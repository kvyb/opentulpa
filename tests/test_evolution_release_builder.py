from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from opentulpa.bootstrap.oci_host import OciCommandResult
from opentulpa.evolution.dependency_resolver import ResolvedDependencyBase
from opentulpa.evolution.generation import StateContract, canonical_json_bytes
from opentulpa.evolution.generation_store import GenerationStore, GenerationStoreError
from opentulpa.evolution.process import BoundedProcessResult
from opentulpa.evolution.release_builder import (
    DependencyAwareWheelReleaseBuilder,
    OciReleaseArtifact,
    OciReleaseBuildPolicy,
    ReleaseBuildError,
    ReleaseBuildRequest,
    TrustedOciReleaseBuilder,
    TrustedWheelReleaseBuilder,
    WheelReleaseBuildPolicy,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / "Dockerfile").write_text("FROM scratch\nRUN candidate-controlled-command\n")
    (root / ".dockerignore").write_text("*\n")
    (root / "payload.txt").write_text("evaluated source\n")
    (root / "start.sh").write_text("#!/bin/sh\nexec true\n")
    (root / "start.sh").chmod(0o755)
    workers = root / "src" / "opentulpa" / "capability_workers"
    workers.mkdir(parents=True)
    (workers / "__init__.py").write_text("# fixed package hook\n")
    (workers / "__main__.py").write_text("# fixed package entrypoint\n")
    (workers / "telegram_worker.py").write_text("VERSION = 1\n")
    manifests = root / "src" / "opentulpa" / "capability_manifests"
    manifests.mkdir()
    (manifests / "telegram.json").write_text('{"version":1}\n')
    client = root / "src" / "opentulpa" / "client"
    client.mkdir()
    (client / "tui.py").write_text("VERSION = 1\n")
    fixtures = root / "tests"
    fixtures.mkdir()
    (fixtures / "credential_fixture.py").write_text(
        'API_KEY = "sk-proj-baselinefixturevalue1234567890"\n'
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "seed")
    (workers / "telegram_worker.py").write_text("VERSION = 2\n")
    (manifests / "telegram.json").write_text('{"version":2}\n')
    (client / "tui.py").write_text("VERSION = 2\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "allowed runtime overlay")
    return root, _git(root, "rev-parse", "HEAD")


class FakeBuildRunner:
    image_id = f"sha256:{'a' * 64}"
    base_image_id = f"sha256:{'d' * 64}"

    def __init__(self, *, valid_labels: bool = True) -> None:
        self.valid_labels = valid_labels
        self.commands: list[tuple[str, ...]] = []
        self.labels: dict[str, str] = {}
        self.context_entries: set[str] = set()
        self.context_files: dict[str, bytes] = {}

    async def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> OciCommandResult:
        del timeout_seconds, max_output_bytes
        command = tuple(argv)
        self.commands.append(command)
        if command[1] == "info":
            return OciCommandResult(returncode=0, output=b'["name=rootless"]')
        if command[1] == "build":
            iid_path = Path(command[command.index("--iidfile") + 1])
            iid_path.write_text(self.image_id, encoding="ascii")
            for index, item in enumerate(command):
                if item == "--label":
                    key, value = command[index + 1].split("=", 1)
                    self.labels[key] = value
            context = Path(command[-1])
            recipe = Path(command[command.index("--file") + 1])
            recipe_text = recipe.read_text(encoding="ascii")
            assert recipe.parent != context
            assert f"FROM {self.base_image_id}" in recipe_text
            assert "candidate-controlled-command" not in recipe_text
            assert ".venv" not in recipe_text
            assert "COPY --chown=65532:65532 . /app/" in recipe_text
            assert "mv /app/.opentulpa-candidate-dockerignore /app/.dockerignore" in recipe_text
            self.context_entries = {str(path.relative_to(context)) for path in context.rglob("*")}
            self.context_files = {
                str(path.relative_to(context)): path.read_bytes()
                for path in context.rglob("*")
                if path.is_file()
            }
            assert (context / "payload.txt").read_text() == "evaluated source\n"
            assert (context / "start.sh").read_text() == "#!/bin/sh\nexec true\n"
            assert (context / "Dockerfile").read_text().startswith("FROM scratch")
            assert (context / "src/opentulpa/capability_workers/__init__.py").exists()
            assert (context / "src/opentulpa/capability_workers/__main__.py").exists()
            assert (
                context / "src/opentulpa/capability_workers/telegram_worker.py"
            ).read_text() == "VERSION = 2\n"
            assert (
                context / "src/opentulpa/capability_manifests/telegram.json"
            ).read_text() == '{"version":2}\n'
            assert (context / "src/opentulpa/client/tui.py").read_text() == "VERSION = 2\n"
            return OciCommandResult(returncode=0, output=b"built")
        if command[1:3] == ("image", "inspect"):
            if command[3] == self.base_image_id:
                payload = {"Id": self.base_image_id, "Config": {"Labels": {}}}
                return OciCommandResult(returncode=0, output=json.dumps(payload).encode())
            labels = dict(self.labels)
            if not self.valid_labels:
                labels["org.opentulpa.release.source-commit"] = "0" * 40
            payload = {"Id": self.image_id, "Config": {"Labels": labels}}
            return OciCommandResult(returncode=0, output=json.dumps(payload).encode())
        raise AssertionError(f"unexpected command: {command}")


def _request(root: Path, commit: str) -> ReleaseBuildRequest:
    return ReleaseBuildRequest(
        candidate_id="candidate_test",
        workspace=root,
        base_commit=_git(root, "rev-parse", f"{commit}^"),
        source_commit=commit,
        dependency_lock_hash="b" * 64,
        evaluator_version="test-v1",
        evaluator_fingerprint=f"sha256:{'c' * 64}",
    )


def _policy(tmp_path: Path) -> OciReleaseBuildPolicy:
    return OciReleaseBuildPolicy(
        state_root=tmp_path / "build-state",
        base_image_digest=FakeBuildRunner.base_image_id,
        base_dependency_lock_hash="b" * 64,
    )


@pytest.mark.asyncio
async def test_trusted_builder_returns_real_image_id_and_binds_required_labels(
    tmp_path: Path,
) -> None:
    root, commit = _repository(tmp_path)
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(
        policy=_policy(tmp_path),
        runner=runner,
    )

    artifact = await builder.build(_request(root, commit))

    assert artifact.artifact_digest == runner.image_id
    assert artifact.manifest_digest.startswith("sha256:")
    assert artifact.entrypoint == (
        "/opt/opentulpa-install/controller/generations/image/bin/python",
        "-P",
        "-m",
        "opentulpa",
    )
    assert runner.labels == {
        "org.opentulpa.release.manifest-digest": artifact.manifest_digest,
        "org.opentulpa.release.source-commit": commit,
        "org.opentulpa.release.protocol-version": "1",
        "org.opentulpa.release.source-layout": "full-source-v1",
    }
    build = next(command for command in runner.commands if command[1] == "build")
    assert "--network=none" in build
    assert "--pull=false" in build
    assert "--no-cache" in build
    assert ".git" not in runner.context_entries
    assert ".env" not in runner.context_entries
    assert (".dockerignore" in runner.context_entries)
    assert ".opentulpa-candidate-dockerignore" in runner.context_entries


@pytest.mark.asyncio
async def test_trusted_builder_rejects_source_changed_after_evaluation(tmp_path: Path) -> None:
    root, commit = _repository(tmp_path)
    (root / "payload.txt").write_text("changed\n")
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(
        policy=_policy(tmp_path),
        runner=runner,
    )

    with pytest.raises(ReleaseBuildError, match="changed after evaluation"):
        await builder.build(_request(root, commit))

    assert runner.commands == []


@pytest.mark.asyncio
async def test_trusted_builder_rejects_committed_secret_paths(tmp_path: Path) -> None:
    root, _ = _repository(tmp_path)
    (root / ".env").write_text("TOKEN=secret\n")
    _git(root, "add", ".env")
    _git(root, "commit", "-m", "bad secret")
    commit = _git(root, "rev-parse", "HEAD")
    builder = TrustedOciReleaseBuilder(
        policy=_policy(tmp_path),
        runner=FakeBuildRunner(),
    )

    request = replace(_request(root, commit), base_commit=commit)
    with pytest.raises(ReleaseBuildError, match="forbidden secret path"):
        await builder.build(request)


@pytest.mark.asyncio
async def test_trusted_builder_allows_public_environment_template(tmp_path: Path) -> None:
    root, previous = _repository(tmp_path)
    (root / ".env.example").write_text("PUBLIC_SETTING=\n")
    _git(root, "add", ".env.example")
    _git(root, "commit", "-m", "document public environment")
    commit = _git(root, "rev-parse", "HEAD")
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(policy=_policy(tmp_path), runner=runner)

    artifact = await builder.build(replace(_request(root, commit), base_commit=previous))

    assert artifact.artifact_digest == runner.image_id
    assert ".env.example" in runner.context_entries


@pytest.mark.asyncio
async def test_trusted_builder_fails_closed_when_image_labels_do_not_match(tmp_path: Path) -> None:
    root, commit = _repository(tmp_path)
    builder = TrustedOciReleaseBuilder(
        policy=_policy(tmp_path),
        runner=FakeBuildRunner(valid_labels=False),
    )

    with pytest.raises(ReleaseBuildError, match="labels failed verification"):
        await builder.build(_request(root, commit))


@pytest.mark.asyncio
async def test_trusted_builder_rejects_dependency_changes_before_oci_access(
    tmp_path: Path,
) -> None:
    root, commit = _repository(tmp_path)
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(policy=_policy(tmp_path), runner=runner)
    request = _request(root, commit)

    with pytest.raises(ReleaseBuildError, match="trusted runtime base rebuild"):
        await builder.build(replace(request, dependency_lock_hash="e" * 64))

    assert runner.commands == []


@pytest.mark.asyncio
async def test_trusted_builder_exports_exact_blobs_despite_candidate_git_attributes(
    tmp_path: Path,
) -> None:
    root, previous = _repository(tmp_path)
    (root / ".gitattributes").write_text(
        "payload.txt export-ignore\nraw.txt export-subst\n",
        encoding="utf-8",
    )
    (root / "raw.txt").write_text("$Format:%H$\n", encoding="utf-8")
    _git(root, "add", ".gitattributes", "raw.txt")
    _git(root, "commit", "-m", "adversarial export attributes")
    commit = _git(root, "rev-parse", "HEAD")
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(policy=_policy(tmp_path), runner=runner)

    artifact = await builder.build(replace(_request(root, commit), base_commit=previous))

    assert artifact.artifact_digest == runner.image_id
    assert runner.context_files["payload.txt"] == b"evaluated source\n"
    assert runner.context_files["raw.txt"] == b"$Format:%H$\n"


@pytest.mark.asyncio
async def test_trusted_builder_rejects_credential_content_before_oci_access(
    tmp_path: Path,
) -> None:
    root, previous = _repository(tmp_path)
    (root / "provider.py").write_text(
        'API_KEY = "sk-proj-K7mQ2vN9xR4pL8cD6sW3uY5tH1jF0aBz"\n',
        encoding="utf-8",
    )
    _git(root, "add", "provider.py")
    _git(root, "commit", "-m", "bad credential")
    commit = _git(root, "rev-parse", "HEAD")
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(policy=_policy(tmp_path), runner=runner)

    with pytest.raises(ReleaseBuildError, match="credential material"):
        await builder.build(replace(_request(root, commit), base_commit=previous))

    assert runner.commands == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "candidate_path",
    (
        "src/opentulpa/api/app.py",
        "src/opentulpa/capabilities/bundled.py",
        "src/opentulpa/capability_workers/__init__.py",
        "src/opentulpa/capability_workers/__main__.py",
        "src/opentulpa/capability_workers/sitecustomize.py",
        "src/opentulpa/capability_workers/.venv/evil.py",
        "src/opentulpa/capability_manifests/nested/unsafe.json",
        "src/opentulpa/client/unsafe.py",
        "opentulpa.config.yaml",
    ),
)
async def test_trusted_builder_accepts_full_source_changes(
    tmp_path: Path,
    candidate_path: str,
) -> None:
    root, previous = _repository(tmp_path)
    path = root / candidate_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("candidate change\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "unsafe runtime change")
    commit = _git(root, "rev-parse", "HEAD")
    runner = FakeBuildRunner()
    builder = TrustedOciReleaseBuilder(policy=_policy(tmp_path), runner=runner)

    if ".venv" in Path(candidate_path).parts:
        with pytest.raises(ReleaseBuildError, match="contribution-only"):
            await builder.build(replace(_request(root, commit), base_commit=previous))
        assert runner.commands == []
        return

    artifact = await builder.build(replace(_request(root, commit), base_commit=previous))
    assert artifact.artifact_digest == runner.image_id


def _wheel_repository(
    tmp_path: Path,
    *,
    directory_name: str = "wheel-source",
    backend_path: bool = False,
    hook_marker: Path | None = None,
) -> tuple[Path, str, str, str]:
    root = tmp_path / directory_name
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / "pyproject.toml").write_text(
        """[project]
name = "tinyapp"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[project.scripts]
tinyapp = "tinyapp:main"

[build-system]
requires = ["hatchling"]
build-backend = "{build_backend}"
{backend_path}

[tool.hatch.build.targets.wheel]
packages = ["src/tinyapp"]
""".format(
            backend_path='backend-path = ["."]' if backend_path else "",
            build_backend="candidate_backend" if hook_marker is not None else "hatchling.build",
        ),
        encoding="utf-8",
    )
    if hook_marker is not None:
        (root / "candidate_backend.py").write_text(
            f"from pathlib import Path\nPath({str(hook_marker)!r}).write_text('executed')\n",
            encoding="utf-8",
        )
    lock = b'version = 1\nrevision = 1\nrequires-python = ">=3.12"\n'
    (root / "uv.lock").write_bytes(lock)
    package = root / "src" / "tinyapp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "VERSION = 1\ndef main() -> None:\n    print('tiny')\n",
        encoding="utf-8",
    )
    (package / "__main__.py").write_text("from . import main\nmain()\n", encoding="utf-8")
    (package / "data.txt").write_text("resource\n", encoding="utf-8")
    fixtures = root / "tests"
    fixtures.mkdir()
    (fixtures / "credential_fixture.py").write_text(
        'API_KEY = "sk-proj-baselinefixturevalue1234567890"\n',
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "trusted package metadata")
    base_commit = _git(root, "rev-parse", "HEAD")
    (package / "__init__.py").write_text(
        "VERSION = 2\ndef main() -> None:\n    print('tiny')\n",
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "evaluated source change")
    return root, base_commit, _git(root, "rev-parse", "HEAD"), hashlib.sha256(lock).hexdigest()


def _state_contract() -> StateContract:
    return StateContract(
        runtime_protocol=1,
        controller_min=1,
        controller_max=1,
        product_state_schema=1,
        workspace_api=1,
    )


def _wheel_policy(
    tmp_path: Path,
    root: Path,
    lock_hash: str,
    *,
    directory_name: str = "generations",
) -> WheelReleaseBuildPolicy:
    wheelhouse = tmp_path / "trusted wheelhouse"
    wheelhouse.mkdir(mode=0o700, exist_ok=True)
    return WheelReleaseBuildPolicy(
        generations_root=tmp_path / directory_name,
        build_root=tmp_path / "generation builds",
        base_dependency_lock_hash=lock_hash,
        state_contract=_state_contract(),
        trusted_metadata_hashes={
            "pyproject.toml": hashlib.sha256((root / "pyproject.toml").read_bytes()).hexdigest()
        },
        trusted_wheelhouse=wheelhouse,
        external_python_runtime_policy_sha256="9" * 64,
        entrypoint=("venv/bin/python", "-I", "-m", "tinyapp"),
        import_name="tinyapp",
        resource_paths=("data.txt",),
        package_roots=("src/tinyapp",),
        timeout_seconds=60,
    )


def _wheel_request(
    root: Path,
    base_commit: str,
    source_commit: str,
    lock_hash: str,
) -> ReleaseBuildRequest:
    return ReleaseBuildRequest(
        candidate_id="candidate-wheel",
        workspace=root,
        base_commit=base_commit,
        source_commit=source_commit,
        dependency_lock_hash=lock_hash,
        evaluator_version="test-v1",
        evaluator_fingerprint=f"sha256:{'c' * 64}",
        evaluation_input_sha256="d" * 64,
    )


@pytest.mark.asyncio
async def test_dependency_aware_builder_selects_sealed_resolver_policy(tmp_path: Path) -> None:
    root, base_commit, source_commit, base_lock = _wheel_repository(tmp_path)
    policy = _wheel_policy(tmp_path, root, base_lock)
    resolved_root = tmp_path / "resolved" / ("a" * 64)
    wheelhouse = resolved_root / "wheelhouse"
    site = resolved_root / "site"
    wheelhouse.mkdir(parents=True)
    site.mkdir()
    resolved_lock = "b" * 64
    resolved = ResolvedDependencyBase(
        id="a" * 64,
        root=resolved_root,
        lock_sha256=resolved_lock,
        requirements_sha256="c" * 64,
        wheelhouse_sha256="d" * 64,
        inventory_sha256="e" * 64,
        pyproject_sha256="f" * 64,
        site_sha256="1" * 64,
        resolver_fingerprint="sha256:" + "2" * 64,
    )
    artifact = OciReleaseArtifact(
        artifact_kind="python_generation",
        artifact_digest="sha256:" + "3" * 64,
        manifest_digest="sha256:" + "4" * 64,
        image_reference="python-generation:" + "5" * 64,
        entrypoint=policy.entrypoint,
    )

    class Builder:
        def __init__(self) -> None:
            self.requests: list[ReleaseBuildRequest] = []

        async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
            self.requests.append(request)
            return artifact

    class Resolver:
        async def resolve(self, workspace: Path) -> ResolvedDependencyBase:
            del workspace
            return resolved

        def base_for_lock(self, lock_sha256: str) -> ResolvedDependencyBase | None:
            return resolved if lock_sha256 == resolved.lock_sha256 else None

    base_builder = Builder()
    dynamic_builders: list[tuple[WheelReleaseBuildPolicy, Builder]] = []

    def build_for_policy(selected: WheelReleaseBuildPolicy) -> Builder:
        builder = Builder()
        dynamic_builders.append((selected, builder))
        return builder

    builder = DependencyAwareWheelReleaseBuilder(
        base_builder=base_builder,
        base_policy=policy,
        resolver=Resolver(),
        builder_factory=build_for_policy,
    )

    base_request = _wheel_request(root, base_commit, source_commit, base_lock)
    resolved_request = replace(base_request, dependency_lock_hash=resolved_lock)
    assert await builder.build(base_request) == artifact
    assert await builder.build(resolved_request) == artifact

    assert base_builder.requests == [base_request]
    selected_policy, selected_builder = dynamic_builders[0]
    assert selected_builder.requests == [resolved_request]
    assert selected_policy.base_dependency_lock_hash == resolved_lock
    assert selected_policy.trusted_wheelhouse == wheelhouse
    assert selected_policy.trusted_metadata_hashes["pyproject.toml"] == "f" * 64
    assert selected_policy.dependency_base_id == resolved.id
    with pytest.raises(ReleaseBuildError, match="trusted_runtime_base_rebuild_required"):
        await builder.build(replace(base_request, dependency_lock_hash="9" * 64))


def test_wheel_builder_honors_validated_configured_git_executable(tmp_path: Path) -> None:
    root, _, _, lock_hash = _wheel_repository(tmp_path)
    real_git = shutil.which("git")
    assert real_git is not None
    marker = tmp_path / "configured-git-ran"
    executable_dir = tmp_path / "configured-bin"
    executable_dir.mkdir()
    configured_git = executable_dir / "git"
    configured_git.write_text(
        "#!/bin/sh\n"
        f"printf called > {shlex.quote(str(marker))}\n"
        f"exec {shlex.quote(real_git)} \"$@\"\n",
        encoding="utf-8",
    )
    configured_git.chmod(0o755)
    policy = replace(_wheel_policy(tmp_path, root, lock_hash), git_cli=str(configured_git))
    builder = TrustedWheelReleaseBuilder(policy=policy)

    result = builder._run_trusted_git(root, "rev-parse", "--verify", "HEAD^{commit}")

    assert result.output.strip()
    assert marker.read_text(encoding="utf-8") == "called"


class FakeWheelProcessRunner:
    def __init__(self, source_workspace: Path) -> None:
        self.source_workspace = source_workspace
        self.commands: list[tuple[str, ...]] = []
        self.environments: list[dict[str, str]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> BoundedProcessResult:
        command = tuple(argv)
        self.commands.append(command)
        self.environments.append(dict(env))
        assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"}.intersection(env)
        if command[0] == sys.executable and command[1:4] == ("-I", "-m", "venv"):
            venv_path = Path(command[-1])
            (venv_path / "bin").mkdir(parents=True)
            (venv_path / "lib").mkdir()
            interpreter = venv_path / "bin/python"
            interpreter.write_bytes(b"fake final interpreter\n")
            interpreter.chmod(0o755)
            return BoundedProcessResult(0, b"", False, False)
        if command[0] == sys.executable and command[1:3] == ("-I", "-c"):
            runtime = {
                "cpython_version": platform.python_version(),
                "cpython_cache_tag": sys.implementation.cache_tag,
                "cpython_abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
                "os_name": os.name,
                "platform": sysconfig.get_platform(),
                "machine": platform.machine(),
                "implementation": "cpython",
                "libpython": "",
            }
            return BoundedProcessResult(0, json.dumps(runtime).encode(), False, False)
        if command[:2] == ("uv", "export"):
            assert cwd != self.source_workspace
            assert not (cwd / ".git").exists()
            assert "VERSION = 2" in (cwd / "src/tinyapp/__init__.py").read_text()
            output = Path(command[command.index("--output-file") + 1])
            output.write_bytes(b"")
            return BoundedProcessResult(0, b"exported", False, False)
        if command[:3] == ("uv", "pip", "install"):
            if "--requirements" not in command:
                interpreter = Path(command[command.index("--python") + 1])
                entrypoint = interpreter.parent / "tinyapp"
                entrypoint.write_text(f"#!{interpreter}\n", encoding="utf-8")
                entrypoint.chmod(0o755)
            return BoundedProcessResult(0, b"installed", False, False)
        raise AssertionError(f"unexpected generation command: {command}")


@pytest.mark.asyncio
async def test_trusted_wheel_builder_installs_exact_blobs_into_final_path_generation(
    tmp_path: Path,
) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    runner = FakeWheelProcessRunner(root)
    policy = _wheel_policy(tmp_path, root, lock_hash)
    builder = TrustedWheelReleaseBuilder(policy=policy, process_runner=runner)

    artifact = await builder.build(_wheel_request(root, base_commit, source_commit, lock_hash))

    generation_id = artifact.image_reference.removeprefix("python-generation:")
    installed = GenerationStore(policy.generations_root).open(
        generation_id,
        expected_manifest_digest=artifact.manifest_digest,
        expected_state_contract_digest=policy.state_contract.sha256(),
        expected_evaluator_fingerprint=f"sha256:{'c' * 64}",
        expected_install_profile="runtime",
        controller_protocol=1,
    )
    assert artifact.artifact_kind == "python_generation"
    assert artifact.artifact_digest == artifact.manifest_digest
    assert artifact.manifest_digest == installed.manifest_digest
    interpreter_sha256 = hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest()
    expected_python_runtime = hashlib.sha256(
        canonical_json_bytes(
            {
                "external_python_runtime_policy_sha256": (
                    policy.external_python_runtime_policy_sha256
                ),
                "interpreter_sha256": interpreter_sha256,
            }
        )
    ).hexdigest()
    assert installed.manifest.identity.python_runtime_sha256 == expected_python_runtime
    assert artifact.entrypoint == ("venv/bin/python", "-I", "-m", "tinyapp")
    assert installed.entrypoint_path == policy.generations_root / generation_id / "venv/bin/python"
    assert not (installed.path / "BUILDING").exists()
    assert (installed.path / "COMPLETE").read_bytes() == b""

    venv_command = next(command for command in runner.commands if command[1:4] == ("-I", "-m", "venv"))
    assert Path(venv_command[-1]) == installed.path / "venv"
    export = next(command for command in runner.commands if command[:2] == ("uv", "export"))
    assert {"--frozen", "--no-dev", "--no-emit-project"}.issubset(export)
    dependency_install = next(
        command
        for command in runner.commands
        if command[:3] == ("uv", "pip", "install") and "--requirements" in command
    )
    assert {"--offline", "--no-index", "--require-hashes", "--only-binary=:all:"}.issubset(
        dependency_install
    )
    assert not any(command[1:2] == ("build",) for command in runner.commands)
    assert not any(
        Path(command[0]).name == "python" and command[1:3] == ("-I", "-c")
        for command in runner.commands
        if command[0] != sys.executable
    )
    assert all("VIRTUAL_ENV" not in environment for environment in runner.environments)
    assert stat.S_IMODE(installed.path.stat().st_mode) == 0o555
    assert not any(path.stat().st_mode & 0o222 for path in installed.path.rglob("*"))
    wheel = installed.path / installed.manifest.descriptor.wheel_path
    with zipfile.ZipFile(wheel) as archive:
        assert "tinyapp/__init__.py" in archive.namelist()
        assert archive.read("tinyapp/__init__.py").startswith(b"VERSION = 2")


@pytest.mark.asyncio
async def test_trusted_wheel_builder_rejects_candidate_pyproject_change(tmp_path: Path) -> None:
    root, _, previous, lock_hash = _wheel_repository(tmp_path)
    with (root / "pyproject.toml").open("a", encoding="utf-8") as stream:
        stream.write("\n[tool.candidate]\nenabled = true\n")
    _git(root, "add", "pyproject.toml")
    _git(root, "commit", "-m", "candidate packaging change")
    source_commit = _git(root, "rev-parse", "HEAD")
    runner = FakeWheelProcessRunner(root)
    policy = _wheel_policy(tmp_path, root, lock_hash)
    builder = TrustedWheelReleaseBuilder(
        policy=policy,
        process_runner=runner,
    )

    with pytest.raises(ReleaseBuildError, match="trusted packaging metadata pin|metadata changed"):
        await builder.build(_wheel_request(root, previous, source_commit, lock_hash))

    assert all(command[0] != "uv" for command in runner.commands)


@pytest.mark.asyncio
async def test_trusted_wheel_builder_retains_runtime_base_failure_for_lock_change(
    tmp_path: Path,
) -> None:
    root, _, previous, lock_hash = _wheel_repository(tmp_path)
    (root / "uv.lock").write_text("candidate lock\n", encoding="utf-8")
    _git(root, "add", "uv.lock")
    _git(root, "commit", "-m", "candidate lock change")
    source_commit = _git(root, "rev-parse", "HEAD")
    runner = FakeWheelProcessRunner(root)
    builder = TrustedWheelReleaseBuilder(
        policy=_wheel_policy(tmp_path, root, lock_hash),
        process_runner=runner,
    )

    with pytest.raises(ReleaseBuildError, match="trusted_runtime_base_rebuild_required"):
        await builder.build(_wheel_request(root, previous, source_commit, lock_hash))

    assert all(command[0] != "uv" for command in runner.commands)


@pytest.mark.asyncio
async def test_trusted_wheel_builder_accepts_resolver_bound_dependency_change(
    tmp_path: Path,
) -> None:
    root, _, previous, _ = _wheel_repository(tmp_path)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            "dependencies = []",
            'dependencies = ["demo==1.0.0"]',
        ),
        encoding="utf-8",
    )
    lock = b'version = 1\nrevision = 2\nrequires-python = ">=3.12"\n'
    (root / "uv.lock").write_bytes(lock)
    _git(root, "add", "pyproject.toml", "uv.lock")
    _git(root, "commit", "-m", "resolved dependency change")
    source_commit = _git(root, "rev-parse", "HEAD")
    lock_hash = hashlib.sha256(lock).hexdigest()
    policy = replace(
        _wheel_policy(tmp_path, root, lock_hash),
        dependency_base_id="a" * 64,
        build_recipe_version="resolved-wheel-v1-" + "a" * 64,
    )
    builder = TrustedWheelReleaseBuilder(
        policy=policy,
        process_runner=FakeWheelProcessRunner(root),
    )

    artifact = await builder.build(_wheel_request(root, previous, source_commit, lock_hash))

    assert artifact.artifact_kind == "python_generation"


@pytest.mark.asyncio
async def test_trusted_wheel_builder_never_executes_candidate_build_backend(tmp_path: Path) -> None:
    marker = tmp_path / "backend-executed"
    root, base_commit, source_commit, lock_hash = _wheel_repository(
        tmp_path,
        backend_path=True,
        hook_marker=marker,
    )
    runner = FakeWheelProcessRunner(root)
    builder = TrustedWheelReleaseBuilder(
        policy=_wheel_policy(tmp_path, root, lock_hash),
        process_runner=runner,
    )

    with pytest.raises(ReleaseBuildError, match="backend-path"):
        await builder.build(_wheel_request(root, base_commit, source_commit, lock_hash))

    assert not marker.exists()
    assert all(command[:2] != ("uv", "build") for command in runner.commands)


@pytest.mark.asyncio
async def test_trusted_wheel_builder_rejects_local_fsmonitor_without_executing_it(
    tmp_path: Path,
) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    marker = tmp_path / "fsmonitor-executed"
    monitor = tmp_path / "malicious-fsmonitor"
    monitor.write_text(f"#!/bin/sh\ntouch {marker!s}\n", encoding="utf-8")
    monitor.chmod(0o755)
    _git(root, "config", "core.fsmonitor", str(monitor))
    builder = TrustedWheelReleaseBuilder(
        policy=_wheel_policy(tmp_path, root, lock_hash),
        process_runner=FakeWheelProcessRunner(root),
    )

    with pytest.raises(ReleaseBuildError, match="configuration is unsafe"):
        await builder.build(_wheel_request(root, base_commit, source_commit, lock_hash))

    assert not marker.exists()


@pytest.mark.asyncio
async def test_trusted_wheel_builder_ignores_git_replace_refs(tmp_path: Path) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    package = root / "src/tinyapp/__init__.py"
    package.write_text("VERSION = 999\ndef main() -> None:\n    raise SystemExit(99)\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "malicious replacement")
    replacement_commit = _git(root, "rev-parse", "HEAD")
    _git(root, "replace", source_commit, replacement_commit)
    _git(root, "--no-replace-objects", "checkout", "--detach", source_commit)
    policy = _wheel_policy(tmp_path, root, lock_hash)
    artifact = await TrustedWheelReleaseBuilder(
        policy=policy,
        process_runner=FakeWheelProcessRunner(root),
    ).build(_wheel_request(root, base_commit, source_commit, lock_hash))
    generation_id = artifact.image_reference.removeprefix("python-generation:")
    installed = GenerationStore(policy.generations_root).open_untrusted_for_test(generation_id)

    with zipfile.ZipFile(installed.path / installed.manifest.descriptor.wheel_path) as archive:
        assert archive.read("tinyapp/__init__.py").startswith(b"VERSION = 2")


@pytest.mark.asyncio
async def test_failed_publication_verification_is_quarantined_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    policy = _wheel_policy(tmp_path, root, lock_hash)
    runner = FakeWheelProcessRunner(root)
    builder = TrustedWheelReleaseBuilder(policy=policy, process_runner=runner)
    real_open = builder._store.open

    def fail_verification(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise GenerationStoreError("injected post-publication failure")

    monkeypatch.setattr(builder._store, "open", fail_verification)
    with pytest.raises(GenerationStoreError, match="injected"):
        await builder.build(_wheel_request(root, base_commit, source_commit, lock_hash))

    assert not tuple(policy.generations_root.glob("[0-9a-f]" * 64))
    assert tuple(builder._store.quarantine_root.iterdir())
    monkeypatch.setattr(builder._store, "open", real_open)
    artifact = await builder.build(_wheel_request(root, base_commit, source_commit, lock_hash))
    assert artifact.artifact_kind == "python_generation"


@pytest.mark.asyncio
async def test_concurrent_same_identity_builders_serialize_and_reuse_generation(
    tmp_path: Path,
) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    policy = _wheel_policy(tmp_path, root, lock_hash)
    first_runner = FakeWheelProcessRunner(root)
    second_runner = FakeWheelProcessRunner(root)
    request = _wheel_request(root, base_commit, source_commit, lock_hash)

    first, second = await asyncio.gather(
        TrustedWheelReleaseBuilder(policy=policy, process_runner=first_runner).build(request),
        TrustedWheelReleaseBuilder(policy=policy, process_runner=second_runner).build(request),
    )

    assert first == second
    commands = first_runner.commands + second_runner.commands
    assert sum(command[1:4] == ("-I", "-m", "venv") for command in commands) == 1


@pytest.mark.asyncio
async def test_same_build_identity_with_different_runtime_trees_has_distinct_artifact_digest(
    tmp_path: Path,
) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    request = _wheel_request(root, base_commit, source_commit, lock_hash)
    first_policy = _wheel_policy(
        tmp_path,
        root,
        lock_hash,
        directory_name="runtime-generations-one",
    )
    second_policy = _wheel_policy(
        tmp_path,
        root,
        lock_hash,
        directory_name="runtime-generations-two",
    )

    first = await TrustedWheelReleaseBuilder(
        policy=first_policy,
        process_runner=FakeWheelProcessRunner(root),
    ).build(request)
    second = await TrustedWheelReleaseBuilder(
        policy=second_policy,
        process_runner=FakeWheelProcessRunner(root),
    ).build(request)

    assert first.image_reference == second.image_reference
    assert first.artifact_digest == first.manifest_digest
    assert second.artifact_digest == second.manifest_digest
    assert first.artifact_digest != second.artifact_digest


@pytest.mark.asyncio
async def test_generation_paths_with_spaces_preserve_absolute_shebang_hashing(
    tmp_path: Path,
) -> None:
    root, base_commit, source_commit, lock_hash = _wheel_repository(
        tmp_path,
        directory_name="source with spaces",
    )
    policy = _wheel_policy(
        tmp_path,
        root,
        lock_hash,
        directory_name="generations with spaces",
    )
    artifact = await TrustedWheelReleaseBuilder(
        policy=policy,
        process_runner=FakeWheelProcessRunner(root),
    ).build(_wheel_request(root, base_commit, source_commit, lock_hash))
    generation_id = artifact.image_reference.removeprefix("python-generation:")
    installed = GenerationStore(policy.generations_root).open_untrusted_for_test(generation_id)

    console_script = installed.path / "venv/bin/tinyapp"
    shebang = console_script.read_text(encoding="utf-8").splitlines()[0]
    assert shebang == f"#!{installed.interpreter_path}"
    assert " " in shebang


def test_missing_trusted_wheelhouse_requires_runtime_base_rebuild(tmp_path: Path) -> None:
    root, _, _, lock_hash = _wheel_repository(tmp_path)
    policy = replace(
        _wheel_policy(tmp_path, root, lock_hash),
        trusted_wheelhouse=tmp_path / "missing-wheelhouse",
    )

    with pytest.raises(ReleaseBuildError, match="trusted_runtime_base_rebuild_required"):
        TrustedWheelReleaseBuilder(policy=policy, process_runner=FakeWheelProcessRunner(root))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.asyncio
async def test_controller_assembled_wheel_real_final_path_install_and_import(
    tmp_path: Path,
) -> None:
    from opentulpa.evolution.process import run_bounded_process

    def run_real_command(argv: Sequence[str], *, timeout_seconds: int) -> BoundedProcessResult:
        result = run_bounded_process(
            argv,
            cwd=tmp_path,
            env=os.environ.copy(),
            timeout_seconds=timeout_seconds,
            max_output_bytes=32 * 1024,
        )
        output = result.output.decode(errors="replace")
        if result.truncated:
            output += "\n[output truncated]"
        if result.timed_out:
            pytest.fail(
                f"command timed out after {timeout_seconds}s: {tuple(argv)!r}\n{output}",
                pytrace=False,
            )
        if result.returncode != 0:
            pytest.fail(
                f"command failed with exit code {result.returncode}: {tuple(argv)!r}\n{output}",
                pytrace=False,
            )
        return result

    root, base_commit, source_commit, lock_hash = _wheel_repository(tmp_path)
    policy = _wheel_policy(tmp_path, root, lock_hash)
    artifact = await TrustedWheelReleaseBuilder(
        policy=policy,
        process_runner=FakeWheelProcessRunner(root),
    ).build(_wheel_request(root, base_commit, source_commit, lock_hash))
    generation_id = artifact.image_reference.removeprefix("python-generation:")
    assembled = GenerationStore(policy.generations_root).open_untrusted_for_test(generation_id)
    wheel = assembled.path / assembled.manifest.descriptor.wheel_path

    runtime_root = tmp_path / "real runtime root"
    final_generation = runtime_root / "tiny final generation"
    venv = final_generation / "venv"
    runtime_root.mkdir(mode=0o711)
    runtime_root.chmod(0o711)
    final_generation.mkdir(mode=0o700)
    run_real_command(
        [
            sys.executable,
            "-I",
            "-m",
            "venv",
            "--copies",
            "--without-pip",
            str(venv),
        ],
        timeout_seconds=60,
    )
    library_name = sysconfig.get_config_var("LDLIBRARY")
    library_root = sysconfig.get_config_var("LIBDIR")
    if library_name and library_root:
        source_library = Path(library_root) / library_name
        destination_library = venv / "lib" / library_name
        if source_library.is_file() and not destination_library.exists():
            shutil.copyfile(source_library, destination_library)
    lib64 = venv / "lib64"
    if lib64.is_symlink() and os.readlink(lib64) == "lib":
        lib64.unlink()
    run_real_command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv / "bin/python"),
            "--offline",
            "--no-index",
            "--no-deps",
            "--link-mode=copy",
            str(wheel),
        ],
        timeout_seconds=60,
    )
    directories: list[Path] = []
    for path in final_generation.rglob("*"):
        metadata = path.lstat()
        assert not stat.S_ISLNK(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            directories.append(path)
        else:
            path.chmod(0o555 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o444)
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        directory.chmod(0o555)
    final_generation.chmod(0o555)

    completed = run_real_command(
        [str(venv / "bin/python"), "-I", "-c", "import tinyapp; assert tinyapp.VERSION == 2"],
        timeout_seconds=30,
    )

    assert completed.returncode == 0
    assert stat.S_IMODE(runtime_root.stat().st_mode) == 0o711
    assert stat.S_IMODE(final_generation.stat().st_mode) == 0o555
