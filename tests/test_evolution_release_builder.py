from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from opentulpa.bootstrap.oci_host import OciCommandResult
from opentulpa.evolution.release_builder import (
    OciReleaseBuildPolicy,
    ReleaseBuildError,
    ReleaseBuildRequest,
    TrustedOciReleaseBuilder,
)

# opentulpa: allow-test-credential


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
    web_assets = root / "src" / "opentulpa" / "web_assets"
    web_assets.mkdir()
    (web_assets / "index.html").write_text("<h1>version 1</h1>\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "seed")
    (workers / "telegram_worker.py").write_text("VERSION = 2\n")
    (manifests / "telegram.json").write_text('{"version":2}\n')
    (web_assets / "index.html").write_text("<h1>version 2</h1>\n")
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
            assert "! -name .venv" in recipe_text
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
            assert (context / "src/opentulpa/web_assets/index.html").read_text() == (
                "<h1>version 2</h1>\n"
            )
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
        "src/opentulpa/web_assets/unsafe.py",
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
