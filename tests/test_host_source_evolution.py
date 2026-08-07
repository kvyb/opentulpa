from __future__ import annotations

import os
import shlex
import shutil
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from opentulpa.bootstrap.models import ReleaseRecord
from opentulpa.core.config import Settings
from opentulpa.evolution.evaluator import (
    EvaluationCommand,
    IsolatedProcessEvaluationRunner,
    LocalEvaluationRunner,
)
from opentulpa.evolution.release_provenance import live_repo_artifact_digest
from opentulpa.evolution.sandbox import (
    CandidateContainerBackend,
    CandidateProcessBackend,
    CandidateSandboxPolicy,
    TrustedLocalCandidateBackend,
)
from opentulpa.host import paths as host_paths_module
from opentulpa.host.evolution import (
    HostReleaseActivator,
    TrustedLiveRepoReleaseBuilder,
)
from opentulpa.host.evolution_composition import (
    _trusted_local_evaluation_context,
    _trusted_local_evaluation_executables,
    build_host_evolution_runtime,
)
from opentulpa.host.runtime import (
    RuntimeLiveSourceSpec,
    RuntimeUnavailableError,
)
from opentulpa.host.runtime_environment import LiveSourceRuntimeEnvironment

_EVALUATOR_CONTROLLER_ROOT = Path(sys.executable).absolute().parent.parent
_EVALUATOR_WHEELHOUSE = Path(
    os.environ.get("OPENTULPA_TRUSTED_WHEELHOUSE", "/opt/opentulpa-wheelhouse")
)
_EVALUATOR_UNAVAILABLE = IsolatedProcessEvaluationRunner.unavailable_reason(
    controller_root=_EVALUATOR_CONTROLLER_ROOT,
    wheelhouse=_EVALUATOR_WHEELHOUSE,
)


def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "seed"
    package = root / "src" / "opentulpa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Seed\n", encoding="utf-8")
    return root


def _live_repo_release(source_commit: str = "d" * 40) -> ReleaseRecord:
    digest = live_repo_artifact_digest(source_commit)
    return ReleaseRecord(
        id="release-live-repo",
        candidate_id="candidate-live-repo",
        source_commit=source_commit,
        artifact_digest=digest,
        manifest_digest=digest,
        entrypoint=("python", "-P", "-m", "opentulpa"),
        metadata={
            "artifact_kind": "live_repo",
            "image_reference": f"git-commit:{source_commit}",
        },
    )


class _LiveSourceRuntime:
    def __init__(
        self,
        previous: RuntimeLiveSourceSpec | None = None,
        *,
        fail: bool = False,
        failure_status: str = "ready",
    ) -> None:
        self.live_source = previous
        self.status = "ready"
        self.endpoint = "http://runtime.test"
        self.fail = fail
        self.failure_status = failure_status
        self.verifications: list[RuntimeLiveSourceSpec] = []
        self.replacements: list[tuple[RuntimeLiveSourceSpec, object | None]] = []
        self.project_root = Path("/")

    async def verify_live_source(
        self,
        live_source: RuntimeLiveSourceSpec,
    ) -> RuntimeLiveSourceSpec:
        self.verifications.append(live_source)
        return live_source

    async def replace_live_source(
        self,
        live_source: RuntimeLiveSourceSpec,
        *,
        rollback: object | None = None,
    ) -> None:
        self.replacements.append((live_source, rollback))
        if self.fail:
            self.status = self.failure_status
            if self.status != "ready":
                self.endpoint = None
            raise RuntimeUnavailableError("candidate failed and prior source was restored")
        self.live_source = live_source

    def set_live_source(self, live_source: RuntimeLiveSourceSpec) -> None:
        self.live_source = live_source


class _RuntimeEnvironmentStore:
    def __init__(self, tmp_path: Path) -> None:
        self.interpreter = tmp_path / "runtime-env" / "bin" / "python"
        self.calls: list[tuple[str, Path | None]] = []

    def prepare(
        self,
        source_commit: str,
        *,
        workspace: Path | None = None,
    ) -> LiveSourceRuntimeEnvironment:
        self.calls.append((source_commit, workspace))
        return LiveSourceRuntimeEnvironment(
            id="e" * 64,
            source_commit=source_commit,
            python_interpreter=self.interpreter,
            dependency_lock_hash="f" * 64,
            pyproject_sha256="1" * 64,
            install_profile="runtime-no-dev-no-install-project-v1",
        )


@pytest.mark.asyncio
async def test_live_repo_activation_selects_exact_source_commit(tmp_path: Path) -> None:
    del tmp_path
    release = _live_repo_release()
    runtime = _LiveSourceRuntime()
    activator = HostReleaseActivator(runtime=runtime)  # type: ignore[arg-type]

    result = await activator.activate(
        release,
        activation_id="activation-live-repo",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "active"
    assert runtime.verifications == [RuntimeLiveSourceSpec(source_commit="d" * 40)]
    assert runtime.replacements == [
        (RuntimeLiveSourceSpec(source_commit="d" * 40), None)
    ]
    assert runtime.live_source == RuntimeLiveSourceSpec(source_commit="d" * 40)


def test_live_repo_release_builder_persists_runtime_environment_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    (workspace / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(workspace, "init")
    _git(workspace, "config", "user.name", "Candidate Test")
    _git(workspace, "config", "user.email", "candidate@example.test")
    _git(workspace, "add", "--all")
    _git(workspace, "commit", "-m", "candidate")
    source_commit = _git(workspace, "rev-parse", "HEAD")
    store = _RuntimeEnvironmentStore(tmp_path)
    builder = TrustedLiveRepoReleaseBuilder(runtime_environment_store=store)  # type: ignore[arg-type]

    artifact = builder._build(  # noqa: SLF001
        SimpleNamespace(
            candidate_id="candidate-1",
            workspace=workspace,
            base_commit=source_commit,
            source_commit=source_commit,
            dependency_lock_hash="f" * 64,
            evaluator_version="v1",
            evaluator_fingerprint="sha256:" + "a" * 64,
            evaluation_input_sha256="b" * 64,
        )
    )

    assert artifact.artifact_kind == "live_repo"
    assert artifact.metadata["runtime_environment_id"] == "e" * 64
    assert artifact.metadata["runtime_python_interpreter"] == str(store.interpreter)
    assert store.calls == [(source_commit, workspace)]


@pytest.mark.asyncio
async def test_live_repo_activation_prepares_runtime_environment_before_replacement(
    tmp_path: Path,
) -> None:
    release = _live_repo_release()
    runtime = _LiveSourceRuntime()
    store = _RuntimeEnvironmentStore(tmp_path)
    activator = HostReleaseActivator(
        runtime=runtime,  # type: ignore[arg-type]
        runtime_environment_store=store,  # type: ignore[arg-type]
    )

    result = await activator.activate(
        release,
        activation_id="activation-live-repo-env",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "active"
    assert runtime.replacements
    activated = runtime.replacements[0][0]
    assert activated.runtime_environment_id == "e" * 64
    assert activated.runtime_python_interpreter == str(store.interpreter)
    assert store.calls == [("d" * 40, None)]


@pytest.mark.asyncio
async def test_live_repo_activation_reports_rolled_back_previous_source(
    tmp_path: Path,
) -> None:
    del tmp_path
    previous = RuntimeLiveSourceSpec(source_commit="c" * 40)
    runtime = _LiveSourceRuntime(previous, fail=True)
    activator = HostReleaseActivator(runtime=runtime)  # type: ignore[arg-type]

    result = await activator.activate(
        _live_repo_release(),
        activation_id="activation-live-repo-rollback",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert result.status == "rolled_back"
    assert runtime.replacements[0][0] == RuntimeLiveSourceSpec(source_commit="d" * 40)
    assert runtime.live_source == previous


def test_composition_defaults_to_trusted_local_source_mutation_without_strong_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _seed(tmp_path)
    _git(source, "init")
    _git(source, "config", "user.name", "Source Test")
    _git(source, "config", "user.email", "source@example.test")
    _git(source, "add", "--all")
    _git(source, "commit", "-m", "source")
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o500)
    monkeypatch.setenv("OPENTULPA_UV_BIN", str(uv))
    data = tmp_path / "data"
    data.mkdir(mode=0o700)

    composed = build_host_evolution_runtime(
        runtime=SimpleNamespace(project_root=source),  # type: ignore[arg-type]
        data_root=data,
        control_root=data / "bootstrap",
        settings=Settings(_env_file=None, evolution_enabled=True),
    )

    assert composed is not None
    assert composed.service.source_mutation_enabled is True
    assert isinstance(composed.service._evaluator._runner, LocalEvaluationRunner)  # noqa: SLF001
    factory = composed.service._candidate_backend_factory  # noqa: SLF001
    assert factory is not None
    workspace = data / "bootstrap" / "evolution" / "worktrees" / "candidate"
    workspace.mkdir()
    assert isinstance(factory(workspace), TrustedLocalCandidateBackend)


def test_trusted_local_evaluator_context_hashes_exact_tool_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binary_root = tmp_path / "bin"
    binary_root.mkdir()
    for name in ("python", "ruff", "mypy", "pytest"):
        path = binary_root / name
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o700)
    monkeypatch.setattr(sys, "executable", str(binary_root / "python"))

    executables = _trusted_local_evaluation_executables()
    first = _trusted_local_evaluation_context(executables)
    runner = LocalEvaluationRunner(
        fingerprint_context=lambda: _trusted_local_evaluation_context(executables)
    )
    first_fingerprint = runner.fingerprint
    (binary_root / "ruff").write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    (binary_root / "ruff").chmod(0o700)
    second = _trusted_local_evaluation_context(executables)

    assert executables.ruff == str(binary_root / "ruff")
    assert first["tool.python.path"] == str(binary_root / "python")
    assert first["tool.ruff.sha256"] != second["tool.ruff.sha256"]
    assert first_fingerprint != runner.fingerprint


def test_host_runtime_and_candidate_identities_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(host_paths_module.sys, "platform", "linux")
    monkeypatch.setattr(host_paths_module.os, "geteuid", lambda: 0)

    paths = host_paths_module.HostPaths.from_environment(
        {"OPENTULPA_DATA_ROOT": str(tmp_path / "data")}
    )

    assert (paths.runtime_uid, paths.runtime_gid) == (65_532, 65_532)
    assert (paths.candidate_uid, paths.candidate_gid) == (65_533, 65_533)


def test_final_image_installs_strong_sandbox_and_distinct_candidate_identity() -> None:
    recipe = Path("Dockerfile").read_text(encoding="utf-8")

    for package in ("bubblewrap", "curl", "git", "openssh-client", "util-linux"):
        assert package in recipe
    assert "opentulpa-runtime:x:65532:65532" in recipe
    assert "opentulpa-candidate:x:65533:65533" in recipe
    assert "test -x /usr/bin/bwrap" in recipe
    assert "stat -c %u /usr/bin/bwrap" in recipe
    assert "find /usr/bin/bwrap -perm /022" in recipe


def test_live_repo_reuse_requires_complete_release_provenance() -> None:
    from opentulpa.host import evolution as host_evolution

    expected = _live_repo_release()
    archived = expected.model_copy(deep=True)

    assert host_evolution._same_release_provenance(archived, expected)
    assert not host_evolution._same_release_provenance(
        archived.model_copy(update={"entrypoint": ("python", "-m", "other")}),
        expected,
    )
    assert not host_evolution._same_release_provenance(
        archived.model_copy(
            update={
                "metadata": {
                    **archived.metadata,
                    "image_reference": f"git-commit:{'f' * 40}",
                }
            }
        ),
        expected,
    )


@pytest.mark.skipif(
    not CandidateProcessBackend.is_supported(),
    reason=CandidateProcessBackend.unavailable_reason()
    or "root Linux bubblewrap isolation is required",
)
@pytest.mark.slow
def test_candidate_process_backend_drops_identity_and_has_no_host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    workspace.mkdir(parents=True)
    allowed.chmod(0o700)
    (workspace / "source.txt").write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "must-not-be-visible")
    backend = CandidateProcessBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=False),
    )
    trusted_python = shutil.which("python")
    assert trusted_python is not None

    result = backend.execute(
        'printf \'%s|%s|%s\' "$(id -u)" "${OPENAI_COMPATIBLE_API_KEY-unset}" '
        '"$(command -v python)"; '
        "printf 'after\\n' > source.txt"
    )

    assert result.exit_code == 0
    uid, api_key, python_path = result.output.split("|")
    assert uid == "65533"
    assert api_key == "unset"
    assert python_path == trusted_python
    assert (workspace / "source.txt").read_text(encoding="utf-8") == "after\n"
    assert workspace.stat().st_uid == 0
    assert (workspace / "source.txt").stat().st_uid == 0
    assert allowed.stat().st_mode & 0o777 == 0o700


@pytest.mark.skipif(
    not CandidateProcessBackend.is_supported(),
    reason=CandidateProcessBackend.unavailable_reason()
    or "root Linux bubblewrap isolation is required",
)
@pytest.mark.slow
def test_candidate_process_cannot_see_host_state_processes_or_network(tmp_path: Path) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    workspace.mkdir(parents=True)
    controller_sentinel = Path("/var/tmp") / f"opentulpa-controller-{os.getpid()}"
    product_sentinel = Path("/var/tmp") / f"opentulpa-product-{os.getpid()}"
    controller_sentinel.write_text("controller-secret", encoding="utf-8")
    product_sentinel.write_text("product-secret", encoding="utf-8")
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen()
    port = int(server.getsockname()[1])
    trusted_python = shutil.which("python3")
    assert trusted_python is not None
    script = (
        "import os,socket; from pathlib import Path; "
        f"assert not Path({str(controller_sentinel)!r}).exists(); "
        f"assert not Path({str(product_sentinel)!r}).exists(); "
        f"assert not Path('/proc/{os.getpid()}').exists(); "
        "s=socket.socket(); s.settimeout(0.2); "
        f"assert s.connect_ex(('127.0.0.1',{port})) != 0"
    )
    backend = CandidateProcessBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=False),
    )

    try:
        result = backend.execute(
            f"{shlex.quote(trusted_python)} -I -c {shlex.quote(script)}",
            timeout=10,
        )
    finally:
        server.close()
        controller_sentinel.unlink(missing_ok=True)
        product_sentinel.unlink(missing_ok=True)

    assert result.exit_code == 0, result.output


def test_candidate_sandbox_allows_only_internal_symlinks_when_enabled(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    target = workspace / "packages" / "shared.txt"
    target.parent.mkdir(parents=True)
    target.write_text("shared\n", encoding="utf-8")
    (workspace / "shared.txt").symlink_to("packages/shared.txt")

    CandidateContainerBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(allow_internal_symlinks=True),
    )

    (workspace / "escaped.txt").symlink_to("/etc/hosts")
    with pytest.raises(RuntimeError, match="escaped"):
        CandidateContainerBackend(
            workspace=workspace,
            allowed_root=allowed,
            policy=CandidateSandboxPolicy(allow_internal_symlinks=True),
        )


@pytest.mark.skipif(
    _EVALUATOR_UNAVAILABLE is not None,
    reason=_EVALUATOR_UNAVAILABLE or "root Linux evaluator isolation is required",
)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_isolated_evaluator_uses_writable_copy_without_mutating_source(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    source = workspace / "source.txt"
    source.write_text("trusted\n", encoding="utf-8")
    original_mode = source.stat().st_mode
    runner = IsolatedProcessEvaluationRunner(
        controller_root=_EVALUATOR_CONTROLLER_ROOT,
        wheelhouse=_EVALUATOR_WHEELHOUSE,
    )

    result = await runner.run(
        workspace=workspace,
        command=EvaluationCommand(
            name="readonly",
            argv=(
                runner.executables.python,
                "-I",
                "-c",
                "from pathlib import Path; Path('source.txt').write_text('changed')",
            ),
            timeout_seconds=10,
        ),
    )

    assert result.passed is True
    assert source.read_text(encoding="utf-8") == "trusted\n"
    assert source.stat().st_mode == original_mode


@pytest.mark.skipif(
    _EVALUATOR_UNAVAILABLE is not None,
    reason=_EVALUATOR_UNAVAILABLE or "root Linux evaluator isolation is required",
)
@pytest.mark.slow
@pytest.mark.asyncio
async def test_isolated_evaluator_rejects_inherited_path_executable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    runner = IsolatedProcessEvaluationRunner(
        controller_root=_EVALUATOR_CONTROLLER_ROOT,
        wheelhouse=_EVALUATOR_WHEELHOUSE,
    )

    with pytest.raises(ValueError, match="exact trusted executable"):
        await runner.run(
            workspace=workspace,
            command=EvaluationCommand(
                name="trusted-path",
                argv=("python", "-c", "raise SystemExit(0)"),
                timeout_seconds=10,
            ),
        )


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    import subprocess

    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
