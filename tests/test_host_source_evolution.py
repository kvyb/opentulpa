from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from opentulpa.bootstrap.evolution_runtime import TrustedSourceReleaseProvider
from opentulpa.evolution.evaluator import (
    EvaluationCommand,
    IsolatedProcessEvaluationRunner,
)
from opentulpa.evolution.release_builder import (
    ReleaseBuildError,
    ReleaseBuildRequest,
    SourceOverlayBuildPolicy,
    TrustedSourceOverlayBuilder,
)
from opentulpa.evolution.sandbox import CandidateProcessBackend, CandidateSandboxPolicy
from opentulpa.host.evolution import SourceOverlayReleaseActivator, seed_source_repository


def _seed(tmp_path: Path) -> Path:
    root = tmp_path / "seed"
    package = root / "src" / "opentulpa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "README.md").write_text("# Seed\n", encoding="utf-8")
    return root


def _builder(repository: Path) -> TrustedSourceOverlayBuilder:
    lock_hash = hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
    return TrustedSourceOverlayBuilder(
        policy=SourceOverlayBuildPolicy(base_dependency_lock_hash=lock_hash)
    )


def test_source_repository_import_is_persistent_and_tracks_new_bundled_source(
    tmp_path: Path,
) -> None:
    seed = _seed(tmp_path)
    repository = seed_source_repository(seed_root=seed, repository=tmp_path / "state" / "source")
    first = _git(repository, "rev-parse", "HEAD")

    assert seed_source_repository(seed_root=seed, repository=repository) == repository
    assert _git(repository, "rev-parse", "HEAD") == first

    (seed / "README.md").write_text("# Updated\n", encoding="utf-8")
    seed_source_repository(seed_root=seed, repository=repository)
    second = _git(repository, "rev-parse", "HEAD")

    assert second != first
    assert _git(repository, "show", f"{second}:README.md") == "# Updated"
    assert _git(repository, "merge-base", "--is-ancestor", first, second, check=False) == ""


@pytest.mark.asyncio
async def test_source_overlay_builder_binds_exact_commit_and_dependency_lock(
    tmp_path: Path,
) -> None:
    repository = seed_source_repository(
        seed_root=_seed(tmp_path),
        repository=tmp_path / "state" / "source",
    )
    head = _git(repository, "rev-parse", "HEAD")
    lock_hash = hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
    builder = _builder(repository)
    request = ReleaseBuildRequest(
        candidate_id="candidate-test",
        workspace=repository,
        base_commit=head,
        source_commit=head,
        dependency_lock_hash=lock_hash,
        evaluator_version="test-v1",
        evaluator_fingerprint=f"sha256:{'1' * 64}",
    )

    artifact = await builder.build(request)

    assert artifact.artifact_kind == "source_overlay"
    assert artifact.image_reference == f"source-overlay:{head}"
    assert artifact.artifact_digest.startswith("sha256:")
    with pytest.raises(ReleaseBuildError, match="dependency lock changed"):
        await builder.build(replace(request, dependency_lock_hash="0" * 64))


class _Runtime:
    def __init__(self, root: Path, *, fail: bool = False) -> None:
        self.project_root = root
        self.status = "ready"
        self.fail = fail
        self.replacements: list[Path] = []

    async def replace_source(self, root: Path) -> None:
        from opentulpa.host.runtime import RuntimeUnavailableError

        self.replacements.append(root)
        if self.fail:
            raise RuntimeUnavailableError("candidate failed")
        self.project_root = root


@pytest.mark.asyncio
async def test_source_overlay_activation_materializes_and_reports_health_rollback(
    tmp_path: Path,
) -> None:
    repository = seed_source_repository(
        seed_root=_seed(tmp_path),
        repository=tmp_path / "state" / "source",
    )
    builder = _builder(repository)
    provider = TrustedSourceReleaseProvider(
        source_repository=repository,
        builder=builder,
        evaluator_version="test-v1",
        evaluator_fingerprint=f"sha256:{'2' * 64}",
    )
    release = await provider.build()
    runtime = _Runtime(repository)
    activator = SourceOverlayReleaseActivator(
        repository=repository,
        releases_root=tmp_path / "releases",
        runtime=runtime,  # type: ignore[arg-type]
    )

    active = await activator.activate(
        release,
        activation_id="activation-ok",
        origin=None,
        reason="test",
        rollback=False,
    )

    assert active.status == "active"
    assert runtime.project_root.name == release.source_commit
    assert (runtime.project_root / "src" / "opentulpa" / "__init__.py").is_file()

    failed_runtime = _Runtime(repository, fail=True)
    failed = SourceOverlayReleaseActivator(
        repository=repository,
        releases_root=tmp_path / "failed-releases",
        runtime=failed_runtime,  # type: ignore[arg-type]
    )
    rolled_back = await failed.activate(
        release,
        activation_id="activation-failed",
        origin=None,
        reason="test",
        rollback=False,
    )
    assert rolled_back.status == "rolled_back"
    assert rolled_back.failure_code == "release_unhealthy"


@pytest.mark.skipif(
    os.name != "posix"
    or not hasattr(os, "geteuid")
    or os.geteuid() != 0
    or shutil.which("setpriv") is None
    or shutil.which("prlimit") is None,
    reason="Linux root process isolation is required",
)
def test_candidate_process_backend_drops_identity_and_has_no_host_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed = tmp_path / "worktrees"
    workspace = allowed / "candidate"
    workspace.mkdir(parents=True)
    (workspace / "source.txt").write_text("before\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "must-not-be-visible")
    backend = CandidateProcessBackend(
        workspace=workspace,
        allowed_root=allowed,
        policy=CandidateSandboxPolicy(network_enabled=True),
    )
    trusted_python = shutil.which("python")
    assert trusted_python is not None

    result = backend.execute(
        'printf \'%s|%s|%s\' "$(id -u)" "${OPENAI_COMPATIBLE_API_KEY-unset}" '
        '"$(command -v python)"; '
        "printf 'after\\n' > source.txt"
    )

    assert result.exit_code == 0
    assert result.output == f"65532|unset|{trusted_python}"
    assert (workspace / "source.txt").read_text(encoding="utf-8") == "after\n"
    assert workspace.stat().st_uid == 0
    assert (workspace / "source.txt").stat().st_uid == 0


@pytest.mark.skipif(
    os.name != "posix"
    or not hasattr(os, "geteuid")
    or os.geteuid() != 0
    or shutil.which("setpriv") is None
    or shutil.which("prlimit") is None,
    reason="Linux root process isolation is required",
)
@pytest.mark.asyncio
async def test_isolated_evaluator_uses_writable_copy_without_mutating_source(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    source = workspace / "source.txt"
    source.write_text("trusted\n", encoding="utf-8")
    original_mode = source.stat().st_mode
    runner = IsolatedProcessEvaluationRunner()

    result = await runner.run(
        workspace=workspace,
        command=EvaluationCommand(
            name="readonly",
            argv=("/bin/sh", "-lc", "printf changed > source.txt"),
            timeout_seconds=10,
        ),
    )

    assert result.passed is True
    assert source.read_text(encoding="utf-8") == "trusted\n"
    assert source.stat().st_mode == original_mode


@pytest.mark.skipif(
    os.name != "posix"
    or not hasattr(os, "geteuid")
    or os.geteuid() != 0
    or shutil.which("setpriv") is None
    or shutil.which("prlimit") is None,
    reason="Linux root process isolation is required",
)
@pytest.mark.asyncio
async def test_isolated_evaluator_preserves_trusted_runtime_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "candidate"
    workspace.mkdir()
    probe_root = Path("/var/tmp") / f"opentulpa-evaluator-probe-{os.getpid()}"
    probe_root.mkdir(mode=0o755)
    probe = probe_root / "trusted-evaluator-probe"
    probe.write_text("#!/bin/sh\nprintf available", encoding="utf-8")
    probe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{probe_root}:{os.environ.get('PATH', os.defpath)}")
    runner = IsolatedProcessEvaluationRunner()

    try:
        result = await runner.run(
            workspace=workspace,
            command=EvaluationCommand(
                name="trusted-path",
                argv=(probe.name,),
                timeout_seconds=10,
            ),
        )
    finally:
        probe.unlink(missing_ok=True)
        probe_root.rmdir()

    assert result.passed is True
    assert result.output == "available"


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    import subprocess

    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
