"""Compose live-source evolution for the stable host."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import sys
from pathlib import Path

from opentulpa.core.config import Settings
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    EvaluationExecutables,
    EvaluationRunner,
    LocalEvaluationRunner,
    trusted_default_commands,
)
from opentulpa.evolution.lineage import GitLineage
from opentulpa.evolution.release import AtomicReleasePointer
from opentulpa.evolution.sandbox import (
    CandidateSandboxPolicy,
    TrustedLocalCandidateBackend,
)
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.evolution.workspace import GitCandidateWorkspace
from opentulpa.host.evolution import (
    HostEvolutionControlService,
    HostEvolutionRuntime,
    HostReleaseActivator,
    RuntimeEvolutionEventSink,
    TrustedLiveRepoReleaseBuilder,
    TrustedLiveRepoReleaseProvider,
    prepare_live_source_repository,
)
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.host.runtime_environment import (
    LiveSourceRuntimeEnvironmentStore,
    RuntimeEnvFileManager,
)

_LIVE_REPO_EVALUATOR_VERSION = "host-live-repo-v1"


def build_host_evolution_runtime(
    *,
    runtime: RuntimeSupervisor,
    data_root: Path,
    settings: Settings,
    control_root: Path | None = None,
) -> HostEvolutionRuntime | None:
    """Build the host evolution controller for the live-source repository."""

    if not settings.evolution_enabled:
        return None
    live_repository = _live_source_repository(runtime)

    resolved_data = data_root.expanduser().absolute()
    resolved_control = (
        control_root.expanduser().absolute()
        if control_root is not None
        else resolved_data / "bootstrap"
    )
    evolution_root = resolved_control / "evolution"
    _private_directory(evolution_root)
    worktrees_root = evolution_root / "worktrees"
    artifacts_root = evolution_root / "artifacts"
    for root in (worktrees_root, artifacts_root):
        _private_directory(root)

    if live_repository is None:
        return None

    sandbox_policy = CandidateSandboxPolicy(
        cpu_limit=settings.sandbox_cpu_limit,
        memory_limit=settings.sandbox_memory_limit,
        pid_limit=settings.sandbox_pid_limit,
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=settings.sandbox_max_output_bytes,
        network_enabled=False,
    )

    def live_candidate_backend(workspace: Path) -> TrustedLocalCandidateBackend:
        return TrustedLocalCandidateBackend(
            workspace=workspace,
            allowed_root=worktrees_root,
            policy=sandbox_policy,
        )

    evaluator_executables = _trusted_local_evaluation_executables()
    evaluator_runner: EvaluationRunner
    evaluator_runner = LocalEvaluationRunner(
        fingerprint_context=lambda: _trusted_local_evaluation_context(evaluator_executables)
    )
    evaluator = CandidateEvaluator(
        runner=evaluator_runner,
        # ponytail: full validation belongs in CI; live releases keep fast integrity gates.
        commands=tuple(
            command
            for command in trusted_default_commands(
                timeout_seconds=900,
                executables=evaluator_executables,
            )
            if command.stage != "public"
        ),
    )

    runtime_environment_store = LiveSourceRuntimeEnvironmentStore(
        source_repository=live_repository,
        envs_root=resolved_data / "runtime-source-envs",
        worktrees_root=evolution_root / "runtime-env-worktrees",
        uv_cli=_trusted_uv_cli(),
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=settings.sandbox_max_output_bytes,
    )
    runtime_env_manager = RuntimeEnvFileManager(
        source_root=live_repository,
        runtime=runtime,
    )
    lineage = GitLineage(live_repository, worktrees_root=worktrees_root)
    archive = EvolutionArchive(evolution_root / "archive.db")
    activator = HostReleaseActivator(
        runtime=runtime,
        runtime_environment_store=runtime_environment_store,
    )
    service = EvolutionSupervisor(
        archive=archive,
        workspaces=GitCandidateWorkspace(
            source_repository=live_repository,
            worktrees_root=worktrees_root,
            artifacts_root=artifacts_root,
        ),
        candidate_backend_factory=live_candidate_backend,
        evaluator=evaluator,
        evaluator_version=_LIVE_REPO_EVALUATOR_VERSION,
        release_pointer=AtomicReleasePointer(evolution_root / "current_release.json"),
        source_ref="refs/opentulpa/instance",
        upstream_repository=settings.evolution_upstream_repository,
        upstream_ref=settings.evolution_upstream_ref,
        release_builder=TrustedLiveRepoReleaseBuilder(
            runtime_environment_store=runtime_environment_store,
        ),
        release_activator=activator,
        lineage=lineage,
        event_sink=RuntimeEvolutionEventSink(runtime),
        source_mutation_enabled=True,
        source_mutation_unavailable_reason=None,
    )
    live_initial = TrustedLiveRepoReleaseProvider(
        source_repository=live_repository,
        evaluator_version=_LIVE_REPO_EVALUATOR_VERSION,
        evaluator_fingerprint=lambda: evaluator.fingerprint,
        runtime_environment_store=runtime_environment_store,
    )
    return HostEvolutionRuntime(
        runtime=runtime,
        archive=archive,
        evolution=service,
        activator=activator,
        initial_release=live_initial,
        control_service=HostEvolutionControlService(
            evolution=service,
            runtime_env_file_manager=runtime_env_manager,
        ),
    )


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise RuntimeError("host evolution controller root is unsafe")
    path.chmod(0o700)


def _live_source_repository(runtime: RuntimeSupervisor) -> Path | None:
    raw_root = getattr(runtime, "project_root", None)
    if raw_root is None:
        return None
    root = Path(raw_root).expanduser()
    if not (
        root.is_dir()
        and (root / ".git").exists()
        and (root / "src" / "opentulpa" / "__init__.py").is_file()
        and (root / "uv.lock").is_file()
    ):
        return None
    return prepare_live_source_repository(root)


def _trusted_local_evaluation_executables() -> EvaluationExecutables:
    binary_root = Path(sys.executable).absolute().parent

    def tool(name: str) -> str:
        candidate = binary_root / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        resolved = shutil.which(name)
        if resolved is not None:
            path = Path(resolved).expanduser().absolute()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        raise RuntimeError(f"trusted local evaluation requires {name} on PATH")

    return EvaluationExecutables(
        python=sys.executable,
        ruff=tool("ruff"),
        mypy=tool("mypy"),
        pytest=tool("pytest"),
    )


def _trusted_local_evaluation_context(executables: EvaluationExecutables) -> dict[str, str]:
    context: dict[str, str] = {"mode": "trusted-local"}
    tools = {
        "python": executables.python,
        "ruff": executables.ruff,
        "mypy": executables.mypy,
        "pytest": executables.pytest,
    }
    for name, value in tools.items():
        path = Path(value).expanduser().absolute()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"trusted local evaluation requires executable {name}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise RuntimeError(f"trusted local evaluation requires executable {name}")
        context[f"tool.{name}.path"] = str(path)
        context[f"tool.{name}.resolved_path"] = str(resolved)
        context[f"tool.{name}.sha256"] = _local_file_sha256(resolved)
    return context


def _local_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trusted_uv_cli() -> str:
    configured = str(os.environ.get("OPENTULPA_UV_BIN") or "").strip()
    candidate = Path(configured or shutil.which("uv") or "").expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise RuntimeError("host runtime environment builder requires the trusted uv executable")
    if candidate.name != "uv" or not os.access(candidate, os.X_OK):
        raise RuntimeError("host runtime environment builder requires the trusted uv executable")
    return str(candidate)


__all__ = ["build_host_evolution_runtime"]
