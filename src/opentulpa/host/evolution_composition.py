"""Compose source evolution for a stable host without an OCI engine."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from opentulpa.bootstrap.evolution_runtime import TrustedSourceReleaseProvider
from opentulpa.core.config import Settings
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    IsolatedProcessEvaluationRunner,
    trusted_default_commands,
)
from opentulpa.evolution.release import AtomicReleasePointer
from opentulpa.evolution.release_builder import (
    SourceOverlayBuildPolicy,
    TrustedSourceOverlayBuilder,
)
from opentulpa.evolution.sandbox import (
    CandidateProcessBackend,
    CandidateSandboxPolicy,
)
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.evolution.workspace import GitCandidateWorkspace
from opentulpa.host.evolution import (
    HostEvolutionRuntime,
    SourceOverlayReleaseActivator,
    seed_source_repository,
)
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.notifications.service import NotificationService
from opentulpa.notifications.sinks import EvolutionNotificationSink
from opentulpa.notifications.store import NotificationStore


def build_host_evolution_runtime(
    *,
    runtime: RuntimeSupervisor,
    data_root: Path,
    settings: Settings,
) -> HostEvolutionRuntime | None:
    """Build the Railway/default-host source controller when its trusted seed exists."""

    if not settings.evolution_enabled:
        return None
    configured_seed = str(os.environ.get("OPENTULPA_SOURCE_SEED_ROOT") or "").strip()
    seed_root = Path(configured_seed or "/opt/opentulpa-source")
    if not seed_root.is_dir():
        return None
    if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise RuntimeError("host source evolution requires the isolated Linux host image")

    evolution_root = data_root / "bootstrap" / "evolution"
    repository = seed_source_repository(
        seed_root=seed_root,
        repository=evolution_root / "source",
    )
    runtime_root = Path("/tmp") / (
        "opentulpa-evolution-" + hashlib.sha256(str(data_root).encode("utf-8")).hexdigest()[:12]
    )
    if runtime_root.is_symlink():
        raise RuntimeError("host evolution runtime root cannot be a symlink")
    worktrees_root = runtime_root / "worktrees"
    releases_root = runtime_root / "releases"
    worktrees_root.mkdir(parents=True, exist_ok=True, mode=0o711)
    worktrees_root.chmod(0o711)
    releases_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    artifacts_root = evolution_root / "artifacts"
    policy = CandidateSandboxPolicy(
        cpu_limit=settings.sandbox_cpu_limit,
        memory_limit=settings.sandbox_memory_limit,
        pid_limit=settings.sandbox_pid_limit,
        timeout_seconds=max(900, settings.sandbox_timeout_seconds),
        max_output_bytes=settings.sandbox_max_output_bytes,
        network_enabled=True,
    )

    def candidate_backend(workspace: Path) -> CandidateProcessBackend:
        return CandidateProcessBackend(
            workspace=workspace,
            allowed_root=worktrees_root,
            policy=policy,
        )

    evaluator = CandidateEvaluator(
        runner=IsolatedProcessEvaluationRunner(
            pid_limit=settings.sandbox_pid_limit,
            max_output_bytes=settings.sandbox_max_output_bytes,
        ),
        commands=trusted_default_commands(timeout_seconds=900),
    )
    lock_hash = hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
    builder = TrustedSourceOverlayBuilder(
        policy=SourceOverlayBuildPolicy(base_dependency_lock_hash=lock_hash)
    )
    archive = EvolutionArchive(evolution_root / "archive.db")
    activator = SourceOverlayReleaseActivator(
        repository=repository,
        releases_root=releases_root,
        runtime=runtime,
    )
    service = EvolutionSupervisor(
        archive=archive,
        workspaces=GitCandidateWorkspace(
            source_repository=repository,
            worktrees_root=worktrees_root,
            artifacts_root=artifacts_root,
        ),
        candidate_backend_factory=candidate_backend,
        evaluator=evaluator,
        release_pointer=AtomicReleasePointer(evolution_root / "current_release.json"),
        source_ref="HEAD",
        release_builder=builder,
        release_activator=activator,
        event_sink=EvolutionNotificationSink(
            NotificationService(NotificationStore(data_root / "notifications.db"))
        ),
    )
    initial = TrustedSourceReleaseProvider(
        source_repository=repository,
        builder=builder,
        evaluator_version="host-source-install-v1",
        evaluator_fingerprint=evaluator.fingerprint,
    )
    return HostEvolutionRuntime(
        runtime=runtime,
        archive=archive,
        evolution=service,
        activator=activator,
        initial_release=initial,
    )


__all__ = ["build_host_evolution_runtime"]
