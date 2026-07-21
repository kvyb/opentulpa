"""Composition of the fixed self-improvement service inside the bootstrap process."""

from __future__ import annotations

import hashlib
from pathlib import Path

from opentulpa.bootstrap.evolution_runtime import (
    BootstrapEvolutionEventSink,
    ManagedEvolutionRuntime,
    TrustedSourceReleaseProvider,
)
from opentulpa.bootstrap.supervisor import BootstrapSupervisor
from opentulpa.core.config import Settings
from opentulpa.evolution.activation import BootstrapReleaseActivator
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    OciEvaluationRunner,
    trusted_default_commands,
)
from opentulpa.evolution.release import AtomicReleasePointer
from opentulpa.evolution.release_builder import (
    OciReleaseBuildPolicy,
    TrustedOciReleaseBuilder,
)
from opentulpa.evolution.sandbox import (
    CandidateContainerBackend,
    CandidateSandboxPolicy,
    resolve_local_oci_image,
)
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.evolution.workspace import GitCandidateWorkspace


def build_managed_evolution_runtime(
    *,
    bootstrap: BootstrapSupervisor,
    project_root: Path,
    state_root: Path,
    settings: Settings,
    runtime_base_image: str,
) -> ManagedEvolutionRuntime | None:
    """Build evolution in the stable host, never inside the mutable release."""

    if not settings.evolution_enabled:
        return None
    source = (
        Path(settings.evolution_source_repository).expanduser()
        if settings.evolution_source_repository
        else project_root
    ).resolve()
    if not (source / ".git").exists() or not (source / "uv.lock").is_file():
        raise RuntimeError("managed evolution requires a canonical Git source checkout")
    container_cli = settings.sandbox_container_cli
    sandbox_image = resolve_local_oci_image(
        container_cli=container_cli,
        image=settings.evolution_sandbox_image,
        cwd=project_root,
    )
    evaluator_image = resolve_local_oci_image(
        container_cli=container_cli,
        image=settings.evolution_evaluator_image,
        cwd=project_root,
    )
    base_image = resolve_local_oci_image(
        container_cli=container_cli,
        image=runtime_base_image,
        cwd=project_root,
    )
    evolution_root = state_root.expanduser().resolve() / "evolution"
    worktrees_root = evolution_root / "worktrees"
    artifacts_root = evolution_root / "artifacts"
    sandbox_policy = CandidateSandboxPolicy(
        image=sandbox_image,
        cpu_limit=settings.sandbox_cpu_limit,
        memory_limit=settings.sandbox_memory_limit,
        pid_limit=settings.sandbox_pid_limit,
        timeout_seconds=max(300, settings.sandbox_timeout_seconds),
        max_output_bytes=settings.sandbox_max_output_bytes,
        network_enabled=True,
    )

    def candidate_backend(workspace: Path) -> CandidateContainerBackend:
        return CandidateContainerBackend(
            workspace=workspace,
            allowed_root=worktrees_root,
            policy=sandbox_policy,
            container_cli=container_cli,
        )

    evaluator = CandidateEvaluator(
        runner=OciEvaluationRunner(
            image=evaluator_image,
            container_cli=container_cli,
            cpu_limit=settings.sandbox_cpu_limit,
            memory_limit=settings.sandbox_memory_limit,
            pid_limit=settings.sandbox_pid_limit,
            max_output_bytes=settings.sandbox_max_output_bytes,
        ),
        commands=trusted_default_commands(
            timeout_seconds=max(300, settings.sandbox_timeout_seconds),
        ),
    )
    lock_hash = hashlib.sha256((source / "uv.lock").read_bytes()).hexdigest()
    builder = TrustedOciReleaseBuilder(
        policy=OciReleaseBuildPolicy(
            base_image_digest=base_image,
            base_dependency_lock_hash=lock_hash,
            container_cli=container_cli,
            state_root=evolution_root / "release-builds",
        )
    )
    archive = EvolutionArchive(evolution_root / "archive.db")
    service = EvolutionSupervisor(
        archive=archive,
        workspaces=GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=worktrees_root,
            artifacts_root=artifacts_root,
        ),
        candidate_backend_factory=candidate_backend,
        evaluator=evaluator,
        release_pointer=AtomicReleasePointer(evolution_root / "current_release.json"),
        source_ref="HEAD",
        release_builder=builder,
        release_activator=BootstrapReleaseActivator(bootstrap),
        event_sink=BootstrapEvolutionEventSink(bootstrap),
    )
    initial = TrustedSourceReleaseProvider(
        source_repository=source,
        builder=builder,
        evaluator_version="bootstrap-trusted-install-v1",
        evaluator_fingerprint=evaluator.fingerprint,
    )
    return ManagedEvolutionRuntime(
        bootstrap=bootstrap,
        archive=archive,
        evolution=service,
        initial_release=initial,
    )


__all__ = ["build_managed_evolution_runtime"]
