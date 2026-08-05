"""Trusted source release evaluation, activation, rollback, and sharing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import stat
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from pydantic import JsonValue

from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.core.ids import new_short_id
from opentulpa.evolution.activation import (
    ReleaseActivationStatus,
    ReleaseActivator,
)
from opentulpa.evolution.archive import EvolutionArchive, PromotionAttemptConflictError
from opentulpa.evolution.dependency_resolver import DependencyResolver, ResolvedDependencyBase
from opentulpa.evolution.evaluator import CandidateEvaluator, EvaluationCommandResult
from opentulpa.evolution.generation import (
    UPSTREAM_LINEAGE_METADATA_KEY,
    UpstreamLineage,
)
from opentulpa.evolution.lineage import GitLineage, GitLineageError, GitLineageSnapshot
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    ContributionMetadata,
    EvaluationCheck,
    EvaluationReport,
    EvolutionEvent,
    PromotionAttempt,
    PromotionAttemptStatus,
    Release,
    SourceReleaseOperation,
    SourceReleaseOperationStatus,
)
from opentulpa.evolution.release import ReleasePointer
from opentulpa.evolution.release_builder import (
    DependencyAwareWheelReleaseBuilder,
    OciReleaseArtifact,
    ReleaseBuilder,
    ReleaseBuildError,
    ReleaseBuildRequest,
    TrustedWheelReleaseBuilder,
)
from opentulpa.evolution.sanitizer import (
    ContributionSanitizationError,
    sanitize_contribution_patch,
)
from opentulpa.evolution.workspace import (
    CandidateCommit,
    CandidateWorkspace,
    GitCandidateWorkspace,
)

logger = logging.getLogger(__name__)

_GENERATION_REFERENCE_RE = re.compile(r"python-generation:([0-9a-f]{64})\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_VERIFIED_UPSTREAM_MERGE_COMMIT_KEY = "opentulpa.evolution.upstream_merge_commit"


class EvolutionSupervisorError(RuntimeError):
    """Public evolution failure without model, Git, or filesystem internals."""


class EvolutionEventSink(Protocol):
    async def deliver(self, event: EvolutionEvent) -> None: ...


class InMemoryEvolutionEventSink:
    def __init__(self) -> None:
        self.events: list[EvolutionEvent] = []

    async def deliver(self, event: EvolutionEvent) -> None:
        self.events.append(event)


class EvolutionSupervisor:
    """Keep source release and rollback authority outside the mutable application."""

    def __init__(
        self,
        *,
        archive: EvolutionArchive,
        workspaces: GitCandidateWorkspace,
        evaluator: CandidateEvaluator,
        release_pointer: ReleasePointer,
        candidate_backend_factory: Callable[[Path], Any] | None = None,
        evaluator_version: str = "opentulpa-evaluator-v1",
        source_ref: str = "HEAD",
        upstream_repository: str = "https://github.com/kvyb/opentulpa",
        upstream_ref: str = "refs/heads/main",
        promotion_retry_interval_seconds: float = 1.0,
        release_builder: ReleaseBuilder | None = None,
        release_activator: ReleaseActivator | None = None,
        event_sink: EvolutionEventSink | None = None,
        lineage: GitLineage | None = None,
        source_mutation_enabled: bool = True,
        source_mutation_unavailable_reason: str | None = None,
        dependency_resolver: DependencyResolver | None = None,
        dependency_evaluator_factory: Callable[[ResolvedDependencyBase], CandidateEvaluator]
        | None = None,
    ) -> None:
        if not evaluator_version.strip():
            raise ValueError("evaluator_version is required")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        if not upstream_repository.strip():
            raise ValueError("upstream_repository is required")
        if not upstream_ref.strip():
            raise ValueError("upstream_ref is required")
        if not 0.05 <= promotion_retry_interval_seconds <= 60:
            raise ValueError("promotion retry interval must be between 0.05 and 60 seconds")
        self._archive = archive
        self._workspaces = workspaces
        self._candidate_backend_factory = candidate_backend_factory
        self._evaluator = evaluator
        self._release_pointer = release_pointer
        self._evaluator_version = evaluator_version.strip()
        self._source_ref = source_ref.strip()
        self._upstream_repository = upstream_repository.strip()
        self._upstream_ref = upstream_ref.strip()
        self._promotion_retry_interval_seconds = promotion_retry_interval_seconds
        self._activation_lock = asyncio.Lock()
        self._release_builder = release_builder
        self._release_activator = release_activator
        self._event_sink = event_sink
        self._lineage = lineage
        self._source_mutation_enabled = bool(source_mutation_enabled)
        self._source_mutation_unavailable_reason = str(
            source_mutation_unavailable_reason
            or "source mutation is disabled by the stable host policy"
        ).strip()
        self._dependency_resolver = dependency_resolver
        self._dependency_evaluator_factory = dependency_evaluator_factory
        self._dependency_evaluators: dict[str, CandidateEvaluator] = {}
        self._source_locks: dict[str, asyncio.Lock] = {}
        self._promotion_wake = asyncio.Event()
        self._promotion_dispatcher: asyncio.Task[None] | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def source_mutation_enabled(self) -> bool:
        return self._source_mutation_enabled

    async def start(self) -> None:
        if self._started:
            return
        await self._archive.start()
        current = await self._archive.get_current_release()
        if self._source_mutation_enabled:
            await self._reconcile_lineage_projection(current)
        pointer = await self._release_pointer.current()
        if current is None:
            if pointer is not None:
                await self._release_pointer.clear()
        else:
            projected = (
                await self._release_with_accepted_upstream(current)
                if self._source_mutation_enabled
                else current
            )
            if pointer != projected:
                await self._release_pointer.activate(projected)
        if self._source_mutation_enabled:
            stale = await self._archive.list_candidates(
                status=CandidateStatus.BUILDING,
                limit=1_000,
            )
            for candidate in stale:
                if self._is_source_session(candidate):
                    await self._reconcile_source_session(candidate)
                else:
                    await self._cleanup_stale_candidate(candidate)
        await self._ensure_terminal_candidate_events()
        await self._ensure_terminal_promotion_events()
        await self.flush_events()
        self._started = True
        if self._source_mutation_enabled:
            await self._resume_pending_source_releases()
        if self._source_mutation_enabled and self._release_activator is not None:
            self._promotion_wake.set()
            self._promotion_dispatcher = asyncio.create_task(
                self._promotion_dispatch_loop(),
                name="opentulpa-evolution-promotion-dispatcher",
            )

    async def shutdown(self) -> None:
        self._started = False
        dispatcher = self._promotion_dispatcher
        self._promotion_dispatcher = None
        if dispatcher is not None:
            dispatcher.cancel()
            await asyncio.gather(dispatcher, return_exceptions=True)
        await self.flush_events()
        await self._archive.shutdown()

    async def get_candidate(self, candidate_id: str) -> Candidate | None:
        self._require_started()
        return await self._archive.get_candidate(candidate_id)

    async def list_candidates(
        self,
        *,
        status: CandidateStatus | str | None = None,
        limit: int = 100,
    ) -> list[Candidate]:
        self._require_started()
        return await self._archive.list_candidates(status=status, limit=limit)

    async def source_status(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Inspect the single persistent source session owned by one tenant."""

        self._require_started()
        if not self._source_mutation_enabled:
            return {
                "available": False,
                "active": False,
                "session_active": False,
                "candidate_id": None,
                "reason": self._source_mutation_unavailable_reason,
                "source_mutation_enabled": False,
                "dependency_resolution_available": False,
                "diff_sha256": hashlib.sha256(b"").hexdigest(),
                "conflict_paths": [],
            }
        session_key, audit = self._source_context(audit_context)
        async with self._source_lock(session_key):
            lineage = await self._source_lineage()
            candidate = await self._find_source_session(
                session_key,
                tenant_id=str(audit["tenant_id"]),
            )
            if candidate is None:
                return {
                    "available": True,
                    "source_mutation_enabled": True,
                    "dependency_resolution_available": self._dependency_resolver is not None,
                    "active": False,
                    "session_active": False,
                    "candidate_id": None,
                    "diff_sha256": hashlib.sha256(b"").hexdigest(),
                    "conflict_paths": [],
                    **lineage,
                }
            workspace = self._source_workspace(candidate)
            return {
                "available": True,
                "source_mutation_enabled": True,
                "dependency_resolution_available": self._dependency_resolver is not None,
                "session_active": True,
                **await self._source_snapshot(candidate, workspace),
                **lineage,
            }

    async def source_shell(
        self,
        *,
        command: str,
        timeout_seconds: int = 300,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Run one command and return concise candidate status; source_status includes the diff."""

        self._require_started()
        self._require_source_mutation()
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command or len(safe_command) > 100_000:
            raise ValueError("source shell command is invalid")
        timeout = int(timeout_seconds)
        if timeout < 1 or timeout > 3_600:
            raise ValueError("source shell timeout must be between 1 and 3600 seconds")
        backend_factory = self._candidate_backend_factory
        if backend_factory is None:
            raise EvolutionSupervisorError("source shell is unavailable")
        session_key, audit = self._source_context(audit_context)
        async with self._source_lock(session_key):
            candidate, workspace = await self._open_source_session(
                session_key=session_key,
                audit=audit,
            )
            await self._require_current_source_base(candidate)
            conflict_paths = await self._source_conflict_paths(candidate, workspace)
            backend = backend_factory(workspace.path)
            with self._hide_git_metadata(workspace):
                response = await backend.aexecute(safe_command, timeout=timeout)
            if self._lineage is not None and conflict_paths:
                try:
                    await asyncio.to_thread(
                        self._lineage.stage_resolved_conflicts,
                        workspace,
                        conflict_paths,
                    )
                except GitLineageError as exc:
                    raise EvolutionSupervisorError(
                        "source merge conflict resolution is invalid"
                    ) from exc
            current = await self._required_candidate(candidate.id)
            snapshot = await self._source_snapshot(current, workspace, include_diff=False)
            output, output_truncated = self._bounded_text(response.output)
            return {
                **snapshot,
                "exit_code": int(response.exit_code),
                "output": output,
                "output_truncated": bool(response.truncated or output_truncated),
            }

    async def source_sync_upstream(
        self,
        *,
        expected_active_release_id: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Fetch remote main through the controller and open its isolated reconciliation."""

        self._require_started()
        self._require_source_mutation()
        expected_release = str(expected_active_release_id or "").strip()
        if not expected_release or len(expected_release) > 100:
            raise ValueError("expected active release is invalid")
        lineage = self._lineage
        if lineage is None:
            raise EvolutionSupervisorError("source lineage is unavailable")
        session_key, audit = self._source_context(audit_context)
        tenant_id = str(audit["tenant_id"])
        async with self._source_lock(session_key):
            current = await self._archive.get_current_release()
            if current is None or current.id != expected_release:
                raise EvolutionSupervisorError("active release changed before upstream sync")
            if await self._find_source_session(session_key, tenant_id=tenant_id) is not None:
                raise EvolutionSupervisorError(
                    "finish or release the active source session before upstream sync"
                )
            try:
                synced = await asyncio.to_thread(
                    lineage.sync_upstream,
                    self._upstream_repository,
                    self._upstream_ref,
                )
            except (GitLineageError, ValueError) as exc:
                raise EvolutionSupervisorError("remote upstream synchronization failed") from exc
            values = {
                "synced": synced.changed,
                "previous_upstream_commit": synced.previous_commit,
                "upstream_commit": synced.upstream_commit,
            }
            if not synced.changed:
                return {**values, **await self._source_lineage()}
            candidate, workspace = await self._open_source_session(
                session_key=session_key,
                audit=audit,
            )
            return {
                **values,
                "session_active": True,
                **await self._source_snapshot(candidate, workspace),
                **await self._source_lineage(),
            }

    async def source_resolve_dependencies(
        self,
        *,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Resolve an exact dependency proposal through the trusted network worker."""

        self._require_started()
        self._require_source_mutation()
        resolver = self._dependency_resolver
        if resolver is None:
            raise EvolutionSupervisorError("autonomous dependency resolution is unavailable")
        candidate_id = str(expected_candidate_id or "").strip()
        diff_sha256 = str(expected_diff_sha256 or "").strip().lower()
        if not candidate_id or len(candidate_id) > 100:
            raise ValueError("expected source candidate is invalid")
        if len(diff_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in diff_sha256
        ):
            raise ValueError("expected source diff digest is invalid")
        session_key, audit = self._source_context(audit_context)
        async with self._source_lock(session_key):
            candidate, workspace = await self._open_source_session(
                session_key=session_key,
                audit=audit,
            )
            if candidate.id != candidate_id:
                raise EvolutionSupervisorError("source dependency proposal identity changed")
            await self._require_current_source_base(candidate)
            before = await self._source_snapshot(candidate, workspace, include_diff=False)
            if before["diff_sha256"] != diff_sha256 or not before["dirty"]:
                raise EvolutionSupervisorError("source dependency proposal changed")
            if before["conflict_paths"]:
                raise EvolutionSupervisorError("source dependency proposal has unresolved conflicts")
            resolved = await resolver.resolve(workspace.path)
            await asyncio.to_thread(self._install_resolved_lock, workspace.path, resolved)
            current = await self._required_candidate(candidate.id)
            after = await self._source_snapshot(current, workspace)
            return {
                **after,
                "dependency_base_id": resolved.id,
                "dependency_lock_hash": resolved.lock_sha256,
                "dependency_inventory_sha256": resolved.inventory_sha256,
                "dependency_resolver_fingerprint": resolved.resolver_fingerprint,
                "dependency_wheelhouse_sha256": resolved.wheelhouse_sha256,
            }

    async def source_release(
        self,
        *,
        idempotency_key: str,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        message: str = "OpenTulpa self-update",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Commit exact session bytes, run fixed gates, and queue safe activation."""

        self._require_started()
        self._require_source_mutation()
        safe_key = str(idempotency_key or "").strip()
        if not safe_key or len(safe_key) > 200:
            raise ValueError("source release idempotency key is invalid")
        safe_candidate_id = str(expected_candidate_id or "").strip()
        if not safe_candidate_id or len(safe_candidate_id) > 100:
            raise ValueError("expected source candidate is invalid")
        safe_diff_sha256 = str(expected_diff_sha256 or "").strip().lower()
        if len(safe_diff_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in safe_diff_sha256
        ):
            raise ValueError("expected source diff digest is invalid")
        safe_message = " ".join(str(message or "").split())[:500]
        if not safe_message:
            safe_message = "OpenTulpa self-update"
        session_key, audit = self._source_context(audit_context)
        tenant_id = str(audit["tenant_id"])
        async with self._source_lock(session_key):
            operation = await self._archive.get_source_release_operation(
                tenant_id=tenant_id,
                idempotency_key=safe_key,
            )
            if operation is not None:
                if (
                    operation.message != safe_message
                    or operation.candidate_id != safe_candidate_id
                    or operation.expected_diff_sha256 != safe_diff_sha256
                ):
                    raise EvolutionSupervisorError(
                        "source release idempotency key was used for another request"
                    )
                if operation.status is SourceReleaseOperationStatus.COMPLETED:
                    return dict(operation.result or {})
                return await self._execute_source_release(operation)

            candidate = await self._find_source_session(
                session_key,
                tenant_id=tenant_id,
            )
            if candidate is None:
                raise EvolutionSupervisorError("no active source session exists")
            if candidate.id != safe_candidate_id:
                raise EvolutionSupervisorError("source session changed before approval")
            await self._require_current_source_base(candidate)
            workspace = self._source_workspace(candidate)
            self._require_source_diff_binding(
                workspace,
                expected_diff_sha256=safe_diff_sha256,
            )
            current_release = await self._archive.get_current_release()
            operation_digest = hashlib.sha256(f"{tenant_id}\x00{safe_key}".encode()).hexdigest()
            operation = await self._archive.create_source_release_operation(
                SourceReleaseOperation(
                    id=f"source_release_{operation_digest[:48]}",
                    tenant_id=tenant_id,
                    idempotency_key=safe_key,
                    candidate_id=candidate.id,
                    expected_diff_sha256=safe_diff_sha256,
                    base_release_id=(current_release.id if current_release is not None else None),
                    message=safe_message,
                    audit_context=dict(audit),
                )
            )
            return await self._execute_source_release(operation)

    async def source_rollback(
        self,
        *,
        idempotency_key: str,
        expected_current_release_id: str,
        expected_target_release_id: str,
        reason: str = "Owner requested rollback",
        audit_context: Mapping[str, str] | None = None,
    ) -> PromotionAttempt:
        """Idempotently queue the exact owner-approved rollback transition."""

        self._require_started()
        self._require_source_mutation()
        safe_key = str(idempotency_key or "").strip()
        if not safe_key or len(safe_key) > 200:
            raise ValueError("source rollback idempotency key is invalid")
        safe_current = str(expected_current_release_id or "").strip()
        safe_target = str(expected_target_release_id or "").strip()
        if not safe_current or len(safe_current) > 100:
            raise ValueError("expected current release is invalid")
        if not safe_target or len(safe_target) > 100:
            raise ValueError("expected rollback target is invalid")
        safe_reason = str(reason or "").strip()[:4_000] or "Owner requested rollback"
        _, audit = self._source_context(audit_context)
        tenant_id = str(audit["tenant_id"])
        digest = hashlib.sha256(f"{tenant_id}\x00{safe_key}".encode()).hexdigest()
        return await self.queue_rollback(
            reason=safe_reason,
            audit_context=audit,
            attempt_id=f"rollback_{digest[:48]}",
            release_id=f"release_rollback_{digest[:48]}",
            expected_current_release_id=safe_current,
            expected_target_release_id=safe_target,
            idempotency_digest=digest,
            expected_tenant_id=tenant_id,
        )

    async def queue_promotion(
        self,
        candidate_id: str,
        *,
        reason: str = "Owner approved",
        expected_revision: int | None = None,
        audit_context: Mapping[str, str] | None = None,
        attempt_id: str | None = None,
        release_id: str | None = None,
        expected_previous_source_commit: str | None = None,
    ) -> PromotionAttempt:
        self._require_started()
        self._require_source_mutation()
        async with self._activation_lock:
            if attempt_id is not None:
                existing = await self._archive.get_promotion_attempt(attempt_id)
                if existing is not None:
                    if existing.candidate_id != candidate_id or (
                        release_id is not None and existing.release.id != release_id
                    ):
                        raise EvolutionSupervisorError("source release promotion binding changed")
                    return existing
            candidate = await self._required_candidate(candidate_id)
            self._require_revision(candidate, expected_revision)
            if (
                candidate.status is not CandidateStatus.READY
                or candidate.evaluation_report is None
                or not candidate.evaluation_report.passed
                or not candidate.source_commit
                or not candidate.artifact_digest
            ):
                raise EvolutionSupervisorError("candidate is not ready for promotion")
            if self._release_activator is None:
                raise EvolutionSupervisorError("release activation is unavailable")
            current = await self._archive.get_current_release()
            if self._lineage is not None and self._is_source_session(candidate):
                await self._require_current_source_base(candidate)
            if expected_previous_source_commit is not None and (
                current is None or current.source_commit != expected_previous_source_commit
            ):
                raise EvolutionSupervisorError("source session is based on an inactive release")
            audit = self._audit_context(audit_context)
            release_metadata = self._release_artifact_metadata(candidate)
            release_values: dict[str, Any] = {
                "candidate_id": candidate.id,
                "source_commit": candidate.source_commit,
                "artifact_digest": candidate.artifact_digest,
                "previous_release_id": current.id if current is not None else None,
                "reason": str(reason or "")[:4_000],
                "metadata": {
                    "activation_state": "desired",
                    **release_metadata,
                    **(
                        {"requested_by": candidate.metadata["requested_by"]}
                        if "requested_by" in candidate.metadata
                        else {}
                    ),
                    **({"requested_by": audit} if audit else {}),
                },
            }
            if release_id is not None:
                release_values["id"] = release_id
            release = Release.model_validate(release_values)
            attempt_values: dict[str, Any] = {
                "candidate_id": candidate.id,
                "candidate_revision": candidate.revision,
                "release": release,
                "origin": audit or self._candidate_origin(candidate),
            }
            if attempt_id is not None:
                attempt_values["id"] = attempt_id
            try:
                attempt = await self._archive.create_promotion_attempt(
                    PromotionAttempt.model_validate(attempt_values)
                )
            except PromotionAttemptConflictError:
                if attempt_id is None:
                    raise
                existing = await self._archive.get_promotion_attempt(attempt_id)
                if (
                    existing is None
                    or existing.candidate_id != candidate.id
                    or existing.release.id != release.id
                ):
                    raise
                attempt = existing
        self._promotion_wake.set()
        return attempt

    async def queue_rollback(
        self,
        *,
        reason: str = "Owner requested rollback",
        audit_context: Mapping[str, str] | None = None,
        attempt_id: str | None = None,
        release_id: str | None = None,
        expected_current_release_id: str | None = None,
        expected_target_release_id: str | None = None,
        idempotency_digest: str | None = None,
        expected_tenant_id: str | None = None,
    ) -> PromotionAttempt:
        self._require_started()
        self._require_source_mutation()
        async with self._activation_lock:
            if attempt_id is not None:
                existing = await self._archive.get_promotion_attempt(attempt_id)
                if existing is not None:
                    self._validate_source_rollback_replay(
                        existing,
                        release_id=release_id,
                        reason=reason,
                        expected_current_release_id=expected_current_release_id,
                        expected_target_release_id=expected_target_release_id,
                        idempotency_digest=idempotency_digest,
                        expected_tenant_id=expected_tenant_id,
                    )
                    return existing
            current = await self._archive.get_current_release()
            target = await self._archive.get_rollback_target()
            if current is None or target is None:
                raise EvolutionSupervisorError("no rollback target is available")
            await self._require_active_release_lineage(current)
            if (
                expected_current_release_id is not None
                and current.id != expected_current_release_id
            ):
                raise EvolutionSupervisorError("rollback source changed before approval")
            if expected_target_release_id is not None and target.id != expected_target_release_id:
                raise EvolutionSupervisorError("rollback target changed before approval")
            if self._release_activator is None:
                raise EvolutionSupervisorError("release activation is unavailable")
            target_candidate = await self._required_candidate(target.candidate_id)
            audit = self._audit_context(audit_context)
            accepted_upstream = await self._accepted_upstream_for_release(target)
            release_values: dict[str, Any] = {
                "candidate_id": target.candidate_id,
                "source_commit": target.source_commit,
                "artifact_digest": target.artifact_digest,
                "previous_release_id": current.id,
                "reason": str(reason or "")[:4_000],
                "metadata": {
                    "activation_state": "desired",
                    "rollback_of": current.id,
                    "rollback_target": target.id,
                    **self._release_metadata_from_release(target),
                    **(
                        {"accepted_upstream_commit": accepted_upstream}
                        if accepted_upstream is not None
                        else {}
                    ),
                    **(
                        {
                            "source_rollback_idempotency_digest": idempotency_digest,
                            "source_rollback_tenant_id": expected_tenant_id,
                        }
                        if idempotency_digest is not None and expected_tenant_id is not None
                        else {}
                    ),
                    **({"requested_by": audit} if audit else {}),
                },
            }
            if release_id is not None:
                release_values["id"] = release_id
            attempt_values: dict[str, Any] = {
                "candidate_id": target_candidate.id,
                "candidate_revision": target_candidate.revision,
                "release": Release.model_validate(release_values),
                "origin": audit or self._candidate_origin(target_candidate),
            }
            if attempt_id is not None:
                attempt_values["id"] = attempt_id
            try:
                attempt = await self._archive.create_promotion_attempt(
                    PromotionAttempt.model_validate(attempt_values)
                )
            except PromotionAttemptConflictError:
                if attempt_id is None:
                    raise
                existing = await self._archive.get_promotion_attempt(attempt_id)
                if existing is None:
                    raise
                self._validate_source_rollback_replay(
                    existing,
                    release_id=release_id,
                    reason=reason,
                    expected_current_release_id=expected_current_release_id,
                    expected_target_release_id=expected_target_release_id,
                    idempotency_digest=idempotency_digest,
                    expected_tenant_id=expected_tenant_id,
                )
                attempt = existing
        self._promotion_wake.set()
        return attempt

    @staticmethod
    def _validate_source_rollback_replay(
        attempt: PromotionAttempt,
        *,
        release_id: str | None,
        reason: str,
        expected_current_release_id: str | None,
        expected_target_release_id: str | None,
        idempotency_digest: str | None,
        expected_tenant_id: str | None,
    ) -> None:
        metadata = attempt.release.metadata
        if (
            not metadata.get("rollback_target")
            or (release_id is not None and attempt.release.id != release_id)
            or attempt.release.reason != str(reason or "")[:4_000]
            or metadata.get("rollback_of") != expected_current_release_id
            or metadata.get("rollback_target") != expected_target_release_id
            or metadata.get("source_rollback_idempotency_digest") != idempotency_digest
            or metadata.get("source_rollback_tenant_id") != expected_tenant_id
            or attempt.origin.get("tenant_id") != expected_tenant_id
        ):
            raise EvolutionSupervisorError(
                "source rollback idempotency key was used for another request"
            )

    async def get_promotion_attempt(self, attempt_id: str) -> PromotionAttempt | None:
        self._require_started()
        return await self._archive.get_promotion_attempt(attempt_id)

    async def process_queued_promotions(self, *, limit: int = 20) -> int:
        """Run durable attempts outside the request that queued them."""

        self._require_started()
        self._require_source_mutation()
        if self._release_activator is None:
            return 0
        attempts = await self._archive.list_incomplete_promotion_attempts(limit=limit)
        processed = 0
        for listed in attempts:
            async with self._activation_lock:
                attempt = await self._archive.get_promotion_attempt(listed.id)
                if attempt is None or attempt.status not in {
                    PromotionAttemptStatus.QUEUED,
                    PromotionAttemptStatus.ACTIVATING,
                }:
                    continue
                await self._resume_promotion_attempt(attempt)
                processed += 1
        return processed

    async def _promotion_dispatch_loop(self) -> None:
        while True:
            self._promotion_wake.clear()
            try:
                await self.process_queued_promotions()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("evolution promotion dispatch failed")
            try:
                await asyncio.wait_for(
                    self._promotion_wake.wait(),
                    timeout=self._promotion_retry_interval_seconds,
                )
            except TimeoutError:
                continue

    async def _activate_promotion_attempt(
        self,
        attempt: PromotionAttempt,
        *,
        rollback_target: Release | None = None,
        current_candidate_revision: int | None = None,
    ) -> Release:
        activator = self._release_activator
        if activator is None:
            await self._fail_promotion_attempt(
                attempt,
                code="activation_unavailable",
                message="The immutable release bootstrap is unavailable.",
            )
            raise EvolutionSupervisorError("release activation is unavailable")
        try:
            await self._validate_attempt_for_activation(
                attempt,
                rollback_target=rollback_target,
            )
        except EvolutionSupervisorError:
            await self._fail_promotion_attempt(
                attempt,
                code="release_stale",
                message="The release request changed before activation.",
            )
            raise
        if attempt.status is PromotionAttemptStatus.QUEUED:
            attempt = await self._archive.transition_promotion_attempt(
                attempt.id,
                expected_status=PromotionAttemptStatus.QUEUED,
                new_status=PromotionAttemptStatus.ACTIVATING,
                expected_revision=attempt.revision,
                bootstrap_activation_id=attempt.id,
            )
        if attempt.status is not PromotionAttemptStatus.ACTIVATING:
            raise EvolutionSupervisorError("promotion attempt is not resumable")

        await self._publish_build_event(
            event_key=f"promotion:{attempt.id}:switching",
            event_type="build.switching",
            candidate_id=attempt.candidate_id,
            origin=attempt.origin,
            payload={
                "attempt_id": attempt.id,
                "candidate_id": attempt.candidate_id,
                "release_id": attempt.release.id,
                "status": "switching",
                "rollback": rollback_target is not None,
            },
        )
        try:
            result = await activator.activate(
                self._bootstrap_release(attempt.release),
                activation_id=attempt.id,
                origin=self._release_origin(attempt.origin),
                reason=attempt.release.reason,
                rollback=rollback_target is not None,
            )
        except Exception as exc:
            code, message = self._activation_error(exc)
            await self._fail_promotion_attempt(attempt, code=code, message=message)
            raise EvolutionSupervisorError(message) from exc
        if result.status is not ReleaseActivationStatus.ACTIVE:
            code = result.failure_code or "activation_failed"
            message = result.failure_message or "Release activation did not become active."
            await self._fail_promotion_attempt(attempt, code=code, message=message)
            raise EvolutionSupervisorError(message)

        active = attempt.release.model_copy(
            update={
                "metadata": {
                    **attempt.release.metadata,
                    "activation_state": "active",
                    "bootstrap_activation_id": result.activation_id,
                }
            }
        )
        try:
            if rollback_target is None:
                _, recorded = await self._archive.promote_candidate(
                    active,
                    expected_revision=attempt.candidate_revision,
                )
            else:
                rollback_of = str(active.metadata.get("rollback_of") or "")
                if not rollback_of:
                    raise EvolutionSupervisorError("rollback attempt lost its predecessor")
                if current_candidate_revision is None:
                    current = await self._archive.get_current_release()
                    if current is None or current.id != rollback_of:
                        raise EvolutionSupervisorError("rollback predecessor changed")
                    current_candidate = await self._required_candidate(current.candidate_id)
                    current_candidate_revision = current_candidate.revision
                _, recorded = await self._archive.activate_rollback(
                    rollback_target.id,
                    activation=active,
                    expected_current_release_id=rollback_of,
                    expected_current_candidate_revision=current_candidate_revision,
                )
        except Exception as exc:
            logger.exception(
                "bootstrap activated release but archive commit is pending: attempt=%s",
                attempt.id,
            )
            raise EvolutionSupervisorError(
                "release activated; archive reconciliation is pending"
            ) from exc
        try:
            await self._project_release(recorded)
        except Exception as exc:
            logger.exception(
                "archive activated release but projection is pending: attempt=%s",
                attempt.id,
            )
            raise EvolutionSupervisorError(
                "release activated; projection reconciliation is pending"
            ) from exc
        completed_attempt = await self._archive.transition_promotion_attempt(
            attempt.id,
            expected_status=PromotionAttemptStatus.ACTIVATING,
            new_status=PromotionAttemptStatus.ACTIVE,
            expected_revision=attempt.revision,
            bootstrap_activation_id=result.activation_id,
        )
        await self._publish_promotion_event(completed_attempt, active_release=recorded)
        return recorded

    async def _resume_promotion_attempt(self, attempt: PromotionAttempt) -> None:
        candidate = await self._required_candidate(attempt.candidate_id)
        recorded_failure = candidate.metadata.get("last_activation_failure")
        if isinstance(recorded_failure, dict) and str(
            recorded_failure.get("attempt_id") or ""
        ) == attempt.id:
            await self._fail_promotion_attempt(
                attempt,
                code=str(recorded_failure.get("code") or "activation_failed"),
                message=str(
                    recorded_failure.get("message") or "Release activation failed."
                ),
            )
            return
        current = await self._archive.get_current_release()
        if current is not None and current.id == attempt.release.id:
            await self._project_release(current)
            if attempt.status is PromotionAttemptStatus.ACTIVATING:
                attempt = await self._archive.transition_promotion_attempt(
                    attempt.id,
                    expected_status=PromotionAttemptStatus.ACTIVATING,
                    new_status=PromotionAttemptStatus.ACTIVE,
                    expected_revision=attempt.revision,
                    bootstrap_activation_id=attempt.id,
                )
            await self._publish_promotion_event(attempt, active_release=current)
            return
        rollback_target_id = str(attempt.release.metadata.get("rollback_target") or "")
        rollback_target = (
            await self._archive.get_release(rollback_target_id) if rollback_target_id else None
        )
        if rollback_target_id and rollback_target is None:
            await self._fail_promotion_attempt(
                attempt,
                code="rollback_target_missing",
                message="The rollback target is no longer available.",
            )
            return
        try:
            await self._activate_promotion_attempt(attempt, rollback_target=rollback_target)
        except EvolutionSupervisorError:
            logger.warning(
                "promotion attempt did not resume: attempt=%s",
                attempt.id,
                exc_info=True,
            )

    async def _fail_promotion_attempt(
        self,
        attempt: PromotionAttempt,
        *,
        code: str,
        message: str,
    ) -> None:
        safe_code = str(code or "activation_failed")[:100]
        safe_message = str(message or "Release activation failed.")[:2_000]
        current_attempt = await self._archive.get_promotion_attempt(attempt.id)
        if current_attempt is None or current_attempt.status in {
            PromotionAttemptStatus.ACTIVE,
            PromotionAttemptStatus.FAILED,
        }:
            return
        candidate = await self._required_candidate(current_attempt.candidate_id)
        failure_record: dict[str, JsonValue] = {
            "attempt_id": current_attempt.id,
            "code": safe_code,
            "message": safe_message,
        }
        if candidate.metadata.get("last_activation_failure") != failure_record:
            await self._archive.update_candidate(
                candidate.model_copy(
                    update={
                        "metadata": {
                            **candidate.metadata,
                            "last_activation_failure": failure_record,
                        }
                    }
                ),
                expected_revision=candidate.revision,
            )
        failed = await self._archive.transition_promotion_attempt(
            current_attempt.id,
            expected_status=current_attempt.status,
            new_status=PromotionAttemptStatus.FAILED,
            expected_revision=current_attempt.revision,
            bootstrap_activation_id=current_attempt.bootstrap_activation_id or current_attempt.id,
            failure_code=safe_code,
            failure_message=safe_message,
        )
        await self._publish_promotion_event(failed)

    async def _validate_attempt_for_activation(
        self,
        attempt: PromotionAttempt,
        *,
        rollback_target: Release | None,
    ) -> None:
        candidate = await self._required_candidate(attempt.candidate_id)
        if candidate.revision != attempt.candidate_revision:
            raise EvolutionSupervisorError("release request candidate changed")
        current = await self._archive.get_current_release()
        if rollback_target is None:
            current_id = current.id if current is not None else None
            if current_id != attempt.release.previous_release_id:
                raise EvolutionSupervisorError("release predecessor changed")
            report = candidate.evaluation_report
            if (
                candidate.status is not CandidateStatus.READY
                or report is None
                or not report.passed
                or candidate.source_commit != attempt.release.source_commit
                or candidate.artifact_digest != attempt.release.artifact_digest
            ):
                raise EvolutionSupervisorError("release candidate is no longer promotable")
            if self._lineage is not None and self._is_source_session(candidate):
                await self._require_current_source_base(candidate)
            return
        rollback_of = str(attempt.release.metadata.get("rollback_of") or "")
        if current is None or current.id != rollback_of:
            raise EvolutionSupervisorError("rollback predecessor changed")
        await self._require_active_release_lineage(current)
        if (
            str(attempt.release.metadata.get("rollback_target") or "") != rollback_target.id
            or rollback_target.candidate_id != attempt.candidate_id
            or rollback_target.source_commit != attempt.release.source_commit
            or rollback_target.artifact_digest != attempt.release.artifact_digest
            or candidate.status not in {CandidateStatus.PROMOTED, CandidateStatus.ROLLED_BACK}
        ):
            raise EvolutionSupervisorError("rollback target changed")

    async def _require_active_release_lineage(self, release: Release) -> None:
        if self._lineage is None:
            return
        snapshot = await self._lineage_snapshot()
        accepted = await self._accepted_upstream_for_release(release)
        if (
            accepted is None
            or snapshot.instance_commit != release.source_commit
            or snapshot.accepted_upstream_commit != accepted
        ):
            raise EvolutionSupervisorError("active source lineage diverged from the archive")

    @staticmethod
    def _activation_error(exc: Exception) -> tuple[str, str]:
        code = str(getattr(exc, "code", "") or "activation_failed")[:100]
        message = str(getattr(exc, "public_message", "") or "Release activation failed.")[:2_000]
        return code, message

    @staticmethod
    def _release_origin(origin: Mapping[str, JsonValue]) -> ReleaseOrigin | None:
        required = ("tenant_id", "actor_id", "thread_id", "channel", "correlation_id")
        values = {key: str(origin.get(key) or "").strip() for key in required}
        if any(not value for value in values.values()):
            return None
        return ReleaseOrigin(**values)

    @staticmethod
    def _candidate_origin(candidate: Candidate) -> dict[str, JsonValue]:
        value = candidate.metadata.get("requested_by")
        return dict(value) if isinstance(value, dict) else {}

    @staticmethod
    def _release_artifact_metadata(candidate: Candidate) -> dict[str, JsonValue]:
        manifest_digest = str(candidate.metadata.get("manifest_digest") or "")
        raw_entrypoint = candidate.metadata.get("release_entrypoint")
        raw_changed_paths = candidate.metadata.get("changed_paths")
        diff_sha256 = str(candidate.metadata.get("diff_sha256") or "")
        report = candidate.evaluation_report
        if (
            not manifest_digest
            or not isinstance(raw_entrypoint, list)
            or not raw_entrypoint
            or not isinstance(raw_changed_paths, list)
            or not diff_sha256
            or report is None
            or not candidate.evaluator_fingerprint
        ):
            raise EvolutionSupervisorError("candidate has no verified release manifest")
        artifact_kind = str(candidate.metadata.get("artifact_kind") or "")
        if artifact_kind not in {"oci_image", "source_overlay", "python_generation"}:
            raise EvolutionSupervisorError("candidate release artifact kind is invalid")
        accepted_upstream = str(candidate.metadata.get("accepted_upstream_commit") or "")
        evaluation_input_digest = str(candidate.metadata.get("evaluation_input_digest") or "")
        metadata: dict[str, JsonValue] = {
            "artifact_kind": artifact_kind,
            "manifest_digest": manifest_digest,
            "release_entrypoint": [str(item) for item in raw_entrypoint],
            "base_commit": candidate.base_commit,
            "changed_paths": [str(item) for item in raw_changed_paths],
            "diff_sha256": diff_sha256,
            **(
                {"evaluation_input_digest": evaluation_input_digest}
                if evaluation_input_digest
                else {}
            ),
            **(
                {"dependency_lock_hash": candidate.dependency_lock_hash}
                if candidate.dependency_lock_hash is not None
                else {}
            ),
            **(
                {"accepted_upstream_commit": accepted_upstream}
                if accepted_upstream
                else {}
            ),
            "evaluation_report_id": report.id,
            "evaluation_summary": report.summary,
            "evaluator_fingerprint": candidate.evaluator_fingerprint,
            "evaluator_version": report.evaluator_version,
            **(
                {"image_reference": str(candidate.metadata["image_reference"])}
                if "image_reference" in candidate.metadata
                else {}
            ),
        }
        if artifact_kind == "python_generation":
            image_reference = str(candidate.metadata.get("image_reference") or "")
            match = _GENERATION_REFERENCE_RE.fullmatch(image_reference)
            if match is None:
                raise EvolutionSupervisorError("candidate Python generation identity is invalid")
            metadata["generation_id"] = match.group(1)
        for key in (
            "dependency_base_id",
            "dependency_inventory_sha256",
            "dependency_resolver_fingerprint",
            "dependency_site_sha256",
            "dependency_wheelhouse_sha256",
            "state_contract",
            "state_contract_sha256",
            "state_contract_digest",
            "install_profile",
            "controller_protocol",
        ):
            if key in candidate.metadata:
                metadata[key] = candidate.metadata[key]
        return metadata

    @staticmethod
    def _release_metadata_from_release(release: Release) -> dict[str, JsonValue]:
        return {
            key: release.metadata[key]
            for key in (
                "artifact_kind",
                "manifest_digest",
                "release_entrypoint",
                "image_reference",
                "generation_id",
                "dependency_lock_hash",
                "dependency_base_id",
                "dependency_inventory_sha256",
                "dependency_resolver_fingerprint",
                "dependency_site_sha256",
                "dependency_wheelhouse_sha256",
                "evaluation_input_digest",
                "accepted_upstream_commit",
                "state_contract",
                "state_contract_sha256",
                "state_contract_digest",
                "install_profile",
                "controller_protocol",
                "base_commit",
                "changed_paths",
                "diff_sha256",
                "evaluation_report_id",
                "evaluation_summary",
                "evaluator_fingerprint",
                "evaluator_version",
            )
            if key in release.metadata
        }

    @staticmethod
    def _bootstrap_release(release: Release) -> ReleaseRecord:
        manifest_digest = str(release.metadata.get("manifest_digest") or "")
        raw_entrypoint = release.metadata.get("release_entrypoint")
        if not isinstance(raw_entrypoint, list):
            raise EvolutionSupervisorError("release entrypoint is unavailable")
        return ReleaseRecord(
            id=release.id,
            candidate_id=release.candidate_id,
            source_commit=release.source_commit,
            artifact_digest=release.artifact_digest,
            manifest_digest=manifest_digest,
            entrypoint=tuple(str(item) for item in raw_entrypoint),
            metadata={
                key: value for key, value in release.metadata.items() if key != "requested_by"
            },
        )

    async def flush_events(self, *, limit: int = 100) -> int:
        sink = self._event_sink
        if sink is None:
            return 0
        delivered = 0
        for event in await self._archive.pending_events(limit=limit):
            try:
                await sink.deliver(event)
            except Exception:
                logger.exception("evolution event delivery failed: event=%s", event.id)
                await self._archive.mark_event_attempt(event.id, delivered=False)
                continue
            await self._archive.mark_event_attempt(event.id, delivered=True)
            delivered += 1
        return delivered

    async def _ensure_terminal_candidate_events(self) -> None:
        for status in (CandidateStatus.READY, CandidateStatus.FAILED):
            for candidate in await self._archive.list_candidates(status=status, limit=1_000):
                await self._publish_candidate_event(candidate)

    async def _ensure_terminal_promotion_events(self) -> None:
        terminal: dict[str, tuple[PromotionAttempt, Release | None]] = {}
        for release in await self._archive.list_release_history(limit=1_000):
            attempt_id = str(release.metadata.get("bootstrap_activation_id") or "")
            if not attempt_id:
                continue
            attempt = await self._archive.get_promotion_attempt(attempt_id)
            if attempt is not None and attempt.status is PromotionAttemptStatus.ACTIVE:
                terminal[attempt.id] = (attempt, release)
        for candidate in await self._archive.list_candidates(limit=1_000):
            failure = candidate.metadata.get("last_activation_failure")
            if not isinstance(failure, dict):
                continue
            attempt_id = str(failure.get("attempt_id") or "")
            if not attempt_id or attempt_id in terminal:
                continue
            attempt = await self._archive.get_promotion_attempt(attempt_id)
            if attempt is not None and attempt.status is PromotionAttemptStatus.FAILED:
                terminal[attempt.id] = (attempt, None)
        for attempt, active_release in terminal.values():
            await self._publish_promotion_event(attempt, active_release=active_release)

    async def _publish_candidate_event(self, candidate: Candidate) -> None:
        if candidate.status not in {CandidateStatus.READY, CandidateStatus.FAILED}:
            return
        report = candidate.evaluation_report
        completion_id = (
            report.id if report is not None else f"{candidate.status.value}:{candidate.revision}"
        )
        event = EvolutionEvent(
            event_key=f"candidate:{candidate.id}:completed:{completion_id}",
            event_type=(
                "candidate.ready"
                if candidate.status is CandidateStatus.READY
                else "candidate.failed"
            ),
            candidate_id=candidate.id,
            origin=self._candidate_origin(candidate),
            payload={
                "candidate_id": candidate.id,
                "status": candidate.status.value,
                "source_commit": candidate.source_commit,
                "summary": (
                    report.summary
                    if report is not None
                    else "Candidate improvement did not complete."
                ),
                **(
                    {"failure_code": str(candidate.metadata["failure_code"])}
                    if "failure_code" in candidate.metadata
                    else {}
                ),
            },
        )
        await self._archive.enqueue_event(event)
        await self.flush_events()

    async def _publish_build_event(
        self,
        *,
        event_key: str,
        event_type: str,
        candidate_id: str,
        origin: Mapping[str, JsonValue],
        payload: dict[str, JsonValue],
    ) -> None:
        await self._archive.enqueue_event(
            EvolutionEvent(
                event_key=event_key,
                event_type=event_type,
                candidate_id=candidate_id,
                origin=dict(origin),
                payload=payload,
            )
        )
        await self.flush_events()

    async def _publish_promotion_event(
        self,
        attempt: PromotionAttempt,
        *,
        active_release: Release | None = None,
    ) -> None:
        failed = attempt.status is PromotionAttemptStatus.FAILED
        rollback = bool(attempt.release.metadata.get("rollback_target"))
        event = EvolutionEvent(
            event_key=f"promotion:{attempt.id}:terminal",
            event_type=(
                "rollback.failed"
                if failed and rollback
                else "promotion.failed"
                if failed
                else "rollback.active"
                if rollback
                else "promotion.active"
            ),
            candidate_id=attempt.candidate_id,
            origin=attempt.origin,
            payload={
                "attempt_id": attempt.id,
                "candidate_id": attempt.candidate_id,
                "status": attempt.status.value,
                "release_id": (
                    active_release.id if active_release is not None else attempt.release.id
                ),
                **({"failure_code": attempt.failure_code} if attempt.failure_code else {}),
                **({"failure_message": attempt.failure_message} if attempt.failure_message else {}),
            },
        )
        await self._archive.enqueue_event(event)
        await self.flush_events()

    async def prepare_contribution(
        self,
        candidate_id: str,
        *,
        expected_revision: int | None = None,
        audit_context: Mapping[str, str] | None = None,
    ) -> Candidate:
        self._require_started()
        self._require_source_mutation()
        async with self._activation_lock:
            candidate = await self._required_candidate(candidate_id)
            self._require_revision(candidate, expected_revision)
            if not candidate.source_commit or candidate.evaluation_report is None:
                raise EvolutionSupervisorError("candidate has no evaluated source artifact")
            artifact = await asyncio.to_thread(
                self._workspaces.contribution_metadata,
                candidate_id=candidate.id,
                base_commit=candidate.base_commit,
                head_commit=candidate.source_commit,
            )
            try:
                attestation = await asyncio.to_thread(
                    sanitize_contribution_patch,
                    artifact.patch_path,
                    expected_sha256=artifact.patch_sha256,
                )
            except ContributionSanitizationError as exc:
                raise EvolutionSupervisorError(str(exc)) from exc
            audit = self._audit_context(audit_context)
            contribution = ContributionMetadata(
                upstream_repository=self._upstream_repository,
                base_commit=artifact.base_commit,
                branch_name=artifact.branch_name,
                head_commit=artifact.head_commit,
                sanitized=True,
                metadata={
                    "patch_sha256": artifact.patch_sha256,
                    "patch_filename": artifact.patch_path.name,
                    "sanitation_scanner": attestation.scanner_version,
                    "sanitation_bytes": attestation.bytes_scanned,
                    "requires_owner_review": True,
                    **({"prepared_by": audit} if audit else {}),
                },
            )
            updated = candidate.model_copy(update={"contribution": contribution})
            return await self._archive.update_candidate(
                updated,
                expected_revision=candidate.revision,
            )

    async def review_patch(self, candidate_id: str) -> Path:
        """Return a digest-checked immutable patch for authenticated owner review."""

        self._require_started()
        candidate = await self._required_candidate(candidate_id)
        if not candidate.source_commit:
            raise EvolutionSupervisorError("candidate has no source artifact")
        artifact = await asyncio.to_thread(
            self._workspaces.review_artifact,
            candidate_id=candidate.id,
            base_commit=candidate.base_commit,
            head_commit=candidate.source_commit,
        )
        recorded_digest = str(candidate.metadata.get("diff_sha256", "") or "")
        if not recorded_digest or artifact.patch_sha256 != recorded_digest:
            raise EvolutionSupervisorError("candidate review artifact failed digest validation")
        return artifact.patch_path

    async def _build_release_artifact(
        self,
        *,
        candidate_id: str,
        workspace: Path,
        base_commit: str,
        source_commit: str,
        dependency_lock_hash: str | None,
        evaluation_input_digest: str,
        evaluator_fingerprint: str,
    ) -> tuple[OciReleaseArtifact | None, EvaluationCommandResult]:
        started = asyncio.get_running_loop().time()
        builder = self._release_builder
        if builder is None:
            return None, EvaluationCommandResult(
                name="release.artifact",
                stage="build",
                passed=False,
                exit_code=1,
                duration_seconds=0,
                output="Trusted release builder is unavailable.",
            )
        try:
            artifact = await builder.build(
                ReleaseBuildRequest(
                    candidate_id=candidate_id,
                    workspace=workspace,
                    base_commit=base_commit,
                    source_commit=source_commit,
                    dependency_lock_hash=dependency_lock_hash,
                    evaluator_version=self._evaluator_version,
                    evaluator_fingerprint=evaluator_fingerprint,
                    evaluation_input_sha256=evaluation_input_digest.removeprefix("sha256:"),
                )
            )
        except ReleaseBuildError as exc:
            message = str(exc or "Candidate release build failed.")[:4_000]
            return None, EvaluationCommandResult(
                name="release.artifact",
                stage="build",
                passed=False,
                exit_code=1,
                duration_seconds=asyncio.get_running_loop().time() - started,
                output=message,
            )
        except Exception:
            logger.exception("trusted candidate release build failed: candidate=%s", candidate_id)
            return None, EvaluationCommandResult(
                name="release.artifact",
                stage="build",
                passed=False,
                exit_code=1,
                duration_seconds=asyncio.get_running_loop().time() - started,
                output="Candidate release build failed.",
            )
        return artifact, EvaluationCommandResult(
            name="release.artifact",
            stage="build",
            passed=True,
            exit_code=0,
            duration_seconds=asyncio.get_running_loop().time() - started,
            output=f"Verified immutable {artifact.artifact_kind} release manifest.",
        )

    def _artifact_metadata(
        self,
        artifact: OciReleaseArtifact,
        *,
        dependency_lock_hash: str | None,
    ) -> dict[str, JsonValue]:
        metadata: dict[str, JsonValue] = {
            "artifact_kind": artifact.artifact_kind,
            "manifest_digest": artifact.manifest_digest,
            "image_reference": artifact.image_reference,
            "release_entrypoint": list(artifact.entrypoint),
            **(
                {"dependency_lock_hash": dependency_lock_hash}
                if dependency_lock_hash is not None
                else {}
            ),
        }
        if artifact.artifact_kind == "python_generation":
            match = _GENERATION_REFERENCE_RE.fullmatch(artifact.image_reference)
            if match is None:
                raise EvolutionSupervisorError("trusted builder returned an invalid generation")
            metadata["generation_id"] = match.group(1)
            builder = self._release_builder
            if isinstance(builder, TrustedWheelReleaseBuilder):
                policy = builder._policy
                metadata.update(
                    {
                        "state_contract": policy.state_contract.model_dump(mode="json"),
                        "state_contract_sha256": policy.state_contract.sha256(),
                        "install_profile": policy.install_profile,
                        "controller_protocol": policy.state_contract.runtime_protocol,
                    }
                )
            elif isinstance(builder, DependencyAwareWheelReleaseBuilder):
                contract = builder.state_contract
                metadata.update(
                    {
                        "state_contract": contract.model_dump(mode="json"),
                        "state_contract_sha256": contract.sha256(),
                        "install_profile": builder.install_profile,
                        "controller_protocol": contract.runtime_protocol,
                    }
                )
        return metadata

    def _evaluation_report(
        self,
        candidate_id: str,
        *,
        source_commit: str,
        artifact_digest: str,
        evaluator_fingerprint: str,
        results: tuple[EvaluationCommandResult, ...],
        source_release_operation_id: str | None = None,
    ) -> EvaluationReport:
        checks = tuple(
            EvaluationCheck(
                name=f"{result.stage}:{result.name}",
                passed=result.passed,
                summary=result.output[:4_000],
                details={"exit_code": result.exit_code, "stage": result.stage},
                duration_seconds=result.duration_seconds,
            )
            for result in results
        )
        if not checks:
            checks = (
                EvaluationCheck(
                    name="evaluator:no_checks",
                    passed=False,
                    summary="The evaluator returned no checks.",
                ),
            )
        passed = all(check.passed for check in checks if check.required)
        return EvaluationReport(
            candidate_id=candidate_id,
            source_commit=source_commit,
            artifact_digest=artifact_digest,
            evaluator_fingerprint=evaluator_fingerprint,
            evaluator_version=self._evaluator_version,
            passed=passed,
            checks=checks,
            summary="Candidate passed all required checks." if passed else "Candidate failed.",
            metadata=(
                {"source_release_operation_id": source_release_operation_id}
                if source_release_operation_id is not None
                else {}
            ),
        )

    async def _fail_candidate(
        self,
        candidate_id: str,
        workspace: CandidateWorkspace,
        *,
        code: str,
    ) -> None:
        try:
            if workspace.path.exists():
                git_path = workspace.path / ".git"
                if not git_path.exists():
                    hidden = workspace.path.parent / f".{workspace.candidate_id}.git-link"
                    if hidden.exists():
                        os.replace(hidden, git_path)
                await asyncio.to_thread(self._workspaces.remove, workspace)
        except Exception:
            logger.exception("failed to clean candidate worktree: candidate=%s", candidate_id)
        candidate = await self._archive.get_candidate(candidate_id)
        if candidate is None:
            return
        try:
            candidate = await self._archive.update_candidate(
                candidate.model_copy(
                    update={
                        "worktree_path": None,
                        "metadata": {**candidate.metadata, "failure_code": code},
                    }
                ),
                expected_revision=candidate.revision,
            )
        except Exception:
            logger.exception("failed to record candidate failure: candidate=%s", candidate_id)
            candidate = await self._archive.get_candidate(candidate_id)
            if candidate is None:
                return
        if candidate.status is CandidateStatus.BUILDING:
            try:
                candidate = await self._archive.transition_status(
                    candidate.id,
                    expected_status=CandidateStatus.BUILDING,
                    new_status=CandidateStatus.FAILED,
                    expected_revision=candidate.revision,
                )
            except Exception:
                logger.exception(
                    "failed to transition candidate failure: candidate=%s", candidate_id
                )
        if candidate.status is CandidateStatus.FAILED:
            try:
                await self._publish_candidate_event(candidate)
            except Exception:
                logger.exception(
                    "failed to record candidate completion event: candidate=%s", candidate_id
                )

    async def _execute_source_release(
        self,
        operation: SourceReleaseOperation,
    ) -> dict[str, Any]:
        if operation.status is SourceReleaseOperationStatus.COMPLETED:
            return dict(operation.result or {})
        candidate = await self._required_candidate(operation.candidate_id)
        if candidate.status in {CandidateStatus.READY, CandidateStatus.PROMOTED}:
            if candidate.metadata.get("diff_sha256") != operation.expected_diff_sha256:
                raise EvolutionSupervisorError("source release approval binding changed")
            return await self._finish_source_release(operation, candidate)
        if candidate.status is not CandidateStatus.BUILDING or not self._is_source_session(
            candidate
        ):
            raise EvolutionSupervisorError("source release candidate is not resumable")
        current_release = await self._archive.get_current_release()
        if (
            current_release.id if current_release is not None else None
        ) != operation.base_release_id or (
            current_release is not None and current_release.source_commit != candidate.base_commit
        ):
            raise EvolutionSupervisorError("source session is based on an inactive release")
        await self._require_current_source_base(candidate)

        workspace = self._source_workspace(candidate)
        self._require_source_diff_binding(
            workspace,
            expected_diff_sha256=operation.expected_diff_sha256,
        )
        commit = await self._source_commit(
            candidate=candidate,
            workspace=workspace,
            message=operation.message,
        )
        if commit.diff_sha256 != operation.expected_diff_sha256:
            raise EvolutionSupervisorError("source release approval binding changed")
        current, lock_hash, evaluator = await self._bind_source_commit(
            candidate=candidate,
            workspace=workspace,
            commit=commit,
        )
        prior_report = current.evaluation_report
        if (
            prior_report is not None
            and prior_report.source_commit == commit.source_commit
            and prior_report.artifact_digest == current.artifact_digest
            and prior_report.evaluator_fingerprint == evaluator.fingerprint
            and prior_report.evaluator_version == self._evaluator_version
            and prior_report.metadata.get("source_release_operation_id") == operation.id
        ):
            if not prior_report.passed:
                snapshot = await self._source_snapshot(current, workspace)
                return await self._complete_source_release_operation(
                    operation,
                    {**snapshot, "promotion": None},
                )
            ready = await self._archive.transition_status(
                current.id,
                expected_status=CandidateStatus.BUILDING,
                new_status=CandidateStatus.READY,
                expected_revision=current.revision,
            )
            return await self._finish_source_release(operation, ready)

        await self._publish_build_event(
            event_key=f"source-release:{operation.id}:preparing",
            event_type="build.preparing",
            candidate_id=current.id,
            origin=operation.audit_context,
            payload={
                "candidate_id": current.id,
                "operation_id": operation.id,
                "status": "preparing",
            },
        )
        command_results = await evaluator.evaluate(workspace.path)
        await self._assert_source_unchanged(workspace, commit.source_commit, "evaluation")
        await self._require_current_source_base(current)
        artifact: OciReleaseArtifact | None = None
        if all(result.passed for result in command_results):
            evaluation_input_digest = str(current.metadata["evaluation_input_digest"])
            artifact, build_result = await self._build_release_artifact(
                candidate_id=current.id,
                workspace=workspace.path,
                base_commit=commit.base_commit,
                source_commit=commit.source_commit,
                dependency_lock_hash=lock_hash,
                evaluation_input_digest=evaluation_input_digest,
                evaluator_fingerprint=evaluator.fingerprint,
            )
            command_results = (*command_results, build_result)
        await self._assert_source_unchanged(workspace, commit.source_commit, "release build")
        await self._require_current_source_base(current)
        evaluation_input_digest = str(current.metadata["evaluation_input_digest"])
        artifact_digest = (
            artifact.artifact_digest if artifact is not None else evaluation_input_digest
        )
        metadata = dict(current.metadata)
        if artifact is not None:
            metadata.update(
                self._artifact_metadata(
                    artifact,
                    dependency_lock_hash=lock_hash,
                )
            )
        current = await self._archive.update_candidate(
            current.model_copy(
                update={
                    "artifact_digest": artifact_digest,
                    "metadata": metadata,
                }
            ),
            expected_revision=current.revision,
        )
        report = self._evaluation_report(
            current.id,
            source_commit=commit.source_commit,
            artifact_digest=artifact_digest,
            evaluator_fingerprint=evaluator.fingerprint,
            results=command_results,
            source_release_operation_id=operation.id,
        )
        current = await self._archive.append_evaluation(
            report,
            expected_revision=current.revision,
        )
        if not report.passed:
            snapshot = await self._source_snapshot(current, workspace)
            return await self._complete_source_release_operation(
                operation,
                {**snapshot, "promotion": None},
            )
        ready = await self._archive.transition_status(
            current.id,
            expected_status=CandidateStatus.BUILDING,
            new_status=CandidateStatus.READY,
            expected_revision=current.revision,
        )
        return await self._finish_source_release(operation, ready)

    async def _bind_source_commit(
        self,
        *,
        candidate: Candidate,
        workspace: CandidateWorkspace,
        commit: CandidateCommit,
    ) -> tuple[Candidate, str | None, CandidateEvaluator]:
        lock_hash = self._dependency_lock_hash(workspace.path)
        evaluator, dependency_base = self._evaluator_for_lock(lock_hash)
        evaluation_input_digest = self._evaluation_input_digest(
            source_commit=commit.source_commit,
            dependency_lock_hash=lock_hash,
            evaluator_version=self._evaluator_version,
            evaluator_fingerprint=evaluator.fingerprint,
        )
        current = await self._required_candidate(candidate.id)
        verified_merge = await self._verify_candidate_merge_commit(
            current,
            commit.source_commit,
        )
        old_paths = current.metadata.get("changed_paths")
        changed_paths = sorted(
            {
                *(str(path) for path in old_paths if isinstance(path, str)),
                *commit.changed_paths,
            }
            if isinstance(old_paths, list)
            else set(commit.changed_paths)
        )
        source_changed = current.source_commit != commit.source_commit
        metadata: dict[str, JsonValue] = dict(current.metadata)
        if verified_merge is None:
            metadata.pop(_VERIFIED_UPSTREAM_MERGE_COMMIT_KEY, None)
        else:
            metadata[_VERIFIED_UPSTREAM_MERGE_COMMIT_KEY] = verified_merge
        if source_changed:
            for key in (
                "artifact_kind",
                "image_reference",
                "generation_id",
                "manifest_digest",
                "release_entrypoint",
            ):
                metadata.pop(key, None)
        for key in (
            "dependency_base_id",
            "dependency_inventory_sha256",
            "dependency_resolver_fingerprint",
            "dependency_site_sha256",
            "dependency_wheelhouse_sha256",
        ):
            metadata.pop(key, None)
        if dependency_base is not None:
            metadata.update(
                {
                    "dependency_base_id": dependency_base.id,
                    "dependency_inventory_sha256": dependency_base.inventory_sha256,
                    "dependency_resolver_fingerprint": dependency_base.resolver_fingerprint,
                    "dependency_site_sha256": dependency_base.site_sha256,
                    "dependency_wheelhouse_sha256": dependency_base.wheelhouse_sha256,
                }
            )
        metadata.update(
            {
                "changed_paths": list(changed_paths),
                "diff_sha256": commit.diff_sha256,
                "evaluation_input_digest": evaluation_input_digest,
                "promotion_eligible": bool(
                    commit.promotion_eligible and current.metadata.get("promotion_eligible", True)
                ),
            }
        )
        if (
            current.source_commit == commit.source_commit
            and current.dependency_lock_hash == lock_hash
            and current.evaluator_fingerprint == evaluator.fingerprint
            and current.metadata == metadata
        ):
            return current, lock_hash, evaluator
        current = await self._archive.update_candidate(
            current.model_copy(
                update={
                    "source_commit": commit.source_commit,
                    "dependency_lock_hash": lock_hash,
                    "artifact_digest": None if source_changed else current.artifact_digest,
                    "evaluator_fingerprint": evaluator.fingerprint,
                    "metadata": metadata,
                }
            ),
            expected_revision=current.revision,
        )
        return current, lock_hash, evaluator

    async def _finish_source_release(
        self,
        operation: SourceReleaseOperation,
        candidate: Candidate,
    ) -> dict[str, Any]:
        attempt_digest = hashlib.sha256(f"{operation.id}:promotion".encode()).hexdigest()
        attempt_id = f"promotion_{attempt_digest[:48]}"
        release_id = f"release_{attempt_digest[:48]}"
        existing = await self._archive.get_promotion_attempt(attempt_id)
        if existing is None:
            await self._require_current_source_base(candidate)
            current_release = await self._archive.get_current_release()
            current_release_id = current_release.id if current_release is not None else None
            if current_release_id != operation.base_release_id:
                raise EvolutionSupervisorError("source session is based on an inactive release")
            candidate = await self._cleanup_released_source_workspace(candidate)
            await self._publish_candidate_event(candidate)
        candidate_data = self._source_candidate_data(candidate)
        promotion = await self.queue_promotion(
            candidate.id,
            reason=operation.message,
            expected_revision=candidate.revision,
            audit_context={key: str(value) for key, value in operation.audit_context.items()},
            attempt_id=attempt_id,
            release_id=release_id,
            expected_previous_source_commit=(
                candidate.base_commit if operation.base_release_id is not None else None
            ),
        )
        result = {
            "active": False,
            "candidate": candidate_data,
            "promotion": promotion.model_dump(mode="json"),
        }
        return await self._complete_source_release_operation(
            operation,
            result,
            promotion_attempt_id=promotion.id,
        )

    async def _cleanup_released_source_workspace(self, candidate: Candidate) -> Candidate:
        if not candidate.worktree_path:
            return candidate
        workspace = CandidateWorkspace(
            candidate_id=candidate.id,
            path=Path(candidate.worktree_path),
            base_commit=candidate.base_commit,
        )
        try:
            self._restore_git_metadata(workspace)
            await asyncio.to_thread(self._workspaces.remove, workspace)
        except Exception:
            logger.exception(
                "failed to remove released source session: candidate=%s",
                candidate.id,
            )
            return candidate
        return await self._archive.update_candidate(
            candidate.model_copy(update={"worktree_path": None}),
            expected_revision=candidate.revision,
        )

    async def _complete_source_release_operation(
        self,
        operation: SourceReleaseOperation,
        result: dict[str, Any],
        *,
        promotion_attempt_id: str | None = None,
    ) -> dict[str, Any]:
        completed = await self._archive.complete_source_release_operation(
            operation.id,
            expected_revision=operation.revision,
            result=result,
            promotion_attempt_id=promotion_attempt_id,
        )
        return dict(completed.result or {})

    async def _resume_pending_source_releases(self) -> None:
        operations = await self._archive.list_pending_source_release_operations(limit=1_000)
        for operation in operations:
            session_key = hashlib.sha256(operation.tenant_id.encode("utf-8")).hexdigest()
            try:
                async with self._source_lock(session_key):
                    current = await self._archive.get_source_release_operation(
                        tenant_id=operation.tenant_id,
                        idempotency_key=operation.idempotency_key,
                    )
                    if (
                        current is not None
                        and current.status is SourceReleaseOperationStatus.PENDING
                    ):
                        await self._execute_source_release(current)
            except Exception:
                candidate = await self._archive.get_candidate(operation.candidate_id)
                if candidate is None or candidate.status is CandidateStatus.FAILED:
                    current = await self._archive.get_source_release_operation(
                        tenant_id=operation.tenant_id,
                        idempotency_key=operation.idempotency_key,
                    )
                    if (
                        current is not None
                        and current.status is SourceReleaseOperationStatus.PENDING
                    ):
                        await self._complete_source_release_operation(
                            current,
                            {
                                "active": False,
                                "candidate_id": current.candidate_id,
                                "candidate": (
                                    {
                                        "id": candidate.id,
                                        "status": candidate.status.value,
                                    }
                                    if candidate is not None
                                    else None
                                ),
                                "promotion": None,
                                "error": {
                                    "code": "source_release_unrecoverable",
                                    "message": (
                                        "Source release could not be recovered; start a new "
                                        "source session."
                                    ),
                                    "retryable": False,
                                },
                            },
                        )
                    logger.warning(
                        "unrecoverable source release was closed: operation=%s",
                        operation.id,
                    )
                else:
                    logger.exception(
                        "source release recovery remains pending: operation=%s",
                        operation.id,
                    )

    async def _require_current_source_base(self, candidate: Candidate) -> None:
        current = await self._archive.get_current_release()
        if current is not None and current.source_commit != candidate.base_commit:
            raise EvolutionSupervisorError("source session is based on an inactive release")
        lineage = self._lineage
        if lineage is None:
            return
        if current is None:
            raise EvolutionSupervisorError("source lineage has no active release")
        snapshot = await self._lineage_snapshot()
        accepted = await self._accepted_upstream_for_release(current)
        recorded = self._candidate_upstream_lineage(candidate)
        target_accepted = str(candidate.metadata.get("accepted_upstream_commit") or "")
        allowed_targets = {
            snapshot.accepted_upstream_commit,
            str(recorded.upstream_commit or ""),
        }
        if (
            snapshot.instance_commit != candidate.base_commit
            or accepted is None
            or snapshot.accepted_upstream_commit != accepted
            or recorded.upstream_commit != snapshot.upstream_commit
            or recorded.merge_base_commit != snapshot.merge_base_commit
            or target_accepted not in allowed_targets
        ):
            raise EvolutionSupervisorError("source lineage changed before release")

    async def _lineage_snapshot(self) -> GitLineageSnapshot:
        lineage = self._lineage
        if lineage is None:
            raise EvolutionSupervisorError("source lineage is unavailable")
        try:
            return await asyncio.to_thread(lineage.snapshot)
        except GitLineageError as exc:
            raise EvolutionSupervisorError("source lineage is unavailable") from exc

    @staticmethod
    def _candidate_upstream_lineage(candidate: Candidate) -> UpstreamLineage:
        raw = candidate.metadata.get(UPSTREAM_LINEAGE_METADATA_KEY)
        try:
            return UpstreamLineage.model_validate(raw)
        except (TypeError, ValueError) as exc:
            raise EvolutionSupervisorError("candidate upstream lineage is invalid") from exc

    async def _adopt_source_lineage(
        self,
        candidate: Candidate,
        workspace: CandidateWorkspace,
    ) -> Candidate:
        lineage = self._lineage
        if lineage is None:
            return candidate
        if UPSTREAM_LINEAGE_METADATA_KEY in candidate.metadata:
            self._candidate_upstream_lineage(candidate)
            return candidate
        snapshot = await self._lineage_snapshot()
        if snapshot.instance_commit != candidate.base_commit:
            raise EvolutionSupervisorError("source session is based on a stale instance")
        try:
            native = await asyncio.to_thread(lineage.inspect_native_merge, workspace)
        except GitLineageError as exc:
            raise EvolutionSupervisorError("source session native merge state is invalid") from exc
        recorded = native.upstream_lineage if native is not None else snapshot.upstream_lineage
        accepted = snapshot.accepted_upstream_commit
        verified_merge: str | None = None
        if native is not None:
            accepted = native.upstream_commit
        else:
            head = await asyncio.to_thread(self._workspaces.head, workspace)
            if head != candidate.base_commit:
                try:
                    verified_merge = await asyncio.to_thread(
                        lineage.verify_merged_tip,
                        head,
                        instance_commit=candidate.base_commit,
                        upstream_commit=snapshot.upstream_commit,
                    )
                except GitLineageError:
                    verified_merge = None
                else:
                    accepted = snapshot.upstream_commit
        metadata: dict[str, JsonValue] = {
            **candidate.metadata,
            UPSTREAM_LINEAGE_METADATA_KEY: recorded.model_dump(mode="json"),
            "accepted_upstream_commit": accepted,
        }
        if verified_merge is not None:
            metadata[_VERIFIED_UPSTREAM_MERGE_COMMIT_KEY] = verified_merge
        return await self._archive.update_candidate(
            candidate.model_copy(update={"metadata": metadata}),
            expected_revision=candidate.revision,
        )

    @staticmethod
    def _is_source_session(candidate: Candidate) -> bool:
        return candidate.metadata.get("source_session") is True

    async def _reconcile_source_session(self, candidate: Candidate) -> None:
        """Keep a valid interactive worktree across bootstrap restarts."""

        if not candidate.worktree_path:
            await self._archive.transition_status(
                candidate.id,
                expected_status=CandidateStatus.BUILDING,
                new_status=CandidateStatus.FAILED,
                expected_revision=candidate.revision,
            )
            return
        workspace = CandidateWorkspace(
            candidate_id=candidate.id,
            path=Path(candidate.worktree_path),
            base_commit=candidate.base_commit,
        )
        try:
            self._restore_git_metadata(workspace)
            candidate = await self._adopt_source_lineage(candidate, workspace)
            head = await asyncio.to_thread(self._workspaces.head, workspace)
            expected_head = candidate.source_commit or candidate.base_commit
            if head != expected_head:
                operation = await self._archive.get_pending_source_release_operation(candidate.id)
                if operation is not None:
                    commit = await asyncio.to_thread(
                        self._workspaces.recover_commit,
                        workspace,
                    )
                    if commit.diff_sha256 != operation.expected_diff_sha256:
                        raise EvolutionSupervisorError(
                            "recovered source commit does not match its approval"
                        )
                    await self._verify_candidate_merge_commit(candidate, commit.source_commit)
                    await self._bind_source_commit(
                        candidate=candidate,
                        workspace=workspace,
                        commit=commit,
                    )
                else:
                    raise EvolutionSupervisorError("source session commit changed unexpectedly")
            elif self._lineage is not None:
                if candidate.source_commit is not None:
                    await self._verify_candidate_merge_commit(candidate, head)
                else:
                    recorded = self._candidate_upstream_lineage(candidate)
                    try:
                        native = await asyncio.to_thread(
                            self._lineage.inspect_native_merge,
                            workspace,
                        )
                    except GitLineageError as exc:
                        raise EvolutionSupervisorError(
                            "source session native merge state changed"
                        ) from exc
                    if native is not None and native.upstream_lineage != recorded:
                        raise EvolutionSupervisorError(
                            "source session native merge state changed"
                        )
        except Exception:
            logger.exception(
                "interactive source session recovery failed: candidate=%s",
                candidate.id,
            )
            await self._fail_candidate(
                candidate.id,
                workspace,
                code="source_session_recovery_failed",
            )

    def _source_context(
        self,
        audit_context: Mapping[str, str] | None,
    ) -> tuple[str, dict[str, str]]:
        sanitized = self._audit_context(audit_context)
        audit = {key: str(value) for key, value in sanitized.items()}
        tenant_id = audit.get("tenant_id", "")
        if not tenant_id:
            raise EvolutionSupervisorError("source session context is incomplete")
        digest = hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()
        return digest, audit

    def _source_lock(self, session_key: str) -> asyncio.Lock:
        return self._source_locks.setdefault(session_key, asyncio.Lock())

    async def _find_source_session(
        self,
        session_key: str,
        *,
        tenant_id: str,
    ) -> Candidate | None:
        candidates = await self._archive.list_candidates(
            status=CandidateStatus.BUILDING,
            limit=1_000,
        )
        matches: list[Candidate] = []
        for candidate in candidates:
            requested_by = candidate.metadata.get("requested_by")
            if self._is_source_session(candidate) and (
                candidate.metadata.get("source_session_key") == session_key
                or candidate.metadata.get("source_tenant_id") == tenant_id
                or (isinstance(requested_by, dict) and requested_by.get("tenant_id") == tenant_id)
            ):
                matches.append(candidate)
        if len(matches) > 1:
            raise EvolutionSupervisorError("source session state is ambiguous")
        return matches[0] if matches else None

    async def _open_source_session(
        self,
        *,
        session_key: str,
        audit: Mapping[str, str],
    ) -> tuple[Candidate, CandidateWorkspace]:
        tenant_id = str(audit["tenant_id"])
        existing = await self._find_source_session(
            session_key,
            tenant_id=tenant_id,
        )
        if existing is not None:
            return existing, self._source_workspace(existing)
        current_release = await self._archive.get_current_release()
        lineage_snapshot: GitLineageSnapshot | None = None
        merge_prepared = False
        if self._lineage is not None:
            if current_release is None:
                raise EvolutionSupervisorError("source lineage has no active release")
            lineage_snapshot = await self._lineage_snapshot()
            if lineage_snapshot.instance_commit != current_release.source_commit:
                raise EvolutionSupervisorError("active source lineage is inconsistent")
            base_ref = lineage_snapshot.instance_commit
            try:
                incorporated = await asyncio.to_thread(
                    self._lineage.is_ancestor,
                    lineage_snapshot.upstream_commit,
                    lineage_snapshot.instance_commit,
                )
            except GitLineageError as exc:
                raise EvolutionSupervisorError("source lineage is unavailable") from exc
            merge_prepared = not incorporated
        else:
            base_ref = (
                current_release.source_commit if current_release is not None else self._source_ref
            )
        candidate_id = new_short_id("candidate", suffix_chars=12)
        workspace = await asyncio.to_thread(
            self._workspaces.create,
            candidate_id=candidate_id,
            base_ref=base_ref,
        )
        metadata: dict[str, JsonValue] = {
            "source_session": True,
            "source_session_key": session_key,
            "source_tenant_id": tenant_id,
            "requested_by": dict(audit),
        }
        if lineage_snapshot is not None:
            metadata.update(
                {
                    UPSTREAM_LINEAGE_METADATA_KEY: (
                        lineage_snapshot.upstream_lineage.model_dump(mode="json")
                    ),
                    "accepted_upstream_commit": (
                        lineage_snapshot.upstream_commit
                        if merge_prepared
                        else lineage_snapshot.accepted_upstream_commit
                    ),
                }
            )
        candidate = Candidate(
            id=candidate_id,
            base_commit=workspace.base_commit,
            requested_improvement="Interactive OpenTulpa source session",
            worktree_path=str(workspace.path),
            metadata=metadata,
        )
        try:
            if merge_prepared:
                assert self._lineage is not None and lineage_snapshot is not None
                await asyncio.to_thread(
                    self._lineage.prepare_merge,
                    workspace,
                    lineage_snapshot.upstream_lineage,
                )
            await self._archive.create_candidate(candidate)
        except GitLineageError as exc:
            await asyncio.to_thread(self._workspaces.remove, workspace)
            raise EvolutionSupervisorError("upstream merge could not be prepared") from exc
        except Exception:
            await asyncio.to_thread(self._workspaces.remove, workspace)
            raise
        return candidate, workspace

    def _source_workspace(self, candidate: Candidate) -> CandidateWorkspace:
        if (
            candidate.status is not CandidateStatus.BUILDING
            or not self._is_source_session(candidate)
            or not candidate.worktree_path
        ):
            raise EvolutionSupervisorError("source session is unavailable")
        workspace = CandidateWorkspace(
            candidate_id=candidate.id,
            path=Path(candidate.worktree_path),
            base_commit=candidate.base_commit,
        )
        self._restore_git_metadata(workspace)
        return workspace

    @staticmethod
    def _restore_git_metadata(workspace: CandidateWorkspace) -> None:
        git_path = workspace.path / ".git"
        hidden = workspace.path.parent / f".{workspace.candidate_id}.git-link"
        if hidden.exists():
            if os.path.lexists(git_path):
                raise EvolutionSupervisorError("candidate Git metadata is ambiguous")
            os.replace(hidden, git_path)

    async def _source_commit(
        self,
        *,
        candidate: Candidate,
        workspace: CandidateWorkspace,
        message: str,
    ) -> CandidateCommit:
        if self._lineage is not None:
            recorded = self._candidate_upstream_lineage(candidate)
            try:
                merge = await asyncio.to_thread(
                    self._lineage.inspect_native_merge,
                    workspace,
                )
            except GitLineageError as exc:
                raise EvolutionSupervisorError(
                    "source session native merge state changed"
                ) from exc
            if merge is not None:
                if merge.upstream_lineage != recorded:
                    raise EvolutionSupervisorError("source session native merge state changed")
                if merge.conflicted_paths:
                    raise EvolutionSupervisorError("source merge has unresolved conflicts")
        status = await asyncio.to_thread(self._workspaces.status, workspace)
        if status:
            commit = await asyncio.to_thread(
                self._workspaces.commit,
                workspace,
                message=message,
            )
            await self._verify_candidate_merge_commit(candidate, commit.source_commit)
            return commit
        head = await asyncio.to_thread(self._workspaces.head, workspace)
        if head == candidate.base_commit:
            raise EvolutionSupervisorError("source session has no changes to release")
        if candidate.source_commit is None:
            commit = await asyncio.to_thread(self._workspaces.recover_commit, workspace)
            await self._verify_candidate_merge_commit(candidate, commit.source_commit)
            return commit
        if candidate.source_commit != head:
            raise EvolutionSupervisorError("source session commit changed unexpectedly")
        await self._verify_candidate_merge_commit(candidate, head)
        diff_sha256 = str(candidate.metadata.get("diff_sha256") or "")
        raw_paths = candidate.metadata.get("changed_paths")
        changed_paths = (
            tuple(str(path) for path in raw_paths if isinstance(path, str))
            if isinstance(raw_paths, list)
            else ()
        )
        if len(diff_sha256) != 64 or not changed_paths:
            raise EvolutionSupervisorError("source session commit evidence is unavailable")
        return CandidateCommit(
            candidate_id=candidate.id,
            base_commit=candidate.base_commit,
            source_commit=head,
            diff_sha256=diff_sha256,
            changed_paths=changed_paths,
            promotion_eligible=bool(candidate.metadata.get("promotion_eligible", False)),
        )

    async def _verify_candidate_merge_commit(
        self,
        candidate: Candidate,
        source_commit: str,
    ) -> str | None:
        lineage = self._lineage
        if lineage is None:
            return None
        recorded = self._candidate_upstream_lineage(candidate)
        if recorded.upstream_commit is None:
            raise EvolutionSupervisorError("candidate upstream merge state is invalid")
        snapshot = await self._lineage_snapshot()
        target_accepted = str(candidate.metadata.get("accepted_upstream_commit") or "")
        if target_accepted == snapshot.accepted_upstream_commit:
            return None
        if target_accepted != recorded.upstream_commit:
            raise EvolutionSupervisorError("candidate accepted upstream is invalid")
        raw_verified = str(
            candidate.metadata.get(_VERIFIED_UPSTREAM_MERGE_COMMIT_KEY) or ""
        )
        if raw_verified and _COMMIT_RE.fullmatch(raw_verified) is None:
            raise EvolutionSupervisorError("candidate verified merge metadata is invalid")
        try:
            return await asyncio.to_thread(
                lineage.verify_merged_tip,
                source_commit,
                instance_commit=candidate.base_commit,
                upstream_commit=recorded.upstream_commit,
                expected_merge_commit=raw_verified or None,
            )
        except GitLineageError as exc:
            raise EvolutionSupervisorError("source merge commit is invalid") from exc

    async def _assert_source_unchanged(
        self,
        workspace: CandidateWorkspace,
        expected_head: str,
        operation: str,
    ) -> None:
        if await asyncio.to_thread(self._workspaces.status, workspace):
            raise EvolutionSupervisorError(f"{operation} modified candidate source")
        if await asyncio.to_thread(self._workspaces.head, workspace) != expected_head:
            raise EvolutionSupervisorError(f"{operation} changed candidate commit")

    async def _source_snapshot(
        self,
        candidate: Candidate,
        workspace: CandidateWorkspace,
        *,
        include_diff: bool = True,
    ) -> dict[str, Any]:
        conflict_paths = await self._source_conflict_paths(candidate, workspace)
        diff = (
            "".join(f"unresolved upstream conflict: {path}\n" for path in conflict_paths)
            if conflict_paths
            else await asyncio.to_thread(self._source_diff, workspace)
        )
        status = await asyncio.to_thread(self._workspaces.status, workspace)
        metadata_paths = candidate.metadata.get("changed_paths")
        changed_files = (
            {str(path) for path in metadata_paths if isinstance(path, str)}
            if isinstance(metadata_paths, list)
            else set()
        )
        changed_files.update(self._status_path(item) for item in status)
        changed_files.discard("")
        snapshot = {
            "active": True,
            "candidate_id": candidate.id,
            "candidate": self._source_candidate_data(candidate),
            "dirty": bool(status),
            "changed_files": sorted(changed_files)[:1_000],
            "working_tree_status": [item[:1_000] for item in status[:1_000]],
            "conflict_paths": list(conflict_paths),
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        }
        if include_diff:
            bounded_diff, diff_truncated = self._bounded_text(diff)
            snapshot["diff"] = bounded_diff
            snapshot["diff_truncated"] = diff_truncated
        return snapshot

    async def _source_conflict_paths(
        self,
        candidate: Candidate,
        workspace: CandidateWorkspace,
    ) -> tuple[str, ...]:
        lineage = self._lineage
        if lineage is None:
            return ()
        self._candidate_upstream_lineage(candidate)
        try:
            paths = await asyncio.to_thread(lineage.conflicted_paths, workspace)
        except GitLineageError as exc:
            raise EvolutionSupervisorError("source merge conflict state is unavailable") from exc
        return tuple(path[:1_000] for path in paths[:1_000])

    def _source_diff(self, workspace: CandidateWorkspace) -> str:
        return self._workspaces.full_diff(workspace)

    def _require_source_diff_binding(
        self,
        workspace: CandidateWorkspace,
        *,
        expected_diff_sha256: str,
    ) -> None:
        actual = hashlib.sha256(self._source_diff(workspace).encode("utf-8")).hexdigest()
        if actual != expected_diff_sha256:
            raise EvolutionSupervisorError("source changed before approval")

    async def _source_lineage(self) -> dict[str, Any]:
        current = await self._archive.get_current_release()
        target = await self._archive.get_rollback_target()
        values: dict[str, Any] = {
            "current_release_id": current.id if current is not None else None,
            "rollback_target_release_id": target.id if target is not None else None,
            "active_commit": current.source_commit if current is not None else None,
            "active_source_commit": current.source_commit if current is not None else None,
            "instance_commit": None,
            "upstream_commit": None,
            "accepted_upstream_commit": None,
            "merge_base_commit": None,
            "upstream_pending": False,
        }
        if self._lineage is None:
            return values
        snapshot = await self._lineage_snapshot()
        values.update(
            {
                "instance_commit": snapshot.instance_commit,
                "upstream_commit": snapshot.upstream_commit,
                "accepted_upstream_commit": snapshot.accepted_upstream_commit,
                "merge_base_commit": snapshot.merge_base_commit,
                "upstream_pending": (
                    snapshot.upstream_commit != snapshot.accepted_upstream_commit
                ),
            }
        )
        return values

    @staticmethod
    def _source_candidate_data(candidate: Candidate) -> dict[str, Any]:
        report = candidate.evaluation_report
        raw_paths = candidate.metadata.get("changed_paths")
        return {
            "id": candidate.id,
            "status": candidate.status.value,
            "revision": candidate.revision,
            "base_commit": candidate.base_commit,
            "source_commit": candidate.source_commit,
            "artifact_digest": candidate.artifact_digest,
            "changed_paths": (
                [str(path) for path in raw_paths if isinstance(path, str)]
                if isinstance(raw_paths, list)
                else []
            ),
            "diff_sha256": str(candidate.metadata.get("diff_sha256") or "") or None,
            "evaluation": report.model_dump(mode="json") if report is not None else None,
            "created_at": candidate.created_at.isoformat(),
            "updated_at": candidate.updated_at.isoformat(),
        }

    @staticmethod
    def _status_path(status: str) -> str:
        value = status[3:] if len(status) > 3 else status
        return value.rsplit(" -> ", maxsplit=1)[-1]

    @staticmethod
    def _bounded_text(value: object, *, limit: int = 50_000) -> tuple[str, bool]:
        text = str(value or "").replace("\x00", "")
        return text[:limit], len(text) > limit

    async def _cleanup_stale_candidate(self, candidate: Candidate) -> None:
        if candidate.worktree_path:
            workspace = CandidateWorkspace(
                candidate_id=candidate.id,
                path=Path(candidate.worktree_path),
                base_commit=candidate.base_commit,
            )
            await self._fail_candidate(candidate.id, workspace, code="process_restarted")
            return
        await self._archive.transition_status(
            candidate.id,
            expected_status=CandidateStatus.BUILDING,
            new_status=CandidateStatus.FAILED,
            expected_revision=candidate.revision,
        )

    async def _required_candidate(self, candidate_id: str) -> Candidate:
        candidate = await self._archive.get_candidate(candidate_id)
        if candidate is None:
            raise EvolutionSupervisorError("candidate was not found")
        return candidate

    @staticmethod
    def _require_revision(candidate: Candidate, expected_revision: int | None) -> None:
        if expected_revision is not None and candidate.revision != expected_revision:
            raise EvolutionSupervisorError("candidate revision changed")

    @staticmethod
    def _audit_context(value: Mapping[str, str] | None) -> dict[str, JsonValue]:
        if value is None:
            return {}
        limits = {
            "tenant_id": 200,
            "actor_id": 200,
            "thread_id": 8_192,
            "channel": 64,
            "run_kind": 64,
            "correlation_id": 8_192,
            "origin": 4_000,
            "authority": 100,
            "reason": 4_000,
        }
        return {
            key: cleaned[: limits[key]]
            for key in limits
            if (cleaned := str(value.get(key, "") or "").strip())
        }

    async def _accepted_upstream_for_release(self, release: Release) -> str | None:
        recorded = str(release.metadata.get("accepted_upstream_commit") or "").strip()
        if recorded and _COMMIT_RE.fullmatch(recorded) is None:
            raise EvolutionSupervisorError("release accepted upstream metadata is invalid")
        lineage = self._lineage
        if lineage is None:
            return recorded or None
        try:
            upstream = await asyncio.to_thread(lineage.resolve_ref, lineage.upstream_ref)
            accepted = recorded or await asyncio.to_thread(
                lineage.merge_base,
                release.source_commit,
                upstream,
            )
            in_release, in_upstream = await asyncio.gather(
                asyncio.to_thread(lineage.is_ancestor, accepted, release.source_commit),
                asyncio.to_thread(lineage.is_ancestor, accepted, upstream),
            )
        except GitLineageError as exc:
            raise EvolutionSupervisorError("release source lineage is invalid") from exc
        if not in_release or not in_upstream:
            raise EvolutionSupervisorError("release accepted upstream is not in its lineage")
        return accepted

    async def _release_with_accepted_upstream(self, release: Release) -> Release:
        accepted = await self._accepted_upstream_for_release(release)
        if accepted is None or release.metadata.get("accepted_upstream_commit") == accepted:
            return release
        return release.model_copy(
            update={
                "metadata": {
                    **release.metadata,
                    "accepted_upstream_commit": accepted,
                }
            }
        )

    async def _optional_lineage_ref(self, ref: str) -> str | None:
        lineage = self._lineage
        if lineage is None:
            return None
        try:
            return await asyncio.to_thread(lineage.resolve_ref, ref)
        except GitLineageError:
            return None

    async def _reconcile_lineage_projection(self, current: Release | None) -> None:
        lineage = self._lineage
        if lineage is None:
            return
        instance, accepted_ref = await asyncio.gather(
            self._optional_lineage_ref(lineage.instance_ref),
            self._optional_lineage_ref(lineage.accepted_upstream_ref),
        )
        if current is None:
            if instance is not None or accepted_ref is not None:
                raise EvolutionSupervisorError("source lineage exists without an archived release")
            return
        target_accepted = await self._accepted_upstream_for_release(current)
        assert target_accepted is not None
        if instance is None and accepted_ref is None:
            try:
                await asyncio.to_thread(
                    lineage.initialize,
                    current.source_commit,
                    target_accepted,
                )
            except GitLineageError as exc:
                raise EvolutionSupervisorError("source lineage initialization failed") from exc
            return
        if instance is None or accepted_ref is None:
            raise EvolutionSupervisorError("source lineage refs are incomplete")
        if instance == current.source_commit and accepted_ref == target_accepted:
            return
        previous = (
            await self._archive.get_release(current.previous_release_id)
            if current.previous_release_id is not None
            else None
        )
        if previous is None:
            raise EvolutionSupervisorError("source lineage diverged from the archive")
        previous_accepted = await self._accepted_upstream_for_release(previous)
        assert previous_accepted is not None
        if instance != previous.source_commit or accepted_ref != previous_accepted:
            raise EvolutionSupervisorError("source lineage diverged from the archive")
        try:
            await asyncio.to_thread(
                lineage.project,
                current.source_commit,
                target_accepted,
                expected_instance_commit=previous.source_commit,
                expected_accepted_upstream_commit=previous_accepted,
            )
        except GitLineageError as exc:
            raise EvolutionSupervisorError("source lineage projection repair failed") from exc

    async def _project_lineage_release(self, release: Release) -> None:
        lineage = self._lineage
        if lineage is None:
            return
        accepted = await self._accepted_upstream_for_release(release)
        assert accepted is not None
        snapshot = await self._lineage_snapshot()
        if (
            snapshot.instance_commit == release.source_commit
            and snapshot.accepted_upstream_commit == accepted
        ):
            return
        previous = (
            await self._archive.get_release(release.previous_release_id)
            if release.previous_release_id is not None
            else None
        )
        if previous is None:
            raise EvolutionSupervisorError("release predecessor lineage is unavailable")
        previous_accepted = await self._accepted_upstream_for_release(previous)
        assert previous_accepted is not None
        if (
            snapshot.instance_commit != previous.source_commit
            or snapshot.accepted_upstream_commit != previous_accepted
        ):
            raise EvolutionSupervisorError("source lineage diverged before projection")
        try:
            await asyncio.to_thread(
                lineage.project,
                release.source_commit,
                accepted,
                expected_instance_commit=previous.source_commit,
                expected_accepted_upstream_commit=previous_accepted,
            )
        except GitLineageError as exc:
            raise EvolutionSupervisorError("active source lineage projection failed") from exc

    async def _project_release(self, release: Release) -> None:
        """Project the archive-authoritative release for inspection and recovery."""

        projected = await self._release_with_accepted_upstream(release)
        await self._project_lineage_release(projected)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                await self._release_pointer.activate(projected)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    await asyncio.sleep(0.1 * (attempt + 1))
        if last_error is not None:
            raise EvolutionSupervisorError(
                "active release pointer projection failed"
            ) from last_error

    @contextmanager
    def _hide_git_metadata(self, workspace: CandidateWorkspace):  # type: ignore[no-untyped-def]
        git_path = workspace.path / ".git"
        hidden = workspace.path.parent / f".{workspace.candidate_id}.git-link"
        if git_path.is_symlink() or not git_path.is_file() or hidden.exists():
            raise EvolutionSupervisorError("candidate Git metadata is invalid")
        os.replace(git_path, hidden)
        try:
            yield
        finally:
            if os.path.lexists(git_path):
                if git_path.is_symlink() or git_path.is_file():
                    git_path.unlink()
                elif git_path.is_dir():
                    shutil.rmtree(git_path)
                else:
                    raise EvolutionSupervisorError("candidate created invalid Git metadata")
            os.replace(hidden, git_path)

    @staticmethod
    def _dependency_lock_hash(workspace: Path) -> str | None:
        lockfile = workspace / "uv.lock"
        if not lockfile.is_file() or lockfile.is_symlink():
            return None
        digest = hashlib.sha256()
        with lockfile.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _evaluator_for_lock(
        self,
        lock_hash: str | None,
    ) -> tuple[CandidateEvaluator, ResolvedDependencyBase | None]:
        resolver = self._dependency_resolver
        if resolver is None or lock_hash is None:
            return self._evaluator, None
        resolved = resolver.base_for_lock(lock_hash)
        if resolved is None:
            return self._evaluator, None
        evaluator = self._dependency_evaluators.get(resolved.id)
        if evaluator is None:
            factory = self._dependency_evaluator_factory
            if factory is None:
                raise EvolutionSupervisorError(
                    "resolved dependency evaluation environment is unavailable"
                )
            evaluator = factory(resolved)
            self._dependency_evaluators[resolved.id] = evaluator
        return evaluator, resolved

    @staticmethod
    def _install_resolved_lock(
        workspace: Path,
        resolved: ResolvedDependencyBase,
    ) -> None:
        if (
            len(resolved.id) != 64
            or any(character not in "0123456789abcdef" for character in resolved.id)
            or len(resolved.lock_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in resolved.lock_sha256
            )
            or resolved.root.name != resolved.id
            or resolved.root.is_symlink()
            or not resolved.root.is_dir()
        ):
            raise EvolutionSupervisorError("resolved dependency base identity is invalid")
        source = resolved.lock_path
        try:
            source_metadata = source.lstat()
        except OSError as exc:
            raise EvolutionSupervisorError("resolved dependency lock is unavailable") from exc
        if (
            not stat.S_ISREG(source_metadata.st_mode)
            or source_metadata.st_nlink != 1
            or stat.S_IMODE(source_metadata.st_mode) & 0o222
        ):
            raise EvolutionSupervisorError("resolved dependency lock is unsafe")
        lock_bytes = source.read_bytes()
        if hashlib.sha256(lock_bytes).hexdigest() != resolved.lock_sha256:
            raise EvolutionSupervisorError("resolved dependency lock identity changed")
        target = workspace / "uv.lock"
        if os.path.lexists(target):
            metadata = target.lstat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise EvolutionSupervisorError("candidate dependency lock path is unsafe")
            owner = (metadata.st_uid, metadata.st_gid)
        else:
            metadata = workspace.stat()
            owner = (metadata.st_uid, metadata.st_gid)
        temporary = workspace / f".uv.lock.resolved-{new_short_id('write', suffix_chars=12)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(temporary, flags, 0o600)
            try:
                if os.geteuid() == 0:
                    os.fchown(descriptor, *owner)
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(lock_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
            finally:
                os.close(descriptor)
            os.replace(temporary, target)
            target.chmod(0o600)
            directory = os.open(workspace, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise EvolutionSupervisorError("resolved dependency lock could not be installed") from exc

    @staticmethod
    def _evaluation_input_digest(
        *,
        source_commit: str,
        dependency_lock_hash: str | None,
        evaluator_version: str,
        evaluator_fingerprint: str,
    ) -> str:
        payload = (
            f"{source_commit}:{dependency_lock_hash or 'none'}:"
            f"{evaluator_version}:{evaluator_fingerprint}"
        )
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("evolution supervisor has not been started")

    def _require_source_mutation(self) -> None:
        if not self._source_mutation_enabled:
            raise EvolutionSupervisorError(self._source_mutation_unavailable_reason)


__all__ = [
    "EvolutionEventSink",
    "EvolutionSupervisor",
    "EvolutionSupervisorError",
    "InMemoryEvolutionEventSink",
]
