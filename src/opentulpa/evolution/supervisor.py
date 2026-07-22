"""Trusted source release evaluation, activation, rollback, and sharing."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
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
from opentulpa.evolution.evaluator import CandidateEvaluator, EvaluationCommandResult
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
    OciReleaseArtifact,
    ReleaseBuilder,
    ReleaseBuildError,
    ReleaseBuildRequest,
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
        promotion_retry_interval_seconds: float = 1.0,
        release_builder: ReleaseBuilder | None = None,
        release_activator: ReleaseActivator | None = None,
        event_sink: EvolutionEventSink | None = None,
    ) -> None:
        if not evaluator_version.strip():
            raise ValueError("evaluator_version is required")
        if not source_ref.strip():
            raise ValueError("source_ref is required")
        if not upstream_repository.strip():
            raise ValueError("upstream_repository is required")
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
        self._promotion_retry_interval_seconds = promotion_retry_interval_seconds
        self._activation_lock = asyncio.Lock()
        self._release_builder = release_builder
        self._release_activator = release_activator
        self._event_sink = event_sink
        self._source_locks: dict[str, asyncio.Lock] = {}
        self._promotion_wake = asyncio.Event()
        self._promotion_dispatcher: asyncio.Task[None] | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    async def start(self) -> None:
        if self._started:
            return
        await self._archive.start()
        current = await self._archive.get_current_release()
        pointer = await self._release_pointer.current()
        if current is None:
            if pointer is not None:
                await self._release_pointer.clear()
        elif pointer != current:
            await self._release_pointer.activate(current)
        stale = await self._archive.list_candidates(status=CandidateStatus.BUILDING, limit=1_000)
        for candidate in stale:
            if self._is_source_session(candidate):
                await self._reconcile_source_session(candidate)
            else:
                await self._cleanup_stale_candidate(candidate)
        await self._ensure_terminal_candidate_events()
        await self.flush_events()
        self._started = True
        await self._resume_pending_source_releases()
        if self._release_activator is not None:
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
        session_key, audit = self._source_context(audit_context)
        async with self._source_lock(session_key):
            lineage = await self._source_lineage()
            candidate = await self._find_source_session(
                session_key,
                tenant_id=str(audit["tenant_id"]),
            )
            if candidate is None:
                return {
                    "active": False,
                    "candidate_id": None,
                    "diff_sha256": hashlib.sha256(b"").hexdigest(),
                    **lineage,
                }
            workspace = self._source_workspace(candidate)
            return {
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
        """Run one command in the isolated, writable source candidate."""

        self._require_started()
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
            backend = backend_factory(workspace.path)
            with self._hide_git_metadata(workspace):
                response = await backend.aexecute(safe_command, timeout=timeout)
            current = await self._required_candidate(candidate.id)
            snapshot = await self._source_snapshot(current, workspace)
            output, output_truncated = self._bounded_text(response.output)
            return {
                **snapshot,
                "exit_code": int(response.exit_code),
                "output": output,
                "output_truncated": bool(response.truncated or output_truncated),
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
            operation_digest = hashlib.sha256(
                f"{tenant_id}\x00{safe_key}".encode()
            ).hexdigest()
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
        async with self._activation_lock:
            if attempt_id is not None:
                existing = await self._archive.get_promotion_attempt(attempt_id)
                if existing is not None:
                    if (
                        existing.candidate_id != candidate_id
                        or (release_id is not None and existing.release.id != release_id)
                    ):
                        raise EvolutionSupervisorError(
                            "source release promotion binding changed"
                        )
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
            if expected_previous_source_commit is not None and (
                current is None
                or current.source_commit != expected_previous_source_commit
            ):
                raise EvolutionSupervisorError(
                    "source session is based on an inactive release"
                )
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
                    **(
                        {"requested_by": audit}
                        if audit
                        else {}
                    ),
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
            if (
                expected_current_release_id is not None
                and current.id != expected_current_release_id
            ):
                raise EvolutionSupervisorError("rollback source changed before approval")
            if (
                expected_target_release_id is not None
                and target.id != expected_target_release_id
            ):
                raise EvolutionSupervisorError("rollback target changed before approval")
            if self._release_activator is None:
                raise EvolutionSupervisorError("release activation is unavailable")
            target_candidate = await self._required_candidate(target.candidate_id)
            audit = self._audit_context(audit_context)
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
                        {
                            "source_rollback_idempotency_digest": idempotency_digest,
                            "source_rollback_tenant_id": expected_tenant_id,
                        }
                        if idempotency_digest is not None and expected_tenant_id is not None
                        else {}
                    ),
                    **(
                        {"requested_by": audit}
                        if audit
                        else {}
                    ),
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

        source_release = rollback_target or attempt.release
        try:
            result = await activator.activate(
                self._bootstrap_release(source_release),
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
        completed_attempt = await self._archive.transition_promotion_attempt(
            attempt.id,
            expected_status=PromotionAttemptStatus.ACTIVATING,
            new_status=PromotionAttemptStatus.ACTIVE,
            expected_revision=attempt.revision,
            bootstrap_activation_id=result.activation_id,
        )
        await self._publish_promotion_event(completed_attempt, active_release=recorded)
        await self._project_release(recorded)
        return recorded

    async def _resume_promotion_attempt(self, attempt: PromotionAttempt) -> None:
        current = await self._archive.get_current_release()
        if current is not None and current.id == attempt.release.id:
            if attempt.status is PromotionAttemptStatus.ACTIVATING:
                attempt = await self._archive.transition_promotion_attempt(
                    attempt.id,
                    expected_status=PromotionAttemptStatus.ACTIVATING,
                    new_status=PromotionAttemptStatus.ACTIVE,
                    expected_revision=attempt.revision,
                    bootstrap_activation_id=attempt.id,
                )
            await self._publish_promotion_event(attempt, active_release=current)
            await self._project_release(current)
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
            logger.warning("promotion attempt did not resume: attempt=%s", attempt.id)

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
        failed = await self._archive.transition_promotion_attempt(
            current_attempt.id,
            expected_status=current_attempt.status,
            new_status=PromotionAttemptStatus.FAILED,
            expected_revision=current_attempt.revision,
            bootstrap_activation_id=current_attempt.bootstrap_activation_id or current_attempt.id,
            failure_code=safe_code,
            failure_message=safe_message,
        )
        candidate = await self._archive.get_candidate(failed.candidate_id)
        if candidate is None:
            return
        await self._archive.update_candidate(
            candidate.model_copy(
                update={
                    "metadata": {
                        **candidate.metadata,
                        "last_activation_failure": {
                            "attempt_id": failed.id,
                            "code": safe_code,
                            "message": safe_message,
                        },
                    }
                }
            ),
            expected_revision=candidate.revision,
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
            return
        rollback_of = str(attempt.release.metadata.get("rollback_of") or "")
        if current is None or current.id != rollback_of:
            raise EvolutionSupervisorError("rollback predecessor changed")
        if (
            str(attempt.release.metadata.get("rollback_target") or "")
            != rollback_target.id
            or rollback_target.candidate_id != attempt.candidate_id
            or rollback_target.source_commit != attempt.release.source_commit
            or rollback_target.artifact_digest != attempt.release.artifact_digest
            or candidate.status
            not in {CandidateStatus.PROMOTED, CandidateStatus.ROLLED_BACK}
        ):
            raise EvolutionSupervisorError("rollback target changed")

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
            raise EvolutionSupervisorError("candidate has no verified OCI release manifest")
        return {
            "artifact_kind": "oci_image",
            "manifest_digest": manifest_digest,
            "release_entrypoint": [str(item) for item in raw_entrypoint],
            "base_commit": candidate.base_commit,
            "changed_paths": [str(item) for item in raw_changed_paths],
            "diff_sha256": diff_sha256,
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

    @staticmethod
    def _release_metadata_from_release(release: Release) -> dict[str, JsonValue]:
        return {
            key: release.metadata[key]
            for key in (
                "artifact_kind",
                "manifest_digest",
                "release_entrypoint",
                "image_reference",
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
                key: value
                for key, value in release.metadata.items()
                if key != "requested_by"
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
    ) -> tuple[OciReleaseArtifact | None, EvaluationCommandResult]:
        started = asyncio.get_running_loop().time()
        builder = self._release_builder
        if builder is None:
            return None, EvaluationCommandResult(
                name="oci.release",
                stage="build",
                passed=False,
                exit_code=1,
                duration_seconds=0,
                output="Trusted OCI release builder is unavailable.",
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
                    evaluator_fingerprint=self._evaluator.fingerprint,
                )
            )
        except ReleaseBuildError as exc:
            message = str(exc or "Candidate OCI image build failed.")[:4_000]
            return None, EvaluationCommandResult(
                name="oci.release",
                stage="build",
                passed=False,
                exit_code=1,
                duration_seconds=asyncio.get_running_loop().time() - started,
                output=message,
            )
        except Exception:
            logger.exception("trusted candidate OCI build failed: candidate=%s", candidate_id)
            return None, EvaluationCommandResult(
                name="oci.release",
                stage="build",
                passed=False,
                exit_code=1,
                duration_seconds=asyncio.get_running_loop().time() - started,
                output="Candidate OCI image build failed.",
            )
        return artifact, EvaluationCommandResult(
            name="oci.release",
            stage="build",
            passed=True,
            exit_code=0,
            duration_seconds=asyncio.get_running_loop().time() - started,
            output="Verified immutable OCI image and release manifest.",
        )

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
            (current_release.id if current_release is not None else None)
            != operation.base_release_id
            or (
                current_release is not None
                and current_release.source_commit != candidate.base_commit
            )
        ):
            raise EvolutionSupervisorError("source session is based on an inactive release")

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
        current, lock_hash = await self._bind_source_commit(
            candidate=candidate,
            workspace=workspace,
            commit=commit,
        )
        prior_report = current.evaluation_report
        if (
            prior_report is not None
            and prior_report.source_commit == commit.source_commit
            and prior_report.artifact_digest == current.artifact_digest
            and prior_report.evaluator_fingerprint == self._evaluator.fingerprint
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

        command_results = await self._evaluator.evaluate(workspace.path)
        await self._assert_source_unchanged(workspace, commit.source_commit, "evaluation")
        artifact: OciReleaseArtifact | None = None
        if all(result.passed for result in command_results):
            artifact, build_result = await self._build_release_artifact(
                candidate_id=current.id,
                workspace=workspace.path,
                base_commit=commit.base_commit,
                source_commit=commit.source_commit,
                dependency_lock_hash=lock_hash,
            )
            command_results = (*command_results, build_result)
        await self._assert_source_unchanged(workspace, commit.source_commit, "release build")
        evaluation_input_digest = str(current.metadata["evaluation_input_digest"])
        artifact_digest = (
            artifact.artifact_digest if artifact is not None else evaluation_input_digest
        )
        metadata = dict(current.metadata)
        if artifact is not None:
            metadata.update(
                {
                    "artifact_kind": "oci_image",
                    "manifest_digest": artifact.manifest_digest,
                    "image_reference": artifact.image_reference,
                    "release_entrypoint": list(artifact.entrypoint),
                }
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
            evaluator_fingerprint=self._evaluator.fingerprint,
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
    ) -> tuple[Candidate, str | None]:
        lock_hash = self._dependency_lock_hash(workspace.path)
        evaluation_input_digest = self._evaluation_input_digest(
            source_commit=commit.source_commit,
            dependency_lock_hash=lock_hash,
            evaluator_version=self._evaluator_version,
            evaluator_fingerprint=self._evaluator.fingerprint,
        )
        current = await self._required_candidate(candidate.id)
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
        if source_changed:
            for key in (
                "artifact_kind",
                "image_reference",
                "manifest_digest",
                "release_entrypoint",
            ):
                metadata.pop(key, None)
        metadata.update(
            {
                "changed_paths": list(changed_paths),
                "diff_sha256": commit.diff_sha256,
                "evaluation_input_digest": evaluation_input_digest,
                "promotion_eligible": bool(
                    commit.promotion_eligible
                    and current.metadata.get("promotion_eligible", True)
                ),
            }
        )
        if (
            current.source_commit == commit.source_commit
            and current.dependency_lock_hash == lock_hash
            and current.evaluator_fingerprint == self._evaluator.fingerprint
            and current.metadata == metadata
        ):
            return current, lock_hash
        current = await self._archive.update_candidate(
            current.model_copy(
                update={
                    "source_commit": commit.source_commit,
                    "dependency_lock_hash": lock_hash,
                    "artifact_digest": None if source_changed else current.artifact_digest,
                    "evaluator_fingerprint": self._evaluator.fingerprint,
                    "metadata": metadata,
                }
            ),
            expected_revision=current.revision,
        )
        return current, lock_hash

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
            current_release = await self._archive.get_current_release()
            current_release_id = current_release.id if current_release is not None else None
            if current_release_id != operation.base_release_id:
                raise EvolutionSupervisorError(
                    "source session is based on an inactive release"
                )
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
                logger.exception(
                    "source release recovery remains pending: operation=%s",
                    operation.id,
                )

    async def _require_current_source_base(self, candidate: Candidate) -> None:
        current = await self._archive.get_current_release()
        if current is not None and current.source_commit != candidate.base_commit:
            raise EvolutionSupervisorError("source session is based on an inactive release")

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
            head = await asyncio.to_thread(self._workspaces.head, workspace)
            expected_head = candidate.source_commit or candidate.base_commit
            if head != expected_head:
                operation = await self._archive.get_pending_source_release_operation(
                    candidate.id
                )
                if operation is not None:
                    commit = await asyncio.to_thread(
                        self._workspaces.recover_commit,
                        workspace,
                    )
                    if commit.diff_sha256 != operation.expected_diff_sha256:
                        raise EvolutionSupervisorError(
                            "recovered source commit does not match its approval"
                        )
                    await self._bind_source_commit(
                        candidate=candidate,
                        workspace=workspace,
                        commit=commit,
                    )
                else:
                    raise EvolutionSupervisorError(
                        "source session commit changed unexpectedly"
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
                or (
                    isinstance(requested_by, dict)
                    and requested_by.get("tenant_id") == tenant_id
                )
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
        base_ref = current_release.source_commit if current_release is not None else self._source_ref
        candidate_id = new_short_id("candidate", suffix_chars=12)
        workspace = await asyncio.to_thread(
            self._workspaces.create,
            candidate_id=candidate_id,
            base_ref=base_ref,
        )
        candidate = Candidate(
            id=candidate_id,
            base_commit=workspace.base_commit,
            requested_improvement="Interactive OpenTulpa source session",
            worktree_path=str(workspace.path),
            metadata={
                "source_session": True,
                "source_session_key": session_key,
                "source_tenant_id": tenant_id,
                "requested_by": dict(audit),
            },
        )
        try:
            await self._archive.create_candidate(candidate)
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
        status = await asyncio.to_thread(self._workspaces.status, workspace)
        if status:
            return await asyncio.to_thread(
                self._workspaces.commit,
                workspace,
                message=message,
            )
        head = await asyncio.to_thread(self._workspaces.head, workspace)
        if head == candidate.base_commit:
            raise EvolutionSupervisorError("source session has no changes to release")
        if candidate.source_commit is None:
            return await asyncio.to_thread(self._workspaces.recover_commit, workspace)
        if candidate.source_commit != head:
            raise EvolutionSupervisorError("source session commit changed unexpectedly")
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
    ) -> dict[str, Any]:
        diff = await asyncio.to_thread(self._source_diff, workspace)
        status = await asyncio.to_thread(self._workspaces.status, workspace)
        bounded_diff, diff_truncated = self._bounded_text(diff)
        metadata_paths = candidate.metadata.get("changed_paths")
        changed_files = (
            {str(path) for path in metadata_paths if isinstance(path, str)}
            if isinstance(metadata_paths, list)
            else set()
        )
        changed_files.update(self._status_path(item) for item in status)
        changed_files.discard("")
        return {
            "active": True,
            "candidate_id": candidate.id,
            "candidate": self._source_candidate_data(candidate),
            "dirty": bool(status),
            "changed_files": sorted(changed_files)[:1_000],
            "working_tree_status": [item[:1_000] for item in status[:1_000]],
            "diff": bounded_diff,
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            "diff_truncated": diff_truncated,
        }

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

    async def _source_lineage(self) -> dict[str, str | None]:
        current = await self._archive.get_current_release()
        target = await self._archive.get_rollback_target()
        return {
            "current_release_id": current.id if current is not None else None,
            "rollback_target_release_id": target.id if target is not None else None,
        }

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

    async def _project_release(self, release: Release) -> None:
        """Project the already-active release for local inspection and recovery."""

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                await self._release_pointer.activate(release)
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


__all__ = [
    "EvolutionEventSink",
    "EvolutionSupervisor",
    "EvolutionSupervisorError",
    "InMemoryEvolutionEventSink",
]
