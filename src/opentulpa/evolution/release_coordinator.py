"""Durable release activation saga above the immutable bootstrap boundary."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping

from pydantic import JsonValue

from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.evolution.activation import (
    ReleaseActivationStatus,
    ReleaseActivator,
)
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.models import (
    PromotionAttempt,
    PromotionAttemptStatus,
    Release,
)

logger = logging.getLogger(__name__)

AttemptValidator = Callable[[PromotionAttempt, Release | None], Awaitable[None]]
BootstrapReleaseFactory = Callable[[Release], ReleaseRecord]
OriginFactory = Callable[[Mapping[str, JsonValue]], ReleaseOrigin | None]
ReleaseProjection = Callable[[Release], Awaitable[None]]
SwitchingPublisher = Callable[[PromotionAttempt, bool], Awaitable[None]]
TerminalPublisher = Callable[[PromotionAttempt, Release | None], Awaitable[None]]


class ReleaseCoordinationError(RuntimeError):
    """A release saga could not reach or reconcile a terminal state."""


class ReleaseCoordinator:
    """Own promotion-attempt transitions and crash-safe release projection."""

    def __init__(
        self,
        *,
        archive: EvolutionArchive,
        activator: ReleaseActivator | None,
        validate_attempt: AttemptValidator,
        bootstrap_release: BootstrapReleaseFactory,
        release_origin: OriginFactory,
        project_release: ReleaseProjection,
        publish_switching: SwitchingPublisher,
        publish_terminal: TerminalPublisher,
    ) -> None:
        self._archive = archive
        self._activator = activator
        self._validate_attempt = validate_attempt
        self._bootstrap_release = bootstrap_release
        self._release_origin = release_origin
        self._project_release = project_release
        self._publish_switching = publish_switching
        self._publish_terminal = publish_terminal

    async def resume(self, attempt: PromotionAttempt) -> None:
        candidate = await self._archive.get_candidate(attempt.candidate_id)
        if candidate is None:
            await self.fail(
                attempt,
                code="candidate_missing",
                message="The release candidate is no longer available.",
            )
            return
        recorded_failure = candidate.metadata.get("last_activation_failure")
        if isinstance(recorded_failure, dict) and str(
            recorded_failure.get("attempt_id") or ""
        ) == attempt.id:
            await self.fail(
                attempt,
                code=str(recorded_failure.get("code") or "activation_failed"),
                message=str(recorded_failure.get("message") or "Release activation failed."),
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
            await self._publish_terminal(attempt, current)
            return
        rollback_target_id = str(attempt.release.metadata.get("rollback_target") or "")
        rollback_target = (
            await self._archive.get_release(rollback_target_id) if rollback_target_id else None
        )
        if rollback_target_id and rollback_target is None:
            await self.fail(
                attempt,
                code="rollback_target_missing",
                message="The rollback target is no longer available.",
            )
            return
        try:
            await self._activate(attempt, rollback_target=rollback_target)
        except ReleaseCoordinationError:
            logger.warning(
                "promotion attempt did not resume: attempt=%s",
                attempt.id,
                exc_info=True,
            )

    async def fail(
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
        candidate = await self._archive.get_candidate(current_attempt.candidate_id)
        if candidate is None:
            raise ReleaseCoordinationError("promotion candidate disappeared during failure commit")
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
        await self._publish_terminal(failed, None)

    async def _activate(
        self,
        attempt: PromotionAttempt,
        *,
        rollback_target: Release | None,
    ) -> Release:
        activator = self._activator
        if activator is None:
            await self.fail(
                attempt,
                code="activation_unavailable",
                message="The immutable release bootstrap is unavailable.",
            )
            raise ReleaseCoordinationError("release activation is unavailable")
        try:
            await self._validate_attempt(attempt, rollback_target)
        except ReleaseCoordinationError:
            await self.fail(
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
            raise ReleaseCoordinationError("promotion attempt is not resumable")

        await self._publish_switching(attempt, rollback_target is not None)
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
            await self.fail(attempt, code=code, message=message)
            raise ReleaseCoordinationError(message) from exc
        if result.status is not ReleaseActivationStatus.ACTIVE:
            code = result.failure_code or "activation_failed"
            message = result.failure_message or "Release activation did not become active."
            await self.fail(attempt, code=code, message=message)
            raise ReleaseCoordinationError(message)

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
                    raise ReleaseCoordinationError("rollback attempt lost its predecessor")
                current = await self._archive.get_current_release()
                if current is None or current.id != rollback_of:
                    raise ReleaseCoordinationError("rollback predecessor changed")
                current_candidate = await self._archive.get_candidate(current.candidate_id)
                if current_candidate is None:
                    raise ReleaseCoordinationError("rollback predecessor candidate is unavailable")
                _, recorded = await self._archive.activate_rollback(
                    rollback_target.id,
                    activation=active,
                    expected_current_release_id=rollback_of,
                    expected_current_candidate_revision=current_candidate.revision,
                )
        except Exception as exc:
            logger.exception(
                "bootstrap activated release but archive commit is pending: attempt=%s",
                attempt.id,
            )
            raise ReleaseCoordinationError(
                "release activated; archive reconciliation is pending"
            ) from exc
        try:
            await self._project_release(recorded)
        except Exception as exc:
            logger.exception(
                "archive activated release but projection is pending: attempt=%s",
                attempt.id,
            )
            raise ReleaseCoordinationError(
                "release activated; projection reconciliation is pending"
            ) from exc
        completed_attempt = await self._archive.transition_promotion_attempt(
            attempt.id,
            expected_status=PromotionAttemptStatus.ACTIVATING,
            new_status=PromotionAttemptStatus.ACTIVE,
            expected_revision=attempt.revision,
            bootstrap_activation_id=result.activation_id,
        )
        await self._publish_terminal(completed_attempt, recorded)
        return recorded

    @staticmethod
    def _activation_error(exc: Exception) -> tuple[str, str]:
        code = str(getattr(exc, "code", "") or "activation_failed")[:100]
        message = str(
            getattr(exc, "public_message", "") or "Release activation failed."
        )[:2_000]
        return code, message


__all__ = ["ReleaseCoordinationError", "ReleaseCoordinator"]
