"""Durable evolution event construction, recovery, and outbox delivery."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Protocol

from pydantic import JsonValue

from opentulpa.evolution.archive import EvolutionArchive, EvolutionArchiveCorruptionError
from opentulpa.evolution.context import EvolutionAuditContext
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvolutionEvent,
    PromotionAttempt,
    PromotionAttemptStatus,
    Release,
)

logger = logging.getLogger(__name__)


class EvolutionEventSink(Protocol):
    async def deliver(self, event: EvolutionEvent) -> None: ...


class InMemoryEvolutionEventSink:
    def __init__(self) -> None:
        self.events: list[EvolutionEvent] = []

    async def deliver(self, event: EvolutionEvent) -> None:
        self.events.append(event)


class EvolutionEventPublisher:
    """Own event payloads and reliable delivery through the archive outbox."""

    def __init__(
        self,
        *,
        archive: EvolutionArchive,
        sink: EvolutionEventSink | None,
    ) -> None:
        self._archive = archive
        self._sink = sink

    async def flush(self, *, limit: int = 100) -> int:
        if self._sink is None:
            return 0
        delivered = 0
        for event in await self._archive.pending_events(limit=limit):
            try:
                await self._sink.deliver(event)
            except Exception:
                logger.exception("evolution event delivery failed: event=%s", event.id)
                await self._archive.mark_event_attempt(event.id, delivered=False)
                continue
            await self._archive.mark_event_attempt(event.id, delivered=True)
            delivered += 1
        return delivered

    async def recover_terminal_events(self) -> None:
        for status in (CandidateStatus.READY, CandidateStatus.FAILED):
            for candidate in await self._archive.list_candidates(status=status, limit=1_000):
                try:
                    await self.publish_candidate(candidate)
                except EvolutionArchiveCorruptionError as exc:
                    if not self._is_recovered_outbox_payload_conflict(exc):
                        raise
                    logger.warning("skipping conflicting recovered candidate event: %s", candidate.id)

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
            try:
                await self.publish_promotion(attempt, active_release=active_release)
            except EvolutionArchiveCorruptionError as exc:
                if not self._is_recovered_outbox_payload_conflict(exc):
                    raise
                logger.warning("skipping conflicting recovered promotion event: %s", attempt.id)

    async def publish_candidate(self, candidate: Candidate) -> None:
        if candidate.status not in {CandidateStatus.READY, CandidateStatus.FAILED}:
            return
        report = candidate.evaluation_report
        completion_id = (
            report.id if report is not None else f"{candidate.status.value}:{candidate.revision}"
        )
        await self._enqueue(
            EvolutionEvent(
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
        )

    async def publish_build(
        self,
        *,
        event_key: str,
        event_type: str,
        candidate_id: str,
        origin: Mapping[str, JsonValue],
        payload: dict[str, JsonValue],
    ) -> None:
        await self._enqueue(
            EvolutionEvent(
                event_key=event_key,
                event_type=event_type,
                candidate_id=candidate_id,
                origin=dict(origin),
                payload=payload,
            )
        )

    async def publish_promotion(
        self,
        attempt: PromotionAttempt,
        *,
        active_release: Release | None = None,
    ) -> None:
        failed = attempt.status is PromotionAttemptStatus.FAILED
        rollback = bool(attempt.release.metadata.get("rollback_target"))
        await self._enqueue(
            EvolutionEvent(
                event_key=f"promotion:{attempt.id}:terminal",
                event_type=(
                    "rollback.failed"
                    if failed and rollback
                    else "promotion.failed"
                    if failed
                    else "build.rolled_back"
                    if rollback
                    else "build.active"
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
                    **(
                        {"failure_message": attempt.failure_message}
                        if attempt.failure_message
                        else {}
                    ),
                },
            )
        )

    async def _enqueue(self, event: EvolutionEvent) -> None:
        await self._archive.enqueue_event(event)
        await self.flush()

    @staticmethod
    def _is_recovered_outbox_payload_conflict(exc: EvolutionArchiveCorruptionError) -> bool:
        return "evolution event key is bound to another payload" in str(exc)

    @staticmethod
    def _candidate_origin(candidate: Candidate) -> dict[str, JsonValue]:
        value = candidate.metadata.get("requested_by")
        return EvolutionAuditContext.from_mapping(
            value if isinstance(value, Mapping) else None
        ).as_metadata()


__all__ = [
    "EvolutionEventPublisher",
    "EvolutionEventSink",
    "InMemoryEvolutionEventSink",
]
