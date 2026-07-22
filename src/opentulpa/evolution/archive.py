"""Async SQLite archive for candidates, evaluations, and release lineage."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel, JsonValue, ValidationError

from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvaluationReport,
    EvolutionEvent,
    PromotionAttempt,
    PromotionAttemptStatus,
    Release,
    SourceReleaseOperation,
    SourceReleaseOperationStatus,
)


class EvolutionArchiveError(RuntimeError):
    """Base error for archive operations."""


class EvolutionArchiveNotStartedError(EvolutionArchiveError):
    """The archive operation requires an open database."""


class EvolutionArchiveCorruptionError(EvolutionArchiveError):
    """A persisted JSON record no longer satisfies its typed contract."""


class CandidateAlreadyExistsError(EvolutionArchiveError):
    """A candidate identifier has already been archived."""


class CandidateNotFoundError(EvolutionArchiveError):
    """The requested candidate does not exist."""


class CandidateConflictError(EvolutionArchiveError):
    """An optimistic candidate revision or status check failed."""


class InvalidCandidateTransitionError(EvolutionArchiveError):
    """The requested candidate lifecycle transition is not allowed."""


class EvaluationAlreadyExistsError(EvolutionArchiveError):
    """An evaluator run with this identifier was already appended."""


class ReleaseConflictError(EvolutionArchiveError):
    """Release history or its current pointer changed unexpectedly."""


class PromotionAttemptConflictError(EvolutionArchiveError):
    """A promotion attempt changed or reached a terminal state."""


class SourceReleaseOperationConflictError(EvolutionArchiveError):
    """A source-release idempotency record conflicts with the request."""


_ALLOWED_TRANSITIONS: dict[CandidateStatus, frozenset[CandidateStatus]] = {
    CandidateStatus.BUILDING: frozenset(
        {
            CandidateStatus.FAILED,
            CandidateStatus.READY,
            CandidateStatus.REJECTED,
        }
    ),
    CandidateStatus.FAILED: frozenset(
        {
            CandidateStatus.BUILDING,
            CandidateStatus.REJECTED,
        }
    ),
    CandidateStatus.READY: frozenset(
        {
            CandidateStatus.BUILDING,
            CandidateStatus.FAILED,
            CandidateStatus.PROMOTED,
            CandidateStatus.REJECTED,
        }
    ),
    CandidateStatus.PROMOTED: frozenset({CandidateStatus.ROLLED_BACK}),
    CandidateStatus.REJECTED: frozenset(),
    CandidateStatus.ROLLED_BACK: frozenset({CandidateStatus.PROMOTED}),
}

_ALLOWED_PROMOTION_TRANSITIONS: dict[
    PromotionAttemptStatus,
    frozenset[PromotionAttemptStatus],
] = {
    PromotionAttemptStatus.QUEUED: frozenset(
        {PromotionAttemptStatus.ACTIVATING, PromotionAttemptStatus.FAILED}
    ),
    PromotionAttemptStatus.ACTIVATING: frozenset(
        {PromotionAttemptStatus.ACTIVE, PromotionAttemptStatus.FAILED}
    ),
    PromotionAttemptStatus.ACTIVE: frozenset(),
    PromotionAttemptStatus.FAILED: frozenset(),
}


class EvolutionArchive:
    """One-process async archive with atomic, optimistic state changes."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        raw_path = str(db_path)
        if raw_path == ":memory:":
            self.db_path: Path | None = None
            self._database = raw_path
        else:
            self.db_path = Path(raw_path).expanduser().resolve()
            self._database = str(self.db_path)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._connection: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Open and initialize the archive; repeated calls are harmless."""

        async with self._lock:
            if self._connection is not None:
                return
            if self.db_path is not None:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = await aiosqlite.connect(self._database)
            connection.row_factory = aiosqlite.Row
            try:
                await connection.execute("PRAGMA foreign_keys=ON")
                await connection.execute("PRAGMA busy_timeout=5000")
                await connection.execute("PRAGMA journal_mode=WAL")
                await connection.execute("PRAGMA synchronous=NORMAL")
                await connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS evolution_candidates (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL CHECK (
                            status IN (
                                'building', 'failed', 'ready', 'promoted',
                                'rejected', 'rolled_back'
                            )
                        ),
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        base_commit TEXT NOT NULL,
                        parent_candidate_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_evolution_candidates_status_updated
                    ON evolution_candidates (status, updated_at DESC, id);

                    CREATE TABLE IF NOT EXISTS evolution_evaluations (
                        candidate_id TEXT NOT NULL,
                        id TEXT NOT NULL,
                        evaluated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        PRIMARY KEY (candidate_id, id),
                        FOREIGN KEY (candidate_id)
                            REFERENCES evolution_candidates(id) ON DELETE RESTRICT
                    );

                    CREATE INDEX IF NOT EXISTS idx_evolution_evaluations_candidate_time
                    ON evolution_evaluations (candidate_id, evaluated_at ASC, id ASC);

                    CREATE TABLE IF NOT EXISTS evolution_releases (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        id TEXT NOT NULL UNIQUE,
                        candidate_id TEXT NOT NULL,
                        previous_release_id TEXT,
                        promoted_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (candidate_id)
                            REFERENCES evolution_candidates(id) ON DELETE RESTRICT
                    );

                    CREATE INDEX IF NOT EXISTS idx_evolution_releases_candidate
                    ON evolution_releases (candidate_id, sequence DESC);

                    CREATE TABLE IF NOT EXISTS evolution_promotion_attempts (
                        id TEXT PRIMARY KEY,
                        candidate_id TEXT NOT NULL,
                        release_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('queued', 'activating', 'active', 'failed')
                        ),
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (candidate_id)
                            REFERENCES evolution_candidates(id) ON DELETE RESTRICT
                    );

                    CREATE INDEX IF NOT EXISTS idx_evolution_promotion_attempts_status
                    ON evolution_promotion_attempts (status, updated_at ASC, id);

                    CREATE TABLE IF NOT EXISTS evolution_source_release_operations (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        candidate_id TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('pending', 'completed')),
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        promotion_attempt_id TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE (tenant_id, idempotency_key),
                        FOREIGN KEY (candidate_id)
                            REFERENCES evolution_candidates(id) ON DELETE RESTRICT
                    );

                    CREATE INDEX IF NOT EXISTS idx_source_release_operations_status
                    ON evolution_source_release_operations (status, created_at ASC, id);

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_source_release_pending_candidate
                    ON evolution_source_release_operations (candidate_id)
                    WHERE status = 'pending';

                    CREATE TABLE IF NOT EXISTS evolution_outbox (
                        id TEXT PRIMARY KEY,
                        event_key TEXT NOT NULL UNIQUE,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('pending', 'delivered')),
                        attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
                        created_at TEXT NOT NULL,
                        delivered_at TEXT,
                        payload_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_evolution_outbox_delivery
                    ON evolution_outbox (status, created_at ASC, id);

                    CREATE TABLE IF NOT EXISTS evolution_state (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        current_release_id TEXT,
                        FOREIGN KEY (current_release_id)
                            REFERENCES evolution_releases(id) ON DELETE RESTRICT
                    );

                    INSERT OR IGNORE INTO evolution_state (singleton, current_release_id)
                    VALUES (1, NULL);
                    """
                )
                await self._migrate_legacy_promotion_approvals(connection)
                await connection.commit()
            except BaseException:
                await connection.close()
                raise
            self._connection = connection

    async def create_promotion_attempt(self, attempt: PromotionAttempt) -> PromotionAttempt:
        if attempt.revision != 1 or attempt.status is not PromotionAttemptStatus.QUEUED:
            raise ValueError("a new promotion attempt must be queued at revision 1")
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                candidate = await self._candidate_in_transaction(connection, attempt.candidate_id)
                if candidate.revision != attempt.candidate_revision:
                    raise CandidateConflictError("candidate changed before promotion was queued")
                cursor = await connection.execute(
                    """
                    SELECT id FROM evolution_promotion_attempts
                    WHERE status IN ('queued', 'activating')
                    LIMIT 1
                    """
                )
                incomplete = await cursor.fetchone()
                await cursor.close()
                if incomplete is not None:
                    raise PromotionAttemptConflictError(
                        "another release activation is already pending"
                    )
                await connection.execute(
                    """
                    INSERT INTO evolution_promotion_attempts (
                        id, candidate_id, release_id, status, revision,
                        created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.id,
                        attempt.candidate_id,
                        attempt.release.id,
                        attempt.status.value,
                        attempt.revision,
                        attempt.created_at.isoformat(),
                        attempt.updated_at.isoformat(),
                        self._serialize(attempt),
                    ),
                )
                await connection.commit()
            except sqlite3.IntegrityError as exc:
                await connection.rollback()
                raise PromotionAttemptConflictError(
                    f"promotion attempt {attempt.id!r} already exists"
                ) from exc
            except BaseException:
                await connection.rollback()
                raise
        return attempt

    async def create_source_release_operation(
        self,
        operation: SourceReleaseOperation,
    ) -> SourceReleaseOperation:
        """Bind an idempotency key to a candidate before any Git side effect."""

        if (
            operation.revision != 1
            or operation.status is not SourceReleaseOperationStatus.PENDING
        ):
            raise ValueError("a new source release operation must be pending at revision 1")
        async with self._lock:
            connection = self._require_connection()
            try:
                await connection.execute(
                    """
                    INSERT INTO evolution_source_release_operations (
                        id, tenant_id, idempotency_key, candidate_id, status,
                        revision, promotion_attempt_id, created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        operation.id,
                        operation.tenant_id,
                        operation.idempotency_key,
                        operation.candidate_id,
                        operation.status.value,
                        operation.revision,
                        operation.promotion_attempt_id,
                        operation.created_at.isoformat(),
                        operation.updated_at.isoformat(),
                        self._serialize(operation),
                    ),
                )
                await connection.commit()
            except sqlite3.IntegrityError as exc:
                await connection.rollback()
                raise SourceReleaseOperationConflictError(
                    "source release operation already exists or the candidate is busy"
                ) from exc
        return operation

    async def get_source_release_operation(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
    ) -> SourceReleaseOperation | None:
        tenant = self._bounded_value(tenant_id, field="tenant_id", max_length=500)
        key = self._bounded_value(
            idempotency_key,
            field="idempotency_key",
            max_length=200,
        )
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_source_release_operations
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant, key),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._source_release_operation_from_row(row) if row is not None else None

    async def get_pending_source_release_operation(
        self,
        candidate_id: str,
    ) -> SourceReleaseOperation | None:
        safe_id = self._identifier(candidate_id, field="candidate_id")
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_source_release_operations
                WHERE candidate_id = ? AND status = 'pending'
                LIMIT 1
                """,
                (safe_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._source_release_operation_from_row(row) if row is not None else None

    async def list_pending_source_release_operations(
        self,
        *,
        limit: int = 100,
    ) -> list[SourceReleaseOperation]:
        safe_limit = max(1, min(int(limit), 1_000))
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_source_release_operations
                WHERE status = 'pending'
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._source_release_operation_from_row(row) for row in rows]

    async def complete_source_release_operation(
        self,
        operation_id: str,
        *,
        expected_revision: int,
        result: dict[str, JsonValue],
        promotion_attempt_id: str | None = None,
    ) -> SourceReleaseOperation:
        safe_id = self._identifier(operation_id, field="operation_id")
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    """
                    SELECT payload_json FROM evolution_source_release_operations
                    WHERE id = ?
                    """,
                    (safe_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise SourceReleaseOperationConflictError(
                        "source release operation was not found"
                    )
                current = self._source_release_operation_from_row(row)
                if current.status is SourceReleaseOperationStatus.COMPLETED:
                    if (
                        current.result == result
                        and current.promotion_attempt_id == promotion_attempt_id
                    ):
                        await connection.commit()
                        return current
                    raise SourceReleaseOperationConflictError(
                        "source release operation already has a different result"
                    )
                if current.revision != int(expected_revision):
                    raise SourceReleaseOperationConflictError(
                        "source release operation changed"
                    )
                updated = SourceReleaseOperation.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "status": SourceReleaseOperationStatus.COMPLETED,
                        "promotion_attempt_id": promotion_attempt_id,
                        "result": result,
                        "revision": current.revision + 1,
                        "updated_at": self._next_timestamp(current.updated_at),
                    }
                )
                changed = await connection.execute(
                    """
                    UPDATE evolution_source_release_operations
                    SET status = ?, revision = ?, promotion_attempt_id = ?,
                        updated_at = ?, payload_json = ?
                    WHERE id = ? AND status = 'pending' AND revision = ?
                    """,
                    (
                        updated.status.value,
                        updated.revision,
                        updated.promotion_attempt_id,
                        updated.updated_at.isoformat(),
                        self._serialize(updated),
                        updated.id,
                        current.revision,
                    ),
                )
                if changed.rowcount != 1:
                    raise SourceReleaseOperationConflictError(
                        "source release operation changed"
                    )
                await connection.commit()
                return updated
            except BaseException:
                await connection.rollback()
                raise

    async def get_promotion_attempt(self, attempt_id: str) -> PromotionAttempt | None:
        safe_id = self._identifier(attempt_id, field="attempt_id")
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                "SELECT payload_json FROM evolution_promotion_attempts WHERE id = ?",
                (safe_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._promotion_attempt_from_row(row) if row is not None else None

    async def list_incomplete_promotion_attempts(
        self, *, limit: int = 100
    ) -> list[PromotionAttempt]:
        safe_limit = max(1, min(int(limit), 1_000))
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_promotion_attempts
                WHERE status IN ('queued', 'activating')
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._promotion_attempt_from_row(row) for row in rows]

    async def transition_promotion_attempt(
        self,
        attempt_id: str,
        *,
        expected_status: PromotionAttemptStatus | str,
        new_status: PromotionAttemptStatus | str,
        expected_revision: int,
        bootstrap_activation_id: str | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> PromotionAttempt:
        safe_id = self._identifier(attempt_id, field="attempt_id")
        expected = PromotionAttemptStatus(expected_status)
        target = PromotionAttemptStatus(new_status)
        if target not in _ALLOWED_PROMOTION_TRANSITIONS[expected]:
            raise PromotionAttemptConflictError(
                f"cannot transition promotion attempt {expected.value} to {target.value}"
            )
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "SELECT payload_json FROM evolution_promotion_attempts WHERE id = ?",
                    (safe_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise PromotionAttemptConflictError("promotion attempt was not found")
                current = self._promotion_attempt_from_row(row)
                if current.status is not expected or current.revision != int(expected_revision):
                    raise PromotionAttemptConflictError("promotion attempt changed")
                updated = PromotionAttempt.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "status": target,
                        "revision": current.revision + 1,
                        "bootstrap_activation_id": (
                            bootstrap_activation_id or current.bootstrap_activation_id
                        ),
                        "failure_code": failure_code,
                        "failure_message": failure_message,
                        "updated_at": self._next_timestamp(current.updated_at),
                    }
                )
                changed = await connection.execute(
                    """
                    UPDATE evolution_promotion_attempts
                    SET status = ?, revision = ?, updated_at = ?, payload_json = ?
                    WHERE id = ? AND status = ? AND revision = ?
                    """,
                    (
                        updated.status.value,
                        updated.revision,
                        updated.updated_at.isoformat(),
                        self._serialize(updated),
                        updated.id,
                        current.status.value,
                        current.revision,
                    ),
                )
                if changed.rowcount != 1:
                    raise PromotionAttemptConflictError("promotion attempt changed")
                await connection.commit()
                return updated
            except BaseException:
                await connection.rollback()
                raise

    async def enqueue_event(self, event: EvolutionEvent) -> EvolutionEvent:
        if event.status != "pending" or event.attempt_count != 0:
            raise ValueError("new evolution events must be pending and unattempted")
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                "SELECT payload_json FROM evolution_outbox WHERE event_key = ?",
                (event.event_key,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is not None:
                existing = self._event_from_row(row)
                if (
                    existing.event_type != event.event_type
                    or existing.candidate_id != event.candidate_id
                    or existing.origin != event.origin
                    or existing.payload != event.payload
                ):
                    raise EvolutionArchiveCorruptionError(
                        "evolution event key is bound to another payload"
                    )
                return existing
            try:
                await connection.execute(
                    """
                    INSERT INTO evolution_outbox (
                        id, event_key, event_type, status, attempt_count,
                        created_at, delivered_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.id,
                        event.event_key,
                        event.event_type,
                        event.status,
                        event.attempt_count,
                        event.created_at.isoformat(),
                        None,
                        self._serialize(event),
                    ),
                )
                await connection.commit()
            except sqlite3.IntegrityError as exc:
                await connection.rollback()
                raise EvolutionArchiveError("evolution event already exists") from exc
        return event

    async def pending_events(self, *, limit: int = 100) -> list[EvolutionEvent]:
        safe_limit = max(1, min(int(limit), 1_000))
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_outbox
                WHERE status = 'pending'
                ORDER BY created_at ASC, id ASC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._event_from_row(row) for row in rows]

    async def mark_event_attempt(self, event_id: str, *, delivered: bool) -> EvolutionEvent:
        safe_id = self._identifier(event_id, field="event_id")
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = await connection.execute(
                    "SELECT payload_json FROM evolution_outbox WHERE id = ?",
                    (safe_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise EvolutionArchiveError("evolution event was not found")
                current = self._event_from_row(row)
                if current.status == "delivered":
                    await connection.commit()
                    return current
                updated = current.model_copy(
                    update={
                        "status": "delivered" if delivered else "pending",
                        "attempt_count": current.attempt_count + 1,
                        "delivered_at": self._next_timestamp(current.created_at)
                        if delivered
                        else None,
                    }
                )
                changed = await connection.execute(
                    """
                    UPDATE evolution_outbox
                    SET status = ?, attempt_count = ?, delivered_at = ?, payload_json = ?
                    WHERE id = ? AND status = 'pending' AND attempt_count = ?
                    """,
                    (
                        updated.status,
                        updated.attempt_count,
                        updated.delivered_at.isoformat() if updated.delivered_at else None,
                        self._serialize(updated),
                        updated.id,
                        current.attempt_count,
                    ),
                )
                if changed.rowcount != 1:
                    raise EvolutionArchiveError("evolution event changed")
                await connection.commit()
                return updated
            except BaseException:
                await connection.rollback()
                raise

    async def shutdown(self) -> None:
        """Commit pending work and close the archive; repeated calls are harmless."""

        async with self._lock:
            connection = self._connection
            if connection is None:
                return
            self._connection = None
            await connection.close()

    async def create_candidate(self, candidate: Candidate) -> Candidate:
        if candidate.revision != 1:
            raise ValueError("a new candidate must start at revision 1")
        payload = self._serialize(candidate)
        async with self._lock:
            connection = self._require_connection()
            try:
                await connection.execute(
                    """
                    INSERT INTO evolution_candidates (
                        id, status, revision, base_commit, parent_candidate_id,
                        created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate.id,
                        candidate.status.value,
                        candidate.revision,
                        candidate.base_commit,
                        candidate.parent_candidate_id,
                        candidate.created_at.isoformat(),
                        candidate.updated_at.isoformat(),
                        payload,
                    ),
                )
                await connection.commit()
            except sqlite3.IntegrityError as exc:
                await connection.rollback()
                raise CandidateAlreadyExistsError(candidate.id) from exc
        return candidate

    async def get_candidate(self, candidate_id: str) -> Candidate | None:
        safe_id = self._identifier(candidate_id, field="candidate_id")
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                "SELECT payload_json FROM evolution_candidates WHERE id = ?",
                (safe_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._candidate_from_row(row) if row is not None else None

    async def list_candidates(
        self,
        *,
        status: CandidateStatus | str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Candidate]:
        safe_limit = max(1, min(int(limit), 1_000))
        safe_offset = max(0, int(offset))
        normalized_status = CandidateStatus(status) if status is not None else None
        async with self._lock:
            connection = self._require_connection()
            if normalized_status is None:
                cursor = await connection.execute(
                    """
                    SELECT payload_json FROM evolution_candidates
                    ORDER BY created_at DESC, id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (safe_limit, safe_offset),
                )
            else:
                cursor = await connection.execute(
                    """
                    SELECT payload_json FROM evolution_candidates
                    WHERE status = ?
                    ORDER BY created_at DESC, id ASC
                    LIMIT ? OFFSET ?
                    """,
                    (normalized_status.value, safe_limit, safe_offset),
                )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._candidate_from_row(row) for row in rows]

    async def update_candidate(
        self,
        candidate: Candidate,
        *,
        expected_revision: int | None = None,
    ) -> Candidate:
        """Replace candidate metadata without bypassing status or evaluation APIs."""

        expected = candidate.revision if expected_revision is None else int(expected_revision)
        if expected < 1 or candidate.revision != expected:
            raise ValueError("candidate revision must equal expected_revision")
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._candidate_in_transaction(connection, candidate.id)
                if current.revision != expected:
                    raise CandidateConflictError(
                        f"candidate revision is {current.revision}, expected {expected}"
                    )
                if candidate.status != current.status:
                    raise ValueError("use transition_status to change candidate status")
                if candidate.evaluation_report != current.evaluation_report:
                    raise ValueError("use append_evaluation to change evaluation evidence")
                if candidate.created_at != current.created_at:
                    raise ValueError("candidate created_at is immutable")
                updated = Candidate.model_validate(
                    {
                        **candidate.model_dump(mode="python"),
                        "revision": expected + 1,
                        "updated_at": self._next_timestamp(current.updated_at),
                    }
                )
                cursor = await connection.execute(
                    """
                    UPDATE evolution_candidates
                    SET status = ?, revision = ?, base_commit = ?,
                        parent_candidate_id = ?, updated_at = ?, payload_json = ?
                    WHERE id = ? AND revision = ? AND status = ?
                    """,
                    (
                        updated.status.value,
                        updated.revision,
                        updated.base_commit,
                        updated.parent_candidate_id,
                        updated.updated_at.isoformat(),
                        self._serialize(updated),
                        updated.id,
                        expected,
                        current.status.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CandidateConflictError("candidate changed during update")
                await connection.commit()
                return updated
            except BaseException:
                await connection.rollback()
                raise

    async def append_evaluation(
        self,
        report: EvaluationReport,
        *,
        expected_revision: int | None = None,
    ) -> Candidate:
        """Append evaluation evidence and expose it as the candidate's latest report."""

        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._candidate_in_transaction(connection, report.candidate_id)
                if expected_revision is not None and current.revision != int(expected_revision):
                    raise CandidateConflictError(
                        f"candidate revision is {current.revision}, expected {expected_revision}"
                    )
                if report.evaluated_at < current.created_at:
                    raise ValueError("evaluation cannot predate its candidate")
                try:
                    await connection.execute(
                        """
                        INSERT INTO evolution_evaluations (
                            candidate_id, id, evaluated_at, payload_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (
                            report.candidate_id,
                            report.id,
                            report.evaluated_at.isoformat(),
                            self._serialize(report),
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise EvaluationAlreadyExistsError(report.id) from exc
                updated = Candidate.model_validate(
                    {
                        **current.model_dump(mode="python"),
                        "evaluation_report": report,
                        "revision": current.revision + 1,
                        "updated_at": self._next_timestamp(current.updated_at),
                    }
                )
                cursor = await connection.execute(
                    """
                    UPDATE evolution_candidates
                    SET revision = ?, updated_at = ?, payload_json = ?
                    WHERE id = ? AND revision = ?
                    """,
                    (
                        updated.revision,
                        updated.updated_at.isoformat(),
                        self._serialize(updated),
                        current.id,
                        current.revision,
                    ),
                )
                if cursor.rowcount != 1:
                    raise CandidateConflictError("candidate changed while appending evaluation")
                await connection.commit()
                return updated
            except BaseException:
                await connection.rollback()
                raise

    async def list_evaluations(self, candidate_id: str) -> list[EvaluationReport]:
        safe_id = self._identifier(candidate_id, field="candidate_id")
        async with self._lock:
            connection = self._require_connection()
            exists = await connection.execute(
                "SELECT 1 FROM evolution_candidates WHERE id = ?",
                (safe_id,),
            )
            exists_row = await exists.fetchone()
            await exists.close()
            if exists_row is None:
                raise CandidateNotFoundError(safe_id)
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_evaluations
                WHERE candidate_id = ?
                ORDER BY evaluated_at ASC, id ASC
                """,
                (safe_id,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._report_from_row(row) for row in rows]

    async def transition_status(
        self,
        candidate_id: str,
        *,
        expected_status: CandidateStatus | str,
        new_status: CandidateStatus | str,
        expected_revision: int | None = None,
    ) -> Candidate:
        """Atomically compare status/revision and advance one lifecycle edge."""

        safe_id = self._identifier(candidate_id, field="candidate_id")
        expected = CandidateStatus(expected_status)
        target = CandidateStatus(new_status)
        if target not in _ALLOWED_TRANSITIONS[expected]:
            raise InvalidCandidateTransitionError(
                f"cannot transition {expected.value} to {target.value}"
            )
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._candidate_in_transaction(connection, safe_id)
                if current.status != expected:
                    raise CandidateConflictError(
                        f"candidate status is {current.status.value}, expected {expected.value}"
                    )
                if expected_revision is not None and current.revision != int(expected_revision):
                    raise CandidateConflictError(
                        f"candidate revision is {current.revision}, expected {expected_revision}"
                    )
                if target in {CandidateStatus.READY, CandidateStatus.PROMOTED}:
                    await self._require_promotion_evidence(connection, current)
                updated = await self._write_status_transition(connection, current, target)
                await connection.commit()
                return updated
            except BaseException:
                await connection.rollback()
                raise

    async def transition_candidate_status(
        self,
        candidate_id: str,
        *,
        expected_status: CandidateStatus | str,
        new_status: CandidateStatus | str,
        expected_revision: int | None = None,
    ) -> Candidate:
        """Explicit alias for callers that prefer the fully qualified operation name."""

        return await self.transition_status(
            candidate_id,
            expected_status=expected_status,
            new_status=new_status,
            expected_revision=expected_revision,
        )

    async def promote_candidate(
        self,
        release: Release,
        *,
        expected_revision: int | None = None,
    ) -> tuple[Candidate, Release]:
        """Atomically promote a ready candidate and activate its release."""

        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                candidate = await self._candidate_in_transaction(connection, release.candidate_id)
                if candidate.status is not CandidateStatus.READY:
                    raise CandidateConflictError(
                        f"candidate status is {candidate.status.value}, expected ready"
                    )
                if expected_revision is not None and candidate.revision != int(expected_revision):
                    raise CandidateConflictError(
                        f"candidate revision is {candidate.revision}, expected {expected_revision}"
                    )
                await self._require_promotion_evidence(connection, candidate)
                current = await self._current_release_in_transaction(connection)
                recorded = self._release_for_activation(
                    candidate=candidate,
                    release=release,
                    current=current,
                )
                promoted = await self._write_status_transition(
                    connection,
                    candidate,
                    CandidateStatus.PROMOTED,
                )
                await self._insert_release(connection, recorded)
                await connection.commit()
                return promoted, recorded
            except BaseException:
                await connection.rollback()
                raise

    async def record_promotion(self, release: Release) -> Release:
        """Record an already-promoted candidate, primarily for release bootstrap."""

        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                candidate = await self._candidate_in_transaction(connection, release.candidate_id)
                if candidate.status is not CandidateStatus.PROMOTED:
                    raise CandidateConflictError(
                        "candidate must be promoted before recording release"
                    )
                await self._require_promotion_evidence(connection, candidate)
                current = await self._current_release_in_transaction(connection)
                recorded = self._release_for_activation(
                    candidate=candidate,
                    release=release,
                    current=current,
                )
                await self._insert_release(connection, recorded)
                await connection.commit()
                return recorded
            except BaseException:
                await connection.rollback()
                raise

    async def activate_rollback(
        self,
        target_release_id: str,
        *,
        activation: Release,
        expected_current_release_id: str,
        expected_current_candidate_revision: int | None = None,
    ) -> tuple[Candidate, Release]:
        """Atomically reactivate the predecessor and mark the current candidate rolled back."""

        safe_target_id = self._identifier(target_release_id, field="target_release_id")
        safe_current_id = self._identifier(
            expected_current_release_id,
            field="expected_current_release_id",
        )
        async with self._lock:
            connection = self._require_connection()
            await connection.execute("BEGIN IMMEDIATE")
            try:
                current = await self._current_release_in_transaction(connection)
                if current is None or current.id != safe_current_id:
                    actual = current.id if current is not None else None
                    raise ReleaseConflictError(
                        f"current release is {actual!r}, expected {safe_current_id!r}"
                    )
                if current.previous_release_id != safe_target_id:
                    raise ReleaseConflictError(
                        "requested release is not the current rollback target"
                    )
                target = await self._release_in_transaction(connection, safe_target_id)
                if target is None:
                    raise EvolutionArchiveCorruptionError("rollback target release is missing")
                if current.candidate_id == target.candidate_id:
                    raise ReleaseConflictError("rollback target must belong to a prior candidate")

                current_candidate = await self._candidate_in_transaction(
                    connection,
                    current.candidate_id,
                )
                if current_candidate.status is not CandidateStatus.PROMOTED:
                    raise CandidateConflictError("current release candidate is not promoted")
                if (
                    expected_current_candidate_revision is not None
                    and current_candidate.revision != int(expected_current_candidate_revision)
                ):
                    raise CandidateConflictError(
                        "current candidate revision changed before rollback activation"
                    )
                target_candidate = await self._candidate_in_transaction(
                    connection,
                    target.candidate_id,
                )
                if target_candidate.status not in {
                    CandidateStatus.PROMOTED,
                    CandidateStatus.ROLLED_BACK,
                }:
                    raise CandidateConflictError(
                        "rollback target candidate was never safely promoted"
                    )
                await self._require_promotion_evidence(connection, target_candidate)
                if (
                    activation.candidate_id != target.candidate_id
                    or activation.source_commit != target.source_commit
                    or activation.artifact_digest != target.artifact_digest
                ):
                    raise ReleaseConflictError(
                        "rollback activation does not match the target release artifact"
                    )
                recorded = self._release_for_activation(
                    candidate=target_candidate,
                    release=activation,
                    current=current,
                )
                rolled_back = await self._write_status_transition(
                    connection,
                    current_candidate,
                    CandidateStatus.ROLLED_BACK,
                )
                if target_candidate.status is CandidateStatus.ROLLED_BACK:
                    await self._write_status_transition(
                        connection,
                        target_candidate,
                        CandidateStatus.PROMOTED,
                    )
                await self._insert_release(connection, recorded)
                await connection.commit()
                return rolled_back, recorded
            except BaseException:
                await connection.rollback()
                raise

    async def get_release(self, release_id: str) -> Release | None:
        safe_id = self._identifier(release_id, field="release_id")
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                "SELECT payload_json FROM evolution_releases WHERE id = ?",
                (safe_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._release_from_row(row) if row is not None else None

    async def get_current_release(self) -> Release | None:
        async with self._lock:
            connection = self._require_connection()
            return await self._current_release_in_transaction(connection)

    async def list_release_history(self, *, limit: int = 100) -> list[Release]:
        safe_limit = max(1, min(int(limit), 1_000))
        async with self._lock:
            connection = self._require_connection()
            cursor = await connection.execute(
                """
                SELECT payload_json FROM evolution_releases
                ORDER BY sequence DESC
                LIMIT ?
                """,
                (safe_limit,),
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._release_from_row(row) for row in rows]

    async def get_rollback_target(self) -> Release | None:
        """Return the predecessor captured when the current release was activated."""

        async with self._lock:
            connection = self._require_connection()
            current = await self._current_release_in_transaction(connection)
            if current is None or current.previous_release_id is None:
                return None
            cursor = await connection.execute(
                "SELECT payload_json FROM evolution_releases WHERE id = ?",
                (current.previous_release_id,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                raise EvolutionArchiveCorruptionError(
                    "current release references a missing rollback target"
                )
            return self._release_from_row(row)

    async def current_release(self) -> Release | None:
        return await self.get_current_release()

    async def release_history(self, *, limit: int = 100) -> list[Release]:
        return await self.list_release_history(limit=limit)

    async def rollback_target(self) -> Release | None:
        return await self.get_rollback_target()

    async def _require_promotion_evidence(
        self,
        connection: aiosqlite.Connection,
        candidate: Candidate,
    ) -> None:
        report = candidate.evaluation_report
        if report is None or not report.passed:
            raise InvalidCandidateTransitionError("candidate requires a passing evaluation report")
        if candidate.source_commit is None:
            raise InvalidCandidateTransitionError("candidate requires a source commit")
        if candidate.artifact_digest is None:
            raise InvalidCandidateTransitionError("candidate requires an artifact digest")
        if report.source_commit != candidate.source_commit:
            raise InvalidCandidateTransitionError(
                "candidate evaluation does not match its source commit"
            )
        if report.artifact_digest != candidate.artifact_digest:
            raise InvalidCandidateTransitionError(
                "candidate evaluation does not match its artifact digest"
            )
        if report.evaluator_fingerprint != candidate.evaluator_fingerprint:
            raise InvalidCandidateTransitionError(
                "candidate evaluation does not match its evaluator configuration"
            )
        cursor = await connection.execute(
            """
            SELECT 1 FROM evolution_evaluations
            WHERE candidate_id = ? AND id = ?
            """,
            (candidate.id, report.id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise InvalidCandidateTransitionError(
                "candidate's latest evaluation was not appended to the archive"
            )

    async def _write_status_transition(
        self,
        connection: aiosqlite.Connection,
        current: Candidate,
        target: CandidateStatus,
    ) -> Candidate:
        updated = Candidate.model_validate(
            {
                **current.model_dump(mode="python"),
                "status": target,
                "revision": current.revision + 1,
                "updated_at": self._next_timestamp(current.updated_at),
            }
        )
        cursor = await connection.execute(
            """
            UPDATE evolution_candidates
            SET status = ?, revision = ?, updated_at = ?, payload_json = ?
            WHERE id = ? AND status = ? AND revision = ?
            """,
            (
                target.value,
                updated.revision,
                updated.updated_at.isoformat(),
                self._serialize(updated),
                current.id,
                current.status.value,
                current.revision,
            ),
        )
        if cursor.rowcount != 1:
            raise CandidateConflictError("candidate changed during status transition")
        return updated

    @staticmethod
    def _release_for_activation(
        *,
        candidate: Candidate,
        release: Release,
        current: Release | None,
    ) -> Release:
        if candidate.source_commit != release.source_commit:
            raise ReleaseConflictError("release source commit does not match its candidate")
        if candidate.artifact_digest != release.artifact_digest:
            raise ReleaseConflictError("release artifact digest does not match its candidate")
        current_id = current.id if current is not None else None
        if release.previous_release_id is not None and release.previous_release_id != current_id:
            raise ReleaseConflictError("release predecessor does not match the current release")
        if current is not None and release.promoted_at < current.promoted_at:
            raise ReleaseConflictError("promotion timestamp predates the current release")
        return Release.model_validate(
            {
                **release.model_dump(mode="python"),
                "previous_release_id": current_id,
            }
        )

    async def _insert_release(
        self,
        connection: aiosqlite.Connection,
        release: Release,
    ) -> None:
        try:
            await connection.execute(
                """
                INSERT INTO evolution_releases (
                    id, candidate_id, previous_release_id, promoted_at, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    release.id,
                    release.candidate_id,
                    release.previous_release_id,
                    release.promoted_at.isoformat(),
                    self._serialize(release),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise ReleaseConflictError(f"release {release.id!r} already exists") from exc
        await connection.execute(
            """
            UPDATE evolution_state
            SET current_release_id = ?
            WHERE singleton = 1
            """,
            (release.id,),
        )

    def _require_connection(self) -> aiosqlite.Connection:
        if self._connection is None:
            raise EvolutionArchiveNotStartedError("evolution archive has not been started")
        return self._connection

    async def _candidate_in_transaction(
        self,
        connection: aiosqlite.Connection,
        candidate_id: str,
    ) -> Candidate:
        cursor = await connection.execute(
            "SELECT payload_json FROM evolution_candidates WHERE id = ?",
            (candidate_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise CandidateNotFoundError(candidate_id)
        return self._candidate_from_row(row)

    async def _migrate_legacy_promotion_approvals(
        self,
        connection: aiosqlite.Connection,
    ) -> None:
        """Remove the retired approval envelope without auto-running pending rows."""

        cursor = await connection.execute(
            """
            SELECT id, payload_json FROM evolution_promotion_attempts
            ORDER BY created_at ASC, id ASC
            """
        )
        rows = await cursor.fetchall()
        await cursor.close()
        for row in rows:
            try:
                raw = json.loads(str(row["payload_json"]))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise EvolutionArchiveCorruptionError(
                    "invalid persisted PromotionAttempt record"
                ) from exc
            if not isinstance(raw, dict) or (
                "approval_status" not in raw and "approval_audit" not in raw
            ):
                continue
            approval_status = str(raw.pop("approval_status", "not_required"))
            raw.pop("approval_audit", None)
            if approval_status not in {
                "not_required",
                "pending",
                "approved",
                "rejected",
            }:
                raise EvolutionArchiveCorruptionError(
                    "persisted promotion attempt has an invalid approval state"
                )
            if approval_status == "pending" and raw.get("status") == "queued":
                raw.update(
                    {
                        "status": PromotionAttemptStatus.FAILED.value,
                        "revision": int(raw.get("revision", 1)) + 1,
                        "failure_code": "obsolete_approval_state",
                        "failure_message": (
                            "A retired stable approval request was cancelled during upgrade."
                        ),
                    }
                )
            try:
                attempt = PromotionAttempt.model_validate(raw)
            except ValidationError as exc:
                raise EvolutionArchiveCorruptionError(
                    "invalid persisted PromotionAttempt record"
                ) from exc
            await connection.execute(
                """
                UPDATE evolution_promotion_attempts
                SET status = ?, revision = ?, updated_at = ?, payload_json = ?
                WHERE id = ?
                """,
                (
                    attempt.status.value,
                    attempt.revision,
                    attempt.updated_at.isoformat(),
                    self._serialize(attempt),
                    attempt.id,
                ),
            )

    async def _release_in_transaction(
        self,
        connection: aiosqlite.Connection,
        release_id: str,
    ) -> Release | None:
        cursor = await connection.execute(
            "SELECT payload_json FROM evolution_releases WHERE id = ?",
            (release_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._release_from_row(row) if row is not None else None

    async def _current_release_in_transaction(
        self,
        connection: aiosqlite.Connection,
    ) -> Release | None:
        cursor = await connection.execute(
            """
            SELECT releases.payload_json
            FROM evolution_state AS state
            LEFT JOIN evolution_releases AS releases
              ON releases.id = state.current_release_id
            WHERE state.singleton = 1
            """
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None or row["payload_json"] is None:
            return None
        return self._release_from_row(row)

    def _next_timestamp(self, current: datetime) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("evolution archive clock must return an aware datetime")
        return max(now.astimezone(UTC), current.astimezone(UTC))

    @staticmethod
    def _identifier(value: str, *, field: str) -> str:
        safe = str(value or "").strip()
        if not safe:
            raise ValueError(f"{field} is required")
        if len(safe) > 100:
            raise ValueError(f"{field} must be at most 100 characters")
        return safe

    @staticmethod
    def _bounded_value(value: str, *, field: str, max_length: int) -> str:
        safe = str(value or "").strip()
        if not safe:
            raise ValueError(f"{field} is required")
        if len(safe) > max_length:
            raise ValueError(f"{field} must be at most {max_length} characters")
        return safe

    @staticmethod
    def _serialize(value: BaseModel) -> str:
        return json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _deserialize(cls, row: aiosqlite.Row, model: type[BaseModel]) -> BaseModel:
        try:
            raw: Any = json.loads(str(row["payload_json"]))
            return model.model_validate(raw)
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise EvolutionArchiveCorruptionError(
                f"invalid persisted {model.__name__} record"
            ) from exc

    @classmethod
    def _candidate_from_row(cls, row: aiosqlite.Row) -> Candidate:
        value = cls._deserialize(row, Candidate)
        if not isinstance(value, Candidate):  # pragma: no cover - type narrowing
            raise EvolutionArchiveCorruptionError("persisted candidate has an invalid type")
        return value

    @classmethod
    def _report_from_row(cls, row: aiosqlite.Row) -> EvaluationReport:
        value = cls._deserialize(row, EvaluationReport)
        if not isinstance(value, EvaluationReport):  # pragma: no cover - type narrowing
            raise EvolutionArchiveCorruptionError("persisted evaluation has an invalid type")
        return value

    @classmethod
    def _release_from_row(cls, row: aiosqlite.Row) -> Release:
        value = cls._deserialize(row, Release)
        if not isinstance(value, Release):  # pragma: no cover - type narrowing
            raise EvolutionArchiveCorruptionError("persisted release has an invalid type")
        return value

    @classmethod
    def _promotion_attempt_from_row(cls, row: aiosqlite.Row) -> PromotionAttempt:
        value = cls._deserialize(row, PromotionAttempt)
        if not isinstance(value, PromotionAttempt):  # pragma: no cover - type narrowing
            raise EvolutionArchiveCorruptionError("persisted promotion attempt has an invalid type")
        return value

    @classmethod
    def _source_release_operation_from_row(
        cls,
        row: aiosqlite.Row,
    ) -> SourceReleaseOperation:
        value = cls._deserialize(row, SourceReleaseOperation)
        if not isinstance(value, SourceReleaseOperation):  # pragma: no cover
            raise EvolutionArchiveCorruptionError(
                "persisted source release operation has an invalid type"
            )
        return value

    @classmethod
    def _event_from_row(cls, row: aiosqlite.Row) -> EvolutionEvent:
        value = cls._deserialize(row, EvolutionEvent)
        if not isinstance(value, EvolutionEvent):  # pragma: no cover - type narrowing
            raise EvolutionArchiveCorruptionError("persisted evolution event has an invalid type")
        return value


__all__ = [
    "CandidateAlreadyExistsError",
    "CandidateConflictError",
    "CandidateNotFoundError",
    "EvaluationAlreadyExistsError",
    "EvolutionArchive",
    "EvolutionArchiveCorruptionError",
    "EvolutionArchiveError",
    "EvolutionArchiveNotStartedError",
    "InvalidCandidateTransitionError",
    "PromotionAttemptConflictError",
    "ReleaseConflictError",
    "SourceReleaseOperationConflictError",
]
