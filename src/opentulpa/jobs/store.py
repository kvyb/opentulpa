"""Tenant-owned SQLite persistence for deterministic jobs."""

from __future__ import annotations

import hmac
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.jobs.models import (
    Job,
    JobArtifact,
    JobError,
    JobEvent,
    JobEventType,
    JobHandlerResult,
)
from opentulpa.persistence.sqlite import connect_sqlite

_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


class JobNotFoundError(KeyError):
    """The tenant-owned job does not exist."""


class JobIdempotencyConflictError(RuntimeError):
    """An idempotency key was reused for a different job request."""


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    handler_name TEXT NOT NULL,
                    handler_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    idempotency_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    PRIMARY KEY (tenant_id, id),
                    UNIQUE (tenant_id, idempotency_key)
                );

                CREATE INDEX IF NOT EXISTS idx_jobs_tenant_created
                ON jobs (tenant_id, created_at DESC, id);

                CREATE INDEX IF NOT EXISTS idx_jobs_recovery
                ON jobs (status, updated_at, id);

                CREATE TABLE IF NOT EXISTS job_events (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, job_id, sequence)
                );

                CREATE TABLE IF NOT EXISTS job_artifacts (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    uri TEXT NOT NULL,
                    size_bytes INTEGER,
                    sha256 TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, job_id, id)
                );

                CREATE INDEX IF NOT EXISTS idx_job_artifacts_job
                ON job_artifacts (tenant_id, job_id, created_at, id);
                """
            )
            conn.commit()

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _row_to_job(cls, row: sqlite3.Row) -> Job:
        return Job.model_validate(
            {
                "id": row["id"],
                "tenant_id": row["tenant_id"],
                "handler_name": row["handler_name"],
                "handler_version": row["handler_version"],
                "status": row["status"],
                "arguments": json.loads(str(row["arguments_json"])),
                "result": json.loads(str(row["result_json"])) if row["result_json"] else None,
                "error": json.loads(str(row["error_json"])) if row["error_json"] else None,
                "idempotency_key": row["idempotency_key"],
                "attempt_count": row["attempt_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
            }
        )

    @classmethod
    def _append_event(
        cls,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        job_id: str,
        event_type: JobEventType,
        payload: dict[str, Any],
        now: datetime,
    ) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
            FROM job_events WHERE tenant_id = ? AND job_id = ?
            """,
            (tenant_id, job_id),
        ).fetchone()
        sequence = int(row["next_sequence"])
        conn.execute(
            """
            INSERT INTO job_events (
                tenant_id, job_id, sequence, event_type, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                job_id,
                sequence,
                event_type,
                cls._json(payload),
                now.isoformat(),
            ),
        )
        return sequence

    def create(
        self,
        *,
        tenant_id: str,
        job_id: str,
        handler_name: str,
        handler_version: int,
        arguments: dict[str, Any],
        idempotency_key: str,
        request_hash: str,
        now: datetime,
    ) -> tuple[Job, bool]:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND idempotency_key = ?",
                (tenant_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                if not hmac.compare_digest(str(existing["request_hash"]), request_hash):
                    conn.rollback()
                    raise JobIdempotencyConflictError(
                        "idempotency key is already bound to a different job request"
                    )
                conn.commit()
                return self._row_to_job(existing), False
            conn.execute(
                """
                INSERT INTO jobs (
                    tenant_id, id, handler_name, handler_version, status,
                    arguments_json, result_json, error_json, idempotency_key,
                    request_hash, attempt_count, created_at, updated_at,
                    started_at, finished_at
                ) VALUES (?, ?, ?, ?, 'queued', ?, NULL, NULL, ?, ?, 0, ?, ?, NULL, NULL)
                """,
                (
                    tenant_id,
                    job_id,
                    handler_name,
                    handler_version,
                    self._json(arguments),
                    idempotency_key,
                    request_hash,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            self._append_event(
                conn,
                tenant_id=tenant_id,
                job_id=job_id,
                event_type="queued",
                payload={"handler": handler_name, "version": handler_version},
                now=now,
            )
            conn.commit()
        job = self.get(tenant_id=tenant_id, job_id=job_id)
        return job, True

    def get(self, *, tenant_id: str, job_id: str) -> Job:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return self._row_to_job(row)

    def events(
        self,
        *,
        tenant_id: str,
        job_id: str,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> list[JobEvent]:
        self.get(tenant_id=tenant_id, job_id=job_id)
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_events
                WHERE tenant_id = ? AND job_id = ? AND sequence > ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (tenant_id, job_id, max(0, after_sequence), max(1, min(limit, 500))),
            ).fetchall()
        return [
            JobEvent.model_validate(
                {
                    "tenant_id": row["tenant_id"],
                    "job_id": row["job_id"],
                    "sequence": row["sequence"],
                    "event_type": row["event_type"],
                    "payload": json.loads(str(row["payload_json"])),
                    "created_at": row["created_at"],
                }
            )
            for row in rows
        ]

    def artifacts(self, *, tenant_id: str, job_id: str) -> list[JobArtifact]:
        self.get(tenant_id=tenant_id, job_id=job_id)
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM job_artifacts
                WHERE tenant_id = ? AND job_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (tenant_id, job_id),
            ).fetchall()
        return [
            JobArtifact.model_validate(
                {
                    "tenant_id": row["tenant_id"],
                    "job_id": row["job_id"],
                    "id": row["id"],
                    "name": row["name"],
                    "media_type": row["media_type"],
                    "uri": row["uri"],
                    "size_bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                    "metadata": json.loads(str(row["metadata_json"])),
                    "created_at": row["created_at"],
                }
            )
            for row in rows
        ]

    def get_artifact(self, *, tenant_id: str, artifact_id: str) -> JobArtifact:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM job_artifacts WHERE tenant_id = ? AND id = ?",
                (tenant_id, artifact_id),
            ).fetchone()
        if row is None:
            raise JobNotFoundError(artifact_id)
        return JobArtifact.model_validate(
            {
                "tenant_id": row["tenant_id"],
                "job_id": row["job_id"],
                "id": row["id"],
                "name": row["name"],
                "media_type": row["media_type"],
                "uri": row["uri"],
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "metadata": json.loads(str(row["metadata_json"])),
                "created_at": row["created_at"],
            }
        )

    def recover(self, *, now: datetime) -> list[Job]:
        recovered_ids: list[tuple[str, str]] = []
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status IN ('queued', 'running', 'cancel_requested')
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
            for row in rows:
                tenant_id = str(row["tenant_id"])
                job_id = str(row["id"])
                status = str(row["status"])
                if status == "cancel_requested":
                    conn.execute(
                        """
                        UPDATE jobs SET status = 'cancelled', updated_at = ?, finished_at = ?
                        WHERE tenant_id = ? AND id = ? AND status = 'cancel_requested'
                        """,
                        (now.isoformat(), now.isoformat(), tenant_id, job_id),
                    )
                    self._append_event(
                        conn,
                        tenant_id=tenant_id,
                        job_id=job_id,
                        event_type="cancelled",
                        payload={"reason": "restart_after_cancel_request"},
                        now=now,
                    )
                    continue
                if status == "running":
                    conn.execute(
                        """
                        UPDATE jobs SET status = 'queued', updated_at = ?,
                            started_at = NULL, finished_at = NULL
                        WHERE tenant_id = ? AND id = ? AND status = 'running'
                        """,
                        (now.isoformat(), tenant_id, job_id),
                    )
                    self._append_event(
                        conn,
                        tenant_id=tenant_id,
                        job_id=job_id,
                        event_type="recovered",
                        payload={"reason": "service_restart"},
                        now=now,
                    )
                recovered_ids.append((tenant_id, job_id))
            conn.commit()
        return [self.get(tenant_id=tenant, job_id=job) for tenant, job in recovered_ids]

    def claim(self, *, tenant_id: str, job_id: str, now: datetime) -> Job | None:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            if str(row["status"]) != "queued":
                conn.commit()
                return None
            conn.execute(
                """
                UPDATE jobs SET status = 'running', attempt_count = attempt_count + 1,
                    updated_at = ?, started_at = ?, finished_at = NULL, error_json = NULL
                WHERE tenant_id = ? AND id = ? AND status = 'queued'
                """,
                (now.isoformat(), now.isoformat(), tenant_id, job_id),
            )
            attempt = int(
                conn.execute(
                    "SELECT attempt_count FROM jobs WHERE tenant_id = ? AND id = ?",
                    (tenant_id, job_id),
                ).fetchone()["attempt_count"]
            )
            self._append_event(
                conn,
                tenant_id=tenant_id,
                job_id=job_id,
                event_type="running",
                payload={"attempt": attempt},
                now=now,
            )
            conn.commit()
        return self.get(tenant_id=tenant_id, job_id=job_id)

    def progress(
        self,
        *,
        tenant_id: str,
        job_id: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> None:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            if str(row["status"]) != "running":
                conn.commit()
                return
            self._append_event(
                conn,
                tenant_id=tenant_id,
                job_id=job_id,
                event_type="progress",
                payload=payload,
                now=now,
            )
            conn.commit()

    def complete(
        self,
        *,
        tenant_id: str,
        job_id: str,
        result: JobHandlerResult,
        now: datetime,
    ) -> Job:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            status = str(row["status"])
            if status == "cancel_requested":
                self._cancel_in_connection(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    reason="cancelled_before_completion",
                    now=now,
                )
                conn.commit()
                return self.get(tenant_id=tenant_id, job_id=job_id)
            if status != "running":
                conn.commit()
                return self.get(tenant_id=tenant_id, job_id=job_id)
            for artifact_write in result.artifacts:
                artifact_id = new_short_id("artifact")
                conn.execute(
                    """
                    INSERT INTO job_artifacts (
                        tenant_id, job_id, id, name, media_type, uri,
                        size_bytes, sha256, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        job_id,
                        artifact_id,
                        artifact_write.name,
                        artifact_write.media_type,
                        artifact_write.uri,
                        artifact_write.size_bytes,
                        artifact_write.sha256,
                        self._json(artifact_write.metadata),
                        now.isoformat(),
                    ),
                )
                self._append_event(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    event_type="artifact.ready",
                    payload={"artifact_id": artifact_id, "name": artifact_write.name},
                    now=now,
                )
            conn.execute(
                """
                UPDATE jobs SET status = 'succeeded', result_json = ?, error_json = NULL,
                    updated_at = ?, finished_at = ?
                WHERE tenant_id = ? AND id = ? AND status = 'running'
                """,
                (
                    self._json(result.model_dump(mode="json")),
                    now.isoformat(),
                    now.isoformat(),
                    tenant_id,
                    job_id,
                ),
            )
            self._append_event(
                conn,
                tenant_id=tenant_id,
                job_id=job_id,
                event_type="completed",
                payload={"summary": result.summary, "artifact_count": len(result.artifacts)},
                now=now,
            )
            conn.commit()
        return self.get(tenant_id=tenant_id, job_id=job_id)

    def fail(
        self,
        *,
        tenant_id: str,
        job_id: str,
        error: JobError,
        now: datetime,
    ) -> Job:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            if str(row["status"]) == "cancel_requested":
                self._cancel_in_connection(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    reason="cancelled_during_failure",
                    now=now,
                )
            elif str(row["status"]) == "running":
                conn.execute(
                    """
                    UPDATE jobs SET status = 'failed', error_json = ?, result_json = NULL,
                        updated_at = ?, finished_at = ?
                    WHERE tenant_id = ? AND id = ? AND status = 'running'
                    """,
                    (
                        self._json(error.model_dump(mode="json")),
                        now.isoformat(),
                        now.isoformat(),
                        tenant_id,
                        job_id,
                    ),
                )
                self._append_event(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    event_type="failed",
                    payload=error.model_dump(mode="json"),
                    now=now,
                )
            conn.commit()
        return self.get(tenant_id=tenant_id, job_id=job_id)

    def request_cancel(self, *, tenant_id: str, job_id: str, now: datetime) -> Job:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            status = str(row["status"])
            if status in _TERMINAL_STATUSES or status == "cancel_requested":
                conn.commit()
                return self.get(tenant_id=tenant_id, job_id=job_id)
            self._append_event(
                conn,
                tenant_id=tenant_id,
                job_id=job_id,
                event_type="cancel_requested",
                payload={},
                now=now,
            )
            if status == "queued":
                self._cancel_in_connection(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    reason="cancelled_before_start",
                    now=now,
                )
            else:
                conn.execute(
                    """
                    UPDATE jobs SET status = 'cancel_requested', updated_at = ?
                    WHERE tenant_id = ? AND id = ? AND status = 'running'
                    """,
                    (now.isoformat(), tenant_id, job_id),
                )
            conn.commit()
        return self.get(tenant_id=tenant_id, job_id=job_id)

    def mark_cancelled(
        self,
        *,
        tenant_id: str,
        job_id: str,
        reason: str,
        now: datetime,
    ) -> Job:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            if str(row["status"]) not in _TERMINAL_STATUSES:
                self._cancel_in_connection(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    reason=reason,
                    now=now,
                )
            conn.commit()
        return self.get(tenant_id=tenant_id, job_id=job_id)

    def requeue_interrupted(
        self,
        *,
        tenant_id: str,
        job_id: str,
        now: datetime,
    ) -> Job:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status FROM jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise JobNotFoundError(job_id)
            status = str(row["status"])
            if status == "cancel_requested":
                self._cancel_in_connection(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    reason="cancelled_during_shutdown",
                    now=now,
                )
            elif status == "running":
                conn.execute(
                    """
                    UPDATE jobs SET status = 'queued', updated_at = ?, started_at = NULL
                    WHERE tenant_id = ? AND id = ? AND status = 'running'
                    """,
                    (now.isoformat(), tenant_id, job_id),
                )
                self._append_event(
                    conn,
                    tenant_id=tenant_id,
                    job_id=job_id,
                    event_type="recovered",
                    payload={"reason": "service_shutdown"},
                    now=now,
                )
            conn.commit()
        return self.get(tenant_id=tenant_id, job_id=job_id)

    @classmethod
    def _cancel_in_connection(
        cls,
        conn: sqlite3.Connection,
        *,
        tenant_id: str,
        job_id: str,
        reason: str,
        now: datetime,
    ) -> None:
        conn.execute(
            """
            UPDATE jobs SET status = 'cancelled', updated_at = ?, finished_at = ?
            WHERE tenant_id = ? AND id = ?
              AND status NOT IN ('succeeded', 'failed', 'cancelled')
            """,
            (now.isoformat(), now.isoformat(), tenant_id, job_id),
        )
        cls._append_event(
            conn,
            tenant_id=tenant_id,
            job_id=job_id,
            event_type="cancelled",
            payload={"reason": reason},
            now=now,
        )


__all__ = [
    "JobIdempotencyConflictError",
    "JobNotFoundError",
    "JobStore",
]
