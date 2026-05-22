"""Durable wake-event queue with ordered async dispatch and retries."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sqlite3
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from opentulpa.persistence.sqlite import connect_sqlite

logger = logging.getLogger(__name__)

WakeHandler = Callable[[dict[str, Any]], Awaitable[None]]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class WakeQueueService:
    """SQLite-backed queue for wake payloads."""

    def __init__(self, db_path: Path, handler: WakeHandler) -> None:
        self.db_path = db_path
        self._handler = handler
        self._runner: asyncio.Task[Any] | None = None
        self._stop = asyncio.Event()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS wake_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    next_attempt_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_wake_events_pending
                    ON wake_events(status, next_attempt_at, id);
                """
            )

    async def start(self) -> None:
        with self._conn() as conn:
            # Recover events left "processing" during previous shutdown.
            conn.execute(
                """
                UPDATE wake_events
                SET status='pending', updated_at=?
                WHERE status='processing'
                """,
                (_utc_now_iso(),),
            )
            conn.commit()
        self._stop.clear()
        self._runner = asyncio.create_task(self._run_loop())

    async def shutdown(self) -> None:
        self._stop.set()
        if self._runner:
            self._runner.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._runner
            self._runner = None

    async def enqueue(self, payload: dict[str, Any]) -> int:
        now = _utc_now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO wake_events (
                    payload_json, status, attempts, last_error, next_attempt_at, created_at, updated_at
                ) VALUES (?, 'pending', 0, NULL, ?, ?, ?)
                """,
                (json.dumps(payload), now, now, now),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            pending = conn.execute(
                "SELECT COUNT(*) FROM wake_events WHERE status='pending'"
            ).fetchone()[0]
            processing = conn.execute(
                "SELECT COUNT(*) FROM wake_events WHERE status='processing'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM wake_events WHERE status='failed'"
            ).fetchone()[0]
            last = conn.execute(
                """
                SELECT id, status, attempts, last_error, next_attempt_at, created_at, updated_at
                FROM wake_events
                ORDER BY id DESC
                LIMIT 20
                """
            ).fetchall()
        return {
            "pending": int(pending),
            "processing": int(processing),
            "failed": int(failed),
            "recent": [dict(r) for r in last],
        }

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            row = self._claim_next()
            if row is None:
                await asyncio.sleep(1.0)
                continue

            item_id = int(row["id"])
            try:
                payload = json.loads(row["payload_json"])
                if not isinstance(payload, dict):
                    raise ValueError("wake payload must be a JSON object")
                await self._handler(payload)
                self._mark_done(item_id)
            except asyncio.CancelledError:
                self._requeue(item_id, "dispatcher_cancelled", attempts=int(row["attempts"]))
                raise
            except Exception as exc:
                self._requeue(item_id, str(exc), attempts=int(row["attempts"]))

    def _claim_next(self) -> sqlite3.Row | None:
        now = _utc_now_iso()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT id, payload_json, attempts
                FROM wake_events
                WHERE status='pending' AND next_attempt_at <= ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (now,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                """
                UPDATE wake_events
                SET status='processing', updated_at=?
                WHERE id=?
                """,
                (now, row["id"]),
            )
            conn.commit()
            return cast("sqlite3.Row | None", row)

    def _mark_done(self, item_id: int) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE wake_events
                SET status='done', updated_at=?
                WHERE id=?
                """,
                (_utc_now_iso(), item_id),
            )
            conn.commit()

    def _requeue(self, item_id: int, error: str, *, attempts: int) -> None:
        attempts_used = max(0, int(attempts)) + 1
        delay_seconds = min(60, 2 ** min(6, attempts_used))
        next_attempt = (_utc_now() + timedelta(seconds=delay_seconds)).isoformat()
        status = "failed" if attempts_used >= 20 else "pending"
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE wake_events
                SET status=?, attempts=?, last_error=?, next_attempt_at=?, updated_at=?
                WHERE id=?
                """,
                (status, attempts_used, error[:1000], next_attempt, _utc_now_iso(), item_id),
            )
            conn.commit()
        logger.warning("Wake event %s requeued (attempt=%s): %s", item_id, attempts_used, error)
