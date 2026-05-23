"""Durable customer-scoped deferred context event storage."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.persistence.sqlite import connect_sqlite


class EventContextService:
    """Persist non-urgent events to inject into the next user turn."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS context_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    customer_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_context_events_customer
                    ON context_events(customer_id, id ASC);
                """
            )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def add_event(
        self,
        *,
        customer_id: str,
        source: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        cid = str(customer_id or "").strip()
        if not cid:
            raise ValueError("customer_id is required")
        safe_payload = payload if isinstance(payload, dict) else {}
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO context_events (customer_id, source, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cid,
                    str(source or "unknown"),
                    str(event_type or "event"),
                    json.dumps(safe_payload, ensure_ascii=False),
                    self._utc_now_iso(),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)

    def list_events(self, customer_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        if not cid:
            return []
        safe_limit = max(1, min(int(limit), 200))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, customer_id, source, event_type, payload_json, created_at
                FROM context_events
                WHERE customer_id=?
                ORDER BY id ASC
                LIMIT ?
                """,
                (cid, safe_limit),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            raw_payload = row["payload_json"] or "{}"
            try:
                payload = json.loads(raw_payload)
            except Exception:
                payload = {"raw": str(raw_payload)}
            if not isinstance(payload, dict):
                payload = {"raw": payload}
            out.append(
                {
                    "id": int(row["id"]),
                    "customer_id": str(row["customer_id"]),
                    "source": str(row["source"]),
                    "event_type": str(row["event_type"]),
                    "payload": payload,
                    "created_at": str(row["created_at"]),
                }
            )
        return out

    def clear_events(self, customer_id: str, *, through_id: int | None = None) -> int:
        cid = str(customer_id or "").strip()
        if not cid:
            return 0
        with self._conn() as conn:
            if through_id is None:
                cur = conn.execute("DELETE FROM context_events WHERE customer_id=?", (cid,))
            else:
                cur = conn.execute(
                    "DELETE FROM context_events WHERE customer_id=? AND id<=?",
                    (cid, int(through_id)),
                )
            conn.commit()
            return int(cur.rowcount or 0)
