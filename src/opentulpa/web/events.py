"""Durable web-visible event outbox."""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WebEventStore:
    """SQLite-backed event outbox for dashboard consumers."""

    def __init__(self, db_path: Path) -> None:
        assert db_path is not None
        self._db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS web_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_web_events_id ON web_events(id)")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_web_events_customer_thread
                ON web_events(customer_id, thread_id, id)
                """
            )

    def append(
        self,
        *,
        customer_id: str,
        thread_id: str,
        source: str,
        kind: str,
        text: str,
        metadata_json: str = "{}",
    ) -> int:
        assert source.strip()
        assert kind.strip()
        body = str(text or "").strip()
        if not body:
            return 0
        created_at = datetime.now(UTC).isoformat()
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO web_events (
                    created_at, customer_id, thread_id, source, kind, text, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    str(customer_id or "").strip(),
                    str(thread_id or "").strip(),
                    source.strip(),
                    kind.strip(),
                    body,
                    metadata_json.strip() or "{}",
                ),
            )
            event_id = int(cursor.lastrowid or 0)
        assert event_id > 0
        return event_id

    def list_events(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
        customer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 500))
        conditions = ["id > ?"]
        params: list[Any] = [max(0, int(after_id))]
        if customer_id:
            conditions.append("customer_id = ?")
            params.append(customer_id.strip())
        query = (
            "SELECT id, created_at, customer_id, thread_id, source, kind, text, metadata_json "
            f"FROM web_events WHERE {' AND '.join(conditions)} ORDER BY id ASC LIMIT ?"
        )
        params.append(safe_limit)
        with self._lock, self._connect() as conn:
            rows: Iterable[sqlite3.Row] = conn.execute(query, params).fetchall()
        return [
            {
                "id": int(row["id"]),
                "created_at": str(row["created_at"]),
                "customer_id": str(row["customer_id"]),
                "thread_id": str(row["thread_id"]),
                "source": str(row["source"]),
                "kind": str(row["kind"]),
                "text": str(row["text"]),
                "metadata_json": str(row["metadata_json"] or "{}"),
            }
            for row in rows
        ]


_DEFAULT_STORE: WebEventStore | None = None


def set_default_web_event_store(store: WebEventStore | None) -> None:
    global _DEFAULT_STORE
    _DEFAULT_STORE = store


def append_web_event(
    *,
    customer_id: str,
    thread_id: str,
    source: str,
    kind: str,
    text: str,
    metadata_json: str = "{}",
) -> int:
    store = _DEFAULT_STORE
    if store is None:
        return 0
    try:
        return store.append(
            customer_id=customer_id,
            thread_id=thread_id,
            source=source,
            kind=kind,
            text=text,
            metadata_json=metadata_json,
        )
    except Exception:
        logger.exception("failed to append web event source=%s kind=%s", source, kind)
        return 0
