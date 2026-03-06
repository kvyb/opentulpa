"""Durable thread-scoped compressed context summaries."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class ThreadRollupService:
    """Store one rolling summary per LangGraph thread."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS thread_rollups (
                    thread_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def get_rollup(self, thread_id: str) -> str | None:
        tid = str(thread_id or "").strip()
        if not tid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT summary_text
                FROM thread_rollups
                WHERE thread_id=?
                """,
                (tid,),
            ).fetchone()
        if not row:
            return None
        text = str(row["summary_text"] or "").strip()
        return text or None

    def set_rollup(self, thread_id: str, summary: str) -> None:
        tid = str(thread_id or "").strip()
        text = str(summary or "").strip()
        if not tid:
            raise ValueError("thread_id is required")
        if not text:
            raise ValueError("summary is required")
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO thread_rollups (thread_id, summary_text, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(thread_id)
                DO UPDATE SET
                    summary_text=excluded.summary_text,
                    updated_at=excluded.updated_at
                """,
                (tid, text, self._utc_now_iso()),
            )
            conn.commit()

    def clear_rollup(self, thread_id: str) -> bool:
        tid = str(thread_id or "").strip()
        if not tid:
            return False
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM thread_rollups WHERE thread_id=?", (tid,))
            conn.commit()
            return bool(cur.rowcount)
