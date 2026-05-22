"""Durable thread-scoped compressed context summaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.persistence.sqlite import connect_sqlite


class ThreadRollupService:
    """Store one rolling summary per LangGraph thread."""

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
                CREATE TABLE IF NOT EXISTS thread_rollups (
                    thread_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def get_rollup(self, thread_id: str) -> str | None:
        payload = self.get_rollup_payload(thread_id)
        if not payload:
            return None
        summary = str(payload.get("conversation_summary") or "").strip()
        if summary:
            return summary
        open_loops = str(payload.get("open_loops") or "").strip()
        durable_facts = str(payload.get("durable_facts") or "").strip()
        pieces = [part for part in (open_loops, durable_facts) if part]
        return "\n\n".join(pieces).strip() or None

    def get_rollup_payload(self, thread_id: str) -> dict[str, str] | None:
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
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            return {
                "conversation_summary": str(parsed.get("conversation_summary") or "").strip(),
                "open_loops": str(parsed.get("open_loops") or "").strip(),
                "durable_facts": str(parsed.get("durable_facts") or "").strip(),
                "sensitive_refs": str(parsed.get("sensitive_refs") or "").strip(),
                "style_notes": str(parsed.get("style_notes") or "").strip(),
            }
        return {
            "conversation_summary": text,
            "open_loops": "",
            "durable_facts": "",
            "sensitive_refs": "",
            "style_notes": "",
        }

    def set_rollup(self, thread_id: str, summary: str) -> None:
        self.set_rollup_payload(
            thread_id,
            {
                "conversation_summary": summary,
                "open_loops": "",
                "durable_facts": "",
                "sensitive_refs": "",
                "style_notes": "",
            },
        )

    def set_rollup_payload(self, thread_id: str, payload: dict[str, Any]) -> None:
        tid = str(thread_id or "").strip()
        normalized = {
            "conversation_summary": str(payload.get("conversation_summary") or "").strip(),
            "open_loops": str(payload.get("open_loops") or "").strip(),
            "durable_facts": str(payload.get("durable_facts") or "").strip(),
            "sensitive_refs": str(payload.get("sensitive_refs") or "").strip(),
            "style_notes": str(payload.get("style_notes") or "").strip(),
        }
        text = json.dumps(normalized, ensure_ascii=False)
        if not tid:
            raise ValueError("thread_id is required")
        if not any(normalized.values()):
            raise ValueError("rollup payload is required")
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
