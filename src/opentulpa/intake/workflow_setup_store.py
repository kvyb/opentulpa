"""Persistent storage for intake workflow setup sessions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite

SetupSessionStatus = Literal["active", "paused", "completed", "cancelled"]
SetupSessionMode = Literal["create", "edit"]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class WorkflowSetupSessionStore:
    """CRUD storage for intake workflow setup sessions."""

    _LIVE_STATUSES: tuple[SetupSessionStatus, ...] = ("active", "paused")

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, wal=True)

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS intake_workflow_setup_sessions (
                    session_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    target_workflow_id TEXT,
                    target_workflow_snapshot_json TEXT NOT NULL DEFAULT '{}',
                    draft_upsert_json TEXT NOT NULL DEFAULT '{}',
                    scratchpad_json TEXT NOT NULL DEFAULT '{}',
                    last_proposed_draft_hash TEXT,
                    confirmed_draft_hash TEXT,
                    created_or_updated_workflow_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_setup_sessions_thread
                    ON intake_workflow_setup_sessions(customer_id, thread_id, updated_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_setup_sessions_live_thread
                    ON intake_workflow_setup_sessions(customer_id, thread_id)
                    WHERE status IN ('active', 'paused');
                """
            )

    @staticmethod
    def _hydrate_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": str(row["session_id"] or ""),
            "customer_id": str(row["customer_id"] or ""),
            "thread_id": str(row["thread_id"] or ""),
            "status": str(row["status"] or ""),
            "mode": str(row["mode"] or ""),
            "target_workflow_id": str(row["target_workflow_id"] or ""),
            "target_workflow_snapshot": json.loads(
                str(row["target_workflow_snapshot_json"] or "{}")
            ),
            "draft_upsert": json.loads(str(row["draft_upsert_json"] or "{}")),
            "scratchpad": json.loads(str(row["scratchpad_json"] or "{}")),
            "last_proposed_draft_hash": str(row["last_proposed_draft_hash"] or ""),
            "confirmed_draft_hash": str(row["confirmed_draft_hash"] or ""),
            "created_or_updated_workflow_id": str(row["created_or_updated_workflow_id"] or ""),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
            "completed_at": str(row["completed_at"] or ""),
        }

    def get_session(self, *, session_id: str) -> dict[str, Any] | None:
        safe_session_id = str(session_id or "").strip()
        if not safe_session_id:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM intake_workflow_setup_sessions
                WHERE session_id = ?
                """,
                (safe_session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._hydrate_row(row)

    def get_thread_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        statuses: tuple[SetupSessionStatus, ...] | None = None,
    ) -> dict[str, Any] | None:
        safe_customer = str(customer_id or "").strip()
        safe_thread = str(thread_id or "").strip()
        if not safe_customer or not safe_thread:
            return None
        query = """
            SELECT * FROM intake_workflow_setup_sessions
            WHERE customer_id = ? AND thread_id = ?
        """
        params: list[Any] = [safe_customer, safe_thread]
        safe_statuses = tuple(str(item or "").strip() for item in (statuses or ()))
        if safe_statuses:
            placeholders = ", ".join("?" for _ in safe_statuses)
            query += f" AND status IN ({placeholders})"
            params.extend(safe_statuses)
        query += " ORDER BY updated_at DESC LIMIT 1"
        with self._conn() as conn:
            row = conn.execute(query, params).fetchone()
        if row is None:
            return None
        return self._hydrate_row(row)

    def create_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        mode: SetupSessionMode,
        target_workflow_id: str | None,
        target_workflow_snapshot: dict[str, Any] | None,
        draft_upsert: dict[str, Any] | None,
        scratchpad: dict[str, Any] | None,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_thread = str(thread_id or "").strip()
        safe_mode = str(mode or "").strip().lower()
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_thread:
            raise ValueError("thread_id is required")
        if safe_mode not in {"create", "edit"}:
            raise ValueError("mode must be create|edit")
        session_id = new_short_id("iwsetup")
        now = _utc_now_iso()
        target_id = str(target_workflow_id or "").strip()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO intake_workflow_setup_sessions (
                    session_id,
                    customer_id,
                    thread_id,
                    status,
                    mode,
                    target_workflow_id,
                    target_workflow_snapshot_json,
                    draft_upsert_json,
                    scratchpad_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    safe_customer,
                    safe_thread,
                    "active",
                    safe_mode,
                    target_id or None,
                    _json_dumps(_safe_dict(target_workflow_snapshot)),
                    _json_dumps(_safe_dict(draft_upsert)),
                    _json_dumps(_safe_dict(scratchpad)),
                    now,
                    now,
                ),
            )
            conn.commit()
        created = self.get_session(session_id=session_id)
        if created is None:
            raise RuntimeError("failed to create workflow setup session")
        return created

    def update_session(
        self,
        *,
        session_id: str,
        status: SetupSessionStatus | None = None,
        target_workflow_id: str | None = None,
        target_workflow_snapshot: dict[str, Any] | None = None,
        draft_upsert: dict[str, Any] | None = None,
        scratchpad: dict[str, Any] | None = None,
        last_proposed_draft_hash: str | None = None,
        confirmed_draft_hash: str | None = None,
        created_or_updated_workflow_id: str | None = None,
        completed_at: str | None = None,
    ) -> dict[str, Any]:
        current = self.get_session(session_id=session_id)
        if current is None:
            raise ValueError("workflow setup session not found")
        assignments: list[str] = ["updated_at = ?"]
        params: list[Any] = [_utc_now_iso()]
        if status is not None:
            assignments.append("status = ?")
            params.append(str(status or "").strip())
        if target_workflow_id is not None:
            assignments.append("target_workflow_id = ?")
            params.append(str(target_workflow_id or "").strip() or None)
        if target_workflow_snapshot is not None:
            assignments.append("target_workflow_snapshot_json = ?")
            params.append(_json_dumps(_safe_dict(target_workflow_snapshot)))
        if draft_upsert is not None:
            assignments.append("draft_upsert_json = ?")
            params.append(_json_dumps(_safe_dict(draft_upsert)))
        if scratchpad is not None:
            assignments.append("scratchpad_json = ?")
            params.append(_json_dumps(_safe_dict(scratchpad)))
        if last_proposed_draft_hash is not None:
            assignments.append("last_proposed_draft_hash = ?")
            params.append(str(last_proposed_draft_hash or "").strip() or None)
        if confirmed_draft_hash is not None:
            assignments.append("confirmed_draft_hash = ?")
            params.append(str(confirmed_draft_hash or "").strip() or None)
        if created_or_updated_workflow_id is not None:
            assignments.append("created_or_updated_workflow_id = ?")
            params.append(str(created_or_updated_workflow_id or "").strip() or None)
        if completed_at is not None:
            assignments.append("completed_at = ?")
            params.append(str(completed_at or "").strip() or None)
        params.append(str(session_id or "").strip())
        with self._conn() as conn:
            conn.execute(
                f"""
                UPDATE intake_workflow_setup_sessions
                SET {", ".join(assignments)}
                WHERE session_id = ?
                """,
                params,
            )
            conn.commit()
        updated = self.get_session(session_id=session_id)
        if updated is None:
            raise RuntimeError("workflow setup session disappeared during update")
        return updated

    def delete_thread_live_sessions(self, *, customer_id: str, thread_id: str) -> None:
        safe_customer = str(customer_id or "").strip()
        safe_thread = str(thread_id or "").strip()
        if not safe_customer or not safe_thread:
            return
        with self._conn() as conn:
            conn.execute(
                """
                DELETE FROM intake_workflow_setup_sessions
                WHERE customer_id = ? AND thread_id = ? AND status IN ('active', 'paused')
                """,
                (safe_customer, safe_thread),
            )
            conn.commit()

    def clear_all(self) -> None:
        with self._conn() as conn, suppress(sqlite3.OperationalError):
            conn.execute("DELETE FROM intake_workflow_setup_sessions")
            conn.commit()
