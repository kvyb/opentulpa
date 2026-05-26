"""SQLite persistence for intake workflows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opentulpa.persistence.sqlite import connect_sqlite


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        with suppress(ValueError):
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
    return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


class IntakeWorkflowStore:
    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path.resolve()

    def conn(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, wal=True)

    def init_db(
        self,
        *,
        normalize_sink_config: Callable[..., dict[str, Any]],
    ) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS intake_workflows (
                    workflow_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_config_json TEXT NOT NULL,
                    intent_description TEXT NOT NULL,
                    required_fields_json TEXT NOT NULL,
                    field_guidance_json TEXT NOT NULL,
                    assistant_instructions TEXT NOT NULL DEFAULT '',
                    business_facts_json TEXT NOT NULL DEFAULT '{}',
                    knowledge_file_ids_json TEXT NOT NULL DEFAULT '[]',
                    sink_type TEXT NOT NULL,
                    sink_config_json TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    notify_user INTEGER NOT NULL,
                    enabled INTEGER NOT NULL,
                    routine_id TEXT NOT NULL,
                    reply_mode TEXT NOT NULL DEFAULT 'auto',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_intake_workflows_customer
                    ON intake_workflows(customer_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS intake_bookings (
                    booking_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    extracted_fields_json TEXT NOT NULL,
                    sink_write_status TEXT NOT NULL,
                    sink_record_ref_json TEXT NOT NULL,
                    conversation_summary TEXT NOT NULL,
                    last_customer_message_at TEXT,
                    opened_at TEXT NOT NULL,
                    completed_at TEXT,
                    edit_window_until TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_intake_bookings_scope
                    ON intake_bookings(workflow_id, conversation_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS intake_conversation_cursors (
                    workflow_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    last_seen_inbound_message_id TEXT,
                    last_seen_inbound_message_time TEXT,
                    last_seen_conversation_updated_time TEXT,
                    last_seen_latest_outbound_message_id TEXT,
                    last_agent_action_at TEXT,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workflow_id, conversation_id)
                );

                CREATE TABLE IF NOT EXISTS intake_pending_runs (
                    workflow_id TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    owner_chat_id TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL,
                    running_generation INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    last_inbound_message_id TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (workflow_id, conversation_id)
                );
                CREATE INDEX IF NOT EXISTS idx_intake_pending_runs_due
                    ON intake_pending_runs(status, due_at);
                """
            )
            self._ensure_cursor_columns(conn)
            self._ensure_workflow_columns(conn)
            self._migrate_legacy_sink_configs(conn, normalize_sink_config=normalize_sink_config)

    @staticmethod
    def _ensure_cursor_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(intake_conversation_cursors)").fetchall()
        existing = {str(row["name"] or "") for row in rows}
        required_columns = {
            "last_seen_conversation_updated_time": "TEXT",
            "last_seen_latest_outbound_message_id": "TEXT",
            "last_agent_action_at": "TEXT",
        }
        for column, column_type in required_columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE intake_conversation_cursors ADD COLUMN {column} {column_type}")
        conn.commit()

    @staticmethod
    def _ensure_workflow_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(intake_workflows)").fetchall()
        existing = {str(row["name"] or "") for row in rows}
        required_columns = {
            "assistant_instructions": "TEXT NOT NULL DEFAULT ''",
            "business_facts_json": "TEXT NOT NULL DEFAULT '{}'",
            "knowledge_file_ids_json": "TEXT NOT NULL DEFAULT '[]'",
            "reply_mode": "TEXT NOT NULL DEFAULT 'auto'",
        }
        for column, column_type in required_columns.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE intake_workflows ADD COLUMN {column} {column_type}")
        conn.commit()

    @staticmethod
    def _migrate_legacy_sink_configs(
        conn: sqlite3.Connection,
        *,
        normalize_sink_config: Callable[..., dict[str, Any]],
    ) -> None:
        rows = conn.execute(
            "SELECT workflow_id, customer_id, sink_type, sink_config_json FROM intake_workflows"
        ).fetchall()
        for row in rows:
            sink_type = str(row["sink_type"] or "").strip().lower()
            original = json.loads(row["sink_config_json"] or "{}")
            normalized = normalize_sink_config(
                sink_type=sink_type,
                sink_config=original,
                workflow_id=str(row["workflow_id"] or "").strip(),
                customer_id=str(row["customer_id"] or "").strip(),
                validate_target=False,
            )
            if normalized != original:
                conn.execute(
                    "UPDATE intake_workflows SET sink_config_json = ? WHERE workflow_id = ?",
                    (_json_dumps(normalized), str(row["workflow_id"] or "").strip()),
                )
        conn.commit()

    def upsert_workflow_record(
        self,
        *,
        workflow: dict[str, Any],
        created_at: str,
        updated_at: str,
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO intake_workflows (
                    workflow_id, customer_id, name, channel, provider, source_config_json,
                    intent_description, required_fields_json, field_guidance_json,
                    assistant_instructions, business_facts_json, knowledge_file_ids_json, sink_type,
                    sink_config_json, schedule, notify_user, enabled, routine_id,
                    reply_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    name=excluded.name,
                    channel=excluded.channel,
                    provider=excluded.provider,
                    source_config_json=excluded.source_config_json,
                    intent_description=excluded.intent_description,
                    required_fields_json=excluded.required_fields_json,
                    field_guidance_json=excluded.field_guidance_json,
                    assistant_instructions=excluded.assistant_instructions,
                    business_facts_json=excluded.business_facts_json,
                    knowledge_file_ids_json=excluded.knowledge_file_ids_json,
                    sink_type=excluded.sink_type,
                    sink_config_json=excluded.sink_config_json,
                    schedule=excluded.schedule,
                    notify_user=excluded.notify_user,
                    enabled=excluded.enabled,
                    routine_id=excluded.routine_id,
                    reply_mode=excluded.reply_mode,
                    updated_at=excluded.updated_at
                """,
                (
                    workflow["workflow_id"],
                    workflow["customer_id"],
                    workflow["name"],
                    workflow["channel"],
                    workflow["provider"],
                    _json_dumps(workflow["source_config"]),
                    workflow["intent_description"],
                    _json_dumps(workflow["required_fields"]),
                    _json_dumps(workflow["field_guidance"]),
                    workflow["assistant_instructions"],
                    _json_dumps(workflow["business_facts"]),
                    _json_dumps(workflow["knowledge_file_ids"]),
                    workflow["sink_type"],
                    _json_dumps(workflow["sink_config"]),
                    workflow["schedule"],
                    1 if workflow["notify_user"] else 0,
                    1 if workflow["enabled"] else 0,
                    workflow["routine_id"],
                    workflow["reply_mode"],
                    created_at,
                    updated_at,
                ),
            )
            conn.commit()

    def list_workflows(self, *, customer_id: str, include_disabled: bool = False) -> list[dict[str, Any]]:
        safe_customer = str(customer_id or "").strip()
        if not safe_customer:
            return []
        query = "SELECT * FROM intake_workflows WHERE customer_id = ?"
        params: list[Any] = [safe_customer]
        if not include_disabled:
            query += " AND enabled = 1"
        query += " ORDER BY updated_at DESC"
        with self.conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._hydrate_workflow_row(row) for row in rows]

    def list_customer_summaries(self) -> list[dict[str, Any]]:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT customer_id, COUNT(*) AS workflow_count, MAX(updated_at) AS last_workflow_at
                FROM intake_workflows
                GROUP BY customer_id
                ORDER BY last_workflow_at DESC
                """
            ).fetchall()
        return [
            {
                "customer_id": str(row["customer_id"]),
                "workflow_count": int(row["workflow_count"] or 0),
                "last_workflow_at": str(row["last_workflow_at"] or ""),
            }
            for row in rows
        ]

    def get_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any] | None:
        safe_customer = str(customer_id or "").strip()
        safe_workflow = str(workflow_id or "").strip()
        if not safe_customer or not safe_workflow:
            return None
        with self.conn() as conn:
            row = conn.execute(
                "SELECT * FROM intake_workflows WHERE workflow_id = ? AND customer_id = ?",
                (safe_workflow, safe_customer),
            ).fetchone()
        return self._hydrate_workflow_row(row) if row is not None else None

    def delete_workflow_records(self, *, workflow_id: str) -> None:
        safe_workflow = str(workflow_id or "").strip()
        if not safe_workflow:
            return
        with self.conn() as conn:
            conn.execute("DELETE FROM intake_workflows WHERE workflow_id = ?", (safe_workflow,))
            conn.execute("DELETE FROM intake_bookings WHERE workflow_id = ?", (safe_workflow,))
            conn.execute("DELETE FROM intake_conversation_cursors WHERE workflow_id = ?", (safe_workflow,))
            conn.execute("DELETE FROM intake_pending_runs WHERE workflow_id = ?", (safe_workflow,))
            conn.commit()

    def list_bookings(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        safe_customer = str(customer_id or "").strip()
        safe_workflow = str(workflow_id or "").strip()
        safe_conversation = str(conversation_id or "").strip()
        if not safe_customer or not safe_workflow:
            return []
        query = "SELECT * FROM intake_bookings WHERE customer_id = ? AND workflow_id = ?"
        params: list[Any] = [safe_customer, safe_workflow]
        if safe_conversation:
            query += " AND conversation_id = ?"
            params.append(safe_conversation)
        query += " ORDER BY updated_at DESC"
        with self.conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._hydrate_booking_row(row) for row in rows]

    def get_cursor(self, *, workflow_id: str, conversation_id: str) -> dict[str, str]:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT
                    last_seen_inbound_message_id,
                    last_seen_inbound_message_time,
                    last_seen_conversation_updated_time,
                    last_seen_latest_outbound_message_id,
                    last_agent_action_at
                FROM intake_conversation_cursors
                WHERE workflow_id = ? AND conversation_id = ?
                """,
                (workflow_id, conversation_id),
            ).fetchone()
        if row is None:
            return {}
        return {
            "last_seen_inbound_message_id": str(row["last_seen_inbound_message_id"] or ""),
            "last_seen_inbound_message_time": str(row["last_seen_inbound_message_time"] or ""),
            "last_seen_conversation_updated_time": str(row["last_seen_conversation_updated_time"] or ""),
            "last_seen_latest_outbound_message_id": str(row["last_seen_latest_outbound_message_id"] or ""),
            "last_agent_action_at": str(row["last_agent_action_at"] or ""),
        }

    def set_cursor(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        latest_inbound_message_id: str,
        latest_inbound_message_time: str,
        conversation_updated_time: str = "",
        latest_outbound_message_id: str = "",
        agent_action_at: str = "",
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO intake_conversation_cursors (
                    workflow_id, conversation_id, last_seen_inbound_message_id,
                    last_seen_inbound_message_time, last_seen_conversation_updated_time,
                    last_seen_latest_outbound_message_id, last_agent_action_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id, conversation_id) DO UPDATE SET
                    last_seen_inbound_message_id=excluded.last_seen_inbound_message_id,
                    last_seen_inbound_message_time=excluded.last_seen_inbound_message_time,
                    last_seen_conversation_updated_time=excluded.last_seen_conversation_updated_time,
                    last_seen_latest_outbound_message_id=excluded.last_seen_latest_outbound_message_id,
                    last_agent_action_at=excluded.last_agent_action_at,
                    updated_at=excluded.updated_at
                """,
                (
                    workflow_id,
                    conversation_id,
                    latest_inbound_message_id,
                    latest_inbound_message_time,
                    conversation_updated_time,
                    latest_outbound_message_id,
                    agent_action_at,
                    _utc_now_iso(),
                ),
            )
            conn.commit()

    def upsert_booking(self, booking: dict[str, Any]) -> None:
        now = _utc_now_iso()
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO intake_bookings (
                    booking_id, workflow_id, customer_id, conversation_id, status,
                    extracted_fields_json, sink_write_status, sink_record_ref_json,
                    conversation_summary, last_customer_message_at, opened_at,
                    completed_at, edit_window_until, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(booking_id) DO UPDATE SET
                    status=excluded.status,
                    extracted_fields_json=excluded.extracted_fields_json,
                    sink_write_status=excluded.sink_write_status,
                    sink_record_ref_json=excluded.sink_record_ref_json,
                    conversation_summary=excluded.conversation_summary,
                    last_customer_message_at=excluded.last_customer_message_at,
                    opened_at=excluded.opened_at,
                    completed_at=excluded.completed_at,
                    edit_window_until=excluded.edit_window_until,
                    updated_at=excluded.updated_at
                """,
                (
                    str(booking["booking_id"]),
                    str(booking["workflow_id"]),
                    str(booking["customer_id"]),
                    str(booking["conversation_id"]),
                    str(booking.get("status", "active") or "active"),
                    _json_dumps(_safe_dict(booking.get("extracted_fields"))),
                    str(booking.get("sink_write_status", "pending") or "pending"),
                    _json_dumps(_safe_dict(booking.get("sink_record_ref"))),
                    str(booking.get("conversation_summary", "") or ""),
                    str(booking.get("last_customer_message_at", "") or ""),
                    str(booking.get("opened_at", "") or now),
                    str(booking.get("completed_at", "") or ""),
                    str(booking.get("edit_window_until", "") or ""),
                    str(booking.get("created_at", "") or now),
                    str(booking.get("updated_at", "") or now),
                ),
            )
            conn.commit()

    def recover_interrupted_pending_runs(self) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                UPDATE intake_pending_runs
                SET status = 'pending',
                    running_generation = 0,
                    due_at = ?,
                    updated_at = ?
                WHERE status = 'running'
                """,
                (_utc_now_iso(), _utc_now_iso()),
            )
            conn.commit()

    def queue_pending_run(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        event_type: str,
        due_at: str,
        owner_chat_id: str = "",
        last_inbound_message_id: str = "",
    ) -> dict[str, Any]:
        safe_workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        safe_customer_id = str(workflow.get("customer_id", "") or "").strip()
        safe_conversation_id = str(conversation_id or "").strip()
        if not safe_workflow_id or not safe_customer_id or not safe_conversation_id:
            return {"ok": False, "queued": False, "summary": "pending run requires workflow and conversation ids"}
        now = _utc_now_iso()
        safe_owner_chat_id = str(owner_chat_id or "").strip()
        safe_last_inbound_id = str(last_inbound_message_id or "").strip()
        safe_event_type = str(event_type or "").strip()
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT generation, status, owner_chat_id, created_at
                FROM intake_pending_runs
                WHERE workflow_id = ? AND conversation_id = ?
                """,
                (safe_workflow_id, safe_conversation_id),
            ).fetchone()
            generation = int(row["generation"] or 0) + 1 if row is not None else 1
            status = str(row["status"] or "").strip() if row is not None else ""
            next_status = "running" if status == "running" else "pending"
            created_at = str(row["created_at"] or now) if row is not None else now
            if not safe_owner_chat_id and row is not None:
                safe_owner_chat_id = str(row["owner_chat_id"] or "").strip()
            conn.execute(
                """
                INSERT INTO intake_pending_runs (
                    workflow_id, conversation_id, customer_id, event_type, owner_chat_id,
                    generation, running_generation, status, due_at,
                    last_inbound_message_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id, conversation_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    event_type=excluded.event_type,
                    owner_chat_id=excluded.owner_chat_id,
                    generation=excluded.generation,
                    status=excluded.status,
                    due_at=excluded.due_at,
                    last_inbound_message_id=excluded.last_inbound_message_id,
                    updated_at=excluded.updated_at
                """,
                (
                    safe_workflow_id,
                    safe_conversation_id,
                    safe_customer_id,
                    safe_event_type,
                    safe_owner_chat_id,
                    generation,
                    next_status,
                    due_at,
                    safe_last_inbound_id,
                    created_at,
                    now,
                ),
            )
            conn.commit()
        return {
            "ok": True,
            "queued": True,
            "workflow_id": safe_workflow_id,
            "conversation_id": safe_conversation_id,
            "generation": generation,
            "due_at": due_at,
        }

    def claim_due_pending_runs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit or 10), 50))
        now = _utc_now_iso()
        claimed: list[dict[str, Any]] = []
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM intake_pending_runs
                WHERE status = 'pending' AND due_at <= ?
                ORDER BY due_at ASC, updated_at ASC
                LIMIT ?
                """,
                (now, safe_limit),
            ).fetchall()
            for row in rows:
                generation = int(row["generation"] or 0)
                result = conn.execute(
                    """
                    UPDATE intake_pending_runs
                    SET status = 'running',
                        running_generation = ?,
                        updated_at = ?
                    WHERE workflow_id = ?
                      AND conversation_id = ?
                      AND generation = ?
                      AND status = 'pending'
                    """,
                    (generation, now, str(row["workflow_id"]), str(row["conversation_id"]), generation),
                )
                if int(getattr(result, "rowcount", 0) or 0) == 1:
                    claimed.append(dict(row))
            conn.commit()
        return claimed

    def pending_run_is_still_running(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        generation: int,
    ) -> bool:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT status, running_generation
                FROM intake_pending_runs
                WHERE workflow_id = ? AND conversation_id = ?
                """,
                (workflow_id, conversation_id),
            ).fetchone()
        if row is None:
            return False
        return str(row["status"] or "").strip() == "running" and int(row["running_generation"] or 0) == int(
            generation or 0
        )

    def finish_pending_run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        generation: int,
        stale_requeue_seconds: float,
    ) -> None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT generation, due_at FROM intake_pending_runs WHERE workflow_id = ? AND conversation_id = ?",
                (workflow_id, conversation_id),
            ).fetchone()
            if row is None:
                return
            current_generation = int(row["generation"] or 0)
            if current_generation > int(generation or 0):
                due_at = str(row["due_at"] or "").strip()
                parsed_due = _parse_datetime(due_at)
                min_due = _utc_now() + timedelta(seconds=max(0.0, float(stale_requeue_seconds)))
                next_due_at = due_at if parsed_due is not None and parsed_due > _utc_now() else min_due.isoformat()
                conn.execute(
                    """
                    UPDATE intake_pending_runs
                    SET status = 'pending',
                        running_generation = 0,
                        due_at = ?,
                        updated_at = ?
                    WHERE workflow_id = ? AND conversation_id = ?
                    """,
                    (next_due_at, _utc_now_iso(), workflow_id, conversation_id),
                )
            else:
                conn.execute(
                    "DELETE FROM intake_pending_runs WHERE workflow_id = ? AND conversation_id = ?",
                    (workflow_id, conversation_id),
                )
            conn.commit()

    @staticmethod
    def _hydrate_workflow_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "workflow_id": str(row["workflow_id"]),
            "customer_id": str(row["customer_id"]),
            "name": str(row["name"]),
            "channel": str(row["channel"]),
            "provider": str(row["provider"]),
            "source_config": json.loads(row["source_config_json"] or "{}"),
            "intent_description": str(row["intent_description"]),
            "required_fields": json.loads(row["required_fields_json"] or "[]"),
            "field_guidance": json.loads(row["field_guidance_json"] or "{}"),
            "assistant_instructions": str(row["assistant_instructions"] or ""),
            "business_facts": json.loads(row["business_facts_json"] or "{}"),
            "knowledge_file_ids": json.loads(row["knowledge_file_ids_json"] or "[]"),
            "sink_type": str(row["sink_type"]),
            "sink_config": json.loads(row["sink_config_json"] or "{}"),
            "schedule": str(row["schedule"]),
            "notify_user": bool(row["notify_user"]),
            "enabled": bool(row["enabled"]),
            "routine_id": str(row["routine_id"]),
            "reply_mode": "auto",
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _hydrate_booking_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "booking_id": str(row["booking_id"]),
            "workflow_id": str(row["workflow_id"]),
            "customer_id": str(row["customer_id"]),
            "conversation_id": str(row["conversation_id"]),
            "status": str(row["status"]),
            "extracted_fields": json.loads(row["extracted_fields_json"] or "{}"),
            "sink_write_status": str(row["sink_write_status"]),
            "sink_record_ref": json.loads(row["sink_record_ref_json"] or "{}"),
            "conversation_summary": str(row["conversation_summary"] or ""),
            "last_customer_message_at": str(row["last_customer_message_at"] or ""),
            "opened_at": str(row["opened_at"] or ""),
            "completed_at": str(row["completed_at"] or ""),
            "edit_window_until": str(row["edit_window_until"] or ""),
            "created_at": str(row["created_at"] or ""),
            "updated_at": str(row["updated_at"] or ""),
        }
