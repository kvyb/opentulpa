"""Durable owner handoff state."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from opentulpa.handoffs.models import (
    HANDOFF_LIFECYCLE_STATUSES,
    HANDOFF_NON_TERMINAL_STATUSES,
    HANDOFF_STATUS_AWAITING_OWNER,
    HANDOFF_STATUS_FAILED_REPLY,
    HANDOFF_STATUS_OWNER_RESPONDED,
    HANDOFF_STATUS_RESOLVED,
    HANDOFF_STATUS_RESOLVED_NO_REPLY,
    HANDOFF_STATUS_RESUMING,
    HandoffLead,
    HandoffMessage,
    HandoffMessages,
    HandoffOpenRequest,
    HandoffRecord,
    HandoffStatus,
    HandoffTrigger,
)
from opentulpa.persistence.sqlite import connect_sqlite


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads_dict(value: Any) -> dict[str, Any]:
    try:
        loaded = json.loads(str(value or "{}"))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_loads_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        loaded = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return loaded if isinstance(loaded, list) else []


class IntakeHandoffStore:
    """SQLite-backed handoff source of truth."""

    def __init__(self, *, db_path: Path) -> None:
        self._db_path = db_path.resolve()

    def conn(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, wal=True)

    def init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS intake_owner_handoffs (
                    handoff_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    workflow_name TEXT NOT NULL,
                    source_channel TEXT NOT NULL,
                    source_provider TEXT NOT NULL,
                    conversation_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_rule_id TEXT NOT NULL,
                    trigger_rule_label TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    owner_prompt TEXT NOT NULL,
                    customer_wait_reply TEXT NOT NULL DEFAULT '',
                    lead_json TEXT NOT NULL DEFAULT '{}',
                    messages_json TEXT NOT NULL DEFAULT '{}',
                    latest_inbound_message_id TEXT NOT NULL DEFAULT '',
                    latest_customer_message_preview TEXT NOT NULL DEFAULT '',
                    owner_feedback TEXT NOT NULL DEFAULT '',
                    failure_reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    responded_at TEXT,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_intake_handoffs_customer_status
                    ON intake_owner_handoffs(customer_id, status, updated_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_intake_handoffs_one_open_conversation
                    ON intake_owner_handoffs(workflow_id, conversation_id)
                    WHERE status IN ('awaiting_owner', 'owner_responded', 'resuming');
                """
            )

    def open_or_update(self, payload: HandoffOpenRequest) -> HandoffRecord:
        handoff_id = payload.handoff_id.strip()
        workflow_id = payload.workflow_id.strip()
        conversation_id = payload.conversation_id.strip()
        assert handoff_id.startswith("hnd_")
        assert workflow_id and conversation_id
        existing = self._get_non_terminal(workflow_id=workflow_id, conversation_id=conversation_id)
        if existing is not None:
            return self._update_open_handoff(handoff_id=existing.handoff_id, payload=payload)
        return self._insert_handoff(payload)

    def _get_non_terminal(self, *, workflow_id: str, conversation_id: str) -> HandoffRecord | None:
        placeholders = ",".join("?" for _ in HANDOFF_NON_TERMINAL_STATUSES)
        query = (
            "SELECT * FROM intake_owner_handoffs "
            f"WHERE workflow_id = ? AND conversation_id = ? AND status IN ({placeholders}) "
            "ORDER BY updated_at DESC LIMIT 1"
        )
        with self.conn() as conn:
            row = conn.execute(query, (workflow_id, conversation_id, *HANDOFF_NON_TERMINAL_STATUSES)).fetchone()
        return self._hydrate(row) if row is not None else None

    def _insert_handoff(self, payload: HandoffOpenRequest) -> HandoffRecord:
        now = _utc_now_iso()
        values = self._row_values(payload, created_at=now, updated_at=now)
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO intake_owner_handoffs (
                    handoff_id, customer_id, workflow_id, workflow_name, source_channel,
                    source_provider, conversation_id, status, trigger_rule_id,
                    trigger_rule_label, trigger_reason, owner_prompt, customer_wait_reply,
                    lead_json, messages_json, latest_inbound_message_id,
                    latest_customer_message_preview, owner_feedback, failure_reason,
                    created_at, updated_at, responded_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?, ?, NULL, NULL)
                """,
                values,
            )
            conn.commit()
        created = self.get(customer_id=payload.customer_id, handoff_id=payload.handoff_id)
        assert created is not None
        return created.model_copy(update={"created": True})

    def _update_open_handoff(self, *, handoff_id: str, payload: HandoffOpenRequest) -> HandoffRecord:
        now = _utc_now_iso()
        with self.conn() as conn:
            conn.execute(
                """
                UPDATE intake_owner_handoffs
                SET workflow_name = ?,
                    source_channel = ?,
                    source_provider = ?,
                    trigger_reason = ?,
                    owner_prompt = ?,
                    customer_wait_reply = ?,
                    lead_json = ?,
                    messages_json = ?,
                    latest_inbound_message_id = ?,
                    latest_customer_message_preview = ?,
                    updated_at = ?
                WHERE handoff_id = ? AND status IN ('awaiting_owner', 'owner_responded', 'resuming')
                """,
                (
                    payload.workflow_name,
                    payload.source_channel,
                    payload.source_provider,
                    payload.trigger.reason,
                    payload.trigger.owner_prompt,
                    payload.trigger.customer_wait_reply,
                    _json_dumps(payload.lead.model_dump()),
                    _json_dumps(payload.messages.to_api_dict()),
                    payload.latest_inbound_message_id,
                    payload.latest_customer_message_preview,
                    now,
                    handoff_id,
                ),
            )
            conn.commit()
        updated = self.get(customer_id=payload.customer_id, handoff_id=handoff_id)
        assert updated is not None
        return updated.model_copy(update={"created": False})

    def _row_values(self, payload: HandoffOpenRequest, *, created_at: str, updated_at: str) -> tuple[Any, ...]:
        return (
            payload.handoff_id,
            payload.customer_id,
            payload.workflow_id,
            payload.workflow_name,
            payload.source_channel,
            payload.source_provider,
            payload.conversation_id,
            HANDOFF_STATUS_AWAITING_OWNER,
            payload.trigger.rule_id,
            payload.trigger.rule_label,
            payload.trigger.reason,
            payload.trigger.owner_prompt,
            payload.trigger.customer_wait_reply,
            _json_dumps(payload.lead.model_dump()),
            _json_dumps(payload.messages.to_api_dict()),
            payload.latest_inbound_message_id,
            payload.latest_customer_message_preview,
            created_at,
            updated_at,
        )

    def get(self, *, customer_id: str, handoff_id: str) -> HandoffRecord | None:
        with self.conn() as conn:
            row = conn.execute(
                "SELECT * FROM intake_owner_handoffs WHERE customer_id = ? AND handoff_id = ?",
                (customer_id, handoff_id),
            ).fetchone()
        return self._hydrate(row) if row is not None else None

    def list_handoffs(
        self,
        *,
        customer_id: str,
        status: str = "",
        limit: int = 50,
    ) -> list[HandoffRecord]:
        conditions = ["customer_id = ?"]
        params: list[Any] = [customer_id]
        if status:
            conditions.append("status = ?")
            params.append(status)
        params.append(max(1, min(int(limit or 50), 100)))
        query = (
            "SELECT * FROM intake_owner_handoffs "
            f"WHERE {' AND '.join(conditions)} ORDER BY updated_at DESC LIMIT ?"
        )
        with self.conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._hydrate(row) for row in rows]

    def accept_owner_response(self, *, customer_id: str, handoff_id: str, owner_feedback: str) -> HandoffRecord | None:
        now = _utc_now_iso()
        with self.conn() as conn:
            result = conn.execute(
                """
                UPDATE intake_owner_handoffs
                SET status = ?, owner_feedback = ?, responded_at = ?, updated_at = ?
                WHERE customer_id = ? AND handoff_id = ? AND status = ?
                """,
                (
                    HANDOFF_STATUS_OWNER_RESPONDED,
                    owner_feedback,
                    now,
                    now,
                    customer_id,
                    handoff_id,
                    HANDOFF_STATUS_AWAITING_OWNER,
                ),
            )
            conn.commit()
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            return None
        return self.get(customer_id=customer_id, handoff_id=handoff_id)

    def mark_resuming(self, *, customer_id: str, handoff_id: str) -> HandoffRecord | None:
        return self._transition(
            customer_id=customer_id,
            handoff_id=handoff_id,
            status=HANDOFF_STATUS_RESUMING,
            allowed_statuses=(HANDOFF_STATUS_OWNER_RESPONDED,),
        )

    def mark_resolved(self, *, customer_id: str, handoff_id: str, no_reply: bool = False) -> HandoffRecord | None:
        status = HANDOFF_STATUS_RESOLVED_NO_REPLY if no_reply else HANDOFF_STATUS_RESOLVED
        return self._transition(
            customer_id=customer_id,
            handoff_id=handoff_id,
            status=status,
            resolved=True,
            allowed_statuses=(HANDOFF_STATUS_RESUMING,),
        )

    def mark_failed_reply(self, *, customer_id: str, handoff_id: str, failure_reason: str) -> HandoffRecord | None:
        return self._transition(
            customer_id=customer_id,
            handoff_id=handoff_id,
            status=HANDOFF_STATUS_FAILED_REPLY,
            failure_reason=failure_reason,
            resolved=True,
            allowed_statuses=(HANDOFF_STATUS_OWNER_RESPONDED, HANDOFF_STATUS_RESUMING),
        )

    def _transition(
        self,
        *,
        customer_id: str,
        handoff_id: str,
        status: str,
        failure_reason: str = "",
        resolved: bool = False,
        allowed_statuses: tuple[str, ...] = (),
    ) -> HandoffRecord | None:
        now = _utc_now_iso()
        resolved_at = now if resolved else None
        status_filter = ""
        params: list[Any] = [status, failure_reason, now, resolved_at, customer_id, handoff_id]
        if allowed_statuses:
            placeholders = ",".join("?" for _ in allowed_statuses)
            status_filter = f" AND status IN ({placeholders})"
            params.extend(allowed_statuses)
        with self.conn() as conn:
            result = conn.execute(
                f"""
                UPDATE intake_owner_handoffs
                SET status = ?, failure_reason = ?, updated_at = ?, resolved_at = COALESCE(?, resolved_at)
                WHERE customer_id = ? AND handoff_id = ?
                {status_filter}
                """,
                params,
            )
            conn.commit()
        if int(getattr(result, "rowcount", 0) or 0) != 1:
            return None
        return self.get(customer_id=customer_id, handoff_id=handoff_id)

    def _hydrate(self, row: sqlite3.Row) -> HandoffRecord:
        messages = _json_loads_dict(row["messages_json"])
        status = str(row["status"])
        if status not in HANDOFF_LIFECYCLE_STATUSES:
            raise ValueError(f"unknown handoff status: {status}")
        return HandoffRecord(
            handoff_id=str(row["handoff_id"]),
            status=cast(HandoffStatus, status),
            customer_id=str(row["customer_id"]),
            workflow_id=str(row["workflow_id"]),
            workflow_name=str(row["workflow_name"]),
            source_channel=str(row["source_channel"]),
            source_provider=str(row["source_provider"]),
            conversation_id=str(row["conversation_id"]),
            lead=HandoffLead(**_json_loads_dict(row["lead_json"])),
            trigger=HandoffTrigger(
                rule_id=str(row["trigger_rule_id"]),
                rule_label=str(row["trigger_rule_label"]),
                reason=str(row["trigger_reason"]),
                owner_prompt=str(row["owner_prompt"]),
                customer_wait_reply=str(row["customer_wait_reply"]),
            ),
            messages=HandoffMessages(
                latest=[HandoffMessage.model_validate(item) for item in _json_loads_list(messages.get("latest"))],
                previous=[
                    HandoffMessage.model_validate(item)
                    for item in _json_loads_list(messages.get("previous"))
                ],
            ),
            latest_inbound_message_id=str(row["latest_inbound_message_id"]),
            latest_customer_message_preview=str(row["latest_customer_message_preview"]),
            owner_feedback=str(row["owner_feedback"]),
            failure_reason=str(row["failure_reason"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            responded_at=str(row["responded_at"] or ""),
            resolved_at=str(row["resolved_at"] or ""),
        )
