"""Persistent intake workflow storage and wake-time execution."""

from __future__ import annotations

import csv
import json
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN
from opentulpa.scheduler.models import Routine
from opentulpa.skills.service import build_skill_markdown

_ALLOWED_CHANNELS = {"instagram_dm"}
_ALLOWED_PROVIDERS = {"composio"}
_ALLOWED_SINK_TYPES = {"google_sheets_composio", "local_csv", "generic_composio_write"}
_DEFAULT_SCHEDULE = "*/5 * * * *"
_DEFAULT_EDIT_WINDOW = timedelta(hours=2)
_MAX_DECISION_RECOVERY_ATTEMPTS = 2


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


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_optional_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _clean_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_field in value.items():
        key = str(raw_key or "").strip()
        field = str(raw_field or "").strip()
        if key and field:
            out[key] = field
    return out


def _normalize_toolkit_slug(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("-", "")


def _normalize_composio_tool_slug(value: Any) -> str:
    safe = str(value or "").strip()
    if not safe:
        return ""
    if "_" not in safe:
        return safe
    prefix, remainder = safe.split("_", 1)
    if prefix and prefix == prefix.lower():
        upper_prefix = prefix.upper()
        if remainder.upper().startswith(f"{upper_prefix}_"):
            return remainder
    return safe


def _infer_toolkit_from_tool_slug(value: Any) -> str:
    safe = _normalize_composio_tool_slug(value)
    if not safe:
        return ""
    if "_" not in safe:
        return _normalize_toolkit_slug(safe)
    prefix, _ = safe.split("_", 1)
    return _normalize_toolkit_slug(prefix)


def _infer_operation_hint_from_tool_slug(value: Any) -> str:
    safe = _normalize_composio_tool_slug(value)
    if not safe:
        return ""
    if "_" in safe:
        _, remainder = safe.split("_", 1)
    else:
        remainder = safe
    return str(remainder).replace("_", " ").strip().lower()


def _unique_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


class IntakeWorkflowService:
    """Stores intake workflows and runs them on scheduled wake events."""

    def __init__(
        self,
        *,
        db_path: Path,
        project_root: Path,
        scheduler: Any | None = None,
        skill_store: Any | None = None,
        composio: Any | None = None,
        get_agent_runtime: Any | None = None,
    ) -> None:
        self._db_path = db_path.resolve()
        self._project_root = project_root.resolve()
        self._scheduler = scheduler
        self._skill_store = skill_store
        self._composio = composio
        self._get_agent_runtime = get_agent_runtime
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
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
                    sink_type TEXT NOT NULL,
                    sink_config_json TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    notify_user INTEGER NOT NULL,
                    enabled INTEGER NOT NULL,
                    routine_id TEXT NOT NULL,
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
                """
            )
            self._ensure_cursor_columns(conn)
            self._migrate_legacy_sink_configs(conn)

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
            if column in existing:
                continue
            conn.execute(
                f"ALTER TABLE intake_conversation_cursors ADD COLUMN {column} {column_type}"
            )
        conn.commit()

    def _migrate_legacy_sink_configs(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT workflow_id, customer_id, sink_type, sink_config_json FROM intake_workflows"
        ).fetchall()
        for row in rows:
            sink_type = str(row["sink_type"] or "").strip().lower()
            original = json.loads(row["sink_config_json"] or "{}")
            normalized = self._normalize_sink_config(
                sink_type=sink_type,
                sink_config=original,
                workflow_id=str(row["workflow_id"] or "").strip(),
                customer_id=str(row["customer_id"] or "").strip(),
                validate_target=False,
            )
            if normalized == original:
                continue
            conn.execute(
                "UPDATE intake_workflows SET sink_config_json = ? WHERE workflow_id = ?",
                (_json_dumps(normalized), str(row["workflow_id"] or "").strip()),
            )
        conn.commit()

    def _normalize_workflow_payload(
        self,
        *,
        workflow_id: str | None,
        customer_id: str,
        name: str,
        channel: str,
        provider: str,
        source_config: dict[str, Any] | None,
        intent_description: str,
        required_fields: list[str],
        field_guidance: dict[str, Any] | None,
        sink_type: str,
        sink_config: dict[str, Any] | None,
        schedule: str,
        notify_user: bool,
        enabled: bool,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_name = str(name or "").strip()
        safe_channel = str(channel or "instagram_dm").strip().lower() or "instagram_dm"
        safe_provider = str(provider or "composio").strip().lower() or "composio"
        safe_intent = str(intent_description or "").strip()
        safe_schedule = str(schedule or _DEFAULT_SCHEDULE).strip() or _DEFAULT_SCHEDULE
        safe_required_fields = _unique_string_list(required_fields)
        safe_source_config = _safe_dict(source_config)
        safe_field_guidance = _safe_dict(field_guidance)
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_name:
            raise ValueError("name is required")
        if safe_channel not in _ALLOWED_CHANNELS:
            raise ValueError("channel must be instagram_dm")
        if safe_provider not in _ALLOWED_PROVIDERS:
            raise ValueError("provider must be composio")
        if not safe_intent:
            raise ValueError("intent_description is required")
        if not safe_required_fields:
            raise ValueError("required_fields must contain at least one field")
        safe_sink_type = str(sink_type or "").strip().lower()
        if safe_sink_type not in _ALLOWED_SINK_TYPES:
            raise ValueError(
                "sink_type must be one of google_sheets_composio|local_csv|generic_composio_write"
            )
        existing_record = existing or {}
        safe_workflow_id = (
            _normalize_optional_id(workflow_id)
            or _normalize_optional_id(existing_record.get("workflow_id", ""))
            or new_short_id("iwf")
        )
        safe_sink_config = self._normalize_sink_config(
            sink_type=safe_sink_type,
            sink_config=sink_config,
            workflow_id=safe_workflow_id,
            customer_id=safe_customer,
        )
        safe_routine_id = (
            str(existing_record.get("routine_id", "")).strip()
            if existing is not None
            else ""
        ) or new_short_id("rtn")
        return {
            "workflow_id": safe_workflow_id,
            "customer_id": safe_customer,
            "name": safe_name,
            "channel": safe_channel,
            "provider": safe_provider,
            "source_config": safe_source_config,
            "intent_description": safe_intent,
            "required_fields": safe_required_fields,
            "field_guidance": safe_field_guidance,
            "sink_type": safe_sink_type,
            "sink_config": safe_sink_config,
            "schedule": safe_schedule,
            "notify_user": bool(notify_user),
            "enabled": bool(enabled),
            "routine_id": safe_routine_id,
        }

    def _normalize_sink_config(
        self,
        *,
        sink_type: str,
        sink_config: dict[str, Any] | None,
        workflow_id: str,
        customer_id: str,
        validate_target: bool = True,
    ) -> dict[str, Any]:
        safe_config = _safe_dict(sink_config)
        if sink_type == "local_csv":
            requested_path = str(safe_config.get("file_path", "") or "").strip()
            file_path = requested_path or f"tulpa_stuff/intake_{workflow_id or 'workflow'}.csv"
            return {"file_path": file_path}

        field_mapping = _clean_mapping(safe_config.get("field_mapping"))
        if not field_mapping:
            raise ValueError("sink_config.field_mapping is required for composio sink types")
        static_arguments = _safe_dict(safe_config.get("static_arguments"))
        connected_account_id = str(safe_config.get("connected_account_id", "") or "").strip()
        legacy_tool_slug = str(safe_config.get("tool_slug", "") or "").strip()
        toolkit = _normalize_toolkit_slug(safe_config.get("toolkit"))
        operation_hint = str(safe_config.get("operation_hint", "") or "").strip().lower()
        if sink_type == "google_sheets_composio":
            toolkit = toolkit or _infer_toolkit_from_tool_slug(legacy_tool_slug) or "googlesheets"
            operation_hint = operation_hint or _infer_operation_hint_from_tool_slug(legacy_tool_slug) or "upsert rows"
        else:
            toolkit = toolkit or _infer_toolkit_from_tool_slug(legacy_tool_slug)
            operation_hint = operation_hint or _infer_operation_hint_from_tool_slug(legacy_tool_slug)
            if not toolkit:
                raise ValueError("sink_config.toolkit is required for generic_composio_write")
            if not operation_hint:
                raise ValueError(
                    "sink_config.operation_hint is required for generic_composio_write"
                )
        normalized = {
            "toolkit": toolkit,
            "operation_hint": operation_hint,
            "field_mapping": field_mapping,
            "static_arguments": static_arguments,
        }
        if connected_account_id:
            normalized["connected_account_id"] = connected_account_id
        if validate_target:
            self._validate_sink_target(
                customer_id=customer_id,
                sink_type=sink_type,
                sink_config=normalized,
            )
        return normalized

    def _validate_sink_target(
        self,
        *,
        customer_id: str,
        sink_type: str,
        sink_config: dict[str, Any],
    ) -> None:
        del customer_id
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            return
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        if not toolkit:
            raise ValueError("sink_config.toolkit is required for composio sink types")
        operation_hint = str(sink_config.get("operation_hint", "") or "").strip().lower()
        try:
            resolved_slug = self._resolve_composio_sink_tool_slug(
                sink_type=sink_type,
                sink_config=sink_config,
            )
        except Exception as exc:
            raise ValueError(f"unable to resolve sink tool from toolkit={toolkit}: {exc}") from exc
        if not resolved_slug:
            raise ValueError(
                f"unable to resolve sink tool from toolkit={toolkit}"
            )
        if sink_type == "generic_composio_write" and not operation_hint:
            raise ValueError("sink_config.operation_hint is required for generic_composio_write")

    def _hydrate_workflow_row(self, row: sqlite3.Row) -> dict[str, Any]:
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
            "sink_type": str(row["sink_type"]),
            "sink_config": json.loads(row["sink_config_json"] or "{}"),
            "schedule": str(row["schedule"]),
            "notify_user": bool(row["notify_user"]),
            "enabled": bool(row["enabled"]),
            "routine_id": str(row["routine_id"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def _hydrate_booking_row(self, row: sqlite3.Row) -> dict[str, Any]:
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

    def upsert_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str | None = None,
        name: str,
        channel: str = "instagram_dm",
        provider: str = "composio",
        source_config: dict[str, Any] | None = None,
        intent_description: str,
        required_fields: list[str],
        field_guidance: dict[str, Any] | None = None,
        sink_type: str,
        sink_config: dict[str, Any] | None = None,
        schedule: str = _DEFAULT_SCHEDULE,
        notify_user: bool = True,
        enabled: bool = True,
    ) -> dict[str, Any]:
        existing = None
        safe_workflow_id = _normalize_optional_id(workflow_id)
        if safe_workflow_id:
            existing = self.get_workflow(customer_id=customer_id, workflow_id=safe_workflow_id)
        workflow = self._normalize_workflow_payload(
            workflow_id=safe_workflow_id or None,
            customer_id=customer_id,
            name=name,
            channel=channel,
            provider=provider,
            source_config=source_config,
            intent_description=intent_description,
            required_fields=required_fields,
            field_guidance=field_guidance,
            sink_type=sink_type,
            sink_config=sink_config,
            schedule=schedule,
            notify_user=notify_user,
            enabled=enabled,
            existing=existing,
        )
        now = _utc_now_iso()
        created_at = str(existing.get("created_at", "")).strip() if existing else ""
        if not created_at:
            created_at = now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO intake_workflows (
                    workflow_id, customer_id, name, channel, provider, source_config_json,
                    intent_description, required_fields_json, field_guidance_json,
                    sink_type, sink_config_json, schedule, notify_user, enabled, routine_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    name=excluded.name,
                    channel=excluded.channel,
                    provider=excluded.provider,
                    source_config_json=excluded.source_config_json,
                    intent_description=excluded.intent_description,
                    required_fields_json=excluded.required_fields_json,
                    field_guidance_json=excluded.field_guidance_json,
                    sink_type=excluded.sink_type,
                    sink_config_json=excluded.sink_config_json,
                    schedule=excluded.schedule,
                    notify_user=excluded.notify_user,
                    enabled=excluded.enabled,
                    routine_id=excluded.routine_id,
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
                    workflow["sink_type"],
                    _json_dumps(workflow["sink_config"]),
                    workflow["schedule"],
                    1 if workflow["notify_user"] else 0,
                    1 if workflow["enabled"] else 0,
                    workflow["routine_id"],
                    created_at,
                    now,
                ),
            )
            conn.commit()
        self._sync_routine(workflow)
        self._sync_skill(workflow)
        return self.get_workflow(
            customer_id=workflow["customer_id"],
            workflow_id=workflow["workflow_id"],
        ) or workflow

    def list_workflows(
        self,
        *,
        customer_id: str,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        safe_customer = str(customer_id or "").strip()
        if not safe_customer:
            return []
        query = """
            SELECT * FROM intake_workflows
            WHERE customer_id = ?
        """
        params: list[Any] = [safe_customer]
        if not include_disabled:
            query += " AND enabled = 1"
        query += " ORDER BY updated_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._hydrate_workflow_row(row) for row in rows]

    def get_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any] | None:
        safe_customer = str(customer_id or "").strip()
        safe_workflow = str(workflow_id or "").strip()
        if not safe_customer or not safe_workflow:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM intake_workflows
                WHERE workflow_id = ? AND customer_id = ?
                """,
                (safe_workflow, safe_customer),
            ).fetchone()
        if row is None:
            return None
        return self._hydrate_workflow_row(row)

    def delete_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any]:
        workflow = self.get_workflow(customer_id=customer_id, workflow_id=workflow_id)
        if workflow is None:
            return {"ok": False, "deleted": False}
        with self._conn() as conn:
            conn.execute("DELETE FROM intake_workflows WHERE workflow_id = ?", (workflow["workflow_id"],))
            conn.execute("DELETE FROM intake_bookings WHERE workflow_id = ?", (workflow["workflow_id"],))
            conn.execute(
                "DELETE FROM intake_conversation_cursors WHERE workflow_id = ?",
                (workflow["workflow_id"],),
            )
            conn.commit()
        if self._scheduler is not None:
            with suppress(Exception):
                self._scheduler.remove_routine(str(workflow.get("routine_id", "")).strip())
        if self._skill_store is not None:
            with suppress(Exception):
                self._skill_store.delete_skill(
                    scope="user",
                    customer_id=str(workflow["customer_id"]),
                    name=self._workflow_skill_name(str(workflow["workflow_id"])),
                )
        return {"ok": True, "deleted": True, "workflow_id": workflow["workflow_id"]}

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
        query = """
            SELECT * FROM intake_bookings
            WHERE customer_id = ? AND workflow_id = ?
        """
        params: list[Any] = [safe_customer, safe_workflow]
        if safe_conversation:
            query += " AND conversation_id = ?"
            params.append(safe_conversation)
        query += " ORDER BY updated_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._hydrate_booking_row(row) for row in rows]

    def _sync_routine(self, workflow: dict[str, Any]) -> None:
        if self._scheduler is None:
            return
        payload = {
            "instruction": (
                "Run the configured intake workflow, inspect recent Instagram DMs through Composio, "
                "continue conversations when needed, and save completed bookings using the stored sink."
            ),
            "customer_id": workflow["customer_id"],
            "notify_user": bool(workflow["notify_user"]),
            "notification_opt_out": not bool(workflow["notify_user"]),
            "workflow_type": "intake_workflow",
            "workflow_id": workflow["workflow_id"],
            "channel": workflow["channel"],
            "provider": workflow["provider"],
        }
        routine = Routine(
            id=str(workflow["routine_id"]),
            name=str(workflow["name"]),
            schedule=str(workflow["schedule"]),
            payload=payload,
            enabled=bool(workflow["enabled"]),
            is_cron=" " in str(workflow["schedule"]) and len(str(workflow["schedule"]).split()) >= 5,
        )
        self._scheduler.add_routine(routine)

    def _workflow_skill_name(self, workflow_id: str) -> str:
        return f"intake-workflow-{workflow_id}"

    def _sync_skill(self, workflow: dict[str, Any]) -> None:
        if self._skill_store is None:
            return
        workflow_id = str(workflow["workflow_id"])
        name = self._workflow_skill_name(workflow_id)
        description = (
            f"Operate the {workflow['name']} Instagram intake workflow for this user."
        )
        instructions = (
            "## Purpose\n"
            f"Support the durable intake workflow `{workflow['name']}`.\n\n"
            "## Matching Rule\n"
            f"- Match conversations that fit this intent: {workflow['intent_description']}\n\n"
            "## Required Fields\n"
            f"- Collect these fields before save: {', '.join(workflow['required_fields'])}\n\n"
            "## Behavioral Rules\n"
            "- Ask concise follow-up questions in the Instagram DM when fields are missing.\n"
            "- When all required fields are present, save through the configured sink.\n"
            "- Treat the same DM thread as one active booking until completion.\n"
            "- If the last completed booking is still inside the edit window, follow-up changes may edit it.\n"
            "- Otherwise, a clearly new request should create a new booking.\n"
            "- Telegram notifications should stay concise and only summarize booking success or failures.\n"
        )
        supporting_files = {
            "workflow.json": _json_dumps(
                {
                    "workflow_id": workflow_id,
                    "name": workflow["name"],
                    "channel": workflow["channel"],
                    "provider": workflow["provider"],
                    "intent_description": workflow["intent_description"],
                    "required_fields": workflow["required_fields"],
                    "sink_type": workflow["sink_type"],
                }
            )
            + "\n"
        }
        skill_markdown = build_skill_markdown(
            name=name,
            description=description,
            instructions=instructions,
        )
        self._skill_store.upsert_skill(
            scope="user",
            customer_id=str(workflow["customer_id"]),
            name=name,
            skill_markdown=skill_markdown,
            source="intake_workflow",
            enabled=True,
            supporting_files=supporting_files,
        )

    def _get_active_booking(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        bookings = self.list_bookings(
            customer_id=customer_id,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
        )
        for booking in bookings:
            if str(booking.get("status", "")).strip().lower() == "active":
                return booking
        return None

    def _get_recent_completed_booking(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        bookings = self.list_bookings(
            customer_id=customer_id,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
        )
        now = _utc_now()
        for booking in bookings:
            if str(booking.get("status", "")).strip().lower() != "completed":
                continue
            edit_window_until = _parse_datetime(booking.get("edit_window_until"))
            if edit_window_until is None or edit_window_until < now:
                continue
            return booking
        return None

    def _get_cursor(self, *, workflow_id: str, conversation_id: str) -> dict[str, str]:
        with self._conn() as conn:
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
            "last_seen_conversation_updated_time": str(
                row["last_seen_conversation_updated_time"] or ""
            ),
            "last_seen_latest_outbound_message_id": str(
                row["last_seen_latest_outbound_message_id"] or ""
            ),
            "last_agent_action_at": str(row["last_agent_action_at"] or ""),
        }

    def _set_cursor(
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
        with self._conn() as conn:
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

    def _has_new_inbound_signal(
        self,
        *,
        conversation_summary: dict[str, Any],
        cursor: dict[str, str],
        force: bool,
    ) -> bool:
        if force:
            return True
        latest_inbound_id = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip()
        if not latest_inbound_id:
            return False
        last_seen_inbound_id = str(cursor.get("last_seen_inbound_message_id", "")).strip()
        if last_seen_inbound_id and latest_inbound_id == last_seen_inbound_id:
            return False

        latest_message_id = str(conversation_summary.get("latest_message_id", "") or "").strip()
        latest_outbound_id = str(
            conversation_summary.get("latest_outbound_message_id", "") or ""
        ).strip()
        latest_inbound_time = _parse_datetime(
            conversation_summary.get("latest_inbound_message_created_time")
        )
        last_seen_inbound_time = _parse_datetime(cursor.get("last_seen_inbound_message_time"))
        if latest_message_id and latest_outbound_id and latest_message_id == latest_outbound_id:
            if last_seen_inbound_time and latest_inbound_time and latest_inbound_time <= last_seen_inbound_time:
                return False
            if not last_seen_inbound_id:
                return False
        return True

    def _normalize_conversation_messages(
        self,
        *,
        conversation: dict[str, Any],
        recipient_id: str | None,
    ) -> list[dict[str, Any]]:
        payload = _safe_dict(conversation.get("data")) if "data" in conversation else _safe_dict(conversation)
        participants = _safe_list(_safe_dict(payload.get("participants")).get("data"))
        messages = _safe_list(_safe_dict(payload.get("messages")).get("data"))
        participant_usernames = {
            str(_safe_dict(item).get("id", "") or "").strip(): str(
                _safe_dict(item).get("username", "") or ""
            ).strip()
            for item in participants
            if str(_safe_dict(item).get("id", "") or "").strip()
        }
        safe_recipient = str(recipient_id or "").strip()
        normalized: list[dict[str, Any]] = []
        for item in messages:
            msg = _safe_dict(item)
            sender = _safe_dict(msg.get("from"))
            sender_id = str(sender.get("id", "") or "").strip()
            normalized.append(
                {
                    "id": str(msg.get("id", "") or "").strip(),
                    "created_time": str(msg.get("created_time", "") or "").strip(),
                    "sender_id": sender_id,
                    "sender_username": str(sender.get("username", "") or "").strip()
                    or participant_usernames.get(sender_id, ""),
                    "sender_role": "customer" if safe_recipient and sender_id == safe_recipient else "assistant",
                    "text": str(msg.get("message", "") or "").strip(),
                }
            )
        normalized.sort(key=lambda item: str(item.get("created_time", "")))
        return normalized[-12:]

    async def run_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        event_type: str = "scheduled",
        force: bool = False,
    ) -> dict[str, Any]:
        workflow = self.get_workflow(customer_id=customer_id, workflow_id=workflow_id)
        if workflow is None:
            return {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": f"Intake workflow {workflow_id} was not found.",
            }
        if not bool(workflow.get("enabled")) and not force:
            return {
                "ok": True,
                "workflow_id": workflow_id,
                "summary": NO_NOTIFY_TOKEN,
                "reason": "workflow_disabled",
            }
        if str(workflow.get("channel")) != "instagram_dm" or str(workflow.get("provider")) != "composio":
            return {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": (
                    f"Workflow {workflow['name']} failed: unsupported source "
                    f"{workflow.get('channel')}/{workflow.get('provider')}."
                ),
            }
        composio = self._composio
        if composio is None or not bool(getattr(composio, "enabled", False)):
            return {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": f"Workflow {workflow['name']} failed: Composio is not available.",
            }

        source_config = _safe_dict(workflow.get("source_config"))
        connected_account_id = str(source_config.get("connected_account_id", "") or "").strip() or None
        scan_limit = max(1, min(int(source_config.get("scan_limit", 10) or 10), 25))
        configured_conversation_ids = _unique_string_list(
            source_config.get("conversation_ids")
            if isinstance(source_config.get("conversation_ids"), list)
            else (
                [source_config.get("conversation_id")]
                if str(source_config.get("conversation_id", "")).strip()
                else []
            )
        )
        try:
            if configured_conversation_ids:
                items = []
                for conversation_id in configured_conversation_ids:
                    detailed = composio.get_instagram_conversation(
                        customer_id=str(workflow["customer_id"]),
                        conversation_id=conversation_id,
                        connected_account_id=connected_account_id,
                    )
                    items.append(_safe_dict(detailed.get("summary")))
            else:
                conversations_payload = composio.list_instagram_conversations(
                    customer_id=str(workflow["customer_id"]),
                    connected_account_id=connected_account_id,
                    limit=scan_limit,
                )
                items = _safe_list(conversations_payload.get("items"))
        except Exception as exc:
            return {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": f"Workflow {workflow['name']} failed while reading Instagram DMs: {exc}",
            }

        processed = 0
        matched = 0
        saved_notifications: list[str] = []
        errors: list[str] = []
        result_items: list[dict[str, Any]] = []
        for item in items:
            conversation_summary = _safe_dict(item)
            conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
            latest_inbound_id = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip()
            latest_inbound_time = str(
                conversation_summary.get("latest_inbound_message_created_time", "") or ""
            ).strip()
            conversation_updated_time = str(
                conversation_summary.get("conversation_updated_time", "") or ""
            ).strip()
            latest_outbound_id = str(
                conversation_summary.get("latest_outbound_message_id", "") or ""
            ).strip()
            if not conversation_id:
                continue
            cursor = self._get_cursor(
                workflow_id=str(workflow["workflow_id"]),
                conversation_id=conversation_id,
            )
            if not self._has_new_inbound_signal(
                conversation_summary=conversation_summary,
                cursor=cursor,
                force=force,
            ):
                continue

            processed += 1
            try:
                detailed = composio.get_instagram_conversation(
                    customer_id=str(workflow["customer_id"]),
                    conversation_id=conversation_id,
                    connected_account_id=connected_account_id,
                )
            except Exception as exc:
                errors.append(f"{conversation_id}: {exc}")
                continue

            detailed_summary = _safe_dict(detailed.get("summary"))
            conversation = _safe_dict(detailed.get("conversation"))
            cursor_summary = detailed_summary or conversation_summary
            latest_inbound_id = str(cursor_summary.get("latest_inbound_message_id", "") or "").strip()
            latest_inbound_time = str(
                cursor_summary.get("latest_inbound_message_created_time", "") or ""
            ).strip()
            conversation_updated_time = str(
                cursor_summary.get("conversation_updated_time", "") or ""
            ).strip()
            latest_outbound_id = str(
                cursor_summary.get("latest_outbound_message_id", "") or ""
            ).strip()
            active_booking = self._get_active_booking(
                customer_id=str(workflow["customer_id"]),
                workflow_id=str(workflow["workflow_id"]),
                conversation_id=conversation_id,
            )
            recent_completed_booking = self._get_recent_completed_booking(
                customer_id=str(workflow["customer_id"]),
                workflow_id=str(workflow["workflow_id"]),
                conversation_id=conversation_id,
            )
            decision, error = await self._decide_workflow_action(
                workflow=workflow,
                conversation_summary=cursor_summary,
                conversation=conversation,
                active_booking=active_booking,
                recent_completed_booking=recent_completed_booking,
            )
            if error:
                errors.append(f"{conversation_id}: {error}")
                continue
            if not bool(decision.get("matches_workflow")):
                self._set_cursor(
                    workflow_id=str(workflow["workflow_id"]),
                    conversation_id=conversation_id,
                    latest_inbound_message_id=latest_inbound_id,
                    latest_inbound_message_time=latest_inbound_time,
                    conversation_updated_time=conversation_updated_time,
                    latest_outbound_message_id=latest_outbound_id,
                )
                result_items.append(
                    {
                        "conversation_id": conversation_id,
                        "matched": False,
                        "status": "ignored",
                    }
                )
                continue

            matched += 1
            recovery_feedback: list[dict[str, Any]] = []
            applied: dict[str, Any] = {}
            apply_error: str | None = None
            for attempt in range(_MAX_DECISION_RECOVERY_ATTEMPTS + 1):
                applied, apply_error, feedback = await self._apply_decision(
                    workflow=workflow,
                    conversation_summary=cursor_summary,
                    conversation=conversation,
                    active_booking=active_booking,
                    recent_completed_booking=recent_completed_booking,
                    decision=decision,
                )
                if apply_error is None:
                    break
                if (
                    attempt >= _MAX_DECISION_RECOVERY_ATTEMPTS
                    or feedback is None
                    or str(feedback.get("phase", "")).strip() == "sink_execution"
                ):
                    break
                recovery_feedback.append(feedback)
                active_booking = self._get_active_booking(
                    customer_id=str(workflow["customer_id"]),
                    workflow_id=str(workflow["workflow_id"]),
                    conversation_id=conversation_id,
                )
                recent_completed_booking = self._get_recent_completed_booking(
                    customer_id=str(workflow["customer_id"]),
                    workflow_id=str(workflow["workflow_id"]),
                    conversation_id=conversation_id,
                )
                decision, error = await self._decide_workflow_action(
                    workflow=workflow,
                    conversation_summary=cursor_summary,
                    conversation=conversation,
                    active_booking=active_booking,
                    recent_completed_booking=recent_completed_booking,
                    execution_feedback=recovery_feedback,
                )
                if error:
                    apply_error = error
                    break
                if not bool(decision.get("matches_workflow")):
                    apply_error = "recovery decision no longer matches workflow"
                    break
            if apply_error:
                errors.append(f"{conversation_id}: {apply_error}")
                continue
            self._set_cursor(
                workflow_id=str(workflow["workflow_id"]),
                conversation_id=conversation_id,
                latest_inbound_message_id=latest_inbound_id,
                latest_inbound_message_time=latest_inbound_time,
                conversation_updated_time=conversation_updated_time,
                latest_outbound_message_id=latest_outbound_id,
                agent_action_at=_utc_now_iso(),
            )
            result_items.append(applied)
            saved_summary = str(applied.get("saved_summary", "") or "").strip()
            if saved_summary:
                saved_notifications.append(saved_summary)

        if errors:
            summary = (
                f"Workflow {workflow['name']} hit errors: " + " | ".join(errors[:3])
            )[:2000]
        elif saved_notifications:
            summary = "\n".join(saved_notifications[:3])[:2000]
        else:
            summary = NO_NOTIFY_TOKEN
        return {
            "ok": not errors,
            "workflow_id": str(workflow["workflow_id"]),
            "event_type": event_type,
            "processed_conversations": processed,
            "matched_conversations": matched,
            "results": result_items,
            "errors": errors,
            "summary": summary,
        }

    async def _decide_workflow_action(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        execution_feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        runtime = self._get_agent_runtime() if callable(self._get_agent_runtime) else None
        if runtime is None or not hasattr(runtime, "decide_intake_workflow"):
            return {}, "agent runtime does not support intake workflow decisions"
        recent_messages = self._normalize_conversation_messages(
            conversation=conversation,
            recipient_id=str(conversation_summary.get("recipient_id", "") or "").strip() or None,
        )
        try:
            decision = await runtime.decide_intake_workflow(
                customer_id=str(workflow["customer_id"]),
                workflow={
                    "workflow_id": workflow["workflow_id"],
                    "name": workflow["name"],
                    "intent_description": workflow["intent_description"],
                    "required_fields": workflow["required_fields"],
                    "field_guidance": workflow["field_guidance"],
                    "sink_type": workflow["sink_type"],
                },
                conversation={
                    "summary": conversation_summary,
                    "recent_messages": recent_messages,
                },
                active_booking=active_booking,
                recent_completed_booking=recent_completed_booking,
                execution_feedback=execution_feedback,
            )
        except Exception as exc:
            return {}, str(exc)
        if not isinstance(decision, dict) or not bool(decision.get("ok", False)):
            return {}, str(decision.get("error", "invalid intake workflow decision"))
        return decision, None

    async def _apply_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        decision: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
        booking_action = str(decision.get("booking_action", "ignore") or "ignore").strip().lower()
        if booking_action not in {
            "ignore",
            "update_active",
            "edit_recent_completed",
            "create_new_booking",
        }:
            return {}, f"unsupported booking_action={booking_action}", self._build_recovery_feedback(
                phase="decision_validation",
                error=f"unsupported booking_action={booking_action}",
                decision=decision,
            )
        if booking_action == "ignore":
            return {
                "conversation_id": str(conversation_summary.get("conversation_id", "") or ""),
                "matched": True,
                "status": "ignored",
            }, None, None

        target_booking: dict[str, Any] | None = None
        normalized_booking_action = booking_action
        if booking_action == "create_new_booking" and active_booking is not None:
            normalized_booking_action = "update_active"
        if normalized_booking_action == "update_active":
            if active_booking is None:
                normalized_booking_action = "create_new_booking"
            else:
                target_booking = dict(active_booking)
        elif normalized_booking_action == "edit_recent_completed":
            if recent_completed_booking is None:
                normalized_booking_action = "create_new_booking"
            else:
                target_booking = dict(recent_completed_booking)
        if normalized_booking_action == "create_new_booking":
            target_booking = {
                "booking_id": new_short_id("bkg"),
                "workflow_id": workflow["workflow_id"],
                "customer_id": workflow["customer_id"],
                "conversation_id": str(conversation_summary.get("conversation_id", "") or ""),
                "status": "active",
                "extracted_fields": {},
                "sink_write_status": "pending",
                "sink_record_ref": {},
                "conversation_summary": "",
                "last_customer_message_at": "",
                "opened_at": _utc_now_iso(),
                "completed_at": "",
                "edit_window_until": "",
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
        if target_booking is None:
            return {}, "workflow decision did not resolve a booking target", self._build_recovery_feedback(
                phase="decision_validation",
                error="workflow decision did not resolve a booking target",
                decision=decision,
            )

        extracted_fields = dict(_safe_dict(target_booking.get("extracted_fields")))
        extracted_fields.update(_safe_dict(decision.get("extracted_fields")))
        conversation_summary_text = str(decision.get("conversation_summary", "") or "").strip()
        if not conversation_summary_text:
            conversation_summary_text = str(
                conversation_summary.get("latest_inbound_message_text_preview", "") or ""
            ).strip()[:300]
        target_booking["extracted_fields"] = extracted_fields
        target_booking["conversation_summary"] = conversation_summary_text
        target_booking["last_customer_message_at"] = str(
            conversation_summary.get("latest_inbound_message_created_time", "") or ""
        ).strip()

        ready_to_save = bool(decision.get("ready_to_save"))
        reply_action = str(decision.get("reply_action", "none") or "none").strip().lower()
        reply_text = str(decision.get("reply_text", "") or "").strip()
        sink_ref = dict(_safe_dict(target_booking.get("sink_record_ref")))
        sink_status = str(target_booking.get("sink_write_status", "pending") or "pending").strip()
        saved_summary = ""
        if ready_to_save:
            save_payload = dict(_safe_dict(decision.get("save_payload")))
            if not save_payload:
                save_payload = dict(extracted_fields)
            missing = [
                field
                for field in _unique_string_list(workflow.get("required_fields"))
                if not str(save_payload.get(field, "") or "").strip()
            ]
            if missing:
                return {}, (
                    f"decision marked ready_to_save but required fields are missing: {', '.join(missing)}"
                ), self._build_recovery_feedback(
                    phase="decision_validation",
                    error=f"ready_to_save missing required fields: {', '.join(missing)}",
                    decision=decision,
                )
            sink_result, sink_error = self._write_to_sink(
                workflow=workflow,
                booking=target_booking,
                payload=save_payload,
            )
            if sink_error is not None:
                target_booking["status"] = "active"
                target_booking["sink_write_status"] = "failed"
                target_booking["sink_record_ref"] = sink_ref
                self._upsert_booking(target_booking)
                return {}, sink_error, self._build_recovery_feedback(
                    phase="sink_execution",
                    error=sink_error,
                    decision=decision,
                )
            sink_status = "succeeded"
            sink_ref = _safe_dict(sink_result)
            now = _utc_now()
            target_booking["status"] = "completed"
            target_booking["completed_at"] = now.isoformat()
            target_booking["edit_window_until"] = (now + _DEFAULT_EDIT_WINDOW).isoformat()
            target_booking["sink_write_status"] = sink_status
            target_booking["sink_record_ref"] = sink_ref
            target_booking["extracted_fields"] = save_payload
            target_booking["updated_at"] = now.isoformat()
            self._upsert_booking(target_booking)
            saved_summary = self._build_saved_summary(
                workflow=workflow,
                booking=target_booking,
                conversation_summary=conversation_summary,
            )
        else:
            if reply_action == "mark_cancelled":
                target_booking["status"] = "cancelled"
            else:
                target_booking["status"] = "active"
            target_booking["sink_write_status"] = sink_status
            target_booking["sink_record_ref"] = sink_ref
            target_booking["updated_at"] = _utc_now_iso()
            self._upsert_booking(target_booking)

        if reply_action == "send_reply":
            reply_error = await self._send_instagram_reply(
                workflow=workflow,
                conversation_summary=conversation_summary,
                reply_text=reply_text,
            )
            if reply_error is not None:
                return {}, reply_error, self._build_recovery_feedback(
                    phase="reply_execution",
                    error=reply_error,
                    decision=decision,
                )

        return {
            "conversation_id": str(conversation_summary.get("conversation_id", "") or ""),
            "matched": True,
            "status": str(target_booking.get("status", "") or "active"),
            "booking_id": str(target_booking.get("booking_id", "") or ""),
            "saved_summary": saved_summary,
        }, None, None

    def _build_recovery_feedback(
        self,
        *,
        phase: str,
        error: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "phase": str(phase or "").strip() or "unknown",
            "error": str(error or "").strip()[:1000],
            "prior_decision": {
                "booking_action": str(decision.get("booking_action", "") or "").strip().lower(),
                "reply_action": str(decision.get("reply_action", "") or "").strip().lower(),
                "ready_to_save": bool(decision.get("ready_to_save")),
                "missing_fields": _unique_string_list(decision.get("missing_fields")),
            },
        }

    async def _send_instagram_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None:
        if not reply_text:
            return "reply_action=send_reply requires non-empty reply_text"
        recipient_id = str(conversation_summary.get("recipient_id", "") or "").strip()
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
        if not recipient_id or not conversation_id:
            return "Instagram reply requires verified recipient_id and conversation_id"
        arguments = {
            "recipient_id": recipient_id,
            "conversation_id": conversation_id,
            "text": reply_text,
        }
        latest_inbound = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip()
        if latest_inbound:
            arguments["reply_to_message_id"] = latest_inbound
        source_config = _safe_dict(workflow.get("source_config"))
        connected_account_id = str(source_config.get("connected_account_id", "") or "").strip() or None
        try:
            result = self._composio.execute_tool(
                customer_id=str(workflow["customer_id"]),
                tool_slug="INSTAGRAM_SEND_TEXT_MESSAGE",
                arguments=arguments,
                connected_account_id=connected_account_id,
            )
        except Exception as exc:
            return f"failed to send Instagram DM reply: {exc}"
        if not bool(result.get("successful", False)):
            return str(result.get("error") or "Instagram DM reply failed")
        return None

    def _write_to_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        if sink_type == "local_csv":
            return self._write_to_local_csv(
                workflow=workflow,
                booking=booking,
                payload=payload,
            )
        if sink_type in {"google_sheets_composio", "generic_composio_write"}:
            return self._write_to_composio_sink(
                workflow=workflow,
                booking=booking,
                payload=payload,
            )
        return {}, f"unsupported sink_type={sink_type}"

    def _write_to_local_csv(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        sink_config = _safe_dict(workflow.get("sink_config"))
        relative_path = str(sink_config.get("file_path", "") or "").strip()
        if not relative_path:
            return {}, "local_csv sink is missing file_path"
        absolute_path = (self._project_root / relative_path).resolve()
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        base_row = {
            "booking_id": str(booking["booking_id"]),
            "workflow_id": str(workflow["workflow_id"]),
            "workflow_name": str(workflow["name"]),
            "conversation_id": str(booking["conversation_id"]),
            "customer_id": str(workflow["customer_id"]),
            "status": "completed",
            "completed_at": _utc_now_iso(),
        }
        for key, value in payload.items():
            base_row[str(key)] = str(value or "")
        rows: list[dict[str, str]] = []
        fieldnames: list[str] = list(base_row.keys())
        if absolute_path.exists():
            with absolute_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                existing_fields = list(reader.fieldnames or [])
                for item in reader:
                    rows.append({str(k): str(v or "") for k, v in item.items()})
                for field in existing_fields:
                    if field not in fieldnames:
                        fieldnames.append(field)
                for field in base_row:
                    if field not in fieldnames:
                        fieldnames.append(field)
        updated = False
        for row in rows:
            if str(row.get("booking_id", "")).strip() != str(booking["booking_id"]):
                continue
            row.update(base_row)
            updated = True
            break
        if not updated:
            rows.append(base_row)
        with absolute_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fieldnames})
        return {
            "sink_type": "local_csv",
            "file_path": relative_path,
            "booking_id": str(booking["booking_id"]),
        }, None

    def _write_to_composio_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            return {}, "Composio is not available for sink execution"
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        enriched_payload = {
            **payload,
            "booking_id": str(booking["booking_id"]),
            "workflow_id": str(workflow["workflow_id"]),
            "conversation_id": str(booking["conversation_id"]),
            "customer_id": str(workflow["customer_id"]),
        }
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        tool_slug = self._resolve_composio_sink_tool_slug(
            sink_type=sink_type,
            sink_config=sink_config,
        )
        if not tool_slug:
            return {}, f"could not resolve a Composio tool for toolkit={toolkit or 'unknown'}"
        arguments: dict[str, Any]
        if sink_type == "google_sheets_composio":
            key_source = "booking_id"
            key_header = str(field_mapping.get(key_source, "Booking ID") or "Booking ID").strip()
            headers = [key_header]
            row = [enriched_payload.get(key_source)]
            for source_key, header_name in field_mapping.items():
                safe_source = str(source_key or "").strip()
                safe_header = str(header_name or "").strip()
                if not safe_source or not safe_header or safe_source == key_source:
                    continue
                headers.append(safe_header)
                row.append(enriched_payload.get(safe_source))
            arguments = {
                **static_arguments,
                "headers": headers,
                "rows": [row],
                "keyColumn": key_header,
            }
        else:
            arguments = dict(static_arguments)
            for target_key, source_key in field_mapping.items():
                arguments[target_key] = enriched_payload.get(source_key)
        connected_account_id = str(sink_config.get("connected_account_id", "") or "").strip() or None
        try:
            result = self._composio.execute_tool(
                customer_id=str(workflow["customer_id"]),
                tool_slug=tool_slug,
                arguments=arguments,
                connected_account_id=connected_account_id,
            )
        except Exception as exc:
            return {}, f"sink execution failed: {exc}"
        if not bool(result.get("successful", False)):
            return {}, str(result.get("error") or "sink execution failed")
        return {
            "sink_type": str(workflow["sink_type"]),
            "toolkit": toolkit,
            "booking_id": str(booking["booking_id"]),
            "data": result.get("data"),
        }, None

    def _resolve_composio_sink_tool_slug(
        self,
        *,
        sink_type: str,
        sink_config: dict[str, Any],
    ) -> str:
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            raise ValueError("Composio is not available for sink tool resolution")
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        if not toolkit:
            raise ValueError("sink_config.toolkit is required")
        operation_hint = str(sink_config.get("operation_hint", "") or "").strip().lower()
        queries: list[str] = []
        if operation_hint:
            queries.append(operation_hint)
        if sink_type == "google_sheets_composio":
            queries.extend(["upsert rows", "append rows", "add row", "rows"])
        elif operation_hint:
            queries.append("write")
        seen_queries: set[str] = set()
        candidates: list[dict[str, Any]] = []
        for query in queries:
            safe_query = str(query or "").strip().lower()
            if not safe_query or safe_query in seen_queries:
                continue
            seen_queries.add(safe_query)
            result = self._composio.search_tools(
                query=safe_query,
                toolkits=[toolkit],
                limit=20,
            )
            if not bool(result.get("ok", False)):
                continue
            items = _safe_list(result.get("items"))
            if not items:
                continue
            candidates.extend(item for item in items if isinstance(item, dict))
            selected = self._select_composio_sink_candidate(
                sink_type=sink_type,
                toolkit=toolkit,
                operation_hint=operation_hint or safe_query,
                candidates=candidates,
            )
            if selected:
                return selected
        selected = self._select_composio_sink_candidate(
            sink_type=sink_type,
            toolkit=toolkit,
            operation_hint=operation_hint,
            candidates=candidates,
        )
        if selected:
            return selected
        raise ValueError(f"no matching tool found in toolkit={toolkit}")

    @staticmethod
    def _select_composio_sink_candidate(
        *,
        sink_type: str,
        toolkit: str,
        operation_hint: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        hint_tokens = {token for token in operation_hint.replace("_", " ").split() if token}
        best_slug = ""
        best_score = -1
        for item in candidates:
            slug = str(item.get("slug", "") or "").strip()
            if not slug:
                continue
            item_toolkit = _normalize_toolkit_slug(item.get("toolkit_slug"))
            if item_toolkit and item_toolkit != toolkit:
                continue
            haystack = " ".join(
                [
                    slug,
                    str(item.get("name", "") or ""),
                    str(item.get("description", "") or ""),
                ]
            ).lower()
            score = 0
            upper_slug = slug.upper()
            if sink_type == "google_sheets_composio":
                if "UPSERT_ROWS" in upper_slug:
                    score += 100
                if "APPEND_ROWS" in upper_slug:
                    score += 80
                if "ADD_ROW" in upper_slug:
                    score += 60
                if "ROW" in upper_slug:
                    score += 15
            for token in hint_tokens:
                if token in haystack:
                    score += 8
            input_schema = item.get("input_schema")
            if isinstance(input_schema, dict):
                schema_text = json.dumps(input_schema, ensure_ascii=False).lower()
                if "rows" in schema_text:
                    score += 10
                if "headers" in schema_text:
                    score += 6
                if "keycolumn" in schema_text:
                    score += 6
            if score > best_score:
                best_score = score
                best_slug = slug
        return best_slug

    def _upsert_booking(self, booking: dict[str, Any]) -> None:
        now = _utc_now_iso()
        with self._conn() as conn:
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

    def _build_saved_summary(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> str:
        contact = str(conversation_summary.get("latest_inbound_sender_username", "") or "").strip()
        if not contact:
            contact = str(conversation_summary.get("recipient_id", "") or "").strip() or "instagram_contact"
        fields = _safe_dict(booking.get("extracted_fields"))
        parts = [
            f"Booking saved for {workflow['name']}:",
            f"contact={contact}",
            f"booking_id={booking['booking_id']}",
        ]
        for key in _unique_string_list(list(fields.keys()))[:5]:
            value = str(fields.get(key, "") or "").strip()
            if value:
                parts.append(f"{key}={value}")
        parts.append(f"sink={workflow['sink_type']}")
        return " ".join(parts)[:1000]
