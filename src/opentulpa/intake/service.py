"""Persistent intake workflow storage and wake-time execution."""

from __future__ import annotations

import asyncio
import csv
import json
import re
import sqlite3
import threading
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.ids import new_short_id
from opentulpa.intake.workflow_skill import build_intake_workflow_skill, workflow_skill_name
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN
from opentulpa.scheduler.models import Routine

_ALLOWED_CHANNELS = {"instagram_dm", "telegram_business_dm"}
_ALLOWED_PROVIDERS = {"composio", "telegram_bot_api"}
_ALLOWED_SINK_TYPES = {"google_sheets_composio", "local_csv", "generic_composio_write"}
_DEFAULT_SCHEDULE = "*/5 * * * *"
_DEFAULT_EDIT_WINDOW = timedelta(hours=2)
_MAX_LATEST_INBOUND_AGE = timedelta(minutes=1)
_MAX_DECISION_RECOVERY_ATTEMPTS = 2
_TELEGRAM_BUSINESS_WEBHOOK_DEBOUNCE_SECONDS = 1.5


def _channel_uses_scheduler(channel: str) -> bool:
    return str(channel or "").strip().lower() != "telegram_business_dm"


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


def _is_older_than(value: Any, *, max_age: timedelta) -> bool:
    parsed = _parse_datetime(value)
    if parsed is None:
        return False
    return (_utc_now() - parsed) > max_age


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


def _sheet_cell_value(value: Any) -> Any:
    if value is None:
        return ""
    return value


def _normalize_google_sheets_arguments(value: dict[str, Any]) -> dict[str, Any]:
    out = dict(value)
    for canonical, aliases in {
        "spreadsheetId": ("spreadsheet_id", "spreadsheetID", "spreadsheet"),
        "sheetName": ("sheet_name", "worksheet", "worksheet_name", "tab_name"),
    }.items():
        if str(out.get(canonical, "") or "").strip():
            continue
        for alias in aliases:
            alias_value = out.pop(alias, None)
            if str(alias_value or "").strip():
                out[canonical] = alias_value
                break
    return out


def _normalize_google_sheets_field_mapping(
    field_mapping: dict[str, str],
    *,
    payload_keys: set[str],
) -> dict[str, str]:
    """Return source-field -> sheet-header mapping.

    Models sometimes produce the inverse shape for human-friendly headers, e.g.
    {"Booking ID": "booking_id"}. Flip those entries when the value is a known
    payload key.
    """

    out: dict[str, str] = {}
    for raw_key, raw_value in field_mapping.items():
        key = str(raw_key or "").strip()
        value = str(raw_value or "").strip()
        if not key or not value:
            continue
        if key in payload_keys:
            out[key] = value
            continue
        if value in payload_keys:
            out[value] = key
            continue
        out[key] = value
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


def _looks_like_cyrillic(*values: Any) -> bool:
    text = " ".join(str(value or "") for value in values)
    return any("\u0400" <= char <= "\u04ff" for char in text)


def _extract_phone_hint(value: Any) -> str:
    text = str(value or "")
    match = re.search(r"\+?\d[\d\s().-]{6,}\d", text)
    return match.group(0).strip(" .,-") if match else ""


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
        telegram_business: Any | None = None,
        file_vault: FileVaultService | None = None,
        get_agent_runtime: Any | None = None,
    ) -> None:
        self._db_path = db_path.resolve()
        self._project_root = project_root.resolve()
        self._scheduler = scheduler
        self._skill_store = skill_store
        self._composio = composio
        self._telegram_business = telegram_business
        self._file_vault = file_vault
        self._get_agent_runtime = get_agent_runtime
        self._conversation_locks_guard = threading.Lock()
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._init_db()

    def _runtime_for_observability(self) -> Any | None:
        getter = self._get_agent_runtime
        if not callable(getter):
            return None
        with suppress(Exception):
            return getter()
        return None

    @staticmethod
    def _intake_trace_id(*, workflow: dict[str, Any], conversation_summary: dict[str, Any]) -> str:
        workflow_id = str(workflow.get("workflow_id", "") or "").strip() or "workflow"
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip() or "conversation"
        latest_inbound_id = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip() or "latest"
        return f"intake_{workflow_id}_{conversation_id}_{latest_inbound_id}"

    @staticmethod
    def _intake_thread_id(*, workflow: dict[str, Any], conversation_summary: dict[str, Any]) -> str:
        workflow_id = str(workflow.get("workflow_id", "") or "").strip() or "workflow"
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip() or "conversation"
        return f"intake_decision_{workflow_id}_{conversation_id}"

    def _intake_observability_fields(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        **extra: Any,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "trace_id": self._intake_trace_id(workflow=workflow, conversation_summary=conversation_summary),
            "thread_id": self._intake_thread_id(workflow=workflow, conversation_summary=conversation_summary),
            "customer_id": str(workflow.get("customer_id", "") or "").strip(),
            "workflow_id": str(workflow.get("workflow_id", "") or "").strip(),
            "workflow_name": str(workflow.get("name", "") or "").strip(),
            "routine_id": str(workflow.get("routine_id", "") or "").strip(),
            "channel": str(workflow.get("channel", "") or "").strip() or "instagram_dm",
            "provider": str(workflow.get("provider", "") or "").strip() or "composio",
            "sink_type": str(workflow.get("sink_type", "") or "").strip(),
            "conversation_id": str(conversation_summary.get("conversation_id", "") or "").strip(),
            "recipient_id": str(conversation_summary.get("recipient_id", "") or "").strip(),
            "latest_inbound_message_id": str(
                conversation_summary.get("latest_inbound_message_id", "") or ""
            ).strip(),
            "latest_inbound_sender_username": str(
                conversation_summary.get("latest_inbound_sender_username", "") or ""
            ).strip(),
        }
        for key, value in extra.items():
            safe_key = str(key or "").strip()
            if safe_key:
                fields[safe_key] = value
        return fields

    def _emit_observability(
        self,
        *,
        event: str,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        **extra: Any,
    ) -> None:
        runtime = self._runtime_for_observability()
        fields = self._intake_observability_fields(
            workflow=workflow,
            conversation_summary=conversation_summary,
            **extra,
        )
        customer_id = str(fields.pop("customer_id", "") or "").strip() or None
        record = getattr(runtime, "record_observability_event", None)
        if callable(record):
            record(
                event=event,
                customer_id=customer_id,
                posthog_event=event,
                **fields,
            )
            return
        log_event = getattr(runtime, "log_behavior_event", None)
        if callable(log_event):
            log_event(event=event, **fields)

    def _conversation_lock(self, *, workflow_id: str, conversation_id: str) -> asyncio.Lock:
        key = f"{workflow_id}:{conversation_id}"
        with self._conversation_locks_guard:
            lock = self._conversation_locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._conversation_locks[key] = lock
            return lock

    @staticmethod
    def _conversation_debounce_seconds(*, workflow: dict[str, Any], event_type: str) -> float:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        if channel == "telegram_business_dm" and str(event_type or "").strip() == "telegram_business_webhook":
            return _TELEGRAM_BUSINESS_WEBHOOK_DEBOUNCE_SECONDS
        return 0.0

    def _reload_conversation_summary(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        fallback: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        items, source_error = self._load_source_items(workflow=workflow)
        if source_error is not None:
            return fallback, source_error
        for item in items:
            summary = _safe_dict(item)
            if str(summary.get("conversation_id", "") or "").strip() == conversation_id:
                return summary, None
        return fallback, None

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
                    assistant_instructions TEXT NOT NULL DEFAULT '',
                    knowledge_file_ids_json TEXT NOT NULL DEFAULT '[]',
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
            self._ensure_workflow_columns(conn)
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

    @staticmethod
    def _ensure_workflow_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(intake_workflows)").fetchall()
        existing = {str(row["name"] or "") for row in rows}
        required_columns = {
            "assistant_instructions": "TEXT NOT NULL DEFAULT ''",
            "knowledge_file_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        }
        for column, column_type in required_columns.items():
            if column in existing:
                continue
            conn.execute(
                f"ALTER TABLE intake_workflows ADD COLUMN {column} {column_type}"
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
        assistant_instructions: str,
        knowledge_file_ids: list[str],
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
        safe_assistant_instructions = str(assistant_instructions or "").strip()
        safe_knowledge_file_ids = _unique_string_list(knowledge_file_ids)
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_name:
            raise ValueError("name is required")
        if safe_channel not in _ALLOWED_CHANNELS:
            raise ValueError("channel must be instagram_dm|telegram_business_dm")
        if safe_provider not in _ALLOWED_PROVIDERS:
            raise ValueError("provider must be composio|telegram_bot_api")
        if safe_channel == "instagram_dm" and safe_provider != "composio":
            raise ValueError("instagram_dm workflows require provider=composio")
        if safe_channel == "telegram_business_dm" and safe_provider != "telegram_bot_api":
            raise ValueError("telegram_business_dm workflows require provider=telegram_bot_api")
        if safe_channel == "telegram_business_dm":
            safe_source_config = self._resolve_telegram_business_source_config(
                customer_id=safe_customer,
                source_config=safe_source_config,
            )
            business_connection_id = str(safe_source_config.get("business_connection_id", "") or "").strip()
            if not business_connection_id:
                raise ValueError("telegram_business_dm workflows require source_config.business_connection_id")
            safe_schedule = ""
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
        if _channel_uses_scheduler(safe_channel):
            safe_routine_id = (
                str(existing_record.get("routine_id", "")).strip()
                if existing is not None
                else ""
            ) or new_short_id("rtn")
        else:
            safe_routine_id = ""
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
            "assistant_instructions": safe_assistant_instructions,
            "knowledge_file_ids": safe_knowledge_file_ids,
            "sink_type": safe_sink_type,
            "sink_config": safe_sink_config,
            "schedule": safe_schedule,
            "notify_user": bool(notify_user),
            "enabled": bool(enabled),
            "routine_id": safe_routine_id,
        }

    def _resolve_telegram_business_source_config(
        self,
        *,
        customer_id: str,
        source_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        safe_source_config = _safe_dict(source_config)
        business_connection_id = str(
            safe_source_config.get("business_connection_id", "") or ""
        ).strip()
        if business_connection_id:
            return safe_source_config
        telegram_business = self._telegram_business
        if telegram_business is None or not hasattr(telegram_business, "status"):
            return safe_source_config
        status = _safe_dict(telegram_business.status(customer_id=customer_id))
        enabled_connections = [
            _safe_dict(item)
            for item in _safe_list(status.get("connections"))
            if bool(_safe_dict(item).get("is_enabled"))
        ]
        if len(enabled_connections) == 1:
            resolved = dict(safe_source_config)
            resolved["business_connection_id"] = str(
                enabled_connections[0].get("business_connection_id", "") or ""
            ).strip()
            return resolved
        if not enabled_connections:
            raise ValueError(
                "telegram_business_dm workflows require a connected Telegram Business account"
            )
        raise ValueError(
            "telegram_business_dm workflows found multiple connected business accounts; "
            "specify source_config.business_connection_id"
        )

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
            requested_path = str(
                safe_config.get("file_path", "")
                or safe_config.get("filename", "")
                or ""
            ).strip()
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
            "assistant_instructions": str(row["assistant_instructions"] or ""),
            "knowledge_file_ids": json.loads(row["knowledge_file_ids_json"] or "[]"),
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
        assistant_instructions: str = "",
        knowledge_file_ids: list[str] | None = None,
        sink_type: str,
        sink_config: dict[str, Any] | None = None,
        schedule: str = _DEFAULT_SCHEDULE,
        notify_user: bool = True,
        enabled: bool = True,
    ) -> dict[str, Any]:
        existing = None
        safe_workflow_id = _normalize_optional_id(workflow_id)
        safe_channel = str(channel or "instagram_dm").strip().lower() or "instagram_dm"
        if safe_workflow_id:
            existing = self.get_workflow(customer_id=customer_id, workflow_id=safe_workflow_id)
        if safe_channel == "telegram_business_dm":
            existing_telegram_workflow = self._get_unique_telegram_business_workflow(
                customer_id=customer_id
            )
            if existing_telegram_workflow is not None:
                existing_telegram_workflow_id = str(
                    existing_telegram_workflow.get("workflow_id", "") or ""
                ).strip()
                if existing is not None and safe_workflow_id == existing_telegram_workflow_id:
                    raise ValueError(
                        "telegram_business_dm workflows cannot be edited in place; fetch the "
                        "existing workflow for context, delete it, then create a new workflow"
                    )
                if not safe_workflow_id:
                    raise ValueError(
                        "telegram_business_dm workflows cannot be updated in place; fetch the "
                        "existing workflow for context, delete it, then create a new workflow"
                    )
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
            assistant_instructions=assistant_instructions,
            knowledge_file_ids=knowledge_file_ids or [],
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
                    assistant_instructions, knowledge_file_ids_json, sink_type,
                    sink_config_json, schedule, notify_user, enabled, routine_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    knowledge_file_ids_json=excluded.knowledge_file_ids_json,
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
                    workflow["assistant_instructions"],
                    _json_dumps(workflow["knowledge_file_ids"]),
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

    def _get_unique_telegram_business_workflow(
        self,
        *,
        customer_id: str,
    ) -> dict[str, Any] | None:
        workflows = [
            workflow
            for workflow in self.list_workflows(customer_id=customer_id, include_disabled=True)
            if str(workflow.get("channel", "") or "").strip().lower() == "telegram_business_dm"
        ]
        if not workflows:
            return None
        if len(workflows) > 1:
            raise ValueError(
                "telegram_business_dm supports only one workflow per customer; multiple existing workflows found"
            )
        return workflows[0]

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

    def list_customer_summaries(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
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
                    name=workflow_skill_name(str(workflow["workflow_id"])),
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

    def _knowledge_files_for_workflow(self, *, customer_id: str, workflow: dict[str, Any]) -> list[dict[str, Any]]:
        file_vault = self._file_vault
        if file_vault is None:
            return []
        return file_vault.get_many(
            customer_id=str(customer_id or "").strip(),
            file_ids=_unique_string_list(workflow.get("knowledge_file_ids")),
        )

    def _sync_routine(self, workflow: dict[str, Any]) -> None:
        if self._scheduler is None:
            return
        if not _channel_uses_scheduler(str(workflow.get("channel", "") or "").strip()):
            routine_id = str(workflow.get("routine_id", "") or "").strip()
            if routine_id:
                with suppress(Exception):
                    self._scheduler.remove_routine(routine_id)
            return
        payload = {
            "instruction": (
                "Run the configured intake workflow, inspect recent external customer messages, "
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

    def _sync_skill(self, workflow: dict[str, Any]) -> None:
        if self._skill_store is None:
            return
        skill = build_intake_workflow_skill(workflow)
        self._skill_store.upsert_skill(
            scope="user",
            customer_id=str(workflow["customer_id"]),
            name=str(skill["name"]),
            skill_markdown=str(skill["skill_markdown"]),
            source="intake_workflow",
            enabled=True,
            supporting_files=dict(skill.get("supporting_files") or {}),
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
        workflow: dict[str, Any],
        conversation: dict[str, Any],
        recipient_id: str | None,
    ) -> list[dict[str, Any]]:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        if channel == "telegram_business_dm":
            messages = _safe_list(conversation.get("messages"))
            normalized: list[dict[str, Any]] = []
            for item in messages:
                msg = _safe_dict(item)
                normalized.append(
                    {
                        "id": str(msg.get("message_id", msg.get("id", "")) or "").strip(),
                        "created_time": str(msg.get("date_iso", msg.get("created_time", "")) or "").strip(),
                        "sender_id": str(msg.get("from_user_id", msg.get("sender_id", "")) or "").strip(),
                        "sender_username": str(msg.get("from_username", msg.get("sender_username", "")) or "").strip(),
                        "sender_role": str(msg.get("sender_role", "") or "").strip() or "customer",
                        "text": str(msg.get("text", "") or "").strip(),
                    }
                )
            normalized.sort(key=lambda item: str(item.get("created_time", "")))
            return normalized[-12:]
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

    def _source_matches_workflow(
        self,
        *,
        workflow: dict[str, Any],
        business_connection_id: str,
        conversation_id: str,
    ) -> bool:
        source_config = _safe_dict(workflow.get("source_config"))
        expected_business_connection_id = str(
            source_config.get("business_connection_id", "") or ""
        ).strip()
        if expected_business_connection_id and expected_business_connection_id != business_connection_id:
            return False
        configured_conversation_ids = _unique_string_list(
            source_config.get("conversation_ids")
            if isinstance(source_config.get("conversation_ids"), list)
            else (
                [source_config.get("conversation_id")]
                if str(source_config.get("conversation_id", "")).strip()
                else []
            )
        )
        return not (
            configured_conversation_ids and conversation_id not in configured_conversation_ids
        )

    def _fallback_out_of_scope_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        decision: dict[str, Any],
    ) -> str:
        latest_text = str(
            conversation_summary.get("latest_inbound_message_text_preview", "") or ""
        ).strip()
        decision_text = " ".join(
            [
                str(decision.get("conversation_summary", "") or ""),
                str(decision.get("reason", "") or ""),
            ]
        ).lower()
        if not latest_text:
            return ""
        out_of_scope_markers = (
            "out of scope",
            "outside scope",
            "outside the workflow",
            "outside this workflow",
            "outside the scoped workflow",
            "вне scope",
            "вне workflow",
            "вне этого workflow",
            "не входит",
            "не покрывает",
        )
        if not any(marker in decision_text for marker in out_of_scope_markers):
            return ""
        latest_lower = latest_text.lower()
        inquiry_markers = (
            "?",
            "сколько",
            "стоим",
            "цен",
            "прайс",
            "запис",
            "можно",
            "услуг",
            "how much",
            "price",
            "cost",
            "book",
            "appointment",
            "service",
        )
        if not any(marker in latest_lower for marker in inquiry_markers):
            return ""
        scope = str(workflow.get("name") or workflow.get("intent_description") or "").strip()
        if len(scope) > 120:
            scope = scope[:117].rstrip() + "..."
        if not scope:
            scope = "this workflow"
        instructions = str(workflow.get("assistant_instructions", "") or "")
        phone = _extract_phone_hint(instructions)
        if _looks_like_cyrillic(latest_text, instructions, scope):
            phone_part = f" по телефону {phone}" if phone else " напрямую"
            return (
                f"Здравствуйте! Сейчас я могу помочь только по workflow «{scope}». "
                f"По этой услуге, пожалуйста, обратитесь{phone_part}."
            )
        phone_part = f" at {phone}" if phone else " directly"
        return (
            f"Hi! I can only help with the current workflow: {scope}. "
            f"For this service, please contact the business{phone_part}."
        )

    def _load_source_items(
        self,
        *,
        workflow: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None]:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        if channel == "instagram_dm" and provider == "composio":
            composio = self._composio
            if composio is None or not bool(getattr(composio, "enabled", False)):
                return [], f"Workflow {workflow['name']} failed: Composio is not available."
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
                return [], f"Workflow {workflow['name']} failed while reading Instagram DMs: {exc}"
            return items, None
        if channel == "telegram_business_dm" and provider == "telegram_bot_api":
            telegram_business = self._telegram_business
            if telegram_business is None:
                return [], f"Workflow {workflow['name']} failed: Telegram Business is not available."
            source_config = _safe_dict(workflow.get("source_config"))
            business_connection_id = str(source_config.get("business_connection_id", "") or "").strip()
            if not business_connection_id:
                return [], f"Workflow {workflow['name']} failed: source_config.business_connection_id is required."
            scan_limit = max(1, min(int(source_config.get("scan_limit", 10) or 10), 50))
            configured_conversation_ids = _unique_string_list(
                source_config.get("conversation_ids")
                if isinstance(source_config.get("conversation_ids"), list)
                else (
                    [source_config.get("conversation_id")]
                    if str(source_config.get("conversation_id", "")).strip()
                    else []
                )
            )
            payload = telegram_business.list_conversations(
                customer_id=str(workflow["customer_id"]),
                business_connection_id=business_connection_id,
                limit=scan_limit,
                chat_ids=configured_conversation_ids or None,
            )
            return _safe_list(payload.get("items")), None
        return [], (
            f"Workflow {workflow['name']} failed: unsupported source "
            f"{workflow.get('channel')}/{workflow.get('provider')}."
        )

    def _load_source_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        if channel == "instagram_dm" and provider == "composio":
            source_config = _safe_dict(workflow.get("source_config"))
            connected_account_id = str(source_config.get("connected_account_id", "") or "").strip() or None
            try:
                detailed = self._composio.get_instagram_conversation(
                    customer_id=str(workflow["customer_id"]),
                    conversation_id=conversation_id,
                    connected_account_id=connected_account_id,
                )
            except Exception as exc:
                return {}, {}, str(exc)
            return _safe_dict(detailed.get("summary")), _safe_dict(detailed.get("conversation")), None
        if channel == "telegram_business_dm" and provider == "telegram_bot_api":
            source_config = _safe_dict(workflow.get("source_config"))
            business_connection_id = str(source_config.get("business_connection_id", "") or "").strip()
            detailed = self._telegram_business.get_conversation(
                customer_id=str(workflow["customer_id"]),
                business_connection_id=business_connection_id,
                conversation_id=conversation_id,
            )
            if not bool(detailed.get("ok", False)):
                return {}, {}, str(detailed.get("error") or "conversation not found")
            return _safe_dict(detailed.get("summary")), _safe_dict(detailed.get("conversation")), None
        return {}, {}, "unsupported source"

    async def _send_source_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        if channel == "instagram_dm" and provider == "composio":
            return await self._send_instagram_reply(
                workflow=workflow,
                conversation_summary=conversation_summary,
                reply_text=reply_text,
            )
        if channel == "telegram_business_dm" and provider == "telegram_bot_api":
            return await self._send_telegram_business_reply(
                workflow=workflow,
                conversation_summary=conversation_summary,
                reply_text=reply_text,
            )
        return f"unsupported reply source {channel}/{provider}"

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
        items, source_error = self._load_source_items(workflow=workflow)
        if source_error is not None:
            return {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": source_error,
            }

        processed = 0
        matched = 0
        saved_notifications: list[str] = []
        errors: list[str] = []
        result_items: list[dict[str, Any]] = []
        for item in items:
            conversation_summary = _safe_dict(item)
            conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
            if not conversation_id:
                continue
            async with self._conversation_lock(
                workflow_id=str(workflow["workflow_id"]),
                conversation_id=conversation_id,
            ):
                debounce_seconds = self._conversation_debounce_seconds(
                    workflow=workflow,
                    event_type=event_type,
                )
                if debounce_seconds > 0:
                    await asyncio.sleep(debounce_seconds)
                    conversation_summary, refresh_error = self._reload_conversation_summary(
                        workflow=workflow,
                        conversation_id=conversation_id,
                        fallback=conversation_summary,
                    )
                    if refresh_error:
                        errors.append(f"{conversation_id}: {refresh_error}")
                        self._emit_observability(
                            event="intake.conversation.error",
                            workflow=workflow,
                            conversation_summary=conversation_summary,
                            phase="reload_summary",
                            error=refresh_error,
                        )
                        continue

                latest_inbound_id = str(
                    conversation_summary.get("latest_inbound_message_id", "") or ""
                ).strip()
                latest_inbound_time = str(
                    conversation_summary.get("latest_inbound_message_created_time", "") or ""
                ).strip()
                conversation_updated_time = str(
                    conversation_summary.get("conversation_updated_time", "") or ""
                ).strip()
                latest_outbound_id = str(
                    conversation_summary.get("latest_outbound_message_id", "") or ""
                ).strip()
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
                if not force and _is_older_than(
                    conversation_summary.get("latest_inbound_message_created_time"),
                    max_age=_MAX_LATEST_INBOUND_AGE,
                ):
                    self._set_cursor(
                        workflow_id=str(workflow["workflow_id"]),
                        conversation_id=conversation_id,
                        latest_inbound_message_id=latest_inbound_id,
                        latest_inbound_message_time=latest_inbound_time,
                        conversation_updated_time=conversation_updated_time,
                        latest_outbound_message_id=latest_outbound_id,
                    )
                    continue

                processed += 1
                self._emit_observability(
                    event="intake.conversation.start",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    force=bool(force),
                    cursor_latest_inbound_message_id=str(
                        cursor.get("latest_inbound_message_id", "") or ""
                    ).strip(),
                    cursor_latest_outbound_message_id=str(
                        cursor.get("latest_outbound_message_id", "") or ""
                    ).strip(),
                )
                try:
                    detailed_summary, conversation, detail_error = self._load_source_conversation(
                        workflow=workflow,
                        conversation_id=conversation_id,
                    )
                except Exception as exc:
                    detailed_summary = {}
                    conversation = {}
                    detail_error = str(exc)
                if detail_error:
                    error_text = str(detail_error)
                    errors.append(f"{conversation_id}: {error_text}")
                    self._emit_observability(
                        event="intake.conversation.error",
                        workflow=workflow,
                        conversation_summary=conversation_summary,
                        phase="load_conversation",
                        error=error_text,
                    )
                    continue

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
                    self._emit_observability(
                        event="intake.conversation.error",
                        workflow=workflow,
                        conversation_summary=cursor_summary,
                        phase="decision",
                        error=error,
                    )
                    continue
                if not bool(decision.get("matches_workflow")):
                    reply_action = str(decision.get("reply_action", "none") or "none").strip().lower()
                    reply_text = str(decision.get("reply_text", "") or "").strip()
                    if reply_action != "send_reply" or not reply_text:
                        fallback_reply = self._fallback_out_of_scope_reply(
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            decision=decision,
                        )
                        if fallback_reply:
                            reply_action = "send_reply"
                            reply_text = fallback_reply
                            self._emit_observability(
                                event="intake.reply.fallback_out_of_scope",
                                workflow=workflow,
                                conversation_summary=cursor_summary,
                                reply_text=reply_text,
                            )
                    if reply_action == "send_reply" and reply_text:
                        self._emit_observability(
                            event="intake.apply.start",
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            booking_action="ignore",
                            reply_action=reply_action,
                            ready_to_save=False,
                        )
                        self._emit_observability(
                            event="intake.reply.start",
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            booking_id="",
                            reply_text=reply_text,
                        )
                        reply_error = await self._send_source_reply(
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            reply_text=reply_text,
                        )
                        if reply_error is not None:
                            errors.append(f"{conversation_id}: {reply_error}")
                            self._emit_observability(
                                event="intake.reply.error",
                                workflow=workflow,
                                conversation_summary=cursor_summary,
                                booking_id="",
                                error=reply_error,
                            )
                            self._emit_observability(
                                event="intake.conversation.error",
                                workflow=workflow,
                                conversation_summary=cursor_summary,
                                phase="reply_execution",
                                error=reply_error,
                            )
                            continue
                        self._emit_observability(
                            event="intake.reply.ok",
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            booking_id="",
                        )
                        self._emit_observability(
                            event="intake.apply.ok",
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            status="ignored",
                            booking_action="ignore",
                            reply_action=reply_action,
                            ready_to_save=False,
                        )
                        self._set_cursor(
                            workflow_id=str(workflow["workflow_id"]),
                            conversation_id=conversation_id,
                            latest_inbound_message_id=latest_inbound_id,
                            latest_inbound_message_time=latest_inbound_time,
                            conversation_updated_time=conversation_updated_time,
                            latest_outbound_message_id=latest_outbound_id,
                            agent_action_at=_utc_now_iso(),
                        )
                        result_items.append(
                            {
                                "conversation_id": conversation_id,
                                "matched": False,
                                "status": "ignored",
                                "replied": True,
                            }
                        )
                        self._emit_observability(
                            event="intake.conversation.complete",
                            workflow=workflow,
                            conversation_summary=cursor_summary,
                            matched=False,
                            status="ignored",
                            replied=True,
                        )
                        continue
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
                    self._emit_observability(
                        event="intake.conversation.complete",
                        workflow=workflow,
                        conversation_summary=cursor_summary,
                        matched=False,
                        status="ignored",
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
                    self._emit_observability(
                        event="intake.conversation.error",
                        workflow=workflow,
                        conversation_summary=cursor_summary,
                        phase="apply",
                        error=apply_error,
                    )
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
                self._emit_observability(
                    event="intake.conversation.complete",
                    workflow=workflow,
                    conversation_summary=cursor_summary,
                    matched=True,
                    status=str(applied.get("status", "") or "").strip(),
                    booking_id=str(applied.get("booking_id", "") or "").strip(),
                    saved_summary=saved_summary,
                )

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
            error = "agent runtime does not support intake workflow decisions"
            self._emit_observability(
                event="intake.decision.error",
                workflow=workflow,
                conversation_summary=conversation_summary,
                error=error,
            )
            return {}, error
        recent_messages = self._normalize_conversation_messages(
            workflow=workflow,
            conversation=conversation,
            recipient_id=str(conversation_summary.get("recipient_id", "") or "").strip() or None,
        )
        workflow_context = {
            "workflow_id": workflow.get("workflow_id"),
            "name": workflow.get("name"),
            "intent_description": workflow.get("intent_description"),
            "required_fields": workflow.get("required_fields"),
            "field_guidance": workflow.get("field_guidance"),
            "assistant_instructions": workflow.get("assistant_instructions", ""),
            "knowledge_file_ids": _unique_string_list(workflow.get("knowledge_file_ids")),
            "knowledge_files": self._knowledge_files_for_workflow(
                customer_id=str(workflow["customer_id"]),
                workflow=workflow,
            ),
            "sink_type": workflow.get("sink_type"),
            "sink_config": workflow.get("sink_config"),
            "channel": workflow.get("channel"),
            "provider": workflow.get("provider"),
        }
        self._emit_observability(
            event="intake.decision.start",
            workflow=workflow,
            conversation_summary=conversation_summary,
            active_booking_id=str((active_booking or {}).get("booking_id", "") or "").strip(),
            recent_completed_booking_id=str(
                (recent_completed_booking or {}).get("booking_id", "") or ""
            ).strip(),
            recent_message_count=len(recent_messages),
            execution_feedback_count=len(execution_feedback or []),
            execution_feedback=_safe_list(execution_feedback),
        )
        try:
            decision = await runtime.decide_intake_workflow(
                customer_id=str(workflow["customer_id"]),
                workflow=workflow_context,
                conversation={
                    "summary": conversation_summary,
                    "recent_messages": recent_messages,
                },
                active_booking=active_booking,
                recent_completed_booking=recent_completed_booking,
                execution_feedback=execution_feedback,
            )
        except Exception as exc:
            error = str(exc)
            self._emit_observability(
                event="intake.decision.error",
                workflow=workflow,
                conversation_summary=conversation_summary,
                error=error,
            )
            return {}, error
        if not isinstance(decision, dict) or not bool(decision.get("ok", False)):
            error = str(decision.get("error", "invalid intake workflow decision"))
            self._emit_observability(
                event="intake.decision.error",
                workflow=workflow,
                conversation_summary=conversation_summary,
                error=error,
                decision=_safe_dict(decision),
            )
            return {}, error
        self._emit_observability(
            event="intake.decision.ok",
            workflow=workflow,
            conversation_summary=conversation_summary,
            matches_workflow=bool(decision.get("matches_workflow")),
            confidence=decision.get("confidence"),
            booking_action=str(decision.get("booking_action", "") or "").strip().lower(),
            reply_action=str(decision.get("reply_action", "") or "").strip().lower(),
            ready_to_save=bool(decision.get("ready_to_save")),
            missing_fields=_unique_string_list(decision.get("missing_fields")),
            extracted_fields=_safe_dict(decision.get("extracted_fields")),
            save_payload=_safe_dict(decision.get("save_payload")),
            reason=str(decision.get("reason", "") or "").strip(),
        )
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
        ready_to_save = bool(decision.get("ready_to_save"))
        reply_action = str(decision.get("reply_action", "none") or "none").strip().lower()
        reply_text = str(decision.get("reply_text", "") or "").strip()
        self._emit_observability(
            event="intake.apply.start",
            workflow=workflow,
            conversation_summary=conversation_summary,
            booking_action=booking_action,
            reply_action=reply_action,
            ready_to_save=ready_to_save,
        )
        if booking_action not in {
            "ignore",
            "update_active",
            "edit_recent_completed",
            "create_new_booking",
        }:
            error = f"unsupported booking_action={booking_action}"
            self._emit_observability(
                event="intake.apply.error",
                workflow=workflow,
                conversation_summary=conversation_summary,
                phase="decision_validation",
                error=error,
                booking_action=booking_action,
            )
            return {}, error, self._build_recovery_feedback(
                phase="decision_validation",
                error=error,
                decision=decision,
            )
        if booking_action == "ignore":
            if reply_action == "send_reply":
                self._emit_observability(
                    event="intake.reply.start",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    booking_id="",
                    reply_text=reply_text,
                )
                reply_error = await self._send_source_reply(
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    reply_text=reply_text,
                )
                if reply_error is not None:
                    self._emit_observability(
                        event="intake.reply.error",
                        workflow=workflow,
                        conversation_summary=conversation_summary,
                        booking_id="",
                        error=reply_error,
                    )
                    self._emit_observability(
                        event="intake.apply.error",
                        workflow=workflow,
                        conversation_summary=conversation_summary,
                        phase="reply_execution",
                        error=reply_error,
                        booking_id="",
                    )
                    return {}, reply_error, self._build_recovery_feedback(
                        phase="reply_execution",
                        error=reply_error,
                        decision=decision,
                    )
                self._emit_observability(
                    event="intake.reply.ok",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    booking_id="",
                )
            self._emit_observability(
                event="intake.apply.ok",
                workflow=workflow,
                conversation_summary=conversation_summary,
                status="ignored",
                booking_action=booking_action,
                reply_action=reply_action,
                ready_to_save=ready_to_save,
            )
            ignored_result = {
                "conversation_id": str(conversation_summary.get("conversation_id", "") or ""),
                "matched": True,
                "status": "ignored",
            }
            if reply_action == "send_reply":
                ignored_result["replied"] = True
            return ignored_result, None, None

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
            error = "workflow decision did not resolve a booking target"
            self._emit_observability(
                event="intake.apply.error",
                workflow=workflow,
                conversation_summary=conversation_summary,
                phase="decision_validation",
                error=error,
                booking_action=booking_action,
            )
            return {}, error, self._build_recovery_feedback(
                phase="decision_validation",
                error=error,
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

        sink_ref = dict(_safe_dict(target_booking.get("sink_record_ref")))
        sink_status = str(target_booking.get("sink_write_status", "pending") or "pending").strip()
        saved_summary = ""
        sink_arguments = dict(_safe_dict(decision.get("sink_arguments")))
        if ready_to_save:
            save_payload = dict(extracted_fields)
            save_payload.update(_safe_dict(decision.get("save_payload")))
            missing = [
                field
                for field in _unique_string_list(workflow.get("required_fields"))
                if not str(save_payload.get(field, "") or "").strip()
            ]
            if missing:
                error = (
                    "decision marked ready_to_save but required fields are missing: "
                    + ", ".join(missing)
                )
                self._emit_observability(
                    event="intake.apply.error",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    phase="decision_validation",
                    error=error,
                    booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                    missing_fields=missing,
                )
                return {}, (
                    error
                ), self._build_recovery_feedback(
                    phase="decision_validation",
                    error=f"ready_to_save missing required fields: {', '.join(missing)}",
                    decision=decision,
                )
            self._emit_observability(
                event="intake.sink_write.start",
                workflow=workflow,
                conversation_summary=conversation_summary,
                booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                sink_type=str(workflow.get("sink_type", "") or "").strip(),
                payload=save_payload,
            )
            sink_result, sink_error = self._write_to_sink(
                workflow=workflow,
                booking=target_booking,
                payload=save_payload,
                sink_arguments=sink_arguments,
            )
            if sink_error is not None:
                target_booking["status"] = "active"
                target_booking["sink_write_status"] = "failed"
                target_booking["sink_record_ref"] = sink_ref
                self._upsert_booking(target_booking)
                self._emit_observability(
                    event="intake.sink_write.error",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                    sink_type=str(workflow.get("sink_type", "") or "").strip(),
                    error=sink_error,
                )
                self._emit_observability(
                    event="intake.apply.error",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    phase="sink_execution",
                    error=sink_error,
                    booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                )
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
            self._emit_observability(
                event="intake.sink_write.ok",
                workflow=workflow,
                conversation_summary=conversation_summary,
                booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                sink_type=str(workflow.get("sink_type", "") or "").strip(),
                sink_result=sink_ref,
            )
            saved_summary = self._build_saved_summary(
                workflow=workflow,
                booking=target_booking,
                conversation_summary=conversation_summary,
            )
            if self._should_enforce_completion_reply(workflow=workflow) and (
                reply_action != "send_reply" or not reply_text
            ):
                reply_action = "send_reply"
                reply_text = self._build_completion_confirmation_reply(
                    workflow=workflow,
                    booking=target_booking,
                    conversation_summary=conversation_summary,
                )
                self._emit_observability(
                    event="intake.reply.normalized_completion_confirmation",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                    reply_text=reply_text,
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
            self._emit_observability(
                event="intake.reply.start",
                workflow=workflow,
                conversation_summary=conversation_summary,
                booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                reply_text=reply_text,
            )
            reply_error = await self._send_source_reply(
                workflow=workflow,
                conversation_summary=conversation_summary,
                reply_text=reply_text,
            )
            if reply_error is not None:
                self._emit_observability(
                    event="intake.reply.error",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                    error=reply_error,
                )
                self._emit_observability(
                    event="intake.apply.error",
                    workflow=workflow,
                    conversation_summary=conversation_summary,
                    phase="reply_execution",
                    error=reply_error,
                    booking_id=str(target_booking.get("booking_id", "") or "").strip(),
                )
                return {}, reply_error, self._build_recovery_feedback(
                    phase="reply_execution",
                    error=reply_error,
                    decision=decision,
                )
            self._emit_observability(
                event="intake.reply.ok",
                workflow=workflow,
                conversation_summary=conversation_summary,
                booking_id=str(target_booking.get("booking_id", "") or "").strip(),
            )

        self._emit_observability(
            event="intake.apply.ok",
            workflow=workflow,
            conversation_summary=conversation_summary,
            status=str(target_booking.get("status", "") or "active"),
            booking_id=str(target_booking.get("booking_id", "") or "").strip(),
            sink_write_status=str(target_booking.get("sink_write_status", "") or "").strip(),
            booking_action=booking_action,
            reply_action=reply_action,
            ready_to_save=ready_to_save,
            saved_summary=saved_summary,
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
                "sink_arguments": _safe_dict(decision.get("sink_arguments")),
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

    async def _send_telegram_business_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None:
        if not reply_text:
            return "reply_action=send_reply requires non-empty reply_text"
        telegram_business = self._telegram_business
        if telegram_business is None:
            return "Telegram Business is not available"
        source_config = _safe_dict(workflow.get("source_config"))
        business_connection_id = str(source_config.get("business_connection_id", "") or "").strip()
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
        if not business_connection_id or not conversation_id:
            return "Telegram Business reply requires business_connection_id and conversation_id"
        latest_inbound = str(conversation_summary.get("latest_inbound_message_id", "") or "").strip()
        client = getattr(telegram_business, "client", None)
        if client is None:
            return "Telegram Business client is not available"
        try:
            sent = await client.send_message(
                chat_id=conversation_id,
                text=reply_text,
                parse_mode="HTML",
                business_connection_id=business_connection_id,
                reply_to_message_id=int(latest_inbound) if latest_inbound.isdigit() else None,
            )
        except Exception as exc:
            return f"failed to send Telegram Business reply: {exc}"
        if not sent:
            return "Telegram Business reply failed"
        result_message = {}
        if isinstance(sent, dict):
            candidate = sent.get("result")
            if isinstance(candidate, dict):
                result_message = dict(candidate)
        if result_message:
            result_message.setdefault("message_id", new_short_id("tgmsg"))
            result_message.setdefault("date", int(_utc_now().timestamp()))
            result_message.setdefault(
                "chat",
                {
                    "id": conversation_id,
                    "type": "private",
                },
            )
            result_message.setdefault("text", reply_text)
            result_message.setdefault("business_connection_id", business_connection_id)
            result_message.setdefault("sender_business_bot", {"id": "opentulpa"})
            with suppress(Exception):
                telegram_business.upsert_message(
                    business_connection_id=business_connection_id,
                    customer_id=str(workflow["customer_id"]),
                    message=result_message,
                )
        return None

    def _write_to_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        payload: dict[str, Any],
        sink_arguments: dict[str, Any] | None = None,
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
                sink_arguments=sink_arguments,
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
        sink_arguments: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        if self._composio is None or not bool(getattr(self._composio, "enabled", False)):
            return {}, "Composio is not available for sink execution"
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "")).strip().lower()
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        override_arguments = _safe_dict(sink_arguments)
        if sink_type == "google_sheets_composio":
            top_level_arguments = {
                key: sink_config.get(key)
                for key in ("spreadsheetId", "spreadsheet_id", "sheetName", "sheet_name")
                if key in sink_config
            }
            static_arguments = _normalize_google_sheets_arguments(
                {**top_level_arguments, **static_arguments}
            )
            override_arguments = _normalize_google_sheets_arguments(override_arguments)
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
            field_mapping = _normalize_google_sheets_field_mapping(
                field_mapping,
                payload_keys=set(enriched_payload.keys()),
            )
            key_source = "booking_id"
            key_header = str(field_mapping.get(key_source, "Booking ID") or "Booking ID").strip()
            headers = [key_header]
            row = [_sheet_cell_value(enriched_payload.get(key_source))]
            for source_key, header_name in field_mapping.items():
                safe_source = str(source_key or "").strip()
                safe_header = str(header_name or "").strip()
                if not safe_source or not safe_header or safe_source == key_source:
                    continue
                headers.append(safe_header)
                row.append(_sheet_cell_value(enriched_payload.get(safe_source)))
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
        arguments.update(override_arguments)
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
            contact = str(conversation_summary.get("recipient_id", "") or "").strip() or "customer_contact"
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

    def _should_enforce_completion_reply(self, *, workflow: dict[str, Any]) -> bool:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        return channel == "telegram_business_dm" and provider == "telegram_bot_api"

    def _build_completion_confirmation_reply(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> str:
        fields = _safe_dict(booking.get("extracted_fields"))
        context_text = " ".join(
            [
                str(workflow.get("assistant_instructions", "") or ""),
                str(workflow.get("intent_description", "") or ""),
                str(conversation_summary.get("latest_inbound_message_text_preview", "") or ""),
                " ".join(str(value or "") for value in fields.values()),
            ]
        )
        russian = any("\u0400" <= char <= "\u04ff" for char in context_text)

        def _first_value(*keys: str) -> str:
            for key in keys:
                value = str(fields.get(key, "") or "").strip()
                if value:
                    return value
            return ""

        service = _first_value("service_name", "service", "wash_type", "service_category")
        date = _first_value("desired_date", "date", "day")
        time_value = _first_value("desired_time", "time")
        price = _first_value("quoted_price", "price", "cost")

        if russian:
            parts = ["Готово, запись сохранена."]
            if service:
                parts.append(f"Услуга: {service}.")
            if date or time_value:
                parts.append(f"Дата и время: {' '.join([date, time_value]).strip()}.")
            if price:
                parts.append(f"Цена: {price}.")
            return " ".join(parts)[:1000]

        parts = ["Done, your booking is saved."]
        if service:
            parts.append(f"Service: {service}.")
        if date or time_value:
            parts.append(f"Date/time: {' '.join([date, time_value]).strip()}.")
        if price:
            parts.append(f"Price: {price}.")
        return " ".join(parts)[:1000]
