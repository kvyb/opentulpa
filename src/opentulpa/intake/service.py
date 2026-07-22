"""Persistent intake workflow storage and wake-time execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from collections.abc import Callable
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.ids import new_short_id
from opentulpa.intake.decision_applier import DecisionApplier
from opentulpa.intake.decision_maker import DecisionMaker
from opentulpa.intake.messaging_adapters import (
    AdapterRegistry,
    ConversationSummary,
    build_messaging_adapter_registry,
    messaging_adapter_context,
)
from opentulpa.intake.reply_policy import (
    looks_like_cyrillic as _looks_like_cyrillic,
)
from opentulpa.intake.sink_utils import (
    clean_mapping as _clean_mapping,
)
from opentulpa.intake.sink_utils import (
    google_sheets_top_level_arguments as _google_sheets_top_level_arguments,
)
from opentulpa.intake.sink_utils import (
    incoming_user_id as _incoming_user_id,
)
from opentulpa.intake.sink_utils import (
    incoming_username as _incoming_username,
)
from opentulpa.intake.sink_utils import (
    infer_operation_hint_from_tool_slug as _infer_operation_hint_from_tool_slug,
)
from opentulpa.intake.sink_utils import (
    infer_toolkit_from_tool_slug as _infer_toolkit_from_tool_slug,
)
from opentulpa.intake.sink_utils import (
    normalize_google_sheets_arguments as _normalize_google_sheets_arguments,
)
from opentulpa.intake.sink_utils import (
    normalize_google_sheets_field_mapping as _normalize_google_sheets_field_mapping,
)
from opentulpa.intake.sink_utils import (
    normalize_toolkit_slug as _normalize_toolkit_slug,
)
from opentulpa.intake.sink_utils import (
    sheet_cell_value as _sheet_cell_value,
)
from opentulpa.intake.sink_writer import SinkWriter, normalize_local_csv_path
from opentulpa.intake.store import IntakeWorkflowStore
from opentulpa.intake.workflow_boundaries import (
    DECISION_BOOKING_ACTIONS,
    BookingTargetResolution,
)
from opentulpa.intake.workflow_runner import WorkflowRunner
from opentulpa.intake.workflow_runtime import (
    normalize_source_config as _normalize_source_config,
)
from opentulpa.intake.workflow_runtime import (
    parse_datetime as _parse_datetime,
)
from opentulpa.intake.workflow_runtime import (
    safe_dict as _safe_dict,
)
from opentulpa.intake.workflow_runtime import (
    safe_list as _safe_list,
)
from opentulpa.intake.workflow_runtime import (
    unique_string_list as _unique_string_list,
)
from opentulpa.intake.workflow_runtime import (
    utc_now as _utc_now,
)
from opentulpa.interfaces.telegram.constants import NO_NOTIFY_TOKEN
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.specs import AgentSpecRef

_ALLOWED_CHANNELS = {"instagram_dm", "telegram_business_dm"}
_ALLOWED_PROVIDERS = {"composio", "telegram_bot_api"}
_ALLOWED_SINK_TYPES = {"google_sheets_composio", "local_csv", "generic_composio_write"}
_ALLOWED_REPLY_MODES = {"auto"}
_DEFAULT_SCHEDULE = "*/2 * * * *"
_MAX_LATEST_INBOUND_AGE = timedelta(minutes=1)
_SCHEDULED_INBOUND_AGE_GRACE = timedelta(minutes=1)
_DEFAULT_SCHEDULED_INBOUND_AGE = timedelta(minutes=5)
_MAX_SCHEDULED_INBOUND_AGE = timedelta(hours=1)
_MAX_TELEGRAM_BUSINESS_WEBHOOK_INBOUND_AGE = timedelta(hours=24)
_TELEGRAM_BUSINESS_WEBHOOK_DEBOUNCE_SECONDS = 1.5
_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS = 5.0
_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS = 3.0
_TELEGRAM_BUSINESS_SETTLED_EVENT_TYPE = "telegram_business_webhook_settled"
_UNANSWERED_CUSTOMER_BURST_WINDOW = timedelta(minutes=5)
_RECENT_MESSAGE_HISTORY_LIMIT = 80
_PENDING_RUN_POLL_SECONDS = 0.2
_PENDING_RUN_MAX_CONCURRENCY = 4
_BUSINESS_FACTS_MAX_KEYS = 32
_BUSINESS_FACTS_MAX_LIST_ITEMS = 20
_BUSINESS_FACTS_MAX_STRING_CHARS = 500
_BUSINESS_FACTS_MAX_JSON_CHARS = 12000
logger = logging.getLogger(__name__)
_PUBLIC_TELEGRAM_INTAKE_ERROR = (
    "Telegram Business intake could not process an update. Check server logs."
)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _is_older_than(value: Any, *, max_age: timedelta) -> bool:
    parsed = _parse_datetime(value)
    if parsed is None:
        return False
    return (_utc_now() - parsed) > max_age


def _scheduled_poll_interval(schedule: Any) -> timedelta | None:
    parts = str(schedule or "").strip().split()
    if len(parts) < 5:
        return None
    minute = parts[0].strip()
    if minute == "*":
        return timedelta(minutes=1)
    for prefix in ("*/", "0/"):
        if not minute.startswith(prefix):
            continue
        raw_step = minute[len(prefix) :].strip()
        if not raw_step.isdigit():
            return None
        step = int(raw_step)
        if step <= 0:
            return None
        return timedelta(minutes=step)
    return None


def _scheduled_inbound_max_age(workflow: dict[str, Any]) -> timedelta:
    interval = _scheduled_poll_interval(workflow.get("schedule"))
    if interval is None:
        return _DEFAULT_SCHEDULED_INBOUND_AGE
    return min(
        max(interval + _SCHEDULED_INBOUND_AGE_GRACE, _DEFAULT_SCHEDULED_INBOUND_AGE),
        _MAX_SCHEDULED_INBOUND_AGE,
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _compact_business_fact_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for raw_key, raw_value in list(value.items())[:_BUSINESS_FACTS_MAX_KEYS]:
            key = str(raw_key or "").strip()
            if not key:
                continue
            out[key] = _compact_business_fact_value(raw_value)
        return out
    if isinstance(value, list):
        return [
            _compact_business_fact_value(item)
            for item in value[:_BUSINESS_FACTS_MAX_LIST_ITEMS]
        ]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    text = str(value or "").strip()
    if len(text) <= _BUSINESS_FACTS_MAX_STRING_CHARS:
        return text
    return text[: _BUSINESS_FACTS_MAX_STRING_CHARS - 3].rstrip() + "..."


def _normalize_business_facts(value: Any) -> dict[str, Any]:
    facts = _safe_dict(value)
    if len(_json_dumps(facts)) > _BUSINESS_FACTS_MAX_JSON_CHARS:
        return {
            "summary": (
                "Owner-provided business facts were too large to store inline. "
                "Keep large source material in knowledge files."
            )
        }
    compact = _compact_business_fact_value(facts)
    if not isinstance(compact, dict):
        return {}
    rendered = _json_dumps(compact)
    if len(rendered) <= _BUSINESS_FACTS_MAX_JSON_CHARS:
        return compact
    return {
        "summary": (
            "Owner-provided business facts were too large to store inline. "
            "Keep large source material in knowledge files."
        )
    }


def _unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
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


def _normalize_optional_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


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
        sink_root: Path | None = None,
        idempotency: IdempotencyStore | None = None,
        composio: Any | None = None,
        sink_composio: Any | None = None,
        telegram_business: Any | None = None,
        file_vault: FileVaultService | None = None,
        knowledge_service: Any | None = None,
        get_intake_agent: Any | None = None,
        resolve_agent_spec: Callable[[str, str], AgentSpecRef] | None = None,
    ) -> None:
        self._db_path = db_path.resolve()
        self._store = IntakeWorkflowStore(db_path=self._db_path)
        self._project_root = project_root.resolve()
        self._sink_root = (
            sink_root.expanduser()
            if sink_root is not None
            else self._project_root / ".opentulpa" / "intake_sinks"
        )
        self._idempotency = idempotency or IdempotencyStore(
            self._db_path.with_name(f"{self._db_path.stem}_external_effects.db")
        )
        self._composio = composio
        self._sink_composio = sink_composio
        self._telegram_business = telegram_business
        self._messaging_adapters: AdapterRegistry = build_messaging_adapter_registry(
            composio=composio,
            telegram_business=telegram_business,
        )
        self._workflow_runner = WorkflowRunner(
            self,
            is_older_than=_is_older_than,
            utc_now_iso=_utc_now_iso,
        )
        self._decision_applier = DecisionApplier(
            self,
            utc_now=_utc_now,
            utc_now_iso=_utc_now_iso,
        )
        self._decision_maker = DecisionMaker(self)
        self._sink_writer = SinkWriter(
            sink_root=self._sink_root,
            composio=sink_composio,
            idempotency=self._idempotency,
        )
        self._file_vault = file_vault
        self._knowledge_service = knowledge_service
        self._get_intake_agent = get_intake_agent
        self._resolve_agent_spec = resolve_agent_spec
        self._conversation_locks_guard = threading.Lock()
        self._conversation_locks: dict[str, asyncio.Lock] = {}
        self._pending_worker_task: asyncio.Task[None] | None = None
        self._pending_worker_stop: asyncio.Event | None = None
        self._pending_run_tasks: set[asyncio.Task[None]] = set()
        self._store.init_db(normalize_sink_config=self._normalize_sink_config)

    def _conn(self) -> Any:
        return self._store.conn()

    async def start(self) -> None:
        if self._pending_worker_task is not None and not self._pending_worker_task.done():
            return
        self._recover_interrupted_pending_runs()
        self._pending_worker_stop = asyncio.Event()
        self._pending_worker_task = asyncio.create_task(
            self._pending_run_worker_loop(),
            name="opentulpa-intake-pending-runs",
        )

    async def shutdown(self) -> None:
        task = self._pending_worker_task
        stop_event = self._pending_worker_stop
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        active_tasks = list(self._pending_run_tasks)
        for active_task in active_tasks:
            active_task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        self._pending_run_tasks.clear()
        self._pending_worker_task = None
        self._pending_worker_stop = None

    def _agent_for_observability(self) -> Any | None:
        getter = self._get_intake_agent
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
            "channel": str(workflow.get("channel", "") or "").strip() or "instagram_dm",
            "provider": str(workflow.get("provider", "") or "").strip() or "composio",
            "sink_type": str(workflow.get("sink_type", "") or "").strip(),
            "conversation_id": str(conversation_summary.get("conversation_id", "") or "").strip(),
            "recipient_id": str(conversation_summary.get("recipient_id", "") or "").strip(),
            "latest_inbound_message_id": str(
                conversation_summary.get("latest_inbound_message_id", "") or ""
            ).strip(),
            "latest_inbound_sender_id": _incoming_user_id(conversation_summary),
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
        agent = self._agent_for_observability()
        fields = self._intake_observability_fields(
            workflow=workflow,
            conversation_summary=conversation_summary,
            **extra,
        )
        customer_id = str(fields.pop("customer_id", "") or "").strip() or None
        record = getattr(agent, "record_observability_event", None)
        if callable(record):
            record(
                event=event,
                customer_id=customer_id,
                **fields,
            )
            return
        log_event = getattr(agent, "log_behavior_event", None)
        if callable(log_event):
            log_event(event=event, **fields)

    @staticmethod
    def _source_platform(workflow: dict[str, Any]) -> str:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        if channel == "telegram_business_dm" and provider == "telegram_bot_api":
            return "telegram_business"
        if channel == "instagram_dm" and provider == "composio":
            return "instagram"
        return channel or provider or "unknown"

    def _enrich_conversation_summary(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> dict[str, Any]:
        summary = dict(_safe_dict(conversation_summary))
        incoming_user_id = _incoming_user_id(summary)
        username = _incoming_username(summary)
        if incoming_user_id:
            summary["incoming_user_id"] = incoming_user_id
        if username:
            summary["username"] = username
        summary.setdefault("platform", self._source_platform(workflow))
        return summary

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

    @staticmethod
    def _is_telegram_business_webhook_event(event_type: str) -> bool:
        return str(event_type or "").strip() in {
            "telegram_business_webhook",
            _TELEGRAM_BUSINESS_SETTLED_EVENT_TYPE,
        }

    @classmethod
    def _latest_inbound_max_age_for_event(
        cls,
        *,
        event_type: str,
        workflow: dict[str, Any],
    ) -> timedelta:
        if cls._is_telegram_business_webhook_event(event_type):
            return _MAX_TELEGRAM_BUSINESS_WEBHOOK_INBOUND_AGE
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        if channel == "instagram_dm" and provider == "composio":
            return _scheduled_inbound_max_age(workflow)
        return _MAX_LATEST_INBOUND_AGE

    @staticmethod
    def _uses_latest_inbound_stale_guard(
        *,
        workflow: dict[str, Any],
        event_type: str,
        force: bool,
    ) -> bool:
        if force:
            return False
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        if (
            channel == "telegram_business_dm"
            and provider == "telegram_bot_api"
            and IntakeWorkflowService._is_telegram_business_webhook_event(event_type)
        ):
            return True
        return channel == "instagram_dm" and provider == "composio"

    @staticmethod
    def _refreshes_stale_decision_inline(*, workflow: dict[str, Any]) -> bool:
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        return channel == "instagram_dm" and provider == "composio"

    @staticmethod
    def _pending_due_at(delay_seconds: float) -> str:
        return (_utc_now() + timedelta(seconds=max(0.0, float(delay_seconds)))).isoformat()

    @staticmethod
    def _latest_inbound_changed(
        *,
        decided_summary: dict[str, Any],
        latest_summary: dict[str, Any],
    ) -> bool:
        decided_id = str(decided_summary.get("latest_inbound_message_id", "") or "").strip()
        latest_id = str(latest_summary.get("latest_inbound_message_id", "") or "").strip()
        if latest_id and decided_id and latest_id != decided_id:
            return True
        if latest_id and not decided_id:
            return True
        decided_time = _parse_datetime(decided_summary.get("latest_inbound_message_created_time"))
        latest_time = _parse_datetime(latest_summary.get("latest_inbound_message_created_time"))
        return bool(decided_time and latest_time and latest_time > decided_time)

    def _recover_interrupted_pending_runs(self) -> None:
        self._store.recover_interrupted_pending_runs()

    def _queue_pending_run(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        event_type: str,
        owner_chat_id: str = "",
        delay_seconds: float = _TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS,
        last_inbound_message_id: str = "",
    ) -> dict[str, Any]:
        safe_workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        safe_customer_id = str(workflow.get("customer_id", "") or "").strip()
        safe_conversation_id = str(conversation_id or "").strip()
        if not safe_workflow_id or not safe_customer_id or not safe_conversation_id:
            return {"ok": False, "queued": False, "summary": "pending run requires workflow and conversation ids"}
        due_at = self._pending_due_at(delay_seconds)
        safe_event_type = str(event_type or "").strip() or _TELEGRAM_BUSINESS_SETTLED_EVENT_TYPE
        queued = self._store.queue_pending_run(
            workflow=workflow,
            conversation_id=safe_conversation_id,
            event_type=safe_event_type,
            owner_chat_id=owner_chat_id,
            due_at=due_at,
            last_inbound_message_id=last_inbound_message_id,
        )
        queued["summary"] = NO_NOTIFY_TOKEN
        return queued

    async def enqueue_telegram_business_workflow_run(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str,
        owner_chat_id: str = "",
        event_type: str = "telegram_business_webhook",
    ) -> dict[str, Any]:
        workflow = self.get_workflow(customer_id=customer_id, workflow_id=workflow_id)
        if workflow is None:
            return {
                "ok": False,
                "queued": False,
                "workflow_id": workflow_id,
                "summary": f"Intake workflow {workflow_id} was not found.",
            }
        if not bool(workflow.get("enabled")):
            return {
                "ok": True,
                "queued": False,
                "workflow_id": workflow_id,
                "summary": NO_NOTIFY_TOKEN,
                "reason": "workflow_disabled",
            }
        latest_inbound_id = ""
        summary, refresh_error = self._reload_conversation_summary(
            workflow=workflow,
            conversation_id=str(conversation_id or "").strip(),
            fallback={},
        )
        if refresh_error:
            return {
                "ok": False,
                "queued": False,
                "workflow_id": workflow_id,
                "summary": refresh_error,
            }
        latest_inbound_id = str(summary.get("latest_inbound_message_id", "") or "").strip()
        return self._queue_pending_run(
            workflow=workflow,
            conversation_id=conversation_id,
            event_type=_TELEGRAM_BUSINESS_SETTLED_EVENT_TYPE
            if self._is_telegram_business_webhook_event(event_type)
            else event_type,
            owner_chat_id=owner_chat_id,
            delay_seconds=_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS,
            last_inbound_message_id=latest_inbound_id,
        )

    def _claim_due_pending_runs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        return self._store.claim_due_pending_runs(limit=limit)

    async def drain_due_pending_runs(self, *, limit: int = 10) -> int:
        rows = self._claim_due_pending_runs(limit=limit)
        if not rows:
            return 0
        semaphore = asyncio.Semaphore(max(1, int(_PENDING_RUN_MAX_CONCURRENCY)))

        async def _run(row: dict[str, Any]) -> None:
            async with semaphore:
                await self._run_pending_row(row)

        results = await asyncio.gather(*(_run(row) for row in rows), return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self._log_pending_run_failure("intake pending run failed during drain", result)
        return len(rows)

    async def _schedule_due_pending_runs(self, *, limit: int = 10) -> int:
        max_concurrency = max(1, int(_PENDING_RUN_MAX_CONCURRENCY))
        capacity = max(0, min(int(limit or 10), max_concurrency - len(self._pending_run_tasks)))
        if capacity <= 0:
            return 0
        rows = self._claim_due_pending_runs(limit=capacity)
        for row in rows:
            task = asyncio.create_task(
                self._run_pending_row(row),
                name=f"opentulpa-intake-pending-run-{row.get('workflow_id')}-{row.get('conversation_id')}",
            )
            self._pending_run_tasks.add(task)
            task.add_done_callback(self._on_pending_run_task_done)
        return len(rows)

    def _on_pending_run_task_done(self, task: asyncio.Task[None]) -> None:
        self._pending_run_tasks.discard(task)
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            self._log_pending_run_failure("intake pending run failed during worker task", exc)

    @staticmethod
    def _log_pending_run_failure(message: str, exc: BaseException) -> None:
        logger.error(message, exc_info=(type(exc), exc, exc.__traceback__))

    async def _pending_run_worker_loop(self) -> None:
        while True:
            stop_event = self._pending_worker_stop
            if stop_event is not None and stop_event.is_set():
                return
            try:
                await self._schedule_due_pending_runs(limit=10)
            except Exception:
                logger.exception("intake pending run worker failed while claiming due work")
            stop_event = self._pending_worker_stop
            if stop_event is None:
                await asyncio.sleep(_PENDING_RUN_POLL_SECONDS)
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=_PENDING_RUN_POLL_SECONDS)
            except TimeoutError:
                continue

    async def _run_pending_row(self, row: dict[str, Any]) -> None:
        workflow_id = str(row.get("workflow_id", "") or "").strip()
        conversation_id = str(row.get("conversation_id", "") or "").strip()
        customer_id = str(row.get("customer_id", "") or "").strip()
        owner_chat_id = str(row.get("owner_chat_id", "") or "").strip()
        generation = int(row.get("generation") or 0)
        result: dict[str, Any]
        try:
            result = await self.run_workflow(
                customer_id=customer_id,
                workflow_id=workflow_id,
                event_type=_TELEGRAM_BUSINESS_SETTLED_EVENT_TYPE,
            )
        except Exception as exc:
            self._log_pending_run_failure("intake pending run raised", exc)
            result = {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": _PUBLIC_TELEGRAM_INTAKE_ERROR,
            }
        if (
            not bool(result.get("ok", False))
            and owner_chat_id
            and str(result.get("summary", "") or "").strip()
            and str(result.get("summary", "") or "").strip() != NO_NOTIFY_TOKEN
        ):
            logger.error(
                "intake pending run failed: %r",
                result,
                extra={
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                },
            )
            await self._notify_pending_run_owner(
                owner_chat_id=owner_chat_id,
                summary=_PUBLIC_TELEGRAM_INTAKE_ERROR,
            )
        self._finish_pending_run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            generation=generation,
        )

    def _pending_run_is_still_running(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        generation: int,
    ) -> bool:
        return self._store.pending_run_is_still_running(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            generation=generation,
        )

    async def _notify_pending_run_owner(self, *, owner_chat_id: str, summary: str) -> None:
        del summary
        telegram_business = self._telegram_business
        client = getattr(telegram_business, "client", None)
        if client is None:
            return
        with suppress(Exception):
            await client.send_message(
                chat_id=owner_chat_id,
                text=_PUBLIC_TELEGRAM_INTAKE_ERROR,
                parse_mode="HTML",
            )

    def _finish_pending_run(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        generation: int,
    ) -> None:
        self._store.finish_pending_run(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            generation=generation,
            stale_requeue_seconds=_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS,
        )

    def _reload_conversation_summary(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        fallback: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        items, source_error, _source_warnings = self._load_source_items(workflow=workflow)
        if source_error is not None:
            return fallback, source_error
        for item in items:
            summary = _safe_dict(item)
            if str(summary.get("conversation_id", "") or "").strip() == conversation_id:
                return summary, None
        return fallback, None

    def _conversation_became_stale(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        decided_summary: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str | None]:
        latest_summary, refresh_error = self._reload_conversation_summary(
            workflow=workflow,
            conversation_id=conversation_id,
            fallback=decided_summary,
        )
        if refresh_error is not None:
            return False, decided_summary, refresh_error
        return (
            self._latest_inbound_changed(
                decided_summary=decided_summary,
                latest_summary=latest_summary,
            ),
            latest_summary,
            None,
        )

    def _requeue_stale_telegram_business_run(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        latest_summary: dict[str, Any],
    ) -> None:
        self._queue_pending_run(
            workflow=workflow,
            conversation_id=conversation_id,
            event_type=_TELEGRAM_BUSINESS_SETTLED_EVENT_TYPE,
            delay_seconds=_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS,
            last_inbound_message_id=str(
                latest_summary.get("latest_inbound_message_id", "") or ""
            ).strip(),
        )

    def _requeue_if_conversation_stale(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        matched: bool,
    ) -> dict[str, Any] | None:
        stale, latest_summary, stale_error = self._conversation_became_stale(
            workflow=workflow,
            conversation_id=conversation_id,
            decided_summary=conversation_summary,
        )
        if stale_error:
            self._emit_observability(
                event="intake.conversation.error",
                workflow=workflow,
                conversation_summary=conversation_summary,
                phase="stale_check",
                error=stale_error,
            )
        if not stale:
            return None
        channel = str(workflow.get("channel", "") or "").strip().lower()
        provider = str(workflow.get("provider", "") or "").strip().lower()
        requeued = channel == "telegram_business_dm" and provider == "telegram_bot_api"
        if requeued:
            self._requeue_stale_telegram_business_run(
                workflow=workflow,
                conversation_id=conversation_id,
                latest_summary=latest_summary,
            )
        status = "stale_requeued" if requeued else "stale_waiting_for_next_poll"
        self._emit_observability(
            event="intake.conversation.stale",
            workflow=workflow,
            conversation_summary=conversation_summary,
            latest_inbound_message_id=str(
                latest_summary.get("latest_inbound_message_id", "") or ""
            ).strip(),
            requeued=requeued,
        )
        return {
            "conversation_id": conversation_id,
            "matched": bool(matched),
            "status": status,
            "replied": False,
        }

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
        business_facts: dict[str, Any] | None,
        knowledge_file_ids: list[str],
        sink_type: str,
        sink_config: dict[str, Any] | None,
        schedule: str,
        notify_user: bool,
        enabled: bool,
        reply_mode: str,
        existing: dict[str, Any] | None,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_name = str(name or "").strip()
        safe_channel = str(channel or "instagram_dm").strip().lower() or "instagram_dm"
        safe_provider = str(provider or "composio").strip().lower() or "composio"
        safe_intent = str(intent_description or "").strip()
        safe_schedule = str(schedule or _DEFAULT_SCHEDULE).strip() or _DEFAULT_SCHEDULE
        safe_required_fields = _unique_string_list(required_fields)
        safe_source_config = _normalize_source_config(source_config)
        safe_field_guidance = _safe_dict(field_guidance)
        safe_assistant_instructions = str(assistant_instructions or "").strip()
        safe_business_facts = _normalize_business_facts(business_facts)
        safe_knowledge_file_ids = _unique_string_list(knowledge_file_ids)
        safe_reply_mode = "auto"
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
            safe_reply_mode = "auto"
            safe_source_config = self._resolve_telegram_business_source_config(
                customer_id=safe_customer,
                source_config=safe_source_config,
            )
            safe_source_config = _normalize_source_config(safe_source_config)
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
            "business_facts": safe_business_facts,
            "knowledge_file_ids": safe_knowledge_file_ids,
            "sink_type": safe_sink_type,
            "sink_config": safe_sink_config,
            "schedule": safe_schedule,
            "notify_user": bool(notify_user),
            "enabled": bool(enabled),
            "routine_id": "",
            "reply_mode": safe_reply_mode,
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
            file_path = normalize_local_csv_path(
                requested_path,
                workflow_id=workflow_id,
            )
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
            static_arguments = _normalize_google_sheets_arguments(
                {**_google_sheets_top_level_arguments(safe_config), **static_arguments}
            )
            if validate_target and not str(static_arguments.get("spreadsheetId", "") or "").strip():
                raise ValueError(
                    "google_sheets_composio requires sink_config.static_arguments.spreadsheetId"
                )
            static_arguments = self._resolve_google_sheets_sheet_name_for_sink(
                customer_id=customer_id,
                static_arguments=static_arguments,
                connected_account_id=connected_account_id or None,
                validate_target=validate_target,
            )
        else:
            toolkit = toolkit or _infer_toolkit_from_tool_slug(legacy_tool_slug)
            operation_hint = operation_hint or _infer_operation_hint_from_tool_slug(legacy_tool_slug)
            if not toolkit:
                raise ValueError("sink_config.toolkit is required for generic_composio_write")
            if not operation_hint:
                raise ValueError(
                    "sink_config.operation_hint is required for generic_composio_write"
                )
            if "booking_id" not in field_mapping.values():
                raise ValueError(
                    "generic_composio_write requires a field_mapping target sourced from booking_id"
                )
            operation_tokens = operation_hint.replace("_", " ").replace("-", " ").split()
            if validate_target and not any(
                token in operation_tokens for token in ("upsert", "update")
            ):
                raise ValueError(
                    "generic_composio_write operation_hint must describe an upsert or update"
                )
        normalized = {
            "toolkit": toolkit,
            "operation_hint": operation_hint,
            "field_mapping": field_mapping,
            "static_arguments": static_arguments,
        }
        if connected_account_id:
            normalized["connected_account_id"] = connected_account_id
        if legacy_tool_slug:
            normalized["tool_slug"] = legacy_tool_slug
        if validate_target:
            binding = self._validate_sink_target(
                customer_id=customer_id,
                sink_type=sink_type,
                sink_config=normalized,
            )
            normalized["tool_slug"] = str(binding.tool_slug)
            normalized["connected_account_id"] = str(binding.connected_account_id)
        return normalized

    def _validate_sink_target(
        self,
        *,
        customer_id: str,
        sink_type: str,
        sink_config: dict[str, Any],
    ) -> Any:
        configured_slug = str(sink_config.get("tool_slug") or "").strip() or None
        if self._sink_composio is None or not bool(
            getattr(self._sink_composio, "enabled", False)
        ):
            raise ValueError("tenant-aware Composio sink validation is unavailable")
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        if not toolkit:
            raise ValueError("sink_config.toolkit is required for composio sink types")
        operation_hint = str(sink_config.get("operation_hint", "") or "").strip().lower()
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        required_arguments = set(static_arguments)
        if sink_type == "google_sheets_composio":
            required_arguments.update({"headers", "rows", "keyColumn"})
        else:
            required_arguments.update(field_mapping)
        try:
            return self._sink_composio.resolve_sink_binding(
                tenant_id=customer_id,
                toolkit=toolkit,
                connected_account_id=str(
                    sink_config.get("connected_account_id") or ""
                ).strip()
                or None,
                tool_slug=configured_slug,
                operation_hint=operation_hint,
                required_arguments=required_arguments,
                allow_discovery=configured_slug is None,
            )
        except Exception as exc:
            raise ValueError(f"unable to resolve sink tool from toolkit={toolkit}: {exc}") from exc

    def _resolve_google_sheets_sheet_name_for_sink(
        self,
        *,
        customer_id: str,
        static_arguments: dict[str, Any],
        connected_account_id: str | None,
        validate_target: bool,
    ) -> dict[str, Any]:
        return self._sink_writer.resolve_google_sheets_sheet_name_for_sink(
            customer_id=customer_id,
            static_arguments=static_arguments,
            connected_account_id=connected_account_id,
            validate_target=validate_target,
        )

    @staticmethod
    def _workflow_to_upsert_draft(workflow: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": str(workflow.get("name", "") or ""),
            "channel": str(workflow.get("channel", "instagram_dm") or "instagram_dm"),
            "provider": str(workflow.get("provider", "composio") or "composio"),
            "source_config": _safe_dict(workflow.get("source_config")),
            "intent_description": str(workflow.get("intent_description", "") or ""),
            "required_fields": [
                str(item or "").strip()
                for item in _safe_list(workflow.get("required_fields"))
                if str(item or "").strip()
            ],
            "field_guidance": _safe_dict(workflow.get("field_guidance")),
            "assistant_instructions": str(workflow.get("assistant_instructions", "") or ""),
            "business_facts": _safe_dict(workflow.get("business_facts")),
            "knowledge_file_ids": [
                str(item or "").strip()
                for item in _safe_list(workflow.get("knowledge_file_ids"))
                if str(item or "").strip()
            ],
            "sink_type": str(workflow.get("sink_type", "") or ""),
            "sink_config": _safe_dict(workflow.get("sink_config")),
            "schedule": str(workflow.get("schedule", _DEFAULT_SCHEDULE) or _DEFAULT_SCHEDULE),
            "notify_user": bool(workflow.get("notify_user", True)),
            "enabled": bool(workflow.get("enabled", True)),
            "reply_mode": "auto",
        }

    def preflight_workflow_payload(
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
        business_facts: dict[str, Any] | None = None,
        knowledge_file_ids: list[str] | None = None,
        sink_type: str,
        sink_config: dict[str, Any] | None = None,
        schedule: str = _DEFAULT_SCHEDULE,
        notify_user: bool = True,
        enabled: bool = True,
        reply_mode: str = "auto",
    ) -> dict[str, Any]:
        try:
            normalized = self._normalize_workflow_payload(
                workflow_id=workflow_id,
                customer_id=customer_id,
                name=name,
                channel=channel,
                provider=provider,
                source_config=source_config,
                intent_description=intent_description,
                required_fields=required_fields,
                field_guidance=field_guidance,
                assistant_instructions=assistant_instructions,
                business_facts=business_facts,
                knowledge_file_ids=knowledge_file_ids or [],
                sink_type=sink_type,
                sink_config=sink_config,
                schedule=schedule,
                notify_user=notify_user,
                enabled=enabled,
                reply_mode=reply_mode,
                existing=None,
            )
        except Exception as exc:
            error = str(exc)
            return {
                "ok": False,
                "status": "needs_clarification",
                "errors": [error],
                "warnings": [],
                "follow_up_questions": [self._preflight_follow_up_for_error(error)],
            }

        warnings: list[str] = []
        safe_channel = str(normalized.get("channel", "") or "").strip().lower()
        safe_provider = str(normalized.get("provider", "") or "").strip().lower()
        if safe_channel == "instagram_dm" and safe_provider == "composio":
            warnings.append(
                "Instagram DM intake uses scheduled Composio polling, not webhook delivery; "
                "new messages are handled on the configured schedule and only for conversations "
                "Composio can read."
            )
        dry_run = self._build_sink_dry_run_preview(normalized, warnings=warnings)
        return {
            "ok": True,
            "status": "ready",
            "errors": [],
            "warnings": warnings,
            "follow_up_questions": [],
            "normalized_draft": self._workflow_to_upsert_draft(normalized),
            "sink_preflight": {
                "sink_type": str(normalized.get("sink_type", "") or ""),
                "dry_run": dry_run,
            },
        }

    @staticmethod
    def _preflight_follow_up_for_error(error: str) -> str:
        text = str(error or "").strip()
        lowered = text.lower()
        if "multiple sheets" in lowered:
            return "Which Google Sheets tab should this workflow write to?"
        if "sheetname" in lowered or "worksheet" in lowered:
            return "Which Google Sheets tab should this workflow write to?"
        if "spreadsheetid" in lowered:
            return "Please provide the Google Sheet URL or spreadsheet ID."
        if "business_connection_id" in lowered or "telegram business" in lowered:
            return "Please connect Telegram Business or choose the exact business connection for this workflow."
        if "required_fields" in lowered:
            return "Which fields must be collected before saving a completed lead?"
        if "intent_description" in lowered:
            return "What inbound intent should this workflow handle?"
        if "sink_config.field_mapping" in lowered or "field_mapping" in lowered:
            return "How should workflow fields map to the output sink columns or arguments?"
        return f"Please clarify the workflow setup issue: {text}"

    def _build_sink_dry_run_preview(
        self,
        workflow: dict[str, Any],
        *,
        warnings: list[str],
    ) -> dict[str, Any]:
        sink_type = str(workflow.get("sink_type", "") or "").strip().lower()
        sink_config = _safe_dict(workflow.get("sink_config"))
        sample_payload = self._sample_sink_payload(workflow)
        if sink_type == "local_csv":
            relative_path = str(sink_config.get("file_path", "") or "").strip()
            headers = [
                "booking_id",
                "workflow_id",
                "workflow_name",
                "conversation_id",
                "customer_id",
                "status",
                "completed_at",
                *sample_payload.keys(),
            ]
            return {
                "mode": "non_destructive",
                "will_execute": False,
                "target": {"file_path": relative_path},
                "headers_preview": _unique_strings(headers),
                "sample_payload": sample_payload,
            }
        if sink_type in {"google_sheets_composio", "generic_composio_write"}:
            return self._build_composio_sink_dry_run_preview(
                workflow,
                sample_payload=sample_payload,
                warnings=warnings,
            )
        return {"mode": "non_destructive", "will_execute": False, "unsupported_sink_type": sink_type}

    @staticmethod
    def _sample_sink_payload(workflow: dict[str, Any]) -> dict[str, str]:
        sink_config = _safe_dict(workflow.get("sink_config"))
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        required_fields = [
            str(item or "").strip()
            for item in _safe_list(workflow.get("required_fields"))
            if str(item or "").strip()
        ]
        source_fields = [*required_fields]
        sink_type = str(workflow.get("sink_type", "") or "").strip().lower()
        if sink_type == "google_sheets_composio":
            source_fields.extend(field_mapping.keys())
            source_fields.extend(field_mapping.values())
        else:
            source_fields.extend(field_mapping.values())
        payload: dict[str, str] = {}
        for field in _unique_strings(source_fields):
            if field in {
                "booking_id",
                "workflow_id",
                "conversation_id",
                "customer_id",
            }:
                continue
            payload[field] = f"sample_{field}"
        return payload

    def _build_composio_sink_dry_run_preview(
        self,
        workflow: dict[str, Any],
        *,
        sample_payload: dict[str, str],
        warnings: list[str],
    ) -> dict[str, Any]:
        sink_config = _safe_dict(workflow.get("sink_config"))
        sink_type = str(workflow.get("sink_type", "") or "").strip().lower()
        toolkit = _normalize_toolkit_slug(sink_config.get("toolkit"))
        connected_account_id = str(sink_config.get("connected_account_id", "") or "").strip() or None
        tool_slug = str(sink_config.get("tool_slug") or "").strip()
        if self._sink_composio is None or not bool(
            getattr(self._sink_composio, "enabled", False)
        ):
            warnings.append("Composio is not configured, so the write tool could not be validated.")
        else:
            try:
                binding = self._validate_sink_target(
                    customer_id=str(workflow.get("customer_id") or ""),
                    sink_type=sink_type,
                    sink_config=sink_config,
                )
                tool_slug = str(binding.tool_slug)
                connected_account_id = str(binding.connected_account_id)
            except Exception as exc:
                warnings.append(f"Could not validate Composio write tool during dry run: {exc}")
        enriched_payload = {
            **sample_payload,
            "booking_id": "sample_booking_id",
            "workflow_id": str(workflow.get("workflow_id", "") or "sample_workflow_id"),
            "conversation_id": "sample_conversation_id",
            "customer_id": str(workflow.get("customer_id", "") or "sample_customer_id"),
            "incoming_user_id": "sample_incoming_user_id",
            "latest_inbound_sender_id": "sample_incoming_user_id",
        }
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        field_mapping = _clean_mapping(sink_config.get("field_mapping"))
        if sink_type == "google_sheets_composio":
            static_arguments = _normalize_google_sheets_arguments(static_arguments)
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
        return {
            "mode": "non_destructive",
            "will_execute": False,
            "toolkit": toolkit,
            "tool_slug": tool_slug,
            "connected_account_id": connected_account_id,
            "arguments_preview": arguments,
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
        business_facts: dict[str, Any] | None = None,
        knowledge_file_ids: list[str] | None = None,
        sink_type: str,
        sink_config: dict[str, Any] | None = None,
        schedule: str = _DEFAULT_SCHEDULE,
        notify_user: bool = True,
        enabled: bool = True,
        reply_mode: str = "auto",
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
                if safe_workflow_id != existing_telegram_workflow_id:
                    raise ValueError(
                        "telegram_business_dm supports only one active workflow per customer"
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
            business_facts=business_facts,
            knowledge_file_ids=knowledge_file_ids or [],
            sink_type=sink_type,
            sink_config=sink_config,
            schedule=schedule,
            notify_user=notify_user,
            enabled=enabled,
            reply_mode=reply_mode,
            existing=existing,
        )
        now = _utc_now_iso()
        created_at = str(existing.get("created_at", "")).strip() if existing else ""
        if not created_at:
            created_at = now
        revision = self._store.upsert_workflow_record(
            workflow=workflow,
            created_at=created_at,
            updated_at=now,
        )
        workflow["revision"] = revision
        self._index_workflow_knowledge(workflow)
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
        return self._store.list_workflows(
            customer_id=customer_id,
            include_disabled=include_disabled,
        )

    def list_customer_summaries(self) -> list[dict[str, Any]]:
        return self._store.list_customer_summaries()

    def get_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any] | None:
        return self._store.get_workflow(customer_id=customer_id, workflow_id=workflow_id)

    def delete_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        workflow = self.get_workflow(customer_id=customer_id, workflow_id=workflow_id)
        if workflow is None:
            return {"ok": False, "deleted": False}
        self._store.delete_workflow_records(
            customer_id=customer_id,
            workflow_id=str(workflow["workflow_id"]),
            expected_revision=expected_revision,
        )
        return {"ok": True, "deleted": True, "workflow_id": workflow["workflow_id"]}

    def list_bookings(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._store.list_bookings(
            customer_id=customer_id,
            workflow_id=workflow_id,
            conversation_id=conversation_id,
        )

    def reconcile_sink_effect(
        self,
        *,
        customer_id: str,
        actor_id: str,
        workflow_id: str,
        booking_id: str,
        effect_revision: int,
        decision: str,
        reason: str,
        provider_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Apply an explicit owner decision to one indeterminate sink effect."""

        workflow = self.get_workflow(customer_id=customer_id, workflow_id=workflow_id)
        if workflow is None:
            raise LookupError("intake workflow not found")
        booking = next(
            (
                item
                for item in self.list_bookings(
                    customer_id=customer_id,
                    workflow_id=workflow_id,
                )
                if str(item.get("booking_id") or "") == booking_id
            ),
            None,
        )
        if booking is None:
            raise LookupError("intake booking not found")
        revision = int(booking.get("sink_effect_revision") or 0)
        phase = str(booking.get("sink_effect_phase") or "").strip()
        if revision != effect_revision or not phase:
            raise ValueError("sink effect revision conflict")
        if str(booking.get("sink_write_status") or "") not in {
            "pending",
            "indeterminate",
        }:
            raise ValueError("sink effect is not awaiting reconciliation")
        if decision not in {"confirm_applied", "retry_no_effect", "reject"}:
            raise ValueError("unsupported sink reconciliation decision")
        key = SinkWriter.effect_idempotency_key(
            booking_id=booking_id,
            phase=phase,
            revision=revision,
        )
        result = {
            "sink_type": str(workflow.get("sink_type") or ""),
            "toolkit": str(_safe_dict(workflow.get("sink_config")).get("toolkit") or ""),
            "tool_slug": str(
                _safe_dict(workflow.get("sink_config")).get("tool_slug") or ""
            ),
            "booking_id": booking_id,
            "data": _safe_dict(provider_result),
            "reconciled": True,
        }
        self._idempotency.reconcile_pending(
            tenant_id=customer_id,
            idempotency_key=key,
            decision=cast(Any, decision),
            actor_id=actor_id,
            reason=reason,
            result=result if decision == "confirm_applied" else None,
        )
        booking["sink_write_status"] = (
            "pending" if decision == "confirm_applied" else "failed"
        )
        booking["updated_at"] = _utc_now().isoformat()
        self._store.upsert_booking(booking)
        return {
            "booking_id": booking_id,
            "effect_revision": revision,
            "phase": phase,
            "decision": decision,
            "sink_write_status": booking["sink_write_status"],
        }

    def _index_workflow_knowledge(self, workflow: dict[str, Any]) -> None:
        knowledge = self._knowledge_service
        if knowledge is None:
            return
        file_ids = _unique_string_list(workflow.get("knowledge_file_ids"))
        if not file_ids:
            return
        with suppress(Exception):
            knowledge.index_sources(
                customer_id=str(workflow.get("customer_id", "") or "").strip(),
                scope_type="intake_workflow",
                scope_id=str(workflow.get("workflow_id", "") or "").strip(),
                file_ids=file_ids,
            )

    def _business_knowledge_query_text(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        recent_messages: list[dict[str, Any]],
        active_booking: dict[str, Any] | None,
    ) -> str:
        latest_text = str(conversation_summary.get("latest_inbound_message_text_preview", "") or "").strip()
        for item in reversed(recent_messages):
            if not isinstance(item, dict):
                continue
            if str(item.get("sender_role", "") or "").strip().lower() not in {"customer", "lead", "user"}:
                continue
            text = str(item.get("text", "") or "").strip()
            if text:
                latest_text = text
                break
        active_fields: dict[str, Any] = {}
        if isinstance(active_booking, dict):
            active_fields = _safe_dict(active_booking.get("extracted_fields"))
        parts = [
            f"Latest customer message: {latest_text}",
            f"Active booking extracted_fields JSON: {_json_dumps(active_fields)}",
            (
                "Workflow field contract JSON: "
                + _json_dumps(
                    {
                        "required_fields": _safe_list(workflow.get("required_fields")),
                        "field_guidance": _safe_dict(workflow.get("field_guidance")),
                    }
                )
            ),
            "Workflow scope: " + str(workflow.get("intent_description", "") or "").strip(),
            "Workflow instructions: " + str(workflow.get("assistant_instructions", "") or "").strip(),
            (
                "Task: return only source-backed business facts relevant to the latest customer "
                "message or active booking. Respect the configured workflow scope: if the source contains "
                "facts for a category outside this workflow, do not answer those facts for intake; return "
                "NO_SOURCE so the intake agent can redirect from workflow instructions. If the latest "
                "message only cancels, reschedules, or corrects an existing booking and needs no new "
                "business fact, return NO_SOURCE."
            ),
        ]
        return " ".join(part for part in parts if part).strip()

    def _business_knowledge_answer_for_workflow(
        self,
        *,
        customer_id: str,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        recent_messages: list[dict[str, Any]],
        active_booking: dict[str, Any] | None = None,
        query_override: str | None = None,
        include_no_source: bool = False,
    ) -> str:
        knowledge = self._knowledge_service
        if knowledge is None:
            return ""
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        file_ids = _unique_string_list(workflow.get("knowledge_file_ids"))
        if not workflow_id or not file_ids:
            return ""
        query = str(query_override or "").strip()
        if not query:
            query = self._business_knowledge_query_text(
                workflow=workflow,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
                active_booking=active_booking,
            )
        if not query:
            query = str(workflow.get("intent_description", "") or "").strip() or "business knowledge"
        workflow_context = {
            "workflow_id": workflow_id,
            "name": str(workflow.get("name", "") or "").strip(),
            "intent_description": str(workflow.get("intent_description", "") or "").strip(),
            "required_fields": _safe_list(workflow.get("required_fields")),
            "field_guidance": _safe_dict(workflow.get("field_guidance")),
            "assistant_instructions": str(workflow.get("assistant_instructions", "") or "").strip(),
            "active_booking": _safe_dict(active_booking),
        }
        with suppress(Exception):
            result = knowledge.query(
                customer_id=customer_id,
                scope_type="intake_workflow",
                scope_id=workflow_id,
                query=query,
                workflow_context=workflow_context,
            )
            if int(getattr(result, "section_count", 0) or 0) == 0:
                knowledge.index_sources(
                    customer_id=customer_id,
                    scope_type="intake_workflow",
                    scope_id=workflow_id,
                    file_ids=file_ids,
                )
                result = knowledge.query(
                    customer_id=customer_id,
                    scope_type="intake_workflow",
                    scope_id=workflow_id,
                    query=query,
                    workflow_context=workflow_context,
                )
            answer = getattr(result, "answer", None)
            if answer is None:
                if include_no_source:
                    return (
                        f"Business knowledge query: {query[:600]}\n"
                        "Business knowledge answer: NO_SOURCE"
                    )[:3600]
                return ""
            answer_text = str(getattr(answer, "answer_extract", "") or "").strip()
            if not answer_text:
                if include_no_source:
                    return (
                        f"Business knowledge query: {query[:600]}\n"
                        "Business knowledge answer: NO_SOURCE"
                    )[:3600]
                return ""
            return (
                f"Business knowledge query: {query[:600]}\n"
                f"Business knowledge answer: {answer_text[:3000]}"
            )[:3600]
        return ""

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
        return self._store.get_cursor(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
        )

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
        self._store.set_cursor(
            workflow_id=workflow_id,
            conversation_id=conversation_id,
            latest_inbound_message_id=latest_inbound_message_id,
            latest_inbound_message_time=latest_inbound_message_time,
            conversation_updated_time=conversation_updated_time,
            latest_outbound_message_id=latest_outbound_message_id,
            agent_action_at=agent_action_at,
        )

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
            telegram_normalized: list[dict[str, Any]] = []
            for item in messages:
                msg = _safe_dict(item)
                telegram_normalized.append(
                    {
                        "id": str(msg.get("message_id", msg.get("id", "")) or "").strip(),
                        "created_time": str(msg.get("date_iso", msg.get("created_time", "")) or "").strip(),
                        "sender_id": str(msg.get("from_user_id", msg.get("sender_id", "")) or "").strip(),
                        "sender_username": str(msg.get("from_username", msg.get("sender_username", "")) or "").strip(),
                        "sender_role": str(msg.get("sender_role", "") or "").strip() or "customer",
                        "text": str(msg.get("text", "") or "").strip(),
                    }
                )
            telegram_normalized.sort(key=lambda item: str(item.get("created_time", "")))
            return telegram_normalized[-_RECENT_MESSAGE_HISTORY_LIMIT:]
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
        return normalized[-_RECENT_MESSAGE_HISTORY_LIMIT:]

    def _unanswered_customer_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages:
            return []
        last_assistant_index = -1
        for index, item in enumerate(messages):
            if str(item.get("sender_role", "") or "").strip() == "assistant":
                last_assistant_index = index
        unanswered = [
            item
            for item in messages[last_assistant_index + 1 :]
            if str(item.get("sender_role", "") or "").strip() == "customer"
        ]
        if len(unanswered) <= 1:
            return unanswered

        latest_time = _parse_datetime(unanswered[-1].get("created_time"))
        if latest_time is None:
            return unanswered
        cutoff = latest_time - _UNANSWERED_CUSTOMER_BURST_WINDOW
        return [
            item
            for item in unanswered
            if (parsed := _parse_datetime(item.get("created_time"))) is None or parsed >= cutoff
        ]

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

    def _missing_field_for_follow_up(
        self,
        *,
        workflow: dict[str, Any],
        decision: dict[str, Any],
        active_booking: dict[str, Any] | None,
    ) -> str:
        missing_fields = _unique_string_list(decision.get("missing_fields"))
        if missing_fields:
            return missing_fields[0]
        known_fields: dict[str, Any] = {}
        if isinstance(active_booking, dict):
            known_fields.update(_safe_dict(active_booking.get("extracted_fields")))
        known_fields.update(_safe_dict(decision.get("extracted_fields")))
        known_fields.update(_safe_dict(decision.get("save_payload")))
        for field in _unique_string_list(workflow.get("required_fields")):
            if not str(known_fields.get(field, "") or "").strip():
                return field
        return ""

    def _no_file_business_knowledge_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        decision: dict[str, Any],
        active_booking: dict[str, Any] | None,
    ) -> str:
        latest_text = str(
            conversation_summary.get("latest_inbound_message_text_preview", "") or ""
        ).strip()
        query = str(decision.get("business_knowledge_query", "") or "").strip()
        instructions = str(workflow.get("assistant_instructions", "") or "").strip()
        missing_field = self._missing_field_for_follow_up(
            workflow=workflow,
            decision=decision,
            active_booking=active_booking,
        )
        label = missing_field.replace("_", " ").strip()
        if _looks_like_cyrillic(latest_text, query, instructions):
            prefix = "Точную информацию нужно уточнить у бизнеса."
            if label:
                return f"{prefix} Какое значение указать для поля «{label}»?"
            return prefix
        prefix = "I'll need to confirm that with the business."
        if label:
            return f"{prefix} What {label} should I use for the booking?"
        return prefix

    def _normalize_no_file_business_knowledge_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        active_booking: dict[str, Any] | None,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(decision)
        reply_action = str(normalized.get("reply_action", "none") or "none").strip().lower()
        reply_text = str(normalized.get("reply_text", "") or "").strip()
        normalized["needs_business_knowledge"] = False
        normalized["business_knowledge_query"] = ""
        normalized["knowledge_source_refs"] = []
        normalized["grounding_status"] = "no_source"
        normalized["ready_to_save"] = False
        normalized["save_payload"] = {}
        if reply_action != "send_reply" or not reply_text:
            normalized["reply_action"] = "send_reply"
            normalized["reply_text"] = self._no_file_business_knowledge_reply(
                workflow=workflow,
                conversation_summary=conversation_summary,
                decision=decision,
                active_booking=active_booking,
            )
        return normalized

    def _load_source_items(
        self,
        *,
        workflow: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None, list[dict[str, str]]]:
        context = messaging_adapter_context(workflow)
        adapter = self._messaging_adapters.get(context.key)
        if adapter is not None:
            result = adapter.list_source_items(context=context)
            return [dict(item) for item in result.items], result.error, result.warnings
        return [], (
            f"Workflow {context.workflow_name} failed: unsupported source "
            f"{context.channel}/{context.provider}."
        ), []

    def _load_source_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
        context = messaging_adapter_context(workflow)
        adapter = self._messaging_adapters.get(context.key)
        if adapter is not None:
            result = adapter.load_conversation(context=context, conversation_id=conversation_id)
            return dict(result.summary), result.conversation, result.error
        return {}, {}, "unsupported source"

    async def _send_source_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None:
        context = messaging_adapter_context(workflow)
        adapter = self._messaging_adapters.get(context.key)
        if adapter is not None:
            return await adapter.send_reply(
                context=context,
                conversation_summary=cast(ConversationSummary, conversation_summary),
                reply_text=reply_text,
            )
        return f"unsupported reply source {context.channel}/{context.provider}"

    async def _send_intake_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None:
        workflow_id = str(workflow.get("workflow_id", "") or "").strip()
        customer_id = str(workflow.get("customer_id", "") or "").strip()
        conversation_id = str(conversation_summary.get("conversation_id", "") or "").strip()
        source_message_id = str(
            conversation_summary.get("latest_inbound_message_id", "") or ""
        ).strip()
        if not all((workflow_id, customer_id, conversation_id, source_message_id)):
            return "intake reply is missing a durable workflow, conversation, or inbound message identity"

        reply_hash = hashlib.sha256(reply_text.encode("utf-8")).hexdigest()
        claim = self._store.claim_reply_delivery(
            workflow_id=workflow_id,
            customer_id=customer_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            reply_hash=reply_hash,
        )
        if claim != "claimed":
            self._emit_observability(
                event="intake.reply.suppressed",
                workflow=workflow,
                conversation_summary=conversation_summary,
                reason=(
                    "already_delivered"
                    if claim == "completed"
                    else "indeterminate_prior_attempt"
                    if claim == "pending"
                    else "different_reply_already_claimed"
                ),
            )
            return None

        try:
            error = await self._send_source_reply(
                workflow=workflow,
                conversation_summary=conversation_summary,
                reply_text=reply_text,
            )
        except Exception:
            return "intake reply delivery outcome is indeterminate"
        if error is not None:
            return error
        self._store.complete_reply_delivery(
            workflow_id=workflow_id,
            customer_id=customer_id,
            conversation_id=conversation_id,
            source_message_id=source_message_id,
            reply_hash=reply_hash,
        )
        return None

    async def run_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        event_type: str = "scheduled",
        force: bool = False,
    ) -> dict[str, Any]:
        return await self._workflow_runner.run_workflow(
            customer_id=customer_id,
            workflow_id=workflow_id,
            event_type=event_type,
            force=force,
        )

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
        return await self._decision_maker.decide_workflow_action(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
        )

    def _emit_apply_decision_validation_error(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        error: str,
        booking_action: str,
        sink_action: str,
    ) -> None:
        assert error
        assert booking_action == booking_action.strip().lower()
        assert sink_action == sink_action.strip().lower()
        fields: dict[str, Any] = {}
        if error.startswith("unsupported sink_action="):
            fields["sink_action"] = sink_action
        elif error.startswith("unsupported booking_action="):
            fields["booking_action"] = booking_action
        else:
            fields["booking_action"] = booking_action
            fields["sink_action"] = sink_action
        self._emit_observability(
            event="intake.apply.error",
            workflow=workflow,
            conversation_summary=conversation_summary,
            phase="decision_validation",
            error=error,
            **fields,
        )

    def _resolve_booking_target(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        booking_action: str,
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
    ) -> BookingTargetResolution:
        assert booking_action in DECISION_BOOKING_ACTIONS
        assert booking_action != "ignore"
        normalized_booking_action = booking_action
        target_booking: dict[str, Any] | None = None
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
                "sink_effect_revision": 0,
                "sink_effect_phase": "",
                "conversation_summary": "",
                "last_customer_message_at": "",
                "opened_at": _utc_now_iso(),
                "completed_at": "",
                "edit_window_until": "",
                "created_at": _utc_now_iso(),
                "updated_at": _utc_now_iso(),
            }
        return BookingTargetResolution(
            booking_action=normalized_booking_action,
            booking=target_booking,
        )

    async def _apply_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        decision: dict[str, Any],
        stale_guard: bool = False,
    ) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]:
        return await self._decision_applier.apply_decision(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            decision=decision,
            stale_guard=stale_guard,
        )

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
                "extracted_fields": _safe_dict(decision.get("extracted_fields")),
                "save_payload": _safe_dict(decision.get("save_payload")),
                "sink_action": str(decision.get("sink_action", "") or "").strip().lower(),
                "sink_payload": _safe_dict(decision.get("sink_payload")),
            },
        }

    def _write_to_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        return self._sink_writer.write_to_sink(
            workflow=workflow,
            booking=booking,
            conversation_summary=conversation_summary,
            payload=payload,
            record_status=record_status,
        )

    def _write_to_local_csv(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        return self._sink_writer.write_to_local_csv(
            workflow=workflow,
            booking=booking,
            payload=payload,
            record_status=record_status,
        )

    def _write_to_composio_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        return self._sink_writer.write_to_composio_sink(
            workflow=workflow,
            booking=booking,
            conversation_summary=conversation_summary,
            payload=payload,
            record_status=record_status,
        )

    def _upsert_booking(self, booking: dict[str, Any]) -> None:
        self._store.upsert_booking(booking)

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
        status = str(booking.get("status", "") or "").strip().lower()
        action = "Booking cancelled" if status == "cancelled" else "Booking saved"
        parts = [
            f"{action} for {workflow['name']}:",
            f"contact={contact}",
            f"booking_id={booking['booking_id']}",
        ]
        for key in _unique_string_list(list(fields.keys()))[:5]:
            value = str(fields.get(key, "") or "").strip()
            if value:
                parts.append(f"{key}={value}")
        parts.append(f"sink={workflow['sink_type']}")
        return " ".join(parts)[:1000]

    def _build_cancellation_confirmation_reply(
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
        when = " ".join([date, time_value]).strip()
        if russian:
            if service and when:
                return f"Готово, запись на {service} {when} отменена."[:1000]
            if service:
                return f"Готово, запись на {service} отменена."[:1000]
            return "Готово, запись отменена."
        if service and when:
            return f"Done, your {service} booking for {when} is cancelled."[:1000]
        if service:
            return f"Done, your {service} booking is cancelled."[:1000]
        return "Done, your booking is cancelled."

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
