"""
In-process LangGraph runtime for OpenTulpa.

This replaces the Parlant subprocess/session model with a local StateGraph that:
- runs tool-calling in a bounded loop,
- persists thread state via SQLite checkpointer,
- supports token streaming for Telegram,
- and reuses existing /internal/* APIs as tool backends.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.agent.context_compaction import (
    compress_rollup as _compress_rollup,
)
from opentulpa.agent.context_compaction import (
    maybe_compact_thread_context as _maybe_compact_thread_context,
)
from opentulpa.agent.context_compaction import (
    persist_rollup_memory as _persist_rollup_memory,
)
from opentulpa.agent.context_compaction import (
    split_text_chunks as _split_text_chunks,
)
from opentulpa.agent.file_analysis import (
    analyze_uploaded_file as _analyze_uploaded_file,
)
from opentulpa.agent.file_analysis import (
    extract_docx_text as _extract_docx_text,
)
from opentulpa.agent.file_analysis import (
    extract_pdf_text as _extract_pdf_text,
)
from opentulpa.agent.file_analysis import (
    extract_uploaded_text as _extract_uploaded_text,
)
from opentulpa.agent.file_analysis import (
    summarize_uploaded_blob as _summarize_uploaded_blob,
)
from opentulpa.agent.file_analysis import (
    transcribe_audio_blob as _transcribe_audio_blob,
)
from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.internal_api_client import InternalApiClient
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.runtime_input import (
    MergedInputSuppressedError,
    ThreadInputCoordinator,
)
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.agent.turn_policy import (
    normalize_turn_mode as _normalize_turn_mode,
)
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    minutes_to_utc_offset as _minutes_to_utc_offset,
)
from opentulpa.agent.utils import (
    normalize_model_name as _normalize_model_name,
)
from opentulpa.agent.utils import (
    utc_offset_to_minutes as _utc_offset_to_minutes,
)
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.link_aliases import LinkAliasService
from opentulpa.context.service import EventContextService
from opentulpa.context.thread_rollups import ThreadRollupService
from opentulpa.core.ids import new_short_id

logger = logging.getLogger(__name__)
_LINK_ID_TOKEN_RE = re.compile(r"\blink_[A-Za-z0-9]{4,12}\b")
STREAM_WAIT_SIGNAL = "__TULPA_STREAM_WAIT__"
STREAM_APPROVAL_HANDOFF_SIGNAL = "__TULPA_APPROVAL_HANDOFF__"
STREAM_EMPTY_REPLY_FALLBACK = (
    "I couldn't produce a visible user-facing reply for that step. "
    "Please retry, and I will continue from the latest state."
)

APPROVAL_EXECUTION_CUSTOMER_ID_TOOLS: set[str] = {
    "memory_search",
    "memory_add",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_send",
    "tulpa_file_send",
    "web_image_send",
    "uploaded_file_analyze",
    "skill_list",
    "skill_get",
    "skill_upsert",
    "skill_delete",
    "directive_get",
    "directive_set",
    "directive_clear",
    "lessons_learnt",
    "time_profile_get",
    "time_profile_set",
    "routine_list",
    "routine_create",
    "routine_delete",
    "automation_delete",
    "browser_use_run",
    "tulpa_run_terminal",
}


class _WakeClassification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    notify_user: bool = False
    reason: str = ""


class _ClaimCheckDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    applies: bool = False
    mismatch: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    repair_instruction: str = ""


class _GuardrailIntentDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    gate: str = "allow"
    impact_type: str = "read"
    recipient_scope: str = "self"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class _SkillSelectionItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = ""
    score: float = Field(default=0.0)
    reason: str = ""


class _SkillSelectionDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    selected: list[_SkillSelectionItem] = Field(default_factory=list)


@dataclass(slots=True)
class _PreparedTurnContext:
    through_id: int | None
    config: dict[str, Any]
    graph_input: dict[str, Any]


class OpenTulpaLangGraphRuntime:
    def __init__(
        self,
        *,
        app_url: str,
        openrouter_api_key: str,
        model_name: str,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        wake_classifier_model_name: str | None = None,
        guardrail_classifier_model_name: str | None = None,
        checkpoint_db_path: str,
        recursion_limit: int = 30,
        max_completion_tokens: int = 4096,
        max_user_reply_chars: int = 4000,
        context_events: EventContextService | None = None,
        customer_profile_service: CustomerProfileService | None = None,
        thread_rollup_service: ThreadRollupService | None = None,
        link_alias_service: LinkAliasService | None = None,
        context_token_limit: int = 12000,
        context_rollup_tokens: int = 2200,
        context_recent_tokens: int = 3500,
        context_compaction_source_tokens: int = 100000,
        input_debounce_seconds: float = 0.65,
        proactive_heartbeat_default_hours: int = 3,
        behavior_log_enabled: bool = True,
        behavior_log_path: str = ".opentulpa/logs/agent_behavior.jsonl",
        browser_use_headless: bool = True,
        browser_use_model_override: str | None = None,
        browser_use_max_concurrent_tasks: int = 2,
        browser_use_task_retention_seconds: int = 1800,
    ) -> None:
        self.app_url = app_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = str(openrouter_base_url or "").strip() or "https://openrouter.ai/api/v1"
        self.model_name = _normalize_model_name(model_name)
        self._max_completion_tokens = max(128, min(int(max_completion_tokens), 32768))
        self._max_user_reply_chars = max(500, min(int(max_user_reply_chars), 20000))
        self._wake_classifier_model_name = (
            _normalize_model_name(wake_classifier_model_name)
            if str(wake_classifier_model_name or "").strip()
            else self.model_name
        )
        guardrail_model = (
            str(guardrail_classifier_model_name).strip()
            if str(guardrail_classifier_model_name or "").strip()
            else "minimax/minimax-m2.5"
        )
        self._guardrail_classifier_model_name = _normalize_model_name(guardrail_model)
        self.checkpoint_db_path = checkpoint_db_path
        self.recursion_limit = recursion_limit
        self._context_events = context_events
        self._customer_profile_service = customer_profile_service
        self._thread_rollup_service = thread_rollup_service
        self._link_alias_service = link_alias_service
        self._context_token_limit = max(6000, min(24000, int(context_token_limit)))
        self._context_short_term_high_tokens = self._context_token_limit
        self._context_short_term_low_tokens = min(
            max(1500, int(context_recent_tokens)),
            max(1500, self._context_short_term_high_tokens - 500),
        )
        self._context_rollup_tokens = min(
            max(500, int(context_rollup_tokens)),
            max(500, self._context_short_term_low_tokens - 250),
        )
        # Backward-compat aliases consumed by existing helpers/tests.
        self._context_recent_tokens = self._context_short_term_low_tokens
        self._context_compaction_source_tokens = max(
            self._context_rollup_tokens,
            int(context_compaction_source_tokens),
        )
        self._input_debounce_seconds = max(0.0, min(float(input_debounce_seconds), 3.0))
        self._proactive_heartbeat_default_hours = max(1, min(int(proactive_heartbeat_default_hours), 24))
        self._behavior_log_enabled = bool(behavior_log_enabled)
        raw_behavior_path = str(behavior_log_path or "").strip() or ".opentulpa/logs/agent_behavior.jsonl"
        self._behavior_log_path = Path(raw_behavior_path).resolve()
        self._behavior_log_lock = threading.Lock()
        if self._behavior_log_enabled:
            self._behavior_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._browser_use_headless = bool(browser_use_headless)
        self._browser_use_model_override = str(browser_use_model_override or "").strip()
        self._browser_use_max_concurrent_tasks = max(1, int(browser_use_max_concurrent_tasks))
        self._browser_use_task_retention_seconds = max(60, int(browser_use_task_retention_seconds))
        self._browser_use_local_manager: Any | None = None
        self._active_customer_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
            "opentulpa_active_customer_id",
            default="",
        )
        self._active_customer_id = ""

        self._model = init_chat_model(
            self.model_name,
            model_provider="openai",
            api_key=openrouter_api_key,
            base_url=self.openrouter_base_url,
            temperature=0,
            max_completion_tokens=self._max_completion_tokens,
        )
        if self._wake_classifier_model_name == self.model_name:
            self._wake_classifier_model = self._model
        else:
            try:
                self._wake_classifier_model = init_chat_model(
                    self._wake_classifier_model_name,
                    model_provider="openai",
                    api_key=openrouter_api_key,
                    base_url=self.openrouter_base_url,
                    temperature=0,
                    max_completion_tokens=self._max_completion_tokens,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize wake classifier model '%s'; falling back to main model '%s'.",
                    self._wake_classifier_model_name,
                    self.model_name,
                )
                self._wake_classifier_model = self._model
        if self._guardrail_classifier_model_name == self.model_name:
            self._guardrail_classifier_model = self._model
        elif self._guardrail_classifier_model_name == self._wake_classifier_model_name:
            self._guardrail_classifier_model = self._wake_classifier_model
        else:
            try:
                self._guardrail_classifier_model = init_chat_model(
                    self._guardrail_classifier_model_name,
                    model_provider="openai",
                    api_key=openrouter_api_key,
                    base_url=self.openrouter_base_url,
                    temperature=0,
                    max_completion_tokens=self._max_completion_tokens,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize guardrail classifier model '%s'; "
                    "falling back to main model '%s'.",
                    self._guardrail_classifier_model_name,
                    self.model_name,
                )
                self._guardrail_classifier_model = self._model

        self._checkpointer_cm: Any | None = None
        self._checkpointer: Any | None = None
        self._graph = None
        self._tools: dict[str, Any] = {}
        self._model_with_tools = None
        self._thread_inputs = ThreadInputCoordinator(debounce_seconds=self._input_debounce_seconds)
        self._internal_api = InternalApiClient(base_url=self.app_url)

    def get_browser_use_local_manager(self) -> Any:
        if self._browser_use_local_manager is None:
            from opentulpa.agent.browser_use_local import BrowserUseLocalManager

            self._browser_use_local_manager = BrowserUseLocalManager(
                openrouter_api_key=self.openrouter_api_key,
                openrouter_base_url=self.openrouter_base_url,
                default_model=self.model_name,
                model_override=self._browser_use_model_override,
                headless=self._browser_use_headless,
                max_concurrent_tasks=self._browser_use_max_concurrent_tasks,
                task_retention_seconds=self._browser_use_task_retention_seconds,
            )
        return self._browser_use_local_manager

    def log_behavior_event(self, *, event: str, **fields: Any) -> None:
        if not bool(getattr(self, "_behavior_log_enabled", False)):
            return
        event_name = str(event or "").strip()
        if not event_name:
            return
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "event": event_name,
        }
        for key, value in fields.items():
            safe_key = str(key or "").strip()
            if not safe_key:
                continue
            payload[safe_key] = value
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        lock = getattr(self, "_behavior_log_lock", None)
        path = getattr(self, "_behavior_log_path", None)
        if not isinstance(path, Path):
            return
        with suppress(Exception):
            path.parent.mkdir(parents=True, exist_ok=True)
        if lock is None:
            with suppress(Exception), path.open("a", encoding="utf-8") as f:
                f.write(serialized + "\n")
            return
        with suppress(Exception), lock, path.open("a", encoding="utf-8") as f:
            f.write(serialized + "\n")

    def _truncate_user_visible_reply(self, text: str) -> tuple[str, bool]:
        raw = str(text or "").strip()
        if not raw:
            return "", False
        max_chars = int(getattr(self, "_max_user_reply_chars", 4000))
        if len(raw) <= max_chars:
            return raw, False

        suffix = "\n\n[Response truncated to fit chat limits.]"
        keep = max(160, max_chars - len(suffix))
        clipped = raw[:keep].rstrip()
        boundary_floor = max(0, int(keep * 0.6))
        cut_positions = [
            clipped.rfind("\n\n", boundary_floor),
            clipped.rfind("\n", boundary_floor),
            clipped.rfind(". ", boundary_floor),
            clipped.rfind("! ", boundary_floor),
            clipped.rfind("? ", boundary_floor),
            clipped.rfind("; ", boundary_floor),
        ]
        best_cut = max(cut_positions)
        if best_cut > 0:
            clipped = clipped[:best_cut].rstrip()
        return clipped + suffix, True

    async def _invoke_structured_model(
        self,
        *,
        model: Any,
        messages: list[Any],
        schema: type[BaseModel],
    ) -> tuple[BaseModel | None, str | None]:
        last_error: str | None = None
        structured = getattr(model, "with_structured_output", None)
        if callable(structured):
            try:
                runner = structured(schema)
                payload = await runner.ainvoke(messages)
                if isinstance(payload, schema):
                    return payload, None
                if isinstance(payload, dict):
                    return schema.model_validate(payload), None
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        try:
            response = await model.ainvoke(messages)
            raw = _content_to_text(getattr(response, "content", response)).strip()
            if raw:
                return schema.model_validate_json(raw), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        return None, last_error

    @staticmethod
    def _tool_message_indicates_approval_handoff(message: Any) -> bool:
        if not isinstance(message, ToolMessage):
            return False
        extras = getattr(message, "additional_kwargs", {}) or {}
        if isinstance(extras, dict):
            control = extras.get("opentulpa_control", {})
            if isinstance(control, dict):
                status = str(control.get("status", "")).strip().lower()
                if status == "approval_pending":
                    return True
        raw_content = _content_to_text(getattr(message, "content", "")).strip()
        if not raw_content or not raw_content.startswith("{"):
            return False
        try:
            payload = json.loads(raw_content)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        return str(payload.get("status", "")).strip().lower() == "approval_pending"

    @staticmethod
    def _build_approval_handoff_text(result: dict[str, Any]) -> str:
        outcomes = result.get("tool_outcomes", [])
        if not isinstance(outcomes, list):
            return ""
        for item in outcomes:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")).strip().lower() != "approval_pending":
                continue
            approval_id = str(item.get("approval_id", "")).strip()
            action_summary = OpenTulpaLangGraphRuntime._approval_handoff_subject(item)
            follow_up_hint = OpenTulpaLangGraphRuntime._approval_handoff_follow_up_hint(item)
            if approval_id:
                return (
                    f"Approval required before execution of {action_summary}. "
                    f"approval_id={approval_id}. Approve or deny this request in the UI."
                    f"{follow_up_hint}"
                )
            return (
                f"Approval required before execution of {action_summary}. "
                f"Approve or deny in the UI.{follow_up_hint}"
            )
        return ""

    @staticmethod
    def _approval_handoff_subject(item: dict[str, Any]) -> str:
        summary = " ".join(str(item.get("summary", "")).split()).strip()
        if summary:
            return summary[:180] + ("..." if len(summary) > 180 else "")
        action_name = str(item.get("action_name", "") or item.get("tool_name", "")).strip()
        if action_name:
            return action_name
        return "this action"

    @staticmethod
    def _approval_handoff_follow_up_hint(item: dict[str, Any]) -> str:
        haystack = " ".join(
            str(item.get(key, "")).strip().lower()
            for key in ("summary", "action_name", "tool_name")
        )
        if any(token in haystack for token in ("send", "message", "post", "publish")):
            return " If you want, I can draft the content here first before you approve anything."
        if any(token in haystack for token in ("routine", "schedule", "automation", "remind")):
            return (
                " If you wanted discussion first, tell me to keep it in chat "
                "and I will plan it with you before creating any routine."
            )
        return " If you want, I can explain the planned action before you approve it."

    @staticmethod
    def _summarize_pending_payload(payload: Any, *, payload_limit: int = 240) -> str:
        if isinstance(payload, dict):
            allowed_keys = (
                "approval_id",
                "status",
                "action_name",
                "execution_ok",
                "retryable",
                "event_label",
                "routine_id",
                "routine_name",
                "task_id",
                "reason",
            )
            parts: list[str] = []
            for key in allowed_keys:
                value = payload.get(key)
                if value in (None, ""):
                    continue
                text = " ".join(str(value).split())
                if len(text) > 90:
                    text = text[:90] + "..."
                parts.append(f"{key}={text}")
            if not parts:
                keys = sorted(str(k) for k in payload if str(k).strip())
                if keys:
                    shown = ", ".join(keys[:6])
                    more = f" (+{len(keys) - 6})" if len(keys) > 6 else ""
                    return f"payload_keys={shown}{more}"
                return ""
            summary = "; ".join(parts)
        else:
            summary = " ".join(str(payload).split())
        if len(summary) > payload_limit:
            summary = summary[:payload_limit] + "..."
        return summary

    @staticmethod
    def _format_pending_context(events: list[dict[str, Any]], *, payload_limit: int = 240) -> str:
        lines: list[str] = []
        for idx, event in enumerate(events, start=1):
            source = str(event.get("source", "event"))
            event_type = str(event.get("event_type", "update"))
            payload_text = OpenTulpaLangGraphRuntime._summarize_pending_payload(
                event.get("payload", {}),
                payload_limit=payload_limit,
            )
            if payload_text:
                lines.append(f"{idx}. [{source}/{event_type}] {payload_text}")
            else:
                lines.append(f"{idx}. [{source}/{event_type}]")
        return "\n".join(lines)

    def _load_pending_context(
        self,
        *,
        customer_id: str,
        include_pending_context: bool,
    ) -> tuple[list[dict[str, Any]], int | None]:
        if not include_pending_context or self._context_events is None:
            return [], None
        pending = self._context_events.list_events(customer_id, limit=20)
        if not pending:
            return [], None
        through_id = int(pending[-1]["id"])
        return pending, through_id

    def _build_pending_context_summary(
        self,
        *,
        customer_id: str,
        include_pending_context: bool,
    ) -> tuple[str, int | None]:
        pending, through_id = self._load_pending_context(
            customer_id=customer_id,
            include_pending_context=include_pending_context,
        )
        if not pending:
            return "", through_id
        return self._format_pending_context(pending), through_id

    async def _has_pending_approval_lock(self, *, customer_id: str, thread_id: str) -> bool:
        cid = str(customer_id or "").strip()
        tid = str(thread_id or "").strip()
        if not cid or not tid:
            return False
        try:
            response = await self._request_with_backoff(
                "GET",
                "/internal/approvals/pending/status",
                params={"customer_id": cid, "thread_id": tid},
                timeout=5.0,
                retries=0,
            )
        except Exception:
            return False
        if response.status_code != 200:
            return False
        with suppress(Exception):
            payload = response.json()
            return bool(payload.get("pending", False))
        return False

    def register_links_from_text(
        self,
        *,
        customer_id: str,
        text: str,
        source: str,
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        if self._link_alias_service is None:
            return []
        cid = str(customer_id or "").strip()
        if not cid:
            return []
        raw = str(text or "")
        if not raw:
            return []
        with suppress(Exception):
            return self._link_alias_service.register_links_from_text(
                cid,
                raw,
                source=source,
                limit=limit,
            )
        return []

    def expand_link_aliases(self, *, customer_id: str, text: str) -> str:
        if self._link_alias_service is None:
            return str(text or "")
        cid = str(customer_id or "").strip()
        raw = str(text or "")
        if not cid or not raw or "link_" not in raw.lower():
            return raw
        with suppress(Exception):
            return self._link_alias_service.expand_link_ids_in_text(cid, raw)
        return raw

    def resolve_link_aliases_in_args(self, *, customer_id: str, args: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(args, dict):
            return {}

        def _walk(value: Any) -> Any:
            if isinstance(value, str):
                if _LINK_ID_TOKEN_RE.search(value):
                    return self.expand_link_aliases(customer_id=customer_id, text=value)
                return value
            if isinstance(value, list):
                return [_walk(item) for item in value]
            if isinstance(value, dict):
                return {str(k): _walk(v) for k, v in value.items()}
            return value

        return {str(k): _walk(v) for k, v in args.items()}

    def _build_link_alias_context(self, *, customer_id: str, user_text: str) -> str:
        if self._link_alias_service is None:
            return ""
        cid = str(customer_id or "").strip()
        if not cid:
            return ""
        safe_user_text = str(user_text or "")
        seen_ids: set[str] = set()
        selected: list[dict[str, Any]] = []

        try:
            mentioned = self._link_alias_service.extract_link_ids(safe_user_text, limit=8)
        except Exception:
            mentioned = []
        for link_id in mentioned:
            with suppress(Exception):
                item = self._link_alias_service.get_by_id(cid, link_id)
                if not item:
                    continue
                lid = str(item.get("id", "")).strip().lower()
                if not lid or lid in seen_ids:
                    continue
                seen_ids.add(lid)
                selected.append(item)

        max_aliases = 4
        if len(selected) < max_aliases:
            recent: list[dict[str, Any]] = []
            with suppress(Exception):
                recent = self._link_alias_service.list_recent(cid, limit=max_aliases)
            for item in recent:
                lid = str(item.get("id", "")).strip().lower()
                if not lid or lid in seen_ids:
                    continue
                seen_ids.add(lid)
                selected.append(item)
                if len(selected) >= max_aliases:
                    break

        if not selected:
            return ""
        lines = [f"- {item['id']}: {item['url']}" for item in selected[:max_aliases]]
        return (
            "Known long-link aliases for this user:\n"
            + "\n".join(lines)
            + "\nUse alias IDs for long URLs. Outputting a known alias expands to the full URL."
        )

    async def _load_active_directive(self, customer_id: str) -> str | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        if self._customer_profile_service is not None:
            try:
                return self._customer_profile_service.get_directive(cid)
            except Exception:
                pass
        try:
            r = await self._request_with_backoff(
                "POST",
                "/internal/directive/get",
                json_body={"customer_id": cid},
                timeout=5.0,
                retries=1,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            directive = str(data.get("directive") or "").strip()
            return directive or None
        except Exception:
            return None

    async def _load_user_utc_offset(self, customer_id: str) -> str | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        if self._customer_profile_service is not None:
            with suppress(Exception):
                return self._customer_profile_service.get_utc_offset(cid)
        try:
            r = await self._request_with_backoff(
                "POST",
                "/internal/time_profile/get",
                json_body={"customer_id": cid},
                timeout=5.0,
                retries=1,
            )
            if r.status_code != 200:
                return None
            data = r.json()
            offset = str(data.get("utc_offset") or "").strip()
            return offset or None
        except Exception:
            return None

    async def _list_available_skills(self, customer_id: str) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        try:
            r = await self._request_with_backoff(
                "POST",
                "/internal/skills/list",
                json_body={
                    "customer_id": cid,
                    "include_global": True,
                    "include_disabled": False,
                    "limit": 200,
                },
                timeout=8.0,
                retries=1,
            )
            if r.status_code != 200:
                return []
            data = r.json()
            skills = data.get("skills", [])
            if not isinstance(skills, list):
                return []
            out: list[dict[str, Any]] = []
            for item in skills:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                desc = str(item.get("description", "")).strip()
                scope = str(item.get("scope", "")).strip() or "user"
                if not name or not desc:
                    continue
                out.append(
                    {
                        "name": name,
                        "description": desc,
                        "scope": scope,
                    }
                )
            return out
        except Exception:
            return []

    async def _select_relevant_skills(
        self,
        *,
        customer_id: str,
        query: str,
        candidates: list[dict[str, Any]],
        max_skills: int = 2,
    ) -> list[dict[str, Any]]:
        prompt_query = str(query or "").strip()
        if not prompt_query or not candidates:
            return []
        shortlist = candidates[:80]
        catalog = "\n".join(
            [
                f"- name={c['name']} scope={c['scope']} description={c['description'][:300]}"
                for c in shortlist
            ]
        )
        decision, _ = await self._invoke_structured_model(
            model=self._model,
            schema=_SkillSelectionDecision,
            messages=[
                SystemMessage(
                    content=(
                        "You select reusable skills for the current user request.\n"
                        "Return strict JSON object with key 'selected', an array of objects:\n"
                        "  {\"name\": string, \"score\": number, \"reason\": string}\n"
                        "Choose only skills that materially improve answer quality.\n"
                        "If the request is about reminders, schedules, recurring jobs, cron, or automations, "
                        "prefer selecting routine-schedule-composer when available.\n"
                        "Prioritize skills that improve execution reliability and claim accuracy over style-only skills.\n"
                        "If none apply, return {\"selected\": []}."
                    )
                ),
                HumanMessage(
                    content=(
                        f"customer_id={customer_id}\n"
                        f"user_request={prompt_query[:2000]}\n\n"
                        f"available_skills:\n{catalog}"
                    )
                ),
            ],
        )
        if decision is None or not isinstance(decision, _SkillSelectionDecision):
            return []
        by_name = {c["name"]: c for c in shortlist}
        selected: list[dict[str, Any]] = []
        for item in decision.selected:
            name = str(item.name or "").strip()
            if not name or name not in by_name:
                continue
            score = float(item.score)
            if score < 0.45:
                continue
            selected.append(
                {
                    **by_name[name],
                    "score": score,
                    "reason": str(item.reason or "").strip()[:300],
                }
            )
        selected.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return selected[: max(1, min(int(max_skills), 3))]

    async def _resolve_skill_context(self, customer_id: str, user_text: str) -> dict[str, Any]:
        cid = str(customer_id or "").strip()
        query = str(user_text or "").strip()
        if not cid or not query:
            return {"skill_names": [], "context": ""}
        candidates = await self._list_available_skills(cid)
        if not candidates:
            return {"skill_names": [], "context": ""}
        selected = await self._select_relevant_skills(
            customer_id=cid,
            query=query,
            candidates=candidates,
            max_skills=1,
        )
        if not selected:
            return {"skill_names": [], "context": ""}

        sections: list[str] = []
        skill_names: list[str] = []
        total_chars = 0
        max_total_chars = 9000
        for item in selected:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            try:
                r = await self._request_with_backoff(
                    "POST",
                    "/internal/skills/get",
                    json_body={
                        "customer_id": cid,
                        "name": name,
                        "include_files": False,
                        "include_global": True,
                    },
                    timeout=8.0,
                    retries=1,
                )
                if r.status_code != 200:
                    continue
                payload = r.json()
                skill = payload.get("skill", {})
                if not isinstance(skill, dict):
                    continue
                skill_md = str(skill.get("skill_markdown", "")).strip()
                if not skill_md:
                    continue
                snippet = (
                    f"Skill name: {name}\n"
                    f"Scope: {skill.get('scope', '')}\n"
                    f"Description: {skill.get('description', '')}\n"
                    f"Selection reason: {item.get('reason', '')}\n\n"
                    f"SKILL.md:\n{skill_md[:3500]}"
                )
                if total_chars + len(snippet) > max_total_chars:
                    break
                sections.append(snippet)
                skill_names.append(name)
                total_chars += len(snippet)
            except Exception:
                continue
        context = "\n\n---\n\n".join(sections).strip()
        return {"skill_names": skill_names, "context": context}

    async def _build_live_time_context(self, customer_id: str) -> dict[str, str]:
        now_server = datetime.now().astimezone()
        now_utc = datetime.now(UTC)
        server_offset = now_server.utcoffset() or timedelta()
        server_offset_minutes = int(server_offset.total_seconds() // 60)
        server_offset_text = _minutes_to_utc_offset(server_offset_minutes)

        user_offset_text = await self._load_user_utc_offset(customer_id)
        source = "profile"
        user_offset_minutes = (
            _utc_offset_to_minutes(user_offset_text) if user_offset_text else None
        )
        if user_offset_minutes is None:
            user_offset_minutes = server_offset_minutes
            user_offset_text = server_offset_text
            source = "fallback_server_timezone"

        user_local = now_utc + timedelta(minutes=user_offset_minutes)
        return {
            "server_time_local_iso": now_server.isoformat(),
            "server_time_utc_iso": now_utc.isoformat(),
            "server_utc_offset": server_offset_text,
            "user_time_local_iso": user_local.isoformat(),
            "user_utc_offset": user_offset_text,
            "user_time_source": source,
        }

    def _load_thread_rollup(self, thread_id: str) -> str | None:
        tid = str(thread_id or "").strip()
        if not tid or self._thread_rollup_service is None:
            return None
        try:
            text = self._thread_rollup_service.get_rollup(tid)
            return self._cap_rollup_text(text)
        except Exception:
            return None

    def _save_thread_rollup(self, thread_id: str, rollup: str) -> None:
        tid = str(thread_id or "").strip()
        text = self._cap_rollup_text(rollup)
        if not tid or not text or self._thread_rollup_service is None:
            return
        with suppress(Exception):
            self._thread_rollup_service.set_rollup(tid, text)

    def _cap_rollup_text(self, text: str | None) -> str:
        raw = str(text or "").strip()
        if not raw:
            return ""
        max_chars = max(800, int(self._context_rollup_tokens) * 4)
        if len(raw) <= max_chars:
            return raw
        reserve = max(200, max_chars // 2 - 8)
        return f"{raw[:reserve]}\n...\n{raw[-reserve:]}"

    @staticmethod
    def _extract_docx_text(raw_bytes: bytes) -> str:
        return _extract_docx_text(raw_bytes)

    @staticmethod
    def _extract_pdf_text(raw_bytes: bytes) -> str:
        return _extract_pdf_text(raw_bytes)

    @staticmethod
    def _extract_uploaded_text(
        *,
        raw_bytes: bytes,
        filename: str | None,
        mime_type: str | None,
        max_chars: int = 140000,
    ) -> str:
        return _extract_uploaded_text(
            raw_bytes=raw_bytes,
            filename=filename,
            mime_type=mime_type,
            max_chars=max_chars,
        )

    async def summarize_uploaded_blob(
        self,
        *,
        filename: str | None,
        mime_type: str | None,
        kind: str | None,
        raw_bytes: bytes,
        caption: str | None = None,
        question: str | None = None,
    ) -> str:
        return await _summarize_uploaded_blob(
            self,
            filename=filename,
            mime_type=mime_type,
            kind=kind,
            raw_bytes=raw_bytes,
            caption=caption,
            question=question,
        )

    async def transcribe_audio_blob(
        self,
        *,
        filename: str | None,
        mime_type: str | None,
        kind: str | None,
        raw_bytes: bytes,
    ) -> str:
        return await _transcribe_audio_blob(
            self,
            filename=filename,
            mime_type=mime_type,
            kind=kind,
            raw_bytes=raw_bytes,
        )

    async def analyze_uploaded_file(
        self,
        *,
        record: dict[str, Any],
        raw_bytes: bytes,
        question: str | None = None,
    ) -> dict[str, Any]:
        return await _analyze_uploaded_file(
            self,
            record=record,
            raw_bytes=raw_bytes,
            question=question,
        )

    @staticmethod
    def _split_text_chunks(text: str, *, approx_tokens_per_chunk: int = 25000) -> list[str]:
        return _split_text_chunks(text, approx_tokens_per_chunk=approx_tokens_per_chunk)

    async def _compress_rollup(self, existing_rollup: str, additional_text: str) -> str:
        return await _compress_rollup(self, existing_rollup, additional_text)

    async def _persist_rollup_memory(self, *, customer_id: str, thread_id: str, rollup: str) -> None:
        await _persist_rollup_memory(
            self,
            customer_id=customer_id,
            thread_id=thread_id,
            rollup=rollup,
        )

    async def _maybe_compact_thread_context(self, *, thread_id: str, customer_id: str) -> None:
        await _maybe_compact_thread_context(
            self,
            thread_id=thread_id,
            customer_id=customer_id,
        )

    async def _pre_resolve_skill_state(
        self,
        *,
        customer_id: str,
        user_text: str,
    ) -> dict[str, Any]:
        query = str(user_text or "").strip()
        if not query:
            return {
                "active_skill_query": "",
                "active_skill_context": "",
                "active_skill_names": [],
            }
        resolved = await self._resolve_skill_context(customer_id, query)
        context = str(resolved.get("context", "")).strip()
        names_raw = resolved.get("skill_names", [])
        names = [str(n).strip() for n in names_raw if str(n).strip()] if isinstance(names_raw, list) else []
        return {
            "active_skill_query": query,
            "active_skill_context": context,
            "active_skill_names": names,
        }

    async def start(self) -> None:
        if self._graph is not None:
            return
        db_path = Path(self.checkpoint_db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(db_path))
        self._checkpointer = await self._checkpointer_cm.__aenter__()
        if hasattr(self._checkpointer, "setup"):
            maybe_coro = self._checkpointer.setup()
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        self._register_tools()
        self._model_with_tools = self._model.bind_tools(list(self._tools.values()))
        self._graph = self._build_graph()
        manager = self.get_browser_use_local_manager()
        with suppress(Exception):
            preflight_error = await manager.preflight()
            if preflight_error:
                logger.warning("browser_use local preflight warning: %s", preflight_error)

    async def shutdown(self) -> None:
        manager = self._browser_use_local_manager
        if manager is not None:
            with suppress(Exception):
                await manager.shutdown()
        self._browser_use_local_manager = None
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None
        self._checkpointer = None
        self._graph = None

    def healthy(self) -> bool:
        return self._graph is not None

    def _effective_recursion_limit(self, recursion_limit_override: int | None = None) -> int:
        if recursion_limit_override is None:
            return int(self.recursion_limit)
        return max(5, min(int(recursion_limit_override), 200))

    @staticmethod
    def _build_graph_input(
        *,
        user_text: str,
        customer_id: str,
        thread_id: str,
        turn_mode: str,
        pending_context_summary: str,
        trace_id: str,
        skill_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=user_text)],
            "customer_id": customer_id,
            "thread_id": thread_id,
            "turn_mode": _normalize_turn_mode(turn_mode),
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": pending_context_summary,
            "agent_trace_id": trace_id,
            "tool_error_count": 0,
            "approval_handoff": False,
            "claim_check_retry_count": 0,
            "claim_check_needs_retry": False,
            **skill_state,
        }

    async def _prepare_turn_context(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str,
        include_pending_context: bool,
        trace_id: str,
        recursion_limit_override: int | None = None,
    ) -> _PreparedTurnContext | None:
        await self._maybe_compact_thread_context(thread_id=thread_id, customer_id=customer_id)
        if await self._has_pending_approval_lock(customer_id=customer_id, thread_id=thread_id):
            return None
        user_text = str(text or "")
        self.register_links_from_text(
            customer_id=customer_id,
            text=user_text,
            source="user_turn",
            limit=30,
        )
        user_text = self.expand_link_aliases(customer_id=customer_id, text=user_text)
        pending_context_summary, through_id = self._build_pending_context_summary(
            customer_id=customer_id,
            include_pending_context=include_pending_context,
        )
        skill_state = await self._pre_resolve_skill_state(
            customer_id=customer_id,
            user_text=user_text,
        )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._effective_recursion_limit(recursion_limit_override),
        }
        graph_input = self._build_graph_input(
            user_text=user_text,
            customer_id=customer_id,
            thread_id=thread_id,
            turn_mode=turn_mode,
            pending_context_summary=pending_context_summary,
            trace_id=trace_id,
            skill_state=skill_state,
        )
        return _PreparedTurnContext(
            through_id=through_id,
            config=config,
            graph_input=graph_input,
        )

    async def ainvoke_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
        recursion_limit_override: int | None = None,
    ) -> str:
        await self.start()
        assert self._graph is not None
        normalized_turn_mode = _normalize_turn_mode(turn_mode)
        turn_trace_id = new_short_id("turn")
        turn_state, effective_text = await self._thread_inputs.begin_turn(
            thread_id=thread_id, text=text
        )
        if turn_state is None:
            self.log_behavior_event(
                event="turn_merged",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
            )
            return ""
        try:
            self.log_behavior_event(
                event="turn_start",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
                input_chars=len(str(effective_text or "")),
                turn_mode=normalized_turn_mode,
            )
            prepared = await self._prepare_turn_context(
                thread_id=thread_id,
                customer_id=customer_id,
                text=str(effective_text or ""),
                turn_mode=normalized_turn_mode,
                include_pending_context=include_pending_context,
                trace_id=turn_trace_id,
                recursion_limit_override=recursion_limit_override,
            )
            if prepared is None:
                self.log_behavior_event(
                    event="turn_blocked_pending_approval",
                    trace_id=turn_trace_id,
                    mode="ainvoke",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    turn_mode=normalized_turn_mode,
                )
                return ""
            result = await self._graph.ainvoke(prepared.graph_input, config=prepared.config)
            if bool(result.get("approval_handoff", False)) or str(result.get("turn_status", "")) == "approval_pending":
                self.log_behavior_event(
                    event="turn_approval_handoff",
                    trace_id=turn_trace_id,
                    mode="ainvoke",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    turn_mode=normalized_turn_mode,
                )
                handoff_text = self._build_approval_handoff_text(result)
                if handoff_text:
                    self.register_links_from_text(
                        customer_id=customer_id,
                        text=handoff_text,
                        source="assistant_turn",
                        limit=10,
                    )
                return handoff_text
            final_reply = str(result.get("final_response_text", "")).strip()
            if final_reply:
                self.register_links_from_text(
                    customer_id=customer_id,
                    text=final_reply,
                    source="assistant_turn",
                    limit=30,
                )
                cleaned = self.expand_link_aliases(customer_id=customer_id, text=final_reply)
                cleaned, truncated = self._truncate_user_visible_reply(cleaned)
                if truncated:
                    self.log_behavior_event(
                        event="turn_reply_truncated",
                        trace_id=turn_trace_id,
                        mode="ainvoke",
                        thread_id=thread_id,
                        customer_id=customer_id,
                        max_chars=self._max_user_reply_chars,
                        output_chars=len(str(final_reply).strip()),
                        truncated_chars=len(cleaned.strip()),
                    )
                if prepared.through_id is not None and self._context_events is not None:
                    self._context_events.clear_events(customer_id, through_id=prepared.through_id)
                self.log_behavior_event(
                    event="turn_complete",
                    trace_id=turn_trace_id,
                    mode="ainvoke",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    output_chars=len(cleaned.strip()),
                    turn_mode=normalized_turn_mode,
                )
                return cleaned.strip()
            messages = result.get("messages", [])
            for message in reversed(messages):
                if isinstance(message, AIMessage) and (message.content or "").strip():
                    cleaned = str(message.content)
                    self.register_links_from_text(
                        customer_id=customer_id,
                        text=cleaned,
                        source="assistant_turn",
                        limit=30,
                    )
                    cleaned = self.expand_link_aliases(customer_id=customer_id, text=cleaned)
                    cleaned, truncated = self._truncate_user_visible_reply(cleaned)
                    if truncated:
                        self.log_behavior_event(
                            event="turn_reply_truncated",
                            trace_id=turn_trace_id,
                            mode="ainvoke",
                            thread_id=thread_id,
                            customer_id=customer_id,
                            max_chars=self._max_user_reply_chars,
                            output_chars=len(str(message.content or "").strip()),
                            truncated_chars=len(cleaned.strip()),
                        )
                    if prepared.through_id is not None and self._context_events is not None:
                        self._context_events.clear_events(customer_id, through_id=prepared.through_id)
                    self.log_behavior_event(
                        event="turn_complete",
                        trace_id=turn_trace_id,
                        mode="ainvoke",
                        thread_id=thread_id,
                        customer_id=customer_id,
                        output_chars=len(cleaned.strip()),
                        turn_mode=normalized_turn_mode,
                    )
                    return cleaned.strip()
            self.log_behavior_event(
                event="turn_no_visible_reply",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
                turn_mode=normalized_turn_mode,
            )
            return "I ran into an issue and could not produce a final response yet."
        except Exception as exc:
            self.log_behavior_event(
                event="turn_exception",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
                error=str(exc)[:500],
                turn_mode=normalized_turn_mode,
            )
            raise
        finally:
            self._thread_inputs.end_turn(turn_state)

    async def _begin_thread_turn(
        self,
        *,
        thread_id: str,
        text: str,
    ) -> tuple[Any | None, str]:
        """
        Backward-compatible wrapper for tests/internal callers that relied on
        the previous runtime-local turn debounce API.
        """
        return await self._thread_inputs.begin_turn(thread_id=thread_id, text=text)

    @staticmethod
    def _end_thread_turn(state: Any | None) -> None:
        """Backward-compatible wrapper around thread turn release."""
        ThreadInputCoordinator.end_turn(state)

    async def astream_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
    ) -> AsyncIterator[str]:
        await self.start()
        assert self._graph is not None
        normalized_turn_mode = _normalize_turn_mode(turn_mode)
        turn_trace_id = new_short_id("turn")
        turn_state, effective_text = await self._thread_inputs.begin_turn(
            thread_id=thread_id, text=text
        )
        if turn_state is None:
            logger.info(
                "runtime.astream_text merged_input thread_id=%s customer_id=%s",
                thread_id,
                customer_id,
            )
            self.log_behavior_event(
                event="turn_merged",
                trace_id=turn_trace_id,
                mode="astream",
                thread_id=thread_id,
                customer_id=customer_id,
                turn_mode=normalized_turn_mode,
            )
            raise MergedInputSuppressedError("input merged into previous in-flight turn")
        try:
            logger.info(
                "runtime.astream_text start thread_id=%s customer_id=%s text_chars=%s",
                thread_id,
                customer_id,
                len(str(effective_text or "")),
            )
            self.log_behavior_event(
                event="turn_start",
                trace_id=turn_trace_id,
                mode="astream",
                thread_id=thread_id,
                customer_id=customer_id,
                input_chars=len(str(effective_text or "")),
                turn_mode=normalized_turn_mode,
            )
            prepared = await self._prepare_turn_context(
                thread_id=thread_id,
                customer_id=customer_id,
                text=str(effective_text or ""),
                turn_mode=normalized_turn_mode,
                include_pending_context=include_pending_context,
                trace_id=turn_trace_id,
                recursion_limit_override=None,
            )
            if prepared is None:
                yielded_any = True
                self.log_behavior_event(
                    event="turn_blocked_pending_approval",
                    trace_id=turn_trace_id,
                    mode="astream",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    turn_mode=normalized_turn_mode,
                )
                yield STREAM_APPROVAL_HANDOFF_SIGNAL
                return
            config = prepared.config
            segment_accumulated = ""
            stream_key = ""
            yielded_any = False
            saw_agent_output = False
            in_tool_phase = False
            approval_handoff_detected = False
            stream_started_at = time.monotonic()
            stream_no_visible_timeout_s = float(
                str(os.environ.get("AGENT_STREAM_NO_VISIBLE_PROGRESS_SECONDS", "210")).strip()
                or "210"
            )
            stream_total_chunks = 0
            stream_agent_chunks = 0
            stream_tool_chunks = 0
            stream_wait_signals = 0
            stream_visible_yields = 0
            stream_filtered_empty = 0
            stream_filtered_blank_expanded = 0
            first_visible_yield_ms: int | None = None
            self.log_behavior_event(
                event="turn_stream_loop_start",
                trace_id=turn_trace_id,
                thread_id=thread_id,
                customer_id=customer_id,
                stream_no_visible_timeout_s=stream_no_visible_timeout_s,
                turn_mode=normalized_turn_mode,
            )

            def _finalize_segment() -> None:
                nonlocal segment_accumulated
                if not segment_accumulated:
                    return
                cleaned_segment = segment_accumulated
                if cleaned_segment.strip():
                    self.register_links_from_text(
                        customer_id=customer_id,
                        text=cleaned_segment,
                        source="assistant_turn",
                        limit=30,
                    )
                segment_accumulated = ""

            async for message_chunk, metadata in self._graph.astream(
                prepared.graph_input,
                config=config,
                stream_mode="messages",
            ):
                stream_total_chunks += 1
                node_name = str(metadata.get("langgraph_node", "")).strip().lower()
                if stream_total_chunks % 50 == 0:
                    self.log_behavior_event(
                        event="turn_stream_heartbeat",
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        stream_total_chunks=stream_total_chunks,
                        stream_agent_chunks=stream_agent_chunks,
                        stream_tool_chunks=stream_tool_chunks,
                        stream_visible_yields=stream_visible_yields,
                    )
                if node_name != "agent":
                    stream_tool_chunks += 1
                    if node_name == "tools" and self._tool_message_indicates_approval_handoff(message_chunk):
                        approval_handoff_detected = True
                        self.log_behavior_event(
                            event="turn_stream_tool_approval_handoff_detected",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            stream_total_chunks=stream_total_chunks,
                            turn_mode=normalized_turn_mode,
                        )
                    if saw_agent_output and not in_tool_phase:
                        in_tool_phase = True
                        stream_wait_signals += 1
                        self.log_behavior_event(
                            event="turn_stream_wait_signal",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            stream_wait_signals=stream_wait_signals,
                            stream_total_chunks=stream_total_chunks,
                            turn_mode=normalized_turn_mode,
                        )
                        _finalize_segment()
                        yield STREAM_WAIT_SIGNAL
                    if (
                        not yielded_any
                        and stream_no_visible_timeout_s > 0
                        and (time.monotonic() - stream_started_at) >= stream_no_visible_timeout_s
                    ):
                        self.log_behavior_event(
                            event="turn_stream_no_visible_progress_timeout",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            elapsed_ms=int((time.monotonic() - stream_started_at) * 1000),
                            stream_total_chunks=stream_total_chunks,
                            stream_agent_chunks=stream_agent_chunks,
                            stream_tool_chunks=stream_tool_chunks,
                            stream_filtered_empty=stream_filtered_empty,
                            stream_filtered_blank_expanded=stream_filtered_blank_expanded,
                            turn_mode=normalized_turn_mode,
                        )
                        break
                    continue
                stream_agent_chunks += 1
                if in_tool_phase:
                    in_tool_phase = False
                    stream_key = ""
                    _finalize_segment()
                chunk_key = str(getattr(message_chunk, "id", "") or "")
                if chunk_key and stream_key and chunk_key != stream_key:
                    _finalize_segment()
                if chunk_key:
                    stream_key = chunk_key
                if message_chunk.content:
                    saw_agent_output = True
                    segment_accumulated += str(message_chunk.content)
                    cleaned = segment_accumulated
                    if not cleaned.strip():
                        stream_filtered_empty += 1
                        continue
                    expanded = self.expand_link_aliases(customer_id=customer_id, text=cleaned)
                    if expanded.strip():
                        expanded, truncated = self._truncate_user_visible_reply(expanded)
                        yielded_any = True
                        stream_visible_yields += 1
                        if first_visible_yield_ms is None:
                            first_visible_yield_ms = int((time.monotonic() - stream_started_at) * 1000)
                        if stream_visible_yields <= 3 or stream_visible_yields % 20 == 0:
                            self.log_behavior_event(
                                event="turn_stream_chunk_yielded",
                                trace_id=turn_trace_id,
                                thread_id=thread_id,
                                customer_id=customer_id,
                                stream_visible_yields=stream_visible_yields,
                                stream_total_chunks=stream_total_chunks,
                                output_chars=len(expanded.strip()),
                                first_visible_yield_ms=first_visible_yield_ms,
                                turn_mode=normalized_turn_mode,
                            )
                        yield expanded
                        if truncated:
                            self.log_behavior_event(
                                event="turn_stream_reply_truncated",
                                trace_id=turn_trace_id,
                                thread_id=thread_id,
                                customer_id=customer_id,
                                max_chars=self._max_user_reply_chars,
                                output_chars=len(cleaned.strip()),
                                truncated_chars=len(expanded.strip()),
                                turn_mode=normalized_turn_mode,
                            )
                            break
                    else:
                        stream_filtered_blank_expanded += 1

                if (
                    not yielded_any
                    and stream_no_visible_timeout_s > 0
                    and (time.monotonic() - stream_started_at) >= stream_no_visible_timeout_s
                ):
                    self.log_behavior_event(
                        event="turn_stream_no_visible_progress_timeout",
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        elapsed_ms=int((time.monotonic() - stream_started_at) * 1000),
                        stream_total_chunks=stream_total_chunks,
                        stream_agent_chunks=stream_agent_chunks,
                        stream_tool_chunks=stream_tool_chunks,
                        stream_filtered_empty=stream_filtered_empty,
                        stream_filtered_blank_expanded=stream_filtered_blank_expanded,
                        turn_mode=normalized_turn_mode,
                    )
                    break

            if prepared.through_id is not None and self._context_events is not None:
                self._context_events.clear_events(customer_id, through_id=prepared.through_id)
            _finalize_segment()
            if not yielded_any and approval_handoff_detected:
                yielded_any = True
                self.log_behavior_event(
                    event="turn_approval_handoff",
                    trace_id=turn_trace_id,
                    mode="astream",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    turn_mode=normalized_turn_mode,
                )
                yield STREAM_APPROVAL_HANDOFF_SIGNAL
            if not yielded_any:
                logger.warning(
                    "runtime.astream_text no_visible_chunks thread_id=%s customer_id=%s; invoking fallback",
                    thread_id,
                    customer_id,
                )
                self.log_behavior_event(
                    event="turn_stream_no_visible_chunks",
                    trace_id=turn_trace_id,
                    thread_id=thread_id,
                    customer_id=customer_id,
                    elapsed_ms=int((time.monotonic() - stream_started_at) * 1000),
                    stream_total_chunks=stream_total_chunks,
                    stream_agent_chunks=stream_agent_chunks,
                    stream_tool_chunks=stream_tool_chunks,
                    stream_filtered_empty=stream_filtered_empty,
                    stream_filtered_blank_expanded=stream_filtered_blank_expanded,
                    turn_mode=normalized_turn_mode,
                )
                fallback_result = await self._graph.ainvoke(
                    prepared.graph_input,
                    config=config,
                )
                if bool(fallback_result.get("approval_handoff", False)) or str(
                    fallback_result.get("turn_status", "")
                ) == "approval_pending":
                    yielded_any = True
                    self.log_behavior_event(
                        event="turn_approval_handoff",
                        trace_id=turn_trace_id,
                        mode="astream",
                        thread_id=thread_id,
                        customer_id=customer_id,
                        turn_mode=normalized_turn_mode,
                    )
                    yield STREAM_APPROVAL_HANDOFF_SIGNAL
                    fallback_result = {"messages": []}
                fallback_messages = fallback_result.get("messages", [])
                fallback_yielded = False
                fallback_text = str(fallback_result.get("final_response_text", "")).strip()
                if fallback_text:
                    self.register_links_from_text(
                        customer_id=customer_id,
                        text=fallback_text,
                        source="assistant_turn",
                        limit=30,
                    )
                    fallback_text = self.expand_link_aliases(
                        customer_id=customer_id,
                        text=fallback_text,
                    )
                    if fallback_text.strip():
                        fallback_yielded = True
                        self.log_behavior_event(
                            event="turn_stream_fallback_yielded",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            output_chars=len(fallback_text.strip()),
                            turn_mode=normalized_turn_mode,
                        )
                        yield fallback_text.strip()
                for message in reversed(fallback_messages):
                    if fallback_yielded:
                        break
                    if isinstance(message, AIMessage) and (message.content or "").strip():
                        cleaned = str(message.content)
                        if cleaned.strip():
                            self.register_links_from_text(
                                customer_id=customer_id,
                                text=cleaned,
                                source="assistant_turn",
                                limit=30,
                            )
                            cleaned = self.expand_link_aliases(
                                customer_id=customer_id,
                                text=cleaned,
                            )
                            fallback_yielded = True
                            self.log_behavior_event(
                                event="turn_stream_fallback_yielded",
                                trace_id=turn_trace_id,
                                thread_id=thread_id,
                                customer_id=customer_id,
                                output_chars=len(cleaned.strip()),
                                turn_mode=normalized_turn_mode,
                            )
                            yield cleaned.strip()
                            break
                if not fallback_yielded:
                    logger.error(
                        "runtime.astream_text fallback_no_ai_message thread_id=%s customer_id=%s messages_count=%s",
                        thread_id,
                        customer_id,
                        len(fallback_messages),
                    )
                    self.register_links_from_text(
                        customer_id=customer_id,
                        text=STREAM_EMPTY_REPLY_FALLBACK,
                        source="assistant_turn",
                        limit=5,
                    )
                    yielded_any = True
                    self.log_behavior_event(
                        event="turn_stream_fallback_empty",
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        turn_mode=normalized_turn_mode,
                    )
                    yield STREAM_EMPTY_REPLY_FALLBACK
            logger.info(
                "runtime.astream_text complete thread_id=%s customer_id=%s yielded_any=%s",
                thread_id,
                customer_id,
                yielded_any,
            )
            self.log_behavior_event(
                event="turn_complete",
                trace_id=turn_trace_id,
                mode="astream",
                thread_id=thread_id,
                customer_id=customer_id,
                yielded_any=yielded_any,
                elapsed_ms=int((time.monotonic() - stream_started_at) * 1000),
                stream_total_chunks=stream_total_chunks,
                stream_agent_chunks=stream_agent_chunks,
                stream_tool_chunks=stream_tool_chunks,
                stream_wait_signals=stream_wait_signals,
                stream_visible_yields=stream_visible_yields,
                first_visible_yield_ms=first_visible_yield_ms,
                turn_mode=normalized_turn_mode,
            )
        except Exception:
            logger.exception(
                "runtime.astream_text failed thread_id=%s customer_id=%s",
                thread_id,
                customer_id,
            )
            self.log_behavior_event(
                event="turn_exception",
                trace_id=turn_trace_id,
                mode="astream",
                thread_id=thread_id,
                customer_id=customer_id,
                turn_mode=normalized_turn_mode,
            )
            raise
        finally:
            self._thread_inputs.end_turn(turn_state)

    async def classify_wake_event(
        self,
        *,
        customer_id: str,
        event_label: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Let the model decide whether a wake event should interrupt the user now."""
        decision, invoke_error = await self._invoke_structured_model(
            model=self._wake_classifier_model,
            schema=_WakeClassification,
            messages=[
                SystemMessage(
                    content=(
                        "You classify background assistant events.\n"
                        "Return strict JSON with keys: notify_user (bool), reason (string).\n"
                        "Use notify_user=true only when immediate user attention is required."
                    )
                ),
                HumanMessage(
                    content=(
                        f"customer_id={customer_id}\n"
                        f"event_label={event_label}\n"
                        f"payload={json.dumps(payload, ensure_ascii=False)[:5000]}"
                    )
                ),
            ],
        )
        if decision is None or not isinstance(decision, _WakeClassification):
            return {
                "notify_user": False,
                "reason": (
                    f"classifier_error:{invoke_error}"
                    if invoke_error
                    else "classifier_error:invalid_wake_classifier_output"
                ),
            }
        return {
            "notify_user": bool(decision.notify_user),
            "reason": str(decision.reason).strip()[:500],
        }

    async def verify_completion_claim(
        self,
        *,
        user_text: str,
        assistant_text: str,
        recent_tool_outputs: list[str],
        turn_window: str | None = None,
    ) -> dict[str, Any]:
        """
        Verify that immediate-action completion claims are supported by tool evidence.

        This check is intentionally conservative: on uncertainty it should not force a retry.
        """
        safe_assistant = str(assistant_text or "").strip()
        if not safe_assistant:
            return {
                "ok": True,
                "applies": False,
                "mismatch": False,
                "confidence": 0.0,
                "reason": "empty_assistant_text",
                "repair_instruction": "",
                "usable": True,
            }
        safe_user = str(user_text or "").strip()
        safe_turn_window = str(turn_window or "").strip()
        safe_tools: list[str] = []
        for raw in (recent_tool_outputs or []):
            text = " ".join(str(raw or "").split()).strip()
            if text:
                safe_tools.append(text)

        decision, invoke_error = await self._invoke_structured_model(
            model=self._guardrail_classifier_model,
            schema=_ClaimCheckDecision,
            messages=[
                SystemMessage(
                    content=(
                        "You verify assistant execution claims against tool evidence.\n"
                        "Return strict JSON only with keys:\n"
                        "ok (bool), applies (bool), mismatch (bool), confidence (0..1), "
                        "reason (string <= 180 chars), repair_instruction (string <= 220 chars).\n"
                        "Decision policy (conservative, non-aggressive):\n"
                        "- applies=true only if assistant explicitly claims something was already done/launched/sent/posted/scheduled now.\n"
                        "- applies=true if assistant commits to an immediate follow-up action in this same turn "
                        "(e.g., 'doing this now', 'retrying now', 'give me a moment') that should produce tool evidence.\n"
                        "- applies=true if assistant asks the user to approve/deny or says approval is pending now.\n"
                        "- If user_message asks only for an outcome/failure summary, assistant must not promise "
                        "new immediate execution unless tool evidence exists in this turn.\n"
                        "- If assistant is future-tense or conditional without immediate-action claims, set applies=false and mismatch=false.\n"
                        "- mismatch=true only when there is a clear immediate completion claim without matching success evidence in tool outputs.\n"
                        "- mismatch=true when assistant commits immediate follow-up execution now but no matching tool evidence exists.\n"
                        "- If assistant claims completed/updated/created/scheduled now AND also states approval is pending, set mismatch=true.\n"
                        "- If assistant asks for approval (or says approval is pending) but tool evidence lacks a pending-approval artifact "
                        "(e.g., approval_id, APPROVAL_PENDING, or explicit pending challenge), set mismatch=true.\n"
                        "- If evidence is ambiguous/partial, prefer mismatch=false.\n"
                        "- If tool outputs show approval pending, denial, or tool error while assistant claims success now, mismatch=true.\n"
                        "- If assistant claims it fetched/initialized/updated specific content (e.g., named headlines, numbers, concrete facts) "
                        "but those concrete details are not present in tool outputs from this turn, set mismatch=true.\n"
                        "- repair_instruction should tell the agent to either run the missing tool now or restate status honestly.\n"
                        "No markdown. No extra keys."
                    )
                ),
                HumanMessage(
                    content=(
                        f"user_message={safe_user}\n"
                        f"assistant_message={safe_assistant}\n"
                        f"turn_window={safe_turn_window}\n"
                        f"recent_tool_outputs={json.dumps(safe_tools, ensure_ascii=False)}"
                    )
                ),
            ],
        )
        if decision is None or not isinstance(decision, _ClaimCheckDecision):
            return {
                "ok": False,
                "applies": False,
                "mismatch": False,
                "confidence": 0.0,
                "reason": (
                    f"classifier_error:{invoke_error}"
                    if invoke_error
                    else "invalid_checker_output:no_json_object"
                ),
                "repair_instruction": "",
                "usable": False,
            }
        applies = bool(decision.applies)
        mismatch = bool(decision.mismatch) if applies else False
        confidence = float(decision.confidence)
        return {
            "ok": bool(decision.ok),
            "applies": applies,
            "mismatch": mismatch,
            "confidence": max(0.0, min(confidence, 1.0)),
            "reason": str(decision.reason).strip()[:180],
            "repair_instruction": str(decision.repair_instruction).strip()[:220],
            "usable": True,
        }

    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> dict[str, Any]:
        """
        Isolated, compact classifier for approval guardrails.

        Returns strict JSON-like payload:
        {
          "ok": bool,
          "gate": "allow|require_approval|deny",
          "impact_type": "read|write|purchase|costly",
          "recipient_scope": "self|external|unknown",
          "confidence": float,
          "reason": str
        }
        """
        safe_name = str(action_name or "").strip()
        if not safe_name:
            return {"ok": False, "error": "missing_action_name"}

        safe_args: dict[str, Any] = {}
        sensitive_parts = {"key", "token", "secret", "password", "authorization", "api"}
        for key, value in (action_args or {}).items():
            key_text = str(key).strip()
            lower_key = key_text.lower()
            if any(part in lower_key for part in sensitive_parts):
                safe_args[key_text] = "***"
                continue
            if isinstance(value, str):
                if lower_key in {"command", "script", "implementation_command", "code"}:
                    safe_args[key_text] = value[:12000]
                else:
                    safe_args[key_text] = value[:500]
            elif isinstance(value, (int, float, bool)) or value is None:
                safe_args[key_text] = value
            elif isinstance(value, list):
                safe_args[key_text] = [str(item)[:120] for item in value[:12]]
            elif isinstance(value, dict):
                safe_args[key_text] = {
                    str(k)[:40]: str(v)[:120] for k, v in list(value.items())[:12]
                }
            else:
                safe_args[key_text] = str(value)[:200]

        decision, invoke_error = await self._invoke_structured_model(
            model=self._guardrail_classifier_model,
            schema=_GuardrailIntentDecision,
            messages=[
                SystemMessage(
                    content=(
                        "Classify action safety intent for an approval gate.\n"
                        "Return strict JSON object only with keys:\n"
                        "ok (bool), gate (allow|require_approval|deny),\n"
                        "impact_type (read|write|purchase|costly),\n"
                        "recipient_scope (self|external|unknown),\n"
                        "confidence (0..1), reason (string <= 160 chars).\n"
                        "Rules:\n"
                        "- Approval should be required in exactly one case: external write side effects.\n"
                        "- External write means mutating/posting/sending/purchasing/updating data on services "
                        "outside this local project/runtime.\n"
                        "- Treat ANY non-localhost network mutation as external write.\n"
                        "- For shell/terminal commands, classify from literal command intent, not user phrasing.\n"
                        "- If command contains write verbs/flags with remote URLs, set gate=require_approval.\n"
                        "- High-signal external write indicators include: "
                        "curl -X POST|PUT|PATCH|DELETE, --request POST|PUT|PATCH|DELETE, "
                        "--data/-d/--json with http(s) URL, requests.post/put/patch/delete, "
                        "httpx.post/put/patch/delete, fetch(...,{method:'POST'|'PUT'|'PATCH'|'DELETE'}).\n"
                        "- URLs to localhost/127.0.0.1/::1 are local; do not treat as external by URL alone.\n"
                        "- Internal reads/writes (repo files, local artifacts, local config/state) are allow.\n"
                        "- Remote reads/fetch/summarization without external mutation are allow.\n"
                        "- Never set gate=require_approval for read-only actions, including external/API/web "
                        "reads.\n"
                        "- For tulpa_run_terminal, classify from full command/script text in action_args.command.\n"
                        "- For routine_create, evaluate planned downstream behavior from action_args + action_note:\n"
                        "  * inspect implementation_command/implementation fields as the execution artifact.\n"
                        "  * if future scheduled behavior includes external writes, set gate=require_approval.\n"
                        "  * otherwise set gate=allow.\n"
                        "- For non-routine actions, set gate=require_approval only when this immediate action "
                        "implies external write side effects.\n"
                        "- If uncertain on a command that includes a non-localhost URL plus write-like markers, "
                        "escalate to require_approval.\n"
                        "- If uncertain without write-like markers, set gate=allow with recipient_scope=unknown "
                        "or self as appropriate.\n"
                        "- Use deny only for actions that should never run as requested.\n"
                        "- Treat action_note as agent reasoning about next planned action and likely tool path.\n"
                        "Do not include any extra keys or markdown."
                    )
                ),
                HumanMessage(
                    content=(
                        f"action_name={safe_name}\n"
                        f"action_args={json.dumps(safe_args, ensure_ascii=False)[:20000]}\n"
                        f"action_note={str(action_note or '').strip()[:2000]}"
                    )
                ),
            ],
        )
        if decision is None or not isinstance(decision, _GuardrailIntentDecision):
            detail = invoke_error or "invalid_guardrail_output"
            return {"ok": False, "error": f"classifier_error:{detail}"}

        gate = str(decision.gate).strip().lower()
        impact_type = str(decision.impact_type).strip().lower()
        recipient_scope = str(decision.recipient_scope).strip().lower()
        if gate not in {"allow", "require_approval", "deny"}:
            return {"ok": False, "error": "invalid_gate"}
        if impact_type not in {"read", "write", "purchase", "costly"}:
            return {"ok": False, "error": "invalid_impact_type"}
        if recipient_scope not in {"self", "external", "unknown"}:
            return {"ok": False, "error": "invalid_recipient_scope"}
        return {
            "ok": bool(decision.ok),
            "gate": gate,
            "impact_type": impact_type,
            "recipient_scope": recipient_scope,
            "confidence": max(0.0, min(float(decision.confidence), 1.0)),
            "reason": str(decision.reason).strip()[:160],
        }

    async def evaluate_tool_guardrail(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> dict[str, Any]:
        """Call upstream approval broker to evaluate a tool call at action time."""
        safe_cmd = ""
        if action_name == "tulpa_run_terminal":
            safe_cmd = str((action_args or {}).get("command", "")).strip()[:300]
        try:
            response = await self._request_with_backoff(
                "POST",
                "/internal/approvals/evaluate",
                json_body={
                    "customer_id": customer_id,
                    "thread_id": thread_id,
                    "action_name": action_name,
                    "action_args": action_args if isinstance(action_args, dict) else {},
                    "action_note": str(action_note or "").strip()[:2000],
                    "defer_challenge_delivery": True,
                },
                timeout=12.0,
                retries=1,
            )
            if response.status_code != 200:
                self.log_behavior_event(
                    event="guardrail.evaluate.http_error",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    action_name=action_name,
                    command=safe_cmd,
                    status_code=response.status_code,
                    gate="require_approval",
                )
                return {
                    "gate": "require_approval",
                    "reason": f"guardrail_http_{response.status_code}",
                    "summary": f"execute {action_name}",
                }
            payload = response.json()
            if isinstance(payload, dict):
                self.log_behavior_event(
                    event="guardrail.evaluate.decision",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    action_name=action_name,
                    command=safe_cmd,
                    gate=str(payload.get("gate", "")),
                    reason=str(payload.get("reason", ""))[:200],
                    impact_type=str(payload.get("impact_type", "")),
                    recipient_scope=str(payload.get("recipient_scope", "")),
                    confidence=payload.get("confidence"),
                )
                return payload
            self.log_behavior_event(
                event="guardrail.evaluate.invalid_payload",
                thread_id=thread_id,
                customer_id=customer_id,
                action_name=action_name,
                command=safe_cmd,
                gate="require_approval",
            )
            return {
                "gate": "require_approval",
                "reason": "guardrail_invalid_payload",
                "summary": f"execute {action_name}",
            }
        except Exception as exc:
            exc_name = type(exc).__name__
            self.log_behavior_event(
                event="guardrail.evaluate.exception",
                thread_id=thread_id,
                customer_id=customer_id,
                action_name=action_name,
                command=safe_cmd,
                gate="require_approval",
                error=f"{exc_name}: {exc}",
            )
            return {
                "gate": "require_approval",
                "reason": f"guardrail_request_error:{exc_name}",
                "summary": f"execute {action_name}",
            }

    async def _request_with_backoff(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        return await self._internal_api.request_with_backoff(
            method=method,
            path=path,
            params=params,
            json_body=json_body,
            timeout=timeout,
            retries=retries,
        )

    def _register_tools(self) -> None:
        self._tools = register_runtime_tools(self)

    def _build_graph(self):
        return build_runtime_graph(self)

    def set_active_customer_id(self, customer_id: str):
        cid = str(customer_id or "").strip()
        token = self._active_customer_id_ctx.set(cid)
        self._active_customer_id = cid
        return token

    def reset_active_customer_id(self, token: object) -> None:
        self._active_customer_id_ctx.reset(token)
        self._active_customer_id = str(self._active_customer_id_ctx.get() or "").strip()

    def get_active_customer_id(self) -> str:
        return str(self._active_customer_id_ctx.get() or "").strip()

    async def execute_tool(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
        customer_id: str | None = None,
        inject_customer_id: bool = False,
    ) -> Any:
        """
        Public runtime API for tool execution outside normal graph turns.

        Used by approval execution to avoid coupling to private runtime attributes.
        """
        await self.start()
        self.log_behavior_event(
            event="tool_execute_start",
            action_name=str(action_name or "").strip(),
            customer_id=str(customer_id or "").strip(),
        )
        tool_fn = self._tools.get(str(action_name or "").strip())
        if tool_fn is None:
            self.log_behavior_event(
                event="tool_execute_missing",
                action_name=str(action_name or "").strip(),
                customer_id=str(customer_id or "").strip(),
            )
            raise RuntimeError(f"unknown tool: {action_name}")
        cid = str(customer_id or "").strip()
        args = dict(action_args) if isinstance(action_args, dict) else {}
        args.pop("customer_id", None)
        if inject_customer_id and str(action_name or "").strip() in APPROVAL_EXECUTION_CUSTOMER_ID_TOOLS and not cid:
            raise RuntimeError(f"customer_id is required for tool: {action_name}")
        args = self.resolve_link_aliases_in_args(
            customer_id=cid,
            args=args,
        )
        token = self.set_active_customer_id(cid)
        try:
            result = await tool_fn.ainvoke(args)
        except Exception as exc:
            self.log_behavior_event(
                event="tool_execute_error",
                action_name=str(action_name or "").strip(),
                customer_id=str(customer_id or "").strip(),
                error=str(exc)[:500],
            )
            raise
        finally:
            self.reset_active_customer_id(token)
        if cid:
            self.register_links_from_text(
                customer_id=cid,
                text=json.dumps(result, ensure_ascii=False, default=str),
                source=f"tool:{action_name}",
                limit=40,
            )
        self.log_behavior_event(
            event="tool_execute_complete",
            action_name=str(action_name or "").strip(),
            customer_id=str(customer_id or "").strip(),
            result_ok=(not isinstance(result, dict) or bool(result.get("ok", True))),
        )
        return result
