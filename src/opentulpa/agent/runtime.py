"""
In-process LangGraph runtime for OpenTulpa.

This runs the agent in-process with a local StateGraph that:
- runs tool-calling in a bounded loop,
- persists thread state via SQLite checkpointer,
- supports token streaming for Telegram,
- and reuses existing /internal/* APIs as tool backends.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
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
    split_rollup_sections as _split_rollup_sections,
)
from opentulpa.agent.context_compaction import (
    split_text_chunks as _split_text_chunks,
)
from opentulpa.agent.context_engineer import ContextEngineer
from opentulpa.agent.context_engineer import (
    trim_text_to_token_budget as _trim_text_to_token_budget,
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
from opentulpa.agent.prompt_classifier import classify_prompt_mode as _classify_prompt_mode
from opentulpa.agent.runtime_input import (
    MergedInputSuppressedError,
    ThreadInputCoordinator,
)
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.agent.turn_policy import (
    normalize_turn_mode as _normalize_turn_mode,
)
from opentulpa.agent.utils import (
    approx_tokens as _approx_tokens,
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
    safe_json as _safe_json,
)
from opentulpa.agent.utils import (
    utc_offset_to_minutes as _utc_offset_to_minutes,
)
from opentulpa.context.customer_profile_models import (
    CustomerScopedRequest,
    DirectiveGetResponse,
    TimeProfileGetResponse,
)
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.context.link_aliases import LinkAliasService
from opentulpa.context.service import EventContextService
from opentulpa.context.thread_rollups import ThreadRollupService
from opentulpa.core.ids import new_short_id
from opentulpa.logging import create_posthog_logger
from opentulpa.memory.service import MEMORY_KIND_PRIORITY

logger = logging.getLogger(__name__)

_MEMORY_GROUNDING_KIND_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("preferences_and_directives", ("directive_fact", "preference_fact")),
    ("durable_personal_facts", ("user_profile_fact", "life_fact", "relationship_fact", "contact_fact")),
    ("aspirations_and_plans", ("aspirations_fact", "project_fact")),
    ("active_projects_or_workflows", ("workflow_fact", "skill_fact")),
    ("technical_or_code_facts", ("code_fact", "credential_fact")),
    ("relevant_files_or_media", ("file_fact", "media_fact")),
    ("fallback_thread_context", ("thread_context_rollup",)),
)

_LLM_CALL_TRACE_LIMIT = 100
_DEFAULT_OPENROUTER_APP_REFERER = "https://github.com/kvyb/opentulpa"
_DEFAULT_OPENROUTER_APP_TITLE = "OpenTulpa"


def _redact_inline_trace_string(value: str) -> str:
    raw = str(value)
    if not raw.lower().startswith("data:"):
        return raw
    if ";base64," in raw:
        prefix, _, _ = raw.partition(";base64,")
        return f"{prefix};base64,[redacted]"
    prefix, _, _ = raw.partition(",")
    return f"{prefix},[redacted]"


def _prompt_cache_control_payload(*, ttl_1h: bool) -> dict[str, Any]:
    cc: dict[str, Any] = {"type": "ephemeral"}
    if ttl_1h:
        cc["ttl"] = "1h"
    return cc


def _provider_prompt_cache_profile(
    *,
    enabled: bool,
    model_name: str,
    ttl_1h: bool,
) -> dict[str, Any]:
    slug = (model_name or "").strip().lower()
    if not enabled:
        return {
            "enabled": False,
            "strategy": "disabled",
            "supports_top_level": False,
            "supports_breakpoints": False,
            "cache_control": {},
            "model_name": model_name,
        }
    if "anthropic/" in slug or "claude" in slug:
        return {
            "enabled": True,
            "strategy": "top_level",
            "supports_top_level": True,
            "supports_breakpoints": True,
            "cache_control": _prompt_cache_control_payload(ttl_1h=ttl_1h),
            "model_name": model_name,
        }
    if "gemini" in slug or slug.startswith("google/"):
        return {
            "enabled": True,
            "strategy": "breakpoint",
            "supports_top_level": False,
            "supports_breakpoints": True,
            "cache_control": _prompt_cache_control_payload(ttl_1h=ttl_1h),
            "model_name": model_name,
        }
    if any(
        marker in slug
        for marker in (
            "openai/",
            "gpt-",
            "o1",
            "o3",
            "o4",
            "deepseek",
            "grok",
            "x-ai/",
            "moonshot",
            "kimi",
            "groq/",
        )
    ):
        return {
            "enabled": True,
            "strategy": "automatic",
            "supports_top_level": False,
            "supports_breakpoints": False,
            "cache_control": {},
            "model_name": model_name,
        }
    return {
        "enabled": True,
        "strategy": "unknown",
        "supports_top_level": False,
        "supports_breakpoints": False,
        "cache_control": {},
        "model_name": model_name,
    }


def _provider_prompt_cache_invoke_extras(
    *,
    enabled: bool,
    model_name: str,
    ttl_1h: bool,
) -> dict[str, Any]:
    """
    Provider-specific request extras for prompt caching.

    OpenRouter currently accepts top-level `cache_control` for Anthropic Claude.
    Other providers either cache automatically or require per-message breakpoints.
    """
    profile = _provider_prompt_cache_profile(
        enabled=enabled,
        model_name=model_name,
        ttl_1h=ttl_1h,
    )
    if profile.get("strategy") != "top_level":
        return {}
    return {"extra_body": {"cache_control": dict(profile.get("cache_control") or {})}}


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(existing, value)
            continue
        merged[key] = value
    return merged


def _looks_like_openrouter_base_url(base_url: str | None) -> bool:
    normalized = str(base_url or "").strip().lower()
    return "openrouter.ai" in normalized


def _is_glm_51_model_name(model_name: str | None) -> bool:
    normalized = str(model_name or "").strip().lower()
    return normalized.startswith("z-ai/glm-5.1")


_GLM_51_OPENROUTER_PROVIDER_ORDER = [
    "fireworks",
    "siliconflow",
    "friendli",
    "inceptron",
    "atlas-cloud",
]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return _redact_inline_trace_string(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        safe = {str(k): _json_safe(v) for k, v in value.items()}
        if str(safe.get("type", "") or "").strip().lower() == "input_audio":
            input_audio = safe.get("input_audio")
            if isinstance(input_audio, dict) and str(input_audio.get("data", "") or "").strip():
                safe_input_audio = dict(input_audio)
                safe_input_audio["data"] = "[redacted]"
                safe["input_audio"] = safe_input_audio
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        with suppress(Exception):
            return _json_safe(model_dump())
    return _redact_inline_trace_string(str(value))


def _openrouter_app_headers(
    *,
    base_url: str | None,
    env: dict[str, str] | None = None,
) -> dict[str, str]:
    if not _looks_like_openrouter_base_url(base_url):
        return {}
    source = env if env is not None else os.environ
    title = str(source.get("OPENROUTER_APP_TITLE", "")).strip() or _DEFAULT_OPENROUTER_APP_TITLE
    headers: dict[str, str] = {}
    headers["HTTP-Referer"] = _DEFAULT_OPENROUTER_APP_REFERER
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _message_role(message: Any) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, SystemMessage):
        return "system"
    if isinstance(message, ToolMessage):
        return "tool"
    return "unknown"


def _serialize_message(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", "")
    safe_content = _json_safe(content)
    safe_text = _content_to_text(safe_content)
    payload: dict[str, Any] = {
        "role": _message_role(message),
        "type": type(message).__name__,
        "content": safe_content,
        "text": safe_text,
        "approx_tokens": _approx_tokens(safe_text),
    }
    tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = _json_safe(tool_calls)
    name = str(getattr(message, "name", "") or "").strip()
    if name:
        payload["name"] = name
    additional_kwargs = getattr(message, "additional_kwargs", None)
    if additional_kwargs:
        payload["additional_kwargs"] = _json_safe(additional_kwargs)
    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        payload["response_metadata"] = _json_safe(response_metadata)
    return payload


def _message_content_with_cache_breakpoint(
    content: Any,
    *,
    cache_control: dict[str, Any],
) -> Any:
    if isinstance(content, str):
        text = str(content)
        if not text.strip():
            return content
        return [{"type": "text", "text": text, "cache_control": dict(cache_control)}]
    if not isinstance(content, list):
        return content
    updated = list(content)
    for idx in range(len(updated) - 1, -1, -1):
        item = updated[idx]
        if isinstance(item, str):
            text = str(item)
            if not text.strip():
                continue
            updated[idx] = {"type": "text", "text": text, "cache_control": dict(cache_control)}
            return updated
        if isinstance(item, dict):
            item_type = str(item.get("type", "")).strip().lower()
            if item_type != "text" or "cache_control" in item:
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            patched = dict(item)
            patched["cache_control"] = dict(cache_control)
            updated[idx] = patched
            return updated
    return content


def _message_with_cache_breakpoint(message: Any, *, cache_control: dict[str, Any]) -> Any:
    content = _message_content_with_cache_breakpoint(
        getattr(message, "content", None),
        cache_control=cache_control,
    )
    if content == getattr(message, "content", None):
        return message
    model_copy = getattr(message, "model_copy", None)
    copied = model_copy(deep=True) if callable(model_copy) else message.copy(deep=True)
    copied.content = content
    return copied


def _infer_stable_system_prefix_count(messages: list[Any]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, SystemMessage):
            break
        if not _content_to_text(getattr(message, "content", "")).strip():
            break
        count += 1
    return count


def _supports_ainvoke_kwargs(target: Any, kwargs: dict[str, Any]) -> bool:
    if not kwargs:
        return False
    ainvoke = getattr(target, "ainvoke", None)
    if not callable(ainvoke):
        return False
    try:
        sig = inspect.signature(ainvoke)
    except (TypeError, ValueError):
        return False
    params = sig.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params):
        return True
    return all(key in sig.parameters for key in kwargs)


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _usage_object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    result: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens",
        "output_tokens",
    ):
        item = getattr(value, key, None)
        if item is not None:
            result[key] = item
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        with suppress(Exception):
            dumped = model_dump()
            if isinstance(dumped, dict):
                result.update(dumped)
    return result


def _extract_response_usage_fields(response: Any) -> dict[str, Any]:
    usage = _usage_object_to_dict(getattr(response, "usage", None))
    response_metadata = getattr(response, "response_metadata", None)
    if not usage and isinstance(response_metadata, dict):
        usage = _usage_object_to_dict(response_metadata.get("usage"))
        token_usage = response_metadata.get("token_usage")
        if not usage and isinstance(token_usage, dict):
            usage = {
                "prompt_tokens": token_usage.get("prompt_tokens"),
                "completion_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
                "prompt_tokens_details": token_usage.get("prompt_tokens_details"),
                "completion_tokens_details": token_usage.get("completion_tokens_details"),
            }
    if not usage:
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(usage_metadata, dict) and usage_metadata:
            usage = {
                "prompt_tokens": usage_metadata.get("input_tokens"),
                "completion_tokens": usage_metadata.get("output_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
                "input_tokens": usage_metadata.get("input_tokens"),
                "output_tokens": usage_metadata.get("output_tokens"),
            }

    prompt_details = _usage_object_to_dict(usage.get("prompt_tokens_details"))
    completion_details = _usage_object_to_dict(usage.get("completion_tokens_details"))
    prompt_tokens = _maybe_int(usage.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _maybe_int(usage.get("input_tokens"))
    completion_tokens = _maybe_int(usage.get("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _maybe_int(usage.get("output_tokens"))
    total_tokens = _maybe_int(usage.get("total_tokens"))
    cached_tokens = _maybe_int(prompt_details.get("cached_tokens"))
    cache_write_tokens = _maybe_int(prompt_details.get("cache_write_tokens"))
    reasoning_tokens = _maybe_int(completion_details.get("reasoning_tokens"))
    cost = usage.get("cost")
    cost_details = _usage_object_to_dict(usage.get("cost_details"))

    fields: dict[str, Any] = {}
    if prompt_tokens is not None:
        fields["native_tokens_prompt"] = prompt_tokens
    if completion_tokens is not None:
        fields["native_tokens_completion"] = completion_tokens
    if total_tokens is not None:
        fields["native_tokens_total"] = total_tokens
    if cached_tokens is not None:
        fields["native_tokens_cached"] = cached_tokens
        fields["cache_hit"] = cached_tokens > 0
    if cache_write_tokens is not None:
        fields["native_tokens_cache_write"] = cache_write_tokens
    if reasoning_tokens is not None:
        fields["native_tokens_reasoning"] = reasoning_tokens
    with suppress(Exception):
        if cost not in (None, ""):
            fields["native_cost_usd"] = float(cost)
    if cost_details:
        fields["native_cost_details"] = cost_details
        with suppress(Exception):
            if cost_details.get("prompt") not in (None, ""):
                fields["native_cost_prompt_usd"] = float(cost_details.get("prompt"))
        with suppress(Exception):
            if cost_details.get("completion") not in (None, ""):
                fields["native_cost_completion_usd"] = float(cost_details.get("completion"))
    return fields
_LINK_ID_TOKEN_RE = re.compile(r"\blink_[A-Za-z0-9]{4,12}\b")
STREAM_WAIT_SIGNAL = "__TULPA_STREAM_WAIT__"
STREAM_APPROVAL_HANDOFF_SIGNAL = "__TULPA_APPROVAL_HANDOFF__"
STREAM_PROGRESS_PREFIX = "__TULPA_STREAM_PROGRESS__:"
STREAM_EMPTY_REPLY_FALLBACK = (
    "I couldn't produce a visible user-facing reply for that step. "
    "Please retry, and I will continue from the latest state."
)
STREAM_PRECOMMIT_SECONDS = 0.75
_PROVISIONAL_REPLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*i can also\s+", re.IGNORECASE),
    re.compile(r"^\s*i can (?:search|check|look|fetch|inspect|try)\b", re.IGNORECASE),
    re.compile(r"^\s*let me\b", re.IGNORECASE),
    re.compile(r"^\s*i(?:'| a)?ll\b", re.IGNORECASE),
    re.compile(r"^\s*i will\b", re.IGNORECASE),
    re.compile(r"^\s*(?:one sec|one second|still working|working on it)\b", re.IGNORECASE),
    re.compile(r"\bthis will take\b", re.IGNORECASE),
)
_PROGRESS_TOOL_NAME_ALIASES: dict[str, str] = {
    "tulpa_read_file": "Reading a file",
    "tulpa_write_file": "Writing a file",
    "tulpa_validate_file": "Validating a file",
    "tulpa_run_terminal": "Running a terminal command",
    "skill_get": "Loading a skill",
    "skill_list": "Checking available skills",
    "web_search": "Searching the web",
    "fetch_url_content": "Fetching a webpage",
    "fetch_file_content": "Fetching a file",
    "browser_use_run": "Using the browser",
}

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
    "time_profile_get",
    "time_profile_set",
    "routine_list",
    "routine_create",
    "routine_delete",
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


class _IntakeWorkflowDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    matches_workflow: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    conversation_summary: str = ""
    extracted_fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    reply_action: str = "none"
    reply_text: str = ""
    ready_to_save: bool = False
    booking_action: str = "ignore"
    save_payload: dict[str, Any] = Field(default_factory=dict)
    sink_arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


@dataclass(slots=True)
class _PreparedTurnContext:
    through_id: int | None
    config: dict[str, Any]
    graph_input: dict[str, Any]


def _build_intake_workflow_system_prompt() -> str:
    return (
        "You operate an autonomous business intake workflow over external DM conversations.\n"
        "Your job is to classify whether the conversation matches the workflow, extract reliable fields, "
        "decide whether to ask a follow-up question, decide whether the booking is ready to save, and "
        "decide whether the latest customer message updates an active booking, edits a recent completed booking, "
        "starts a new booking, or should be ignored.\n\n"
        "Return strict JSON only with keys:\n"
        "matches_workflow (bool), confidence (0..1), conversation_summary (string), "
        "extracted_fields (object), missing_fields (string array), reply_action (string), "
        "reply_text (string), ready_to_save (bool), booking_action (string), "
        "save_payload (object), sink_arguments (object), reason (string).\n\n"
        "Allowed booking_action values: ignore, update_active, edit_recent_completed, create_new_booking.\n"
        "Allowed reply_action values: none, send_reply, mark_cancelled.\n\n"
        "Decision policy:\n"
        "- Be precise. False positives are worse than ignoring an unrelated DM.\n"
        "- Use matches_workflow=true only when the customer is clearly pursuing the workflow intent now.\n"
        "- If the message is ambiguous, casual, social, or not clearly about the workflow, return matches_workflow=false, booking_action=ignore, reply_action=none.\n"
        "- Confidence should reflect how certain you are in the match and booking decision.\n"
        "- Confidence guide: 0.9+ very clear, 0.7-0.89 likely, 0.4-0.69 ambiguous, below 0.4 weak evidence.\n\n"
        "Field extraction policy:\n"
        "- Extract fields only from evidence in the conversation or saved state.\n"
        "- Do not invent or infer missing business details unless the value is explicitly or near-explicitly stated.\n"
        "- Light normalization is allowed: trim whitespace, standardize obvious time/date phrasing, preserve meaning.\n"
        "- If customer messages conflict, prefer the latest customer-provided value unless the newer message is too vague to override the earlier one.\n"
        "- Do not ask for a field that is already reliably known unless the value is conflicting or unclear.\n"
        "- missing_fields must list only fields that are truly still needed before save.\n\n"
        "Reply policy:\n"
        "- If details are missing, set reply_action=send_reply with one concise, high-leverage follow-up question.\n"
        "- Ask at most one compact question at a time unless a single sentence can naturally request two tightly related missing fields.\n"
        "- reply_text should be plain outbound DM text, not explanations about JSON or system behavior.\n"
        "- If no reply is needed, use reply_action=none and reply_text=\"\".\n"
        "- Use mark_cancelled only when the customer clearly cancels, abandons, or says they no longer want the booking.\n"
        "- Never ask for Telegram approval. This is background workflow execution.\n\n"
        "Booking action policy:\n"
        "- If there is an active booking and the customer is continuing the same request, use update_active.\n"
        "- If there is a recent completed booking inside the edit window and the customer is correcting or changing that booking, use edit_recent_completed.\n"
        "- If the previous booking is done and the customer is clearly starting another separate booking, use create_new_booking.\n"
        "- If the conversation does not currently require workflow action, use ignore.\n\n"
        "Recovery policy:\n"
        "- execution_feedback, when present in the human message, describes a real failure from the last attempted action.\n"
        "- Do not repeat the same failing action unchanged if execution_feedback shows it already failed.\n"
        "- Replan using the error details. For example, change reply wording, inspect the sink with tools, "
        "provide missing sink arguments, avoid an invalid save, or ask a clarifying question instead.\n\n"
        "Save policy:\n"
        "- Set ready_to_save=true only when all required fields are available with enough clarity to create/update the booking.\n"
        "- When ready_to_save=true, save_payload must contain the merged final field set that should be persisted now.\n"
        "- sink_arguments may contain sink-tool arguments or overrides discovered via tools or context "
        "(for example sheetName for Google Sheets). These are merged into the final sink write.\n"
        "- When ready_to_save=false, save_payload should usually be empty.\n"
        "- When no sink overrides are needed, sink_arguments should usually be empty.\n"
        "- conversation_summary should be a short operational summary of what the customer currently wants.\n"
        "- reason should briefly explain the match decision and booking_action.\n\n"
        "Examples:\n"
        "1. Customer asks for a wash, gives day and car type, but no time -> matches_workflow=true, booking_action=create_new_booking or update_active, reply_action=send_reply, missing_fields includes time, ready_to_save=false.\n"
        "2. Customer says 'actually make it 4pm instead' after a recent completed booking -> matches_workflow=true, booking_action=edit_recent_completed, extracted_fields.time='4pm'.\n"
        "3. Customer says 'also book my other car tomorrow evening' after an earlier finished booking -> matches_workflow=true, booking_action=create_new_booking.\n"
        "4. Customer only reacts with 'thanks' or sends unrelated chat -> matches_workflow=false, booking_action=ignore, reply_action=none.\n"
        "No markdown. No extra keys."
    )


def _trim_text_chars(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max(0, int(limit)):
        return text
    if limit <= 3:
        return text[: max(0, int(limit))]
    return text[: max(0, int(limit) - 3)].rstrip() + "..."


def _compact_jsonish_dict(
    value: Any,
    *,
    item_limit: int = 8,
    char_limit: int = 120,
) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_value in list(value.items())[: max(0, int(item_limit))]:
        key = str(raw_key or "").strip()
        if not key:
            continue
        rendered = (
            json.dumps(raw_value, ensure_ascii=False, sort_keys=True)
            if isinstance(raw_value, (dict, list))
            else str(raw_value or "")
        )
        out[key] = _trim_text_chars(rendered, limit=char_limit)
    return out


def _compact_workflow_for_prompt(workflow: dict[str, Any]) -> dict[str, Any]:
    safe_workflow = workflow if isinstance(workflow, dict) else {}
    field_guidance = safe_workflow.get("field_guidance")
    guidance_map = field_guidance if isinstance(field_guidance, dict) else {}
    compact_guidance = {
        str(key or "").strip(): _trim_text_chars(value, limit=120)
        for key, value in guidance_map.items()
        if str(key or "").strip()
    }
    sink_config = safe_workflow.get("sink_config")
    safe_sink_config = sink_config if isinstance(sink_config, dict) else {}
    compact_sink: dict[str, Any] = {}
    toolkit = str(safe_sink_config.get("toolkit", "") or "").strip()
    if toolkit:
        compact_sink["toolkit"] = toolkit
    operation_hint = str(safe_sink_config.get("operation_hint", "") or "").strip()
    if operation_hint:
        compact_sink["operation_hint"] = operation_hint
    field_mapping = safe_sink_config.get("field_mapping")
    if isinstance(field_mapping, dict) and field_mapping:
        compact_sink["field_mapping_keys"] = [
            str(key or "").strip()
            for key in list(field_mapping.keys())[:8]
            if str(key or "").strip()
        ]
    static_arguments = safe_sink_config.get("static_arguments")
    if isinstance(static_arguments, dict) and static_arguments:
        compact_sink["static_argument_keys"] = [
            str(key or "").strip()
            for key in list(static_arguments.keys())[:8]
            if str(key or "").strip()
        ]
        compact_sink["static_arguments"] = _compact_jsonish_dict(static_arguments)
    knowledge_files: list[dict[str, Any]] = []
    for item in list(safe_workflow.get("knowledge_files") or [])[:6]:
        if not isinstance(item, dict):
            continue
        knowledge_files.append(
            {
                "id": str(item.get("id", "") or "").strip(),
                "filename": _trim_text_chars(item.get("original_filename", ""), limit=80),
                "summary": _trim_text_chars(item.get("summary", ""), limit=220),
            }
        )
    return {
        "workflow_id": str(safe_workflow.get("workflow_id", "") or "").strip(),
        "name": _trim_text_chars(safe_workflow.get("name", ""), limit=80),
        "intent_description": _trim_text_chars(
            safe_workflow.get("intent_description", ""),
            limit=500,
        ),
        "required_fields": [
            str(item or "").strip()
            for item in list(safe_workflow.get("required_fields") or [])[:12]
            if str(item or "").strip()
        ],
        "field_guidance": compact_guidance,
        "assistant_instructions": _trim_text_chars(
            safe_workflow.get("assistant_instructions", ""),
            limit=400,
        ),
        "knowledge_file_ids": [
            str(item or "").strip()
            for item in list(safe_workflow.get("knowledge_file_ids") or [])[:12]
            if str(item or "").strip()
        ],
        "knowledge_files": knowledge_files,
        "sink_type": str(safe_workflow.get("sink_type", "") or "").strip(),
        "channel": str(safe_workflow.get("channel", "") or "").strip(),
        "provider": str(safe_workflow.get("provider", "") or "").strip(),
        "sink": compact_sink,
        "policies": safe_workflow.get("policies", {})
        if isinstance(safe_workflow.get("policies"), dict)
        else {},
    }


def _compact_recent_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    compact: list[dict[str, str]] = []
    for item in messages[-6:]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "id": _trim_text_chars(item.get("id", ""), limit=80),
                "created_time": _trim_text_chars(item.get("created_time", ""), limit=64),
                "sender_role": _trim_text_chars(item.get("sender_role", ""), limit=24),
                "sender_username": _trim_text_chars(item.get("sender_username", ""), limit=48),
                "text": _trim_text_chars(item.get("text", ""), limit=300),
            }
        )
    return compact


def _compact_booking_for_prompt(booking: dict[str, Any] | None) -> dict[str, Any]:
    safe_booking = booking if isinstance(booking, dict) else {}
    extracted_fields = safe_booking.get("extracted_fields")
    safe_fields = extracted_fields if isinstance(extracted_fields, dict) else {}
    compact_fields = {
        str(key or "").strip(): _trim_text_chars(value, limit=80)
        for key, value in list(safe_fields.items())[:12]
        if str(key or "").strip()
    }
    return {
        "booking_id": str(safe_booking.get("booking_id", "") or "").strip(),
        "status": str(safe_booking.get("status", "") or "").strip(),
        "opened_at": str(safe_booking.get("opened_at", "") or "").strip(),
        "completed_at": str(safe_booking.get("completed_at", "") or "").strip(),
        "edit_window_until": str(safe_booking.get("edit_window_until", "") or "").strip(),
        "extracted_fields": compact_fields,
    }


def _compact_execution_feedback(feedback: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in list(feedback or [])[-2:]:
        if not isinstance(item, dict):
            continue
        prior = item.get("prior_decision")
        compact.append(
            {
                "phase": _trim_text_chars(item.get("phase", ""), limit=80),
                "error": _trim_text_chars(item.get("error", ""), limit=400),
                "prior_decision": prior if isinstance(prior, dict) else {},
            }
        )
    return compact


def _compact_conversation_for_prompt(conversation: dict[str, Any]) -> dict[str, Any]:
    safe_conversation = conversation if isinstance(conversation, dict) else {}
    summary = safe_conversation.get("summary")
    safe_summary = summary if isinstance(summary, dict) else {}
    compact_summary = {
        "conversation_id": _trim_text_chars(safe_summary.get("conversation_id", ""), limit=120),
        "recipient_id": _trim_text_chars(safe_summary.get("recipient_id", ""), limit=120),
        "latest_inbound_message_id": _trim_text_chars(
            safe_summary.get("latest_inbound_message_id", ""),
            limit=120,
        ),
        "latest_inbound_message_created_time": _trim_text_chars(
            safe_summary.get("latest_inbound_message_created_time", ""),
            limit=64,
        ),
        "latest_inbound_sender_username": _trim_text_chars(
            safe_summary.get("latest_inbound_sender_username", ""),
            limit=64,
        ),
        "latest_inbound_message_text_preview": _trim_text_chars(
            safe_summary.get("latest_inbound_message_text_preview", ""),
            limit=180,
        ),
        "latest_outbound_message_id": _trim_text_chars(
            safe_summary.get("latest_outbound_message_id", ""),
            limit=120,
        ),
        "latest_outbound_message_created_time": _trim_text_chars(
            safe_summary.get("latest_outbound_message_created_time", ""),
            limit=64,
        ),
        "conversation_updated_time": _trim_text_chars(
            safe_summary.get("conversation_updated_time", ""),
            limit=64,
        ),
    }
    return {
        "summary": compact_summary,
        "recent_messages": _compact_recent_messages(safe_conversation.get("recent_messages")),
    }


def _build_intake_workflow_agent_prompt(
    *,
    customer_id: str,
    workflow: dict[str, Any],
    conversation: dict[str, Any],
    active_booking: dict[str, Any] | None,
    recent_completed_booking: dict[str, Any] | None,
    execution_feedback: list[dict[str, Any]] | None = None,
) -> str:
    compact_workflow = _compact_workflow_for_prompt(workflow)
    compact_conversation = _compact_conversation_for_prompt(conversation)
    compact_active_booking = _compact_booking_for_prompt(active_booking)
    compact_recent_booking = _compact_booking_for_prompt(recent_completed_booking)
    compact_feedback = _compact_execution_feedback(execution_feedback)
    return (
        "System update: an intake workflow wake fired for one external DM conversation.\n"
        "Operate like a real OpenTulpa background execution turn and use tools when needed.\n\n"
        "Primary goal:\n"
        "- Decide whether this conversation is an active match for the workflow.\n"
        "- Extract reliable booking fields.\n"
        "- If necessary, inspect external state before deciding, especially for availability checks.\n"
        "- Return strict JSON only as the final answer.\n\n"
        "Tool-use guidance:\n"
        "- You may use normal tools, especially uploaded_file_get, uploaded_file_analyze, uploaded_file_search, "
        "composio_tool_search, composio_tool_schema, composio_tool_execute, and "
        "composio_instagram_reply_precheck when they materially help.\n"
        "- If the workflow uses a Google Sheets or generic Composio sink and availability matters, inspect the "
        "relevant external state before setting ready_to_save=true.\n"
        "- If the sink write needs concrete target metadata such as a Google Sheets tab name, inspect the sink "
        "with Composio tools and return those values in sink_arguments.\n"
        "- If the workflow has bound knowledge files, use them before improvising answers.\n"
        "- Prefer minimal read-only tool usage first.\n"
        "- Do not create, update, delete, or run workflows/routines from inside this turn.\n"
        "- Do not call intake_workflow_upsert, intake_workflow_delete, intake_workflow_run, routine_create, or routine_delete.\n"
        "- Do not ask the user for approval. This is background execution.\n"
        "- Do not send the outbound source reply or perform the final booking write yourself in this turn; "
        "the intake workflow service will do the final idempotent reply/save after your decision.\n\n"
        "Final answer contract:\n"
        "- Return strict JSON only with keys:\n"
        "  matches_workflow, confidence, conversation_summary, extracted_fields, missing_fields, "
        "reply_action, reply_text, ready_to_save, booking_action, save_payload, sink_arguments, reason.\n"
        "- booking_action must be one of: ignore, update_active, edit_recent_completed, create_new_booking.\n"
        "- reply_action must be one of: none, send_reply, mark_cancelled.\n"
        "- If availability is blocked or conflicting, do not set ready_to_save=true.\n"
        "- If details are missing, ask one concise follow-up question in reply_text.\n"
        "- If execution_feedback is present, you are replanning after a real tool or application error. "
        "Read it carefully, do not repeat the same failing action unchanged, and adapt your next decision.\n"
        "- sink_arguments is for sink-specific write arguments or overrides discovered during this turn; "
        "leave it empty when not needed.\n"
        "- False positives are worse than ignoring unrelated DMs.\n\n"
        f"customer_id={customer_id}\n"
        f"workflow={json.dumps(compact_workflow, ensure_ascii=False)}\n"
        f"conversation={json.dumps(compact_conversation, ensure_ascii=False)}\n"
        f"active_booking={json.dumps(compact_active_booking, ensure_ascii=False)}\n"
        f"recent_completed_booking={json.dumps(compact_recent_booking, ensure_ascii=False)}\n"
        f"execution_feedback={json.dumps(compact_feedback, ensure_ascii=False)}"
    )


def _build_intake_workflow_human_prompt(
    *,
    customer_id: str,
    workflow: dict[str, Any],
    conversation: dict[str, Any],
    active_booking: dict[str, Any] | None,
    recent_completed_booking: dict[str, Any] | None,
    execution_feedback: list[dict[str, Any]] | None = None,
) -> str:
    compact_workflow = _compact_workflow_for_prompt(workflow)
    compact_conversation = _compact_conversation_for_prompt(conversation)
    compact_active_booking = _compact_booking_for_prompt(active_booking)
    compact_recent_booking = _compact_booking_for_prompt(recent_completed_booking)
    compact_feedback = _compact_execution_feedback(execution_feedback)
    return (
        f"customer_id={customer_id}\n"
        f"workflow={json.dumps(compact_workflow, ensure_ascii=False)}\n"
        f"conversation={json.dumps(compact_conversation, ensure_ascii=False)}\n"
        f"active_booking={json.dumps(compact_active_booking, ensure_ascii=False)}\n"
        f"recent_completed_booking={json.dumps(compact_recent_booking, ensure_ascii=False)}\n"
        f"execution_feedback={json.dumps(compact_feedback, ensure_ascii=False)}"
    )


def _clean_json_text_block(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_schema_from_text(raw: str, schema: type[BaseModel]) -> BaseModel:
    cleaned = _clean_json_text_block(raw)
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return schema.model_validate_json(cleaned[start : end + 1])
        raise


class OpenTulpaLangGraphRuntime:
    def __init__(
        self,
        *,
        app_url: str,
        openrouter_api_key: str,
        model_name: str,
        reasoning_effort: str | None = None,
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        wake_classifier_model_name: str | None = None,
        wake_execution_model_name: str | None = None,
        telegram_media_model_name: str | None = None,
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
        prompt_caching_enabled: bool = True,
        prompt_cache_ttl_1h: bool = False,
        posthog_api_key: str | None = None,
        posthog_host: str | None = None,
    ) -> None:
        self.app_url = app_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = str(openrouter_base_url or "").strip() or "https://openrouter.ai/api/v1"
        self.model_name = _normalize_model_name(model_name)
        self._reasoning_effort = str(reasoning_effort or "").strip() or None
        self._max_completion_tokens = max(128, min(int(max_completion_tokens), 32768))
        self._max_user_reply_chars = max(500, min(int(max_user_reply_chars), 20000))
        self._wake_classifier_model_name = (
            _normalize_model_name(wake_classifier_model_name)
            if str(wake_classifier_model_name or "").strip()
            else self.model_name
        )
        self._wake_execution_model_name = (
            _normalize_model_name(wake_execution_model_name)
            if str(wake_execution_model_name or "").strip()
            else self.model_name
        )
        self._telegram_media_model_name = (
            _normalize_model_name(telegram_media_model_name)
            if str(telegram_media_model_name or "").strip()
            else "google/gemini-3-flash-preview"
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
        # Compatibility aliases consumed by helper modules and persisted state.
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
        self._llm_call_trace_path = self._behavior_log_path.parent / "llm_call_traces.jsonl"
        self._llm_call_trace_lock = threading.Lock()
        self._llm_call_trace_limit = _LLM_CALL_TRACE_LIMIT
        if self._behavior_log_enabled:
            self._behavior_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._browser_use_headless = bool(browser_use_headless)
        self._browser_use_model_override = str(browser_use_model_override or "").strip()
        self._browser_use_max_concurrent_tasks = max(1, int(browser_use_max_concurrent_tasks))
        self._browser_use_task_retention_seconds = max(60, int(browser_use_task_retention_seconds))
        self._prompt_caching_enabled = bool(prompt_caching_enabled)
        self._prompt_cache_ttl_1h = bool(prompt_cache_ttl_1h)
        self._posthog_logger = create_posthog_logger(
            api_key=posthog_api_key,
            host=posthog_host,
        )
        self._context_engineer = ContextEngineer()
        self._browser_use_local_manager: Any | None = None
        self._headroom_service: Any | None = None
        self._interactive_sessions_lock = asyncio.Lock()
        self._interactive_sessions: dict[str, Any] = {}
        self._active_customer_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
            "opentulpa_active_customer_id",
            default="",
        )
        self._active_customer_id = ""
        self._active_thread_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
            "opentulpa_active_thread_id",
            default="",
        )
        self._active_thread_id = ""

        model_init_kwargs: dict[str, Any] = {
            "model_provider": "openai",
            "api_key": openrouter_api_key,
            "base_url": self.openrouter_base_url,
            "temperature": 0,
            "max_completion_tokens": self._max_completion_tokens,
        }
        default_headers = _openrouter_app_headers(base_url=self.openrouter_base_url)
        if default_headers:
            model_init_kwargs["default_headers"] = default_headers
        if self._reasoning_effort:
            model_init_kwargs["reasoning_effort"] = self._reasoning_effort

        self._model = init_chat_model(
            self.model_name,
            **model_init_kwargs,
        )
        if self._wake_classifier_model_name == self.model_name:
            self._wake_classifier_model = self._model
        else:
            try:
                self._wake_classifier_model = init_chat_model(
                    self._wake_classifier_model_name,
                    **model_init_kwargs,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize wake classifier model '%s'; falling back to main model '%s'.",
                    self._wake_classifier_model_name,
                    self.model_name,
                )
                self._wake_classifier_model = self._model
        if self._wake_execution_model_name == self.model_name:
            self._wake_execution_model = self._model
        elif self._wake_execution_model_name == self._wake_classifier_model_name:
            self._wake_execution_model = self._wake_classifier_model
        else:
            try:
                self._wake_execution_model = init_chat_model(
                    self._wake_execution_model_name,
                    **model_init_kwargs,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize wake execution model '%s'; falling back to main model '%s'.",
                    self._wake_execution_model_name,
                    self.model_name,
                )
                self._wake_execution_model = self._model
        if self._telegram_media_model_name == self.model_name:
            self._telegram_media_model = self._model
        elif self._telegram_media_model_name == self._wake_classifier_model_name:
            self._telegram_media_model = self._wake_classifier_model
        elif self._telegram_media_model_name == self._wake_execution_model_name:
            self._telegram_media_model = self._wake_execution_model
        else:
            try:
                self._telegram_media_model = init_chat_model(
                    self._telegram_media_model_name,
                    **model_init_kwargs,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize Telegram media model '%s'; falling back to main model '%s'.",
                    self._telegram_media_model_name,
                    self.model_name,
                )
                self._telegram_media_model = self._model
        if self._guardrail_classifier_model_name == self.model_name:
            self._guardrail_classifier_model = self._model
        elif self._guardrail_classifier_model_name == self._telegram_media_model_name:
            self._guardrail_classifier_model = self._telegram_media_model
        elif self._guardrail_classifier_model_name == self._wake_classifier_model_name:
            self._guardrail_classifier_model = self._wake_classifier_model
        elif self._guardrail_classifier_model_name == self._wake_execution_model_name:
            self._guardrail_classifier_model = self._wake_execution_model
        else:
            try:
                self._guardrail_classifier_model = init_chat_model(
                    self._guardrail_classifier_model_name,
                    **model_init_kwargs,
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
        self._wake_execution_model_with_tools = None
        self._thread_inputs = ThreadInputCoordinator(debounce_seconds=self._input_debounce_seconds)
        self._internal_api = InternalApiClient(base_url=self.app_url)

    def prompt_cache_profile(self, *, model_name: str | None = None) -> dict[str, Any]:
        target_model_name = str(model_name or getattr(self, "model_name", "") or "").strip()
        return dict(
            _provider_prompt_cache_profile(
                enabled=bool(getattr(self, "_prompt_caching_enabled", False)),
                model_name=target_model_name,
                ttl_1h=bool(getattr(self, "_prompt_cache_ttl_1h", False)),
            )
        )

    def model_invoke_extras(self, *, model_name: str | None = None) -> dict[str, Any]:
        """Extra kwargs for main agent model.ainvoke (e.g. OpenRouter prompt cache_control)."""
        target_model_name = str(model_name or getattr(self, "model_name", "") or "").strip()
        return dict(
            _provider_prompt_cache_invoke_extras(
                enabled=bool(getattr(self, "_prompt_caching_enabled", False)),
                model_name=target_model_name,
                ttl_1h=bool(getattr(self, "_prompt_cache_ttl_1h", False)),
            )
        )

    def _model_request_attempts(self, *, model_name: str | None = None) -> list[dict[str, Any]]:
        target_model_name = str(model_name or getattr(self, "model_name", "") or "").strip()
        if not _looks_like_openrouter_base_url(getattr(self, "openrouter_base_url", "")):
            return [{"name": "default", "invoke_extras": {}, "call_context": {}}]
        if not _is_glm_51_model_name(target_model_name):
            return [{"name": "default", "invoke_extras": {}, "call_context": {}}]
        return [
            {
                "name": "glm51_ordered_providers",
                "invoke_extras": {
                    "extra_body": {
                        "provider": {
                            "order": list(_GLM_51_OPENROUTER_PROVIDER_ORDER),
                            "allow_fallbacks": False,
                        }
                    }
                },
                "call_context": {
                    "provider_route": "glm51_ordered_providers",
                    "provider_order": list(_GLM_51_OPENROUTER_PROVIDER_ORDER),
                    "provider_allow_fallbacks": False,
                },
            },
        ]

    def _resolve_model_name_for_runtime_call(self, model: Any, explicit_name: str | None = None) -> str:
        if explicit_name:
            return str(explicit_name).strip()
        if model is getattr(self, "_wake_classifier_model", None):
            return str(getattr(self, "_wake_classifier_model_name", "") or "").strip()
        if model is getattr(self, "_wake_execution_model", None):
            return str(getattr(self, "_wake_execution_model_name", "") or "").strip()
        if model is getattr(self, "_guardrail_classifier_model", None):
            return str(getattr(self, "_guardrail_classifier_model_name", "") or "").strip()
        if model is getattr(self, "_wake_execution_model_with_tools", None):
            return str(getattr(self, "_wake_execution_model_name", "") or "").strip()
        if model is getattr(self, "_model", None) or model is getattr(self, "_model_with_tools", None):
            return str(getattr(self, "model_name", "") or "").strip()
        model_name = getattr(model, "model_name", None)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
        return str(getattr(self, "model_name", "") or "").strip()

    def model_with_tools_for_turn_mode(self, turn_mode: str) -> Any:
        normalized_turn_mode = str(turn_mode or "").strip().lower()
        if normalized_turn_mode == "routine_wake" and self._wake_execution_model_with_tools is not None:
            return self._wake_execution_model_with_tools
        return self._model_with_tools

    def prepare_messages_for_prompt_cache(
        self,
        messages: list[Any],
        *,
        model_name: str | None = None,
        stable_prefix_count: int = 0,
    ) -> list[Any]:
        profile = self.prompt_cache_profile(model_name=model_name)
        if profile.get("strategy") != "breakpoint":
            return messages
        cache_control = dict(profile.get("cache_control") or {})
        if not cache_control:
            return messages
        effective_stable_prefix_count = (
            int(stable_prefix_count)
            if int(stable_prefix_count) > 0
            else _infer_stable_system_prefix_count(messages)
        )
        if effective_stable_prefix_count <= 0:
            return messages
        patched: list[Any] = list(messages)
        target_index: int | None = None
        for idx in range(min(effective_stable_prefix_count, len(patched)) - 1, -1, -1):
            if getattr(patched[idx], "content", None):
                target_index = idx
                break
        if target_index is None:
            return messages
        patched[target_index] = _message_with_cache_breakpoint(
            patched[target_index],
            cache_control=cache_control,
        )
        return patched

    async def ainvoke_model(
        self,
        model: Any,
        messages: list[Any],
        *,
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        call_context: dict[str, Any] | None = None,
    ) -> Any:
        resolved_model_name = self._resolve_model_name_for_runtime_call(model, explicit_name=model_name)
        prepared_messages = self.prepare_messages_for_prompt_cache(
            list(messages),
            model_name=resolved_model_name,
            stable_prefix_count=stable_prefix_count,
        )
        base_invoke_extras = self.model_invoke_extras(model_name=resolved_model_name)
        attempts = self._model_request_attempts(model_name=resolved_model_name)
        last_exc: Exception | None = None
        for attempt_index, attempt in enumerate(attempts):
            invoke_extras = _deep_merge_dicts(
                dict(base_invoke_extras),
                dict(attempt.get("invoke_extras") or {}),
            )
            attempt_context = dict(call_context or {})
            attempt_context.update(dict(attempt.get("call_context") or {}))
            attempt_context["provider_attempt_name"] = str(attempt.get("name") or "").strip() or "default"
            attempt_context["provider_attempt_index"] = attempt_index + 1
            attempt_context["provider_attempt_count"] = len(attempts)
            callback_target = self._model_with_callbacks(model, call_context=attempt_context)
            response: Any | None = None
            error_text: str | None = None
            try:
                if _supports_ainvoke_kwargs(callback_target, invoke_extras):
                    response = await callback_target.ainvoke(prepared_messages, **invoke_extras)
                else:
                    response = await callback_target.ainvoke(prepared_messages)
                return response
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                last_exc = exc
                if attempt_index + 1 >= len(attempts):
                    raise
                logger.warning(
                    "Model invocation via %s failed for %s; retrying with next provider route: %s",
                    attempt_context["provider_attempt_name"],
                    resolved_model_name,
                    error_text,
                )
                self.log_behavior_event(
                    event="llm.provider_fallback",
                    model_name=resolved_model_name,
                    failed_provider_attempt=attempt_context["provider_attempt_name"],
                    next_provider_attempt=str(attempts[attempt_index + 1].get("name") or "").strip() or "default",
                    error=error_text,
                )
            finally:
                self._record_llm_call_trace(
                    model_name=resolved_model_name,
                    prepared_messages=prepared_messages,
                    stable_prefix_count=stable_prefix_count,
                    response=response,
                    error=error_text,
                    call_context=attempt_context,
                )
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Model invocation failed without attempts.")

    @staticmethod
    def _looks_like_provisional_reply(text: str) -> bool:
        candidate = " ".join(str(text or "").split()).strip()
        if not candidate:
            return False
        return any(pattern.search(candidate) for pattern in _PROVISIONAL_REPLY_PATTERNS)

    @staticmethod
    def _stream_chunk_is_tool_phase(node_name: str, message_chunk: Any) -> bool:
        normalized = str(node_name or "").strip().lower()
        if normalized != "tools":
            return False
        if isinstance(message_chunk, ToolMessage):
            return True
        tool_calls = getattr(message_chunk, "tool_calls", None)
        if isinstance(tool_calls, list) and tool_calls:
            return True
        return True

    @staticmethod
    def _build_progress_signal(text: str) -> str:
        cleaned = " ".join(str(text or "").split()).strip() or "Working on it…"
        return f"{STREAM_PROGRESS_PREFIX}{cleaned}"

    @staticmethod
    def _describe_tool_calls_for_progress(tool_calls: list[Any]) -> str:
        names: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "")).strip()
            if name:
                names.append(name)
        if not names:
            return "Working on it…"
        labels: list[str] = []
        for name in names[:2]:
            label = _PROGRESS_TOOL_NAME_ALIASES.get(name)
            if label is None:
                label = name.replace("tulpa_", "").replace("browser_use_", "").replace("_", " ").strip().capitalize()
            labels.append(label)
        if len(names) == 1:
            return f"{labels[0]}…"
        return f"{labels[0]}, then {labels[1].lower()}…"

    def get_browser_use_local_manager(self) -> Any:
        if self._browser_use_local_manager is None:
            from opentulpa.integrations.browser_use_local import BrowserUseLocalManager

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

    def get_headroom_service(self) -> Any:
        if self._headroom_service is None:
            from opentulpa.integrations import HeadroomService

            self._headroom_service = HeadroomService(model_name=self.model_name)
        return self._headroom_service

    def compress_tool_result_for_model(
        self,
        *,
        tool_name: str,
        args: Any,
        result: Any,
        user_text: str = "",
        model_name: str | None = None,
    ) -> str:
        raw_result_text = _safe_json(result).strip()
        if not raw_result_text:
            return ""
        service = self.get_headroom_service()
        compress = getattr(service, "compress_tool_result", None)
        if not callable(compress):
            return raw_result_text
        try:
            compressed = compress(
                tool_name=tool_name,
                args=args,
                result=result,
                user_text=user_text,
                model_name=model_name or self.model_name,
            )
        except Exception:
            logger.exception("tool result compression failed for %s", str(tool_name or "").strip() or "tool")
            return raw_result_text
        normalized = str(compressed or "").strip()
        return normalized or raw_result_text

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

    def capture_posthog_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        posthog_logger = getattr(self, "_posthog_logger", None)
        capture = getattr(posthog_logger, "capture_event", None)
        if not callable(capture):
            return
        capture(
            distinct_id=str(customer_id or "").strip() or None,
            event=str(event or "").strip(),
            properties=_json_safe(properties or {}),
        )

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        posthog_event: str | None = None,
        **fields: Any,
    ) -> None:
        self.log_behavior_event(event=event, **fields)
        self.capture_posthog_event(
            event=str(posthog_event or event or "").strip(),
            customer_id=customer_id,
            properties={"behavior_event": str(event or "").strip(), **fields},
        )

    @staticmethod
    def _normalize_llm_call_context(call_context: dict[str, Any] | None) -> dict[str, Any]:
        normalized = dict(call_context) if isinstance(call_context, dict) else {}
        prompt_sections = normalized.get("prompt_sections")
        if isinstance(prompt_sections, str):
            normalized["prompt_sections"] = [
                part.strip() for part in prompt_sections.split(",") if part.strip()
            ]
        elif isinstance(prompt_sections, list):
            normalized["prompt_sections"] = [
                str(part).strip() for part in prompt_sections if str(part).strip()
            ]
        normalized["call_site"] = str(
            normalized.get("call_site") or "runtime_model_invoke"
        ).strip()
        return normalized

    def _write_llm_call_trace(self, payload: dict[str, Any]) -> None:
        path = getattr(self, "_llm_call_trace_path", None)
        lock = getattr(self, "_llm_call_trace_lock", None)
        limit = max(1, int(getattr(self, "_llm_call_trace_limit", _LLM_CALL_TRACE_LIMIT)))
        if not isinstance(path, Path):
            return
        serialized = json.dumps(payload, ensure_ascii=False, default=str)

        def _commit() -> None:
            existing: list[str] = []
            with suppress(Exception):
                existing = [
                    line.rstrip("\n")
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            kept = existing[-max(0, limit - 1) :]
            kept.append(serialized)
            with path.open("w", encoding="utf-8") as f:
                if kept:
                    f.write("\n".join(kept) + "\n")

        with suppress(Exception):
            path.parent.mkdir(parents=True, exist_ok=True)
        if lock is None:
            with suppress(Exception):
                _commit()
            return
        with suppress(Exception), lock:
            _commit()

    def _record_llm_call_trace(
        self,
        *,
        model_name: str,
        prepared_messages: list[Any],
        stable_prefix_count: int,
        response: Any | None,
        error: str | None,
        call_context: dict[str, Any] | None = None,
    ) -> None:
        if not bool(getattr(self, "_behavior_log_enabled", True)):
            return
        normalized_context = self._normalize_llm_call_context(call_context)
        usage_fields = (
            self.extract_response_usage_fields(response)
            if response is not None
            else {}
        )
        response_content = getattr(response, "content", response) if response is not None else ""
        safe_response_content = _json_safe(response_content)
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "model_name": str(model_name or "").strip(),
            "stable_prefix_count": int(stable_prefix_count),
            "prompt_messages": [_serialize_message(message) for message in prepared_messages],
            "prompt_message_count": len(prepared_messages),
            "response_type": type(response).__name__ if response is not None else "",
            "response_message": _serialize_message(response) if response is not None else None,
            "response_text": _content_to_text(safe_response_content).strip() if response is not None else "",
            "response_content": safe_response_content,
            "response_tool_calls": _json_safe(getattr(response, "tool_calls", None)),
            "error": str(error or "").strip() or None,
            **usage_fields,
        }
        for key, value in normalized_context.items():
            record[str(key)] = _json_safe(value)
        self._write_llm_call_trace(record)

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
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        call_context: dict[str, Any] | None = None,
    ) -> tuple[BaseModel | None, str | None]:
        last_error: str | None = None
        resolved_model_name = self._resolve_model_name_for_runtime_call(model, explicit_name=model_name)
        prepared_messages = self.prepare_messages_for_prompt_cache(
            list(messages),
            model_name=resolved_model_name,
            stable_prefix_count=stable_prefix_count,
        )
        base_invoke_extras = self.model_invoke_extras(model_name=resolved_model_name)
        attempts = self._model_request_attempts(model_name=resolved_model_name)
        for attempt_index, attempt in enumerate(attempts):
            invoke_extras = _deep_merge_dicts(
                dict(base_invoke_extras),
                dict(attempt.get("invoke_extras") or {}),
            )
            attempt_context = dict(call_context or {})
            attempt_context.update(dict(attempt.get("call_context") or {}))
            attempt_context["provider_attempt_name"] = str(attempt.get("name") or "").strip() or "default"
            attempt_context["provider_attempt_index"] = attempt_index + 1
            attempt_context["provider_attempt_count"] = len(attempts)
            callback_target = self._model_with_callbacks(model, call_context=attempt_context)
            structured = getattr(callback_target, "with_structured_output", None)
            payload: Any | None = None
            error_text: str | None = None
            trace_recorded = False
            if callable(structured):
                try:
                    runner = structured(schema)
                    if _supports_ainvoke_kwargs(runner, invoke_extras):
                        payload = await runner.ainvoke(prepared_messages, **invoke_extras)
                    else:
                        payload = await runner.ainvoke(prepared_messages)
                    if isinstance(payload, schema):
                        self._record_llm_call_trace(
                            model_name=resolved_model_name,
                            prepared_messages=prepared_messages,
                            stable_prefix_count=stable_prefix_count,
                            response=payload,
                            error=None,
                            call_context=attempt_context,
                        )
                        trace_recorded = True
                        return payload, None
                    if isinstance(payload, dict):
                        parsed = schema.model_validate(payload)
                        self._record_llm_call_trace(
                            model_name=resolved_model_name,
                            prepared_messages=prepared_messages,
                            stable_prefix_count=stable_prefix_count,
                            response=parsed,
                            error=None,
                            call_context=attempt_context,
                        )
                        trace_recorded = True
                        return parsed, None
                    error_text = (
                        f"TypeError: structured output returned unsupported type "
                        f"{type(payload).__name__}"
                    )
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"
                finally:
                    if not trace_recorded and (payload is not None or error_text):
                        self._record_llm_call_trace(
                            model_name=resolved_model_name,
                            prepared_messages=prepared_messages,
                            stable_prefix_count=stable_prefix_count,
                            response=payload,
                            error=error_text,
                            call_context=attempt_context,
                        )
            if error_text:
                last_error = error_text
                if attempt_index + 1 >= len(attempts):
                    break
                logger.warning(
                    "Structured model invocation via %s failed for %s; retrying with next provider route: %s",
                    attempt_context["provider_attempt_name"],
                    resolved_model_name,
                    error_text,
                )
                self.log_behavior_event(
                    event="llm.provider_fallback",
                    model_name=resolved_model_name,
                    failed_provider_attempt=attempt_context["provider_attempt_name"],
                    next_provider_attempt=str(attempts[attempt_index + 1].get("name") or "").strip() or "default",
                    error=error_text,
                )
                continue
        try:
            response = await self.ainvoke_model(
                model,
                list(messages),
                model_name=resolved_model_name,
                stable_prefix_count=stable_prefix_count,
                call_context={
                    **dict(call_context or {}),
                    "call_site": str((call_context or {}).get("call_site") or "structured_model_fallback"),
                },
            )
            raw = _content_to_text(getattr(response, "content", response)).strip()
            if raw:
                return schema.model_validate_json(_clean_json_text_block(raw)), None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        return None, last_error

    def extract_response_usage_fields(self, response: Any) -> dict[str, Any]:
        return dict(_extract_response_usage_fields(response))

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
                json_body=CustomerScopedRequest(customer_id=cid).model_dump(mode="json"),
                timeout=5.0,
                retries=1,
            )
            if r.status_code != 200:
                return None
            return DirectiveGetResponse.model_validate(r.json()).directive
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
                json_body=CustomerScopedRequest(customer_id=cid).model_dump(mode="json"),
                timeout=5.0,
                retries=1,
            )
            if r.status_code != 200:
                return None
            return TimeProfileGetResponse.model_validate(r.json()).utc_offset
        except Exception:
            return None

    @staticmethod
    def _has_retrieval_evidence(
        *,
        user_text: str,
        prompt_mode: str,
        skill_candidates: list[dict[str, Any]] | None = None,
        thread_rollup_sections: dict[str, str] | None = None,
    ) -> bool:
        mode = str(prompt_mode or "").strip().lower()
        if mode == "literal_chat":
            return False
        text = str(user_text or "").strip().lower()
        if not text:
            return False
        if mode == "execution":
            return True
        tokens = set(re.findall(r"[a-z0-9][a-z0-9._-]{2,}", text))
        if not tokens:
            return False
        if isinstance(skill_candidates, list):
            for item in skill_candidates:
                if not isinstance(item, dict):
                    continue
                hay = f"{item.get('name', '')} {item.get('description', '')}".lower()
                if any(tok in hay for tok in tokens):
                    return True
        if isinstance(thread_rollup_sections, dict):
            hay = " ".join(str(thread_rollup_sections.get(k) or "").lower() for k in ("conversation_summary", "open_loops", "durable_facts"))
            if any(tok in hay for tok in tokens):
                return True
        return False

    @staticmethod
    def _normalize_memory_search_results(raw: Any) -> list[dict[str, Any]]:
        payload = raw.get("results") if isinstance(raw, dict) and isinstance(raw.get("results"), list) else raw
        if not isinstance(payload, list):
            payload = [payload] if payload not in (None, "") else []
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("memory") or item.get("content") or "").strip()
            if not text:
                continue
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            kind = str(item.get("kind") or metadata.get("kind") or "").strip().lower() or "thread_context_rollup"
            dedupe_key = (kind, text.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(
                {
                    "text": text,
                    "kind": kind,
                    "score": item.get("score"),
                    "metadata": metadata,
                    "thread_id": str(item.get("thread_id") or metadata.get("thread_id") or "").strip(),
                    "skill_name": str(item.get("skill_name") or metadata.get("skill_name") or "").strip(),
                }
            )
        return normalized

    @staticmethod
    def _memory_grounding_sort_key(item: dict[str, Any]) -> tuple[int, float]:
        kind = str(item.get("kind", "") or "").strip().lower()
        priority = int(MEMORY_KIND_PRIORITY.get(kind, 50))
        try:
            score = float(item.get("score"))
        except (TypeError, ValueError):
            score = 0.0
        return priority, -score

    @staticmethod
    def _memory_grounding_section_for_kind(kind: str) -> str:
        normalized = str(kind or "").strip().lower()
        for section_name, kinds in _MEMORY_GROUNDING_KIND_SECTIONS:
            if normalized in kinds:
                return section_name
        return "fallback_thread_context"

    def _build_memory_grounding_block(
        self,
        memories: list[dict[str, Any]],
        *,
        token_budget: int = 380,
    ) -> str:
        if not memories:
            return ""
        budget = max(180, int(token_budget))
        section_labels = {
            "preferences_and_directives": "Preferences and directives",
            "durable_personal_facts": "Durable personal facts",
            "aspirations_and_plans": "Aspirations and plans",
            "active_projects_or_workflows": "Active projects or workflows",
            "technical_or_code_facts": "Technical or code facts",
            "relevant_files_or_media": "Relevant files or media",
            "fallback_thread_context": "Fallback thread context",
        }
        grouped: dict[str, list[str]] = {name: [] for name, _ in _MEMORY_GROUNDING_KIND_SECTIONS}
        used = 0
        max_lines_per_section = 3
        for item in sorted(memories, key=self._memory_grounding_sort_key):
            section_name = self._memory_grounding_section_for_kind(str(item.get("kind", "")))
            line = _trim_text_to_token_budget(str(item.get("text", "")).strip(), token_budget=28)
            if not line:
                continue
            line_tokens = max(1, _approx_tokens(line) + 1)
            if grouped[section_name] and line in grouped[section_name]:
                continue
            if len(grouped[section_name]) >= max_lines_per_section:
                continue
            if used and used + line_tokens > budget:
                continue
            grouped[section_name].append(line)
            used += line_tokens
        parts: list[str] = []
        for section_name, _ in _MEMORY_GROUNDING_KIND_SECTIONS:
            lines = grouped.get(section_name) or []
            if not lines:
                continue
            parts.append(f"{section_labels[section_name]}:\n- " + "\n- ".join(lines))
        block = "\n\n".join(parts).strip()
        return _trim_text_to_token_budget(block, token_budget=budget)

    async def _load_memory_grounding_context(
        self,
        *,
        customer_id: str,
        user_text: str,
        turn_mode: str,
        token_budget: int = 500,
    ) -> str:
        if str(turn_mode or "").strip().lower() != "interactive":
            return ""
        cid = str(customer_id or "").strip()
        if not cid:
            return ""
        primary_query = str(user_text or "").strip()
        queries: list[dict[str, Any]] = []
        if primary_query:
            queries.append({"query": primary_query, "limit": 8, "metadata": None})
        # Favor durable facts first and pull thread rollups only as fallback.
        queries.extend(
            [
                {
                    "query": "important durable preferences, directives, personal facts, projects, workflows, skills, and technical context",
                    "limit": 8,
                    "metadata": {
                        "kind": [
                            "directive_fact",
                            "preference_fact",
                            "style_fact",
                            "user_profile_fact",
                            "life_fact",
                            "relationship_fact",
                            "contact_fact",
                            "project_fact",
                            "aspirations_fact",
                            "workflow_fact",
                            "skill_fact",
                            "code_fact",
                            "credential_fact",
                            "file_fact",
                            "media_fact",
                        ]
                    },
                },
                {
                    "query": "compressed older thread context and unresolved notes",
                    "limit": 4,
                    "metadata": {"kind": "thread_context_rollup"},
                },
            ]
        )
        collected: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for spec in queries:
            query = str(spec.get("query", "") or "").strip()
            if not query:
                continue
            try:
                response = await self._request_with_backoff(
                    "POST",
                    "/internal/memory/search",
                    json_body={
                        "query": query,
                        "user_id": cid,
                        "limit": int(spec.get("limit", 8)),
                        "metadata": spec.get("metadata"),
                    },
                    timeout=8.0,
                    retries=1,
                )
            except Exception:
                continue
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
            except Exception:
                continue
            for item in self._normalize_memory_search_results(payload.get("results", payload)):
                dedupe_key = (
                    str(item.get("kind", "")).strip().lower(),
                    str(item.get("text", "")).strip().lower(),
                )
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                collected.append(item)
            if len(collected) >= 10:
                break
        return self._build_memory_grounding_block(collected, token_budget=token_budget)

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
        prompt_mode: str = "task_chat",
        max_skills: int = 2,
    ) -> list[dict[str, Any]]:
        prompt_query = str(query or "").strip()
        if not prompt_query or not candidates:
            return []
        if str(prompt_mode or "").strip().lower() == "literal_chat":
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
                        "Never select persona, tone, or style-only skills for literal definitions, acronym expansions, translations, or short factual clarifications.\n"
                        "If the request is about reminders, schedules, recurring jobs, or cron, "
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

    async def _resolve_skill_context(
        self,
        customer_id: str,
        user_text: str,
        *,
        prompt_mode: str = "task_chat",
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cid = str(customer_id or "").strip()
        query = str(user_text or "").strip()
        if not cid or not query:
            return {"skill_names": [], "context": ""}
        available = candidates if isinstance(candidates, list) else await self._list_available_skills(cid)
        if not available:
            return {"skill_names": [], "context": ""}
        selected = await self._select_relevant_skills(
            customer_id=cid,
            query=query,
            candidates=available,
            prompt_mode=prompt_mode,
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

    async def _load_skill_context_by_names(
        self,
        customer_id: str,
        skill_names: list[str] | None,
    ) -> dict[str, Any]:
        cid = str(customer_id or "").strip()
        normalized_names: list[str] = []
        for item in skill_names or []:
            name = str(item or "").strip()
            if not name or name in normalized_names:
                continue
            normalized_names.append(name)
        if not cid or not normalized_names:
            return {"skill_names": [], "context": ""}

        sections: list[str] = []
        resolved_names: list[str] = []
        total_chars = 0
        max_total_chars = 12000
        for name in normalized_names[:3]:
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
                if total_chars + len(skill_md) > max_total_chars:
                    break
                sections.append(skill_md)
                resolved_name = str(skill.get("name", "")).strip() or name
                if resolved_name not in resolved_names:
                    resolved_names.append(resolved_name)
                total_chars += len(skill_md)
            except Exception:
                continue
        return {
            "skill_names": resolved_names,
            "context": "\n\n---\n\n".join(sections).strip(),
        }

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

    def _load_thread_rollup_sections(self, thread_id: str) -> dict[str, str]:
        tid = str(thread_id or "").strip()
        empty = {
            "conversation_summary": "",
            "open_loops": "",
            "durable_facts": "",
            "sensitive_refs": "",
            "style_notes": "",
        }
        if not tid or self._thread_rollup_service is None:
            return empty
        try:
            getter = getattr(self._thread_rollup_service, "get_rollup_payload", None)
            payload = getter(tid) if callable(getter) else None
            if isinstance(payload, dict):
                return {key: self._cap_rollup_text(str(payload.get(key) or "")) for key in empty}
            legacy = self._thread_rollup_service.get_rollup(tid)
            return {
                key: self._cap_rollup_text(value)
                for key, value in _split_rollup_sections(legacy or "").items()
            }
        except Exception:
            return empty

    def _save_thread_rollup(self, thread_id: str, rollup: str) -> None:
        tid = str(thread_id or "").strip()
        text = self._cap_rollup_text(rollup)
        if not tid or not text or self._thread_rollup_service is None:
            return
        with suppress(Exception):
            setter = getattr(self._thread_rollup_service, "set_rollup_payload", None)
            if callable(setter):
                setter(tid, _split_rollup_sections(text))
            else:
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
        prompt_mode: str,
        forced_skill_names: list[str] | None = None,
    ) -> dict[str, Any]:
        query = str(user_text or "").strip()
        available_skills = await self._list_available_skills(customer_id)
        forced_names = [
            str(item or "").strip()
            for item in (forced_skill_names or [])
            if str(item or "").strip()
        ]
        forced_skill_context = (
            await self._load_skill_context_by_names(
                customer_id=customer_id,
                skill_names=forced_names,
            )
            if forced_names
            else {"skill_names": [], "context": ""}
        )
        if not query:
            return {
                "prompt_mode": prompt_mode,
                "active_skill_query": "",
                "active_skill_names": forced_names,
                "active_available_skills": available_skills,
                "active_skill_discovery_context": "",
                "active_invoked_skill_context": str(forced_skill_context.get("context", "")).strip(),
                "active_invoked_skill_names": list(forced_skill_context.get("skill_names", []) or []),
                "active_skill_context": str(forced_skill_context.get("context", "")).strip(),
            }
        if forced_names:
            return {
                "prompt_mode": prompt_mode,
                "active_skill_query": query,
                "active_skill_names": forced_names,
                "active_available_skills": available_skills,
                "active_skill_discovery_context": "",
                "active_invoked_skill_context": str(forced_skill_context.get("context", "")).strip(),
                "active_invoked_skill_names": list(forced_skill_context.get("skill_names", []) or []),
                "active_skill_context": str(forced_skill_context.get("context", "")).strip(),
            }
        if prompt_mode == "literal_chat":
            return {
                "prompt_mode": prompt_mode,
                "active_skill_query": query,
                "active_skill_names": [],
                "active_available_skills": available_skills,
                "active_skill_discovery_context": "",
                "active_invoked_skill_context": "",
                "active_invoked_skill_names": [],
                "active_skill_context": "",
            }
        selected = await self._select_relevant_skills(
            customer_id=customer_id,
            query=query,
            candidates=available_skills,
            prompt_mode=prompt_mode,
            max_skills=3,
        )
        names = [str(item.get("name", "")).strip() for item in selected if str(item.get("name", "")).strip()]
        return {
            "prompt_mode": prompt_mode,
            "active_skill_query": query,
            "active_skill_names": names,
            "active_available_skills": available_skills,
            "active_skill_discovery_context": "",
            "active_invoked_skill_context": "",
            "active_invoked_skill_names": [],
            "active_skill_context": "",
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
        if self._wake_execution_model is self._model:
            self._wake_execution_model_with_tools = self._model_with_tools
        else:
            self._wake_execution_model_with_tools = self._wake_execution_model.bind_tools(list(self._tools.values()))
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
        posthog_logger = getattr(self, "_posthog_logger", None)
        if posthog_logger is not None and hasattr(posthog_logger, "shutdown"):
            with suppress(Exception):
                posthog_logger.shutdown()
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None
        self._checkpointer = None
        self._graph = None
        self._model_with_tools = None
        self._wake_execution_model_with_tools = None

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
        prompt_mode: str,
        pending_context_summary: str,
        trace_id: str,
        skill_state: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=user_text)],
            "customer_id": customer_id,
            "thread_id": thread_id,
            "turn_mode": _normalize_turn_mode(turn_mode),
            "prompt_mode": prompt_mode,
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": pending_context_summary,
            "agent_trace_id": trace_id,
            "tool_error_count": 0,
            "approval_handoff": False,
            "claim_check_retry_count": 0,
            "claim_check_needs_retry": False,
            "frozen_prompt_context": None,
            "frozen_history_projection": None,
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
        forced_skill_names: list[str] | None = None,
        prompt_mode_override: str | None = None,
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
        prompt_mode = (
            str(prompt_mode_override or "").strip().lower()
            or _classify_prompt_mode(user_text, turn_mode=turn_mode)
        )
        try:
            skill_state = await self._pre_resolve_skill_state(
                customer_id=customer_id,
                user_text=user_text,
                prompt_mode=prompt_mode,
                forced_skill_names=forced_skill_names,
            )
        except TypeError:
            try:
                skill_state = await self._pre_resolve_skill_state(
                    customer_id=customer_id,
                    user_text=user_text,
                    prompt_mode=prompt_mode,
                )
            except TypeError:
                skill_state = await self._pre_resolve_skill_state(
                    customer_id=customer_id,
                    user_text=user_text,
                )
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._effective_recursion_limit(recursion_limit_override),
        }
        callbacks = self._build_posthog_callbacks(
            customer_id=customer_id,
            trace_id=trace_id,
            thread_id=thread_id,
            turn_mode=turn_mode,
            prompt_mode=prompt_mode,
        )
        if callbacks:
            config["callbacks"] = callbacks
        graph_input = self._build_graph_input(
            user_text=user_text,
            customer_id=customer_id,
            thread_id=thread_id,
            turn_mode=turn_mode,
            prompt_mode=prompt_mode,
            pending_context_summary=pending_context_summary,
            trace_id=trace_id,
            skill_state=skill_state,
        )
        return _PreparedTurnContext(
            through_id=through_id,
            config=config,
            graph_input=graph_input,
        )

    def _build_posthog_callbacks(
        self,
        *,
        customer_id: str | None,
        trace_id: str | None,
        thread_id: str | None,
        turn_mode: str | None,
        prompt_mode: str | None,
        call_site: str | None = None,
        model_name: str | None = None,
    ) -> list[Any]:
        posthog_logger = getattr(self, "_posthog_logger", None)
        if posthog_logger is None:
            return []
        properties: dict[str, Any] = {
            "thread_id": str(thread_id or "").strip(),
            "turn_mode": str(turn_mode or "").strip(),
            "prompt_mode": str(prompt_mode or "").strip(),
            "call_site": str(call_site or "").strip(),
            "model_name": str(model_name or "").strip(),
        }
        return posthog_logger.build_callbacks(
            distinct_id=str(customer_id or "").strip() or None,
            trace_id=str(trace_id or "").strip() or None,
            properties=properties,
        )

    def _model_with_callbacks(self, model: Any, *, call_context: dict[str, Any] | None = None) -> Any:
        if model is None:
            return model
        context = dict(call_context or {})
        callbacks = self._build_posthog_callbacks(
            customer_id=str(
                context.get("customer_id")
                or self.get_active_customer_id()
                or ""
            ).strip()
            or None,
            trace_id=str(context.get("trace_id") or "").strip() or None,
            thread_id=str(context.get("thread_id") or "").strip() or None,
            turn_mode=str(context.get("turn_mode") or "").strip() or None,
            prompt_mode=str(context.get("prompt_mode") or "").strip() or None,
            call_site=str(context.get("call_site") or "").strip() or "runtime_model_invoke",
            model_name=self._resolve_model_name_for_runtime_call(model, explicit_name=context.get("model_name")),
        )
        if not callbacks:
            return model
        with_config = getattr(model, "with_config", None)
        if not callable(with_config):
            return model
        try:
            return with_config({"callbacks": callbacks})
        except Exception:
            logger.exception("Failed to attach PostHog callbacks to model invocation.")
            return model

    async def ainvoke_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
        recursion_limit_override: int | None = None,
        forced_skill_names: list[str] | None = None,
        prompt_mode_override: str | None = None,
    ) -> str:
        await self.start()
        assert self._graph is not None
        normalized_turn_mode = _normalize_turn_mode(turn_mode)
        turn_trace_id = new_short_id("turn")
        interactive_session = await self._get_registered_interactive_session(thread_id=thread_id)
        if normalized_turn_mode == "interactive" and interactive_session is not None:
            turn_state = None
            effective_text = str(text or "")
        else:
            turn_state, effective_text = await self._thread_inputs.begin_turn(
                thread_id=thread_id, text=text
            )
        if turn_state is None and not (
            normalized_turn_mode == "interactive" and interactive_session is not None
        ):
            self.log_behavior_event(
                event="turn_merged",
                trace_id=turn_trace_id,
                mode="ainvoke",
                thread_id=thread_id,
                customer_id=customer_id,
            )
            return ""
        customer_scope_token = self.set_active_customer_id(customer_id)
        thread_scope_token = self.set_active_thread_id(thread_id)
        try:
            if turn_state is not None or not (normalized_turn_mode == "interactive" and interactive_session is not None):
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
                forced_skill_names=forced_skill_names,
                prompt_mode_override=prompt_mode_override,
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
            self.reset_active_thread_id(thread_scope_token)
            self.reset_active_customer_id(customer_scope_token)
            self._thread_inputs.end_turn(turn_state)

    async def astream_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
        forced_skill_names: list[str] | None = None,
        prompt_mode_override: str | None = None,
    ) -> AsyncIterator[str]:
        await self.start()
        assert self._graph is not None
        normalized_turn_mode = _normalize_turn_mode(turn_mode)
        turn_trace_id = new_short_id("turn")
        interactive_session = await self._get_registered_interactive_session(thread_id=thread_id)
        if normalized_turn_mode == "interactive" and interactive_session is not None:
            turn_state = None
            effective_text = str(text or "")
        else:
            turn_state, effective_text = await self._thread_inputs.begin_turn(
                thread_id=thread_id, text=text
            )
        if turn_state is None and normalized_turn_mode == "interactive" and interactive_session is not None:
            logger.info(
                "runtime.astream_text interactive_session_bypass thread_id=%s customer_id=%s",
                thread_id,
                customer_id,
            )
        elif turn_state is None:
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
        customer_scope_token = self.set_active_customer_id(customer_id)
        thread_scope_token = self.set_active_thread_id(thread_id)
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
                forced_skill_names=forced_skill_names,
                prompt_mode_override=prompt_mode_override,
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
            in_tool_phase = False
            suppress_live_text_until_completion = False
            approval_handoff_detected = False
            stream_started_at = time.monotonic()
            stream_no_visible_timeout_s = float(
                str(os.environ.get("AGENT_STREAM_NO_VISIBLE_PROGRESS_SECONDS", "210")).strip()
                or "210"
            )
            stream_precommit_seconds = STREAM_PRECOMMIT_SECONDS
            stream_total_chunks = 0
            stream_agent_chunks = 0
            stream_tool_chunks = 0
            stream_wait_signals = 0
            stream_visible_yields = 0
            stream_filtered_empty = 0
            stream_filtered_blank_expanded = 0
            first_visible_yield_ms: int | None = None
            buffered_visible = ""
            buffered_visible_truncated = False
            buffered_visible_source_chars = 0
            pending_progress_text = "Working on it…"
            self.log_behavior_event(
                event="turn_stream_loop_start",
                trace_id=turn_trace_id,
                thread_id=thread_id,
                customer_id=customer_id,
                stream_no_visible_timeout_s=stream_no_visible_timeout_s,
                stream_precommit_seconds=stream_precommit_seconds,
                turn_mode=normalized_turn_mode,
            )

            def _precommit_active() -> bool:
                if stream_precommit_seconds <= 0 or yielded_any:
                    return False
                return (time.monotonic() - stream_started_at) < stream_precommit_seconds

            def _finalize_segment(*, register_links: bool = True) -> None:
                nonlocal segment_accumulated
                if not segment_accumulated:
                    return
                cleaned_segment = segment_accumulated
                if register_links and cleaned_segment.strip():
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
                        if buffered_visible and not yielded_any:
                            self.log_behavior_event(
                                event="turn_stream_precommit_discarded",
                                trace_id=turn_trace_id,
                                thread_id=thread_id,
                                customer_id=customer_id,
                                output_chars=len(buffered_visible.strip()),
                                reason="approval_handoff",
                                turn_mode=normalized_turn_mode,
                            )
                            buffered_visible = ""
                            buffered_visible_truncated = False
                            buffered_visible_source_chars = 0
                            _finalize_segment(register_links=False)
                        self.log_behavior_event(
                            event="turn_approval_handoff",
                            trace_id=turn_trace_id,
                            mode="astream",
                            thread_id=thread_id,
                            customer_id=customer_id,
                            turn_mode=normalized_turn_mode,
                        )
                        yielded_any = True
                        yield STREAM_APPROVAL_HANDOFF_SIGNAL
                        break
                    if self._stream_chunk_is_tool_phase(node_name, message_chunk) and not in_tool_phase:
                        in_tool_phase = True
                        suppress_live_text_until_completion = True
                        if buffered_visible and not yielded_any:
                            self.log_behavior_event(
                                event="turn_stream_precommit_discarded",
                                trace_id=turn_trace_id,
                                thread_id=thread_id,
                                customer_id=customer_id,
                                output_chars=len(buffered_visible.strip()),
                                reason="tool_phase",
                                turn_mode=normalized_turn_mode,
                            )
                            buffered_visible = ""
                            buffered_visible_truncated = False
                            buffered_visible_source_chars = 0
                            _finalize_segment(register_links=False)
                        stream_wait_signals += 1
                        self.log_behavior_event(
                            event="turn_stream_wait_signal",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            stream_wait_signals=stream_wait_signals,
                            stream_total_chunks=stream_total_chunks,
                            progress_text=pending_progress_text,
                            turn_mode=normalized_turn_mode,
                        )
                        _finalize_segment()
                        yield self._build_progress_signal(pending_progress_text)
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
                tool_calls = getattr(message_chunk, "tool_calls", []) or []
                if tool_calls:
                    pending_progress_text = self._describe_tool_calls_for_progress(tool_calls)
                    suppress_live_text_until_completion = True
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
                    segment_accumulated += str(message_chunk.content)
                    cleaned = segment_accumulated
                    if not cleaned.strip():
                        stream_filtered_empty += 1
                        continue
                    expanded = self.expand_link_aliases(customer_id=customer_id, text=cleaned)
                    if expanded.strip():
                        expanded, truncated = self._truncate_user_visible_reply(expanded)
                        if suppress_live_text_until_completion:
                            buffered_visible = expanded
                            buffered_visible_truncated = truncated
                            buffered_visible_source_chars = len(cleaned.strip())
                            continue
                        if _precommit_active():
                            buffered_visible = expanded
                            buffered_visible_truncated = truncated
                            buffered_visible_source_chars = len(cleaned.strip())
                            continue
                        buffered_visible = ""
                        buffered_visible_truncated = False
                        buffered_visible_source_chars = 0
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
            if buffered_visible and not yielded_any and not approval_handoff_detected:
                buffered_candidate = buffered_visible.strip()
                if self._looks_like_provisional_reply(buffered_candidate):
                    self.log_behavior_event(
                        event="turn_stream_precommit_discarded",
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        output_chars=len(buffered_candidate),
                        reason="provisional_only",
                        turn_mode=normalized_turn_mode,
                    )
                    buffered_visible = ""
                    buffered_visible_truncated = False
                    buffered_visible_source_chars = 0
                else:
                    yielded_any = True
                    stream_visible_yields += 1
                    if first_visible_yield_ms is None:
                        first_visible_yield_ms = int((time.monotonic() - stream_started_at) * 1000)
                    self.log_behavior_event(
                        event="turn_stream_precommit_flushed",
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        output_chars=len(buffered_candidate),
                        elapsed_ms=int((time.monotonic() - stream_started_at) * 1000),
                        turn_mode=normalized_turn_mode,
                    )
                    yield buffered_visible
                    if buffered_visible_truncated:
                        self.log_behavior_event(
                            event="turn_stream_reply_truncated",
                            trace_id=turn_trace_id,
                            thread_id=thread_id,
                            customer_id=customer_id,
                            max_chars=self._max_user_reply_chars,
                            output_chars=buffered_visible_source_chars,
                            truncated_chars=len(buffered_candidate),
                            turn_mode=normalized_turn_mode,
                        )
                    buffered_visible = ""
                    buffered_visible_truncated = False
                    buffered_visible_source_chars = 0
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
                if fallback_text and not self._looks_like_provisional_reply(fallback_text):
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
                elif fallback_text:
                    self.log_behavior_event(
                        event="turn_stream_fallback_discarded",
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        output_chars=len(fallback_text),
                        reason="provisional_only",
                        turn_mode=normalized_turn_mode,
                    )
                latest_human_index = -1
                for index, message in enumerate(fallback_messages):
                    if isinstance(message, HumanMessage):
                        latest_human_index = index
                for message in reversed(fallback_messages[latest_human_index + 1 :]):
                    if fallback_yielded:
                        break
                    if isinstance(message, AIMessage) and (message.content or "").strip():
                        cleaned = str(message.content)
                        if cleaned.strip() and not self._looks_like_provisional_reply(cleaned):
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
                        if cleaned.strip():
                            self.log_behavior_event(
                                event="turn_stream_fallback_discarded",
                                trace_id=turn_trace_id,
                                thread_id=thread_id,
                                customer_id=customer_id,
                                output_chars=len(cleaned.strip()),
                                reason="provisional_only",
                                turn_mode=normalized_turn_mode,
                            )
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
            self.reset_active_thread_id(thread_scope_token)
            self.reset_active_customer_id(customer_scope_token)
            self._thread_inputs.end_turn(turn_state)

    async def _get_registered_interactive_session(self, *, thread_id: str) -> Any | None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id:
            return None
        lock = getattr(self, "_interactive_sessions_lock", None)
        sessions = getattr(self, "_interactive_sessions", None)
        if lock is None or sessions is None:
            return None
        async with lock:
            return sessions.get(safe_thread_id)

    async def register_interactive_session(self, *, thread_id: str, session: Any) -> None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id:
            return
        if getattr(self, "_interactive_sessions_lock", None) is None:
            self._interactive_sessions_lock = asyncio.Lock()
        if getattr(self, "_interactive_sessions", None) is None:
            self._interactive_sessions = {}
        async with self._interactive_sessions_lock:
            self._interactive_sessions[safe_thread_id] = session

    async def clear_interactive_session(self, *, thread_id: str, session: Any | None = None) -> None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id:
            return
        lock = getattr(self, "_interactive_sessions_lock", None)
        sessions = getattr(self, "_interactive_sessions", None)
        if lock is None or sessions is None:
            return
        async with lock:
            current = sessions.get(safe_thread_id)
            if session is None or current is session:
                sessions.pop(safe_thread_id, None)

    async def drain_interactive_fragments(self, *, thread_id: str) -> list[str]:
        session = await self._get_registered_interactive_session(thread_id=thread_id)
        if session is None or not hasattr(session, "drain_graph_fragments"):
            return []
        try:
            drained = await session.drain_graph_fragments()
        except Exception:
            logger.exception(
                "Failed to drain interactive fragments for thread_id=%s",
                thread_id,
            )
            return []
        return [str(item).strip() for item in drained if str(item).strip()]

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

    async def decide_intake_workflow(
        self,
        *,
        customer_id: str,
        workflow: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        execution_feedback: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return a structured decision for one intake workflow conversation."""
        invoke_error: str | None = None
        decision: _IntakeWorkflowDecision | None = None
        workflow_id = str(workflow.get("workflow_id", "") or "").strip() or "workflow"
        conversation_summary = conversation.get("summary") if isinstance(conversation, dict) else {}
        conversation_id = str(
            (conversation_summary or {}).get("conversation_id", "")
            if isinstance(conversation_summary, dict)
            else ""
        ).strip() or "conversation"
        latest_inbound_id = str(
            (conversation_summary or {}).get("latest_inbound_message_id", "")
            if isinstance(conversation_summary, dict)
            else ""
        ).strip() or "latest"
        structured_thread_id = f"intake_decision_{workflow_id}_{conversation_id}"
        structured_trace_id = f"intake_{workflow_id}_{conversation_id}_{latest_inbound_id}"
        tool_enabled_runtime = (
            getattr(self, "_graph", None) is not None
            and getattr(self, "_wake_execution_model_with_tools", None) is not None
            and callable(getattr(self, "ainvoke_text", None))
        )
        sink_type = str(workflow.get("sink_type", "") or "").strip().lower()
        prefer_agent_runtime = tool_enabled_runtime and (
            sink_type in {"google_sheets_composio", "generic_composio_write"}
            or bool(execution_feedback)
        )
        model = getattr(self, "_wake_execution_model", None) or self._model
        if prefer_agent_runtime:
            try:
                raw = await self.ainvoke_text(
                    thread_id=f"wake_intake_{workflow_id}_{conversation_id}_{latest_inbound_id}",
                    customer_id=customer_id,
                    text=_build_intake_workflow_agent_prompt(
                        customer_id=customer_id,
                        workflow=workflow,
                        conversation=conversation,
                        active_booking=active_booking,
                        recent_completed_booking=recent_completed_booking,
                        execution_feedback=execution_feedback,
                    ),
                    turn_mode="routine_wake",
                    include_pending_context=False,
                    prompt_mode_override="literal_chat",
                )
                parsed = _parse_schema_from_text(raw, _IntakeWorkflowDecision)
                if isinstance(parsed, _IntakeWorkflowDecision):
                    decision = parsed
                    invoke_error = None
            except Exception as exc:
                invoke_error = f"{type(exc).__name__}: {exc}"
        if decision is None:
            decision, invoke_error = await self._invoke_structured_model(
                model=model,
                schema=_IntakeWorkflowDecision,
                messages=[
                    SystemMessage(content=_build_intake_workflow_system_prompt()),
                    HumanMessage(
                        content=_build_intake_workflow_human_prompt(
                            customer_id=customer_id,
                            workflow=workflow,
                            conversation=conversation,
                            active_booking=active_booking,
                            recent_completed_booking=recent_completed_booking,
                            execution_feedback=execution_feedback,
                        )
                    ),
                ],
                stable_prefix_count=1,
                call_context={
                    "call_site": "intake_workflow_decision",
                    "trace_id": structured_trace_id,
                    "thread_id": structured_thread_id,
                    "customer_id": customer_id,
                    "turn_mode": "routine_wake",
                    "prompt_mode": "structured_intake",
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                    "latest_inbound_message_id": latest_inbound_id,
                },
            )
        if decision is None or not isinstance(decision, _IntakeWorkflowDecision):
            return {
                "ok": False,
                "error": (
                    f"intake_workflow_decision_error:{invoke_error}"
                    if invoke_error
                    else "intake_workflow_decision_error:invalid_output"
                ),
            }
        return {
            "ok": True,
            "matches_workflow": bool(decision.matches_workflow),
            "confidence": float(decision.confidence),
            "conversation_summary": str(decision.conversation_summary).strip()[:500],
            "extracted_fields": dict(decision.extracted_fields),
            "missing_fields": [str(item).strip() for item in decision.missing_fields if str(item).strip()],
            "reply_action": str(decision.reply_action).strip().lower() or "none",
            "reply_text": str(decision.reply_text).strip(),
            "ready_to_save": bool(decision.ready_to_save),
            "booking_action": str(decision.booking_action).strip().lower() or "ignore",
            "save_payload": dict(decision.save_payload),
            "sink_arguments": dict(decision.sink_arguments),
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
        ctx = self._ensure_active_customer_id_ctx()
        previous = str(ctx.get() or "").strip()
        token = ctx.set(cid)
        self._active_customer_id = cid
        return (token, previous)

    def reset_active_customer_id(self, token: object) -> None:
        ctx = self._ensure_active_customer_id_ctx()
        previous = ""
        raw_token = token
        if isinstance(token, tuple) and len(token) == 2:
            raw_token, previous = token
            previous = str(previous or "").strip()
        try:
            ctx.reset(raw_token)
        except ValueError:
            ctx.set(previous)
        self._active_customer_id = str(ctx.get() or "").strip()

    def get_active_customer_id(self) -> str:
        return str(self._ensure_active_customer_id_ctx().get() or "").strip()

    def _ensure_active_customer_id_ctx(self) -> contextvars.ContextVar[str]:
        ctx = getattr(self, "_active_customer_id_ctx", None)
        if isinstance(ctx, contextvars.ContextVar):
            return ctx
        ctx = contextvars.ContextVar("opentulpa_active_customer_id", default="")
        self._active_customer_id_ctx = ctx
        return ctx

    def set_active_thread_id(self, thread_id: str):
        tid = str(thread_id or "").strip()
        ctx = self._ensure_active_thread_id_ctx()
        previous = str(ctx.get() or "").strip()
        token = ctx.set(tid)
        self._active_thread_id = tid
        return (token, previous)

    def reset_active_thread_id(self, token: object) -> None:
        ctx = self._ensure_active_thread_id_ctx()
        previous = ""
        raw_token = token
        if isinstance(token, tuple) and len(token) == 2:
            raw_token, previous = token
            previous = str(previous or "").strip()
        try:
            ctx.reset(raw_token)
        except ValueError:
            ctx.set(previous)
        self._active_thread_id = str(ctx.get() or "").strip()

    def get_active_thread_id(self) -> str:
        return str(self._ensure_active_thread_id_ctx().get() or "").strip()

    def _ensure_active_thread_id_ctx(self) -> contextvars.ContextVar[str]:
        ctx = getattr(self, "_active_thread_id_ctx", None)
        if isinstance(ctx, contextvars.ContextVar):
            return ctx
        ctx = contextvars.ContextVar("opentulpa_active_thread_id", default="")
        self._active_thread_id_ctx = ctx
        return ctx

    async def execute_tool(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
        customer_id: str | None = None,
        thread_id: str | None = None,
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
        customer_token = self.set_active_customer_id(cid)
        thread_token = self.set_active_thread_id(str(thread_id or "").strip())
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
            self.reset_active_thread_id(thread_token)
            self.reset_active_customer_id(customer_token)
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
