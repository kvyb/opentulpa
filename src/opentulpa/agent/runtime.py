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
import hashlib
import inspect
import json
import logging
import os
import re
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.agent import model_pool as _model_pool
from opentulpa.agent.context_compaction import (
    compact_thread_context_for_turn,
    thread_context_needs_compaction,
)
from opentulpa.agent.context_compaction import (
    compress_rollup as _compress_rollup,
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
from opentulpa.agent.context_engine import ContextEngine
from opentulpa.agent.context_engine import (
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
from opentulpa.agent.model_provider_profile import provider_prompt_cache_profile
from opentulpa.agent.openrouter_chat_factory import openrouter_app_headers
from opentulpa.agent.runtime_context_provider import RuntimeContextSourceProvider
from opentulpa.agent.runtime_input import (
    MergedInputSuppressedError,
    ThreadInputCoordinator,
)
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.agent.turn_context_preparer import prepare_turn_context
from opentulpa.agent.turn_plan import turn_plan_enabled_for_turn_mode
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
from opentulpa.memory.service import MEMORY_KIND_PRIORITY

logger = logging.getLogger(__name__)

_MEMORY_GROUNDING_KIND_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("preferences_and_directives", ("directive_fact", "preference_fact")),
    (
        "durable_personal_facts",
        ("user_profile_fact", "life_fact", "relationship_fact", "contact_fact"),
    ),
    ("aspirations_and_plans", ("aspirations_fact", "project_fact")),
    ("active_projects_or_workflows", ("workflow_fact", "skill_fact")),
    ("technical_or_code_facts", ("code_fact", "credential_fact")),
    ("relevant_files_or_media", ("file_fact", "media_fact")),
    ("fallback_thread_context", ("thread_context_rollup",)),
)

_LLM_CALL_TRACE_LIMIT = 100


def _redact_inline_trace_string(value: str) -> str:
    raw = str(value)
    if not raw.lower().startswith("data:"):
        return raw
    if ";base64," in raw:
        prefix, _, _ = raw.partition(";base64,")
        return f"{prefix};base64,[redacted]"
    prefix, _, _ = raw.partition(",")
    return f"{prefix},[redacted]"


def _init_runtime_chat_model(
    model_name: str,
    *,
    base_kwargs: dict[str, Any],
    openrouter_base_url: str | None,
    reasoning_effort: str | None,
) -> Any:
    return _model_pool.init_runtime_chat_model(
        model_name,
        base_kwargs=base_kwargs,
        openrouter_base_url=openrouter_base_url,
        reasoning_effort=reasoning_effort,
        init_chat_model_func=init_chat_model,
        chat_openai_cls=ChatOpenAI,
    )


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


def _langchain_callback_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        safe_key = str(key)
        safe_value = _json_safe(value)
        if isinstance(safe_value, str):
            safe[safe_key] = safe_value
            continue
        safe[safe_key] = json.dumps(
            safe_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    return safe


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


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except Exception:
        return None


def _maybe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _first_float(mapping: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _maybe_float(mapping.get(key))
        if value is not None:
            return value
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
            input_details = _usage_object_to_dict(usage_metadata.get("input_token_details"))
            output_details = _usage_object_to_dict(usage_metadata.get("output_token_details"))
            usage = {
                "prompt_tokens": usage_metadata.get("input_tokens"),
                "completion_tokens": usage_metadata.get("output_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
                "input_tokens": usage_metadata.get("input_tokens"),
                "output_tokens": usage_metadata.get("output_tokens"),
                "prompt_tokens_details": {
                    "cached_tokens": input_details.get("cache_read"),
                    "cache_write_tokens": input_details.get("cache_creation"),
                },
                "completion_tokens_details": {
                    "reasoning_tokens": output_details.get("reasoning"),
                },
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
    if cached_tokens is None:
        cached_tokens = _maybe_int(usage.get("prompt_cache_hit_tokens"))
    cache_write_tokens = _maybe_int(prompt_details.get("cache_write_tokens"))
    if cache_write_tokens is None:
        cache_write_tokens = _maybe_int(usage.get("prompt_cache_miss_tokens"))
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
    total_cost = _maybe_float(cost)
    if cost_details:
        fields["native_cost_details"] = cost_details
        total_cost = total_cost or _first_float(
            cost_details,
            "total",
            "cost",
            "total_cost",
            "upstream_inference_cost",
        )
        prompt_cost = _first_float(
            cost_details,
            "prompt",
            "input",
            "prompt_cost",
            "input_cost",
            "upstream_inference_prompt_cost",
        )
        completion_cost = _first_float(
            cost_details,
            "completion",
            "completions",
            "output",
            "completion_cost",
            "completions_cost",
            "output_cost",
            "upstream_inference_completion_cost",
            "upstream_inference_completions_cost",
        )
        if prompt_cost is not None:
            fields["native_cost_prompt_usd"] = prompt_cost
        if completion_cost is not None:
            fields["native_cost_completion_usd"] = completion_cost
    if total_cost is not None:
        fields["native_cost_usd"] = total_cost
    return fields


def _extract_response_metadata_trace_fields(response: Any) -> dict[str, Any]:
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, dict):
        return {}
    fields: dict[str, Any] = {}
    generation_id = str(metadata.get("id") or "").strip()
    if generation_id:
        fields["openrouter_generation_id"] = generation_id
    model_provider = str(metadata.get("model_provider") or "").strip()
    if model_provider:
        fields["response_model_provider"] = model_provider
    response_model_name = str(metadata.get("model_name") or "").strip()
    if response_model_name:
        fields["response_model_name"] = response_model_name
    system_fingerprint = str(metadata.get("system_fingerprint") or "").strip()
    if system_fingerprint:
        fields["response_system_fingerprint"] = system_fingerprint
    return fields


def _stream_chunk_has_reasoning(message_chunk: Any) -> bool:
    additional_kwargs = getattr(message_chunk, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        if str(additional_kwargs.get("reasoning_content") or "").strip():
            return True
        reasoning_details = additional_kwargs.get("reasoning_details")
        if isinstance(reasoning_details, list) and reasoning_details:
            return True
        if str(additional_kwargs.get("reasoning") or "").strip():
            return True
    response_metadata = getattr(message_chunk, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage = _usage_object_to_dict(response_metadata.get("usage"))
        completion_details = _usage_object_to_dict(
            usage.get("completion_tokens_details")
            or usage.get("completionTokensDetails")
            or response_metadata.get("completion_tokens_details")
        )
        reasoning_tokens = _maybe_int(completion_details.get("reasoning_tokens"))
        if reasoning_tokens is not None and reasoning_tokens > 0:
            return True
    return False


def _tool_trace_name(tool: Any) -> str:
    return str(getattr(tool, "name", "") or "").strip()


def _fallback_tool_schema(tool: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "name": _tool_trace_name(tool),
        "description": str(getattr(tool, "description", "") or ""),
    }
    args_schema = getattr(tool, "args_schema", None)
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    if callable(model_json_schema):
        with suppress(Exception):
            schema["args_schema"] = model_json_schema()
    elif args_schema is not None:
        schema["args_schema"] = str(args_schema)
    return schema


def _tool_schema_trace_fields(runtime: Any, turn_mode: str) -> dict[str, Any]:
    normalized_turn_mode = str(turn_mode or "").strip().lower()
    if not normalized_turn_mode:
        return {}
    tools_for_turn_mode = getattr(runtime, "tools_for_turn_mode", None)
    if not callable(tools_for_turn_mode):
        return {}
    try:
        tools = list(tools_for_turn_mode(normalized_turn_mode))
    except Exception as exc:
        return {"bound_tool_schema_error": f"{type(exc).__name__}: {exc}"}
    if not tools:
        registered_tools = getattr(runtime, "_tools", {})
        if isinstance(registered_tools, dict):
            tools = list(registered_tools.values())

    schemas: list[Any] = []
    names: list[str] = []
    errors: list[str] = []
    for tool in tools:
        name = _tool_trace_name(tool)
        if name:
            names.append(name)
        try:
            schema = convert_to_openai_tool(tool)
        except Exception as exc:
            errors.append(f"{name or type(tool).__name__}: {type(exc).__name__}")
            schema = _fallback_tool_schema(tool)
        schemas.append(_json_safe(schema))

    serialized = json.dumps(schemas, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    fields: dict[str, Any] = {
        "bound_tool_count": len(tools),
        "bound_tool_names": names,
        "bound_tool_schema_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "bound_tool_schema_chars": len(serialized),
    }
    if errors:
        fields["bound_tool_schema_errors"] = errors
    return fields


def _hash_json(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prompt_cache_trace_fields(
    serialized_messages: list[dict[str, Any]],
    *,
    stable_prefix_count: int,
) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "prompt_hash": _hash_json(serialized_messages),
    }
    stable_count = max(0, min(int(stable_prefix_count), len(serialized_messages)))
    if stable_count:
        stable_messages = serialized_messages[:stable_count]
        fields["stable_prefix_hash"] = _hash_json(stable_messages)
        fields["stable_prefix_chars"] = sum(
            len(str(item.get("text", "") or "")) for item in stable_messages
        )
    cache_breakpoint_indexes: list[int] = []
    for index, message in enumerate(serialized_messages):
        content = message.get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and isinstance(item.get("cache_control"), dict)
            for item in content
        ):
            cache_breakpoint_indexes.append(index)
    if cache_breakpoint_indexes:
        breakpoint_index = cache_breakpoint_indexes[-1]
        cacheable_messages = serialized_messages[: breakpoint_index + 1]
        fields["cache_breakpoint_index"] = breakpoint_index
        fields["cache_breakpoint_count"] = len(cache_breakpoint_indexes)
        fields["cache_breakpoint_prefix_hash"] = _hash_json(cacheable_messages)
        fields["cache_breakpoint_prefix_chars"] = sum(
            len(str(item.get("text", "") or "")) for item in cacheable_messages
        )

    first_system = next(
        (item for item in serialized_messages if str(item.get("role", "") or "") == "system"),
        None,
    )
    first_non_system = next(
        (item for item in serialized_messages if str(item.get("role", "") or "") != "system"),
        None,
    )
    if first_system is not None:
        fields["sticky_first_system_hash"] = _hash_json(first_system)
        fields["sticky_first_system_chars"] = len(str(first_system.get("text", "") or ""))
    if first_non_system is not None:
        fields["sticky_first_non_system_hash"] = _hash_json(first_non_system)
        fields["sticky_first_non_system_chars"] = len(str(first_non_system.get("text", "") or ""))
    return fields


_LINK_ID_TOKEN_RE = re.compile(r"\blink_[A-Za-z0-9]{4,12}\b")
STREAM_WAIT_SIGNAL = "__TULPA_STREAM_WAIT__"
STREAM_PROGRESS_PREFIX = "__TULPA_STREAM_PROGRESS__:"
STREAM_EMPTY_REPLY_FALLBACK = (
    "I couldn't produce a visible user-facing reply for that step. "
    "Please retry, and I will continue from the latest state."
)
STREAM_PRECOMMIT_SECONDS = 0.75
CONTEXT_COMPACTION_BACKGROUND_DRAIN_SECONDS = 30.0


@dataclass(frozen=True)
class AgentStreamEvent:
    event: str
    payload: dict[str, Any]


_PROVISIONAL_REPLY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*i can also\s+", re.IGNORECASE),
    re.compile(r"^\s*i can (?:search|check|look|fetch|inspect|try)\b", re.IGNORECASE),
    re.compile(r"^\s*let me\b", re.IGNORECASE),
    re.compile(r"^\s*(?:sure|ok|okay|absolutely|got it)[!,.]?\s+let me\b", re.IGNORECASE),
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
    "uploaded_file_inspect_structure": "Inspecting uploaded file",
    "business_knowledge_index": "Preparing business knowledge",
    "business_knowledge_query": "Querying business knowledge",
    "user_context_add_files": "Adding files to user context",
    "user_context_query": "Querying user context",
    "user_context_reindex": "Reindexing user context",
    "browser_use_run": "Using the browser",
    "browser_use_owner_input_submit": "Continuing browser verification",
}
WORKFLOW_SETUP_TOOL_NAMES: set[str] = {
    "send_owner_update",
    "tool_group_list",
    "tool_group_describe",
    "tool_group_exec",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_send",
    "uploaded_file_analyze",
    "uploaded_file_inspect_structure",
    "business_knowledge_index",
    "business_knowledge_query",
    "user_context_add_files",
    "user_context_query",
    "user_context_list_sources",
    "user_context_find_sources",
    "user_context_reindex",
    "user_context_archive_sources",
    "user_context_promote_to_intake",
    "intake_workflow_list",
    "intake_workflow_get",
    "intake_workflow_setup_begin",
    "intake_workflow_setup_get",
    "intake_workflow_setup_update",
    "intake_workflow_setup_preflight",
    "intake_workflow_setup_propose_current",
    "intake_workflow_setup_mark_proposed",
    "intake_workflow_setup_confirm_current",
    "intake_workflow_setup_commit",
    "intake_workflow_setup_finalize_confirmation",
    "intake_workflow_setup_pause",
    "intake_workflow_setup_cancel",
    "telegram_business_status",
    "composio_status",
    "composio_authorize_toolkit",
    "composio_wait_for_connection",
    "composio_toolkits",
    "composio_connected_accounts",
    "composio_tool_search",
    "composio_tool_schema",
}

INTERACTIVE_NATIVE_TOOL_NAMES: set[str] = {
    "send_owner_update",
    "turn_plan",
    "server_time",
    "tool_group_list",
    "tool_group_describe",
    "tool_group_exec",
}

ROUTINE_WAKE_NATIVE_TOOL_NAMES: set[str] = {
    "server_time",
    "tool_group_list",
    "tool_group_describe",
    "tool_group_exec",
}

CUSTOMER_ID_REQUIRED_TOOLS: set[str] = {
    "memory_search",
    "memory_add",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_send",
    "tulpa_file_send",
    "web_image_send",
    "uploaded_file_analyze",
    "uploaded_file_inspect_structure",
    "business_knowledge_index",
    "business_knowledge_query",
    "user_context_add_files",
    "user_context_query",
    "user_context_list_sources",
    "user_context_find_sources",
    "user_context_reindex",
    "user_context_archive_sources",
    "user_context_promote_to_intake",
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
    "browser_use_task_get",
    "browser_use_task_screenshot",
    "browser_use_task_control",
    "browser_use_owner_input_submit",
    "tulpa_run_terminal",
    "tool_group_exec",
}


class _WakeClassification(BaseModel):
    model_config = ConfigDict(extra="ignore")

    notify_user: bool = False
    reason: str = ""


class _RoutineCreateIntentDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    allow_create: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class _WorkflowSetupInterruptionDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = True
    kind: str = "setup_input"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status_reply: str = ""
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
    sink_action: str = "none"
    sink_payload: dict[str, Any] = Field(default_factory=dict)
    sink_arguments: dict[str, Any] = Field(default_factory=dict)
    needs_business_knowledge: bool = False
    business_knowledge_query: str = ""
    knowledge_source_refs: list[Any] = Field(default_factory=list)
    grounding_status: str = ""
    reason: str = ""


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
        "save_payload (object), sink_action (string), sink_payload (object), "
        "sink_arguments (object), needs_business_knowledge (bool), "
        "business_knowledge_query (string), knowledge_source_refs (string array), "
        "grounding_status (string), reason (string).\n\n"
        "Allowed booking_action values: ignore, update_active, edit_recent_completed, create_new_booking.\n"
        "Allowed reply_action values: none, send_reply, mark_cancelled.\n\n"
        "Decision policy:\n"
        "- Treat conversation.unanswered_customer_messages as the active customer turn when present.\n"
        "- If multiple unanswered customer messages are present, interpret them together before deciding fields, booking action, knowledge needs, and reply text.\n"
        "- Do not base the decision only on latest_inbound_message_text_preview when unanswered_customer_messages contains more context.\n"
        "- Default mode is not an intent filter: unless workflow.intent_match_required is true, treat messages from the configured source as candidates for this workflow.\n"
        "- In default mode, set matches_workflow=true for greetings, casual openers, ambiguous early-stage messages, and business-adjacent questions when a useful reply can move the conversation forward.\n"
        "- Only when workflow.intent_match_required is true, use matches_workflow=false for messages that are not clearly pursuing the workflow intent.\n"
        "- Use matches_workflow=false in default mode only for clearly unrelated conversations where no useful workflow reply should be sent.\n"
        "- If the customer asks a business/service/pricing/booking question that is close to the workflow but outside its configured scope, return matches_workflow=false, booking_action=ignore, reply_action=send_reply with a concise redirect based on workflow instructions.\n"
        "- Confidence should reflect how certain you are in the match and booking decision.\n"
        "- Confidence guide: 0.9+ very clear, 0.7-0.89 likely, 0.4-0.69 ambiguous, below 0.4 weak evidence.\n\n"
        "Field extraction policy:\n"
        "- Extract fields only from evidence in the conversation or saved state.\n"
        "- Do not invent or infer missing business details unless the value is explicitly or near-explicitly stated.\n"
        "- workflow.business_facts and workflow.workflow_skill are owner-provided workflow configuration. Use those compact inline facts as authoritative business facts unless workflow.knowledge_answer directly contradicts them.\n"
        "- When answering business facts from source material, use workflow.business_facts, workflow.workflow_skill, workflow.knowledge_answer, or request a service-side lookup only when workflow.knowledge_file_ids is non-empty. If no owner-provided inline fact, bound file, or answer text supports the fact, say you need to confirm instead of inventing it.\n"
        "- Business knowledge returns only plain answer text. Leave knowledge_source_refs empty. Set grounding_status to grounded when workflow.business_facts, workflow.workflow_skill, or that answer text directly supports the business fact; otherwise use no_source.\n"
        "- If an active booking already contains source-backed business facts and the latest customer message only supplies missing customer-provided fields, do not query business knowledge again. Reuse the active booking fields and return the save decision.\n"
        "- Light normalization is allowed: trim whitespace, standardize obvious time/date phrasing, preserve meaning.\n"
        "- If customer messages conflict, prefer the latest customer-provided value unless the newer message is too vague to override the earlier one.\n"
        "- Do not ask for a field that is already reliably known unless the value is conflicting or unclear.\n"
        "- missing_fields must list only fields that are truly still needed before save.\n\n"
        "Source identity policy:\n"
        "- conversation.summary.incoming_user_id, latest_inbound_sender_id, username, and platform are backend-provided source metadata.\n"
        "- Treat source identity values as factual when present. Do not invent them if missing.\n"
        "- If workflow instructions ask to record the inbound user id, Telegram id, Instagram id, username, or similar source identity, use these conversation summary fields.\n\n"
        "Booking-state fast path:\n"
        "- If the latest customer message clearly cancels, reschedules, corrects, or otherwise updates an active "
        "booking or a recent completed booking, treat it as a booking-state operation. Reuse the saved booking "
        "fields, do not require workflow.knowledge_answer or business_knowledge_query unless the latest message "
        "asks for a new source-backed service or price fact, and return the JSON decision directly.\n"
        "- For a clear cancellation of an active or recent completed booking: use matches_workflow=true, "
        "booking_action=update_active or edit_recent_completed, reply_action=mark_cancelled, and a reply_text "
        "that confirms the cancellation. If the sink should be updated, ready_to_save=true and save_payload "
        'should contain the merged booking fields plus status="cancelled".\n\n'
        "Business knowledge request policy:\n"
        "- If workflow has knowledge_file_ids, workflow.knowledge_answer is empty, and the latest message needs a "
        "source-backed price, service, policy, menu, or capability fact that is not already present in saved "
        "booking state or workflow.business_facts/workflow.workflow_skill, set needs_business_knowledge=true and business_knowledge_query to one concise natural "
        "language query. Do not guess the fact and do not ask the customer before the knowledge lookup.\n"
        "- If workflow.knowledge_file_ids is empty, never set needs_business_knowledge=true. Use workflow "
        "business_facts, workflow_skill, instructions, and field guidance when possible; otherwise say the fact needs confirmation and ask the "
        "next missing required field.\n"
        "- When needs_business_knowledge=true, return the partial classification and extracted customer-provided "
        "fields, but set ready_to_save=false, reply_action=none, and keep reply_text empty.\n"
        "- If workflow.knowledge_answer is present, use it and set needs_business_knowledge=false. If it says "
        "NO_SOURCE or does not support the requested fact, ask one clarifying question or redirect according to "
        "workflow scope instead of requesting knowledge again.\n\n"
        "Reply policy:\n"
        "- If details are missing, set reply_action=send_reply with one concise, high-leverage follow-up question.\n"
        "- Ask at most one compact question at a time unless a single sentence can naturally request two tightly related missing fields.\n"
        "- reply_text should be plain outbound DM text, not explanations about JSON or system behavior.\n"
        '- If no reply is needed, use reply_action=none and reply_text="".\n'
        "- Use mark_cancelled only when the customer clearly cancels, abandons, or says they no longer want the booking.\n"
        "- Never ask for extra confirmation. This is background workflow execution.\n\n"
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
        "- When ready_to_save=true, reply_text must describe the completed action. Never ask the customer to confirm "
        "a booking or change that you are saving now. If confirmation is still needed, set ready_to_save=false.\n"
        "- When workflow instructions explicitly require writing fields before all required fields are collected, set "
        "sink_action=upsert_partial and put only those interim fields in sink_payload. Use this for cases like "
        "recording incoming_user_id or username on the first inbound message. Otherwise use sink_action=none and sink_payload={}. "
        "Do not use upsert_partial for ignored conversations.\n"
        "- When needs_business_knowledge=true and workflow.knowledge_file_ids is non-empty, set ready_to_save=false, "
        'reply_action=none, reply_text="", and business_knowledge_query to the exact missing source-backed fact.\n'
        "- sink_arguments may contain sink-tool arguments or overrides discovered via tools or context "
        "(for example sheetName for Google Sheets). These are merged into the final sink write.\n"
        "- When ready_to_save=false, save_payload should usually be empty.\n"
        "- When ready_to_save=true, leave sink_action=none; the service will perform the final save using save_payload.\n"
        "- When no sink overrides are needed, sink_arguments should usually be empty.\n"
        "- conversation_summary should be a short operational summary of what the customer currently wants.\n"
        "- reason should briefly explain the match decision and booking_action.\n\n"
        "Examples:\n"
        "1. Customer asks for a wash, gives day and car type, but no time -> matches_workflow=true, booking_action=create_new_booking or update_active, reply_action=send_reply, missing_fields includes time, ready_to_save=false.\n"
        "2. Customer says 'actually make it 4pm instead' after a recent completed booking -> matches_workflow=true, booking_action=edit_recent_completed, extracted_fields.time='4pm'.\n"
        "3. Customer says 'also book my other car tomorrow evening' after an earlier finished booking -> matches_workflow=true, booking_action=create_new_booking.\n"
        "4. Default mode greeting like 'hi' or 'привет' -> matches_workflow=true, booking_action=ignore, reply_action=send_reply with a concise useful opener.\n"
        "5. Strict intent mode unrelated chat -> matches_workflow=false, booking_action=ignore, reply_action=none.\n"
        "6. Customer says 'cancel it please' after an active or recent completed booking -> matches_workflow=true, booking_action=update_active or edit_recent_completed, reply_action=mark_cancelled, reply_text confirms cancellation.\n"
        "7. Customer asks for a price from bound knowledge and workflow.knowledge_answer is empty -> needs_business_knowledge=true with a concise business_knowledge_query.\n"
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
    business_facts = safe_workflow.get("business_facts")
    compact_business_facts = _compact_jsonish_dict(
        business_facts,
        item_limit=12,
        char_limit=220,
    )
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
    compact_knowledge_answer = _trim_text_chars(
        safe_workflow.get("knowledge_answer", ""), limit=3600
    )
    return {
        "workflow_id": str(safe_workflow.get("workflow_id", "") or "").strip(),
        "name": _trim_text_chars(safe_workflow.get("name", ""), limit=80),
        "intent_description": _trim_text_chars(
            safe_workflow.get("intent_description", ""),
            limit=500,
        ),
        "intent_match_required": bool(safe_workflow.get("intent_match_required", False)),
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
        "business_facts": compact_business_facts,
        "workflow_skill": _trim_text_chars(safe_workflow.get("workflow_skill", ""), limit=3200),
        "knowledge_file_ids": [
            str(item or "").strip()
            for item in list(safe_workflow.get("knowledge_file_ids") or [])[:12]
            if str(item or "").strip()
        ],
        "knowledge_answer": compact_knowledge_answer,
        "sink_type": str(safe_workflow.get("sink_type", "") or "").strip(),
        "channel": str(safe_workflow.get("channel", "") or "").strip(),
        "provider": str(safe_workflow.get("provider", "") or "").strip(),
        "sink": compact_sink,
        "policies": safe_workflow.get("policies", {})
        if isinstance(safe_workflow.get("policies"), dict)
        else {},
    }


def _compact_recent_messages(messages: Any, *, limit: int = 6) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    compact: list[dict[str, str]] = []
    for item in messages[-limit:]:
        if not isinstance(item, dict):
            continue
        compact.append(
            {
                "id": _trim_text_chars(item.get("id", ""), limit=80),
                "created_time": _trim_text_chars(item.get("created_time", ""), limit=64),
                "sender_role": _trim_text_chars(item.get("sender_role", ""), limit=24),
                "sender_id": _trim_text_chars(item.get("sender_id", ""), limit=80),
                "sender_username": _trim_text_chars(item.get("sender_username", ""), limit=48),
                "text": _trim_text_chars(item.get("text", ""), limit=300),
            }
        )
    return compact


def _compact_unanswered_customer_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    return _compact_recent_messages(messages, limit=40)


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
        "platform": _trim_text_chars(safe_summary.get("platform", ""), limit=64),
        "conversation_id": _trim_text_chars(safe_summary.get("conversation_id", ""), limit=120),
        "recipient_id": _trim_text_chars(safe_summary.get("recipient_id", ""), limit=120),
        "incoming_user_id": _trim_text_chars(safe_summary.get("incoming_user_id", ""), limit=120),
        "username": _trim_text_chars(safe_summary.get("username", ""), limit=64),
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
        "latest_inbound_sender_id": _trim_text_chars(
            safe_summary.get("latest_inbound_sender_id", ""),
            limit=120,
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
        "unanswered_customer_messages": _compact_unanswered_customer_messages(
            safe_conversation.get("unanswered_customer_messages")
        ),
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
        "- Use all unanswered customer messages as one active customer turn when provided.\n"
        "- Inspect external state only when the workflow explicitly requires it before this decision.\n"
        "- Return strict JSON only as the final answer.\n\n"
        "Tool-use guidance:\n"
        "- You may use normal tools, especially uploaded_file_get, uploaded_file_analyze, uploaded_file_search, "
        "composio_tool_search, composio_tool_schema, composio_tool_execute, and "
        "composio_instagram_reply_precheck when they materially help.\n"
        "- Do not read or search a sink just because it is Google Sheets or Composio. A write sink is not an "
        "availability source by default.\n"
        "- Check external availability only if the workflow explicitly says to check conflicts or open slots. "
        "If the workflow says not to promise availability or to only record customer preferences, do not check availability.\n"
        "- If sink_config already includes concrete target metadata such as spreadsheetId and sheetName, do not "
        "search for sink schema or read tools; leave sink_arguments empty unless a required write argument is missing.\n"
        "- If the sink write needs missing concrete target metadata such as a Google Sheets tab name, inspect the "
        "sink with Composio tools and return only those discovered values in sink_arguments.\n"
        "- Do not call business_knowledge_query during this intake decision turn. If workflow.knowledge_file_ids is "
        "non-empty, workflow.knowledge_answer is empty, and a source-backed business fact is needed, return "
        "needs_business_knowledge=true with one concise business_knowledge_query. The intake service will run "
        "the oracle and call you again with workflow.knowledge_answer.\n"
        "- workflow.business_facts and workflow.workflow_skill are owner-provided workflow configuration. Use them for compact inline business facts such as prices, service menu highlights, hours, discounts, locations, and policies.\n"
        "- If workflow.knowledge_file_ids is empty, never set needs_business_knowledge=true. Use workflow "
        "business_facts, workflow_skill, instructions, and field guidance when possible; otherwise say the fact needs confirmation and ask the "
        "next missing required field.\n"
        "- If workflow.knowledge_answer is present, use it and return a final decision. Do not request knowledge again.\n"
        "- If active_booking.extracted_fields already contains the needed source-backed business facts and the latest inbound message supplies only missing customer-provided fields, return the merged decision without requesting business knowledge.\n"
        "- Prefer minimal read-only tool usage first.\n"
        "- Do not create, update, delete, or run workflows/routines from inside this turn.\n"
        "- Do not call intake_workflow_upsert, intake_workflow_delete, intake_workflow_run, routine_create, or routine_delete.\n"
        "- Do not ask the user for confirmation. This is background execution.\n"
        "- Do not send the outbound source reply or perform the final booking write yourself in this turn; "
        "the intake workflow service will do the final idempotent reply/save after your decision.\n\n"
        "Final answer contract:\n"
        "- Return strict JSON only with keys:\n"
        "  matches_workflow, confidence, conversation_summary, extracted_fields, missing_fields, "
        "reply_action, reply_text, ready_to_save, booking_action, save_payload, sink_action, "
        "sink_payload, sink_arguments, needs_business_knowledge, business_knowledge_query, "
        "knowledge_source_refs, grounding_status, reason.\n"
        "- booking_action must be one of: ignore, update_active, edit_recent_completed, create_new_booking.\n"
        "- reply_action must be one of: none, send_reply, mark_cancelled.\n"
        "- If availability is blocked or conflicting, do not set ready_to_save=true.\n"
        "- If details are missing, ask one concise follow-up question in reply_text.\n"
        "- If ready_to_save=true, reply_text must be a final saved/updated/cancelled confirmation and must not ask "
        "for confirmation. If you still need confirmation from the customer, use ready_to_save=false.\n"
        "- conversation.summary contains backend-provided source identity fields such as platform, incoming_user_id, "
        "latest_inbound_sender_id, and username. Use them when workflow instructions ask you to record Telegram, "
        "Instagram, user, or username metadata. Do not invent missing ids.\n"
        "- If workflow instructions explicitly require writing fields before all required fields are collected, set "
        "sink_action=upsert_partial and put only those interim fields in sink_payload. Otherwise use sink_action=none.\n"
        "- If needs_business_knowledge=true and workflow.knowledge_file_ids is non-empty, set ready_to_save=false, "
        'reply_action=none, reply_text="", and business_knowledge_query to the exact missing source-backed fact.\n'
        "- If the customer asks a business/service/pricing/booking question that is close to the workflow but outside its configured scope, return matches_workflow=false, booking_action=ignore, reply_action=send_reply with a concise redirect based on workflow instructions.\n"
        "- If the latest customer message is only cancelling, rescheduling, or correcting an active/recent booking, "
        "reuse active_booking or recent_completed_booking and do not call business_knowledge_query unless a new "
        "source-backed business fact is requested.\n"
        "- If execution_feedback is present, you are replanning after a real tool or application error. "
        "Read it carefully, do not repeat the same failing action unchanged, and adapt your next decision.\n"
        "- For business facts in reply_text or save_payload, leave knowledge_source_refs empty and set grounding_status=grounded when workflow.business_facts, workflow.workflow_skill, workflow.knowledge_answer, or business_knowledge_query directly supports the fact. If none supports a fact, set grounding_status=no_source and ask to confirm instead.\n"
        "- sink_arguments is for sink-specific write arguments or overrides discovered during this turn; "
        "leave it empty when not needed.\n"
        "- Unless workflow.intent_match_required is true, do not use intent as a front-door filter; reply usefully when the source conversation can be moved forward.\n\n"
        f"customer_id={customer_id}\n"
        f"workflow={json.dumps(compact_workflow, ensure_ascii=False)}\n"
        f"conversation={json.dumps(compact_conversation, ensure_ascii=False)}\n"
        f"active_booking={json.dumps(compact_active_booking, ensure_ascii=False)}\n"
        f"recent_completed_booking={json.dumps(compact_recent_booking, ensure_ascii=False)}\n"
        f"execution_feedback={json.dumps(compact_feedback, ensure_ascii=False)}"
    )


def _build_intake_workflow_context_prompt(
    *,
    customer_id: str,
    workflow: dict[str, Any],
) -> str:
    compact_workflow = _compact_workflow_for_prompt(workflow)
    return (
        "INTAKE_WORKFLOW_CONTEXT\n"
        "Stable owner-defined workflow configuration for this intake run.\n"
        f"customer_id={customer_id}\n"
        f"workflow={json.dumps(compact_workflow, ensure_ascii=False)}"
    )


def _build_intake_workflow_state_prompt(
    *,
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
        "INTAKE_CONVERSATION_STATE\n"
        "Volatile conversation state for the current inbound message.\n"
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


def _normalize_knowledge_source_refs(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = ""
        if isinstance(item, dict):
            text = str(
                item.get("source_ref")
                or item.get("ref")
                or item.get("chunk_id")
                or item.get("file_id")
                or ""
            ).strip()
            if not text:
                with suppress(Exception):
                    text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        normalized.append(text)
    return normalized


class OpenTulpaLangGraphRuntime:
    def __init__(
        self,
        *,
        app_url: str,
        openrouter_api_key: str,
        model_name: str,
        reasoning_effort: str | None = "medium",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        wake_classifier_model_name: str | None = None,
        wake_execution_model_name: str | None = None,
        telegram_media_model_name: str | None = None,
        workflow_setup_input_classifier_model_name: str | None = None,
        checkpoint_db_path: str,
        recursion_limit: int = 30,
        max_completion_tokens: int = 4096,
        max_user_reply_chars: int = 4000,
        context_events: EventContextService | None = None,
        customer_profile_service: CustomerProfileService | None = None,
        thread_rollup_service: ThreadRollupService | None = None,
        link_alias_service: LinkAliasService | None = None,
        context_token_limit: int = 20000,
        context_rollup_tokens: int = 2200,
        context_recent_tokens: int = 3500,
        context_compaction_source_tokens: int = 12000,
        context_compaction_model_name: str | None = "google/gemini-3-flash-preview",
        input_debounce_seconds: float = 0.65,
        proactive_heartbeat_default_hours: int = 3,
        behavior_log_enabled: bool = True,
        behavior_log_path: str = ".opentulpa/logs/agent_behavior.jsonl",
        browser_use_headless: bool = True,
        browser_use_model_override: str | None = None,
        browser_use_max_concurrent_tasks: int = 2,
        browser_use_task_retention_seconds: int = 1800,
        browser_use_user_data_dir: str | None = ".opentulpa/browser_use_profiles",
        browser_use_api_key: str | None = None,
        browser_use_cloud_proxy_country_code: str | None = "us",
        browser_use_cloud_timeout_minutes: int = 15,
        capsolver_api_key: str | None = None,
        prompt_caching_enabled: bool = True,
        prompt_cache_ttl_1h: bool = False,
        langfuse_tracer: Any | None = None,
    ) -> None:
        self.app_url = app_url.rstrip("/")
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = (
            str(openrouter_base_url or "").strip() or "https://openrouter.ai/api/v1"
        )
        self.model_name = _normalize_model_name(model_name)
        self._reasoning_effort = str(reasoning_effort or "").strip() or None
        self._max_completion_tokens = max(128, min(int(max_completion_tokens), 32768))
        self._max_user_reply_chars = max(500, min(int(max_user_reply_chars), 20000))
        self._wake_classifier_model_name = (
            _normalize_model_name(str(wake_classifier_model_name))
            if str(wake_classifier_model_name or "").strip()
            else self.model_name
        )
        self._wake_execution_model_name = (
            _normalize_model_name(str(wake_execution_model_name))
            if str(wake_execution_model_name or "").strip()
            else self.model_name
        )
        self._telegram_media_model_name = (
            _normalize_model_name(str(telegram_media_model_name))
            if str(telegram_media_model_name or "").strip()
            else "google/gemini-3.1-flash-lite-preview"
        )
        workflow_setup_classifier_model = (
            str(workflow_setup_input_classifier_model_name).strip()
            if str(workflow_setup_input_classifier_model_name or "").strip()
            else "z-ai/glm-5.2"
        )
        self._workflow_setup_input_classifier_model_name = _normalize_model_name(
            workflow_setup_classifier_model
        )
        self._context_compaction_model_name = (
            _normalize_model_name(str(context_compaction_model_name).strip())
            if str(context_compaction_model_name or "").strip()
            else self.model_name
        )
        self.checkpoint_db_path = checkpoint_db_path
        self.recursion_limit = recursion_limit
        self._context_events = context_events
        self._customer_profile_service = customer_profile_service
        self._thread_rollup_service = thread_rollup_service
        self._link_alias_service = link_alias_service
        self._composio_service: Any | None = None
        self._workflow_setup_service: Any | None = None
        self._context_token_limit = max(6000, min(30000, int(context_token_limit)))
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
        self._proactive_heartbeat_default_hours = max(
            1, min(int(proactive_heartbeat_default_hours), 24)
        )
        self._behavior_log_enabled = bool(behavior_log_enabled)
        raw_behavior_path = (
            str(behavior_log_path or "").strip() or ".opentulpa/logs/agent_behavior.jsonl"
        )
        self._behavior_log_path = Path(raw_behavior_path).resolve()
        self._behavior_log_lock = threading.Lock()
        self._llm_call_trace_path = self._behavior_log_path.parent / "llm_call_traces.jsonl"
        self._llm_call_trace_lock = threading.Lock()
        self._llm_call_trace_limit = _LLM_CALL_TRACE_LIMIT
        self._llm_prompt_hashes_by_trace_key: dict[str, list[str]] = {}
        self._thread_checkpoint_locks: dict[str, asyncio.Lock] = {}
        self._thread_checkpoint_locks_guard = asyncio.Lock()
        self._context_compaction_background_tasks: set[asyncio.Task[Any]] = set()
        if self._behavior_log_enabled:
            self._behavior_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._browser_use_headless = bool(browser_use_headless)
        self._browser_use_model_override = str(browser_use_model_override or "").strip()
        self._browser_use_max_concurrent_tasks = max(1, int(browser_use_max_concurrent_tasks))
        self._browser_use_task_retention_seconds = max(60, int(browser_use_task_retention_seconds))
        self._browser_use_user_data_dir = str(browser_use_user_data_dir or "").strip()
        self._browser_use_api_key = str(browser_use_api_key or "").strip()
        self._browser_use_cloud_proxy_country_code = str(
            browser_use_cloud_proxy_country_code or ""
        ).strip()
        self._browser_use_cloud_timeout_minutes = max(
            1, min(int(browser_use_cloud_timeout_minutes), 240)
        )
        self._capsolver_api_key = str(capsolver_api_key or "").strip()
        self._prompt_caching_enabled = bool(prompt_caching_enabled)
        self._prompt_cache_ttl_1h = bool(prompt_cache_ttl_1h)
        self._langfuse_tracer = langfuse_tracer
        self._context_engine = ContextEngine()
        self._context_source_provider = RuntimeContextSourceProvider(self)
        self._browser_use_local_manager: Any | None = None
        self._headroom_service: Any | None = None
        self._interactive_sessions_lock = asyncio.Lock()
        self._interactive_sessions: dict[str, Any] = {}
        self._interactive_update_senders_lock = asyncio.Lock()
        self._interactive_update_senders: dict[str, Any] = {}
        self._interactive_update_sent_keys: dict[str, set[str]] = {}
        self._interactive_file_senders_lock = asyncio.Lock()
        self._interactive_file_senders: dict[str, Any] = {}
        self._active_customer_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
            "opentulpa_active_customer_id",
            default="",
        )
        self._active_turn_mode_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
            "opentulpa_active_turn_mode",
            default="interactive",
        )
        self._active_customer_id = ""
        self._active_turn_mode = "interactive"
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
        default_headers = openrouter_app_headers(base_url=self.openrouter_base_url)
        if default_headers:
            model_init_kwargs["default_headers"] = default_headers
            model_init_kwargs["use_responses_api"] = False
        if self._reasoning_effort:
            model_init_kwargs["reasoning_effort"] = self._reasoning_effort

        self._model = _init_runtime_chat_model(
            self.model_name,
            base_kwargs=model_init_kwargs,
            openrouter_base_url=self.openrouter_base_url,
            reasoning_effort=self._reasoning_effort,
        )
        if self._wake_classifier_model_name == self.model_name:
            self._wake_classifier_model = self._model
        else:
            try:
                self._wake_classifier_model = _init_runtime_chat_model(
                    self._wake_classifier_model_name,
                    base_kwargs=model_init_kwargs,
                    openrouter_base_url=self.openrouter_base_url,
                    reasoning_effort=self._reasoning_effort,
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
                self._wake_execution_model = _init_runtime_chat_model(
                    self._wake_execution_model_name,
                    base_kwargs=model_init_kwargs,
                    openrouter_base_url=self.openrouter_base_url,
                    reasoning_effort=self._reasoning_effort,
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
                self._telegram_media_model = _init_runtime_chat_model(
                    self._telegram_media_model_name,
                    base_kwargs=model_init_kwargs,
                    openrouter_base_url=self.openrouter_base_url,
                    reasoning_effort=self._reasoning_effort,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize Telegram media model '%s'; falling back to main model '%s'.",
                    self._telegram_media_model_name,
                    self.model_name,
                )
                self._telegram_media_model = self._model
        classifier_model_kwargs = dict(model_init_kwargs)
        classifier_model_kwargs["max_completion_tokens"] = min(
            int(classifier_model_kwargs.get("max_completion_tokens", 160) or 160),
            160,
        )
        if self._workflow_setup_input_classifier_model_name == self.model_name:
            self._workflow_setup_input_classifier_model = self._model
        elif self._workflow_setup_input_classifier_model_name == self._wake_classifier_model_name:
            self._workflow_setup_input_classifier_model = self._wake_classifier_model
        elif self._workflow_setup_input_classifier_model_name == self._wake_execution_model_name:
            self._workflow_setup_input_classifier_model = self._wake_execution_model
        elif self._workflow_setup_input_classifier_model_name == self._telegram_media_model_name:
            self._workflow_setup_input_classifier_model = self._telegram_media_model
        else:
            try:
                self._workflow_setup_input_classifier_model = _init_runtime_chat_model(
                    self._workflow_setup_input_classifier_model_name,
                    base_kwargs=classifier_model_kwargs,
                    openrouter_base_url=self.openrouter_base_url,
                    reasoning_effort=None,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize workflow setup input classifier model '%s'; "
                    "falling back to main model '%s'.",
                    self._workflow_setup_input_classifier_model_name,
                    self.model_name,
                )
                self._workflow_setup_input_classifier_model = self._model
        if self._context_compaction_model_name == self.model_name:
            self._context_compaction_model = self._model
        elif self._context_compaction_model_name == self._wake_classifier_model_name:
            self._context_compaction_model = self._wake_classifier_model
        elif self._context_compaction_model_name == self._wake_execution_model_name:
            self._context_compaction_model = self._wake_execution_model
        elif self._context_compaction_model_name == self._telegram_media_model_name:
            self._context_compaction_model = self._telegram_media_model
        elif (
            self._context_compaction_model_name == self._workflow_setup_input_classifier_model_name
        ):
            self._context_compaction_model = self._workflow_setup_input_classifier_model
        else:
            try:
                self._context_compaction_model = _init_runtime_chat_model(
                    self._context_compaction_model_name,
                    base_kwargs=model_init_kwargs,
                    openrouter_base_url=self.openrouter_base_url,
                    reasoning_effort=self._reasoning_effort,
                )
            except Exception:
                logger.exception(
                    "Failed to initialize context compaction model '%s'; "
                    "falling back to main model '%s'.",
                    self._context_compaction_model_name,
                    self.model_name,
                )
                self._context_compaction_model = self._model

        self._checkpointer_cm: Any | None = None
        self._checkpointer: Any | None = None
        self._graph = None
        self._tools: dict[str, Any] = {}
        self._model_with_tools = None
        self._workflow_setup_model_with_tools = None
        self._wake_execution_model_with_tools = None
        self._thread_inputs = ThreadInputCoordinator(debounce_seconds=self._input_debounce_seconds)
        self._internal_api = InternalApiClient(base_url=self.app_url)

    @property
    def link_alias_service(self) -> LinkAliasService | None:
        return self._link_alias_service

    @property
    def composio_service(self) -> Any | None:
        return self._composio_service

    @property
    def workflow_setup_service(self) -> Any | None:
        return self._workflow_setup_service

    def configure_api_services(
        self,
        *,
        link_alias_service: LinkAliasService | None = None,
        composio_service: Any | None = None,
        workflow_setup_service: Any | None = None,
    ) -> None:
        assert composio_service is None or hasattr(composio_service, "status")
        assert workflow_setup_service is None or hasattr(
            workflow_setup_service, "get_thread_session"
        )
        if link_alias_service is not None and self._link_alias_service is None:
            self._link_alias_service = link_alias_service
        if composio_service is not None:
            self._composio_service = composio_service
        if workflow_setup_service is not None:
            self._workflow_setup_service = workflow_setup_service

    def prompt_cache_profile(self, *, model_name: str | None = None) -> dict[str, Any]:
        target_model_name = str(model_name or getattr(self, "model_name", "") or "").strip()
        return dict(
            provider_prompt_cache_profile(
                enabled=bool(getattr(self, "_prompt_caching_enabled", False)),
                model_name=target_model_name,
                ttl_1h=bool(getattr(self, "_prompt_cache_ttl_1h", False)),
            )
        )

    def model_invoke_extras(self, *, model_name: str | None = None) -> dict[str, Any]:
        """Extra kwargs for main agent model.ainvoke (e.g. OpenRouter prompt cache_control)."""
        return _model_pool.model_invoke_extras(self, model_name=model_name)

    def _model_request_attempts(self, *, model_name: str | None = None) -> list[dict[str, Any]]:
        return [{"name": "default", "invoke_extras": {}, "call_context": {}}]

    def _resolve_model_name_for_runtime_call(
        self, model: Any, explicit_name: str | None = None
    ) -> str:
        if explicit_name:
            return str(explicit_name).strip()
        if model is getattr(self, "_wake_classifier_model", None):
            return str(getattr(self, "_wake_classifier_model_name", "") or "").strip()
        if model is getattr(self, "_wake_execution_model", None):
            return str(getattr(self, "_wake_execution_model_name", "") or "").strip()
        if model is getattr(self, "_workflow_setup_input_classifier_model", None):
            return str(
                getattr(self, "_workflow_setup_input_classifier_model_name", "") or ""
            ).strip()
        if model is getattr(self, "_wake_execution_model_with_tools", None):
            return str(getattr(self, "_wake_execution_model_name", "") or "").strip()
        if model is getattr(self, "_workflow_setup_model_with_tools", None):
            return str(getattr(self, "model_name", "") or "").strip()
        if model is getattr(self, "_model", None) or model is getattr(
            self, "_model_with_tools", None
        ):
            return str(getattr(self, "model_name", "") or "").strip()
        model_name = getattr(model, "model_name", None)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
        return str(getattr(self, "model_name", "") or "").strip()

    def model_with_tools_for_turn_mode(self, turn_mode: str) -> Any:
        normalized_turn_mode = str(turn_mode or "").strip().lower()
        if (
            normalized_turn_mode == "routine_wake"
            and self._wake_execution_model_with_tools is not None
        ):
            return self._wake_execution_model_with_tools
        if (
            normalized_turn_mode == "workflow_setup"
            and self._workflow_setup_model_with_tools is not None
        ):
            return self._workflow_setup_model_with_tools
        return self._model_with_tools

    def tools_for_turn_mode(self, turn_mode: str) -> list[Any]:
        normalized_turn_mode = str(turn_mode or "").strip().lower()
        native_names = (
            ROUTINE_WAKE_NATIVE_TOOL_NAMES
            if normalized_turn_mode == "routine_wake"
            else INTERACTIVE_NATIVE_TOOL_NAMES
        )
        blocked_tools: set[str] = set()
        if not turn_plan_enabled_for_turn_mode(normalized_turn_mode):
            blocked_tools.add("turn_plan")
        if normalized_turn_mode == "routine_wake":
            blocked_tools.add("send_owner_update")
        if normalized_turn_mode not in {"interactive", "workflow_setup"}:
            blocked_tools.add("browser_use_owner_input_submit")
        return [
            tool
            for name, tool in self._tools.items()
            if str(name or "").strip() in native_names
            and str(name or "").strip() not in blocked_tools
        ]

    def prepare_messages_for_prompt_cache(
        self,
        messages: list[Any],
        *,
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        cacheable_prefix_count: int | None = None,
    ) -> list[Any]:
        return _model_pool.prepare_messages_for_prompt_cache(
            self,
            messages,
            model_name=model_name,
            stable_prefix_count=stable_prefix_count,
            cacheable_prefix_count=cacheable_prefix_count,
        )

    async def ainvoke_model(
        self,
        model: Any,
        messages: list[Any],
        *,
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        cacheable_prefix_count: int | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> Any:
        return await _model_pool.ainvoke_model(
            self,
            model,
            messages,
            model_name=model_name,
            stable_prefix_count=stable_prefix_count,
            cacheable_prefix_count=cacheable_prefix_count,
            call_context=call_context,
        )

    async def astream_model(
        self,
        model: Any,
        messages: list[Any],
        *,
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        cacheable_prefix_count: int | None = None,
        call_context: dict[str, Any] | None = None,
        stream_config: Any | None = None,
    ) -> Any:
        return await _model_pool.astream_model(
            self,
            model,
            messages,
            model_name=model_name,
            stable_prefix_count=stable_prefix_count,
            cacheable_prefix_count=cacheable_prefix_count,
            call_context=call_context,
            stream_config=stream_config,
        )

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
    def _humanize_tool_identifier(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        cleaned = re.sub(r"[_-]+", " ", text).strip()
        if not cleaned:
            return ""
        if cleaned.isupper():
            cleaned = cleaned.lower()
        return cleaned[:1].upper() + cleaned[1:]

    @classmethod
    def _tool_group_exec_progress_label(cls, args: Any) -> str:
        if not isinstance(args, dict):
            return ""
        calls = args.get("calls")
        if isinstance(calls, list) and calls:
            labels = [
                cls._tool_group_exec_progress_label(call)
                for call in calls[:2]
                if isinstance(call, dict)
            ]
            labels = [label for label in labels if label]
            if labels:
                if len(labels) == 1:
                    return labels[0]
                return f"{labels[0]}, then {labels[1].lower()}"
        group = cls._humanize_tool_identifier(args.get("group"))
        command = cls._humanize_tool_identifier(args.get("command"))
        if group and command:
            group_prefix = f"{group.lower()} "
            if command.lower().startswith(group_prefix):
                command = command[len(group_prefix) :].strip()
                if command:
                    command = command[:1].upper() + command[1:]
            return f"{group}: {command}"
        return group or command

    @classmethod
    def _tool_call_progress_label(cls, call: Any) -> str:
        if not isinstance(call, dict):
            return ""
        name = str(call.get("name", "")).strip()
        if not name:
            return ""
        if name == "tool_group_exec":
            label = cls._tool_group_exec_progress_label(call.get("args"))
            if label:
                return label
        alias = _PROGRESS_TOOL_NAME_ALIASES.get(name)
        if alias is not None:
            return alias
        return cls._humanize_tool_identifier(name.replace("tulpa_", "").replace("browser_use_", ""))

    @staticmethod
    def _safe_tool_names_for_status(tool_calls: list[Any]) -> list[str]:
        assert isinstance(tool_calls, list)
        names: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "")).strip()
            if name:
                names.append(name)
        assert len(names) <= len(tool_calls)
        return names[:8]

    @staticmethod
    def _describe_tool_calls_for_progress(tool_calls: list[Any]) -> str:
        labels = [
            OpenTulpaLangGraphRuntime._tool_call_progress_label(call) for call in tool_calls[:2]
        ]
        labels = [label for label in labels if label]
        if not labels:
            return "Working on it…"
        if len(labels) == 1:
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
                reasoning_effort=self._reasoning_effort,
                headless=self._browser_use_headless,
                max_concurrent_tasks=self._browser_use_max_concurrent_tasks,
                task_retention_seconds=self._browser_use_task_retention_seconds,
                user_data_dir=self._browser_use_user_data_dir,
                capsolver_api_key=self._capsolver_api_key,
                browser_use_api_key=self._browser_use_api_key,
                browser_use_cloud_proxy_country_code=self._browser_use_cloud_proxy_country_code,
                browser_use_cloud_timeout_minutes=self._browser_use_cloud_timeout_minutes,
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
            logger.exception(
                "tool result compression failed for %s", str(tool_name or "").strip() or "tool"
            )
            return raw_result_text
        normalized = str(compressed or "").strip()
        return normalized or raw_result_text

    def log_behavior_event(self, *, event: str, **fields: Any) -> None:
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
        if bool(getattr(self, "_behavior_log_enabled", False)):
            lock = getattr(self, "_behavior_log_lock", None)
            path = getattr(self, "_behavior_log_path", None)
            if isinstance(path, Path):
                with suppress(Exception):
                    path.parent.mkdir(parents=True, exist_ok=True)
                if lock is None:
                    with suppress(Exception), path.open("a", encoding="utf-8") as f:
                        f.write(serialized + "\n")
                else:
                    with suppress(Exception), lock, path.open("a", encoding="utf-8") as f:
                        f.write(serialized + "\n")
        tracer = getattr(self, "_langfuse_tracer", None)
        record_event = getattr(tracer, "record_behavior_event", None)
        if callable(record_event):
            with suppress(Exception):
                record_event(payload)

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        if customer_id:
            fields.setdefault("customer_id", customer_id)
        self.log_behavior_event(event=event, **fields)

    @staticmethod
    def _normalize_llm_call_context(call_context: dict[str, Any] | None) -> dict[str, Any]:
        normalized = dict(call_context) if isinstance(call_context, dict) else {}
        normalized.pop("_langfuse_callback_attached", None)
        normalized.pop("_langfuse_graph_callback_covers_call", None)
        prompt_sections = normalized.get("prompt_sections")
        if isinstance(prompt_sections, str):
            normalized["prompt_sections"] = [
                part.strip() for part in prompt_sections.split(",") if part.strip()
            ]
        elif isinstance(prompt_sections, list):
            normalized["prompt_sections"] = [
                str(part).strip() for part in prompt_sections if str(part).strip()
            ]
        normalized["call_site"] = str(normalized.get("call_site") or "runtime_model_invoke").strip()
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

    def _prompt_change_trace_fields(
        self,
        *,
        model_name: str,
        serialized_prompt_messages: list[dict[str, Any]],
        call_context: dict[str, Any],
    ) -> dict[str, Any]:
        hashes = [_hash_json(message) for message in serialized_prompt_messages]
        trace_key_parts = [
            str(call_context.get("customer_id") or ""),
            str(call_context.get("thread_id") or ""),
            str(call_context.get("turn_mode") or ""),
            str(model_name or ""),
            str(call_context.get("call_site") or ""),
        ]
        trace_key = "|".join(trace_key_parts)
        previous_by_key = getattr(self, "_llm_prompt_hashes_by_trace_key", None)
        if not isinstance(previous_by_key, dict):
            self._llm_prompt_hashes_by_trace_key = {}
            previous_by_key = self._llm_prompt_hashes_by_trace_key
        previous = previous_by_key.get(trace_key)
        previous_by_key[trace_key] = hashes
        if previous is None:
            return {
                "prompt_first_changed_message_index": None,
                "prompt_changed_message_count": None,
                "prompt_previous_message_count": None,
            }
        first_changed: int | None = None
        changed_count = abs(len(hashes) - len(previous))
        for index, current_hash in enumerate(hashes[: len(previous)]):
            if current_hash == previous[index]:
                continue
            changed_count += 1
            if first_changed is None:
                first_changed = index
        if first_changed is None and len(hashes) != len(previous):
            first_changed = min(len(hashes), len(previous))
        return {
            "prompt_first_changed_message_index": first_changed,
            "prompt_changed_message_count": changed_count,
            "prompt_previous_message_count": len(previous),
        }

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
        normalized_context = self._normalize_llm_call_context(call_context)
        usage_fields = self.extract_response_usage_fields(response) if response is not None else {}
        metadata_fields = (
            _extract_response_metadata_trace_fields(response) if response is not None else {}
        )
        tool_schema_fields = _tool_schema_trace_fields(
            self,
            str(normalized_context.get("turn_mode") or ""),
        )
        response_content = getattr(response, "content", response) if response is not None else ""
        safe_response_content = _json_safe(response_content)
        serialized_prompt_messages = [_serialize_message(message) for message in prepared_messages]
        prompt_cache_fields = _prompt_cache_trace_fields(
            serialized_prompt_messages,
            stable_prefix_count=stable_prefix_count,
        )
        prompt_change_fields = self._prompt_change_trace_fields(
            model_name=model_name,
            serialized_prompt_messages=serialized_prompt_messages,
            call_context=normalized_context,
        )
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "model_name": str(model_name or "").strip(),
            "stable_prefix_count": int(stable_prefix_count),
            "prompt_messages": serialized_prompt_messages,
            "prompt_message_count": len(prepared_messages),
            "response_type": type(response).__name__ if response is not None else "",
            "response_message": _serialize_message(response) if response is not None else None,
            "response_text": _content_to_text(safe_response_content).strip()
            if response is not None
            else "",
            "response_content": safe_response_content,
            "response_tool_calls": _json_safe(getattr(response, "tool_calls", None)),
            "error": str(error or "").strip() or None,
            **usage_fields,
            **metadata_fields,
            **tool_schema_fields,
            **prompt_cache_fields,
            **prompt_change_fields,
        }
        for key, value in normalized_context.items():
            record[str(key)] = _json_safe(value)
        if bool(getattr(self, "_behavior_log_enabled", True)):
            self._write_llm_call_trace(record)
        tracer = getattr(self, "_langfuse_tracer", None)
        record_generation = getattr(tracer, "record_generation", None)
        callback_already_records = bool(
            isinstance(call_context, dict)
            and (
                call_context.get("_langfuse_callback_attached")
                or call_context.get("_langfuse_graph_callback_covers_call")
            )
        )
        if callable(record_generation) and not callback_already_records:
            with suppress(Exception):
                record_generation(record)

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

    async def _invoke_structured_model[StructuredModelT: BaseModel](
        self,
        *,
        model: Any,
        messages: list[Any],
        schema: type[StructuredModelT],
        model_name: str | None = None,
        stable_prefix_count: int = 0,
        cacheable_prefix_count: int | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> tuple[StructuredModelT | None, str | None]:
        return await _model_pool.invoke_structured_model(
            self,
            model=model,
            messages=messages,
            schema=schema,
            model_name=model_name,
            stable_prefix_count=stable_prefix_count,
            cacheable_prefix_count=cacheable_prefix_count,
            call_context=call_context,
            clean_json_text_block=_clean_json_text_block,
        )

    def extract_response_usage_fields(self, response: Any) -> dict[str, Any]:
        return dict(_extract_response_usage_fields(response))

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

    def resolve_link_aliases_in_args(
        self, *, customer_id: str, args: dict[str, Any]
    ) -> dict[str, Any]:
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
    def _normalize_memory_search_results(raw: Any) -> list[dict[str, Any]]:
        payload = (
            raw.get("results")
            if isinstance(raw, dict) and isinstance(raw.get("results"), list)
            else raw
        )
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
            kind = (
                str(item.get("kind") or metadata.get("kind") or "").strip().lower()
                or "thread_context_rollup"
            )
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
                    "thread_id": str(
                        item.get("thread_id") or metadata.get("thread_id") or ""
                    ).strip(),
                    "skill_name": str(
                        item.get("skill_name") or metadata.get("skill_name") or ""
                    ).strip(),
                }
            )
        return normalized

    @staticmethod
    def _memory_grounding_sort_key(item: dict[str, Any]) -> tuple[int, float]:
        kind = str(item.get("kind", "") or "").strip().lower()
        priority = int(MEMORY_KIND_PRIORITY.get(kind, 50))
        try:
            raw_score = item.get("score")
            score = float(raw_score) if raw_score is not None else 0.0
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
        selection_model = getattr(self, "_wake_classifier_model", None) or self._model
        decision, _ = await self._invoke_structured_model(
            model=selection_model,
            schema=_SkillSelectionDecision,
            messages=[
                SystemMessage(
                    content=(
                        "You select reusable skills for the current user request.\n"
                        "Return strict JSON object with key 'selected', an array of objects:\n"
                        '  {"name": string, "score": number, "reason": string}\n'
                        "Choose only skills that materially improve answer quality.\n"
                        "Never select persona, tone, or style-only skills for literal definitions, acronym expansions, translations, or short factual clarifications.\n"
                        "If the request is about reminders, schedules, recurring jobs, or cron, "
                        "prefer selecting routine-schedule-composer when available.\n"
                        "Prioritize skills that improve execution reliability and claim accuracy over style-only skills.\n"
                        'If none apply, return {"selected": []}.'
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
        available = (
            candidates if isinstance(candidates, list) else await self._list_available_skills(cid)
        )
        del available
        return {"skill_names": [], "context": ""}

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
        user_offset_minutes = _utc_offset_to_minutes(user_offset_text) if user_offset_text else None
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
            "user_utc_offset": str(user_offset_text or server_offset_text),
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

    async def _persist_rollup_memory(
        self, *, customer_id: str, thread_id: str, rollup: str
    ) -> None:
        await _persist_rollup_memory(
            self,
            customer_id=customer_id,
            thread_id=thread_id,
            rollup=rollup,
        )

    async def _thread_checkpoint_lock(self, thread_id: str) -> asyncio.Lock:
        tid = str(thread_id or "").strip()
        assert tid
        if getattr(self, "_thread_checkpoint_locks_guard", None) is None:
            self._thread_checkpoint_locks_guard = asyncio.Lock()
        if getattr(self, "_thread_checkpoint_locks", None) is None:
            self._thread_checkpoint_locks = {}
        async with self._thread_checkpoint_locks_guard:
            lock = self._thread_checkpoint_locks.get(tid)
            if lock is None:
                lock = asyncio.Lock()
                self._thread_checkpoint_locks[tid] = lock
            return lock

    @asynccontextmanager
    async def _thread_checkpoint_guard(
        self,
        *,
        thread_id: str,
        customer_id: str,
        trace_id: str,
        mode: str,
    ) -> AsyncIterator[None]:
        lock = await self._thread_checkpoint_lock(thread_id)
        started_at = time.monotonic()
        if lock.locked():
            self.log_behavior_event(
                event="turn_waiting_for_context_compaction",
                trace_id=trace_id,
                mode=mode,
                thread_id=thread_id,
                customer_id=customer_id,
            )
        await lock.acquire()
        waited_ms = int((time.monotonic() - started_at) * 1000)
        if waited_ms > 0:
            self.log_behavior_event(
                event="turn_context_lock_acquired",
                trace_id=trace_id,
                mode=mode,
                thread_id=thread_id,
                customer_id=customer_id,
                waited_ms=waited_ms,
            )
        try:
            yield
        finally:
            lock.release()

    @property
    def context_source_provider(self) -> RuntimeContextSourceProvider:
        return self._context_source_provider


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
        self._model_with_tools = self._model.bind_tools(self.tools_for_turn_mode("interactive"))
        self._workflow_setup_model_with_tools = self._model.bind_tools(
            self.tools_for_turn_mode("workflow_setup")
        )
        self._wake_execution_model_with_tools = self._wake_execution_model.bind_tools(
            self.tools_for_turn_mode("routine_wake")
        )
        self._graph = self._build_graph()
        manager = self.get_browser_use_local_manager()
        with suppress(Exception):
            preflight_error = await manager.preflight()
            if preflight_error:
                logger.warning("browser_use local preflight warning: %s", preflight_error)

    async def _drain_context_compaction_background_tasks(self) -> None:
        task_set = getattr(self, "_context_compaction_background_tasks", None)
        if not isinstance(task_set, set):
            return
        pending = {task for task in task_set if not task.done()}
        task_set.intersection_update(pending)
        if not pending:
            task_set.clear()
            return
        done, still_pending = await asyncio.wait(
            pending,
            timeout=CONTEXT_COMPACTION_BACKGROUND_DRAIN_SECONDS,
        )
        for task in done:
            with suppress(BaseException):
                task.result()
        task_set.difference_update(done)
        if still_pending:
            logger.warning(
                "context_compaction background persistence drain timed out pending_tasks=%s",
                len(still_pending),
            )
            for task in still_pending:
                task.cancel()
            task_set.difference_update(still_pending)

    async def shutdown(self) -> None:
        await self._drain_context_compaction_background_tasks()
        manager = self._browser_use_local_manager
        if manager is not None:
            with suppress(Exception):
                await manager.shutdown()
        self._browser_use_local_manager = None
        langfuse_tracer = getattr(self, "_langfuse_tracer", None)
        if langfuse_tracer is not None and hasattr(langfuse_tracer, "shutdown"):
            with suppress(Exception):
                langfuse_tracer.shutdown()
        if self._checkpointer_cm is not None:
            await self._checkpointer_cm.__aexit__(None, None, None)
        self._checkpointer_cm = None
        self._checkpointer = None
        self._graph = None
        self._model_with_tools = None
        self._workflow_setup_model_with_tools = None
        self._wake_execution_model_with_tools = None

    def healthy(self) -> bool:
        return self._graph is not None

    def _effective_recursion_limit(self, recursion_limit_override: int | None = None) -> int:
        if recursion_limit_override is None:
            return int(self.recursion_limit)
        return max(5, min(int(recursion_limit_override), 250))

    def _build_langfuse_callbacks(
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
        langfuse_tracer = getattr(self, "_langfuse_tracer", None)
        build_callbacks = getattr(langfuse_tracer, "build_callbacks", None)
        if not callable(build_callbacks):
            return []
        metadata: dict[str, Any] = {
            "thread_id": str(thread_id or "").strip(),
            "turn_mode": str(turn_mode or "").strip(),
            "prompt_mode": str(prompt_mode or "").strip(),
            "call_site": str(call_site or "").strip(),
            "model_name": str(model_name or "").strip(),
            "opentulpa_trace_id": str(trace_id or "").strip(),
        }
        metadata.update(_tool_schema_trace_fields(self, str(turn_mode or "").strip()))
        return cast(
            "list[Any]",
            build_callbacks(
                user_id=str(customer_id or "").strip() or None,
                trace_id=str(trace_id or "").strip() or None,
                session_id=str(thread_id or "").strip() or None,
                metadata=metadata,
                tags=[
                    item
                    for item in (str(turn_mode or "").strip(), str(prompt_mode or "").strip())
                    if item
                ],
            ),
        )

    def _model_with_callbacks(
        self, model: Any, *, call_context: dict[str, Any] | None = None
    ) -> Any:
        if model is None:
            return model
        context = dict(call_context or {})
        if bool(context.get("_langfuse_graph_callback_covers_call")):
            return model
        callbacks = self._build_langfuse_callbacks(
            customer_id=str(
                context.get("customer_id") or self.get_active_customer_id() or ""
            ).strip()
            or None,
            trace_id=str(context.get("trace_id") or "").strip() or None,
            thread_id=str(context.get("thread_id") or "").strip() or None,
            turn_mode=str(context.get("turn_mode") or "").strip() or None,
            prompt_mode=str(context.get("prompt_mode") or "").strip() or None,
            call_site=str(context.get("call_site") or "").strip() or "runtime_model_invoke",
            model_name=self._resolve_model_name_for_runtime_call(
                model, explicit_name=context.get("model_name")
            ),
        )
        if not callbacks:
            return model
        with_config = getattr(model, "with_config", None)
        if not callable(with_config):
            return model
        try:
            metadata = {
                "langfuse_user_id": str(
                    context.get("customer_id") or self.get_active_customer_id() or ""
                ).strip(),
                "langfuse_session_id": str(context.get("thread_id") or "").strip(),
                "langfuse_tags": [
                    item
                    for item in (
                        str(context.get("turn_mode") or "").strip(),
                        str(context.get("prompt_mode") or "").strip(),
                    )
                    if item
                ],
                "opentulpa_trace_id": str(context.get("trace_id") or "").strip(),
                "thread_id": str(context.get("thread_id") or "").strip(),
                "turn_mode": str(context.get("turn_mode") or "").strip(),
                "prompt_mode": str(context.get("prompt_mode") or "").strip(),
                "call_site": str(context.get("call_site") or "").strip() or "runtime_model_invoke",
            }
            metadata.update(
                _tool_schema_trace_fields(
                    self,
                    str(context.get("turn_mode") or "").strip(),
                )
            )
            configured_model = with_config(
                {
                    "callbacks": callbacks,
                    "metadata": _langchain_callback_metadata(metadata),
                    "tags": list(metadata["langfuse_tags"]),
                }
            )
            if isinstance(call_context, dict):
                call_context["_langfuse_callback_attached"] = True
            return configured_model
        except Exception:
            logger.exception("Failed to attach Langfuse callbacks to model invocation.")
            return model

    def _observability_trace_context(
        self,
        *,
        name: str,
        trace_id: str | None,
        customer_id: str | None,
        thread_id: str | None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        tracer = getattr(self, "_langfuse_tracer", None)
        trace_context = getattr(tracer, "trace_context", None)
        if not callable(trace_context):
            return nullcontext()
        return trace_context(
            name=name,
            trace_id=trace_id,
            user_id=customer_id,
            session_id=thread_id,
            input=input,
            metadata=metadata,
            tags=tags,
        )

    @staticmethod
    def _observability_ids_from_text(text: str) -> dict[str, str]:
        payload: dict[str, str] = {}
        for field in ("workflow_id", "routine_id", "conversation_id"):
            match = re.search(rf"\b{field}\s*[:=]\s*([^\s,;]+)", str(text or ""), re.IGNORECASE)
            if match:
                payload[field] = match.group(1).strip()
        return payload

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
        turn_mode_scope_token = self.set_active_turn_mode(normalized_turn_mode)
        trace_context = self._observability_trace_context(
            name=f"opentulpa.turn.{normalized_turn_mode}",
            trace_id=turn_trace_id,
            customer_id=customer_id,
            thread_id=thread_id,
            input={"text": str(effective_text or ""), "mode": "ainvoke"},
            metadata={
                **self._observability_ids_from_text(str(effective_text or "")),
                "turn_mode": normalized_turn_mode,
                "mode": "ainvoke",
                "prompt_mode_override": str(prompt_mode_override or "").strip() or None,
                "forced_skill_names": forced_skill_names or [],
            },
            tags=[normalized_turn_mode, "ainvoke"],
        )
        trace_context.__enter__()
        try:
            if turn_state is not None or not (
                normalized_turn_mode == "interactive" and interactive_session is not None
            ):
                self.log_behavior_event(
                    event="turn_start",
                    trace_id=turn_trace_id,
                    mode="ainvoke",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    input_chars=len(str(effective_text or "")),
                    turn_mode=normalized_turn_mode,
                )
            async with self._thread_checkpoint_guard(
                thread_id=thread_id,
                customer_id=customer_id,
                trace_id=turn_trace_id,
                mode="ainvoke",
            ):
                await compact_thread_context_for_turn(
                    self,
                    thread_id=thread_id,
                    customer_id=customer_id,
                )
                prepared = await prepare_turn_context(
                    self.context_source_provider,
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=str(effective_text or ""),
                    turn_mode=normalized_turn_mode,
                    include_pending_context=include_pending_context,
                    trace_id=turn_trace_id,
                    recursion_limit_override=recursion_limit_override,
                    forced_skill_names=forced_skill_names,
                    prompt_mode_override=prompt_mode_override,
                    build_langfuse_callbacks=self._build_langfuse_callbacks,
                    tool_schema_trace_fields=lambda mode: _tool_schema_trace_fields(self, mode),
                    langchain_callback_metadata=_langchain_callback_metadata,
                )
                result = await self._graph.ainvoke(prepared.graph_input, config=prepared.config)
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
            latest_human_index = -1
            for index, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    latest_human_index = index
            current_turn_messages = (
                messages[latest_human_index + 1 :] if latest_human_index >= 0 else messages
            )
            for message in reversed(current_turn_messages):
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
                        self._context_events.clear_events(
                            customer_id, through_id=prepared.through_id
                        )
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
            with suppress(Exception):
                trace_context.__exit__(None, None, None)
            self.reset_active_turn_mode(turn_mode_scope_token)
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
        stream_precommit_seconds: float | None = None,
        stream_incremental_deltas: bool = False,
        stream_status_events: bool = False,
    ) -> AsyncIterator[str | AgentStreamEvent]:
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
        if (
            turn_state is None
            and normalized_turn_mode == "interactive"
            and interactive_session is not None
        ):
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
        turn_mode_scope_token = self.set_active_turn_mode(normalized_turn_mode)
        trace_context = self._observability_trace_context(
            name=f"opentulpa.turn.{normalized_turn_mode}",
            trace_id=turn_trace_id,
            customer_id=customer_id,
            thread_id=thread_id,
            input={"text": str(effective_text or ""), "mode": "astream"},
            metadata={
                **self._observability_ids_from_text(str(effective_text or "")),
                "turn_mode": normalized_turn_mode,
                "mode": "astream",
                "prompt_mode_override": str(prompt_mode_override or "").strip() or None,
                "forced_skill_names": forced_skill_names or [],
            },
            tags=[normalized_turn_mode, "astream"],
        )
        trace_context.__enter__()
        checkpoint_lock: asyncio.Lock | None = None
        checkpoint_lock_acquired = False
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
            checkpoint_lock = await self._thread_checkpoint_lock(thread_id)
            checkpoint_wait_started_at = time.monotonic()
            if checkpoint_lock.locked():
                self.log_behavior_event(
                    event="turn_waiting_for_context_compaction",
                    trace_id=turn_trace_id,
                    mode="astream",
                    thread_id=thread_id,
                    customer_id=customer_id,
                )
            await checkpoint_lock.acquire()
            checkpoint_lock_acquired = True
            checkpoint_waited_ms = int((time.monotonic() - checkpoint_wait_started_at) * 1000)
            if checkpoint_waited_ms > 0:
                self.log_behavior_event(
                    event="turn_context_lock_acquired",
                    trace_id=turn_trace_id,
                    mode="astream",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    waited_ms=checkpoint_waited_ms,
                )
            if stream_status_events and await thread_context_needs_compaction(
                self,
                thread_id=thread_id,
            ):
                self.log_behavior_event(
                    event="turn_context_compaction_status_emitted",
                    trace_id=turn_trace_id,
                    mode="astream",
                    thread_id=thread_id,
                    customer_id=customer_id,
                    turn_mode=normalized_turn_mode,
                )
                yield AgentStreamEvent(
                    event="status",
                    payload={
                        "status": "active",
                        "message": "Compacting chat history...",
                    },
                )
            await compact_thread_context_for_turn(
                self,
                thread_id=thread_id,
                customer_id=customer_id,
            )
            prepared = await prepare_turn_context(
                self.context_source_provider,
                thread_id=thread_id,
                customer_id=customer_id,
                text=str(effective_text or ""),
                turn_mode=normalized_turn_mode,
                include_pending_context=include_pending_context,
                trace_id=turn_trace_id,
                recursion_limit_override=None,
                forced_skill_names=forced_skill_names,
                prompt_mode_override=prompt_mode_override,
                build_langfuse_callbacks=self._build_langfuse_callbacks,
                tool_schema_trace_fields=lambda mode: _tool_schema_trace_fields(self, mode),
                langchain_callback_metadata=_langchain_callback_metadata,
            )
            config = prepared.config
            prepared.graph_input["stream_model_calls"] = True
            segment_accumulated = ""
            stream_key = ""
            yielded_any = False
            in_tool_phase = False
            suppress_live_text_until_completion = False
            stream_started_at = time.monotonic()
            stream_no_visible_timeout_s = float(
                str(os.environ.get("AGENT_STREAM_NO_VISIBLE_PROGRESS_SECONDS", "210")).strip()
                or "210"
            )
            effective_stream_precommit_seconds = (
                STREAM_PRECOMMIT_SECONDS
                if stream_precommit_seconds is None
                else max(0.0, float(stream_precommit_seconds))
            )
            stream_total_chunks = 0
            stream_agent_chunks = 0
            stream_tool_chunks = 0
            stream_wait_signals = 0
            stream_visible_yields = 0
            stream_filtered_empty = 0
            stream_filtered_blank_expanded = 0
            reasoning_status_emitted = False
            first_visible_yield_ms: int | None = None
            buffered_visible = ""
            buffered_visible_truncated = False
            buffered_visible_source_chars = 0
            emitted_visible_text = ""
            pending_progress_text = "Working on it…"
            self.log_behavior_event(
                event="turn_stream_loop_start",
                trace_id=turn_trace_id,
                thread_id=thread_id,
                customer_id=customer_id,
                stream_no_visible_timeout_s=stream_no_visible_timeout_s,
                stream_precommit_seconds=effective_stream_precommit_seconds,
                turn_mode=normalized_turn_mode,
            )

            def _precommit_active() -> bool:
                if effective_stream_precommit_seconds <= 0 or yielded_any:
                    return False
                return (time.monotonic() - stream_started_at) < effective_stream_precommit_seconds

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

            def _reasoning_event(message_chunk: Any) -> AgentStreamEvent | None:
                nonlocal reasoning_status_emitted
                if (
                    not stream_status_events
                    or reasoning_status_emitted
                    or not _stream_chunk_has_reasoning(message_chunk)
                ):
                    return None
                reasoning_status_emitted = True
                return AgentStreamEvent(
                    event="reasoning",
                    payload={
                        "status": "active",
                        "message": "Reasoning...",
                    },
                )

            async for message_chunk, metadata in self._graph.astream(
                prepared.graph_input,
                config=config,
                stream_mode="messages",
            ):
                stream_total_chunks += 1
                reasoning_event = _reasoning_event(message_chunk)
                if reasoning_event is not None:
                    yield reasoning_event
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
                    if (
                        self._stream_chunk_is_tool_phase(node_name, message_chunk)
                        and not in_tool_phase
                    ):
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
                if in_tool_phase:
                    in_tool_phase = False
                    suppress_live_text_until_completion = yielded_any
                    stream_key = ""
                    _finalize_segment()
                tool_calls = getattr(message_chunk, "tool_calls", []) or []
                if tool_calls:
                    pending_progress_text = self._describe_tool_calls_for_progress(tool_calls)
                    suppress_live_text_until_completion = True
                    if stream_status_events:
                        yield AgentStreamEvent(
                            event="tool_call",
                            payload={
                                "status": "started",
                                "message": pending_progress_text,
                                "tool_names": self._safe_tool_names_for_status(tool_calls),
                                "tool_call_count": len(tool_calls),
                            },
                        )
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
                        if self._looks_like_provisional_reply(expanded):
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
                            first_visible_yield_ms = int(
                                (time.monotonic() - stream_started_at) * 1000
                            )
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
                        visible_output = expanded
                        if stream_incremental_deltas:
                            visible_output = (
                                expanded[len(emitted_visible_text) :]
                                if emitted_visible_text
                                and expanded.startswith(emitted_visible_text)
                                else expanded
                            )
                            emitted_visible_text = expanded
                        if visible_output:
                            yield visible_output
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
            if buffered_visible:
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
                    flush_event = (
                        "turn_stream_buffered_completion_flushed"
                        if yielded_any
                        else "turn_stream_precommit_flushed"
                    )
                    yielded_any = True
                    stream_visible_yields += 1
                    if first_visible_yield_ms is None:
                        first_visible_yield_ms = int((time.monotonic() - stream_started_at) * 1000)
                    self.log_behavior_event(
                        event=flush_event,
                        trace_id=turn_trace_id,
                        thread_id=thread_id,
                        customer_id=customer_id,
                        output_chars=len(buffered_candidate),
                        elapsed_ms=int((time.monotonic() - stream_started_at) * 1000),
                        turn_mode=normalized_turn_mode,
                    )
                    visible_output = buffered_visible
                    if stream_incremental_deltas:
                        visible_output = (
                            buffered_visible[len(emitted_visible_text) :]
                            if emitted_visible_text
                            and buffered_visible.startswith(emitted_visible_text)
                            else buffered_visible
                        )
                        emitted_visible_text = buffered_visible
                    if visible_output:
                        yield visible_output
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
        except Exception as exc:
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
                error=f"{type(exc).__name__}: {exc}"[:500],
                turn_mode=normalized_turn_mode,
            )
            raise
        finally:
            if checkpoint_lock_acquired and checkpoint_lock is not None:
                checkpoint_lock.release()
            with suppress(Exception):
                trace_context.__exit__(None, None, None)
            self.reset_active_turn_mode(turn_mode_scope_token)
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

    async def clear_interactive_session(
        self, *, thread_id: str, session: Any | None = None
    ) -> None:
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

    async def register_interactive_update_sender(self, *, thread_id: str, sender: Any) -> None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id or sender is None:
            return
        if getattr(self, "_interactive_update_senders_lock", None) is None:
            self._interactive_update_senders_lock = asyncio.Lock()
        if getattr(self, "_interactive_update_senders", None) is None:
            self._interactive_update_senders = {}
        if getattr(self, "_interactive_update_sent_keys", None) is None:
            self._interactive_update_sent_keys = {}
        async with self._interactive_update_senders_lock:
            self._interactive_update_senders[safe_thread_id] = sender
            self._interactive_update_sent_keys[safe_thread_id] = set()

    async def clear_interactive_update_sender(
        self,
        *,
        thread_id: str,
        sender: Any | None = None,
    ) -> None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id:
            return
        lock = getattr(self, "_interactive_update_senders_lock", None)
        senders = getattr(self, "_interactive_update_senders", None)
        sent_keys = getattr(self, "_interactive_update_sent_keys", None)
        if lock is None or senders is None:
            return
        async with lock:
            current = senders.get(safe_thread_id)
            if sender is None or current is sender:
                senders.pop(safe_thread_id, None)
                if isinstance(sent_keys, dict):
                    sent_keys.pop(safe_thread_id, None)

    async def emit_interactive_update(
        self,
        *,
        text: str,
        dedupe_key: str = "",
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_thread_id = str(thread_id or "").strip() or self.get_active_thread_id()
        if not resolved_thread_id:
            return {"ok": False, "sent": False, "reason": "missing_thread_id"}
        safe_text = str(text or "").strip()
        if not safe_text:
            return {"ok": False, "sent": False, "reason": "empty_message"}
        key = str(dedupe_key or "").strip()
        if not key:
            key = hashlib.sha256(safe_text.encode("utf-8")).hexdigest()

        lock = getattr(self, "_interactive_update_senders_lock", None)
        senders = getattr(self, "_interactive_update_senders", None)
        sent_keys_by_thread = getattr(self, "_interactive_update_sent_keys", None)
        if lock is None or senders is None or sent_keys_by_thread is None:
            return {"ok": False, "sent": False, "reason": "interactive_update_unavailable"}

        async with lock:
            sender = senders.get(resolved_thread_id)
            if sender is None:
                return {"ok": False, "sent": False, "reason": "interactive_update_unavailable"}
            sent_keys = sent_keys_by_thread.setdefault(resolved_thread_id, set())
            if key in sent_keys:
                return {"ok": True, "sent": False, "duplicate": True}
            sent_keys.add(key)

        try:
            result = sender(safe_text)
            if inspect.isawaitable(result):
                result = await result
            sent = bool(result.get("sent", True)) if isinstance(result, dict) else bool(result)
        except Exception as exc:
            async with lock:
                sent_keys_by_thread.setdefault(resolved_thread_id, set()).discard(key)
            self.log_behavior_event(
                event="interactive_owner_update_failed",
                thread_id=resolved_thread_id,
                customer_id=self.get_active_customer_id(),
                error=type(exc).__name__,
            )
            return {"ok": False, "sent": False, "error": str(exc)}

        if not sent:
            async with lock:
                sent_keys_by_thread.setdefault(resolved_thread_id, set()).discard(key)
            return {"ok": False, "sent": False, "reason": "send_failed"}

        self.log_behavior_event(
            event="interactive_owner_update_sent",
            thread_id=resolved_thread_id,
            customer_id=self.get_active_customer_id(),
            chars=len(safe_text),
        )
        return {"ok": True, "sent": True}

    async def register_interactive_file_sender(self, *, thread_id: str, sender: Any) -> None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id or sender is None:
            return
        if getattr(self, "_interactive_file_senders_lock", None) is None:
            self._interactive_file_senders_lock = asyncio.Lock()
        if getattr(self, "_interactive_file_senders", None) is None:
            self._interactive_file_senders = {}
        async with self._interactive_file_senders_lock:
            self._interactive_file_senders[safe_thread_id] = sender

    async def clear_interactive_file_sender(
        self,
        *,
        thread_id: str,
        sender: Any | None = None,
    ) -> None:
        safe_thread_id = str(thread_id or "").strip()
        if not safe_thread_id:
            return
        lock = getattr(self, "_interactive_file_senders_lock", None)
        senders = getattr(self, "_interactive_file_senders", None)
        if lock is None or senders is None:
            return
        async with lock:
            current = senders.get(safe_thread_id)
            if sender is None or current is sender:
                senders.pop(safe_thread_id, None)

    async def emit_interactive_file(
        self,
        *,
        file: dict[str, Any],
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        resolved_thread_id = str(thread_id or "").strip() or self.get_active_thread_id()
        if not resolved_thread_id:
            return {"ok": False, "sent": False, "reason": "missing_thread_id"}
        if not isinstance(file, dict) or not file:
            return {"ok": False, "sent": False, "reason": "missing_file"}

        lock = getattr(self, "_interactive_file_senders_lock", None)
        senders = getattr(self, "_interactive_file_senders", None)
        if lock is None or senders is None:
            return {"ok": False, "sent": False, "reason": "interactive_file_unavailable"}
        async with lock:
            sender = senders.get(resolved_thread_id)
        if sender is None:
            return {"ok": False, "sent": False, "reason": "interactive_file_unavailable"}

        try:
            result = sender(file)
            if inspect.isawaitable(result):
                result = await result
            sent = bool(result.get("sent", True)) if isinstance(result, dict) else bool(result)
        except Exception as exc:
            self.log_behavior_event(
                event="interactive_file_send_failed",
                thread_id=resolved_thread_id,
                customer_id=self.get_active_customer_id(),
                error=type(exc).__name__,
            )
            return {"ok": False, "sent": False, "error": str(exc)}
        if not sent:
            return {"ok": False, "sent": False, "reason": "send_failed"}
        self.log_behavior_event(
            event="interactive_file_sent",
            thread_id=resolved_thread_id,
            customer_id=self.get_active_customer_id(),
            file_id=str(file.get("id") or ""),
        )
        return {"ok": True, "sent": True}

    async def drain_interactive_fragments(self, *, thread_id: str) -> list[str]:
        fragments: list[str] = []
        session = await self._get_registered_interactive_session(thread_id=thread_id)
        if session is not None and hasattr(session, "drain_graph_fragments"):
            try:
                drained = await session.drain_graph_fragments()
                fragments.extend(str(item).strip() for item in drained if str(item).strip())
            except Exception:
                logger.exception(
                    "Failed to drain interactive fragments for thread_id=%s",
                    thread_id,
                )
        coordinator = getattr(self, "_thread_inputs", None)
        drain_steering_inputs = getattr(coordinator, "drain_steering_inputs", None)
        if callable(drain_steering_inputs):
            try:
                drained = await drain_steering_inputs(thread_id=thread_id)
                fragments.extend(str(item).strip() for item in drained if str(item).strip())
            except Exception:
                logger.exception(
                    "Failed to drain queued steering inputs for thread_id=%s",
                    thread_id,
                )
        return fragments

    async def classify_workflow_setup_interruption(
        self,
        *,
        user_text: str,
        status: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify a message sent while workflow setup is already running."""

        decision, invoke_error = await self._invoke_structured_model(
            model=self._workflow_setup_input_classifier_model,
            schema=_WorkflowSetupInterruptionDecision,
            messages=[
                SystemMessage(
                    content=(
                        "Classify one user message sent while workflow setup is already running.\n"
                        "Return strict JSON only with keys: ok, kind, confidence, status_reply, reason.\n"
                        "kind must be exactly one of: status_nudge, setup_input.\n"
                        "status_nudge means the user only asks for progress/status, expresses impatience, "
                        "or sends a low-information acknowledgement with no new workflow facts.\n"
                        "setup_input means the message contains any workflow requirement, edit, required "
                        "field, business rule, sink/table/file clarification, confirmation, or other fact "
                        "the setup agent should process.\n"
                        "When kind=status_nudge, write a concise status_reply using only the provided "
                        "status. Do not claim the workflow is complete.\n"
                        "When unsure, use setup_input."
                    )
                ),
                HumanMessage(
                    content=(
                        f"status={json.dumps(status, ensure_ascii=False)[:2000]}\n"
                        f"user_text={str(user_text or '').strip()[:2000]}"
                    )
                ),
            ],
            model_name=self._workflow_setup_input_classifier_model_name,
            call_context={"classifier": "workflow_setup_interruption"},
        )
        if decision is None or not isinstance(decision, _WorkflowSetupInterruptionDecision):
            return {"ok": False, "kind": "setup_input", "error": invoke_error or "invalid_output"}
        kind = str(decision.kind or "").strip().lower()
        if kind not in {"status_nudge", "setup_input"}:
            kind = "setup_input"
        return {
            "ok": bool(decision.ok),
            "kind": kind,
            "confidence": max(0.0, min(float(decision.confidence), 1.0)),
            "status_reply": str(decision.status_reply or "").strip()[:500],
            "reason": str(decision.reason or "").strip()[:300],
        }

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
        conversation_id = (
            str(
                (conversation_summary or {}).get("conversation_id", "")
                if isinstance(conversation_summary, dict)
                else ""
            ).strip()
            or "conversation"
        )
        latest_inbound_id = (
            str(
                (conversation_summary or {}).get("latest_inbound_message_id", "")
                if isinstance(conversation_summary, dict)
                else ""
            ).strip()
            or "latest"
        )
        structured_thread_id = f"intake_decision_{workflow_id}_{conversation_id}"
        structured_trace_id = f"intake_{workflow_id}_{conversation_id}_{latest_inbound_id}"
        tool_enabled_runtime = (
            getattr(self, "_graph", None) is not None
            and getattr(self, "_wake_execution_model_with_tools", None) is not None
            and callable(getattr(self, "ainvoke_text", None))
        )
        sink_type = str(workflow.get("sink_type", "") or "").strip().lower()
        sink_config = (
            workflow.get("sink_config") if isinstance(workflow.get("sink_config"), dict) else {}
        )
        static_arguments = (
            sink_config.get("static_arguments") if isinstance(sink_config, dict) else {}
        )
        sink_needs_tool_metadata = sink_type in {
            "google_sheets_composio",
            "generic_composio_write",
        } and not bool(static_arguments)
        prefer_agent_runtime = tool_enabled_runtime and (
            bool(execution_feedback) or sink_needs_tool_metadata
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
                self.log_behavior_event(
                    event="intake.decision.agent_parse_error",
                    trace_id=structured_trace_id,
                    thread_id=structured_thread_id,
                    customer_id=customer_id,
                    workflow_id=workflow_id,
                    conversation_id=conversation_id,
                    latest_inbound_message_id=latest_inbound_id,
                    error=invoke_error,
                )
        if decision is None:
            with self._observability_trace_context(
                name="opentulpa.intake.turn",
                trace_id=structured_trace_id,
                customer_id=customer_id,
                thread_id=structured_thread_id,
                input={
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                    "incoming_id": latest_inbound_id,
                },
                metadata={
                    "turn_mode": "routine_wake",
                    "prompt_mode": "structured_intake",
                    "workflow_id": workflow_id,
                    "conversation_id": conversation_id,
                    "latest_inbound_message_id": latest_inbound_id,
                    "incoming_id": latest_inbound_id,
                },
                tags=["intake", "routine_wake"],
            ):
                decision, invoke_error = await self._invoke_structured_model(
                    model=model,
                    schema=_IntakeWorkflowDecision,
                    messages=[
                        SystemMessage(
                            content=(
                                _build_intake_workflow_system_prompt()
                                + "\n\n"
                                + _build_intake_workflow_context_prompt(
                                    customer_id=customer_id,
                                    workflow=workflow,
                                )
                            )
                        ),
                        HumanMessage(
                            content=_build_intake_workflow_state_prompt(
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
                        "incoming_id": latest_inbound_id,
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
            "missing_fields": [
                str(item).strip() for item in decision.missing_fields if str(item).strip()
            ],
            "reply_action": str(decision.reply_action).strip().lower() or "none",
            "reply_text": str(decision.reply_text).strip(),
            "ready_to_save": bool(decision.ready_to_save),
            "booking_action": str(decision.booking_action).strip().lower() or "ignore",
            "save_payload": dict(decision.save_payload),
            "sink_action": str(decision.sink_action).strip().lower() or "none",
            "sink_payload": dict(decision.sink_payload),
            "sink_arguments": dict(decision.sink_arguments),
            "needs_business_knowledge": bool(decision.needs_business_knowledge),
            "business_knowledge_query": str(decision.business_knowledge_query).strip()[:500],
            "knowledge_source_refs": _normalize_knowledge_source_refs(
                decision.knowledge_source_refs
            ),
            "grounding_status": str(decision.grounding_status).strip().lower(),
            "reason": str(decision.reason).strip()[:500],
        }

    async def classify_routine_create_intent(
        self,
        *,
        latest_user_text: str,
        prior_assistant_text: str,
        routine_args: dict[str, Any],
        turn_mode: str,
    ) -> dict[str, Any]:
        """Decide whether the current conversation authorizes creating a scheduled routine."""
        safe_args: dict[str, Any] = {}
        for key, value in (routine_args or {}).items():
            key_text = str(key).strip()
            if not key_text:
                continue
            if isinstance(value, str):
                safe_args[key_text] = value[:1800]
            elif isinstance(value, (int, float, bool)) or value is None:
                safe_args[key_text] = value
            elif isinstance(value, list):
                safe_args[key_text] = [str(item)[:160] for item in value[:12]]
            elif isinstance(value, dict):
                safe_args[key_text] = {
                    str(k)[:60]: str(v)[:200] for k, v in list(value.items())[:16]
                }
            else:
                safe_args[key_text] = str(value)[:300]

        decision, invoke_error = await self._invoke_structured_model(
            model=getattr(self, "_wake_classifier_model", None) or self._model,
            schema=_RoutineCreateIntentDecision,
            messages=[
                SystemMessage(
                    content=(
                        "You are a scheduling intent judge for OpenTulpa.\n"
                        "Return strict JSON only with keys: ok (bool), allow_create (bool), "
                        "confidence (0..1), reason (string <= 180 chars).\n"
                        "Task: decide whether the latest user message authorizes the proposed "
                        "routine_create call in this conversation.\n"
                        "Allow when the latest user directly asks for a reminder, schedule, recurring "
                        "job, routine, or automation, or when it is a positive confirmation to the "
                        "prior assistant's explicit question about creating or recreating this routine.\n"
                        "This is multilingual: judge intent semantically, not by exact keywords.\n"
                        "Do not decide safety or external side effects here. Focus only on whether the user authorized creating this routine.\n"
                        "Do not allow if the user asked to only discuss/draft, declined, changed the subject, "
                        "or if the proposed routine materially differs from what the user authorized.\n"
                        "If unsure, set allow_create=false and explain the missing authorization."
                    )
                ),
                HumanMessage(
                    content=(
                        f"turn_mode={str(turn_mode or '').strip()[:80]}\n"
                        f"latest_user_message={str(latest_user_text or '').strip()[:2400]}\n"
                        f"prior_assistant_message={str(prior_assistant_text or '').strip()[:2400]}\n"
                        f"proposed_routine_args={json.dumps(safe_args, ensure_ascii=False)[:6000]}"
                    )
                ),
            ],
            call_context={"classifier": "routine_create_intent"},
        )
        if decision is None or not isinstance(decision, _RoutineCreateIntentDecision):
            detail = invoke_error or "invalid_routine_intent_output"
            return {"ok": False, "allow_create": False, "error": f"classifier_error:{detail}"}
        return {
            "ok": bool(decision.ok),
            "allow_create": bool(decision.allow_create),
            "confidence": max(0.0, min(float(decision.confidence), 1.0)),
            "reason": str(decision.reason).strip()[:180],
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
            ctx.reset(cast("contextvars.Token[str]", raw_token))
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
            ctx.reset(cast("contextvars.Token[str]", raw_token))
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

    def set_active_turn_mode(self, turn_mode: str):
        mode = _normalize_turn_mode(turn_mode)
        ctx = self._ensure_active_turn_mode_ctx()
        previous = str(ctx.get() or "").strip()
        token = ctx.set(mode)
        self._active_turn_mode = mode
        return (token, previous)

    def reset_active_turn_mode(self, token: object) -> None:
        ctx = self._ensure_active_turn_mode_ctx()
        previous = "interactive"
        raw_token = token
        if isinstance(token, tuple) and len(token) == 2:
            raw_token, previous = token
            previous = str(previous or "").strip() or "interactive"
        try:
            ctx.reset(cast("contextvars.Token[str]", raw_token))
        except ValueError:
            ctx.set(previous)
        self._active_turn_mode = _normalize_turn_mode(str(ctx.get() or "interactive"))

    def get_active_turn_mode(self) -> str:
        return _normalize_turn_mode(str(self._ensure_active_turn_mode_ctx().get() or "interactive"))

    def _ensure_active_turn_mode_ctx(self) -> contextvars.ContextVar[str]:
        ctx = getattr(self, "_active_turn_mode_ctx", None)
        if isinstance(ctx, contextvars.ContextVar):
            return ctx
        ctx = contextvars.ContextVar("opentulpa_active_turn_mode", default="interactive")
        self._active_turn_mode_ctx = ctx
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

        Used by API routes and orchestrators without coupling to private runtime attributes.
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
        if (
            inject_customer_id
            and str(action_name or "").strip() in CUSTOMER_ID_REQUIRED_TOOLS
            and not cid
        ):
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
