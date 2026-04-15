"""Graph construction for OpenTulpa runtime."""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
from datetime import datetime, timedelta
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy

from opentulpa.agent.context_engineer import (
    ContextEngineer,
)
from opentulpa.agent.context_engineer import (
    trim_text_to_token_budget as _trim_text_to_token_budget,
)
from opentulpa.agent.lc_messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from opentulpa.agent.models import AgentState
from opentulpa.agent.prompt_policy import (
    build_system_prompt_message as _build_system_prompt_message,
)
from opentulpa.agent.prompt_sections import (
    PROMPT_DYNAMIC_BOUNDARY,
)
from opentulpa.agent.prompt_sections import (
    build_prompt_mode_message as _build_prompt_mode_message,
)
from opentulpa.agent.prompt_sections import (
    build_retrieved_context_message as _build_retrieved_context_message,
)
from opentulpa.agent.tool_message_protocol import (
    enforce_tool_message_protocol as _enforce_tool_message_protocol,
)
from opentulpa.agent.tool_message_protocol import (
    sanitize_history_messages_for_model as _sanitize_history_messages_for_model,
)
from opentulpa.agent.turn_policy import (
    build_turn_mode_system_message as _build_turn_mode_system_message,
)
from opentulpa.agent.turn_policy import (
    execution_origin_for_turn_mode as _execution_origin_for_turn_mode,
)
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
    extract_relative_delay_minutes as _extract_relative_delay_minutes,
)
from opentulpa.agent.utils import (
    is_cron_like_schedule as _is_cron_like_schedule,
)
from opentulpa.agent.utils import (
    latest_user_text as _latest_user_text,
)
from opentulpa.agent.utils import (
    looks_like_shell_command as _looks_like_shell_command,
)
from opentulpa.agent.utils import (
    message_to_text as _message_to_text,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)

logger = logging.getLogger(__name__)


def _compute_claim_check_retry_limit(runtime: Any) -> int:
    """
    Derive retry budget from graph recursion limit so claim-check retries
    can keep driving progress instead of stopping after a tiny fixed count.
    """
    try:
        recursion_limit = int(getattr(runtime, "recursion_limit", 30))
    except Exception:
        recursion_limit = 30
    # Keep a small headroom for non-claim-check hops; still allow many retries.
    return max(3, min(24, recursion_limit - 6))


def _compute_empty_output_retry_limit(runtime: Any) -> int:
    """
    Empty assistant outputs should self-repair quickly and then exit.
    Long retry loops here burn context without producing user-visible progress.
    """
    return min(2, _compute_claim_check_retry_limit(runtime))


def _make_prompt_context_entry(*, section: str, content: str) -> dict[str, str] | None:
    safe_section = str(section or "").strip()
    safe_content = str(content or "").strip()
    if not safe_section or not safe_content:
        return None
    return {"section": safe_section, "content": safe_content}


def _make_retrieved_context_entry(*, section: str, title: str, body: str) -> dict[str, str] | None:
    message = _build_retrieved_context_message(title=title, body=body)
    if message is None:
        return None
    return _make_prompt_context_entry(
        section=section,
        content=_content_to_text(getattr(message, "content", "")).strip(),
    )


def _normalize_prompt_context_entries(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = _make_prompt_context_entry(
            section=str(item.get("section", "")).strip(),
            content=str(item.get("content", "")).strip(),
        )
        if entry is not None:
            normalized.append(entry)
    return normalized


def _normalize_frozen_history_messages(raw: Any) -> list[AnyMessage]:
    if not isinstance(raw, list):
        return []
    normalized: list[AnyMessage] = []
    for item in raw:
        if isinstance(item, (HumanMessage, AIMessage, ToolMessage)):
            normalized.append(item)
    return normalized


def _frozen_prompt_context_matches(
    raw: Any,
    *,
    latest_user: str,
    customer_id: str,
    prompt_mode: str,
    turn_mode: str,
) -> bool:
    if not isinstance(raw, dict):
        return False
    signature = raw.get("signature")
    if not isinstance(signature, dict):
        return False
    return (
        str(signature.get("latest_user", "")).strip() == str(latest_user or "").strip()
        and str(signature.get("customer_id", "")).strip() == str(customer_id or "").strip()
        and str(signature.get("prompt_mode", "")).strip() == str(prompt_mode or "").strip()
        and str(signature.get("turn_mode", "")).strip() == str(turn_mode or "").strip()
    )


def _build_late_turn_control_text(
    *,
    prompt_mode: str,
    turn_mode: str,
    customer_id: str,
    live_time: dict[str, str],
) -> str:
    parts: list[str] = [
        PROMPT_DYNAMIC_BOUNDARY,
        _content_to_text(_build_prompt_mode_message(prompt_mode).content),  # type: ignore[arg-type]
        _content_to_text(_build_turn_mode_system_message(turn_mode).content),
        (
            f"customer_id={customer_id}. "
            "Customer scope for customer-scoped tools is resolved automatically from runtime state."
        ),
        (
            "Live time context (auto-injected this turn):\n"
            f"- server_time_local_iso: {live_time['server_time_local_iso']}\n"
            f"- server_time_utc_iso: {live_time['server_time_utc_iso']}\n"
            f"- server_utc_offset: {live_time['server_utc_offset']}\n"
            f"- user_time_local_iso: {live_time['user_time_local_iso']}\n"
            f"- user_utc_offset: {live_time['user_utc_offset']}\n"
            f"- user_time_source: {live_time['user_time_source']}\n"
            "Use these concrete values for all relative-time reasoning in this turn."
        ),
    ]
    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())


def _prompt_overhead_tokens(messages: list[AnyMessage]) -> int:
    return sum(
        _approx_tokens(_content_to_text(getattr(msg, "content", "")))
        for msg in messages
    )


def _select_optional_prompt_entries(
    entries: list[dict[str, str]],
    *,
    initial_used_tokens: int,
    optional_context_budget: int,
) -> tuple[list[tuple[str, SystemMessage]], int]:
    kept: list[tuple[str, SystemMessage]] = []
    used_tokens = max(0, int(initial_used_tokens))
    for entry in entries:
        content = str(entry.get("content", "")).strip()
        section = str(entry.get("section", "")).strip()
        if not content or not section:
            continue
        msg_tokens = _approx_tokens(content)
        if (used_tokens > 0 or kept) and used_tokens + msg_tokens > optional_context_budget:
            continue
        kept.append((section, SystemMessage(content=content)))
        used_tokens += msg_tokens
    return kept, used_tokens

_WORKING_DIR_PREFIXES: dict[str, str] = {
    "tulpa_stuff": "tulpa_stuff",
    "integrations": "src/opentulpa/integrations",
    "interfaces": "src/opentulpa/interfaces",
    "tools": "src/opentulpa/tools",
    "skills": "src/opentulpa/skills",
    "opentulpa": "src/opentulpa",
}

_CHAT_ONLY_ROUTINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bstay in chat\b"),
    re.compile(r"\b(?:just )?draft (?:it )?(?:with me )?here\b"),
    re.compile(r"\bthink it through with me here\b"),
    re.compile(r"\bwork through it with me here\b"),
    re.compile(r"\bdo not create (?:a )?(?:routine|automation)\b"),
    re.compile(r"\bdon't create (?:a )?(?:routine|automation)\b"),
    re.compile(r"\bdo not automate (?:this|it)(?: yet)?\b"),
    re.compile(r"\bdon't automate (?:this|it)(?: yet)?\b"),
)
_EXPLICIT_ROUTINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bremind me\b"),
    re.compile(r"\bset (?:a )?reminder\b"),
    re.compile(r"\bschedule\b"),
    re.compile(r"\bautomation\b"),
    re.compile(r"\bautomate\b"),
    re.compile(r"\broutine\b"),
    re.compile(r"\brecurr(?:ing|ence)?\b"),
    re.compile(r"\brepeat(?:ing)?\b"),
    re.compile(r"\b(?:daily|weekly|monthly|hourly|nightly)\b"),
    re.compile(r"\bevery (?:morning|afternoon|evening|night|weekday|weekend)\b"),
    re.compile(r"\bevery \d+\s*(?:minute|hour|day|week|month)s?\b"),
    re.compile(r"\beach (?:day|week|month)\b"),
    re.compile(r"\bonce (?:a|per) (?:day|week|month)\b"),
)

def _is_iso_datetime_schedule(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text)
    except Exception:
        return False
    return True


def _has_redundant_working_dir_prefix(command: str, working_dir: str) -> bool:
    prefix = _WORKING_DIR_PREFIXES.get(str(working_dir or "").strip())
    text = str(command or "").strip()
    if not prefix or not text:
        return False
    try:
        parts = shlex.split(text)
    except Exception:
        return False
    if len(parts) <= 1:
        return False
    markers = (f"{prefix}/", f"./{prefix}/")
    for token in parts[1:]:
        raw = str(token)
        candidates = [raw]
        if raw.startswith("--") and "=" in raw:
            _, value = raw.split("=", 1)
            candidates.append(value)
        for candidate in candidates:
            if any(candidate.startswith(marker) for marker in markers):
                return True
    return False


def _has_duplicate_allowed_root_prefix(path: str) -> str | None:
    text = str(path or "").strip()
    if not text:
        return None
    for prefix in _WORKING_DIR_PREFIXES.values():
        normalized = str(prefix or "").strip("/")
        if normalized and text.startswith(f"{normalized}/{normalized}/"):
            return normalized
    return None


def _normalize_user_intent_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def _user_requested_chat_only_for_routine(text: str) -> bool:
    normalized = _normalize_user_intent_text(text)
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _CHAT_ONLY_ROUTINE_PATTERNS)


def _user_explicitly_requested_routine(text: str) -> bool:
    normalized = _normalize_user_intent_text(text)
    if not normalized:
        return False
    if _extract_relative_delay_minutes(text) is not None:
        return True
    return any(pattern.search(normalized) for pattern in _EXPLICIT_ROUTINE_PATTERNS)


def _routine_create_clarification_error(latest_user_text: str, *, turn_mode: str) -> str | None:
    normalized_turn_mode = _normalize_turn_mode(turn_mode)
    if normalized_turn_mode == "routine_wake":
        return None
    if normalized_turn_mode == "event_notification":
        return (
            "TURN_MODE_MISMATCH: this is a background event-notification turn, not a fresh "
            "user scheduling request. Do not call routine_create here unless the event "
            "explicitly instructs schedule management."
        )
    if _user_requested_chat_only_for_routine(latest_user_text):
        return (
            "CHAT_MODE_LOCKED: the latest user message asks to keep this in chat "
            "or avoid creating a routine. Do not call routine_create. "
            "Reply in chat or ask one concise clarifying question instead."
        )
    if not _user_explicitly_requested_routine(latest_user_text):
        return (
            "ACTION_CLARIFICATION_REQUIRED: routine_create is only for explicit reminders, "
            "schedules, or recurring jobs. The latest user message does not clearly request that. "
            "Ask one concise clarifying question or continue in chat instead of creating a routine."
        )
    return None


def _validate_model_tool_call(
    *,
    call_name: str,
    args: Any,
    latest_user_text: str,
    turn_mode: str,
    required_args: dict[str, tuple[str, ...]],
    forbidden_tool_args: dict[str, set[str]],
) -> str | None:
    if not isinstance(args, dict):
        return f"TOOL_VALIDATION_ERROR: arguments for {call_name} must be an object"

    blocked_args = sorted(arg for arg in args if arg in forbidden_tool_args.get(call_name, set()))
    if blocked_args:
        return (
            f"TOOL_VALIDATION_ERROR: {call_name} must not include argument(s): "
            f"{', '.join(blocked_args)}. These are runtime-managed."
        )

    missing = [arg for arg in required_args.get(call_name, ()) if not args.get(arg)]
    if missing:
        if call_name == "routine_create" and "implementation_command" in missing:
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create needs "
                "implementation_command (a concrete shell/script command like "
                "`python3 scripts/digest.py`) describing what will run "
                "on each scheduled execution (the command runs with working_dir=tulpa_stuff "
                "by default, so no tulpa_stuff/ prefix needed). Repair the call and retry."
            )
        return (
            f"TOOL_VALIDATION_ERROR: missing required argument(s) for "
            f"{call_name}: {', '.join(missing)}"
        )

    if call_name == "tulpa_run_terminal":
        command = str(args.get("command", "")).strip()
        if not _looks_like_shell_command(command):
            return (
                "TOOL_VALIDATION_ERROR: command must be a concrete shell command "
                "with executable + args."
            )
        working_dir = str(args.get("working_dir", "tulpa_stuff") or "").strip() or "tulpa_stuff"
        if _has_redundant_working_dir_prefix(command, working_dir):
            return (
                "TOOL_VALIDATION_ERROR: command includes a redundant working-dir path prefix. "
                "When working_dir is set, use paths relative to that directory "
                "(example: use `python3 tg_login.py`, not `python3 tulpa_stuff/tg_login.py`)."
            )

    if call_name in {"tulpa_read_file", "tulpa_write_file", "tulpa_validate_file", "tulpa_file_send"}:
        path_arg = str(args.get("path", "")).strip()
        duplicate_prefix = _has_duplicate_allowed_root_prefix(path_arg)
        if duplicate_prefix:
            return (
                "TOOL_VALIDATION_ERROR: path includes a duplicated allowed-root prefix. "
                f"Use `{duplicate_prefix}/...`, not `{duplicate_prefix}/{duplicate_prefix}/...`."
            )

    if call_name == "routine_create":
        schedule = str(args.get("schedule", "")).strip()
        implementation_command = str(args.get("implementation_command", "")).strip()
        clarification_error = _routine_create_clarification_error(
            latest_user_text,
            turn_mode=turn_mode,
        )
        if clarification_error:
            return clarification_error
        if not (_is_cron_like_schedule(schedule) or _is_iso_datetime_schedule(schedule)):
            return (
                "TOOL_VALIDATION_ERROR: routine_create schedule must be either cron "
                "(five-part expression) or local ISO datetime."
            )
        if not implementation_command:
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create must include "
                "a non-empty implementation_command (shell/script command) so scheduled "
                "runs execute a concrete implementation."
            )
        if not _looks_like_shell_command(implementation_command):
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must "
                "be a concrete shell command (executable + args), not natural language."
            )
        if _has_redundant_working_dir_prefix(implementation_command, "tulpa_stuff"):
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command should be relative "
                "to working_dir=tulpa_stuff (example: `python3 tg_login.py`, "
                "not `python3 tulpa_stuff/tg_login.py`)."
            )
        delay_minutes = _extract_relative_delay_minutes(latest_user_text)
        if delay_minutes is not None and _is_cron_like_schedule(schedule):
            return (
                "TOOL_VALIDATION_ERROR: for one-time relative reminders, "
                "use a local ISO datetime schedule (not cron)."
            )

    return None
def _build_relevant_skill_discovery_context(
    *,
    available_skills: Any,
    selected_names: list[str] | None,
) -> str:
    if not isinstance(available_skills, list) or not selected_names:
        return ""
    wanted = {str(name).strip() for name in selected_names if str(name).strip()}
    if not wanted:
        return ""
    lines = [
        "Skills relevant to this task:",
        "These are discovery hints only. Use skill_get(name) before relying on a skill's actual instructions.",
    ]
    seen: set[str] = set()
    for item in available_skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name not in wanted or name in seen:
            continue
        seen.add(name)
        description = " ".join(str(item.get("description", "")).split()).strip()
        scope = str(item.get("scope", "")).strip() or "user"
        if description:
            lines.append(f"- {name} ({scope}): {description[:220]}")
        else:
            lines.append(f"- {name} ({scope})")
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)


def _extract_invoked_skill_snapshot(result: Any, *, requested_name: str) -> tuple[str, str] | None:
    if not isinstance(result, dict):
        return None
    name = str(result.get("name", "")).strip() or str(requested_name or "").strip()
    if not name:
        return None
    description = str(result.get("description", "")).strip()
    scope = str(result.get("scope", "")).strip() or "user"
    skill_markdown = str(result.get("skill_markdown", "")).strip()
    if not skill_markdown:
        instructions = str(result.get("instructions", "")).strip()
        supporting = result.get("supporting_files")
        if instructions:
            skill_markdown = instructions
        elif isinstance(supporting, dict) and supporting:
            skill_markdown = "\n\n".join(
                f"[{key}]\n{str(value).strip()}"
                for key, value in supporting.items()
                if str(key).strip() and str(value).strip()
            ).strip()
    if not skill_markdown:
        return None
    header = [f"Skill: {name}", f"Scope: {scope}"]
    if description:
        header.append(f"Description: {description}")
    content = "\n".join(header) + f"\n\nSKILL.md:\n{skill_markdown[:3500]}"
    return name, content


def _normalize_approval_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


def _summarize_tool_validation_errors(messages: list[ToolMessage]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for message in messages:
        text = _content_to_text(getattr(message, "content", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " | ".join(parts[:3])


def _build_tool_validation_repair_message(messages: list[ToolMessage]) -> str:
    summary = _summarize_tool_validation_errors(messages)
    if not summary:
        return (
            "VALIDATION_REPAIR_REQUIRED: Your previous tool call was blocked. Do not claim success. "
            "Repair the tool call or clearly state that the action was not completed yet."
        )
    if any(
        marker in summary
        for marker in ("ACTION_CLARIFICATION_REQUIRED", "CHAT_MODE_LOCKED", "TURN_MODE_MISMATCH")
    ):
        return (
            "VALIDATION_REPAIR_REQUIRED: The schedule was not created. Do not say it was scheduled. "
            "Ask one concise clarifying question or continue in chat if automation is not explicit. "
            f"Reason={summary}"
        )
    return (
        "VALIDATION_REPAIR_REQUIRED: The schedule was not created yet. Do not claim success. "
        "Repair the tool call arguments and retry only if you can satisfy the validation error exactly. "
        f"Reason={summary}"
    )


def build_runtime_graph(runtime: Any):
    assert runtime._model_with_tools is not None
    assert runtime._checkpointer is not None

    required_args: dict[str, tuple[str, ...]] = {
        "tulpa_write_file": ("path", "content"),
        "tulpa_validate_file": ("path",),
        "tulpa_reload": (),
        "tulpa_read_file": ("path",),
        "tulpa_run_terminal": ("command",),
        "fetch_url_content": ("url",),
        "fetch_file_content": ("url",),
        "uploaded_file_search": ("query",),
        "uploaded_file_get": ("file_id",),
        "uploaded_file_send": ("file_id",),
        "tulpa_file_send": ("path",),
        "web_image_send": ("url",),
        "uploaded_file_analyze": ("file_id",),
        "skill_get": ("name",),
        "skill_upsert": ("name", "description", "instructions"),
        "skill_delete": ("name",),
        "intake_workflow_upsert": (
            "name",
            "intent_description",
            "required_fields",
            "sink_type",
            "sink_config",
        ),
        "intake_workflow_list": (),
        "intake_workflow_get": ("workflow_id",),
        "intake_workflow_delete": ("workflow_id",),
        "intake_workflow_run": ("workflow_id",),
        "composio_status": (),
        "composio_authorize_toolkit": ("toolkit",),
        "composio_wait_for_connection": ("connection_id",),
        "composio_toolkits": (),
        "composio_connected_accounts": (),
        "composio_disable_connected_account": ("connected_account_id",),
        "composio_delete_connected_account": ("connected_account_id",),
        "composio_tool_search": (),
        "composio_tool_schema": ("tool_slug",),
        "composio_instagram_reply_precheck": (),
        "composio_tool_execute": ("tool_slug",),
        "directive_set": ("directive",),
        "time_profile_set": ("utc_offset",),
        "browser_use_session_list": (),
        "browser_use_run": ("task",),
        "browser_use_task_get": ("task_id",),
        "browser_use_task_screenshot": ("task_id",),
        "browser_use_task_control": ("task_id",),
        "routine_list": (),
        "routine_create": (
            "name",
            "schedule",
            "instruction",
            "implementation_command",
        ),
        "routine_delete": ("routine_id",),
        "guardrail_execute_approved_action": ("approval_id",),
    }
    customer_scoped_tools: set[str] = {
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
        "intake_workflow_upsert",
        "intake_workflow_list",
        "intake_workflow_get",
        "intake_workflow_delete",
        "intake_workflow_run",
        "composio_authorize_toolkit",
        "composio_toolkits",
        "composio_connected_accounts",
        "composio_disable_connected_account",
        "composio_delete_connected_account",
        "composio_tool_search",
        "composio_tool_schema",
        "composio_instagram_reply_precheck",
        "composio_tool_execute",
        "directive_get",
        "directive_set",
        "directive_clear",
        "time_profile_get",
        "time_profile_set",
        "tulpa_run_terminal",
        "routine_list",
        "routine_create",
        "routine_delete",
        "browser_use_run",
        "guardrail_execute_approved_action",
    }
    forbidden_tool_args: dict[str, set[str]] = {name: {"customer_id"} for name in customer_scoped_tools}
    forbidden_tool_args["routine_create"] = {"customer_id", "message"}

    stable_system_message = _build_system_prompt_message()
    context_engineer = getattr(runtime, "_context_engineer", None)
    if not isinstance(context_engineer, ContextEngineer):
        context_engineer = ContextEngineer()

    def _log(state: AgentState | None, event: str, **fields: Any) -> None:
        log_event = getattr(runtime, "log_behavior_event", None)
        if not callable(log_event):
            return
        payload: dict[str, Any] = {}
        if isinstance(state, dict):
            trace_id = str(state.get("agent_trace_id", "")).strip()
            thread_id = str(state.get("thread_id", "")).strip()
            customer_id = str(state.get("customer_id", "")).strip()
            if trace_id:
                payload["trace_id"] = trace_id
            if thread_id:
                payload["thread_id"] = thread_id
            if customer_id:
                payload["customer_id"] = customer_id
        payload.update(fields)
        log_event(event=event, **payload)

    async def agent_node(state: AgentState) -> Command[Literal["validate_tools", "finalize_turn"]]:
        customer_id = state.get("customer_id", "")
        thread_id = state.get("thread_id", "")
        turn_mode = _normalize_turn_mode(state.get("turn_mode"))
        prompt_mode = str(state.get("prompt_mode", "task_chat")).strip().lower() or "task_chat"
        messages = state.get("messages", [])
        injected_messages: list[HumanMessage] = []
        if turn_mode == "interactive":
            drain_fragments = getattr(runtime, "drain_interactive_fragments", None)
            if callable(drain_fragments):
                drained = await drain_fragments(thread_id=thread_id)
                injected_messages = [
                    HumanMessage(content=str(fragment).strip())
                    for fragment in drained
                    if str(fragment).strip()
                ]
                if injected_messages:
                    messages = [*messages, *injected_messages]
        latest_user = _latest_user_text(messages)
        _log(
            state,
            "graph.agent.start",
            message_count=len(messages),
            latest_user_chars=len(latest_user),
            turn_mode=turn_mode,
            injected_user_messages=len(injected_messages),
        )
        cached_query = str(state.get("active_skill_query", "")).strip()
        cached_names = state.get("active_skill_names", []) or []
        cached_available = state.get("active_available_skills", []) or []
        cached_discovery = str(state.get("active_skill_discovery_context", "")).strip()
        cached_invoked_names = state.get("active_invoked_skill_names", []) or []
        cached_invoked_context = str(state.get("active_invoked_skill_context", "")).strip()
        legacy_cached_context = str(state.get("active_skill_context", "")).strip()
        skill_query = cached_query
        skill_names = cached_names if isinstance(cached_names, list) else []
        skill_discovery_context = cached_discovery
        invoked_skill_names = (
            [str(n).strip() for n in cached_invoked_names if str(n).strip()]
            if isinstance(cached_invoked_names, list)
            else []
        )
        invoked_skill_context = cached_invoked_context or legacy_cached_context
        available_skills = cached_available if isinstance(cached_available, list) else []
        prompt_budget = max(4000, int(getattr(runtime, "_context_token_limit", 12000)))
        low_budget = max(1500, int(getattr(runtime, "_context_short_term_low_tokens", 3500)))
        optional_context_budget = max(1000, min(3600, int(low_budget * 0.7)))
        frozen_prompt_context_raw = state.get("frozen_prompt_context")
        prompt_context_update: dict[str, Any] = {}
        if _frozen_prompt_context_matches(
            frozen_prompt_context_raw,
            latest_user=latest_user,
            customer_id=customer_id,
            prompt_mode=prompt_mode,
            turn_mode=turn_mode,
        ):
            frozen_prompt_context = dict(frozen_prompt_context_raw or {})
        else:
            rollup_sections = (
                runtime._load_thread_rollup_sections(thread_id)
                if context_engineer.should_include_optional_context(
                    kind="thread_rollup",
                    prompt_mode=prompt_mode,
                    should_retrieve=True,
                )
                else {}
            )
            should_retrieve = runtime._has_retrieval_evidence(
                user_text=latest_user,
                prompt_mode=prompt_mode,
                skill_candidates=available_skills,
                thread_rollup_sections=rollup_sections,
            )
            if should_retrieve and latest_user and latest_user != cached_query:
                if not available_skills:
                    list_skills = getattr(runtime, "_list_available_skills", None)
                    if callable(list_skills):
                        try:
                            available_skills = await list_skills(customer_id)
                        except Exception:
                            available_skills = []
                selected = await runtime._select_relevant_skills(
                    customer_id=customer_id,
                    query=latest_user,
                    candidates=available_skills,
                    prompt_mode=prompt_mode,
                    max_skills=3,
                )
                skill_names = [
                    str(item.get("name", "")).strip()
                    for item in selected
                    if isinstance(item, dict) and str(item.get("name", "")).strip()
                ]
                skill_query = latest_user
            skill_discovery_context = _build_relevant_skill_discovery_context(
                available_skills=available_skills,
                selected_names=skill_names,
            )
            active_directive = (
                await runtime._load_active_directive(customer_id)
                if context_engineer.should_include_optional_context(
                    kind="task_directive",
                    prompt_mode=prompt_mode,
                    should_retrieve=should_retrieve,
                )
                else None
            )
            memory_grounding = await runtime._load_memory_grounding_context(
                customer_id=customer_id,
                user_text=latest_user,
                turn_mode=turn_mode,
                token_budget=500,
            )
            thread_rollup = (
                "\n\n".join(
                    part
                    for part in (
                        str(rollup_sections.get("open_loops") or "").strip(),
                        str(rollup_sections.get("durable_facts") or "").strip(),
                    )
                    if part
                ).strip()
                if context_engineer.should_include_optional_context(
                    kind="thread_rollup",
                    prompt_mode=prompt_mode,
                    should_retrieve=should_retrieve,
                )
                else None
            )
            live_time = await runtime._build_live_time_context(customer_id)
            pending_context_summary = (
                str(state.get("pending_context_summary", "")).strip()
                if context_engineer.should_include_optional_context(
                    kind="pending_context",
                    prompt_mode=prompt_mode,
                    should_retrieve=should_retrieve,
                )
                else ""
            )
            link_alias_context = (
                runtime._build_link_alias_context(
                    customer_id=customer_id,
                    user_text=latest_user,
                )
                if context_engineer.should_include_optional_context(
                    kind="link_aliases",
                    prompt_mode=prompt_mode,
                    should_retrieve=should_retrieve,
                )
                else ""
            )

            stable_entries: list[dict[str, str]] = []
            late_entries: list[dict[str, str]] = []
            if active_directive:
                directive_text = _trim_text_to_token_budget(
                    active_directive,
                    token_budget=max(120, min(420, int(low_budget * 0.12))),
                )
                directive_entry = _make_retrieved_context_entry(
                    section="task_directive",
                    title="Active persistent task/profile directive.",
                    body=(
                        "Treat this as relevant task context, not conversational topic guidance.\n"
                        f"{directive_text}"
                    ),
                )
                if directive_entry is not None:
                    stable_entries.append(directive_entry)
            if thread_rollup:
                rollup_text = _trim_text_to_token_budget(
                    thread_rollup,
                    token_budget=max(300, min(1400, int(low_budget * 0.4))),
                )
                rollup_entry = _make_retrieved_context_entry(
                    section="thread_rollup",
                    title="Compressed older thread context.",
                    body=rollup_text,
                )
                if rollup_entry is not None:
                    late_entries.append(rollup_entry)
            if pending_context_summary:
                pending_text = _trim_text_to_token_budget(
                    pending_context_summary,
                    token_budget=max(140, min(520, int(low_budget * 0.15))),
                )
                pending_entry = _make_retrieved_context_entry(
                    section="pending_context",
                    title="Background system events summary (not user-authored).",
                    body=(
                        "Use this only to reconcile hidden state and never quote event lines directly.\n"
                        f"{pending_text}"
                    ),
                )
                if pending_entry is not None:
                    late_entries.append(pending_entry)
            if skill_discovery_context and context_engineer.should_include_optional_context(
                kind="skill_discovery",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            ):
                discovery_text = _trim_text_to_token_budget(
                    skill_discovery_context,
                    token_budget=max(160, min(620, int(low_budget * 0.18))),
                )
                discovery_entry = _make_retrieved_context_entry(
                    section="skill_discovery",
                    title="Relevant skill discovery for this turn.",
                    body=discovery_text,
                )
                if discovery_entry is not None:
                    late_entries.append(discovery_entry)
            if invoked_skill_context and context_engineer.should_include_optional_context(
                kind="invoked_skills",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            ):
                invoked_text = _trim_text_to_token_budget(
                    invoked_skill_context,
                    token_budget=max(400, min(1800, int(low_budget * 0.45))),
                )
                invoked_entry = _make_retrieved_context_entry(
                    section="invoked_skills",
                    title=(
                        "Previously invoked skill instructions still relevant in this session "
                        f"(skills: {', '.join(invoked_skill_names) if invoked_skill_names else 'unknown'})."
                    ),
                    body=invoked_text,
                )
                if invoked_entry is not None:
                    late_entries.append(invoked_entry)
            if link_alias_context and context_engineer.should_include_optional_context(
                kind="link_aliases",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            ):
                aliases_text = _trim_text_to_token_budget(
                    link_alias_context,
                    token_budget=max(120, min(320, int(low_budget * 0.08))),
                )
                aliases_entry = _make_prompt_context_entry(
                    section="link_aliases",
                    content=aliases_text,
                )
                if aliases_entry is not None:
                    late_entries.append(aliases_entry)
            if memory_grounding:
                grounding_text = _trim_text_to_token_budget(
                    memory_grounding,
                    token_budget=500,
                )
                grounding_entry = _make_retrieved_context_entry(
                    section="memory_grounding",
                    title="Relevant long-term memory grounding (dynamic retrieval).",
                    body=(
                        "Use this to ground historical facts, preferences, directives, projects, technical details, and recalled files. "
                        "Treat it as retrieved memory, not as a user-authored message in this turn.\n"
                        f"{grounding_text}"
                    ),
                )
                if grounding_entry is not None:
                    late_entries.append(grounding_entry)

            frozen_prompt_context = {
                "signature": {
                    "latest_user": latest_user,
                    "customer_id": customer_id,
                    "prompt_mode": prompt_mode,
                    "turn_mode": turn_mode,
                },
                "late_control_content": _build_late_turn_control_text(
                    prompt_mode=prompt_mode,
                    turn_mode=turn_mode,
                    customer_id=customer_id,
                    live_time=live_time,
                ),
                "late_control_sections": [
                    "volatile_injected",
                    f"prompt_mode:{prompt_mode}",
                    f"turn_mode:{turn_mode}",
                    "customer_scope",
                    "live_time",
                ],
                "stable_entries": stable_entries,
                "late_entries": late_entries,
            }
            prompt_context_update["frozen_prompt_context"] = frozen_prompt_context

        stable_prompt_messages: list[AnyMessage] = [stable_system_message]
        stable_prompt_sections = ["stable_core_policy"]
        stable_entries = _normalize_prompt_context_entries(frozen_prompt_context.get("stable_entries"))
        late_entries = _normalize_prompt_context_entries(frozen_prompt_context.get("late_entries"))
        late_control_content = str(frozen_prompt_context.get("late_control_content", "")).strip()
        late_control_sections = [
            str(section).strip()
            for section in (frozen_prompt_context.get("late_control_sections") or [])
            if str(section).strip()
        ]

        kept_stable_optional_entries, used_optional_tokens = _select_optional_prompt_entries(
            stable_entries,
            initial_used_tokens=0,
            optional_context_budget=optional_context_budget,
        )
        prefix_messages: list[AnyMessage] = [
            *stable_prompt_messages,
            *(message for _, message in kept_stable_optional_entries),
        ]
        prefix_sections = [
            *stable_prompt_sections,
            *(section for section, _ in kept_stable_optional_entries),
        ]

        late_control_message = SystemMessage(content=late_control_content) if late_control_content else None
        selected_frozen_late_entries, used_optional_tokens = _select_optional_prompt_entries(
            late_entries,
            initial_used_tokens=used_optional_tokens,
            optional_context_budget=optional_context_budget,
        )
        prompt_messages_base: list[AnyMessage] = [
            *prefix_messages,
            *([late_control_message] if late_control_message is not None else []),
        ]
        max_overhead_tokens = max(1400, int(prompt_budget * 0.72))
        prompt_messages: list[AnyMessage] = [
            *prompt_messages_base,
            *(message for _, message in selected_frozen_late_entries),
        ]
        prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
        while selected_frozen_late_entries and prompt_overhead_tokens > max_overhead_tokens:
            selected_frozen_late_entries.pop()
            prompt_messages = [
                *prompt_messages_base,
                *(message for _, message in selected_frozen_late_entries),
            ]
            prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
        history_budget = max(800, prompt_budget - prompt_overhead_tokens)
        sanitized_history = _sanitize_history_messages_for_model(messages)
        sanitized_history = _enforce_tool_message_protocol(sanitized_history)
        frozen_history_projection_raw = state.get("frozen_history_projection")
        turn_history_messages = _enforce_tool_message_protocol(_latest_turn_messages(sanitized_history))
        turn_start_index = max(0, len(sanitized_history) - len(turn_history_messages))
        older_history_messages: list[AnyMessage] = []
        stale_summary_text = ""
        history_working_set = context_engineer.build_history_working_set(
            sanitized_history,
            token_budget=history_budget,
        )
        if (
            isinstance(frozen_history_projection_raw, dict)
            and int(frozen_history_projection_raw.get("turn_start_index", -1)) >= 0
            and int(frozen_history_projection_raw.get("turn_start_index", -1)) <= len(sanitized_history)
        ):
            turn_start_index = int(frozen_history_projection_raw.get("turn_start_index", 0))
            older_history_messages = _normalize_frozen_history_messages(
                frozen_history_projection_raw.get("older_history_messages")
            )
            stale_summary_text = str(frozen_history_projection_raw.get("stale_summary_text", "")).strip()
        else:
            initial_turn_messages = _enforce_tool_message_protocol(_latest_turn_messages(sanitized_history))
            turn_start_index = max(0, len(sanitized_history) - len(initial_turn_messages))
            summary_entry = None
            if history_working_set.summary_text:
                summary_entry = _make_retrieved_context_entry(
                    section="stale_history_summary",
                    title="Compressed older in-thread context.",
                    body=history_working_set.summary_text,
                )
            selected_summary_entries: list[tuple[str, SystemMessage]] = []
            if summary_entry is not None:
                selected_summary_entries, _ = _select_optional_prompt_entries(
                    [summary_entry],
                    initial_used_tokens=used_optional_tokens,
                    optional_context_budget=optional_context_budget,
                )
            prompt_messages = [
                *prompt_messages_base,
                *(message for _, message in selected_frozen_late_entries),
                *(message for _, message in selected_summary_entries),
            ]
            prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
            while selected_summary_entries and prompt_overhead_tokens > max_overhead_tokens:
                selected_summary_entries.pop()
                prompt_messages = [
                    *prompt_messages_base,
                    *(message for _, message in selected_frozen_late_entries),
                    *(message for _, message in selected_summary_entries),
                ]
                prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
            if selected_summary_entries:
                history_budget = max(800, prompt_budget - prompt_overhead_tokens)
                history_working_set = context_engineer.build_history_working_set(
                    sanitized_history,
                    token_budget=history_budget,
                )
            bounded_messages = _enforce_tool_message_protocol(history_working_set.raw_messages)
            bounded_latest_turn = _latest_turn_messages(bounded_messages)
            bounded_latest_turn_count = len(bounded_latest_turn)
            if 0 < bounded_latest_turn_count < len(bounded_messages):
                older_history_messages = bounded_messages[:-bounded_latest_turn_count]
            else:
                older_history_messages = []
            stale_summary_text = history_working_set.summary_text
            prompt_context_update["frozen_history_projection"] = {
                "turn_start_index": turn_start_index,
                "older_history_messages": older_history_messages,
                "stale_summary_text": stale_summary_text,
            }
        latest_turn_messages = _enforce_tool_message_protocol(sanitized_history[turn_start_index:])
        summary_entry = (
            _make_retrieved_context_entry(
                section="stale_history_summary",
                title="Compressed older in-thread context.",
                body=stale_summary_text,
            )
            if stale_summary_text
            else None
        )
        selected_summary_entries: list[tuple[str, SystemMessage]] = []
        if summary_entry is not None:
            selected_summary_entries, _ = _select_optional_prompt_entries(
                [summary_entry],
                initial_used_tokens=used_optional_tokens,
                optional_context_budget=optional_context_budget,
            )
        prompt_messages = [
            *prompt_messages_base,
            *(message for _, message in selected_frozen_late_entries),
            *(message for _, message in selected_summary_entries),
        ]
        prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
        frozen_late_messages: list[AnyMessage] = [
            *([late_control_message] if late_control_message is not None else []),
            *(message for _, message in selected_frozen_late_entries),
            *(message for _, message in selected_summary_entries),
        ]
        dynamic_late_messages: list[AnyMessage] = []
        prompt_section_names = [
            *prefix_sections,
            *late_control_sections,
            *(section for section, _ in selected_frozen_late_entries),
            *(section for section, _ in selected_summary_entries),
        ]
        optional_context_messages = (
            len(kept_stable_optional_entries)
            + len(selected_frozen_late_entries)
            + len(selected_summary_entries)
        )
        cache_profile: dict[str, Any] = {}
        cache_profile_fn = getattr(runtime, "prompt_cache_profile", None)
        if callable(cache_profile_fn):
            try:
                cache_profile = cache_profile_fn()
            except Exception:
                cache_profile = {}
        stable_prefix_count = len(prefix_messages) + len(older_history_messages) + len(frozen_late_messages)
        actual_history_messages = [*older_history_messages, *latest_turn_messages]
        raw_chat_history_count = sum(
            1 for msg in actual_history_messages if isinstance(msg, (HumanMessage, AIMessage))
        )
        raw_tool_history_count = sum(1 for msg in actual_history_messages if isinstance(msg, ToolMessage))
        protected_history_count = len(context_engineer._protected_suffix_indices(actual_history_messages))
        model_messages: list[AnyMessage] = [
            *prefix_messages,
            *older_history_messages,
            *frozen_late_messages,
            *dynamic_late_messages,
            *latest_turn_messages,
        ]
        _log(
            state,
            "graph.agent.prompt_ready",
            prompt_message_count=len(model_messages),
            prompt_overhead_tokens=prompt_overhead_tokens,
            history_budget=history_budget,
            history_message_count=len(actual_history_messages),
            raw_chat_history_count=raw_chat_history_count,
            raw_tool_history_count=raw_tool_history_count,
            protected_history_count=protected_history_count,
            optional_context_messages=optional_context_messages,
            prompt_sections=",".join(prompt_section_names),
            prompt_cache_strategy=str(cache_profile.get("strategy", "")),
            prompt_cache_enabled=bool(cache_profile.get("enabled", False)),
            prompt_cache_breakpoints=bool(cache_profile.get("supports_breakpoints", False)),
            prompt_cache_top_level=bool(cache_profile.get("supports_top_level", False)),
            stable_prefix_count=stable_prefix_count,
            turn_mode=turn_mode,
        )
        model_with_tools = runtime.model_with_tools_for_turn_mode(turn_mode)
        assert model_with_tools is not None
        ainvoke_fn = getattr(runtime, "ainvoke_model", None)
        if callable(ainvoke_fn):
            call_context = {
                "call_site": "graph_agent",
                "trace_id": state.get("agent_trace_id"),
                "thread_id": thread_id,
                "customer_id": customer_id,
                "turn_mode": turn_mode,
                "prompt_mode": prompt_mode,
                "prompt_sections": prompt_section_names,
                "stable_prefix_count": stable_prefix_count,
                "prompt_overhead_tokens": prompt_overhead_tokens,
                "history_message_count": len(actual_history_messages),
                "raw_chat_history_count": raw_chat_history_count,
                "raw_tool_history_count": raw_tool_history_count,
                "protected_history_count": protected_history_count,
                "optional_context_messages": optional_context_messages,
            }
            response = await ainvoke_fn(
                model_with_tools,
                model_messages,
                stable_prefix_count=stable_prefix_count,
                call_context=call_context,
            )
        else:
            response = await model_with_tools.ainvoke(model_messages)
        response_text = _content_to_text(getattr(response, "content", ""))
        usage_fields: dict[str, Any] = {}
        usage_fields_fn = getattr(runtime, "extract_response_usage_fields", None)
        if callable(usage_fields_fn):
            try:
                usage_fields = dict(usage_fields_fn(response))
            except Exception:
                usage_fields = {}
        _log(
            state,
            "graph.agent.response",
            response_chars=len(response_text.strip()),
            tool_call_count=len(getattr(response, "tool_calls", []) or []),
            turn_mode=turn_mode,
            **usage_fields,
        )
        update: dict[str, Any] = {
            "messages": [*injected_messages, response],
            "turn_status": "running",
            **prompt_context_update,
        }
        if skill_query:
            update["active_skill_query"] = skill_query
            update["active_skill_names"] = skill_names
            update["active_available_skills"] = available_skills
            update["active_skill_discovery_context"] = skill_discovery_context
            update["active_invoked_skill_context"] = invoked_skill_context
            update["active_invoked_skill_names"] = invoked_skill_names
            update["active_skill_context"] = invoked_skill_context
        goto: Literal["validate_tools", "finalize_turn"] = (
            "validate_tools"
            if isinstance(response, AIMessage) and bool(getattr(response, "tool_calls", []))
            else "finalize_turn"
        )
        return Command(update=update, goto=goto)

    async def validate_tool_calls_node(state: AgentState) -> Command[Literal["tools", "agent"]]:
        messages = state.get("messages", [])
        if not messages:
            return Command(update={"tool_validation_passed": True}, goto="tools")
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return Command(update={"tool_validation_passed": True}, goto="tools")
        _log(
            state,
            "graph.validate_tools.start",
            tool_call_count=len(last.tool_calls),
            turn_mode=_normalize_turn_mode(state.get("turn_mode")),
        )

        validation_errors: list[ToolMessage] = []
        latest_user = _latest_user_text(messages)
        turn_mode = _normalize_turn_mode(state.get("turn_mode"))
        for call in last.tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            args = call.get("args", {}) or {}
            validation_error = _validate_model_tool_call(
                call_name=call_name,
                args=args,
                latest_user_text=latest_user,
                turn_mode=turn_mode,
                required_args=required_args,
                forbidden_tool_args=forbidden_tool_args,
            )
            if validation_error:
                validation_errors.append(ToolMessage(content=validation_error, tool_call_id=call_id))
                continue
        if validation_errors:
            error_summary = _summarize_tool_validation_errors(validation_errors)
            repair_message = _build_tool_validation_repair_message(validation_errors)
            _log(
                state,
                "graph.validate_tools.failed",
                error_count=len(validation_errors),
                error_summary=error_summary,
                repair_message=repair_message,
                turn_mode=turn_mode,
            )
            logger.warning(
                "graph.validate_tools.failed thread_id=%s customer_id=%s errors=%s",
                str(state.get("thread_id", "")).strip(),
                str(state.get("customer_id", "")).strip(),
                error_summary or len(validation_errors),
            )
            return Command(
                update={
                    "messages": [
                        *validation_errors,
                        SystemMessage(content=repair_message),
                    ],
                    "tool_validation_passed": False,
                    "tool_error_count": int(state.get("tool_error_count", 0)) + 1,
                    "last_tool_error": error_summary or "tool validation failed",
                    "turn_status": "running",
                },
                goto="agent",
            )
        _log(
            state,
            "graph.validate_tools.passed",
            tool_call_count=len(last.tool_calls),
            turn_mode=turn_mode,
        )
        return Command(update={"tool_validation_passed": True}, goto="tools")

    async def tools_node(state: AgentState) -> Command[Literal["agent", "__end__"]]:
        messages = state.get("messages", [])
        if not messages:
            return Command(update={"turn_status": "running"}, goto="agent")
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return Command(update={"turn_status": "running"}, goto="agent")

        customer_id = state.get("customer_id", "")
        thread_id = str(state.get("thread_id", "")).strip()
        turn_mode = _normalize_turn_mode(state.get("turn_mode"))
        execution_origin = _execution_origin_for_turn_mode(turn_mode, thread_id=thread_id)
        _log(
            state,
            "graph.tools.start",
            requested_tool_calls=len(last.tool_calls),
            execution_origin=execution_origin,
            turn_mode=turn_mode,
        )

        latest_user_for_guard = _latest_user_text(messages)
        prior_assistant_for_guard = ""
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                candidate = _content_to_text(getattr(msg, "content", "")).strip()
                if candidate:
                    prior_assistant_for_guard = candidate
                    break

        tool_messages: list[ToolMessage] = []
        tool_outcomes: list[dict[str, Any]] = []
        approval_handoff = False
        had_error = False
        failed_tool_names: list[str] = []
        failed_tool_errors: list[str] = []
        invoked_skill_names = state.get("active_invoked_skill_names", []) or []
        invoked_skill_list = (
            [str(n).strip() for n in invoked_skill_names if str(n).strip()]
            if isinstance(invoked_skill_names, list)
            else []
        )
        invoked_skill_context = str(state.get("active_invoked_skill_context", "")).strip() or str(
            state.get("active_skill_context", "")
        ).strip()
        for call in last.tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            args = call.get("args", {}) or {}
            try:
                tool_fn = runtime._tools.get(call_name)
                if tool_fn is None:
                    raise ValueError(f"Unknown tool: {call_name}")
                if call_name in customer_scoped_tools and not str(customer_id or "").strip():
                    raise ValueError(f"{call_name} requires customer scope, but customer_id is missing")
                if call_name in {"tulpa_run_terminal", "routine_create"}:
                    args = {
                        **args,
                        "thread_id": thread_id,
                        "execution_origin": execution_origin,
                        "guard_context": {
                            "previous_user_message": latest_user_for_guard[:2000],
                            "previous_assistant_message": prior_assistant_for_guard[:2000],
                        },
                    }
                if call_name == "routine_create":
                    latest_user = _latest_user_text(messages)
                    corrected_args = dict(args)
                    delay_minutes = _extract_relative_delay_minutes(latest_user)
                    if delay_minutes is not None and _is_cron_like_schedule(
                        str(corrected_args.get("schedule", ""))
                    ):
                        run_at_local = datetime.now().astimezone() + timedelta(
                            minutes=max(1, delay_minutes)
                        )
                        corrected_args["schedule"] = run_at_local.isoformat()
                    args = corrected_args
                args = runtime.resolve_link_aliases_in_args(customer_id=customer_id, args=args)
                scope_token = None
                set_customer_scope = getattr(runtime, "set_active_customer_id", None)
                if callable(set_customer_scope):
                    scope_token = set_customer_scope(customer_id)
                try:
                    result = await tool_fn.ainvoke(args)
                finally:
                    reset_customer_scope = getattr(runtime, "reset_active_customer_id", None)
                    if scope_token is not None and callable(reset_customer_scope):
                        reset_customer_scope(scope_token)
                runtime.register_links_from_text(
                    customer_id=customer_id,
                    text=_safe_json(result),
                    source=f"tool:{call_name}",
                    limit=40,
                )
                result_text = _safe_json(result)
                model_visible_result_text = result_text
                _log(
                    state,
                    "graph.tools.success",
                    tool_name=call_name,
                    tool_call_id=call_id,
                    result_chars=len(result_text),
                    model_visible_result_chars=len(model_visible_result_text),
                    tool_result_compressed=model_visible_result_text != result_text,
                )
                if (
                    isinstance(result, dict)
                    and str(result.get("status", "")).strip().lower() == "approval_pending"
                    and _normalize_approval_id(result.get("approval_id"))
                ):
                    approval_id = _normalize_approval_id(result.get("approval_id"))
                    approval_handoff = True
                    compact_payload = {
                        "status": "approval_pending",
                        "approval_id": approval_id,
                        "action_name": str(result.get("action_name", "")).strip() or call_name,
                        "summary": str(result.get("summary", "")).strip() or None,
                        "reason": str(result.get("reason", "")).strip() or None,
                    }
                    tool_messages.append(
                        ToolMessage(
                            content=_safe_json(compact_payload),
                            tool_call_id=call_id,
                            additional_kwargs={"opentulpa_control": compact_payload},
                        )
                    )
                    tool_outcomes.append(
                        {
                            "tool_name": call_name,
                            "tool_call_id": call_id,
                            "status": "approval_pending",
                            "approval_id": approval_id,
                            "action_name": str(result.get("action_name", "")).strip() or call_name,
                            "summary": str(result.get("summary", "")).strip() or None,
                            "reason": str(result.get("reason", "")).strip() or None,
                            "result_text": _safe_json(compact_payload),
                        }
                    )
                    _log(
                        state,
                        "graph.tools.approval_handoff",
                        tool_name=call_name,
                        tool_call_id=call_id,
                    )
                else:
                    tool_messages.append(
                        ToolMessage(
                            content=model_visible_result_text,
                            tool_call_id=call_id,
                            additional_kwargs={"opentulpa_control": {"status": "ok"}},
                        )
                    )
                    tool_outcomes.append(
                        {
                            "tool_name": call_name,
                            "tool_call_id": call_id,
                            "status": "ok",
                            "result_text": result_text,
                        }
                    )
                    if call_name == "skill_get":
                        requested_name = str(args.get("name", "")).strip()
                        snapshot = _extract_invoked_skill_snapshot(result, requested_name=requested_name)
                        if snapshot is not None:
                            skill_name, skill_text = snapshot
                            merged_names = [*invoked_skill_list]
                            if skill_name not in merged_names:
                                merged_names.append(skill_name)
                            invoked_skill_list = merged_names[-3:]
                            if invoked_skill_context:
                                invoked_skill_context = f"{invoked_skill_context}\n\n---\n\n{skill_text}"
                            else:
                                invoked_skill_context = skill_text
            except Exception as exc:
                had_error = True
                error_text = f"TOOL_ERROR: {call_name} failed: {exc}"
                failed_tool_names.append(call_name)
                failed_tool_errors.append(str(exc).strip())
                _log(
                    state,
                    "graph.tools.error",
                    tool_name=call_name,
                    tool_call_id=call_id,
                    error=str(exc)[:500],
                )
                tool_messages.append(
                    ToolMessage(
                        content=error_text,
                        tool_call_id=call_id,
                        additional_kwargs={
                            "opentulpa_control": {
                                "status": "error",
                                "error": str(exc)[:500],
                            }
                        },
                    )
                )
                tool_outcomes.append(
                    {
                        "tool_name": call_name,
                        "tool_call_id": call_id,
                        "status": "error",
                        "error": str(exc)[:500],
                        "result_text": error_text,
                    }
                )
        update: dict[str, Any] = {
            "messages": tool_messages,
            "tool_outcomes": tool_outcomes,
            "approval_handoff": approval_handoff,
            "turn_status": "approval_pending" if approval_handoff else "running",
            "active_invoked_skill_names": invoked_skill_list,
            "active_invoked_skill_context": invoked_skill_context,
            "active_skill_context": invoked_skill_context,
        }
        if had_error:
            next_tool_error_count = int(state.get("tool_error_count", 0)) + 1
            last_tool_error = next(
                (item for item in reversed(failed_tool_errors) if item),
                "tool execution failed",
            )
            update["tool_error_count"] = next_tool_error_count
            update["last_tool_error"] = last_tool_error
            if (
                turn_mode == "routine_wake"
                and next_tool_error_count >= 2
                and "composio_tool_execute" in failed_tool_names
            ):
                failure_summary = (
                    "AUTOMATION_EXECUTION_FAILED: repeated composio_tool_execute errors during "
                    f"wake execution. Latest error: {last_tool_error[:500]}"
                )
                _log(
                    state,
                    "graph.tools.abort_after_repeated_error",
                    tool_name="composio_tool_execute",
                    tool_error_count=next_tool_error_count,
                    error=last_tool_error[:500],
                    turn_mode=turn_mode,
                )
                update["messages"] = [
                    *tool_messages,
                    AIMessage(content=failure_summary),
                ]
                update["turn_status"] = "failed"
                return Command(update=update, goto="finalize_turn")
        _log(
            state,
            "graph.tools.complete",
            emitted_messages=len(tool_messages),
            had_error=had_error,
        )
        goto: Literal["agent", "__end__"] = END if approval_handoff else "agent"
        return Command(update=update, goto=goto)

    def _latest_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
        if not messages:
            return []
        start = 0
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], HumanMessage):
                start = idx
                break
        return messages[start:]

    def _collect_recent_tool_outputs(turn_messages: list[AnyMessage]) -> list[str]:
        if not turn_messages:
            return []
        outputs: list[str] = []
        for msg in turn_messages:
            if isinstance(msg, ToolMessage):
                text = _content_to_text(getattr(msg, "content", "")).strip()
                if text:
                    outputs.append(text)
        return outputs

    def _serialize_turn_window(turn_messages: list[AnyMessage]) -> str:
        parts: list[str] = []
        for msg in turn_messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, ToolMessage):
                role = "tool"
            else:
                continue
            text = _content_to_text(getattr(msg, "content", "")).strip()
            if not text:
                continue
            chunk = f"[{role}] {text}"
            parts.append(chunk)
        return "\n".join(parts)

    def _tail_messages_to_token_budget(
        all_messages: list[AnyMessage],
        *,
        token_budget: int,
    ) -> list[AnyMessage]:
        if not all_messages:
            return []
        budget = max(200, int(token_budget))
        kept_rev: list[AnyMessage] = []
        used = 0
        for msg in reversed(all_messages):
            text = _message_to_text(msg)
            tok = max(1, _approx_tokens(text))
            if kept_rev and used + tok > budget:
                break
            kept_rev.append(msg)
            used += tok
            if used >= budget:
                break
        kept_rev.reverse()
        return kept_rev

    async def claim_check_node(state: AgentState) -> Command[Literal["agent", "finalize_turn"]]:
        def _retry_backoff_seconds(retry_count: int) -> float:
            safe_retry = max(0, int(retry_count))
            return min(3.2, 0.2 * (2**safe_retry))

        messages = state.get("messages", [])
        if not messages:
            return Command(
                update={
                    "claim_check_needs_retry": False,
                    "claim_check_retry_count": 0,
                    "claim_check_verdict": {"usable": True, "mismatch": False},
                },
                goto="finalize_turn",
            )
        last = messages[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            return Command(
                update={
                    "claim_check_needs_retry": False,
                    "claim_check_retry_count": 0,
                    "claim_check_verdict": {"usable": True, "mismatch": False},
                },
                goto="finalize_turn",
            )

        retry_count = int(state.get("claim_check_retry_count", 0))
        max_claim_check_retries = _compute_claim_check_retry_limit(runtime)
        max_empty_output_retries = _compute_empty_output_retry_limit(runtime)
        _log(
            state,
            "graph.claim_check.start",
            retry_count=retry_count,
            max_claim_check_retries=max_claim_check_retries,
            max_empty_output_retries=max_empty_output_retries,
        )

        assistant_text = _content_to_text(getattr(last, "content", "")).strip()
        if not assistant_text:
            if retry_count >= max_empty_output_retries:
                _log(
                    state,
                    "graph.claim_check.empty_output_exhausted",
                    retry_count=retry_count,
                )
                return Command(
                    update={
                        "claim_check_needs_retry": False,
                        "claim_check_retry_count": 0,
                        "claim_check_verdict": {
                            "usable": True,
                            "mismatch": False,
                            "reason": "empty_output_exhausted",
                        },
                    },
                    goto="finalize_turn",
                )
            backoff_seconds = _retry_backoff_seconds(retry_count)
            _log(
                state,
                "graph.claim_check.empty_output_retry",
                retry_count=retry_count,
                backoff_seconds=backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)
            return Command(
                update={
                    "messages": [
                        SystemMessage(
                            content=(
                                "SELF_CHECK_EMPTY_OUTPUT: Your previous response produced no visible output. "
                                "Continue and provide a concrete answer or execute the needed tools now. "
                                "Do not stop silently."
                            )
                        )
                    ],
                    "claim_check_needs_retry": True,
                    "claim_check_retry_count": retry_count + 1,
                    "claim_check_verdict": {
                        "usable": True,
                        "mismatch": True,
                        "reason": "empty_output_retry",
                    },
                },
                goto="agent",
            )

        turn_messages = _latest_turn_messages(messages)
        if not turn_messages:
            return Command(
                update={
                    "claim_check_needs_retry": False,
                    "claim_check_retry_count": 0,
                    "claim_check_verdict": {"usable": True, "mismatch": False},
                },
                goto="finalize_turn",
            )
        user_text = _content_to_text(getattr(turn_messages[0], "content", "")).strip()
        recent_tool_outputs = _collect_recent_tool_outputs(turn_messages)
        turn_window = _serialize_turn_window(turn_messages)
        tool_budget = max(
            300,
            min(3000, int(getattr(runtime, "_context_short_term_low_tokens", 3500) * 0.25)),
        )
        recent_tool_outputs = [
            _trim_text_to_token_budget(x, token_budget=tool_budget)
            for x in recent_tool_outputs[-8:]
        ]
        turn_window_budget = max(
            1200,
            min(6000, int(getattr(runtime, "_context_short_term_low_tokens", 3500) * 0.6)),
        )
        turn_window = _trim_text_to_token_budget(turn_window, token_budget=turn_window_budget)
        verdict = await runtime.verify_completion_claim(
            user_text=user_text,
            assistant_text=assistant_text,
            recent_tool_outputs=recent_tool_outputs,
            turn_window=turn_window,
        )
        _log(
            state,
            "graph.claim_check.verdict",
            retry_count=retry_count,
            usable=bool(verdict.get("usable", True)),
            mismatch=bool(verdict.get("mismatch", False)),
            applies=bool(verdict.get("applies", False)),
            confidence=verdict.get("confidence"),
        )
        if not bool(verdict.get("usable", True)):
            if retry_count >= max_claim_check_retries:
                _log(
                    state,
                    "graph.claim_check.unusable_exhausted",
                    retry_count=retry_count,
                )
                return Command(
                    update={
                        "claim_check_needs_retry": False,
                        "claim_check_retry_count": 0,
                        "claim_check_verdict": verdict,
                    },
                    goto="finalize_turn",
                )
            backoff_seconds = _retry_backoff_seconds(retry_count)
            _log(
                state,
                "graph.claim_check.unusable_retry",
                retry_count=retry_count,
                backoff_seconds=backoff_seconds,
            )
            await asyncio.sleep(backoff_seconds)
            reason = str(verdict.get("reason", "")).strip()[:180]
            note = (
                "SELF_CHECK_UNAVAILABLE: Claim checker returned an unusable decision. "
                "Continue the turn and either execute the needed tool action now or restate status clearly."
            )
            if reason:
                note += f" Reason={reason}."
            return Command(
                update={
                    "messages": [SystemMessage(content=note)],
                    "claim_check_needs_retry": True,
                    "claim_check_retry_count": retry_count + 1,
                    "claim_check_verdict": verdict,
                },
                goto="agent",
            )
        mismatch = bool(verdict.get("mismatch", False))
        if not mismatch:
            _log(state, "graph.claim_check.passed", retry_count=retry_count)
            return Command(
                update={
                    "claim_check_needs_retry": False,
                    "claim_check_retry_count": 0,
                    "claim_check_verdict": verdict,
                },
                goto="finalize_turn",
            )
        if retry_count >= max_claim_check_retries:
            _log(
                state,
                "graph.claim_check.mismatch_exhausted",
                retry_count=retry_count,
            )
            return Command(
                update={
                    "claim_check_needs_retry": False,
                    "claim_check_retry_count": 0,
                    "claim_check_verdict": verdict,
                },
                goto="finalize_turn",
            )

        reason = str(verdict.get("reason", "")).strip()[:180]
        repair = str(verdict.get("repair_instruction", "")).strip()[:220]
        backoff_seconds = _retry_backoff_seconds(retry_count)
        _log(
            state,
            "graph.claim_check.mismatch_retry",
            retry_count=retry_count,
            backoff_seconds=backoff_seconds,
            reason=reason,
        )
        await asyncio.sleep(backoff_seconds)
        note = (
            "SELF_CHECK_FAILED: Your last reply likely claimed an immediate action was done "
            "without confirmed tool evidence in this turn. "
            "Do not repeat the claim. Either execute the required tool now or state pending status clearly."
        )
        if reason:
            note += f" Reason={reason}."
        if repair:
            note += f" Fix={repair}."
        return Command(
            update={
                "messages": [SystemMessage(content=note)],
                "claim_check_needs_retry": True,
                "claim_check_retry_count": retry_count + 1,
                "claim_check_verdict": verdict,
            },
            goto="agent",
        )

    async def finalize_turn_node(state: AgentState) -> dict[str, Any]:
        if bool(state.get("approval_handoff", False)):
            return {
                "turn_status": "approval_pending",
                "final_response_text": "",
            }
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                if bool(getattr(message, "tool_calls", [])):
                    continue
                text = _content_to_text(getattr(message, "content", "")).strip()
                if text:
                    return {
                        "turn_status": "completed",
                        "final_response_text": text,
                    }
        return {
            "turn_status": "completed",
            "final_response_text": "",
        }

    builder = StateGraph(AgentState)
    builder.add_node(
        "agent",
        agent_node,
        retry_policy=RetryPolicy(max_attempts=3),
        destinations=("validate_tools", "finalize_turn"),
    )
    builder.add_node(
        "validate_tools",
        validate_tool_calls_node,
        retry_policy=RetryPolicy(max_attempts=2),
        destinations=("tools", "agent"),
    )
    builder.add_node(
        "tools",
        tools_node,
        retry_policy=RetryPolicy(max_attempts=3),
        destinations=("agent", END),
    )
    builder.add_node("finalize_turn", finalize_turn_node, retry_policy=RetryPolicy(max_attempts=1))
    builder.add_edge(START, "agent")
    return builder.compile(checkpointer=runtime._checkpointer)
