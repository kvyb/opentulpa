"""Tool-validation node for the runtime graph."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.types import Command

from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_validation import (
    _build_tool_validation_repair_message,
    _routine_create_intent_validation_error,
    _summarize_tool_validation_errors,
    _validate_model_tool_call,
)
from opentulpa.agent.turn_policy import normalize_turn_mode as _normalize_turn_mode
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import latest_user_text as _latest_user_text

logger = logging.getLogger(__name__)

ValidateToolsCommand = Command[Literal["tools", "agent", "finalize_turn"]]
GraphLogFn = Callable[..., None]
LoopLimitNearFn = Callable[[AgentState], bool]
RemainingStepsFn = Callable[[AgentState], int | None]
ValidateToolsNode = Callable[[AgentState], Awaitable[ValidateToolsCommand]]
_MAX_WEB_SEARCH_CALLS_PER_TURN = 5
LOOP_LIMIT_REPAIR_MESSAGE = (
    "LOOP_LIMIT_APPROACHING: Do not call more tools in this turn. Write natural "
    "user-facing prose now using the previous tool results and current context. "
    "If enough information exists, give the proposal, confirmation, or answer. "
    "If not, state the exact blocker and next step."
)


def _successful_tool_calls_in_latest_turn(messages: list[Any]) -> list[dict[str, Any]]:
    latest_user_idx = 0
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            latest_user_idx = idx
            break

    calls_by_id: dict[str, dict[str, Any]] = {}
    for message in messages[latest_user_idx:]:
        if isinstance(message, AIMessage):
            for call in getattr(message, "tool_calls", []) or []:
                call_id = str((call or {}).get("id", "")).strip()
                call_name = str((call or {}).get("name", "")).strip()
                if call_id and call_name:
                    calls_by_id[call_id] = {
                        "name": call_name,
                        "args": (call or {}).get("args", {}) or {},
                    }

    successful: list[dict[str, Any]] = []
    for message in messages[latest_user_idx:]:
        if not isinstance(message, ToolMessage):
            continue
        call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        control = getattr(message, "additional_kwargs", {}).get("opentulpa_control", {})
        status = str(control.get("status", "") if isinstance(control, dict) else "").strip()
        if status == "ok":
            call = calls_by_id.get(call_id)
            if call is not None:
                successful.append(call)
    return successful


def _web_search_call_count(call: Any) -> int:
    if not isinstance(call, dict):
        return 0
    call_name = str(call.get("name", "")).strip()
    args = call.get("args", {}) or {}
    if call_name == "web_search":
        return 1
    if call_name != "tool_group_exec" or not isinstance(args, dict):
        return 0
    group = str(args.get("group", "")).strip().lower()
    command = str(args.get("command", "")).strip()
    if group == "web" and command == "web_search":
        return 1
    calls = _coerce_tool_group_calls(args.get("calls"))
    count = 0
    for item in calls:
        if not isinstance(item, dict):
            continue
        item_group = str(item.get("group", "")).strip().lower()
        item_command = str(item.get("command", "")).strip()
        if item_group == "web" and item_command == "web_search":
            count += 1
    return count


def _coerce_tool_group_calls(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _web_search_success_count_in_latest_turn(messages: list[Any]) -> int:
    return sum(_web_search_call_count(call) for call in _successful_tool_calls_in_latest_turn(messages))


def _web_search_budget_error(*, prior_success_count: int) -> str:
    if prior_success_count >= _MAX_WEB_SEARCH_CALLS_PER_TURN:
        return (
            "WEB_SEARCH_BUDGET_EXCEEDED: web_search is limited to "
            f"{_MAX_WEB_SEARCH_CALLS_PER_TURN} calls per turn. Do not call web_search again "
            "in this turn. Tell the user the maximum web_search cap was reached if more "
            "web discovery is needed. Otherwise use browser_use_run for dynamic web "
            "investigation, fetch_url_content for already found URLs, or synthesize and "
            "report the best current answer from existing results."
        )
    remaining = _MAX_WEB_SEARCH_CALLS_PER_TURN - prior_success_count
    return (
        "WEB_SEARCH_BATCH_TOO_LARGE: web_search is limited to "
        f"{_MAX_WEB_SEARCH_CALLS_PER_TURN} calls per turn. This turn has {remaining} "
        "web_search call(s) remaining. Retry with no more than that many web_search calls "
        "in the same batch. If that is not enough, use browser_use_run for dynamic web "
        "investigation or report to the user that the maximum web_search cap was reached."
    )


def build_validate_tool_calls_node(
    *,
    runtime: Any,
    required_args: dict[str, tuple[str, ...]],
    forbidden_tool_args: dict[str, set[str]],
    log: GraphLogFn,
    loop_limit_near: LoopLimitNearFn,
    remaining_steps: RemainingStepsFn,
) -> ValidateToolsNode:
    async def validate_tool_calls_node(state: AgentState) -> ValidateToolsCommand:
        messages = state.get("messages", [])
        if not messages:
            return Command(update={"tool_validation_passed": True}, goto="tools")
        last = messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return Command(update={"tool_validation_passed": True}, goto="tools")
        log(
            state,
            "graph.validate_tools.start",
            tool_call_count=len(last.tool_calls),
            turn_mode=_normalize_turn_mode(state.get("turn_mode")),
        )

        validation_errors: list[ToolMessage] = []
        latest_user = _latest_user_text(messages)
        prior_assistant = ""
        turn_mode = _normalize_turn_mode(state.get("turn_mode"))
        if loop_limit_near(state):
            log(
                state,
                "graph.loop_limit_tool_call_blocked",
                tool_call_count=len(last.tool_calls),
                remaining_steps=remaining_steps(state),
                turn_mode=turn_mode,
            )
            blocked_messages = [
                ToolMessage(
                    content=(
                        "TOOL_NOT_RUN_LOOP_LIMIT: this requested tool call was not executed "
                        "because the turn is near its graph step budget. Use previous tool "
                        "results and current context to write the final user-facing reply now."
                    ),
                    tool_call_id=str(call.get("id", "")),
                    additional_kwargs={"opentulpa_control": {"status": "error"}},
                )
                for call in last.tool_calls
            ]
            return Command(
                update={
                    "messages": [
                        *blocked_messages,
                        SystemMessage(content=LOOP_LIMIT_REPAIR_MESSAGE),
                    ],
                    "tool_validation_passed": False,
                    "turn_status": "running",
                },
                goto="agent",
            )
        for msg in reversed(messages[:-1]):
            if isinstance(msg, AIMessage):
                candidate = _content_to_text(getattr(msg, "content", "")).strip()
                if candidate:
                    prior_assistant = candidate
                    break
        for call_idx, call in enumerate(last.tool_calls):
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            args = call.get("args", {}) or {}
            requested_web_search_count = _web_search_call_count(call)
            if requested_web_search_count:
                prior_web_search_count = _web_search_success_count_in_latest_turn(messages[:-1])
                current_batch_web_search_count = sum(
                    _web_search_call_count(existing_call) for existing_call in last.tool_calls[:call_idx]
                )
                consumed_web_search_count = prior_web_search_count + current_batch_web_search_count
                if (
                    consumed_web_search_count >= _MAX_WEB_SEARCH_CALLS_PER_TURN
                    or consumed_web_search_count + requested_web_search_count > _MAX_WEB_SEARCH_CALLS_PER_TURN
                ):
                    validation_errors.append(
                        ToolMessage(
                            content=_web_search_budget_error(
                                prior_success_count=consumed_web_search_count,
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue
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
            if call_name == "routine_create":
                intent_error = await _routine_create_intent_validation_error(
                    runtime,
                    args=args,
                    latest_user_text=latest_user,
                    prior_assistant_text=prior_assistant,
                    turn_mode=turn_mode,
                )
                if intent_error:
                    validation_errors.append(ToolMessage(content=intent_error, tool_call_id=call_id))
                    continue
        if validation_errors:
            error_summary = _summarize_tool_validation_errors(validation_errors)
            repair_message = _build_tool_validation_repair_message(validation_errors)
            log(
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
        log(
            state,
            "graph.validate_tools.passed",
            tool_call_count=len(last.tool_calls),
            turn_mode=turn_mode,
        )
        return Command(update={"tool_validation_passed": True}, goto="tools")

    return validate_tool_calls_node
