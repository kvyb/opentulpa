"""Tool-validation node for the runtime graph."""

from __future__ import annotations

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
_MAX_WEB_SEARCH_CALLS_PER_TURN = 2


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


def _successful_tool_count_in_latest_turn(messages: list[Any], *, tool_name: str) -> int:
    return sum(
        1
        for call in _successful_tool_calls_in_latest_turn(messages)
        if str(call.get("name", "")) == tool_name
    )


def _web_search_budget_error(*, prior_success_count: int) -> str:
    if prior_success_count >= _MAX_WEB_SEARCH_CALLS_PER_TURN:
        return (
            "WEB_SEARCH_BUDGET_EXCEEDED: web_search is limited to "
            f"{_MAX_WEB_SEARCH_CALLS_PER_TURN} calls per turn. Do not call web_search again "
            "in this turn. Use browser_use_run for dynamic web investigation, "
            "fetch_url_content for already found URLs, or synthesize from existing results."
        )
    remaining = _MAX_WEB_SEARCH_CALLS_PER_TURN - prior_success_count
    return (
        "WEB_SEARCH_BATCH_TOO_LARGE: web_search is limited to "
        f"{_MAX_WEB_SEARCH_CALLS_PER_TURN} calls per turn. This turn has {remaining} "
        "web_search call(s) remaining. Retry with no more than that many web_search calls "
        "in the same batch, or use browser_use_run for dynamic web investigation."
    )


def build_validate_tool_calls_node(
    *,
    runtime: Any,
    required_args: dict[str, tuple[str, ...]],
    forbidden_tool_args: dict[str, set[str]],
    log: GraphLogFn,
    loop_limit_near: LoopLimitNearFn,
    remaining_steps: RemainingStepsFn,
    loop_limit_final_status_text: str,
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
            return Command(
                update={
                    "messages": [AIMessage(content=loop_limit_final_status_text)],
                    "tool_validation_passed": False,
                    "turn_status": "running",
                    "loop_limit_status_update_sent": True,
                },
                goto="finalize_turn",
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
            if call_name == "web_search":
                prior_web_search_count = _successful_tool_count_in_latest_turn(
                    messages[:-1],
                    tool_name="web_search",
                )
                current_batch_web_search_count = sum(
                    1
                    for existing_call in last.tool_calls[:call_idx]
                    if str(existing_call.get("name", "")) == "web_search"
                )
                if prior_web_search_count + current_batch_web_search_count >= _MAX_WEB_SEARCH_CALLS_PER_TURN:
                    validation_errors.append(
                        ToolMessage(
                            content=_web_search_budget_error(
                                prior_success_count=prior_web_search_count,
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
