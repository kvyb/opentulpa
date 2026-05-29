"""Tool-validation node for the runtime graph."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from langgraph.types import Command

from opentulpa.agent.lc_messages import AIMessage, SystemMessage, ToolMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_budget import apply_tool_call_budget
from opentulpa.agent.tool_loop_guardrails import find_duplicate_tool_calls
from opentulpa.agent.tool_validation import (
    _build_tool_validation_repair_message,
    _routine_create_intent_validation_error,
    _summarize_tool_validation_errors,
    _validate_model_tool_call,
)
from opentulpa.agent.turn_control import graph_recursion_limit_from_config
from opentulpa.agent.turn_policy import normalize_turn_mode as _normalize_turn_mode
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import latest_user_text as _latest_user_text

logger = logging.getLogger(__name__)

ValidateToolsCommand = Command[Literal["tools", "agent", "finalize_turn"]]
GraphLogFn = Callable[..., None]
LoopLimitNearFn = Callable[[AgentState], bool]
RemainingStepsFn = Callable[[AgentState], int | None]
ValidateToolsNode = Callable[[AgentState], Awaitable[ValidateToolsCommand]]
LOOP_LIMIT_REPAIR_MESSAGE = (
    "RUNTIME_BUDGET_APPROACHING: Do not call more tools in this turn. Write natural "
    "user-facing prose now using the previous tool results and current context. "
    "If enough information exists, give the proposal, confirmation, or answer. "
    "If not, state the exact blocker and next step."
)


def _set_message_tool_calls(message: AIMessage, tool_calls: list[Any]) -> None:
    message.tool_calls = tool_calls  # type: ignore[assignment]


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

        original_tool_calls = list(last.tool_calls)
        original_tool_call_count = len(original_tool_calls)
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
                        "because the turn is near its runtime budget. Use previous tool "
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
        budget_decision = apply_tool_call_budget(
            state.get("turn_budget"),
            original_tool_calls,
            turn_mode=turn_mode,
            graph_recursion_limit=graph_recursion_limit_from_config(runtime, None),
        )
        allowed_tool_calls: list[Any] = []
        if budget_decision.trimmed and budget_decision.allowed_tool_calls:
            log(
                state,
                "graph.validate_tools.trimmed_tool_budget",
                original_tool_call_count=original_tool_call_count,
                allowed_tool_call_count=len(budget_decision.allowed_tool_calls),
                turn_mode=turn_mode,
            )
        for blocked in budget_decision.blocked_calls:
            validation_errors.append(
                ToolMessage(
                    content=blocked.error,
                    tool_call_id=blocked.tool_call_id,
                )
            )
        duplicate_tool_calls = find_duplicate_tool_calls(
            requested_calls=budget_decision.allowed_tool_calls,
            prior_tool_outcomes=state.get("tool_outcomes"),
            trace_id=str(state.get("agent_trace_id", "") or "").strip(),
        )
        for duplicate in duplicate_tool_calls:
            validation_errors.append(
                ToolMessage(
                    content=duplicate.error,
                    tool_call_id=duplicate.tool_call_id,
                )
            )
        for call in budget_decision.allowed_tool_calls:
            call_name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            if any(message.tool_call_id == call_id for message in validation_errors):
                continue
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
            if call_name == "routine_create":
                intent_error = await _routine_create_intent_validation_error(
                    runtime,
                    args=args,
                    latest_user_text=latest_user,
                    prior_assistant_text=prior_assistant,
                    turn_mode=turn_mode,
                )
                if intent_error:
                    validation_errors.append(
                        ToolMessage(
                            content=intent_error,
                            tool_call_id=call_id,
                        )
                    )
                    continue
            allowed_tool_calls.append(call)
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
        if budget_decision.trimmed and allowed_tool_calls:
            _set_message_tool_calls(last, allowed_tool_calls)
            log(
                state,
                "graph.validate_tools.trimmed_tool_budget_validated",
                original_tool_call_count=original_tool_call_count,
                allowed_tool_call_count=len(allowed_tool_calls),
                turn_mode=turn_mode,
            )
        log(
            state,
            "graph.validate_tools.passed",
            tool_call_count=len(allowed_tool_calls) if budget_decision.trimmed else original_tool_call_count,
            turn_mode=turn_mode,
        )
        return Command(update={"tool_validation_passed": True}, goto="tools")

    return validate_tool_calls_node
