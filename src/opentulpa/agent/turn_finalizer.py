"""No-tools final response generation for a turn."""

from __future__ import annotations

import logging
from typing import Any

from opentulpa.agent.lc_messages import AIMessage, HumanMessage, ToolMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_outcome_finalizers import (
    final_response_hint_from_tool_outcomes,
    generate_final_response_from_tool_hint,
)
from opentulpa.agent.turn_budget import mark_finalizer_used
from opentulpa.agent.turn_policy import normalize_turn_mode
from opentulpa.agent.utils import content_to_text, latest_user_text

logger = logging.getLogger(__name__)


async def finalize_turn_response(*, runtime: Any, state: AgentState) -> dict[str, Any]:
    messages = state.get("messages", [])
    latest_human_index = -1
    for index, message in enumerate(messages):
        if isinstance(message, HumanMessage):
            latest_human_index = index
    current_turn_messages = messages[latest_human_index + 1 :] if latest_human_index >= 0 else messages
    for message in reversed(current_turn_messages):
        if isinstance(message, AIMessage):
            if bool(getattr(message, "tool_calls", [])):
                continue
            text = content_to_text(getattr(message, "content", "")).strip()
            if text:
                return {"turn_status": "completed", "final_response_text": text}

    final_response_hint = final_response_hint_from_tool_outcomes(state.get("tool_outcomes"))
    if final_response_hint:
        fallback_text = await generate_final_response_from_tool_hint(
            runtime=runtime,
            state=state,
            hint=final_response_hint,
        )
        if fallback_text:
            return {"turn_status": "completed", "final_response_text": fallback_text}

    finalization_reason = str(state.get("turn_finalization_reason", "") or "").strip()
    if finalization_reason:
        fallback_text = await generate_final_response_from_turn_state(
            runtime=runtime,
            state=state,
            reason=finalization_reason,
        )
        if fallback_text:
            return _finalized_with_budget(runtime=runtime, state=state, text=fallback_text)

    return {"turn_status": "completed", "final_response_text": ""}


async def generate_final_response_from_turn_state(
    *,
    runtime: Any,
    state: AgentState,
    reason: str,
) -> str:
    """Ask for one final answer with tools stripped from the prompt contract."""

    ainvoke_fn = getattr(runtime, "ainvoke_model", None)
    if not callable(ainvoke_fn):
        return ""
    turn_mode = normalize_turn_mode(state.get("turn_mode"))
    model = _finalizer_model(runtime, turn_mode=turn_mode)
    if model is None:
        return ""
    safe_reason = str(reason or "runtime_budget_exhausted").strip()
    messages = [
        HumanMessage(
            content=(
                "You are finishing the current OpenTulpa turn. Do not call tools.\n"
                "Use only the current request, assistant notes, verified tool results, and current plan below.\n"
                "Fulfill the requested deliverable directly with the best evidence available now.\n"
                "If the task is incomplete, give the best useful partial result and the exact blocker, "
                "but do not end by asking the user to continue unless no useful deliverable can be produced.\n"
                "Do not mention internal graph state, budgets, traces, or JSON.\n\n"
                f"Finalization reason: {safe_reason}\n\n"
                f"Current turn state:\n{_current_turn_state_for_finalizer(state)}\n\n"
                "Write the final user-facing reply now."
            )
        ),
    ]
    try:
        response = await ainvoke_fn(
            model,
            messages,
            stable_prefix_count=0,
            cacheable_prefix_count=0,
            call_context={
                "call_site": "graph_turn_budget_finalizer",
                "trace_id": state.get("agent_trace_id"),
                "thread_id": state.get("thread_id"),
                "customer_id": state.get("customer_id"),
                "turn_mode": turn_mode,
            },
        )
    except Exception:
        logger.exception("turn_budget_finalizer_model_call_failed")
        return ""
    if bool(getattr(response, "tool_calls", [])):
        return ""
    return content_to_text(getattr(response, "content", "")).strip()


def _finalized_with_budget(*, runtime: Any, state: AgentState, text: str) -> dict[str, Any]:
    return {
        "turn_status": "completed",
        "final_response_text": text,
        "turn_budget": mark_finalizer_used(
            state.get("turn_budget"),
            turn_mode=state.get("turn_mode"),
            graph_recursion_limit=_graph_recursion_limit_from_runtime(runtime),
        ),
    }


def _graph_recursion_limit_from_runtime(runtime: Any) -> int:
    try:
        return max(5, int(getattr(runtime, "recursion_limit", 30)))
    except Exception:
        return 30


def _finalizer_model(runtime: Any, *, turn_mode: str) -> Any:
    if turn_mode == "routine_wake":
        model = getattr(runtime, "_wake_execution_model", None)
        if model is not None:
            return model
    model = getattr(runtime, "_model", None)
    if model is not None:
        return model
    fallback = getattr(runtime, "model_with_tools_for_turn_mode", None)
    if callable(fallback):
        return fallback(turn_mode)
    return None


def _current_turn_state_for_finalizer(state: AgentState) -> str:
    messages = state.get("messages", [])
    latest_user_index = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            latest_user_index = index
            break
    lines = [f"User request: {latest_user_text(messages)}"]
    turn_plan = state.get("turn_plan")
    if isinstance(turn_plan, list) and turn_plan:
        lines.append("Current plan:")
        for item in turn_plan[:8]:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('id', '')}: {item.get('content', '')} "
                    f"({item.get('status', '')})"
                )
    tool_outcomes = state.get("tool_outcomes")
    if isinstance(tool_outcomes, list) and tool_outcomes:
        lines.append("Verified tool outcomes:")
        for outcome in tool_outcomes[-8:]:
            if not isinstance(outcome, dict):
                continue
            name = str(outcome.get("tool_name", "tool") or "tool").strip()
            status = str(outcome.get("status", "") or "").strip()
            result = str(outcome.get("result_text", "") or outcome.get("error", "") or "").strip()
            if result:
                lines.append(f"- {name} [{status}]: {_trim_finalizer_text(result)}")
    notes: list[str] = []
    for message in messages[latest_user_index + 1 :]:
        if isinstance(message, AIMessage) and not bool(getattr(message, "tool_calls", [])):
            text = content_to_text(getattr(message, "content", "")).strip()
            if text:
                notes.append(f"Assistant note: {_trim_finalizer_text(text)}")
        elif isinstance(message, ToolMessage):
            text = content_to_text(getattr(message, "content", "")).strip()
            if text:
                notes.append(f"Tool result: {_trim_finalizer_text(text)}")
    lines.extend(notes[-8:])
    return "\n".join(line for line in lines if line.strip()).strip()


def _trim_finalizer_text(text: str, *, limit: int = 1200) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return f"{raw[: limit - 3].rstrip()}..."
