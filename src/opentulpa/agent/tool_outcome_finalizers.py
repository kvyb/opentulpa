"""Internal final-response hints extracted from successful tool outcomes."""

from __future__ import annotations

import json
import logging
from typing import Any

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.turn_policy import normalize_turn_mode as _normalize_turn_mode
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import latest_user_text as _latest_user_text

logger = logging.getLogger(__name__)


def final_response_hint_from_tool_outcomes(outcomes: Any) -> str:
    if not isinstance(outcomes, list):
        return ""
    for outcome in reversed(outcomes):
        if not isinstance(outcome, dict) or outcome.get("status") != "ok":
            continue
        direct = str(outcome.get("final_response_hint", "") or "").strip()
        if direct:
            return direct
        payload = _tool_outcome_payload(outcome)
        direct_hint = _final_response_hint(payload)
        if direct_hint:
            return direct_hint
    return ""


def _tool_outcome_payload(outcome: dict[str, Any]) -> dict[str, Any]:
    raw = str(outcome.get("result_text", "") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _final_response_hint(payload: dict[str, Any]) -> str:
    for key in ("final_response_hint", "user_visible_reply", "confirmation_text"):
        hint = str(payload.get(key, "") or "").strip()
        if hint:
            return hint
    result = payload.get("result")
    if isinstance(result, dict):
        return _final_response_hint(result)
    return ""


async def generate_final_response_from_tool_hint(
    *,
    runtime: Any,
    state: AgentState,
    hint: str,
) -> str:
    hint = str(hint or "").strip()
    if not hint:
        return ""
    ainvoke_fn = getattr(runtime, "ainvoke_model", None)
    if not callable(ainvoke_fn):
        return ""
    turn_mode = _normalize_turn_mode(state.get("turn_mode"))
    model = (
        getattr(runtime, "_wake_execution_model", None)
        if turn_mode == "routine_wake"
        else getattr(runtime, "_model", None)
    )
    if model is None:
        model = runtime.model_with_tools_for_turn_mode(turn_mode)
    assert model is not None
    messages = [
        SystemMessage(
            content=(
                "Write the final user-facing reply in natural prose. Use only the verified "
                "tool outcome below. Do not mention internal tool names, JSON, traces, or "
                "implementation details. Do not add unsupported details. Be concise."
            )
        ),
        HumanMessage(
            content=(
                f"User asked:\n{_latest_user_text(state.get('messages', []))}\n\n"
                f"Verified tool outcome:\n{hint}\n\n"
                "Write the final reply."
            )
        ),
    ]
    try:
        response = await ainvoke_fn(
            model,
            messages,
            stable_prefix_count=1,
            cacheable_prefix_count=1,
            call_context={
                "call_site": "graph_final_response_from_tool_hint",
                "trace_id": state.get("agent_trace_id"),
                "thread_id": state.get("thread_id"),
                "customer_id": state.get("customer_id"),
                "turn_mode": turn_mode,
            },
        )
    except Exception:
        logger.exception("final_response_hint_model_call_failed")
        return ""
    if bool(getattr(response, "tool_calls", [])):
        return ""
    return _content_to_text(getattr(response, "content", "")).strip()
