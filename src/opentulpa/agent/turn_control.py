"""Runtime turn-control policy for budget and loop status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_budget import web_search_call_count
from opentulpa.agent.turn_budget import (
    TurnBudgetDecision,
    TurnBudgetState,
    consume_model_call,
    record_search_calls,
    record_tool_round,
    remaining_model_calls,
)

LOOP_LIMIT_STATUS_REMAINING_STEPS = 3


@dataclass(frozen=True, slots=True)
class ModelBudgetResult:
    decision: TurnBudgetDecision
    update: dict[str, Any]


def graph_recursion_limit_from_config(runtime: Any, config: Any) -> int:
    if isinstance(config, dict):
        raw_limit = config.get("recursion_limit")
        if raw_limit is not None:
            try:
                return max(5, int(raw_limit))
            except Exception:
                pass
    try:
        return max(5, int(getattr(runtime, "recursion_limit", 30)))
    except Exception:
        return 30


def remaining_graph_steps(state: AgentState) -> int | None:
    try:
        remaining = int(state.get("remaining_steps", 0))
    except Exception:
        return None
    return remaining if remaining > 0 else None


def loop_limit_near(state: AgentState) -> bool:
    remaining = remaining_graph_steps(state)
    if remaining is not None and remaining <= LOOP_LIMIT_STATUS_REMAINING_STEPS:
        return True
    budget_remaining = remaining_model_calls(state.get("turn_budget"))
    return budget_remaining is not None and budget_remaining <= 0


def consume_model_budget_for_turn(
    *,
    runtime: Any,
    config: Any,
    state: AgentState,
    turn_mode: str,
) -> ModelBudgetResult:
    decision = consume_model_call(
        state.get("turn_budget"),
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit_from_config(runtime, config),
    )
    return ModelBudgetResult(decision=decision, update={"turn_budget": decision.state})


def record_tool_round_for_turn(
    *,
    runtime: Any,
    state: AgentState,
    turn_mode: str,
    tool_calls: list[Any] | None = None,
) -> TurnBudgetState:
    graph_recursion_limit = graph_recursion_limit_from_config(runtime, None)
    budget = record_tool_round(
        state.get("turn_budget"),
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    search_call_count = sum(web_search_call_count(call) for call in (tool_calls or []))
    if search_call_count <= 0:
        return budget
    return record_search_calls(
        budget,
        search_call_count=search_call_count,
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
