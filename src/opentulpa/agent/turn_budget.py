"""Runtime-owned per-turn budget for graph agent loops.

Budget refunds are intentionally not supported right now. Failed calls, repair loops,
and retries consume the same turn budget; compaction is tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from opentulpa.agent.turn_policy import normalize_turn_mode
from opentulpa.integrations.web_search import get_web_search_backend_name

DEFAULT_MAX_WEB_SEARCH_CALLS_PER_TURN = 5
MAX_EXA_SEARCH_CALLS_PER_TURN = 2


class TurnBudgetState(TypedDict, total=False):
    max_model_calls: int
    used_model_calls: int
    used_tool_rounds: int
    max_search_calls: int
    used_search_calls: int
    finalizer_used: bool
    exhausted_reason: str


@dataclass(frozen=True, slots=True)
class TurnBudgetDecision:
    allowed: bool
    state: TurnBudgetState
    remaining_model_calls: int
    reason: str = ""


def initial_turn_budget(
    *,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> TurnBudgetState:
    max_model_calls = _model_call_budget(
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    assert max_model_calls >= 1
    return {
        "max_model_calls": max_model_calls,
        "used_model_calls": 0,
        "used_tool_rounds": 0,
        "max_search_calls": current_max_search_calls(),
        "used_search_calls": 0,
        "finalizer_used": False,
        "exhausted_reason": "",
    }


def consume_model_call(
    raw_budget: Any,
    *,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> TurnBudgetDecision:
    budget = normalize_turn_budget(
        raw_budget,
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    used = int(budget["used_model_calls"])
    max_calls = int(budget["max_model_calls"])
    if used >= max_calls:
        reason = f"model_call_budget_exhausted({used}/{max_calls})"
        budget["exhausted_reason"] = reason
        return TurnBudgetDecision(
            allowed=False,
            state=budget,
            remaining_model_calls=0,
            reason=reason,
        )
    budget["used_model_calls"] = used + 1
    remaining = max(0, max_calls - int(budget["used_model_calls"]))
    return TurnBudgetDecision(
        allowed=True,
        state=budget,
        remaining_model_calls=remaining,
    )


def record_tool_round(
    raw_budget: Any,
    *,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> TurnBudgetState:
    budget = normalize_turn_budget(
        raw_budget,
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    budget["used_tool_rounds"] = int(budget["used_tool_rounds"]) + 1
    return budget


def record_search_calls(
    raw_budget: Any,
    *,
    search_call_count: int,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> TurnBudgetState:
    budget = normalize_turn_budget(
        raw_budget,
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    budget["used_search_calls"] = min(
        int(budget["max_search_calls"]),
        int(budget["used_search_calls"]) + max(0, int(search_call_count)),
    )
    return budget


def mark_finalizer_used(
    raw_budget: Any,
    *,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> TurnBudgetState:
    budget = normalize_turn_budget(
        raw_budget,
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    budget["finalizer_used"] = True
    return budget


def normalize_turn_budget(
    raw_budget: Any,
    *,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> TurnBudgetState:
    default = initial_turn_budget(
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    if not isinstance(raw_budget, dict):
        return default
    max_model_calls = _positive_int(
        raw_budget.get("max_model_calls"),
        default["max_model_calls"],
    )
    used_model_calls = min(
        max_model_calls,
        _non_negative_int(raw_budget.get("used_model_calls"), 0),
    )
    used_tool_rounds = _non_negative_int(raw_budget.get("used_tool_rounds"), 0)
    max_search_calls = _positive_int(
        raw_budget.get("max_search_calls"),
        default["max_search_calls"],
    )
    used_search_calls = min(
        max_search_calls,
        _non_negative_int(raw_budget.get("used_search_calls"), 0),
    )
    return {
        "max_model_calls": max_model_calls,
        "used_model_calls": used_model_calls,
        "used_tool_rounds": used_tool_rounds,
        "max_search_calls": max_search_calls,
        "used_search_calls": used_search_calls,
        "finalizer_used": bool(raw_budget.get("finalizer_used", False)),
        "exhausted_reason": str(raw_budget.get("exhausted_reason", "") or "").strip(),
    }


def remaining_model_calls(raw_budget: Any) -> int | None:
    if not isinstance(raw_budget, dict):
        return None
    try:
        return max(
            0,
            int(raw_budget.get("max_model_calls", 0))
            - int(raw_budget.get("used_model_calls", 0)),
        )
    except Exception:
        return None


def budget_status_context(raw_budget: Any) -> str:
    if not isinstance(raw_budget, dict):
        return ""
    remaining = remaining_model_calls(raw_budget)
    if remaining is None:
        return ""
    max_calls = _positive_int(raw_budget.get("max_model_calls"), 1)
    used = _non_negative_int(raw_budget.get("used_model_calls"), 0)
    search_used = _non_negative_int(raw_budget.get("used_search_calls"), 0)
    search_max = _positive_int(raw_budget.get("max_search_calls"), 1)
    search_exhausted = "max_search_calls" in raw_budget and search_used >= search_max
    if remaining > 1 and not search_exhausted:
        return ""
    lines = [
        "TURN_BUDGET_STATUS\n",
        f"- Model calls used: {used}/{max_calls}.\n",
    ]
    if "max_search_calls" in raw_budget:
        lines.append(f"- Web searches used: {search_used}/{search_max}.\n")
        if search_exhausted:
            lines.append(
                "- The web_search cap is exhausted for this turn. Do not call web_search again. "
                "Use browser tools for further investigation, fetch already discovered URLs, "
                "or answer from current verified results.\n"
            )
    if remaining <= 1:
        lines.append(
            "- This turn is near its runtime budget. Produce the concrete user-facing "
            "result now from verified context and tool outputs. Do not send only a "
            "progress update, plan recap, promise of later delivery, or ask-to-continue "
            "handoff when a useful deliverable can be produced. Call more tools only if "
            "the answer would otherwise be materially wrong."
        )
    return "".join(lines)


def current_max_search_calls() -> int:
    try:
        provider = get_web_search_backend_name()
    except Exception:
        provider = "unknown"
    return max_search_calls_for_provider(provider)


def max_search_calls_for_provider(provider: Any) -> int:
    safe_provider = str(provider or "unknown").strip().lower()
    if safe_provider == "exa":
        return MAX_EXA_SEARCH_CALLS_PER_TURN
    return DEFAULT_MAX_WEB_SEARCH_CALLS_PER_TURN


def _model_call_budget(*, turn_mode: Any, graph_recursion_limit: int) -> int:
    graph_steps = max(5, int(graph_recursion_limit))
    estimated_model_calls = max(2, ((graph_steps - 5) // 4) + 2)
    mode = normalize_turn_mode(turn_mode)
    if mode == "workflow_setup":
        return max(6, min(32, estimated_model_calls))
    if mode == "interactive":
        return max(2, min(18, estimated_model_calls))
    if mode == "routine_wake":
        return max(2, min(8, estimated_model_calls))
    return max(2, min(5, estimated_model_calls))


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return max(1, int(default))
    return max(1, parsed)


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return max(0, int(default))
    return max(0, parsed)
