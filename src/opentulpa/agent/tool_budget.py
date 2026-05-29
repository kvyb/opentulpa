"""Runtime-owned per-turn tool budgets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from opentulpa.agent.turn_budget import initial_turn_budget, normalize_turn_budget


@dataclass(frozen=True, slots=True)
class ToolBudgetBlockedCall:
    tool_call_id: str
    error: str


@dataclass(frozen=True, slots=True)
class ToolBudgetDecision:
    allowed_tool_calls: list[Any]
    trimmed: bool
    blocked_calls: list[ToolBudgetBlockedCall]


def apply_tool_call_budget(
    raw_budget: Any,
    requested_calls: list[Any],
    *,
    turn_mode: Any,
    graph_recursion_limit: int,
) -> ToolBudgetDecision:
    """Trim current tool calls to the remaining runtime-owned per-turn tool budget."""

    budget = normalize_turn_budget(
        raw_budget or initial_turn_budget(
            turn_mode=turn_mode,
            graph_recursion_limit=graph_recursion_limit,
        ),
        turn_mode=turn_mode,
        graph_recursion_limit=graph_recursion_limit,
    )
    prior_web_search_count = int(budget["used_search_calls"])
    allowed_tool_calls: list[Any] = []
    blocked_calls: list[ToolBudgetBlockedCall] = []
    trimmed = False

    for call in requested_calls:
        requested_web_search_count = web_search_call_count(call)
        if requested_web_search_count <= 0:
            allowed_tool_calls.append(call)
            continue

        current_batch_web_search_count = sum(
            web_search_call_count(existing) for existing in allowed_tool_calls
        )
        remaining = (
            int(budget["max_search_calls"])
            - prior_web_search_count
            - current_batch_web_search_count
        )
        if remaining <= 0:
            trimmed = True
            blocked_calls.append(
                ToolBudgetBlockedCall(
                    tool_call_id=tool_call_id(call),
                    error=web_search_budget_error(
                        prior_success_count=prior_web_search_count + current_batch_web_search_count,
                        max_calls=int(budget["max_search_calls"]),
                    ),
                )
            )
            continue
        if requested_web_search_count > remaining:
            trimmed = True
            blocked_calls.append(
                ToolBudgetBlockedCall(
                    tool_call_id=tool_call_id(call),
                    error=web_search_budget_error(
                        prior_success_count=prior_web_search_count + current_batch_web_search_count,
                        max_calls=int(budget["max_search_calls"]),
                    ),
                )
            )
            continue
        allowed_tool_calls.append(call)

    if blocked_calls:
        return ToolBudgetDecision(
            allowed_tool_calls=allowed_tool_calls,
            trimmed=True,
            blocked_calls=blocked_calls,
        )
    return ToolBudgetDecision(
        allowed_tool_calls=allowed_tool_calls,
        trimmed=trimmed,
        blocked_calls=[],
    )


def tool_call_id(call: Any) -> str:
    if not isinstance(call, dict):
        return ""
    return str(call.get("id", "") or "").strip()


def web_search_call_count(call: Any) -> int:
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
    count = 0
    for item in coerce_tool_group_calls(args.get("calls")):
        if not isinstance(item, dict):
            continue
        item_group = str(item.get("group", "")).strip().lower()
        item_command = str(item.get("command", "")).strip()
        if item_group == "web" and item_command == "web_search":
            count += 1
    return count


def web_search_budget_error(*, prior_success_count: int, max_calls: int) -> str:
    if max_calls == 2:
        tool_label = "Exa web_search"
        error_prefix = "EXA_SEARCH"
    else:
        tool_label = "web_search"
        error_prefix = "WEB_SEARCH"
    if prior_success_count >= max_calls:
        return (
            f"{error_prefix}_BUDGET_EXCEEDED: {tool_label} is limited to {max_calls} "
            "calls per turn. Do not call web_search again in this turn. Use "
            'tool_group_exec(group="browser", command="browser_use_run", args_json={...}) '
            "for more web investigation, fetch_url_content for already found URLs, or tell "
            "the user the maximum web_search cap was reached and report the best current "
            "answer from existing results."
        )
    remaining = max_calls - prior_success_count
    return (
        f"{error_prefix}_BATCH_TOO_LARGE: {tool_label} is limited to {max_calls} "
        f"calls per turn. This turn has {remaining} "
        "web_search call(s) remaining. Retry with no more than that many web_search calls "
        "in the same batch. If that is not enough, use "
        'tool_group_exec(group="browser", command="browser_use_run", args_json={...}) '
        "for more web investigation or report to the user that the maximum web_search cap "
        "was reached."
    )


def coerce_tool_group_calls(raw: Any) -> list[Any]:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []
