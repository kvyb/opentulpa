"""Stateful graph-control tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from opentulpa.agent.turn_plan import build_turn_plan_result, update_turn_plan

GRAPH_CONTROL_TOOL_NAMES = frozenset({"turn_plan"})


@dataclass(frozen=True)
class GraphControlToolResult:
    result: dict[str, Any]
    state_update: dict[str, Any]


def is_graph_control_tool(tool_name: str) -> bool:
    return tool_name in GRAPH_CONTROL_TOOL_NAMES


def execute_graph_control_tool(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    state: Mapping[str, Any],
) -> GraphControlToolResult:
    if tool_name != "turn_plan":
        raise ValueError(f"Unknown graph control tool: {tool_name}")
    items = args.get("items")
    turn_plan = update_turn_plan(
        state.get("turn_plan"),
        items=items,
        merge=args.get("merge", False),
    )
    return GraphControlToolResult(
        result=build_turn_plan_result(turn_plan),
        state_update={"turn_plan": turn_plan},
    )
