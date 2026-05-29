"""Stateful graph-control tool execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain.tools import tool

from opentulpa.agent.turn_plan import build_turn_plan_result, update_turn_plan


@dataclass(frozen=True)
class GraphControlToolResult:
    result: dict[str, Any]
    state_update: dict[str, Any]


@dataclass(frozen=True)
class GraphControlTool:
    name: str
    schema_tool: Any
    execute: Callable[[Mapping[str, Any], Mapping[str, Any]], GraphControlToolResult]


@tool
async def turn_plan(items: list[dict[str, Any]] | None = None, merge: bool = False) -> Any:
    """Plan and track complex current-turn work.

    Use this in interactive chat for longer-horizon research, discovery,
    analysis, report/list creation, or other multi-step work. Items use
    id, content, status: pending|in_progress|completed|cancelled. Keep
    exactly one item in_progress, update statuses as work moves, and keep
    the plan realistic for the current runtime.
    """
    del items, merge
    return {
        "ok": False,
        "error": (
            "GRAPH_CONTROL_TOOL_ONLY: turn_plan must be executed by the "
            "runtime graph because it updates current-turn graph state."
        ),
    }


def _execute_turn_plan(
    args: Mapping[str, Any],
    state: Mapping[str, Any],
) -> GraphControlToolResult:
    items = args.get("items")
    turn_plan_items = update_turn_plan(
        state.get("turn_plan"),
        items=items,
        merge=args.get("merge", False),
    )
    return GraphControlToolResult(
        result=build_turn_plan_result(turn_plan_items),
        state_update={"turn_plan": turn_plan_items},
    )


_GRAPH_CONTROL_TOOL_REGISTRY = {
    "turn_plan": GraphControlTool(
        name="turn_plan",
        schema_tool=turn_plan,
        execute=_execute_turn_plan,
    )
}


def is_graph_control_tool(tool_name: str) -> bool:
    return tool_name in _GRAPH_CONTROL_TOOL_REGISTRY


def graph_control_tool_registry(runtime: Any | None = None) -> dict[str, GraphControlTool]:
    del runtime
    return dict(_GRAPH_CONTROL_TOOL_REGISTRY)


def register_graph_control_tools(runtime: Any) -> dict[str, Any]:
    return {name: item.schema_tool for name, item in graph_control_tool_registry(runtime).items()}


def execute_graph_control_tool(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    state: Mapping[str, Any],
) -> GraphControlToolResult:
    control_tool = graph_control_tool_registry().get(tool_name)
    if control_tool is None:
        raise ValueError(f"Unknown graph control tool: {tool_name}")
    return control_tool.execute(args, state)
