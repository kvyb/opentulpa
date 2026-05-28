"""Current-turn planning tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool


def register_turn_plan_tools(runtime: Any) -> dict[str, Any]:
    del runtime

    @tool
    async def turn_plan(items: list[dict[str, Any]] | None = None, merge: bool = False) -> Any:
        """Manage a private current-turn plan for complex work.

        Items use id, content, status: pending|in_progress|completed|cancelled.
        """
        del items, merge
        return {
            "ok": False,
            "error": (
                "GRAPH_CONTROL_TOOL_ONLY: turn_plan must be executed by the "
                "runtime graph because it updates current-turn graph state."
            ),
        }

    return {"turn_plan": turn_plan}
