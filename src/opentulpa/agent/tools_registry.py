"""Compose the agent runtime tool registry from domain modules."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.tools import (
    register_browser_tools,
    register_composio_tools,
    register_core_tools,
    register_intake_tools,
    register_routine_tools,
    register_skill_tools,
)
from opentulpa.agent.tools.tool_gateway_tools import register_tool_gateway_tools


def register_runtime_tools(runtime: Any) -> dict[str, Any]:
    tools: dict[str, Any] = {}
    tools.update(register_core_tools(runtime))
    tools.update(register_skill_tools(runtime))
    tools.update(register_intake_tools(runtime))
    tools.update(register_composio_tools(runtime))
    tools.update(register_browser_tools(runtime))
    tools.update(register_routine_tools(runtime))
    tools.update(register_tool_gateway_tools(runtime, tools))
    return tools
