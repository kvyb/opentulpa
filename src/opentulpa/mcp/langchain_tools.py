"""Atomic LangChain tool and interrupt bundle over the trusted MCP broker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

from opentulpa.mcp.broker import MCPToolBroker
from opentulpa.mcp.models import MCPToolDescriptor
from opentulpa.specs import AgentRunContext


@dataclass(frozen=True, slots=True)
class MCPToolBundle:
    """Tools and the mandatory Deep Agents interrupt policy for one instance."""

    tools: tuple[BaseTool, ...]
    interrupt_on: Mapping[str, bool]


def build_mcp_tool_bundle(
    broker: MCPToolBroker,
    *,
    instance_id: str,
) -> MCPToolBundle:
    """Expose registered MCP tools with runtime context hidden from model schemas.

    The returned tools and ``interrupt_on`` mapping are one composition unit. A
    side-effecting tool marks its broker call approved only after Deep Agents has
    passed the same call through this persisted interrupt policy.
    """

    descriptors = broker.descriptors(instance_id)
    if not descriptors:
        raise ValueError("MCP capability instance has no registered tools")
    names = [descriptor.name for descriptor in descriptors]
    if len(names) != len(set(names)):
        raise ValueError("MCP capability instance contains duplicate tool names")
    interrupt_on = MappingProxyType(dict(broker.interrupt_on(instance_id)))
    if set(interrupt_on) != set(names):
        raise RuntimeError("MCP interrupt policy does not cover every registered tool")
    tools = tuple(
        _build_broker_tool(
            broker=broker,
            descriptor=descriptor,
            approval_required=interrupt_on[descriptor.name],
        )
        for descriptor in descriptors
    )
    return MCPToolBundle(tools=tools, interrupt_on=interrupt_on)


def _build_broker_tool(
    *,
    broker: MCPToolBroker,
    descriptor: MCPToolDescriptor,
    approval_required: bool,
) -> BaseTool:
    async def execute(
        *,
        runtime: ToolRuntime[AgentRunContext],
        **arguments: Any,
    ) -> dict[str, Any]:
        context = runtime.context
        if not isinstance(context, AgentRunContext):
            return {
                "status": "error",
                "data": None,
                "error": {
                    "code": "missing_run_context",
                    "message": "Trusted agent run context is unavailable.",
                    "retryable": False,
                },
                "audit_id": f"audit_{uuid4().hex}",
                "idempotency_key": None,
                "replayed": False,
            }
        result = await broker.invoke(
            instance_id=descriptor.instance_id,
            tool_name=descriptor.name,
            arguments=arguments,
            context=context,
            tool_call_id=runtime.tool_call_id or f"call_{uuid4().hex}",
            approval_granted=approval_required,
        )
        return result.model_dump(mode="json")

    execute.__name__ = f"execute_mcp_{descriptor.name}"
    execute.__annotations__["runtime"] = ToolRuntime[AgentRunContext]
    return StructuredTool.from_function(
        coroutine=execute,
        name=descriptor.name,
        description=descriptor.description or descriptor.policy.description,
        args_schema=dict(descriptor.input_schema),
        metadata={
            "opentulpa_capability": descriptor.capability_name,
            "opentulpa_capability_revision": descriptor.capability_revision,
            "opentulpa_capability_instance": descriptor.instance_id,
        },
    )


__all__ = ["MCPToolBundle", "build_mcp_tool_bundle"]
