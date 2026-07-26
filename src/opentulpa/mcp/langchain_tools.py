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
    """Tools and their effective Deep Agents interrupt policy."""

    tools: tuple[BaseTool, ...]
    interrupt_on: Mapping[str, bool]


def build_mcp_tool_bundle(
    broker: MCPToolBroker,
    *,
    instance_id: str,
) -> MCPToolBundle:
    """Expose registered MCP tools with runtime context hidden from model schemas.

    OpenTulpa owner and background agents execute capability tools without per-call
    approval. Manifest policy remains available to direct broker callers.
    """

    descriptors = broker.descriptors(instance_id)
    if not descriptors:
        raise ValueError("MCP capability instance has no registered tools")
    names = [descriptor.name for descriptor in descriptors]
    if len(names) != len(set(names)):
        raise ValueError("MCP capability instance contains duplicate tool names")
    interrupt_on = MappingProxyType(dict.fromkeys(names, False))
    tools = tuple(
        _build_broker_tool(
            broker=broker,
            descriptor=descriptor,
        )
        for descriptor in descriptors
    )
    return MCPToolBundle(tools=tools, interrupt_on=interrupt_on)


def _build_broker_tool(
    *,
    broker: MCPToolBroker,
    descriptor: MCPToolDescriptor,
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
            approval_granted=False,
            approval_enforced=False,
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
