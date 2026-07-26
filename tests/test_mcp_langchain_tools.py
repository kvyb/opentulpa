from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from langchain.tools import ToolRuntime

from opentulpa.capabilities import (
    CapabilityManifest,
    EvalCommand,
    ToolExport,
    WorkerKind,
    WorkerSpec,
)
from opentulpa.mcp import (
    InMemoryMCPAuditSink,
    MCPBrokerRegistrationError,
    MCPCallMetadata,
    MCPRemoteTool,
    MCPToolBroker,
    build_mcp_tool_bundle,
    tool_schema_digest,
)
from opentulpa.specs import AgentRunContext, AgentSpecRef, OriginRef
from opentulpa.tooling import (
    AgentChannel,
    AgentRunKind,
    ApprovalMode,
    IdempotencyMode,
    ToolEffect,
)

READ_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
    "additionalProperties": False,
}
SEND_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}


class _Adapter:
    def __init__(self, tools: Sequence[MCPRemoteTool]) -> None:
        self.tools = tuple(tools)
        self.calls: list[tuple[str, dict[str, Any], MCPCallMetadata]] = []

    async def discover(self, worker: WorkerSpec) -> Sequence[MCPRemoteTool]:
        del worker
        return self.tools

    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any:
        del worker
        self.calls.append((tool_name, dict(arguments), metadata))
        return {"received": dict(arguments)}


def _remote(name: str, schema: dict[str, Any]) -> MCPRemoteTool:
    return MCPRemoteTool(
        name=name,
        input_schema=schema,
        schema_digest=tool_schema_digest(schema),
    )


def _manifest() -> CapabilityManifest:
    return CapabilityManifest(
        name="example",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="example_mcp",
                kind=WorkerKind.MCP,
                protocol="mcp-v1",
                command=("example-mcp",),
                tools=(
                    ToolExport(
                        name="lookup",
                        description="Look up data.",
                        schema_digest=tool_schema_digest(READ_SCHEMA),
                        effect=ToolEffect.READ,
                        approval=ApprovalMode.AUTO,
                        idempotency=IdempotencyMode.NONE,
                    ),
                    ToolExport(
                        name="send",
                        description="Send data.",
                        schema_digest=tool_schema_digest(SEND_SCHEMA),
                        effect=ToolEffect.SEND,
                    ),
                ),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


def _context() -> AgentRunContext:
    return AgentRunContext(
        tenant_id="tenant-a",
        actor_id="owner-a",
        thread_id="thread-a",
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id="correlation-a",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id="tenant-a", spec_id="owner", revision=1),
        trust_class="owner",
    )


def _runtime(tool_call_id: str) -> ToolRuntime[AgentRunContext]:
    return ToolRuntime(
        state={},
        context=_context(),
        config={},
        stream_writer=lambda _: None,
        tool_call_id=tool_call_id,
        store=None,
    )


@pytest.mark.asyncio
async def test_bundle_hides_context_and_executes_through_broker_policy() -> None:
    adapter = _Adapter((_remote("lookup", READ_SCHEMA), _remote("send", SEND_SCHEMA)))
    audit = InMemoryMCPAuditSink()
    broker = MCPToolBroker(audit_sink=audit)
    await broker.register(
        instance_id="instance-a",
        manifest=_manifest(),
        worker_name="example_mcp",
        adapter=adapter,
    )
    bundle = build_mcp_tool_bundle(broker, instance_id="instance-a")

    assert bundle.interrupt_on == {"lookup": False, "send": False}
    assert [tool.name for tool in bundle.tools] == ["lookup", "send"]
    lookup_schema = bundle.tools[0].tool_call_schema
    assert isinstance(lookup_schema, dict)
    assert "runtime" not in lookup_schema.get("properties", {})
    assert "tenant_id" not in lookup_schema.get("properties", {})

    result = await bundle.tools[1].ainvoke(
        {"message": "hello", "runtime": _runtime("call-send")}
    )

    assert result["status"] == "ok"
    assert adapter.calls[0][0:2] == ("send", {"message": "hello"})
    assert adapter.calls[0][2].tenant_id == "tenant-a"
    assert adapter.calls[0][2].tool_call_id == "call-send"
    assert [event.approval_granted for event in audit.events] == [False, False]


@pytest.mark.asyncio
async def test_registration_rejects_model_visible_runtime_fields() -> None:
    unsafe_schema = {
        "type": "object",
        "properties": {"tenant_id": {"type": "string"}},
        "additionalProperties": False,
    }
    manifest = CapabilityManifest(
        name="unsafe",
        version="1.0.0",
        workers=(
            WorkerSpec(
                name="unsafe_mcp",
                kind=WorkerKind.MCP,
                protocol="mcp-v1",
                command=("unsafe-mcp",),
                tools=(
                    ToolExport(
                        name="unsafe",
                        description="Unsafe.",
                        schema_digest=tool_schema_digest(unsafe_schema),
                        effect=ToolEffect.READ,
                        approval=ApprovalMode.AUTO,
                        idempotency=IdempotencyMode.NONE,
                    ),
                ),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )
    broker = MCPToolBroker()

    with pytest.raises(MCPBrokerRegistrationError, match="reserved runtime"):
        await broker.register(
            instance_id="instance-a",
            manifest=manifest,
            worker_name="unsafe_mcp",
            adapter=_Adapter((_remote("unsafe", unsafe_schema),)),
        )


def test_bundle_refuses_unregistered_instance() -> None:
    with pytest.raises(ValueError, match="no registered tools"):
        build_mcp_tool_bundle(MCPToolBroker(), instance_id="missing")
