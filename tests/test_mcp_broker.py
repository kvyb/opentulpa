import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

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
    tool_schema_digest,
)
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling import (
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
    ApprovalMode,
    IdempotencyMode,
    ToolEffect,
)

READ_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "api_token": {"type": "string"},
    },
    "required": ["query"],
    "additionalProperties": False,
}
SEND_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}


class FakeAdapter:
    def __init__(self, tools: Sequence[MCPRemoteTool], result: Any = None) -> None:
        self.tools = tuple(tools)
        self.result = result if result is not None else {"ok": True}
        self.calls: list[tuple[str, dict[str, Any], MCPCallMetadata]] = []

    async def discover(self, worker: WorkerSpec) -> Sequence[MCPRemoteTool]:
        return self.tools

    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any:
        self.calls.append((tool_name, dict(arguments), metadata))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class SlowAdapter(FakeAdapter):
    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any:
        await asyncio.sleep(0.01)
        return await super().invoke(worker, tool_name, arguments, metadata)


def _remote(name: str, schema: dict[str, Any]) -> MCPRemoteTool:
    return MCPRemoteTool(
        name=name,
        description=f"Remote {name}",
        input_schema=schema,
        schema_digest=tool_schema_digest(schema),
    )


def _manifest(*, remote_digest_override: str | None = None) -> CapabilityManifest:
    read_digest = remote_digest_override or tool_schema_digest(READ_SCHEMA)
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
                        schema_digest=read_digest,
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


@pytest.mark.asyncio
async def test_discovery_requires_exact_names_and_schema_digests() -> None:
    broker = MCPToolBroker()
    missing = FakeAdapter((_remote("lookup", READ_SCHEMA),))
    with pytest.raises(MCPBrokerRegistrationError, match="exactly match"):
        await broker.register(
            instance_id="instance-a",
            manifest=_manifest(),
            worker_name="example_mcp",
            adapter=missing,
        )

    mismatch = FakeAdapter((_remote("lookup", READ_SCHEMA), _remote("send", SEND_SCHEMA)))
    with pytest.raises(MCPBrokerRegistrationError, match="digest"):
        await broker.register(
            instance_id="instance-a",
            manifest=_manifest(remote_digest_override=f"sha256:{'0' * 64}"),
            worker_name="example_mcp",
            adapter=mismatch,
        )
    assert broker.descriptors("instance-a") == ()


@pytest.mark.asyncio
async def test_broker_injects_hidden_context_and_redacts_results_and_audit() -> None:
    audit = InMemoryMCPAuditSink()
    adapter = FakeAdapter(
        (_remote("lookup", READ_SCHEMA), _remote("send", SEND_SCHEMA)),
        result={"api_token": "server-secret", "value": 42},
    )
    broker = MCPToolBroker(audit_sink=audit)
    await broker.register(
        instance_id="instance-a",
        manifest=_manifest(),
        worker_name="example_mcp",
        adapter=adapter,
    )

    result = await broker.invoke(
        instance_id="instance-a",
        tool_name="lookup",
        arguments={"query": "hello", "api_token": "caller-secret"},
        context=_context(),
        tool_call_id="call-a",
    )

    assert result.status == "ok"
    assert result.data == {"api_token": "[redacted]", "value": 42}
    assert adapter.calls[0][1] == {"query": "hello", "api_token": "caller-secret"}
    metadata = adapter.calls[0][2]
    assert metadata.tenant_id == "tenant-a"
    assert "tenant_id" not in adapter.calls[0][1]
    assert [event.outcome for event in audit.events] == ["started", "ok"]
    assert audit.events[0].arguments["api_token"] == "[redacted]"


@pytest.mark.asyncio
async def test_side_effects_require_approval_and_replay_by_hidden_idempotency() -> None:
    adapter = FakeAdapter((_remote("lookup", READ_SCHEMA), _remote("send", SEND_SCHEMA)))
    broker = MCPToolBroker()
    await broker.register(
        instance_id="instance-a",
        manifest=_manifest(),
        worker_name="example_mcp",
        adapter=adapter,
    )

    interrupted = await broker.invoke(
        instance_id="instance-a",
        tool_name="send",
        arguments={"message": "hello"},
        context=_context(),
        tool_call_id="call-send",
    )
    first = await broker.invoke(
        instance_id="instance-a",
        tool_name="send",
        arguments={"message": "hello"},
        context=_context(),
        tool_call_id="call-send",
        approval_granted=True,
    )
    replay = await broker.invoke(
        instance_id="instance-a",
        tool_name="send",
        arguments={"message": "hello"},
        context=_context(),
        tool_call_id="call-send",
        approval_granted=True,
    )

    assert interrupted.status == "approval_required"
    assert interrupted.idempotency_key == first.idempotency_key == replay.idempotency_key
    assert first.status == replay.status == "ok"
    assert replay.replayed is True
    assert len(adapter.calls) == 1
    assert broker.interrupt_on("instance-a") == {"lookup": False, "send": True}


@pytest.mark.asyncio
async def test_concurrent_retries_execute_external_effect_once() -> None:
    adapter = SlowAdapter((_remote("lookup", READ_SCHEMA), _remote("send", SEND_SCHEMA)))
    broker = MCPToolBroker()
    await broker.register(
        instance_id="instance-a",
        manifest=_manifest(),
        worker_name="example_mcp",
        adapter=adapter,
    )

    async def send() -> Any:
        return await broker.invoke(
            instance_id="instance-a",
            tool_name="send",
            arguments={"message": "hello"},
            context=_context(),
            tool_call_id="same-call",
            approval_granted=True,
        )

    first, second = await asyncio.gather(send(), send())

    assert len(adapter.calls) == 1
    assert sorted((first.replayed, second.replayed)) == [False, True]


@pytest.mark.asyncio
async def test_model_cannot_override_context_or_receive_raw_adapter_errors() -> None:
    adapter = FakeAdapter(
        (_remote("lookup", READ_SCHEMA), _remote("send", SEND_SCHEMA)),
        result=RuntimeError("token=super-secret"),
    )
    broker = MCPToolBroker()
    await broker.register(
        instance_id="instance-a",
        manifest=_manifest(),
        worker_name="example_mcp",
        adapter=adapter,
    )

    reserved = await broker.invoke(
        instance_id="instance-a",
        tool_name="lookup",
        arguments={"query": "hello", "tenant_id": "other"},
        context=_context(),
        tool_call_id="call-reserved",
    )
    failed = await broker.invoke(
        instance_id="instance-a",
        tool_name="lookup",
        arguments={"query": "hello"},
        context=_context(),
        tool_call_id="call-failed",
    )

    assert reserved.error is not None and reserved.error.code == "reserved_argument"
    assert failed.error is not None and failed.error.code == "operation_failed"
    assert "secret" not in failed.error.message
    assert len(adapter.calls) == 1
