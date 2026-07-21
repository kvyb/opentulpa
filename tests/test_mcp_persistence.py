from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
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
    MCPCallMetadata,
    MCPRemoteTool,
    MCPToolBroker,
    SQLiteMCPAuditSink,
    SQLiteMCPIdempotencyStore,
    tool_schema_digest,
)
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling import (
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
    ToolEffect,
)

TOOL_SCHEMA = {
    "type": "object",
    "properties": {"message": {"type": "string"}},
    "required": ["message"],
    "additionalProperties": False,
}


class RecordingAdapter:
    def __init__(self, tools: Sequence[MCPRemoteTool], *, marker: str) -> None:
        self.tools = tuple(tools)
        self.marker = marker
        self.calls: list[tuple[str, MCPCallMetadata]] = []

    async def discover(self, worker: WorkerSpec) -> Sequence[MCPRemoteTool]:
        return self.tools

    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any:
        self.calls.append((tool_name, metadata))
        return {"marker": self.marker, "message": arguments["message"]}


def _remote(name: str) -> MCPRemoteTool:
    return MCPRemoteTool(
        name=name,
        input_schema=TOOL_SCHEMA,
        schema_digest=tool_schema_digest(TOOL_SCHEMA),
    )


def _manifest(name: str, *tool_names: str) -> CapabilityManifest:
    return CapabilityManifest(
        name=name,
        version="1.0.0",
        workers=(
            WorkerSpec(
                name=f"{name}_mcp",
                kind=WorkerKind.MCP,
                protocol="mcp-v1",
                command=(f"{name}-mcp",),
                tools=tuple(
                    ToolExport(
                        name=tool_name,
                        description=f"Run {tool_name}.",
                        schema_digest=tool_schema_digest(TOOL_SCHEMA),
                        effect=ToolEffect.SEND,
                    )
                    for tool_name in tool_names
                ),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


def _context(tenant_id: str) -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id=f"owner-{tenant_id}",
        thread_id=f"thread-{tenant_id}",
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id=f"correlation-{tenant_id}",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
        trust_class="owner",
    )


async def _register(
    broker: MCPToolBroker,
    *,
    instance_id: str,
    manifest: CapabilityManifest,
    adapter: RecordingAdapter,
) -> None:
    await broker.register(
        instance_id=instance_id,
        manifest=manifest,
        worker_name=manifest.workers[0].name,
        adapter=adapter,
    )


@pytest.mark.asyncio
async def test_sqlite_idempotency_and_audit_survive_broker_restart(tmp_path: Path) -> None:
    db_path = tmp_path / "mcp.db"
    manifest = _manifest("alpha", "send")
    first_adapter = RecordingAdapter((_remote("send"),), marker="first")
    first_broker = MCPToolBroker(
        audit_sink=SQLiteMCPAuditSink(db_path),
        idempotency_store=SQLiteMCPIdempotencyStore(db_path),
    )
    await _register(
        first_broker,
        instance_id="alpha-instance",
        manifest=manifest,
        adapter=first_adapter,
    )

    first = await first_broker.invoke(
        instance_id="alpha-instance",
        tool_name="send",
        arguments={"message": "hello"},
        context=_context("tenant-a"),
        tool_call_id="call-before-restart",
        approval_granted=True,
        idempotency_key="owner-request-1",
    )

    restarted_audit = SQLiteMCPAuditSink(db_path)
    restarted_adapter = RecordingAdapter((_remote("send"),), marker="second")
    restarted_broker = MCPToolBroker(
        audit_sink=restarted_audit,
        idempotency_store=SQLiteMCPIdempotencyStore(db_path),
    )
    await _register(
        restarted_broker,
        instance_id="alpha-instance",
        manifest=manifest,
        adapter=restarted_adapter,
    )
    replay = await restarted_broker.invoke(
        instance_id="alpha-instance",
        tool_name="send",
        arguments={"message": "hello"},
        context=_context("tenant-a"),
        tool_call_id="call-after-restart",
        approval_granted=True,
        idempotency_key="owner-request-1",
    )

    assert first.status == replay.status == "ok"
    assert replay.data == {"marker": "first", "message": "hello"}
    assert replay.replayed is True
    assert first.idempotency_key == replay.idempotency_key
    assert first.idempotency_key != "owner-request-1"
    assert len(first_adapter.calls) == 1
    assert restarted_adapter.calls == []
    assert [event.outcome for event in await restarted_audit.list_events()] == [
        "started",
        "ok",
        "replayed",
    ]
    assert db_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_supplied_keys_are_scoped_by_tenant_capability_and_tool() -> None:
    alpha_manifest = _manifest("alpha", "publish", "send")
    beta_manifest = _manifest("beta", "send")
    alpha_adapter = RecordingAdapter(
        (_remote("publish"), _remote("send")),
        marker="alpha",
    )
    beta_adapter = RecordingAdapter((_remote("send"),), marker="beta")
    broker = MCPToolBroker()
    await _register(
        broker,
        instance_id="alpha-instance",
        manifest=alpha_manifest,
        adapter=alpha_adapter,
    )
    await _register(
        broker,
        instance_id="beta-instance",
        manifest=beta_manifest,
        adapter=beta_adapter,
    )

    calls = (
        ("alpha-instance", "send", _context("tenant-a"), "call-a"),
        ("alpha-instance", "send", _context("tenant-b"), "call-b"),
        ("alpha-instance", "publish", _context("tenant-a"), "call-c"),
        ("beta-instance", "send", _context("tenant-a"), "call-d"),
    )
    results = []
    for instance_id, tool_name, context, tool_call_id in calls:
        results.append(
            await broker.invoke(
                instance_id=instance_id,
                tool_name=tool_name,
                arguments={"message": "hello"},
                context=context,
                tool_call_id=tool_call_id,
                approval_granted=True,
                idempotency_key="shared-owner-key",
            )
        )

    keys = {result.idempotency_key for result in results}
    assert len(keys) == len(calls)
    assert None not in keys
    assert len(alpha_adapter.calls) == 3
    assert len(beta_adapter.calls) == 1
