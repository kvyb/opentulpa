from __future__ import annotations

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
from opentulpa.deep_agent import TenantDynamicToolRegistry
from opentulpa.mcp import MCPCallMetadata, MCPRemoteTool, MCPToolBroker, MCPToolRuntime
from opentulpa.mcp.schema import tool_schema_digest
from opentulpa.tooling import (
    ApprovalMode,
    IdempotencyMode,
    ToolEffect,
)

SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
    "required": ["city"],
    "additionalProperties": False,
}


class _Adapter:
    async def discover(self, worker: WorkerSpec) -> Sequence[MCPRemoteTool]:
        export = worker.tools[0]
        return (
            MCPRemoteTool(
                name=export.name,
                description=export.description,
                input_schema=SCHEMA,
                schema_digest=tool_schema_digest(SCHEMA),
            ),
        )

    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any:
        del worker, tool_name, metadata
        return {"city": arguments["city"]}


def _manifest(
    *,
    name: str = "weather",
    revision: int = 1,
    tool_name: str = "weather_lookup",
) -> CapabilityManifest:
    return CapabilityManifest(
        name=name,
        version=f"1.{revision - 1}.0",
        revision=revision,
        workers=(
            WorkerSpec(
                name=f"{name}_mcp",
                kind=WorkerKind.MCP,
                protocol="mcp-v1",
                command=("weather-server",),
                tools=(
                    ToolExport(
                        name=tool_name,
                        description="Read weather",
                        schema_digest=tool_schema_digest(SCHEMA),
                        effect=ToolEffect.READ,
                        approval=ApprovalMode.AUTO,
                        idempotency=IdempotencyMode.NONE,
                    ),
                ),
            ),
        ),
        eval_commands=(EvalCommand(argv=("pytest", "-q")),),
    )


@pytest.mark.asyncio
async def test_mcp_runtime_publishes_and_removes_atomic_tenant_bundle() -> None:
    registry = TenantDynamicToolRegistry(reserved_names=("profile_get",))
    runtime = MCPToolRuntime(
        broker=MCPToolBroker(),
        tools=registry,
        adapter_factory=lambda worker, config, secrets: _Adapter(),
    )

    await runtime.start(
        tenant_id="tenant-a",
        instance_id="weather-g1",
        manifest=_manifest(),
        config={},
        secrets={},
        worker_endpoints={},
    )

    snapshot = registry.snapshot("tenant-a")
    assert [tool.name for tool in snapshot.tools] == ["weather_lookup"]
    assert dict(snapshot.interrupt_on) == {"weather_lookup": False}
    assert registry.snapshot("tenant-b").tools == ()

    await runtime.stop("weather-g1")
    assert registry.snapshot("tenant-a").tools == ()


@pytest.mark.asyncio
async def test_mcp_runtime_passes_full_hidden_environment_to_stdio_adapter() -> None:
    captured: dict[str, Any] = {}

    def adapter_factory(
        worker: WorkerSpec,
        config: Mapping[str, object],
        environment: Mapping[str, str],
    ) -> _Adapter:
        captured.update(
            worker=worker,
            config=dict(config),
            environment=dict(environment),
        )
        return _Adapter()

    runtime = MCPToolRuntime(
        broker=MCPToolBroker(),
        tools=TenantDynamicToolRegistry(reserved_names=("profile_get",)),
        adapter_factory=adapter_factory,
    )

    await runtime.start(
        tenant_id="tenant-a",
        instance_id="weather-g7",
        manifest=_manifest(),
        config={"units": "metric"},
        secrets={"WEATHER_TOKEN": "secret"},
        worker_endpoints={},
    )

    assert captured["config"] == {"units": "metric"}
    assert captured["environment"] == {
        "WEATHER_TOKEN": "secret",
        "OPENTULPA_CAPABILITY_CONFIG": '{"units":"metric"}',
        "OPENTULPA_CAPABILITY_INSTANCE_ID": "weather-g7",
        "OPENTULPA_CAPABILITY_NAME": "weather",
        "OPENTULPA_CAPABILITY_REVISION": "1",
        "OPENTULPA_WORKER_NAME": "weather_mcp",
    }


@pytest.mark.asyncio
async def test_mcp_runtime_atomically_replaces_same_capability_tool_generation() -> None:
    registry = TenantDynamicToolRegistry()
    runtime = MCPToolRuntime(
        broker=MCPToolBroker(),
        tools=registry,
        adapter_factory=lambda worker, config, secrets: _Adapter(),
    )
    await runtime.start(
        tenant_id="tenant-a",
        instance_id="weather-g1",
        manifest=_manifest(revision=1),
        config={},
        secrets={},
        worker_endpoints={},
    )

    await runtime.replace(
        tenant_id="tenant-a",
        previous_instance_id="weather-g1",
        instance_id="weather-g2",
        manifest=_manifest(revision=2),
        config={},
        secrets={},
        worker_endpoints={},
    )

    snapshot = registry.snapshot("tenant-a")
    assert [tool.name for tool in snapshot.tools] == ["weather_lookup"]
    assert dict(snapshot.interrupt_on) == {"weather_lookup": False}
    assert snapshot.generation == 2
    await runtime.stop("weather-g2")
    assert registry.snapshot("tenant-a").tools == ()


@pytest.mark.asyncio
async def test_mcp_runtime_namespaces_intentional_kernel_tool_replacement() -> None:
    registry = TenantDynamicToolRegistry(reserved_names=("browser_start",))
    runtime = MCPToolRuntime(
        broker=MCPToolBroker(),
        tools=registry,
        adapter_factory=lambda worker, config, secrets: _Adapter(),
    )

    await runtime.start(
        tenant_id="tenant-a",
        instance_id="browser-g1",
        manifest=_manifest(name="browser", tool_name="browser_start"),
        config={},
        secrets={},
        worker_endpoints={},
    )

    snapshot = registry.snapshot("tenant-a")
    assert [tool.name for tool in snapshot.tools] == ["browser__browser_start"]
    assert dict(snapshot.interrupt_on) == {"browser__browser_start": False}
