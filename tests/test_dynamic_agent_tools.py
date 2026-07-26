from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.tools import Tool
from pydantic import ValidationError

import opentulpa.deep_agent.service as service_module
from opentulpa.deep_agent import AgentRunContext, DeepAgentService, TenantDynamicToolRegistry
from opentulpa.specs import AgentSpecStore, AgentSpecWrite, OriginRef


def _tool(name: str) -> Tool:
    return Tool(name=name, description=f"Dynamic {name}", func=lambda value: value)


def _service(
    root: Path,
    *,
    specs: AgentSpecStore,
    dynamic: TenantDynamicToolRegistry,
) -> DeepAgentService:
    return DeepAgentService(
        api_key="",
        base_url="",
        model_name="test-model",
        checkpoint_db_path=root / "checkpoints.db",
        store_db_path=root / "store.db",
        runs_db_path=root / "runs.db",
        workspaces_root=root / "workspaces",
        model=object(),
        tools=(_tool("profile_get"),),
        agent_specs=specs,
        dynamic_tools=dynamic,
    )


def test_dynamic_tool_registry_is_tenant_scoped_and_rejects_collisions() -> None:
    registry = TenantDynamicToolRegistry(reserved_names=("profile_get",))
    registry.register(
        tenant_id="tenant-a",
        instance_id="weather-v1",
        tools=(_tool("weather_lookup"),),
        interrupt_on={"weather_lookup": False},
    )

    assert [tool.name for tool in registry.snapshot("tenant-a").tools] == [
        "weather_lookup"
    ]
    assert registry.snapshot("tenant-b").tools == ()
    with pytest.raises(ValueError, match="kernel tools"):
        registry.register(
            tenant_id="tenant-a",
            instance_id="bad",
            tools=(_tool("profile_get"),),
            interrupt_on={"profile_get": False},
        )
    with pytest.raises(ValueError, match="across capabilities"):
        registry.register(
            tenant_id="tenant-a",
            instance_id="duplicate",
            tools=(_tool("weather_lookup"),),
            interrupt_on={"weather_lookup": False},
        )


def test_active_dynamic_generation_recompiles_owner_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = AgentSpecStore(tmp_path / "specs.db")
    owner = specs.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=AgentSpecWrite(
            name="Owner",
            runtime_profile="owner",
            instructions="Help the owner.",
            tool_policy="profile_default",
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=None,
        created_by="owner",
    )
    dynamic = TenantDynamicToolRegistry(reserved_names=("profile_get",))
    dynamic.register(
        tenant_id="tenant-a",
        instance_id="alerts-v1",
        tools=(_tool("send_alert"),),
        interrupt_on={"send_alert": True},
    )
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(service_module, "create_deep_agent", capture)
    service = _service(tmp_path, specs=specs, dynamic=dynamic)
    service._checkpointer = cast("Any", object())
    service._store = cast("Any", object())
    context = AgentRunContext(
        tenant_id="tenant-a",
        actor_id="owner",
        thread_id="thread-1",
        channel="web",
        run_kind="owner",
        correlation_id="correlation-1",
        origin=OriginRef(interface="web", source_id="owner-web"),
        agent_spec=owner.ref,
        trust_class="owner",
    )

    first = service._graph_for_context(context)
    assert service._graph_for_context(context) is first
    assert {tool.name for tool in calls[0]["tools"]} == {"profile_get", "send_alert"}
    assert calls[0]["interrupt_on"] is None

    dynamic.unregister(tenant_id="tenant-a", instance_id="alerts-v1")
    second = service._graph_for_context(context)
    assert second is not first
    assert [tool.name for tool in calls[1]["tools"]] == ["profile_get"]


def test_private_background_spec_can_allowlist_dynamic_tool_without_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = AgentSpecStore(tmp_path / "specs.db")
    background = specs.create_revision(
        tenant_id="tenant-a",
        spec_id="alert-routine",
        write=AgentSpecWrite(
            name="Alert routine",
            runtime_profile="routine",
            instructions="Send the scheduled alert.",
            tools=("send_alert",),
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=None,
        created_by="owner",
    )
    dynamic = TenantDynamicToolRegistry(reserved_names=("profile_get",))
    dynamic.register(
        tenant_id="tenant-a",
        instance_id="alerts-v1",
        tools=(_tool("send_alert"),),
        interrupt_on={"send_alert": False},
    )
    calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(service_module, "create_deep_agent", capture)
    service = _service(tmp_path, specs=specs, dynamic=dynamic)
    service._checkpointer = cast("Any", object())
    service._store = cast("Any", object())
    context = AgentRunContext(
        tenant_id="tenant-a",
        actor_id="scheduler",
        thread_id="trigger:alert:fire-1",
        channel="routine",
        run_kind="routine",
        correlation_id="trigger:alert:fire-1",
        origin=OriginRef(interface="trigger", source_id="alert"),
        agent_spec=background.ref,
        trust_class="background",
    )

    service._graph_for_context(context)

    assert [tool.name for tool in calls[0]["tools"]] == ["send_alert"]
    assert calls[0]["interrupt_on"] is None


def test_external_agent_cannot_configure_dynamic_capability_tool() -> None:
    with pytest.raises(ValidationError, match="non-knowledge tools: send_alert"):
        AgentSpecWrite(
            name="External",
            runtime_profile="intake",
            instructions="Classify one message.",
            isolation="external",
            tools=("send_alert",),
            memory_scope="none",
            workspace_scope="none",
        )
