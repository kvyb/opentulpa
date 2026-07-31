from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from opentulpa.application.product_tools import ProductToolApplication
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.secrets import AesGcmHostKeyCipher, SecretVault, SecretVaultService
from opentulpa.specs import (
    AgentSpecRef,
    AgentSpecService,
    AgentSpecStore,
    OriginRef,
    TriggerSpecService,
    TriggerSpecStore,
    seed_default_agent_spec_refs,
)
from opentulpa.tooling.adapters import _execute_product_tool
from opentulpa.tooling.arguments import OPERATION_ARGUMENT_SCHEMAS
from opentulpa.tooling.contract import (
    TOOL_SPEC_BY_NAME,
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
)


class _UnusedPort:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected port call: {name}")


class _LeakySecretPort:
    def list(self, *, tenant_id: str) -> list[dict[str, Any]]:
        return [
            {
                "tenant_id": tenant_id,
                "id": "leaky",
                "revision": 1,
                "name": "leaky",
                "state": "active",
                "scopes": ["api.invoke"],
                "created_at": "2026-07-20T12:00:00Z",
                "created_by": "owner-a",
                "value": "raw-port-secret",
            }
        ]


def _context(
    *,
    tenant_id: str = "tenant-a",
    run_kind: AgentRunKind = AgentRunKind.OWNER,
) -> AgentRunContext:
    is_routine = run_kind is AgentRunKind.ROUTINE
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="owner-a",
        thread_id="thread-a",
        channel=AgentChannel.ROUTINE if is_routine else AgentChannel.WEB,
        run_kind=run_kind,
        correlation_id="control-plane-test",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(
            tenant_id=tenant_id,
            spec_id="routine" if is_routine else "owner",
            revision=1,
        ),
        trust_class="background" if is_routine else "owner",
    )


def _application(
    tmp_path: Path,
    *,
    secret_handles_override: Any | None = None,
    on_trigger_spec_changed: Any | None = None,
) -> tuple[
    ProductToolApplication,
    AgentSpecService,
    TriggerSpecService,
    SecretVaultService,
]:
    agent_store = AgentSpecStore(tmp_path / "agent-specs.db")
    agent_specs = AgentSpecService(agent_store)
    trigger_specs = TriggerSpecService(
        TriggerSpecStore(tmp_path / "trigger-specs.db", agent_specs=agent_store)
    )
    secret_handles = SecretVaultService(
        SecretVault(
            tmp_path / "secrets.db",
            cipher=AesGcmHostKeyCipher(b"k" * 32),
        )
    )
    unused = _UnusedPort()
    application = ProductToolApplication(
        profiles=unused,
        files=unused,
        artifacts=unused,
        knowledge=unused,
        research=unused,
        browser=unused,
        integrations=unused,
        intake=unused,
        schedules=unused,
        jobs=unused,
        idempotency=IdempotencyStore(tmp_path / "effects.db"),
        agent_specs=agent_specs,
        trigger_specs=trigger_specs,
        secret_handles=secret_handles_override or secret_handles,
        on_trigger_spec_changed=on_trigger_spec_changed,
    )  # type: ignore[arg-type]
    seed_default_agent_spec_refs(
        agent_store,
        tenant_id="tenant-a",
        actor_id="owner-a",
    )
    return application, agent_specs, trigger_specs, secret_handles


async def _execute(
    application: ProductToolApplication,
    name: str,
    arguments: dict[str, Any],
    *,
    tenant_id: str = "tenant-a",
    run_kind: AgentRunKind = AgentRunKind.OWNER,
) -> dict[str, Any]:
    return await _execute_product_tool(
        application=application,
        spec=TOOL_SPEC_BY_NAME[name],
        context=_context(tenant_id=tenant_id, run_kind=run_kind),
        raw_arguments=arguments,
    )


def _agent_spec_payload(*, instructions: str) -> dict[str, Any]:
    return {
        "name": "Research",
        "runtime_profile": "custom",
        "instructions": instructions,
        "isolation": "private",
        "tool_policy": "allowlist",
        "tools": ["knowledge_query"],
        "memory_scope": "spec",
        "workspace_scope": "read_only",
    }


def _trigger_payload(*, instruction: str) -> dict[str, Any]:
    return {
        "name": "Morning research",
        "source": {
            "kind": "cron",
            "expression": "0 9 * * *",
            "timezone": "Europe/Moscow",
        },
        "exposure": "private",
        "agent_spec": {"spec_id": "routine", "revision": 1},
        "instruction": instruction,
        "delivery": {"mode": "owner"},
    }


@pytest.mark.asyncio
async def test_agent_spec_tools_save_activate_rollback_and_hide_host_identity(
    tmp_path: Path,
) -> None:
    application, _, _, _ = _application(tmp_path)

    listed = await _execute(application, "agent_spec_list", {})
    assert listed["status"] == "ok"
    assert {view["spec"]["id"] for view in listed["data"]} == {
        "owner",
        "routine",
        "intake",
    }
    assert "tenant_id" not in json.dumps(listed["data"])

    created = await _execute(
        application,
        "agent_spec_save",
        {
            "spec_id": "research",
            "spec": _agent_spec_payload(instructions="Research the request"),
        },
    )
    assert created["status"] == "ok"
    assert created["data"]["revision"] == 1
    assert created["idempotency_key"].startswith("derived_")

    with pytest.raises(ValidationError):
        await _execute(
            application,
            "agent_spec_activate",
            {
                "spec_id": "research",
                "revision": 1,
                "expected_active_revision": None,
            },
        )
    activated = await _execute(
        application,
        "agent_spec_activate",
        {
            "spec_id": "research",
            "revision": 1,
            "expected_active_revision": None,
            "idempotency_key": "activate-research-v1",
        },
    )
    assert activated["status"] == "ok"

    revised = await _execute(
        application,
        "agent_spec_save",
        {
            "spec_id": "research",
            "expected_revision": 1,
            "spec": _agent_spec_payload(instructions="Research carefully"),
        },
    )
    assert revised["data"]["revision"] == 2
    activated_v2 = await _execute(
        application,
        "agent_spec_activate",
        {
            "spec_id": "research",
            "revision": 2,
            "expected_active_revision": 1,
            "idempotency_key": "activate-research-v2",
        },
    )
    assert activated_v2["data"]["revision"] == 2
    rolled_back = await _execute(
        application,
        "agent_spec_rollback",
        {
            "spec_id": "research",
            "expected_active_revision": 2,
            "idempotency_key": "rollback-research-v2",
        },
    )
    assert rolled_back["data"]["revision"] == 1


@pytest.mark.asyncio
async def test_trigger_spec_tools_inject_tenant_and_manage_revisions(tmp_path: Path) -> None:
    changed_revisions: list[int] = []
    application, _, _, _ = _application(
        tmp_path,
        on_trigger_spec_changed=lambda trigger: changed_revisions.append(trigger.revision),
    )

    created = await _execute(
        application,
        "trigger_spec_save",
        {
            "trigger_id": "morning",
            "trigger": _trigger_payload(instruction="Prepare research"),
        },
    )
    assert created["status"] == "ok"
    assert created["data"]["agent_spec"] == {
        "spec_id": "routine",
        "revision": 1,
    }
    assert "tenant_id" not in json.dumps(created["data"])
    assert created["idempotency_key"].startswith("derived_")

    activated = await _execute(
        application,
        "trigger_spec_activate",
        {
            "trigger_id": "morning",
            "revision": 1,
            "expected_active_revision": None,
            "idempotency_key": "activate-trigger-v1",
        },
    )
    assert activated["status"] == "ok"
    replayed = await _execute(
        application,
        "trigger_spec_activate",
        {
            "trigger_id": "morning",
            "revision": 1,
            "expected_active_revision": None,
            "idempotency_key": "activate-trigger-v1",
        },
    )
    assert replayed["data"] == activated["data"]
    assert changed_revisions == [1]
    revised = await _execute(
        application,
        "trigger_spec_save",
        {
            "trigger_id": "morning",
            "expected_revision": 1,
            "trigger": _trigger_payload(instruction="Prepare concise research"),
        },
    )
    assert revised["data"]["revision"] == 2
    assert (
        await _execute(
            application,
            "trigger_spec_activate",
            {
                "trigger_id": "morning",
                "revision": 2,
                "expected_active_revision": 1,
                "idempotency_key": "activate-trigger-v2",
            },
        )
    )["status"] == "ok"
    rolled_back = await _execute(
        application,
        "trigger_spec_rollback",
        {
            "trigger_id": "morning",
            "expected_active_revision": 2,
            "idempotency_key": "rollback-trigger-v2",
        },
    )
    assert rolled_back["data"]["revision"] == 1
    assert changed_revisions == [1, 2, 1]
    listed = await _execute(application, "trigger_spec_list", {})
    assert listed["data"][0]["active_revision"] == 1

    invalid = _trigger_payload(instruction="Try tenant injection")
    invalid["agent_spec"]["tenant_id"] = "tenant-b"
    with pytest.raises(ValidationError):
        OPERATION_ARGUMENT_SCHEMAS["trigger_spec_save"].model_validate(
            {"trigger_id": "invalid", "trigger": invalid}
        )


@pytest.mark.asyncio
async def test_secret_tools_only_list_and_revoke_safe_handles(tmp_path: Path) -> None:
    application, _, _, secrets = _application(tmp_path)
    plaintext = "must-never-reach-the-agent"
    pending = secrets.create_pending(
        tenant_id="tenant-a",
        actor_id="owner-a",
        secret_id="telegram_bot",
        name="telegram_bot_token",
        scopes=("telegram.send",),
    )
    secrets.store(
        tenant_id="tenant-a",
        actor_id="owner-a",
        secret_id=pending.id,
        expected_revision=1,
        value=SecretStr(plaintext),
    )
    foreign = secrets.create_pending(
        tenant_id="tenant-b",
        actor_id="owner-b",
        secret_id="foreign_token",
        name="foreign_token",
        scopes=("api.invoke",),
    )
    secrets.store(
        tenant_id="tenant-b",
        actor_id="owner-b",
        secret_id=foreign.id,
        expected_revision=1,
        value=SecretStr("foreign-plaintext"),
    )

    listed = await _execute(application, "secret_handle_list", {})
    serialized = json.dumps(listed)
    assert listed["status"] == "ok"
    assert listed["data"] == [
        {
            "id": "telegram_bot",
            "revision": 2,
            "name": "telegram_bot_token",
            "state": "active",
            "scopes": ["telegram.send"],
            "created_at": listed["data"][0]["created_at"],
            "created_by": "owner-a",
        }
    ]
    assert plaintext not in serialized
    assert "foreign_token" not in serialized
    assert "tenant_id" not in serialized
    assert "ciphertext" not in serialized

    with pytest.raises(ValidationError):
        await _execute(
            application,
            "secret_handle_revoke",
            {"secret_id": "telegram_bot", "expected_revision": 2},
        )
    revoked = await _execute(
        application,
        "secret_handle_revoke",
        {
            "secret_id": "telegram_bot",
            "expected_revision": 2,
            "idempotency_key": "revoke-telegram",
        },
    )
    assert revoked["status"] == "ok"
    assert revoked["data"]["state"] == "revoked"
    assert plaintext not in json.dumps(revoked)


def test_secret_tool_schemas_have_no_plaintext_ingress() -> None:
    for name in ("secret_handle_list", "secret_handle_revoke", "sandbox_ssh_diagnostic"):
        schema = OPERATION_ARGUMENT_SCHEMAS[name].model_json_schema()
        serialized = json.dumps(schema).lower()
        assert '"value"' not in serialized
        assert "plaintext" not in serialized
        assert "ciphertext" not in serialized
        assert '"token"' not in serialized
        assert "tenant_id" not in serialized

    ssh = OPERATION_ARGUMENT_SCHEMAS["sandbox_ssh_diagnostic"].model_validate(
        {
            "secret_id": "ssh_password",
            "host": "13928983",
            "command": "uptime",
            "secret_type": "password",
        }
    )
    assert ssh.secret_type == "password"


@pytest.mark.asyncio
async def test_control_tools_are_owner_only_and_fail_closed_on_leaky_secret_ports(
    tmp_path: Path,
) -> None:
    application, _, _, _ = _application(
        tmp_path,
        secret_handles_override=_LeakySecretPort(),
    )

    restricted = await _execute(
        application,
        "agent_spec_list",
        {},
        run_kind=AgentRunKind.ROUTINE,
    )
    assert restricted["status"] == "error"
    assert restricted["error"]["code"] == "capability_unavailable"

    leaked = await _execute(application, "secret_handle_list", {})
    assert leaked["status"] == "error"
    assert leaked["error"]["code"] == "invalid_service_response"
    assert "raw-port-secret" not in json.dumps(leaked)
