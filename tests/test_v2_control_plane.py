from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_control_plane import register_v2_control_plane_routes
from opentulpa.api.routes.v2_principal import V2Principal
from opentulpa.secrets import AesGcmHostKeyCipher, SecretVault, SecretVaultService
from opentulpa.specs import (
    AgentSpecService,
    AgentSpecStore,
    TriggerSpecService,
    TriggerSpecStore,
    seed_default_agent_spec_refs,
)

TENANT_A = {"x-tenant-id": "tenant-a", "x-actor-id": "owner-a"}
TENANT_B = {"x-tenant-id": "tenant-b", "x-actor-id": "owner-b"}


@dataclass
class _Principal:
    tenant_id: str
    actor_id: str


@dataclass(frozen=True)
class _ControlPlane:
    client: TestClient
    agent_store: AgentSpecStore
    trigger_store: TriggerSpecStore
    vault: SecretVault


def _control_plane(tmp_path: Path) -> _ControlPlane:
    agent_store = AgentSpecStore(tmp_path / "agent-specs.db")
    trigger_store = TriggerSpecStore(
        tmp_path / "trigger-specs.db",
        agent_specs=agent_store,
    )
    vault = SecretVault(
        tmp_path / "secrets.db",
        cipher=AesGcmHostKeyCipher(b"k" * 32),
    )
    agent_specs = AgentSpecService(agent_store)
    trigger_specs = TriggerSpecService(trigger_store)
    secrets = SecretVaultService(vault)
    app = FastAPI()

    async def principal(request: Request) -> V2Principal:
        return _Principal(
            tenant_id=request.headers.get("x-tenant-id", ""),
            actor_id=request.headers.get("x-actor-id", ""),
        )

    register_v2_control_plane_routes(
        app,
        get_agent_specs=lambda: agent_specs,
        get_trigger_specs=lambda: trigger_specs,
        get_secret_vault=lambda: secrets,
        resolve_principal=principal,
    )
    return _ControlPlane(TestClient(app), agent_store, trigger_store, vault)


def _agent_write(*, instructions: str) -> dict[str, Any]:
    return {
        "name": "Research assistant",
        "runtime_profile": "custom",
        "instructions": instructions,
        "isolation": "private",
        "tool_policy": "allowlist",
        "tools": ["knowledge_query"],
        "memory_scope": "spec",
        "workspace_scope": "read_only",
    }


def _trigger_write(*, instruction: str = "Prepare the brief") -> dict[str, Any]:
    return {
        "name": "Morning brief",
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


def test_agent_spec_crud_activation_rollback_and_tenant_isolation(tmp_path: Path) -> None:
    control = _control_plane(tmp_path)
    client = control.client

    seeded = client.post("/v2/agent-specs/seed-defaults", headers=TENANT_A)
    assert seeded.status_code == 201
    assert {spec["id"] for spec in seeded.json()["specs"]} == {
        "owner",
        "release-repair",
        "routine",
        "intake",
    }
    assert client.post("/v2/agent-specs/seed-defaults", headers=TENANT_A).status_code == 201
    refs = seed_default_agent_spec_refs(
        control.agent_store,
        tenant_id="tenant-a",
        actor_id="owner-a",
    )
    assert {profile: ref.spec_id for profile, ref in refs.items()} == {
        "owner": "owner",
        "release-repair": "release-repair",
        "routine": "routine",
        "intake": "intake",
    }

    created = client.post(
        "/v2/agent-specs",
        headers=TENANT_A,
        json={"id": "research", "spec": _agent_write(instructions="Research this")},
    )
    assert created.status_code == 201
    assert created.json()["spec"]["revision"] == 1
    assert client.get("/v2/agent-specs/research", headers=TENANT_A).status_code == 404

    activated = client.post(
        "/v2/agent-specs/research/activate",
        headers=TENANT_A,
        json={"revision": 1, "expected_active_revision": None},
    )
    assert activated.status_code == 200
    assert activated.json()["spec"]["revision"] == 1

    revised = client.post(
        "/v2/agent-specs",
        headers=TENANT_A,
        json={
            "id": "research",
            "expected_revision": 1,
            "spec": _agent_write(instructions="Research this carefully"),
        },
    )
    assert revised.status_code == 200
    assert revised.json()["spec"]["revision"] == 2

    stale = client.post(
        "/v2/agent-specs/research/activate",
        headers=TENANT_A,
        json={"revision": 2, "expected_active_revision": 99},
    )
    assert stale.status_code == 409
    activated_v2 = client.post(
        "/v2/agent-specs/research/activate",
        headers=TENANT_A,
        json={"revision": 2, "expected_active_revision": 1},
    )
    assert activated_v2.status_code == 200

    revisions = client.get(
        "/v2/agent-specs/research/revisions",
        headers=TENANT_A,
    )
    assert [item["revision"] for item in revisions.json()["revisions"]] == [2, 1]
    rolled_back = client.post(
        "/v2/agent-specs/research/rollback",
        headers=TENANT_A,
        json={"expected_active_revision": 2},
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["spec"]["revision"] == 1

    deactivated = client.delete(
        "/v2/agent-specs/research?expected_active_revision=1",
        headers=TENANT_A,
    )
    assert deactivated.status_code == 200
    assert client.get("/v2/agent-specs/research", headers=TENANT_A).status_code == 404
    assert (
        client.get("/v2/agent-specs/research?revision=2", headers=TENANT_A).status_code
        == 200
    )
    assert (
        client.get("/v2/agent-specs/research?revision=2", headers=TENANT_B).status_code
        == 404
    )

    hidden_tenant = client.post(
        "/v2/agent-specs",
        headers=TENANT_A,
        json={
            "id": "bad",
            "tenant_id": "tenant-b",
            "spec": _agent_write(instructions="Do not accept tenant input"),
        },
    )
    assert hidden_tenant.status_code == 422
    assert client.get("/v2/agent-specs", headers={}).status_code == 401


def test_trigger_spec_crud_hides_tenant_and_enforces_target_isolation(
    tmp_path: Path,
) -> None:
    control = _control_plane(tmp_path)
    client = control.client
    assert client.post("/v2/agent-specs/seed-defaults", headers=TENANT_A).status_code == 201

    created = client.post(
        "/v2/trigger-specs",
        headers=TENANT_A,
        json={"id": "morning", "trigger": _trigger_write()},
    )
    assert created.status_code == 201
    assert created.json()["trigger"]["agent_spec"] == {
        "tenant_id": "tenant-a",
        "spec_id": "routine",
        "revision": 1,
    }
    assert client.get("/v2/trigger-specs/morning", headers=TENANT_A).status_code == 404
    assert (
        client.post(
            "/v2/trigger-specs/morning/activate",
            headers=TENANT_A,
            json={"revision": 1, "expected_active_revision": None},
        ).status_code
        == 200
    )

    revised_write = _trigger_write(instruction="Prepare a concise morning brief")
    revised = client.post(
        "/v2/trigger-specs",
        headers=TENANT_A,
        json={"id": "morning", "expected_revision": 1, "trigger": revised_write},
    )
    assert revised.status_code == 200
    assert revised.json()["trigger"]["revision"] == 2
    assert (
        client.post(
            "/v2/trigger-specs/morning/activate",
            headers=TENANT_A,
            json={"revision": 2, "expected_active_revision": 1},
        ).status_code
        == 200
    )
    rolled_back = client.post(
        "/v2/trigger-specs/morning/rollback",
        headers=TENANT_A,
        json={"expected_active_revision": 2},
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["trigger"]["revision"] == 1
    assert (
        client.delete(
            "/v2/trigger-specs/morning?expected_active_revision=1",
            headers=TENANT_A,
        ).status_code
        == 200
    )
    assert (
        client.get("/v2/trigger-specs/morning?revision=2", headers=TENANT_B).status_code
        == 404
    )

    injected = _trigger_write()
    agent_ref = dict(injected["agent_spec"])
    agent_ref["tenant_id"] = "tenant-b"
    injected["agent_spec"] = agent_ref
    rejected_tenant = client.post(
        "/v2/trigger-specs",
        headers=TENANT_A,
        json={"id": "injected", "trigger": injected},
    )
    assert rejected_tenant.status_code == 422

    wrong_isolation = _trigger_write()
    wrong_isolation["exposure"] = "external"
    wrong_isolation["source"] = {
        "kind": "event",
        "event_type": "message",
        "source": "webhook",
        "authentication": "required",
    }
    rejected_isolation = client.post(
        "/v2/trigger-specs",
        headers=TENANT_A,
        json={"id": "external_to_private", "trigger": wrong_isolation},
    )
    assert rejected_isolation.status_code == 422


def test_secret_ingress_never_exposes_plaintext_and_is_tenant_scoped(tmp_path: Path) -> None:
    control = _control_plane(tmp_path)
    client = control.client

    pending = client.post(
        "/v2/secrets/pending",
        headers=TENANT_A,
        json={
            "id": "telegram_bot",
            "name": "telegram_bot_token",
            "scopes": ["telegram.send"],
        },
    )
    assert pending.status_code == 201
    assert pending.headers["cache-control"] == "no-store"
    assert pending.json()["secret"]["state"] == "pending"
    assert set(pending.json()["secret"]) == {
        "tenant_id",
        "id",
        "revision",
        "name",
        "state",
        "scopes",
        "created_at",
        "created_by",
    }
    duplicate = client.post(
        "/v2/secrets/pending",
        headers=TENANT_A,
        json={
            "id": "telegram_bot",
            "name": "telegram_bot_token",
            "scopes": ["telegram.send"],
        },
    )
    assert duplicate.status_code == 409

    first_value = "first-telegram-secret"
    stored = client.put(
        "/v2/secrets/telegram_bot",
        headers=TENANT_A,
        json={"expected_revision": 1, "value": first_value},
    )
    assert stored.status_code == 200
    assert stored.headers["cache-control"] == "no-store"
    assert stored.json()["secret"]["state"] == "active"
    assert stored.json()["secret"]["revision"] == 2
    assert first_value not in stored.text

    listed = client.get("/v2/secrets", headers=TENANT_A)
    assert listed.status_code == 200
    assert listed.headers["cache-control"] == "no-store"
    assert listed.json()["secrets"][0]["id"] == "telegram_bot"
    assert first_value not in listed.text
    assert client.get("/v2/secrets/telegram_bot", headers=TENANT_B).status_code == 404

    stale_value = "stale-secret-value"
    stale = client.put(
        "/v2/secrets/telegram_bot",
        headers=TENANT_A,
        json={"expected_revision": 1, "value": stale_value},
    )
    assert stale.status_code == 409
    assert stale_value not in stale.text

    second_value = "second-telegram-secret"
    rotated = client.put(
        "/v2/secrets/telegram_bot",
        headers=TENANT_A,
        json={
            "expected_revision": 2,
            "value": second_value,
            "scopes": ["telegram.send", "telegram.receive"],
        },
    )
    assert rotated.status_code == 200
    assert rotated.json()["secret"]["revision"] == 3
    assert second_value not in rotated.text

    with sqlite3.connect(control.vault.db_path) as conn:
        rows = conn.execute(
            "SELECT ciphertext FROM secret_revisions WHERE tenant_id = ?",
            ("tenant-a",),
        ).fetchall()
    stored_ciphertext = b"".join(bytes(row[0]) for row in rows if row[0] is not None)
    assert first_value.encode() not in stored_ciphertext
    assert second_value.encode() not in stored_ciphertext

    revoked = client.delete(
        "/v2/secrets/telegram_bot?expected_revision=3",
        headers=TENANT_A,
    )
    assert revoked.status_code == 200
    assert revoked.json()["secret"]["state"] == "revoked"
    restore_value = "must-not-restore"
    restore = client.put(
        "/v2/secrets/telegram_bot",
        headers=TENANT_A,
        json={"expected_revision": 4, "value": restore_value},
    )
    assert restore.status_code == 409
    assert restore_value not in restore.text
    assert client.get("/v2/secrets/telegram_bot/value", headers=TENANT_A).status_code == 404

    hidden_tenant = client.post(
        "/v2/secrets/pending",
        headers=TENANT_A,
        json={
            "id": "bad",
            "name": "bad",
            "scopes": ["telegram.send"],
            "tenant_id": "tenant-b",
        },
    )
    assert hidden_tenant.status_code == 422
    assert client.get("/v2/secrets", headers={}).status_code == 401
