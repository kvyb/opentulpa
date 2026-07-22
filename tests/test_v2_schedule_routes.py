from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_schedules import register_v2_schedule_routes
from opentulpa.schedules.models import AgentJob, Cron, ScheduleWrite
from opentulpa.schedules.service import ScheduleService
from opentulpa.specs import (
    AgentSpecStore,
    TriggerSpecService,
    TriggerSpecStore,
    seed_default_agent_spec_refs,
)


@dataclass(frozen=True)
class _Principal:
    tenant_id: str
    actor_id: str


def _client(tmp_path: Path) -> tuple[TestClient, ScheduleService]:
    agent_specs = AgentSpecStore(tmp_path / "agent_specs.db")
    trigger_specs = TriggerSpecStore(
        tmp_path / "trigger_specs.db",
        agent_specs=agent_specs,
    )

    def resolve(tenant_id: str):
        active = agent_specs.get_active_ref(tenant_id=tenant_id, spec_id="routine")
        if active is None:
            active = seed_default_agent_spec_refs(
                agent_specs,
                tenant_id=tenant_id,
                actor_id="test",
            )["routine"]
        return active

    service = ScheduleService(
        TriggerSpecService(trigger_specs),
        resolve_agent_spec=resolve,
        clock=lambda: datetime(2026, 7, 19, 12, tzinfo=UTC),
    )
    app = FastAPI()

    async def resolve_principal(request: Request) -> _Principal:
        return _Principal(
            tenant_id=request.headers.get("x-tenant-id", ""),
            actor_id="actor-1",
        )

    register_v2_schedule_routes(
        app,
        get_schedule_service=lambda: service,
        resolve_principal=resolve_principal,
    )
    return TestClient(app), service


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "Morning brief",
        "trigger": {
            "kind": "cron",
            "timezone": "Europe/Moscow",
            "expression": "0 9 * * *",
        },
        "action": {
            "kind": "agent_job",
            "instruction": "Prepare the morning brief",
        },
        "notify_owner": True,
        "enabled": True,
    }
    payload.update(overrides)
    return payload


def test_v2_schedule_routes_resolve_tenant_and_enforce_revisions(tmp_path: Path) -> None:
    client, service = _client(tmp_path)
    headers = {"x-tenant-id": "tenant-a"}

    created = client.post("/v2/schedules", headers=headers, json=_payload(id="sch_daily"))
    assert created.status_code == 201
    assert created.json()["schedule"]["revision"] == 1
    assert created.json()["schedule"]["tenant_id"] == "tenant-a"

    service.save(
        tenant_id="tenant-b",
        schedule_id="sch_other",
        write=ScheduleWrite(
            name="Other tenant",
            trigger=Cron(timezone="UTC", expression="0 0 * * *"),
            action=AgentJob(instruction="Do not expose this"),
        ),
    )
    listed = client.get("/v2/schedules", headers=headers)
    assert listed.status_code == 200
    assert {item["id"] for item in listed.json()["schedules"]} == {"sch_daily"}

    missing_revision = client.post(
        "/v2/schedules",
        headers=headers,
        json=_payload(id="sch_daily", name="Changed"),
    )
    assert missing_revision.status_code == 409

    updated = client.post(
        "/v2/schedules",
        headers=headers,
        json=_payload(id="sch_daily", expected_revision=1, name="Changed"),
    )
    assert updated.status_code == 200
    assert updated.json()["schedule"]["revision"] == 2

    stale_delete = client.delete(
        "/v2/schedules/sch_daily",
        headers=headers,
        params={"expected_revision": 1},
    )
    assert stale_delete.status_code == 409
    deleted = client.delete(
        "/v2/schedules/sch_daily",
        headers=headers,
        params={"expected_revision": 2},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted": True, "schedule_id": "sch_daily"}


def test_v2_schedule_routes_reject_missing_principal_and_model_visible_tenant(
    tmp_path: Path,
) -> None:
    client, _ = _client(tmp_path)

    unauthorized = client.get("/v2/schedules")
    assert unauthorized.status_code == 401

    exposed_tenant = client.post(
        "/v2/schedules",
        headers={"x-tenant-id": "tenant-a"},
        json=_payload(tenant_id="tenant-b"),
    )
    assert exposed_tenant.status_code == 422

    invalid_timezone = client.post(
        "/v2/schedules",
        headers={"x-tenant-id": "tenant-a"},
        json=_payload(trigger={"kind": "cron", "timezone": "+03:00", "expression": "0 9 * * *"}),
    )
    assert invalid_timezone.status_code == 422
