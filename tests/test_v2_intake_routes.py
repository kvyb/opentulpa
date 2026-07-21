from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_intake import register_v2_intake_routes
from opentulpa.intake.drafts import ActivatedIntakeDraft, IntakeDraftService, IntakeDraftStore


@dataclass(frozen=True)
class _Principal:
    tenant_id: str
    actor_id: str


class _Workflows:
    def __init__(self) -> None:
        self.active: dict[tuple[str, str], dict[str, Any]] = {}
        self.test_calls: list[tuple[str, str]] = []
        self.reconcile_calls: list[dict[str, Any]] = []

    def activate_draft(
        self,
        *,
        draft_store: IntakeDraftStore,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token_hash: str,
        proposal: dict[str, Any],
        now: datetime,
    ) -> ActivatedIntakeDraft:
        workflow_id = str(proposal["workflow_id"])
        attempt_id = "route-test-activation"
        draft_store.claim_activation(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
            confirmation_token_hash=confirmation_token_hash,
            activation_attempt_id=attempt_id,
            now=now,
        )
        workflow = {**proposal, "customer_id": tenant_id, "revision": 1}
        self.active[(tenant_id, workflow_id)] = workflow
        activated = draft_store.finish_activation(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
            activation_attempt_id=attempt_id,
            now=now,
        )
        return ActivatedIntakeDraft(draft=activated, workflow=workflow)

    def list_workflows(
        self,
        *,
        customer_id: str,
        include_disabled: bool,
    ) -> list[dict[str, Any]]:
        assert include_disabled is True
        return [
            workflow for (tenant_id, _), workflow in self.active.items() if tenant_id == customer_id
        ]

    def get_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any] | None:
        return self.active.get((customer_id, workflow_id))

    def delete_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        expected_revision: int,
    ) -> dict[str, Any]:
        current = self.active.get((customer_id, workflow_id))
        if current is not None and current["revision"] != expected_revision:
            raise ValueError("revision conflict")
        deleted = self.active.pop((customer_id, workflow_id), None) is not None
        return {"deleted": deleted}

    async def run_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        event_type: str,
        force: bool,
    ) -> dict[str, Any]:
        assert event_type == "test"
        assert force is True
        self.test_calls.append((customer_id, workflow_id))
        return {"ok": True, "workflow_id": workflow_id}

    def reconcile_sink_effect(self, **kwargs: Any) -> dict[str, Any]:
        self.reconcile_calls.append(dict(kwargs))
        return {
            "booking_id": kwargs["booking_id"],
            "effect_revision": kwargs["effect_revision"],
            "decision": kwargs["decision"],
        }


def _payload() -> dict[str, Any]:
    return {
        "name": "Lead capture",
        "intent_description": "Book a service",
        "required_fields": ["name", "email"],
        "sink_type": "local_csv",
        "sink_config": {"file_path": "leads.csv"},
    }


def _client(tmp_path: Path) -> tuple[TestClient, _Workflows]:
    workflows = _Workflows()
    drafts = IntakeDraftService(
        IntakeDraftStore(tmp_path / "intake-drafts.sqlite"),
        workflow_activator=workflows,
        clock=lambda: datetime(2026, 7, 19, 12, tzinfo=UTC),
        token_factory=lambda: "confirmation-token-that-is-at-least-32-characters",
    )
    app = FastAPI()

    async def resolve_principal(request: Request) -> _Principal:
        return _Principal(
            tenant_id=request.headers.get("x-tenant-id", ""),
            actor_id=request.headers.get("x-actor-id", "owner-1"),
        )

    register_v2_intake_routes(
        app,
        get_draft_service=lambda: drafts,
        get_intake_workflows=lambda: workflows,
        resolve_principal=resolve_principal,
    )
    return TestClient(app), workflows


def test_v2_intake_draft_prepare_activate_and_workflow_actions(tmp_path: Path) -> None:
    client, workflows = _client(tmp_path)
    headers = {"x-tenant-id": "tenant-a"}
    created = client.post(
        "/v2/intake/drafts",
        headers=headers,
        json={"id": "draft-1", "workflow_id": "workflow-1", "payload": _payload()},
    )
    assert created.status_code == 201
    assert created.json()["draft"]["revision"] == 1

    prepared = client.post(
        "/v2/intake/drafts/draft-1/prepare",
        headers=headers,
        json={"expected_revision": 1},
    )
    assert prepared.status_code == 200
    confirmation_token = prepared.json()["prepared"]["confirmation_token"]
    assert prepared.json()["prepared"]["proposal"]["workflow_id"] == "workflow-1"

    activated = client.post(
        "/v2/intake/drafts/draft-1/activate",
        headers=headers,
        json={"expected_revision": 1, "confirmation_token": confirmation_token},
    )
    assert activated.status_code == 200
    assert activated.json()["activated"]["draft"]["status"] == "activated"

    listed = client.get("/v2/intake/workflows", headers=headers)
    assert listed.status_code == 200
    assert [item["workflow_id"] for item in listed.json()["workflows"]] == ["workflow-1"]
    assert (
        client.get(
            "/v2/intake/workflows/workflow-1",
            headers={"x-tenant-id": "tenant-b"},
        ).status_code
        == 404
    )

    tested = client.post(
        "/v2/intake/workflows/workflow-1/test",
        headers=headers,
        json={},
    )
    assert tested.status_code == 200
    assert tested.json()["result"]["ok"] is True
    assert workflows.test_calls == [("tenant-a", "workflow-1")]

    deleted = client.delete(
        "/v2/intake/workflows/workflow-1?expected_revision=1",
        headers=headers,
    )
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True


def test_v2_intake_routes_enforce_principal_revision_and_hidden_context(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    headers = {"x-tenant-id": "tenant-a"}
    assert client.get("/v2/intake/drafts").status_code == 401

    exposed_tenant = client.post(
        "/v2/intake/drafts",
        headers=headers,
        json={"tenant_id": "tenant-b", "payload": _payload()},
    )
    assert exposed_tenant.status_code == 422
    hidden_context_in_payload = client.post(
        "/v2/intake/drafts",
        headers=headers,
        json={"payload": {**_payload(), "customer_id": "tenant-b"}},
    )
    assert hidden_context_in_payload.status_code == 422

    created = client.post(
        "/v2/intake/drafts",
        headers=headers,
        json={"id": "draft-1", "payload": _payload()},
    )
    assert created.status_code == 201
    missing_revision = client.post(
        "/v2/intake/drafts",
        headers=headers,
        json={"id": "draft-1", "payload": {"name": "Changed"}},
    )
    assert missing_revision.status_code == 409
    stale_patch = client.patch(
        "/v2/intake/drafts/draft-1",
        headers=headers,
        json={"expected_revision": 2, "patch": {"name": "Changed"}},
    )
    assert stale_patch.status_code == 409
    updated = client.patch(
        "/v2/intake/drafts/draft-1",
        headers=headers,
        json={"expected_revision": 1, "patch": {"name": "Changed"}},
    )
    assert updated.status_code == 200
    assert updated.json()["draft"]["revision"] == 2
    assert (
        client.get(
            "/v2/intake/drafts/draft-1",
            headers={"x-tenant-id": "tenant-b"},
        ).status_code
        == 404
    )


def test_v2_sink_reconciliation_injects_tenant_and_actor_context(tmp_path: Path) -> None:
    client, workflows = _client(tmp_path)

    response = client.post(
        "/v2/intake/workflows/workflow-1/bookings/booking-1/sink/reconcile",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "owner-a"},
        json={
            "effect_revision": 2,
            "decision": "retry_no_effect",
            "reason": "verified no record exists",
        },
    )

    assert response.status_code == 200
    assert response.json()["reconciliation"] == {
        "booking_id": "booking-1",
        "effect_revision": 2,
        "decision": "retry_no_effect",
    }
    assert workflows.reconcile_calls == [
        {
            "customer_id": "tenant-a",
            "actor_id": "owner-a",
            "workflow_id": "workflow-1",
            "booking_id": "booking-1",
            "effect_revision": 2,
            "decision": "retry_no_effect",
            "reason": "verified no record exists",
            "provider_result": {},
        }
    ]

    exposed_context = client.post(
        "/v2/intake/workflows/workflow-1/bookings/booking-1/sink/reconcile",
        headers={"x-tenant-id": "tenant-a"},
        json={
            "tenant_id": "tenant-b",
            "effect_revision": 2,
            "decision": "retry_no_effect",
            "reason": "bad request",
        },
    )
    assert exposed_context.status_code == 422


def test_v2_workflow_test_redacts_model_and_provider_failures(tmp_path: Path) -> None:
    client, workflows = _client(tmp_path)
    workflows.active[("tenant-a", "workflow-1")] = {
        "workflow_id": "workflow-1",
        "customer_id": "tenant-a",
        "revision": 1,
    }

    async def failed_run(**_: Any) -> dict[str, Any]:
        return {
            "ok": False,
            "workflow_id": "workflow-1",
            "event_type": "test",
            "errors": ["provider token=private-secret from /srv/private/.env"],
            "summary": "raw provider response: private-secret",
        }

    workflows.run_workflow = failed_run  # type: ignore[method-assign]
    response = client.post(
        "/v2/intake/workflows/workflow-1/test",
        headers={"x-tenant-id": "tenant-a"},
        json={"force": True},
    )

    assert response.status_code == 200
    assert response.json()["result"]["summary"] == (
        "Intake workflow test failed. Check server logs."
    )
    assert "private-secret" not in response.text
    assert "/srv/private" not in response.text


def test_v2_draft_activation_redacts_nested_provider_exception(tmp_path: Path) -> None:
    client, workflows = _client(tmp_path)
    headers = {"x-tenant-id": "tenant-a"}
    created = client.post(
        "/v2/intake/drafts",
        headers=headers,
        json={
            "id": "draft-secret",
            "workflow_id": "workflow-secret",
            "payload": _payload(),
        },
    )
    assert created.status_code == 201
    prepared = client.post(
        "/v2/intake/drafts/draft-secret/prepare",
        headers=headers,
        json={"expected_revision": 1},
    )
    token = prepared.json()["prepared"]["confirmation_token"]

    def failed_activation(**_: Any) -> Any:
        raise RuntimeError("provider body token=private-secret from /srv/private/.env")

    workflows.activate_draft = failed_activation  # type: ignore[method-assign]
    response = client.post(
        "/v2/intake/drafts/draft-secret/activate",
        headers=headers,
        json={"expected_revision": 1, "confirmation_token": token},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "intake draft activation failed"
    assert "private-secret" not in response.text
    assert "/srv/private" not in response.text
