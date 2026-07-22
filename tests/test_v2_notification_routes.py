from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.principal import (
    CapabilityPrincipalResolver,
    OwnerOrCapabilityPrincipalResolver,
    OwnerPrincipalResolver,
)
from opentulpa.api.routes.v2_notifications import register_v2_notification_routes
from opentulpa.capabilities import (
    CapabilityAPICredentialService,
    CapabilityAPIScope,
    CapabilityCredentialStore,
)
from opentulpa.notifications import (
    NotificationApproval,
    NotificationService,
    NotificationStore,
    NotificationWrite,
)
from opentulpa.specs import AgentRunBinding, AgentSpecRef


def _issue(
    store: CapabilityCredentialStore,
    *,
    scopes: frozenset[str],
) -> str:
    return store.issue(
        tenant_id="tenant-capability",
        actor_id="capability:telegram-1",
        capability_name="telegram",
        capability_instance_id="telegram-1",
        interface="telegram",
        source_id="telegram-1",
        channel="telegram",
        agent_binding=AgentRunBinding(
            agent_spec=AgentSpecRef(
                tenant_id="tenant-capability",
                spec_id="owner",
                revision=1,
            ),
            run_kind="owner",
            trust_class="owner",
        ),
        scopes=scopes,
    ).token.get_secret_value()


def _client(tmp_path: Path) -> tuple[TestClient, NotificationService, CapabilityCredentialStore]:
    credentials = CapabilityCredentialStore(tmp_path / "credentials.db")
    service = NotificationService(NotificationStore(tmp_path / "notifications.db"))
    principal = OwnerOrCapabilityPrincipalResolver(
        owner=OwnerPrincipalResolver(token="owner-token", tenant_id="tenant-owner"),
        capability=CapabilityPrincipalResolver(
            CapabilityAPICredentialService(credentials)
        ),
    )
    app = FastAPI()
    register_v2_notification_routes(
        app,
        get_notifications=lambda: service,
        resolve_principal=cast(Any, principal),
    )
    return TestClient(app), service, credentials


def test_owner_and_capability_receive_only_their_tenant_and_ack_independently(
    tmp_path: Path,
) -> None:
    client, service, credentials = _client(tmp_path)
    owner = service.publish(
        tenant_id="tenant-owner",
        dedupe_key="owner-event",
        notification=NotificationWrite(
            kind="evolution.candidate.ready",
            text="Candidate passed evaluation.",
            status="ready",
        ),
    )
    capability = service.publish(
        tenant_id="tenant-capability",
        dedupe_key="capability-event",
        notification=NotificationWrite(
            kind="approval.required",
            text="A scheduled run needs approval.",
            status="interrupted",
            thread_id="trigger:daily",
            run_id="run-waiting",
            approvals=(
                NotificationApproval(
                    approval_id="approval-1",
                    tool_name="integration_invoke",
                    description="Send an email.",
                    allowed_decisions=("approve", "edit", "reject"),
                ),
            ),
        ),
    )
    token = _issue(
        credentials,
        scopes=frozenset(
            {
                CapabilityAPIScope.NOTIFICATIONS_READ.value,
                CapabilityAPIScope.NOTIFICATIONS_ACK.value,
            }
        ),
    )

    owner_response = client.get(
        "/v2/notifications",
        headers={"Authorization": "Bearer owner-token"},
    )
    assert owner_response.status_code == 200
    assert [item["id"] for item in owner_response.json()["notifications"]] == [owner.id]
    assert "tenant_id" not in owner_response.json()["notifications"][0]
    assert "dedupe_key" not in owner_response.json()["notifications"][0]

    headers = {"Authorization": f"Bearer {token}"}
    capability_response = client.get("/v2/notifications", headers=headers)
    assert capability_response.status_code == 200
    payload = capability_response.json()["notifications"]
    assert [item["id"] for item in payload] == [capability.id]
    assert payload[0]["approvals"][0]["approval_id"] == "approval-1"
    assert client.post(
        f"/v2/notifications/{capability.id}/ack", headers=headers
    ).status_code == 204
    assert client.get("/v2/notifications", headers=headers).json()["notifications"] == []
    assert client.post(
        f"/v2/notifications/{owner.id}/ack", headers=headers
    ).status_code == 404


def test_capability_notification_scopes_fail_closed(tmp_path: Path) -> None:
    client, service, credentials = _client(tmp_path)
    notification = service.publish(
        tenant_id="tenant-capability",
        dedupe_key="event",
        notification=NotificationWrite(
            kind="run.completed",
            text="Completed.",
            status="completed",
        ),
    )
    read_token = _issue(
        credentials,
        scopes=frozenset({CapabilityAPIScope.NOTIFICATIONS_READ.value}),
    )
    read_headers = {"Authorization": f"Bearer {read_token}"}

    assert client.get("/v2/notifications", headers=read_headers).status_code == 200
    assert client.post(
        f"/v2/notifications/{notification.id}/ack",
        headers=read_headers,
    ).status_code == 403

    ack_token = _issue(
        credentials,
        scopes=frozenset({CapabilityAPIScope.NOTIFICATIONS_ACK.value}),
    )
    ack_headers = {"Authorization": f"Bearer {ack_token}"}
    assert client.get("/v2/notifications", headers=ack_headers).status_code == 403
