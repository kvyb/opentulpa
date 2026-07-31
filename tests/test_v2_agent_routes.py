from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_agent import register_v2_agent_routes
from opentulpa.deep_agent.contracts import (
    AgentApproval,
    AgentRunEvent,
    AgentRunIdempotencyConflictError,
    AgentRunRequest,
    AgentRunSnapshot,
    ApprovalDecision,
)
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling.contract import AgentChannel, AgentRunContext, AgentRunKind


@dataclass(frozen=True)
class _Principal:
    tenant_id: str
    actor_id: str


@dataclass
class _FakeAgentService:
    snapshots: dict[str, AgentRunSnapshot] = field(default_factory=dict)
    requests: list[AgentRunRequest] = field(default_factory=list)
    decisions: list[tuple[str, ApprovalDecision]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    ensured_threads: list[tuple[str, str, str]] = field(default_factory=list)
    stream_error: Exception | None = None
    thread_tenant: str = "tenant-a"

    async def open_stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        if self.stream_error is not None:
            raise self.stream_error
        return self.stream(request)

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield _event("run.started", 1, {"thread_id": request.context.thread_id})
        yield _event(
            "message.delta",
            2,
            {"text": "Hello", "api_key": "must-not-leak"},
        )
        yield _event("run.completed", 3, {"text": "Hello"})

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        self.decisions.append((run_id, decision))
        yield _event("run.started", 1, {"resumed": True}, run_id=run_id)
        yield _event("run.completed", 2, {"text": "Approved"}, run_id=run_id)

    async def open_resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        return self.resume(run_id, decision)

    async def get_run(self, run_id: str) -> AgentRunSnapshot | None:
        return self.snapshots.get(run_id)

    async def events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[AgentRunEvent]:
        for sequence in range(after_sequence + 1, 4):
            yield _event("message.delta", sequence, {"text": str(sequence)}, run_id=run_id)

    async def cancel(self, run_id: str) -> AgentRunSnapshot:
        self.cancelled.append(run_id)
        snapshot = self.snapshots[run_id]
        cancelled = AgentRunSnapshot(
            run_id=snapshot.run_id,
            context=snapshot.context,
            status="cancelled",
            final_text=snapshot.final_text,
            approvals=snapshot.approvals,
            created_at=snapshot.created_at,
            updated_at=snapshot.updated_at,
        )
        self.snapshots[run_id] = cancelled
        return cancelled

    async def cancel_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> AgentRunSnapshot | None:
        for snapshot in reversed(tuple(self.snapshots.values())):
            if (
                snapshot.context.tenant_id == tenant_id
                and snapshot.context.thread_id == thread_id
                and snapshot.status in {"running", "interrupted", "resume_pending"}
            ):
                return await self.cancel(snapshot.run_id)
        return None

    async def create_thread(
        self, *, tenant_id: str, channel: str, title: str | None = None
    ) -> dict[str, Any]:
        self.thread_tenant = tenant_id
        return {
            "thread_id": "thread-created",
            "title": title or "New session",
            "channel": channel,
            "archived": False,
        }

    async def ensure_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        channel: str,
    ) -> None:
        self.ensured_threads.append((tenant_id, thread_id, channel))

    async def list_threads(
        self, *, tenant_id: str, cursor: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        del cursor, limit
        return {
            "threads": (
                [{"thread_id": "thread-1", "title": "Main", "channel": "web"}]
                if tenant_id == self.thread_tenant
                else []
            ),
            "next_cursor": None,
        }

    async def thread_timeline(
        self, *, tenant_id: str, thread_id: str, cursor: int = 0, limit: int = 30
    ) -> dict[str, Any] | None:
        del cursor, limit
        if tenant_id != self.thread_tenant or thread_id != "thread-1":
            return None
        return {"thread": {"thread_id": thread_id}, "entries": [], "next_cursor": None}

    async def update_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None:
        if tenant_id != self.thread_tenant or thread_id != "thread-1":
            return None
        return {"thread_id": thread_id, "title": title or "Main", "archived": bool(archived)}


def _event(
    event_type: Any,
    sequence: int,
    data: dict[str, Any],
    *,
    run_id: str = "run_123",
) -> AgentRunEvent:
    return AgentRunEvent(
        type=event_type,
        run_id=run_id,
        sequence=sequence,
        timestamp="2026-07-19T12:00:00+00:00",
        data=data,
    )


def _context(*, tenant_id: str = "tenant-a") -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="actor-1",
        thread_id="thread-1",
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id="corr-1",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
        trust_class="owner",
    )


def _snapshot(*, tenant_id: str = "tenant-a", status: Any = "interrupted") -> AgentRunSnapshot:
    return AgentRunSnapshot(
        run_id="run_123",
        context=_context(tenant_id=tenant_id),
        status=status,
        final_text="Partial",
        approvals=(
            AgentApproval(
                id="approval-1",
                tool_name="integration_invoke",
                description="Send the message",
                arguments={"token": "must-not-leak", "recipient": "person@example.com"},
                allowed_decisions=("approve", "edit", "reject"),
            ),
            AgentApproval(
                id="approval-decided",
                tool_name="browser_act",
                description="Already rejected",
                arguments={},
                allowed_decisions=("approve", "reject"),
                status="reject",
            ),
        ),
        created_at="2026-07-19T11:59:00+00:00",
        updated_at="2026-07-19T12:00:00+00:00",
    )


def _client(
    service: _FakeAgentService | None = None,
    *,
    secret_ingress: Any | None = None,
) -> tuple[TestClient, _FakeAgentService]:
    fake = service or _FakeAgentService()
    app = FastAPI()

    async def resolve_principal(request: Request) -> _Principal:
        return _Principal(
            tenant_id=request.headers.get("x-tenant-id", ""),
            actor_id=request.headers.get("x-actor-id", ""),
        )

    register_v2_agent_routes(
        app,
        get_agent_service=lambda: fake,
        resolve_principal=resolve_principal,
        secret_ingress=secret_ingress,
    )
    return TestClient(app), fake


def _sse_payloads(body: str) -> list[tuple[str, dict[str, Any]]]:
    parsed: list[tuple[str, dict[str, Any]]] = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event_type = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
        raw_data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
        parsed.append((event_type, json.loads(raw_data)))
    return parsed


def test_start_run_injects_principal_context_and_streams_normalized_sse() -> None:
    client, service = _client()

    response = client.post(
        "/v2/agent/runs",
        headers={
            "x-tenant-id": "tenant-a",
            "x-actor-id": "actor-7",
            "x-correlation-id": "request-42",
            "idempotency-key": "interface-event-42",
        },
        json={"thread_id": "thread-7", "text": "Help me", "file_ids": ["file-1"]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    request = service.requests[0]
    assert request.context == AgentRunContext(
        tenant_id="tenant-a",
        actor_id="actor-7",
        thread_id="thread-7",
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id="request-42",
        origin=OriginRef(interface="web", source_id="owner-web"),
        agent_spec=AgentSpecRef(tenant_id="tenant-a", spec_id="owner", revision=1),
        trust_class="owner",
    )
    assert request.file_ids == ("file-1",)
    assert request.idempotency_key == "interface-event-42"

    events = _sse_payloads(response.text)
    assert [item[0] for item in events] == [
        "run.started",
        "message.delta",
        "run.completed",
    ]
    assert [item[1]["sequence"] for item in events] == [1, 2, 3]
    assert all(item[1]["run_id"] == "run_123" for item in events)
    assert all(item[1]["timestamp"] == "2026-07-19T12:00:00+00:00" for item in events)
    assert events[1][1]["data"] == {"text": "Hello", "api_key": "[redacted]"}


def test_agent_events_and_snapshots_redact_private_key_blocks() -> None:
    private_key = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "not-a-real-private-key-for-tests\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )

    class PrivateKeyService(_FakeAgentService):
        async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
            self.requests.append(request)
            yield _event("run.started", 1, {"thread_id": request.context.thread_id})
            yield _event("run.completed", 2, {"text": f"Here it is:\n{private_key}"})

    service = PrivateKeyService()
    snapshot = _snapshot(status="completed")
    service.snapshots["run_123"] = AgentRunSnapshot(
        run_id=snapshot.run_id,
        context=snapshot.context,
        status=snapshot.status,
        final_text=f"Here it is:\n{private_key}",
        approvals=snapshot.approvals,
        created_at=snapshot.created_at,
        updated_at=snapshot.updated_at,
    )
    client, _ = _client(service)

    response = client.post(
        "/v2/agent/runs",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-7"},
        json={"thread_id": "thread-7", "text": "Help me", "file_ids": []},
    )
    snapshot_response = client.get(
        "/v2/agent/runs/run_123",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-7"},
    )

    events = _sse_payloads(response.text)
    assert events[-1][1]["data"]["text"] == "Here it is:\n[redacted-private-key]"
    assert snapshot_response.json()["final_text"] == "Here it is:\n[redacted-private-key]"
    assert private_key not in response.text
    assert private_key not in snapshot_response.text


def test_thread_routes_are_server_owned_and_tenant_scoped() -> None:
    client, service = _client()
    headers = {"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"}

    created = client.post("/v2/agent/threads", headers=headers, json={"title": "Research"})
    assert created.status_code == 201
    assert created.json()["title"] == "Research"
    ensured = client.put("/v2/agent/threads/thread-telegram", headers=headers)
    assert ensured.status_code == 200
    assert ensured.json() == {"thread_id": "thread-telegram"}
    assert service.ensured_threads == [
        ("tenant-a", "thread-telegram", "web"),
    ]
    assert client.get("/v2/agent/threads", headers=headers).json()["threads"][0]["title"] == "Main"
    assert (
        client.get("/v2/agent/threads/thread-1/timeline", headers=headers).status_code == 200
    )
    updated = client.patch(
        "/v2/agent/threads/thread-1",
        headers=headers,
        json={"title": "Renamed"},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Renamed"

    other = {"x-tenant-id": "tenant-b", "x-actor-id": "actor-2"}
    assert client.get("/v2/agent/threads/thread-1/timeline", headers=other).status_code == 404
    assert client.patch(
        "/v2/agent/threads/thread-1", headers=other, json={"archived": True}
    ).status_code == 404


def test_idempotency_conflict_returns_409_before_sse_headers() -> None:
    service = _FakeAgentService(
        stream_error=AgentRunIdempotencyConflictError("request digest changed")
    )
    client, _ = _client(service)

    response = client.post(
        "/v2/agent/runs",
        headers={
            "x-tenant-id": "tenant-a",
            "x-actor-id": "actor-1",
            "idempotency-key": "same-key",
        },
        json={"thread_id": "thread-1", "text": "Changed", "file_ids": []},
    )

    assert response.status_code == 409
    assert response.headers["content-type"].startswith("application/json")
    assert response.json() == {
        "detail": "idempotency key belongs to a different agent run request"
    }


def test_start_run_rejects_missing_principal_and_model_visible_identity() -> None:
    client, service = _client()
    payload = {"thread_id": "thread-1", "text": "Hello", "file_ids": []}

    assert client.post("/v2/agent/runs", json=payload).status_code == 401
    assert (
        client.post(
            "/v2/agent/runs",
            headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"},
            json={**payload, "tenant_id": "tenant-b"},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/v2/agent/runs",
            headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"},
            json={**payload, "customer_id": "customer-b"},
        ).status_code
        == 422
    )
    assert service.requests == []


def test_start_run_replaces_pasted_secret_before_agent_checkpointing() -> None:
    raw = "1234567890:AAEabcdefghijklmnopqrstuvwxyz012345678"
    calls: list[dict[str, str]] = []

    def ingress(**kwargs: str) -> str:
        calls.append(kwargs)
        return kwargs["text"].replace(raw, "secret://telegram_bot_token")

    client, service = _client(secret_ingress=ingress)
    response = client.post(
        "/v2/agent/runs",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"},
        json={"thread_id": "thread-1", "text": f"Use {raw}", "file_ids": []},
    )

    assert response.status_code == 200
    assert service.requests[0].text == "Use secret://telegram_bot_token"
    assert raw not in service.requests[0].text
    assert calls == [
        {"tenant_id": "tenant-a", "actor_id": "actor-1", "text": f"Use {raw}"}
    ]


def test_get_run_is_tenant_scoped_and_only_returns_redacted_pending_approvals() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot()})
    client, _ = _client(service)

    hidden = client.get(
        "/v2/agent/runs/run_123",
        headers={"x-tenant-id": "tenant-b", "x-actor-id": "actor-2"},
    )
    assert hidden.status_code == 404

    response = client.get(
        "/v2/agent/runs/run_123",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-2"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "interrupted"
    assert payload["thread_id"] == "thread-1"
    assert "tenant_id" not in payload
    assert payload["pending_approvals"] == [
        {
            "approval_id": "approval-1",
            "tool_name": "integration_invoke",
            "description": "Send the message",
            "arguments": {"token": "[redacted]", "recipient": "person@example.com"},
            "allowed_decisions": ["approve", "edit", "reject"],
        }
    ]


def test_resume_validates_approval_and_streams_the_native_events() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot()})
    client, _ = _client(service)
    headers = {"x-tenant-id": "tenant-a", "x-actor-id": "actor-2"}

    response = client.post(
        "/v2/agent/runs/run_123/resume",
        headers=headers,
        json={
            "approval_id": "approval-1",
            "decision": "edit",
            "edited_arguments": {"recipient": "other@example.com"},
        },
    )

    assert response.status_code == 200
    assert service.decisions == [
        (
            "run_123",
            ApprovalDecision(
                approval_id="approval-1",
                decision="edit",
                edited_arguments={"recipient": "other@example.com"},
            ),
        )
    ]
    assert [item[0] for item in _sse_payloads(response.text)] == [
        "run.started",
        "run.completed",
    ]

    missing_arguments = client.post(
        "/v2/agent/runs/run_123/resume",
        headers=headers,
        json={"approval_id": "approval-1", "decision": "edit"},
    )
    assert missing_arguments.status_code == 422

    wrong_tenant = client.post(
        "/v2/agent/runs/run_123/resume",
        headers={"x-tenant-id": "tenant-b", "x-actor-id": "actor-2"},
        json={"approval_id": "approval-1", "decision": "approve"},
    )
    assert wrong_tenant.status_code == 404


def test_resume_rejects_non_interrupted_runs() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot(status="completed")})
    client, _ = _client(service)

    response = client.post(
        "/v2/agent/runs/run_123/resume",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"},
        json={"approval_id": "approval-1", "decision": "approve"},
    )

    assert response.status_code == 409
    assert service.decisions == []


def test_replay_events_honors_cursor_and_cancel_is_tenant_scoped() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot(status="running")})
    client, _ = _client(service)
    headers = {"x-tenant-id": "tenant-a", "x-actor-id": "actor-2"}

    replay = client.get(
        "/v2/agent/runs/run_123/events?after_sequence=1",
        headers={**headers, "last-event-id": "2"},
    )

    assert replay.status_code == 200
    assert [payload[1]["sequence"] for payload in _sse_payloads(replay.text)] == [3]
    assert (
        client.get(
            "/v2/agent/runs/run_123/events",
            headers={**headers, "last-event-id": "bad"},
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/v2/agent/runs/run_123/cancel",
            headers={"x-tenant-id": "tenant-b", "x-actor-id": "actor-2"},
        ).status_code
        == 404
    )

    cancelled = client.post("/v2/agent/runs/run_123/cancel", headers=headers)

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert service.cancelled == ["run_123"]


def test_steer_cancels_active_run_and_continues_on_its_owned_thread() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot(status="running")})
    client, _ = _client(service)
    headers = {
        "x-tenant-id": "tenant-a",
        "x-actor-id": "actor-9",
        "x-correlation-id": "steer-1",
        "idempotency-key": "steer-request-1",
    }

    response = client.post(
        "/v2/agent/runs/run_123/steer",
        headers=headers,
        json={"text": "Focus on the failing test", "file_ids": ["file-2"]},
    )

    assert response.status_code == 200
    assert service.cancelled == ["run_123"]
    steered = service.requests[-1]
    assert steered.text == "Focus on the failing test"
    assert steered.file_ids == ("file-2",)
    assert steered.idempotency_key == "steer-request-1"
    assert steered.context.thread_id == "thread-1"
    assert steered.context.actor_id == "actor-9"
    assert steered.context.correlation_id == "steer-1"

    wrong_tenant = client.post(
        "/v2/agent/runs/run_123/steer",
        headers={"x-tenant-id": "tenant-b", "x-actor-id": "actor-2"},
        json={"text": "Take over", "file_ids": []},
    )
    assert wrong_tenant.status_code == 404


def test_steer_rejects_a_run_that_is_no_longer_active() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot(status="completed")})
    client, _ = _client(service)

    response = client.post(
        "/v2/agent/runs/run_123/steer",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"},
        json={"text": "Too late", "file_ids": []},
    )

    assert response.status_code == 409
    assert service.cancelled == []
    assert service.requests == []


def test_cancel_thread_handles_a_run_before_the_client_has_its_id() -> None:
    service = _FakeAgentService(snapshots={"run_123": _snapshot(status="running")})
    client, _ = _client(service)

    cancelled = client.post(
        "/v2/agent/threads/thread-1/cancel",
        headers={"x-tenant-id": "tenant-a", "x-actor-id": "actor-1"},
    )

    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert service.cancelled == ["run_123"]
    assert client.post(
        "/v2/agent/threads/thread-1/cancel",
        headers={"x-tenant-id": "tenant-b", "x-actor-id": "actor-2"},
    ).status_code == 404
