from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fastapi import Request
from fastapi.testclient import TestClient

from opentulpa.api.app import create_app


@dataclass(frozen=True)
class _Principal:
    tenant_id: str = "tenant-a"
    actor_id: str = "actor-a"


class _AsyncLifecycle:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events
        self.started = False

    def healthy(self) -> bool:
        return self.started

    async def start(self, *, recover_pending_resumes: bool = True) -> None:
        if self.name == "agent" and not recover_pending_resumes:
            self.events.append("start:agent:recovery-deferred")
            self.started = True
            return
        self.events.append(f"start:{self.name}")
        self.started = True

    async def recover_pending_resumes(self) -> None:
        self.events.append(f"recover:{self.name}")

    async def shutdown(self) -> None:
        self.events.append(f"stop:{self.name}")
        self.started = False


class _SyncDispatcher:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    def start(self) -> None:
        self.events.append(f"start:{self.name}")

    def shutdown(self, *, wait: bool = True) -> None:
        assert wait is True
        self.events.append(f"stop:{self.name}")

    def upsert(self, value: Any) -> None:
        _ = value

    def remove(self, **identifiers: str) -> None:
        _ = identifiers

    async def dispatch_event(self, **event: Any) -> None:
        _ = event


class _CloseOnly:
    def __init__(self, name: str, events: list[str]) -> None:
        self.name = name
        self.events = events

    async def shutdown(self) -> None:
        self.events.append(f"stop:{self.name}")


class _TelegramClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def aclose(self) -> None:
        self.events.append("stop:telegram")


def _app(events: list[str], *, with_capability_service: bool = False) -> Any:
    agent = _AsyncLifecycle("agent", events)
    jobs = _AsyncLifecycle("jobs", events)
    intake = _AsyncLifecycle("intake", events)
    trigger_dispatcher = _SyncDispatcher("triggers", events)
    intake_dispatcher = _SyncDispatcher("intake-poller", events)

    def principal(_: Request) -> _Principal:
        return _Principal()

    return create_app(
        agent_service=cast(Any, agent),
        job_service=cast(Any, jobs),
        file_vault_service=cast(Any, object()),
        integration_service=None,
        intake_workflow_service=cast(Any, intake),
        intake_draft_service=cast(Any, object()),
        schedule_service=cast(Any, object()),
        resolve_principal=principal,
        trigger_dispatcher=cast(Any, trigger_dispatcher),
        intake_poll_dispatcher=cast(Any, intake_dispatcher),
        browser_service=cast(Any, _CloseOnly("browser", events)),
        telegram_client=cast(Any, _TelegramClient(events)),
        capability_service=(
            cast(Any, _AsyncLifecycle("capabilities", events))
            if with_capability_service
            else None
        ),
    )


def test_v2_app_exposes_only_cutover_routes_and_deepagents_health() -> None:
    events: list[str] = []
    app = _app(events)
    paths = {route.path for route in app.routes}

    assert "/healthz" in paths
    assert "/" in paths
    assert "/agent/healthz" in paths
    assert "/v2/agent/runs" in paths
    assert "/v2/files" in paths
    assert "/v2/integrations" in paths
    assert "/v2/intake/workflows" in paths
    assert "/v2/schedules" in paths
    assert "/webhook/telegram" in paths
    assert "/webhook/composio/callback" in paths
    assert all(
        path in {"/", "/healthz", "/agent/healthz"}
        or path.startswith("/v2/")
        or path in {"/webhook/telegram", "/webhook/composio/callback"}
        for path in paths
    )

    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        agent_health = client.get("/agent/healthz")
        assert agent_health.status_code == 200
        assert agent_health.json()["backend"] == "deepagents"
        landing = client.get("/")
        assert landing.status_code == 200
        assert "HEADLESS DEEP AGENTS BACKEND" in landing.text
        assert "opentulpa connect http://testserver" in landing.text
        assert "agent chat interface" in landing.text
        assert client.get("/assets/app.js").status_code == 404
        assert landing.headers["cache-control"] == "no-store"


def test_v2_app_owns_service_lifespan_in_dependency_order() -> None:
    events: list[str] = []

    with TestClient(_app(events)):
        assert events == [
            "start:agent",
            "start:jobs",
            "start:intake",
            "start:triggers",
            "start:intake-poller",
        ]

    assert events == [
        "start:agent",
        "start:jobs",
        "start:intake",
        "start:triggers",
        "start:intake-poller",
        "stop:intake-poller",
        "stop:triggers",
        "stop:browser",
        "stop:intake",
        "stop:jobs",
        "stop:agent",
        "stop:telegram",
    ]


def test_v2_app_restores_capabilities_before_resuming_approved_runs() -> None:
    events: list[str] = []

    with TestClient(_app(events, with_capability_service=True)):
        assert events == [
            "start:agent:recovery-deferred",
            "start:jobs",
            "start:intake",
            "start:capabilities",
            "recover:agent",
            "start:triggers",
            "start:intake-poller",
        ]


def test_composio_callback_escapes_provider_query_values() -> None:
    events: list[str] = []

    with TestClient(_app(events)) as client:
        response = client.get(
            "/webhook/composio/callback",
            params={
                "toolkit": "<script>alert(1)</script>",
                "connectedAccountId": "connection-1",
            },
        )

    assert response.status_code == 200
    assert "<script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "connection-1" in response.text
