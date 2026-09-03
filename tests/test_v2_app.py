from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from opentulpa import __main__ as main_module
from opentulpa.api.app import create_app
from opentulpa.bootstrap.models import IngressEnvelope, OutboxEvent
from opentulpa.core.release_runtime import release_consumers_enabled
from opentulpa.evolution.models import EvolutionEvent
from opentulpa.specs import AgentSpecRef
from opentulpa.specs.defaults import default_agent_spec_writes


@dataclass(frozen=True)
class _Principal:
    tenant_id: str = "tenant-a"
    actor_id: str = "actor-a"


class _AsyncLifecycle:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_at: str | None = None,
        shutdown_started: asyncio.Event | None = None,
        finish_shutdown: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_at = fail_at
        self.shutdown_started = shutdown_started
        self.finish_shutdown = finish_shutdown
        self.started = False
        self.requests: list[Any] = []

    def healthy(self) -> bool:
        return self.started

    async def start_standby(self) -> None:
        self.events.append(f"start:{self.name}:standby")
        if self.fail_at == self.name:
            raise RuntimeError(f"failed:{self.name}")
        self.started = True

    async def start(self, *, recover_pending_resumes: bool = True) -> None:
        if self.name == "agent" and not recover_pending_resumes:
            self.events.append("start:agent:recovery-deferred")
            if self.fail_at == self.name:
                raise RuntimeError(f"failed:{self.name}")
            self.started = True
            return
        self.events.append(f"start:{self.name}")
        if self.fail_at == self.name:
            raise RuntimeError(f"failed:{self.name}")
        self.started = True

    async def recover_pending_resumes(self) -> None:
        self.events.append(f"recover:{self.name}")
        if self.fail_at == "recovery":
            raise RuntimeError("failed:recovery")

    async def open_stream(self, request: Any) -> Any:
        self.requests.append(request)

        async def empty() -> Any:
            if False:
                yield None

        return empty()

    async def shutdown(self) -> None:
        self.events.append(f"stop:{self.name}")
        if self.shutdown_started is not None:
            self.shutdown_started.set()
        if self.finish_shutdown is not None:
            await self.finish_shutdown.wait()
        self.started = False


class _SyncDispatcher:
    def __init__(self, name: str, events: list[str], *, fail_at: str | None = None) -> None:
        self.name = name
        self.events = events
        self.fail_at = fail_at

    def start(self) -> None:
        self.events.append(f"start:{self.name}")
        if self.fail_at == self.name:
            raise RuntimeError(f"failed:{self.name}")

    def shutdown(self, *, wait: bool = True) -> None:
        assert wait is True
        self.events.append(f"stop:{self.name}")

    def upsert(self, value: Any) -> None:
        _ = value

    def remove(self, **identifiers: str) -> None:
        _ = identifiers

    async def dispatch_event(self, **event: Any) -> None:
        self.events.append(f"dispatch:{event['source_event_id']}")


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


def _app(
    events: list[str],
    *,
    with_capability_service: bool = False,
    with_evolution_service: bool = False,
    fail_at: str | None = None,
    agent_shutdown_started: asyncio.Event | None = None,
    finish_agent_shutdown: asyncio.Event | None = None,
    startup_callback: Any | None = None,
    startup_callback_timeout_seconds: float = 20.0,
    notification_service: Any | None = None,
) -> Any:
    agent = _AsyncLifecycle(
        "agent",
        events,
        fail_at=fail_at,
        shutdown_started=agent_shutdown_started,
        finish_shutdown=finish_agent_shutdown,
    )
    jobs = _AsyncLifecycle("jobs", events, fail_at=fail_at)
    intake = _AsyncLifecycle("intake", events, fail_at=fail_at)
    trigger_dispatcher = _SyncDispatcher("triggers", events, fail_at=fail_at)
    intake_dispatcher = _SyncDispatcher("intake-poller", events, fail_at=fail_at)

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
        resolve_agent_spec=lambda tenant_id, spec_id: AgentSpecRef(
            tenant_id=tenant_id,
            spec_id=spec_id,
            revision=1,
        ),
        trigger_dispatcher=cast(Any, trigger_dispatcher),
        intake_poll_dispatcher=cast(Any, intake_dispatcher),
        meta_messenger_tenant_id="tenant-a",
        meta_messenger_trigger_id="meta-trigger",
        meta_messenger_verify_token="meta-verify-token",
        meta_app_secret="meta-app-secret",
        browser_service=cast(Any, _CloseOnly("browser", events)),
        telegram_client=cast(Any, _TelegramClient(events)),
        capability_service=(
            cast(Any, _AsyncLifecycle("capabilities", events, fail_at=fail_at))
            if with_capability_service
            else None
        ),
        evolution_service=(
            cast(Any, _AsyncLifecycle("evolution", events, fail_at=fail_at))
            if with_evolution_service
            else None
        ),
        notification_service=notification_service,
        startup_callback=startup_callback,
        startup_callback_timeout_seconds=startup_callback_timeout_seconds,
    )


def test_meta_messenger_uses_trigger_dispatcher_bound_method() -> None:
    events: list[str] = []
    payload = {
        "object": "page",
        "entry": [
            {
                "id": "page-1",
                "messaging": [
                    {
                        "sender": {"id": "user-1"},
                        "recipient": {"id": "page-1"},
                        "message": {"mid": "message-1", "text": "hello"},
                    }
                ],
            }
        ],
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    signature = hmac.new(b"meta-app-secret", body, hashlib.sha256).hexdigest()

    with TestClient(_app(events)) as client:
        response = client.post(
            "/webhook/meta/messenger",
            content=body,
            headers={"x-hub-signature-256": f"sha256={signature}"},
        )

    assert response.status_code == 200
    assert "dispatch:message-1" in events


def test_v2_app_exposes_only_cutover_routes_and_deepagents_health() -> None:
    events: list[str] = []
    app = _app(events)
    paths = {route.path for route in app.routes}

    assert "/healthz" in paths
    assert "/" in paths
    assert "/agent/healthz" in paths
    assert "/_runtime/identity" in paths
    assert "/_runtime/evolution-events" in paths
    assert "/v2/agent/runs" in paths
    assert "/v2/files" in paths
    assert "/v2/integrations" in paths
    assert "/v2/intake/workflows" in paths
    assert "/v2/schedules" in paths
    assert "/webhook/telegram" in paths
    assert "/webhook/meta/messenger" in paths
    assert "/webhook/composio/callback" in paths
    assert all(
        path
        in {
            "/",
            "/healthz",
            "/agent/healthz",
            "/_runtime/identity",
            "/_runtime/evolution-events",
        }
        or path.startswith("/v2/")
        or path in {
            "/webhook/telegram",
            "/webhook/meta/messenger",
            "/webhook/composio/callback",
        }
        for path in paths
    )

    with TestClient(app) as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        health_body = health.json()
        expected_health = {
            "status": "ok",
            "lifecycle": "ready",
            "consumers_enabled": True,
            "source_commit": None,
        }
        assert {key: health_body[key] for key in expected_health} == expected_health
        assert "launch_nonce" not in health_body
        assert health_body["started_at"]
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


def test_health_is_unavailable_when_capability_worker_is_unhealthy() -> None:
    events: list[str] = []
    app = _app(events, with_capability_service=True)

    with TestClient(app) as client:
        app.state.capability_service.started = False

        health = client.get("/healthz")

        assert health.status_code == 503
        assert health.json()["status"] == "unavailable"


def test_private_evolution_event_route_requires_exact_child_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Notifications:
        def __init__(self) -> None:
            self.published: list[dict[str, Any]] = []

        def publish(self, **values: Any) -> None:
            self.published.append(values)

    nonce = "private-event-launch-nonce-0000"
    token = "private-event-owner-token"
    notifications = Notifications()
    monkeypatch.setenv("OPENTULPA_LAUNCH_NONCE", nonce)
    monkeypatch.setenv("OPENTULPA_OWNER_TOKEN", token)
    event = EvolutionEvent(
        event_key="candidate:candidate-1:failed",
        event_type="candidate.failed",
        release_id="release-1",
        origin={
            "tenant_id": "tenant-a",
            "channel": "web",
            "correlation_id": "correlation-1",
        },
        payload={"status": "failed"},
    )

    with TestClient(_app([], notification_service=notifications)) as client:
        assert client.post(
            "/_runtime/evolution-events",
            headers={
                "Authorization": f"Bearer {token}",
                "X-OpenTulpa-Launch-Nonce": "wrong-nonce-0000000000",
            },
            content=event.model_dump_json(),
        ).status_code == 401
        assert client.post(
            "/_runtime/evolution-events",
            headers={
                "Authorization": "Bearer wrong-token",
                "X-OpenTulpa-Launch-Nonce": nonce,
            },
            content=event.model_dump_json(),
        ).status_code == 401
        delivered = client.post(
            "/_runtime/evolution-events",
            headers={
                "Authorization": f"Bearer {token}",
                "X-OpenTulpa-Launch-Nonce": nonce,
            },
            content=event.model_dump_json(),
        )

    assert delivered.status_code == 204
    assert len(notifications.published) == 1
    assert notifications.published[0]["tenant_id"] == "tenant-a"


def test_failed_deployment_schedules_bounded_repair_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Notifications:
        def __init__(self) -> None:
            self.published: list[dict[str, Any]] = []

        def publish(self, **values: Any) -> None:
            self.published.append(values)

    nonce = "repair-event-launch-nonce-0000"
    token = "repair-event-owner-token-000000"
    monkeypatch.setenv("OPENTULPA_LAUNCH_NONCE", nonce)
    monkeypatch.setenv("OPENTULPA_OWNER_TOKEN", token)
    notifications = Notifications()
    app = _app([], notification_service=notifications)
    event = EvolutionEvent(
        event_key="source:activation-1:rolled_back",
        event_type="promotion.failed",
        release_id="release-1",
        origin={"tenant_id": "tenant-a", "correlation_id": "owner-request"},
        payload={
            "status": "rolled_back",
            "failure_phase": "source checks",
            "failure_message": "source activation check python.compile failed",
            "supervision": {
                "status": "completed",
                "summary": "One observation remains.",
                "findings": ["Fix the source boundary.", "[P2] Add a regression test.", 3],
            },
        },
    )

    with TestClient(app) as client:
        response = client.post(
            "/_runtime/evolution-events",
            headers={
                "Authorization": f"Bearer {token}",
                "X-OpenTulpa-Launch-Nonce": nonce,
            },
            content=event.model_dump_json(),
        )
        limited = client.post(
            "/_runtime/evolution-events",
            headers={
                "Authorization": f"Bearer {token}",
                "X-OpenTulpa-Launch-Nonce": nonce,
            },
            content=event.model_copy(
                update={
                    "event_key": "source:activation-3:rolled_back",
                    "origin": {
                        "tenant_id": "tenant-a",
                        "correlation_id": "evolution-repair:3:activation-2",
                    },
                }
            ).model_dump_json(),
        )

    assert response.status_code == 204
    assert limited.status_code == 204
    assert len(app.state.agent_service.requests) == 1
    request = app.state.agent_service.requests[0]
    assert request.context.agent_spec.spec_id == "release-repair"
    assert set(default_agent_spec_writes()["release-repair"].tools) == {
        "source_status",
        "source_read",
        "source_write",
        "source_edit",
        "source_bash",
        "source_activate",
    }
    assert request.context.correlation_id.startswith("evolution-repair:1:")
    assert "source_activate" in request.text
    assert "source checks" in request.text
    assert "source activation check python.compile failed" in request.text
    assert "Fix the source boundary." in request.text
    assert "[P2] Add a regression test." in request.text
    assert "\n3" not in request.text
    kinds = [item["notification"].kind for item in notifications.published]
    assert kinds == [
        "evolution.promotion.failed",
        "evolution.repair.started",
        "evolution.promotion.failed",
        "evolution.repair.exhausted",
    ]


def test_probation_child_rejects_private_evolution_event_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nonce = "probation-event-launch-nonce-000"
    token = "probation-event-owner-token"
    monkeypatch.setenv("OPENTULPA_LAUNCH_NONCE", nonce)
    monkeypatch.setenv("OPENTULPA_OWNER_TOKEN", token)
    monkeypatch.setenv("OPENTULPA_DISABLE_CONSUMERS", "true")
    event = EvolutionEvent(
        event_key="candidate:candidate-1:failed",
        event_type="candidate.failed",
        release_id="release-1",
        payload={"status": "failed"},
    )

    with TestClient(_app([], notification_service=object())) as client:
        response = client.post(
            "/_runtime/evolution-events",
            headers={
                "Authorization": f"Bearer {token}",
                "X-OpenTulpa-Launch-Nonce": nonce,
            },
            content=event.model_dump_json(),
        )

    assert response.status_code == 409


def test_v2_app_owns_service_lifespan_in_dependency_order() -> None:
    events: list[str] = []

    with TestClient(_app(events)):
        assert events == [
            "start:agent:recovery-deferred",
            "recover:agent",
            "start:jobs",
            "start:intake",
            "start:triggers",
            "start:intake-poller",
        ]

    assert events == [
        "start:agent:recovery-deferred",
        "recover:agent",
        "start:jobs",
        "start:intake",
        "start:triggers",
        "start:intake-poller",
        "stop:intake-poller",
        "stop:triggers",
        "stop:intake",
        "stop:jobs",
        "stop:agent",
        "stop:telegram",
        "stop:browser",
    ]


def test_v2_app_restores_capabilities_before_resuming_approved_runs() -> None:
    events: list[str] = []

    with TestClient(_app(events, with_capability_service=True)):
        assert events == [
            "start:agent:recovery-deferred",
            "start:capabilities",
            "recover:agent",
            "start:jobs",
            "start:intake",
            "start:triggers",
            "start:intake-poller",
        ]


def test_live_source_health_echoes_exact_runtime_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    source_commit = "a" * 40
    launch_nonce = "exact launch nonce value"
    monkeypatch.setenv("OPENTULPA_SOURCE_COMMIT", source_commit)
    monkeypatch.setenv("OPENTULPA_LAUNCH_NONCE", launch_nonce)
    events: list[str] = []

    with TestClient(_app(events)) as client:
        for path in ("/healthz", "/agent/healthz"):
            response = client.get(path)
            assert response.status_code == 200
            assert response.json()["source_commit"] == source_commit
            assert "launch_nonce" not in response.json()
        assert client.get("/_runtime/identity").status_code == 401
        assert (
            client.get(
                "/_runtime/identity",
                headers={"X-OpenTulpa-Launch-Nonce": f"{launch_nonce}-wrong"},
            ).status_code
            == 401
        )
        identity = client.get(
            "/_runtime/identity",
            headers={"X-OpenTulpa-Launch-Nonce": launch_nonce},
        )
        assert identity.status_code == 200
        assert identity.json() == {
            "source_commit": source_commit,
            "launch_nonce": launch_nonce,
        }


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"OPENTULPA_SOURCE_COMMIT": "a" * 40}, "launch nonce"),
        (
            {
                "OPENTULPA_SOURCE_COMMIT": f"{'a' * 40} ",
                "OPENTULPA_LAUNCH_NONCE": "n" * 32,
            },
            "identity",
        ),
        (
            {
                "OPENTULPA_SOURCE_COMMIT": "a" * 40,
                "OPENTULPA_LAUNCH_NONCE": "short",
            },
            "launch nonce",
        ),
    ],
)
def test_live_source_mode_rejects_missing_or_invalid_identity(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    message: str,
) -> None:
    for name in ("OPENTULPA_SOURCE_COMMIT", "OPENTULPA_LAUNCH_NONCE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        _app([])


@pytest.mark.asyncio
async def test_health_readiness_before_during_drain_and_after_lifespan() -> None:
    events: list[str] = []
    shutdown_started = asyncio.Event()
    finish_shutdown = asyncio.Event()
    app = _app(
        events,
        agent_shutdown_started=shutdown_started,
        finish_agent_shutdown=finish_shutdown,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        before = await client.get("/healthz")
        assert before.status_code == 503
        assert before.json()["lifecycle"] == "starting"

        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        ready = await client.get("/healthz")
        assert ready.status_code == 200
        assert ready.json()["lifecycle"] == "ready"

        shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
        await shutdown_started.wait()
        draining = await client.get("/healthz")
        assert draining.status_code == 503
        assert draining.json()["lifecycle"] == "stopping"

        finish_shutdown.set()
        await shutdown
        stopped = await client.get("/healthz")
        assert stopped.status_code == 503
        assert stopped.json()["lifecycle"] == "stopped"


def test_consumer_disabled_probation_starts_no_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_DISABLE_CONSUMERS", " YeS ")
    events: list[str] = []
    app = _app(
        events,
        with_capability_service=True,
        with_evolution_service=True,
    )

    with TestClient(app) as client:
        assert events == ["start:agent:standby"]
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["consumers_enabled"] is False

    assert events == ["start:agent:standby", "stop:agent"]


def test_consumer_disabled_partial_standby_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_DISABLE_CONSUMERS", "true")
    events: list[str] = []
    app = _app(events, fail_at="agent")

    with pytest.raises(RuntimeError, match="failed:agent"), TestClient(app):
        pass

    assert events == ["start:agent:standby", "stop:agent"]
    assert app.state.lifecycle_status == "failed"


def test_normal_startup_restores_before_recovery_and_producers() -> None:
    events: list[str] = []

    with TestClient(
        _app(
            events,
            with_capability_service=True,
            with_evolution_service=True,
        )
    ):
        assert events == [
            "start:agent:recovery-deferred",
            "start:capabilities",
            "start:evolution",
            "recover:agent",
            "start:jobs",
            "start:intake",
            "start:triggers",
            "start:intake-poller",
        ]

    assert events[-9:] == [
        "stop:intake-poller",
        "stop:triggers",
        "stop:intake",
        "stop:jobs",
        "stop:agent",
        "stop:telegram",
        "stop:browser",
        "stop:evolution",
        "stop:capabilities",
    ]


@pytest.mark.parametrize(
    ("fail_at", "expected"),
    [
        ("agent", ["start:agent:recovery-deferred", "stop:agent"]),
        (
            "capabilities",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "stop:agent",
                "stop:capabilities",
            ],
        ),
        (
            "evolution",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "start:evolution",
                "stop:agent",
                "stop:evolution",
                "stop:capabilities",
            ],
        ),
        (
            "recovery",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "start:evolution",
                "recover:agent",
                "stop:agent",
                "stop:evolution",
                "stop:capabilities",
            ],
        ),
        (
            "jobs",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "start:evolution",
                "recover:agent",
                "start:jobs",
                "stop:jobs",
                "stop:agent",
                "stop:browser",
                "stop:evolution",
                "stop:capabilities",
            ],
        ),
        (
            "intake",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "start:evolution",
                "recover:agent",
                "start:jobs",
                "start:intake",
                "stop:intake",
                "stop:jobs",
                "stop:agent",
                "stop:telegram",
                "stop:browser",
                "stop:evolution",
                "stop:capabilities",
            ],
        ),
        (
            "triggers",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "start:evolution",
                "recover:agent",
                "start:jobs",
                "start:intake",
                "start:triggers",
                "stop:triggers",
                "stop:intake",
                "stop:jobs",
                "stop:agent",
                "stop:telegram",
                "stop:browser",
                "stop:evolution",
                "stop:capabilities",
            ],
        ),
        (
            "intake-poller",
            [
                "start:agent:recovery-deferred",
                "start:capabilities",
                "start:evolution",
                "recover:agent",
                "start:jobs",
                "start:intake",
                "start:triggers",
                "start:intake-poller",
                "stop:intake-poller",
                "stop:triggers",
                "stop:intake",
                "stop:jobs",
                "stop:agent",
                "stop:telegram",
                "stop:browser",
                "stop:evolution",
                "stop:capabilities",
            ],
        ),
    ],
)
def test_partial_start_failure_cleans_attempted_boundaries_in_shutdown_phases(
    fail_at: str,
    expected: list[str],
) -> None:
    events: list[str] = []
    app = _app(
        events,
        with_capability_service=True,
        with_evolution_service=True,
        fail_at=fail_at,
    )

    with pytest.raises(RuntimeError, match=f"failed:{fail_at}"), TestClient(app):
        pass

    assert app.state.lifecycle_status == "failed"
    assert events == expected


def test_probation_composition_does_not_construct_source_evolution_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_DISABLE_CONSUMERS", "true")
    monkeypatch.setenv("OPENTULPA_BOOTSTRAP_EVOLUTION_URL", "http://127.0.0.1:9000")
    monkeypatch.setenv("OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN", "e" * 48)

    client = main_module._build_evolution_client(
        settings=cast(Any, SimpleNamespace(evolution_enabled=True)),
    )

    assert client is None
    assert release_consumers_enabled() is False


@pytest.mark.parametrize("value", ["0", "false", "NO", " Off "])
def test_consumer_toggle_accepts_explicit_false_values(value: str) -> None:
    assert release_consumers_enabled({"OPENTULPA_DISABLE_CONSUMERS": value}) is True


@pytest.mark.parametrize("value", ["1", "true", "YES", " On "])
def test_consumer_toggle_accepts_explicit_true_values(value: str) -> None:
    assert release_consumers_enabled({"OPENTULPA_DISABLE_CONSUMERS": value}) is False


def test_consumer_toggle_rejects_malformed_explicit_values() -> None:
    with pytest.raises(RuntimeError, match="explicit boolean"):
        release_consumers_enabled({"OPENTULPA_DISABLE_CONSUMERS": "sometimes"})


def test_main_validates_live_source_identity_before_runtime_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def runtime_paths() -> Any:
        nonlocal called
        called = True
        raise AssertionError("runtime paths must not be resolved")

    monkeypatch.setenv("OPENTULPA_SOURCE_COMMIT", "a" * 40)
    monkeypatch.delenv("OPENTULPA_LAUNCH_NONCE", raising=False)
    monkeypatch.setattr(
        main_module.RuntimePaths,
        "from_environment",
        runtime_paths,
    )

    with pytest.raises(RuntimeError, match="launch nonce"):
        main_module.main()

    assert called is False


@pytest.mark.asyncio
async def test_probation_release_control_rejects_ingress_and_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Agent:
        calls = 0

        def healthy(self) -> bool:
            return True

        async def run(self, request: Any) -> None:
            del request
            self.calls += 1

    class Capabilities:
        async def healthy(self) -> bool:
            return True

    agent = Agent()
    monkeypatch.setenv("OPENTULPA_DISABLE_CONSUMERS", "true")
    monkeypatch.setenv("OPENTULPA_RELEASE_ID", "release-probation")
    monkeypatch.setenv("OPENTULPA_CONTROL_TOKEN", "c" * 48)
    service = main_module._build_release_control_service(
        agent_service=cast(Any, agent),
        resolve_agent_spec=lambda *_: object(),
        secret_ingress=cast(Any, lambda **_: "sanitized"),
        notifications=cast(Any, object()),
        capabilities=cast(Any, Capabilities()),
    )
    assert service is not None
    assert service.ingress_handler is None
    assert service.event_handler is None

    envelope = IngressEnvelope(
        tenant_id="tenant",
        thread_id="thread",
        channel="owner",
        idempotency_key="ingress-key",
        payload={"text": "must not run"},
    )
    event = OutboxEvent(
        event_key="event-key",
        event_type="release.completed",
        payload={},
    )
    with pytest.raises(HTTPException) as ingress_error:
        await service.accept_ingress(envelope, idempotency_key=envelope.idempotency_key)
    with pytest.raises(HTTPException) as event_error:
        await service.accept_event(event, idempotency_key=event.event_key)

    assert ingress_error.value.status_code == 503
    assert event_error.value.status_code == 503
    assert agent.calls == 0


def test_startup_callback_runs_after_producers_and_failure_is_best_effort() -> None:
    events: list[str] = []

    async def callback() -> None:
        events.append("startup-callback")
        raise RuntimeError("bounded callback failure")

    with TestClient(_app(events, startup_callback=callback)) as client:
        assert events[-1] == "startup-callback"
        assert client.get("/healthz").status_code == 200

    assert events[-6:] == [
        "stop:triggers",
        "stop:intake",
        "stop:jobs",
        "stop:agent",
        "stop:telegram",
        "stop:browser",
    ]


def test_startup_callback_is_time_bounded() -> None:
    events: list[str] = []

    async def callback() -> None:
        events.append("startup-callback")
        await asyncio.Event().wait()

    with TestClient(
        _app(
            events,
            startup_callback=callback,
            startup_callback_timeout_seconds=0.001,
        )
    ) as client:
        assert client.get("/healthz").status_code == 200
        assert events[-1] == "startup-callback"

    assert "stop:agent" in events


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
