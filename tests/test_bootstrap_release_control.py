from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI

from opentulpa.bootstrap.models import IngressEnvelope, OutboxEvent
from opentulpa.bootstrap.release_control import (
    ReleaseControlConfigurationError,
    ReleaseControlService,
    register_release_control_plane,
)


@pytest.mark.asyncio
async def test_release_control_authenticates_delivery_health_and_drain() -> None:
    ingress: list[IngressEnvelope] = []
    events: list[OutboxEvent] = []
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()

    async def handle_ingress(envelope: IngressEnvelope) -> None:
        ingress.append(envelope)

    async def handle_event(event: OutboxEvent) -> None:
        events.append(event)

    service = ReleaseControlService(
        release_id="release_blue",
        lease_epoch=4,
        control_token="t" * 32,
        health_provider=lambda: {"runtime": True, "agent_api": True},
        ingress_handler=handle_ingress,
        event_handler=handle_event,
    )
    app = FastAPI()

    @app.get("/slow")
    async def slow() -> dict[str, bool]:
        slow_started.set()
        await release_slow.wait()
        return {"complete": True}

    register_release_control_plane(app, service)
    headers = {
        "Authorization": f"Bearer {'t' * 32}",
        "X-OpenTulpa-Release-ID": "release_blue",
        "X-OpenTulpa-Lease-Epoch": "4",
        "X-OpenTulpa-Control-Token": "t" * 32,
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://release",
    ) as client:
        assert (await client.get(service.health_path)).status_code == 401
        health = await client.get(service.health_path, headers=headers)
        assert health.status_code == 200
        assert health.json()["healthy"] is True

        envelope = IngressEnvelope(
            tenant_id="tenant_1",
            thread_id="thread_1",
            channel="telegram",
            idempotency_key="telegram:update:1",
            payload={"text": "hello"},
        )
        accepted = await client.post(
            service.ingress_path,
            headers={**headers, "Idempotency-Key": envelope.idempotency_key},
            json=envelope.model_dump(mode="json"),
        )
        assert accepted.status_code == 204
        assert ingress == [envelope]

        event = OutboxEvent(
            event_key="activation:1:active",
            event_type="release.active",
            payload={"release_id": "release_blue"},
        )
        delivered = await client.post(
            service.event_path,
            headers={**headers, "Idempotency-Key": event.event_key},
            json=event.model_dump(mode="json"),
        )
        assert delivered.status_code == 204
        assert events == [event]

        slow_task = asyncio.create_task(client.get("/slow", headers=headers))
        await slow_started.wait()
        timed_out = await client.post(
            service.drain_path,
            headers=headers,
            json={"timeout_seconds": 0.01},
        )
        assert timed_out.json() == {"drained": False, "in_flight": 1}
        release_slow.set()
        assert (await slow_task).status_code == 200
        drained = await client.post(
            service.drain_path,
            headers=headers,
            json={"timeout_seconds": 0.01},
        )
        assert drained.json() == {"drained": True, "in_flight": 0}


def test_release_control_environment_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENTULPA_RELEASE_ID", raising=False)
    monkeypatch.delenv("OPENTULPA_CONTROL_TOKEN", raising=False)

    with pytest.raises(ReleaseControlConfigurationError):
        ReleaseControlService.from_environment(
            health_provider=lambda: {"runtime": True, "agent_api": True}
        )
