from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.telegram_deep_agent import register_telegram_deep_agent_routes


class _Relay:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.accepted: dict[str, Any] | None = None

    async def accept_update(self, body: dict[str, Any]) -> dict[str, Any]:
        self.accepted = {"durable": True, "body": body}
        self.order.append("accepted")
        return self.accepted

    async def process_update(self, accepted: dict[str, Any]) -> None:
        assert accepted is self.accepted
        self.order.append("processed")


@pytest.mark.asyncio
async def test_webhook_sends_200_after_accept_and_before_background_processing() -> None:
    order: list[str] = []
    relay = _Relay(order)
    app = FastAPI()
    register_telegram_deep_agent_routes(
        app,
        get_relay=lambda: relay,
        webhook_secret="webhook-secret",
    )
    body = json.dumps({"update_id": 1, "business_message": {"message_id": 10}}).encode()
    requests = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict[str, Any]:
        if requests:
            return requests.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and not message.get("more_body", False):
            order.append("response")

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "https",
            "path": "/webhook/telegram",
            "raw_path": b"/webhook/telegram",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-telegram-bot-api-secret-token", b"webhook-secret"),
            ],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 443),
        },
        receive,
        send,
    )

    assert order == ["accepted", "response", "processed"]


def test_webhook_acceptance_failure_is_generic() -> None:
    class FailingRelay:
        async def accept_update(self, body: dict[str, Any]) -> Any:
            del body
            raise RuntimeError(
                "provider body token=private-secret from /srv/private/.env"
            )

        async def process_update(self, accepted: Any) -> None:
            del accepted

    app = FastAPI()
    register_telegram_deep_agent_routes(
        app,
        get_relay=FailingRelay,
        webhook_secret="webhook-secret",
    )
    response = TestClient(app).post(
        "/webhook/telegram",
        headers={"x-telegram-bot-api-secret-token": "webhook-secret"},
        json={"update_id": 1, "business_message": {"message_id": 10}},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Telegram update could not be accepted"
    assert "private-secret" not in response.text
    assert "/srv/private" not in response.text
