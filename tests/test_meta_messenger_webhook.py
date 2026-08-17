from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.meta_messenger import register_meta_messenger_routes

APP_SECRET = "meta-app-secret"
VERIFY_TOKEN = "meta-verify-token"


class _Dispatcher:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, **event: Any) -> None:
        self.events.append(event)


def _app(dispatcher: _Dispatcher | None = None) -> tuple[FastAPI, _Dispatcher]:
    resolved_dispatcher = dispatcher or _Dispatcher()
    app = FastAPI()
    register_meta_messenger_routes(
        app,
        get_dispatcher=lambda: resolved_dispatcher,
        tenant_id="tenant-1",
        trigger_id="meta-messenger-message",
        verify_token=VERIFY_TOKEN,
        app_secret=APP_SECRET,
    )
    return app, resolved_dispatcher


def _signed_body(payload: dict[str, Any]) -> tuple[bytes, str]:
    body = json.dumps(payload, separators=(",", ":")).encode()
    digest = hmac.new(APP_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return body, f"sha256={digest}"


def test_meta_webhook_verification_returns_challenge() -> None:
    app, _ = _app()

    response = TestClient(app).get(
        "/webhook/meta/messenger",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "123456789",
        },
    )

    assert response.status_code == 200
    assert response.text == "123456789"
    assert response.headers["content-type"].startswith("text/plain")


def test_meta_webhook_verification_rejects_wrong_token() -> None:
    app, _ = _app()

    response = TestClient(app).get(
        "/webhook/meta/messenger",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "123456789",
        },
    )

    assert response.status_code == 403


def test_meta_webhook_checks_signature_and_dispatches_page_messages() -> None:
    app, dispatcher = _app()
    payload = {
        "object": "page",
        "entry": [
            {
                "id": "page-1",
                "time": 1_700_000_000,
                "messaging": [
                    {
                        "sender": {"id": "user-1"},
                        "recipient": {"id": "page-1"},
                        "timestamp": 1_700_000_001,
                        "message": {"mid": "message-1", "text": "Привет"},
                    },
                    {
                        "sender": {"id": "page-1"},
                        "recipient": {"id": "user-1"},
                        "timestamp": 1_700_000_002,
                        "message": {
                            "mid": "message-echo",
                            "text": "Не зацикливаться",
                            "is_echo": True,
                        },
                    },
                ],
            }
        ],
    }
    body, signature = _signed_body(payload)

    response = TestClient(app).post(
        "/webhook/meta/messenger",
        content=body,
        headers={
            "content-type": "application/json",
            "x-hub-signature-256": signature,
        },
    )

    assert response.status_code == 200
    assert response.text == "EVENT_RECEIVED"
    assert dispatcher.events == [
        {
            "tenant_id": "tenant-1",
            "trigger_id": "meta-messenger-message",
            "source_event_id": "message-1",
            "event_type": "message_received",
            "source": "meta_messenger",
            "authenticated": True,
            "payload": {
                "page_id": "page-1",
                "sender_id": "user-1",
                "recipient_id": "page-1",
                "message_id": "message-1",
                "timestamp": 1_700_000_001,
                "text": "Привет",
            },
        }
    ]


def test_meta_webhook_rejects_invalid_signature_before_dispatch() -> None:
    app, dispatcher = _app()
    body = json.dumps({"object": "page", "entry": []}).encode()

    response = TestClient(app).post(
        "/webhook/meta/messenger",
        content=body,
        headers={"x-hub-signature-256": f"sha256={'0' * 64}"},
    )

    assert response.status_code == 401
    assert dispatcher.events == []


def test_meta_webhook_ignores_non_message_events() -> None:
    app, dispatcher = _app()
    body, signature = _signed_body(
        {
            "object": "page",
            "entry": [
                {
                    "id": "page-1",
                    "messaging": [
                        {
                            "sender": {"id": "user-1"},
                            "recipient": {"id": "page-1"},
                            "delivery": {"mids": ["message-1"]},
                        }
                    ],
                }
            ],
        }
    )

    response = TestClient(app).post(
        "/webhook/meta/messenger",
        content=body,
        headers={"x-hub-signature-256": signature},
    )

    assert response.status_code == 200
    assert dispatcher.events == []


def test_meta_webhook_verification_does_not_require_app_secret() -> None:
    app = FastAPI()
    register_meta_messenger_routes(
        app,
        get_dispatcher=lambda: None,
        tenant_id="tenant-1",
        trigger_id="meta-messenger-message",
        verify_token=VERIFY_TOKEN,
        app_secret=None,
    )

    response = TestClient(app).get(
        "/webhook/meta/messenger",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "123456789",
        },
    )

    assert response.status_code == 200
    assert response.text == "123456789"


def test_meta_webhook_returns_unavailable_without_verify_token() -> None:
    app = FastAPI()
    register_meta_messenger_routes(
        app,
        get_dispatcher=lambda: None,
        tenant_id="tenant-1",
        trigger_id="meta-messenger-message",
        verify_token=None,
        app_secret=None,
    )

    response = TestClient(app).get(
        "/webhook/meta/messenger",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": VERIFY_TOKEN,
            "hub.challenge": "123456789",
        },
    )

    assert response.status_code == 503
