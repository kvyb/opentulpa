from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.telegram_webhook_health import register_telegram_webhook_health_routes
from opentulpa.interfaces.telegram.constants import TELEGRAM_WEBHOOK_ALLOWED_UPDATES
from opentulpa.interfaces.telegram.webhook_health import (
    TelegramWebhookInfo,
    build_runtime_config,
    evaluate_webhook_readiness,
)


class _FakeTelegramClient:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls = 0

    async def get_webhook_info(self) -> dict[str, Any]:
        self.calls += 1
        return self.result


class _FailingTelegramClient:
    async def get_webhook_info(self) -> dict[str, Any]:
        raise RuntimeError("network down for token 123:abc")


def _settings(*, token: str = "123:abc", secret: str = "secret") -> SimpleNamespace:
    return SimpleNamespace(
        telegram_bot_token=token,
        telegram_webhook_secret=secret,
        opentulpa_web_token="web-secret",
    )


def test_runtime_config_reports_expected_webhook_url(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com/")

    config = build_runtime_config(_settings())

    assert config.bot_token_configured is True
    assert config.webhook_secret_configured is True
    assert config.public_url == "https://app.example.com"
    assert config.expected_webhook_url == "https://app.example.com/webhook/telegram"
    assert config.allowed_updates == tuple(TELEGRAM_WEBHOOK_ALLOWED_UPDATES)


def test_readiness_reports_missing_runtime_config_without_redeploy_opinion(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    runtime = build_runtime_config(_settings(token="", secret=""))

    result = evaluate_webhook_readiness(
        runtime=runtime,
        telegram=TelegramWebhookInfo(
            available=False,
            url="",
            pending_update_count=0,
            allowed_updates=(),
        ),
    )

    assert result.configured is False
    assert result.ready is False
    assert result.requires_webhook_reset is False
    assert result.reasons == (
        "telegram_bot_token_missing",
        "telegram_webhook_secret_missing",
        "public_base_url_missing",
    )


def test_readiness_requires_webhook_reset_for_telegram_url_mismatch(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    runtime = build_runtime_config(_settings())

    result = evaluate_webhook_readiness(
        runtime=runtime,
        telegram=TelegramWebhookInfo(
            available=True,
            url="https://old.example.com/webhook/telegram",
            pending_update_count=0,
            allowed_updates=tuple(TELEGRAM_WEBHOOK_ALLOWED_UPDATES),
        ),
    )

    assert result.configured is True
    assert result.ready is False
    assert result.requires_webhook_reset is True
    assert result.reasons == ("telegram_webhook_url_mismatch",)


def test_readiness_accepts_matching_webhook_and_allowed_updates(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    runtime = build_runtime_config(_settings())

    result = evaluate_webhook_readiness(
        runtime=runtime,
        telegram=TelegramWebhookInfo(
            available=True,
            url="https://app.example.com/webhook/telegram",
            pending_update_count=0,
            allowed_updates=tuple(reversed(TELEGRAM_WEBHOOK_ALLOWED_UPDATES)),
        ),
    )

    assert result.configured is True
    assert result.ready is True
    assert result.requires_webhook_reset is False
    assert result.reasons == ()


def test_readiness_reports_last_error_without_blocking_matching_webhook(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    runtime = build_runtime_config(_settings())

    result = evaluate_webhook_readiness(
        runtime=runtime,
        telegram=TelegramWebhookInfo(
            available=True,
            url="https://app.example.com/webhook/telegram",
            pending_update_count=0,
            allowed_updates=tuple(TELEGRAM_WEBHOOK_ALLOWED_UPDATES),
            last_error_message="historical delivery error",
        ),
    )

    assert result.ready is True
    assert result.requires_webhook_reset is False
    assert result.last_error_message == "historical delivery error"
    assert result.reasons == ()


def test_web_status_route_reports_ready_with_mocked_telegram(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    fake_client = _FakeTelegramClient(
        {
            "url": "https://app.example.com/webhook/telegram",
            "pending_update_count": 0,
            "allowed_updates": list(TELEGRAM_WEBHOOK_ALLOWED_UPDATES),
        }
    )
    app = FastAPI()
    register_telegram_webhook_health_routes(
        app,
        settings=_settings(),
        get_telegram_client=lambda: fake_client,
        web_token="web-secret",
    )

    with TestClient(app) as client:
        rejected = client.get("/web/telegram/status")
        accepted = client.get(
            "/web/telegram/status",
            headers={"authorization": "Bearer web-secret"},
        )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    payload = accepted.json()
    assert payload["configured"] is True
    assert payload["ready"] is True
    assert payload["requires_webhook_reset"] is False
    assert payload["expected_webhook_url"] == "https://app.example.com/webhook/telegram"
    assert payload["telegram_webhook_url"] == "https://app.example.com/webhook/telegram"
    assert payload["reasons"] == []
    assert fake_client.calls == 1


def test_web_status_route_reports_unavailable_when_web_token_missing(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    fake_client = _FakeTelegramClient(
        {
            "url": "https://app.example.com/webhook/telegram",
            "pending_update_count": 0,
            "allowed_updates": list(TELEGRAM_WEBHOOK_ALLOWED_UPDATES),
        }
    )
    app = FastAPI()
    register_telegram_webhook_health_routes(
        app,
        settings=_settings(),
        get_telegram_client=lambda: fake_client,
        web_token=None,
    )

    with TestClient(app) as client:
        response = client.get(
            "/web/telegram/status",
            headers={"authorization": "Bearer web-secret"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "OPENTULPA_WEB_TOKEN is not configured"}
    assert fake_client.calls == 0


def test_web_status_route_returns_unavailable_when_telegram_raises(monkeypatch: Any) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    app = FastAPI()
    register_telegram_webhook_health_routes(
        app,
        settings=_settings(),
        get_telegram_client=lambda: _FailingTelegramClient(),
        web_token="web-secret",
    )

    with TestClient(app) as client:
        response = client.get(
            "/web/telegram/status",
            headers={"authorization": "Bearer web-secret"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["ready"] is False
    assert payload["telegram"]["available"] is False
    assert payload["telegram"]["raw_error"] == "RuntimeError: network down for token [redacted]"
    assert payload["reasons"] == ["telegram_webhook_status_unavailable"]
