"""Telegram webhook readiness routes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel

from opentulpa.api.web_auth import web_auth_error
from opentulpa.interfaces.telegram.webhook_health import (
    TelegramWebhookInfo,
    build_runtime_config,
    evaluate_webhook_readiness,
    telegram_webhook_info_from_result,
)


class TelegramWebhookInfoResponse(BaseModel):
    available: bool
    url: str
    pending_update_count: int
    allowed_updates: list[str]
    last_error_date: int | None = None
    last_error_message: str | None = None
    max_connections: int | None = None
    raw_error: str | None = None


class TelegramWebhookStatusResponse(BaseModel):
    configured: bool
    public_url: str
    expected_webhook_url: str
    telegram_webhook_url: str
    ready: bool
    requires_webhook_reset: bool
    last_error_message: str | None
    reasons: list[str]
    telegram: TelegramWebhookInfoResponse


def register_telegram_webhook_health_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
    web_token: str | None,
) -> None:
    """Register side-effect-free Telegram webhook readiness checks."""

    @app.get(
        "/internal/telegram/status",
        response_model=TelegramWebhookStatusResponse,
    )
    async def internal_telegram_status() -> TelegramWebhookStatusResponse:
        return await _telegram_status_response(
            settings=settings,
            get_telegram_client=get_telegram_client,
        )

    @app.get(
        "/web/telegram/status",
        response_model=TelegramWebhookStatusResponse,
    )
    async def web_telegram_status(request: Request) -> Any:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error
        return await _telegram_status_response(
            settings=settings,
            get_telegram_client=get_telegram_client,
        )


async def _telegram_status_response(
    *,
    settings: Any,
    get_telegram_client: Callable[[], Any],
) -> TelegramWebhookStatusResponse:
    runtime = build_runtime_config(settings)
    telegram = telegram_webhook_info_from_result(None)

    if runtime.bot_token_configured:
        try:
            client = get_telegram_client()
            get_webhook_info = getattr(client, "get_webhook_info", None)
            if callable(get_webhook_info):
                telegram = telegram_webhook_info_from_result(await get_webhook_info())
        except Exception as exc:
            telegram = _telegram_status_error(exc, settings=settings)

    readiness = evaluate_webhook_readiness(runtime=runtime, telegram=telegram)
    return TelegramWebhookStatusResponse(
        **asdict(readiness),
        telegram=TelegramWebhookInfoResponse.model_validate(asdict(telegram)),
    )


def _telegram_status_error(exc: Exception, *, settings: Any) -> TelegramWebhookInfo:
    return TelegramWebhookInfo(
        available=False,
        url="",
        pending_update_count=0,
        allowed_updates=(),
        raw_error=_sanitize_telegram_error(exc, settings=settings),
    )


def _sanitize_telegram_error(exc: Exception, *, settings: Any) -> str:
    message = " ".join(str(exc).split()) or exc.__class__.__name__
    for secret_name in (
        "telegram_bot_token",
        "telegram_webhook_secret",
        "opentulpa_web_token",
    ):
        secret = str(getattr(settings, secret_name, "") or "").strip()
        if secret:
            message = message.replace(secret, "[redacted]")
    return f"{exc.__class__.__name__}: {message[:160]}"
