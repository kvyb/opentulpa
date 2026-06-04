"""Side-effect-free Telegram webhook readiness checks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from opentulpa.core.public_urls import resolve_public_base_url
from opentulpa.interfaces.telegram.constants import TELEGRAM_WEBHOOK_ALLOWED_UPDATES

TELEGRAM_WEBHOOK_PATH = "/webhook/telegram"


@dataclass(frozen=True, slots=True)
class TelegramWebhookRuntimeConfig:
    bot_token_configured: bool
    webhook_secret_configured: bool
    public_url: str
    expected_webhook_url: str
    allowed_updates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TelegramWebhookInfo:
    available: bool
    url: str
    pending_update_count: int
    allowed_updates: tuple[str, ...]
    last_error_date: int | None = None
    last_error_message: str | None = None
    max_connections: int | None = None
    raw_error: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramWebhookReadiness:
    configured: bool
    public_url: str
    expected_webhook_url: str
    telegram_webhook_url: str
    ready: bool
    requires_webhook_reset: bool
    last_error_message: str | None
    reasons: tuple[str, ...]


def build_runtime_config(
    settings: Any,
    *,
    env: Mapping[str, str] | None = None,
) -> TelegramWebhookRuntimeConfig:
    public_url = resolve_public_base_url(env)
    bot_token = str(getattr(settings, "telegram_bot_token", "") or "").strip()
    webhook_secret = str(getattr(settings, "telegram_webhook_secret", "") or "").strip()
    expected_webhook_url = f"{public_url}{TELEGRAM_WEBHOOK_PATH}" if public_url else ""

    return TelegramWebhookRuntimeConfig(
        bot_token_configured=bool(bot_token),
        webhook_secret_configured=bool(webhook_secret),
        public_url=public_url,
        expected_webhook_url=expected_webhook_url,
        allowed_updates=tuple(TELEGRAM_WEBHOOK_ALLOWED_UPDATES),
    )


def telegram_webhook_info_from_result(result: Mapping[str, Any] | None) -> TelegramWebhookInfo:
    if not isinstance(result, Mapping):
        return TelegramWebhookInfo(
            available=False,
            url="",
            pending_update_count=0,
            allowed_updates=(),
            raw_error="telegram getWebhookInfo returned no result",
        )

    return TelegramWebhookInfo(
        available=True,
        url=str(result.get("url", "") or "").strip(),
        pending_update_count=_safe_int(result.get("pending_update_count")),
        allowed_updates=_string_tuple(result.get("allowed_updates")),
        last_error_date=_optional_int(result.get("last_error_date")),
        last_error_message=_optional_str(result.get("last_error_message")),
        max_connections=_optional_int(result.get("max_connections")),
    )


def evaluate_webhook_readiness(
    *,
    runtime: TelegramWebhookRuntimeConfig,
    telegram: TelegramWebhookInfo,
) -> TelegramWebhookReadiness:
    reasons: list[str] = []
    requires_webhook_reset = False

    if not runtime.bot_token_configured:
        reasons.append("telegram_bot_token_missing")
    if not runtime.webhook_secret_configured:
        reasons.append("telegram_webhook_secret_missing")
    if not runtime.public_url or not runtime.expected_webhook_url:
        reasons.append("public_base_url_missing")

    configured = not reasons
    if configured:
        if not telegram.available:
            reasons.append("telegram_webhook_status_unavailable")
        elif not telegram.url:
            reasons.append("telegram_webhook_url_missing")
            requires_webhook_reset = True
        elif telegram.url != runtime.expected_webhook_url:
            reasons.append("telegram_webhook_url_mismatch")
            requires_webhook_reset = True

        if telegram.available:
            if not telegram.allowed_updates:
                reasons.append("telegram_allowed_updates_missing")
                requires_webhook_reset = True
            elif set(telegram.allowed_updates) != set(runtime.allowed_updates):
                reasons.append("telegram_allowed_updates_mismatch")
                requires_webhook_reset = True

    ready = configured and not reasons
    assert not (ready and requires_webhook_reset)
    return TelegramWebhookReadiness(
        configured=configured,
        public_url=runtime.public_url,
        expected_webhook_url=runtime.expected_webhook_url,
        telegram_webhook_url=telegram.url,
        ready=ready,
        requires_webhook_reset=requires_webhook_reset,
        last_error_message=telegram.last_error_message,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _optional_str(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    parsed = _optional_int(value)
    return max(0, parsed or 0)
