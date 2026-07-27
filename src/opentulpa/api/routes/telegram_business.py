"""Authenticated webhook for Telegram Business intake updates."""

from __future__ import annotations

import logging
from collections.abc import Callable
from hmac import compare_digest
from typing import Any, Protocol

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response

logger = logging.getLogger(__name__)


class TelegramBusinessUpdateHandler(Protocol):
    async def accept_update(self, body: dict[str, Any]) -> Any: ...

    async def process_update(self, accepted: Any) -> None: ...


def register_telegram_business_routes(
    app: FastAPI,
    *,
    get_relay: Callable[[], TelegramBusinessUpdateHandler | None],
    webhook_secret: str | None,
) -> None:
    """Persist business ingress before acknowledgement and process it afterward."""

    async def process_safely(
        relay: TelegramBusinessUpdateHandler,
        accepted: Any,
    ) -> None:
        try:
            await relay.process_update(accepted)
        except Exception:
            logger.exception("Telegram Business update processing failed")

    @app.post("/webhook/telegram", status_code=200)
    async def telegram_business_webhook(
        request: Request,
        background: BackgroundTasks,
        x_telegram_bot_api_secret_token: str | None = Header(default=None),
    ) -> Response:
        expected = str(webhook_secret or "").strip()
        provided = str(x_telegram_bot_api_secret_token or "").strip()
        if not expected:
            raise HTTPException(
                status_code=503,
                detail="Telegram Business webhook is not configured",
            )
        if not provided or not compare_digest(provided, expected):
            raise HTTPException(status_code=401, detail="invalid Telegram webhook secret")
        relay = get_relay()
        if relay is None:
            raise HTTPException(
                status_code=503,
                detail="Telegram Business relay unavailable",
            )
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid Telegram update")
        try:
            accepted = await relay.accept_update(body)
        except Exception as exc:
            logger.error(
                "Telegram Business update could not be durably accepted",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise HTTPException(
                status_code=503,
                detail="Telegram Business update could not be accepted",
            ) from exc
        background.add_task(process_safely, relay, accepted)
        return Response(status_code=200)


__all__ = ["TelegramBusinessUpdateHandler", "register_telegram_business_routes"]
