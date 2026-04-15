"""Telegram Business internal API routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request


def register_telegram_business_routes(
    app: FastAPI,
    *,
    get_telegram_business: Callable[[], Any],
) -> None:
    """Register Telegram Business status routes."""

    @app.post("/internal/telegram/business/status")
    async def internal_telegram_business_status(request: Request) -> Any:
        service = get_telegram_business()
        body = await request.json()
        return service.status(customer_id=str(body.get("customer_id", "")).strip())
