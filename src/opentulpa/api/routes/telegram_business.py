"""Telegram Business internal API routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request

from opentulpa.api.customer_ids import resolve_body_customer_id


def register_telegram_business_routes(
    app: FastAPI,
    *,
    get_telegram_business: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register Telegram Business status routes."""

    @app.post("/internal/telegram/business/status")
    async def internal_telegram_business_status(request: Request) -> Any:
        service = get_telegram_business()
        body = await request.json()
        return service.status(customer_id=resolve_body_customer_id(body, resolve_customer_id))
