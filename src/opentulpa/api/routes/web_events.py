"""Authenticated web event feed for dashboard polling."""

from __future__ import annotations

import hmac
import json
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_customer_id as resolve_customer_id_value


def register_web_event_routes(
    app: FastAPI,
    *,
    settings: Any,
    get_web_events: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register dashboard-facing event feed routes."""

    @app.get("/web/events")
    async def list_web_events(request: Request) -> JSONResponse:
        expected_token = str(getattr(settings, "opentulpa_web_token", "") or "").strip()
        if not expected_token:
            return JSONResponse(
                status_code=503,
                content={"detail": "opentulpa web token is not configured"},
            )
        incoming = str(request.headers.get("authorization", "") or "").strip()
        prefix = "Bearer "
        token = incoming[len(prefix) :].strip() if incoming.startswith(prefix) else ""
        if not hmac.compare_digest(token, expected_token):
            return JSONResponse(status_code=403, content={"detail": "invalid opentulpa web token"})

        after_id = _int_query(request, "after_id", default=0)
        limit = _int_query(request, "limit", default=100)
        customer_id = (
            resolve_customer_id_value(request.query_params.get("customer_id", ""), resolve_customer_id)
            or None
        )
        events = get_web_events().list_events(
            after_id=after_id,
            limit=limit,
            customer_id=customer_id,
        )
        return JSONResponse(
            content={
                "events": [_serialize_event(event) for event in events],
                "next_cursor": events[-1]["id"] if events else after_id,
            }
        )


def _int_query(request: Request, name: str, *, default: int) -> int:
    raw = str(request.query_params.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _serialize_event(event: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    try:
        loaded = json.loads(str(event.get("metadata_json") or "{}"))
        if isinstance(loaded, dict):
            metadata = loaded
    except json.JSONDecodeError:
        metadata = {}
    return {
        "id": int(event["id"]),
        "created_at": str(event["created_at"]),
        "customer_id": str(event["customer_id"]),
        "thread_id": str(event["thread_id"]),
        "source": str(event["source"]),
        "kind": str(event["kind"]),
        "text": str(event["text"]),
        "metadata": metadata,
    }
