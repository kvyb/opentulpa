"""Web-facing intake handoff routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_customer_id as resolve_customer_id_value
from opentulpa.api.web_auth import web_auth_error


def register_handoff_routes(
    app: FastAPI,
    *,
    get_handoffs: Callable[[], Any],
    web_token: str | None,
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register authenticated handoff inbox routes."""

    def _customer_id(request: Request) -> str:
        return resolve_customer_id_value(request.query_params.get("customer_id", ""), resolve_customer_id)

    @app.get("/web/intake/handoffs")
    async def web_intake_handoffs_list(request: Request) -> Any:
        auth = web_auth_error(request, web_token)
        if auth is not None:
            return auth
        customer_id = _customer_id(request)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        status = str(request.query_params.get("status", "") or "").strip()
        limit = _bounded_limit(request.query_params.get("limit", "50"))
        return {
            "ok": True,
            "handoffs": get_handoffs().list_handoffs(
                customer_id=customer_id,
                status=status,
                limit=limit,
            ),
        }

    @app.get("/web/intake/handoffs/{handoff_id}")
    async def web_intake_handoffs_get(handoff_id: str, request: Request) -> Any:
        auth = web_auth_error(request, web_token)
        if auth is not None:
            return auth
        customer_id = _customer_id(request)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        handoff = get_handoffs().get_handoff(customer_id=customer_id, handoff_id=handoff_id)
        if handoff is None:
            return JSONResponse(status_code=404, content={"detail": "handoff not found"})
        return {"ok": True, "handoff": handoff}

    @app.post("/web/intake/handoffs/{handoff_id}/respond")
    async def web_intake_handoffs_respond(handoff_id: str, request: Request) -> Any:
        auth = web_auth_error(request, web_token)
        if auth is not None:
            return auth
        customer_id = _customer_id(request)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "payload must be an object"})
        result = await get_handoffs().respond(
            customer_id=customer_id,
            handoff_id=handoff_id,
            owner_feedback=str(body.get("owner_feedback", "") or ""),
        )
        if not bool(result.get("ok", False)):
            status_code = 409 if str(result.get("status")) == "conflict" else 400
            return JSONResponse(status_code=status_code, content=result)
        return result


def _bounded_limit(value: Any) -> int:
    try:
        parsed = int(str(value or "50"))
    except ValueError:
        return 50
    return max(1, min(parsed, 100))
