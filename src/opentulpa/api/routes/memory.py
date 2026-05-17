"""Internal memory route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request

from opentulpa.api.customer_ids import resolve_customer_id as resolve_customer_id_value


def register_memory_routes(
    app: FastAPI,
    *,
    get_memory: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register internal memory add/search endpoints."""

    @app.post("/internal/memory/add")
    async def internal_memory_add(request: Request) -> Any:
        mem = get_memory()
        body = await request.json()
        messages = body.get("messages", [])
        user_id = resolve_customer_id_value(body.get("user_id") or mem.user_id, resolve_customer_id)
        metadata = body.get("metadata") or {}
        infer = bool(body.get("infer", True))
        retries = int(body.get("retries", 1) or 1)
        result = mem.add(
            messages,
            user_id=user_id,
            metadata=metadata,
            infer=infer,
            retries=retries,
        )
        return {"ok": True, "result": result}

    @app.post("/internal/memory/search")
    async def internal_memory_search(request: Request) -> Any:
        mem = get_memory()
        body = await request.json()
        query = str(body.get("query", "") or "")
        user_id = resolve_customer_id_value(body.get("user_id") or mem.user_id, resolve_customer_id)
        try:
            limit = int(body.get("limit", 5))
        except (TypeError, ValueError):
            limit = 5
        limit = max(1, min(limit, 25))
        metadata = body.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else None
        results = mem.search(query, user_id=user_id, limit=limit, metadata=metadata)
        return {"results": results}
