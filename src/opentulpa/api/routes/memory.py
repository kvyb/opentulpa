"""Internal memory route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request


def register_memory_routes(
    app: FastAPI,
    *,
    get_memory: Callable[[], Any],
) -> None:
    """Register internal memory add/search endpoints."""

    @app.post("/internal/memory/add")
    async def internal_memory_add(request: Request) -> Any:
        mem = get_memory()
        body = await request.json()
        messages = body.get("messages", [])
        user_id = body.get("user_id") or mem.user_id
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
        query = body.get("query", "")
        user_id = body.get("user_id") or mem.user_id
        limit = body.get("limit", 5)
        results = mem.search(query, user_id=user_id, limit=limit)
        return {"results": results}
