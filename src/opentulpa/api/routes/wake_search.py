"""Wake queue and web-search route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.integrations.web_search import web_search as run_web_search


def register_wake_and_search_routes(
    app: FastAPI,
    *,
    get_wake_queue: Callable[[], Any],
    llm_model: str | None,
) -> None:
    """Register wake queue APIs and OpenRouter-backed web search endpoint."""
    _ = llm_model

    @app.post("/internal/wake")
    async def internal_wake(request: Request) -> Any:
        """Called by scheduler or external trigger to wake the agent with a payload."""
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400, content={"detail": "wake payload must be JSON object"}
            )
        queue_id = await get_wake_queue().enqueue(body)
        return {"ok": True, "queued": True, "queue_id": queue_id}

    @app.get("/internal/wake/queue")
    async def internal_wake_queue_stats() -> Any:
        """Inspect wake queue health and recent entries."""
        return {"ok": True, "queue": get_wake_queue().stats()}

    @app.post("/internal/web_search")
    async def internal_web_search(request: Request) -> Any:
        """Run OpenRouter web search (default: Perplexity Sonar Pro Search)."""
        body = await request.json()
        query = body.get("query", "").strip()
        if not query:
            return JSONResponse(status_code=400, content={"detail": "query required"})
        result = await run_web_search(query)
        return {"ok": True, "result": result}
