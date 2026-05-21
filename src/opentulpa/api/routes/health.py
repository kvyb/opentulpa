"""Health route registration."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

STARTED_AT = datetime.now(UTC).isoformat()


def register_health_routes(
    app: FastAPI,
    *,
    get_agent_runtime: Callable[[], Any],
    get_shutdown_drain: Callable[[], Any] | None = None,
) -> None:
    """Register liveness and runtime-health endpoints."""

    @app.get("/healthz", response_model=None)
    async def health() -> Any:
        drain = get_shutdown_drain() if get_shutdown_drain is not None else None
        status = drain.status() if drain is not None and hasattr(drain, "status") else None
        if status is not None and bool(getattr(status, "draining", False)):
            return JSONResponse(
                status_code=503,
                content={
                    "status": "draining",
                    "active_turns": int(getattr(status, "active_turns", 0)),
                    **_deployment_identity(),
                },
            )
        return {"status": "ok", **_deployment_identity()}

    @app.get("/agent/healthz")
    async def agent_health() -> dict[str, Any]:
        runtime = get_agent_runtime()
        healthy = bool(runtime and getattr(runtime, "healthy", lambda: False)())
        return {"status": "ok" if healthy else "degraded", "backend": "langgraph", **_deployment_identity()}


def _deployment_identity() -> dict[str, str | None]:
    return {
        "commit_sha": _clean_env("RAILWAY_GIT_COMMIT_SHA") or _clean_env("GIT_COMMIT_SHA"),
        "deployment_id": _clean_env("RAILWAY_DEPLOYMENT_ID") or _clean_env("OPENTULPA_DEPLOYMENT_ID"),
        "started_at": STARTED_AT,
    }


def _clean_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None
