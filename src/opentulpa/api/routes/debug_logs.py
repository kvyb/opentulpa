"""Operator-only debug log route registration."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.responses import Response

from opentulpa.core.debug_logs import get_debug_log_path


def register_debug_log_routes(app: FastAPI) -> None:
    """Register direct API access to the app debug log."""

    @app.get("/debug_logs", response_model=None)
    async def debug_logs() -> Response:
        path = get_debug_log_path()
        if not path.exists() or not path.is_file():
            return JSONResponse(status_code=404, content={"detail": "app.log not found"})
        return FileResponse(
            path=path,
            media_type="text/plain; charset=utf-8",
            filename=path.name,
        )
