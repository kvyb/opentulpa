"""Dynamic tulpa sandbox route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.tasks.sandbox import (
    ALLOWED_TERMINAL_COMMANDS,
    ALLOWED_TERMINAL_DIRS,
    PROJECT_ROOT,
    get_tulpa_catalog,
)
from opentulpa.tasks.sandbox import read_file as sandbox_read_file
from opentulpa.tasks.sandbox import run_terminal as sandbox_run_terminal
from opentulpa.tasks.sandbox import validate_generated_file as sandbox_validate_generated_file
from opentulpa.tasks.sandbox import write_file as sandbox_write_file


def register_tulpa_routes(
    app: FastAPI,
    *,
    get_tulpa_loader: Callable[[], Any],
) -> None:
    """Register tulpa reload/read/write/run validation endpoints."""

    @app.post("/internal/tulpa/reload")
    async def internal_tulpa_reload() -> Any:
        """Reload APIRouter modules from tulpa_stuff."""
        return get_tulpa_loader().reload()

    @app.post("/internal/tulpa/write_file")
    async def internal_tulpa_write_file(request: Request) -> Any:
        """Write a file only inside approved integration/self-modification paths."""
        body = await request.json()
        relative_path = str(body.get("path", "")).strip()
        content = body.get("content")
        if content is None:
            return JSONResponse(status_code=400, content={"detail": "content is required"})
        try:
            target = sandbox_write_file(relative_path, str(content))
            validation = sandbox_validate_generated_file(relative_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {
            "ok": True,
            "path": str(target.relative_to(PROJECT_ROOT)),
            "validation": validation,
        }

    @app.post("/internal/tulpa/validate_file")
    async def internal_tulpa_validate_file(request: Request) -> Any:
        """Validate generated code file contract/syntax before using it."""
        body = await request.json()
        relative_path = str(body.get("path", "")).strip()
        try:
            result = sandbox_validate_generated_file(relative_path)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return result

    @app.get("/internal/tulpa/read_file")
    async def internal_tulpa_read_file(path: str, max_chars: int = 12000) -> Any:
        """Read a file inside approved integration/self-modification paths."""
        try:
            content = sandbox_read_file(path, max_chars=max_chars)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "path": path, "content": content}

    @app.post("/internal/tulpa/run_terminal")
    async def internal_tulpa_run_terminal(request: Request) -> Any:
        """Run a restricted command in approved integration/self-modification paths."""
        body = await request.json()
        command = str(body.get("command", "")).strip()
        working_dir_key = str(body.get("working_dir", "tulpa_stuff")).strip()
        timeout_seconds = int(body.get("timeout_seconds", 90))

        if not command:
            return JSONResponse(status_code=400, content={"detail": "command is required"})
        try:
            return sandbox_run_terminal(
                command=command,
                working_dir=working_dir_key,
                timeout_seconds=timeout_seconds,
            )
        except PermissionError as exc:
            return JSONResponse(
                status_code=403,
                content={"detail": str(exc), "allowed_commands": sorted(ALLOWED_TERMINAL_COMMANDS)},
            )
        except TimeoutError as exc:
            return JSONResponse(status_code=408, content={"detail": str(exc)})
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": str(exc),
                    "allowed_working_dirs": sorted(ALLOWED_TERMINAL_DIRS.keys()),
                },
            )
        except RuntimeError as exc:
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/internal/tulpa/catalog")
    async def internal_tulpa_catalog() -> Any:
        """Return tulpa_stuff catalog/index and recent tracked entries."""
        return {"ok": True, "catalog": get_tulpa_catalog()}
