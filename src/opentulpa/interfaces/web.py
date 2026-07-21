"""Fixed routes for the source-bundled, mutable owner web presentation."""

from __future__ import annotations

import mimetypes
import stat
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from opentulpa.api.web_auth import OWNER_SESSION_COOKIE, local_browser_request

_ASSET_ROOT = Path(__file__).resolve().parents[1] / "web_assets"
_ASSET_SUFFIXES = frozenset({".css", ".html", ".js", ".json", ".svg"})
_MAX_ASSET_BYTES = 5_000_000
_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; base-uri 'none'; connect-src 'self'; "
        "font-src 'self'; form-action 'self'; frame-ancestors 'none'; "
        "img-src 'self' data:; object-src 'none'; script-src 'self'; style-src 'self'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


def _asset_path(value: str) -> Path:
    relative = PurePosixPath(str(value or ""))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.suffix.casefold() not in _ASSET_SUFFIXES
    ):
        raise HTTPException(status_code=404)
    candidate = _ASSET_ROOT.joinpath(*relative.parts)
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise HTTPException(status_code=404) from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size > _MAX_ASSET_BYTES
    ):
        raise HTTPException(status_code=404)
    resolved_root = _ASSET_ROOT.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise HTTPException(status_code=404)
    return resolved


def register_owner_web_interface(
    app: FastAPI,
    *,
    local_owner_token: str | None = None,
) -> None:
    """Serve mutable static presentation through fixed, traversal-safe routes."""

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def owner_interface(request: Request) -> FileResponse:
        response = FileResponse(
            _asset_path("index.html"),
            media_type="text/html; charset=utf-8",
            headers=_SECURITY_HEADERS,
        )
        token = str(local_owner_token or "").strip()
        if token and local_browser_request(request):
            response.set_cookie(
                OWNER_SESSION_COOKIE,
                token,
                httponly=True,
                secure=request.url.scheme.casefold() == "https",
                samesite="strict",
                path="/",
            )
        return response

    @app.get("/assets/{asset_path:path}", response_class=FileResponse, include_in_schema=False)
    async def owner_asset(asset_path: str) -> FileResponse:
        path = _asset_path(asset_path)
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(
            path,
            media_type=media_type or "application/octet-stream",
            headers=_SECURITY_HEADERS,
        )


__all__ = ["register_owner_web_interface"]
