"""Shared bearer auth helpers for web-facing API routes."""

from __future__ import annotations

from hmac import compare_digest

from fastapi import Request
from fastapi.responses import JSONResponse


def bearer_token(request: Request) -> str:
    header = str(request.headers.get("authorization") or "").strip()
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def authorized_web_request(request: Request, expected_token: str | None) -> bool:
    secret = str(expected_token or "").strip()
    if not secret:
        return False
    token = bearer_token(request)
    return bool(token and compare_digest(token, secret))


def web_auth_error(
    request: Request,
    expected_token: str | None,
    *,
    missing_status_code: int = 503,
) -> JSONResponse | None:
    secret = str(expected_token or "").strip()
    if not secret:
        return JSONResponse(
            status_code=missing_status_code,
            content={"detail": "OPENTULPA_WEB_TOKEN is not configured"},
        )
    if not authorized_web_request(request, secret):
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return None
