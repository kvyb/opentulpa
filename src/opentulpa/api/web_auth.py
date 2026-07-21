"""Shared bearer auth helpers for web-facing API routes."""

from __future__ import annotations

from hmac import compare_digest
from ipaddress import ip_address
from urllib.parse import urlsplit

from fastapi import Request
from fastapi.responses import JSONResponse

OWNER_SESSION_COOKIE = "opentulpa_owner_session"
_FORWARDED_HEADERS = (
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
)
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_loopback_host(value: str | None) -> bool:
    """Return whether a configured bind/request host is explicitly loopback."""

    host = str(value or "").strip().casefold().removeprefix("[").removesuffix("]")
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def local_owner_cookie_enabled(*, bind_host: str, public_base_url: str | None) -> bool:
    """Enable passwordless browser auth only for an unproxied local deployment."""

    return is_loopback_host(bind_host) and not str(public_base_url or "").strip()


def local_browser_request(request: Request, *, require_origin: bool = False) -> bool:
    """Validate the network and browser boundary used by local cookie authentication."""

    if any(request.headers.get(name) for name in _FORWARDED_HEADERS):
        return False
    if request.client is None or not is_loopback_host(request.client.host):
        return False
    if not is_loopback_host(request.url.hostname):
        return False
    fetch_site = str(request.headers.get("sec-fetch-site") or "").strip().casefold()
    if fetch_site and fetch_site not in {"none", "same-origin"}:
        return False
    if not require_origin:
        return True
    origin = str(request.headers.get("origin") or "").strip()
    if not origin:
        return False
    parsed = urlsplit(origin)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return False
    try:
        origin_port = parsed.port
        request_port = request.url.port
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == request.url.scheme.casefold()
        and (parsed.hostname or "").casefold()
        == (request.url.hostname or "").casefold()
        and origin_port == request_port
    )


def owner_session_token(request: Request, *, enabled: bool) -> str:
    """Read the local browser cookie only inside its loopback same-origin boundary."""

    if not enabled:
        return ""
    require_origin = request.method.upper() not in _SAFE_METHODS
    if not local_browser_request(request, require_origin=require_origin):
        return ""
    return str(request.cookies.get(OWNER_SESSION_COOKIE) or "").strip()


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


__all__ = [
    "OWNER_SESSION_COOKIE",
    "authorized_web_request",
    "bearer_token",
    "is_loopback_host",
    "local_browser_request",
    "local_owner_cookie_enabled",
    "owner_session_token",
    "web_auth_error",
]
