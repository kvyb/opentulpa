"""Stable setup, recovery, logging, and proxy API for one OpenTulpa deployment."""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import Cookie, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from opentulpa.host.models import HostConfigInput
from opentulpa.host.service import HostActivationError, HostService
from opentulpa.host.store import HostConfigConflictError, HostStore

HOST_SESSION_COOKIE = "opentulpa_host_session"
_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_HOST_HEADERS = {
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


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ClaimRequest(_RequestModel):
    setup_token: str | None = Field(default=None, max_length=500)
    owner_token: str | None = Field(default=None, max_length=500)


class SessionRequest(_RequestModel):
    token: str = Field(min_length=1, max_length=500)


def create_host_app(
    *,
    store: HostStore,
    service: HostService,
    assets_root: Path | None = None,
    local_owner_enabled: bool = False,
    setup_token: str | None = None,
) -> FastAPI:
    """Create the immutable host surface around one mutable Deep Agents child."""

    asset_dir = assets_root or Path(__file__).resolve().parent / "assets"
    proxy_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5, read=None, write=60, pool=5),
        follow_redirects=False,
        trust_env=False,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await service.start()
        try:
            yield
        finally:
            await service.shutdown()
            await proxy_client.aclose()

    app = FastAPI(title="OpenTulpa Host", version="2", lifespan=lifespan)
    app.state.host_store = store
    app.state.host_service = service

    def owner_token(
        request: Request,
        authorization: str | None,
        cookie_token: str | None,
    ) -> str | None:
        if local_owner_enabled and request.client and request.client.host in {"127.0.0.1", "::1"}:
            return "__local_owner__"
        if authorization and authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return cookie_token

    def require_owner(
        request: Request,
        authorization: str | None,
        cookie_token: str | None,
    ) -> None:
        token = owner_token(request, authorization, cookie_token)
        if token == "__local_owner__":
            return
        if not store.authorize_owner(token):
            raise HTTPException(status_code=401, detail="owner authentication required")

    @app.get("/healthz")
    async def health() -> dict[str, Any]:
        return {
            "ok": True,
            "host": "ready",
            "runtime": service.runtime.status,
            "configured": store.active() is not None,
        }

    @app.get("/agent/healthz")
    async def agent_health() -> Response:
        if service.runtime.status != "ready" or service.activating:
            runtime_status = "activating" if service.activating else service.runtime.status
            return JSONResponse(status_code=503, content={"ok": False, "runtime": runtime_status})
        return JSONResponse(content={"ok": True, "runtime": "ready"})

    @app.get("/_host")
    @app.get("/_host/")
    async def host_console() -> FileResponse:
        return FileResponse(asset_dir / "index.html", media_type="text/html", headers=_HOST_HEADERS)

    @app.get("/_host/assets/{asset_name}")
    async def host_asset(asset_name: str) -> FileResponse:
        if asset_name not in {"app.css", "app.js", "favicon.svg"}:
            raise HTTPException(status_code=404)
        media_type = (
            "text/css"
            if asset_name.endswith(".css")
            else "image/svg+xml"
            if asset_name.endswith(".svg")
            else "application/javascript"
        )
        return FileResponse(asset_dir / asset_name, media_type=media_type, headers=_HOST_HEADERS)

    @app.get("/_host/api/status")
    async def host_status(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        session: Annotated[str | None, Cookie(alias=HOST_SESSION_COOKIE)] = None,
    ) -> dict[str, Any]:
        token = owner_token(request, authorization, session)
        authenticated = token == "__local_owner__" or store.authorize_owner(token)
        active = store.active()
        payload: dict[str, Any] = {
            "claimed": store.claimed,
            "authenticated": authenticated,
            "configured": active is not None,
            "runtime": {
                "status": "activating" if service.activating else service.runtime.status,
                "revision": service.runtime.revision,
                "error": service.runtime.error if authenticated else None,
            },
        }
        if authenticated:
            payload["config"] = store.view(active) if active is not None else None
            payload["revisions"] = store.list_views()
        return payload

    @app.post("/_host/api/claim")
    async def claim(body: ClaimRequest, request: Request, response: Response) -> dict[str, Any]:
        if store.claimed:
            raise HTTPException(status_code=409, detail="host is already claimed")
        is_local = bool(
            local_owner_enabled and request.client and request.client.host in {"127.0.0.1", "::1"}
        )
        supplied_setup = setup_token if is_local else body.setup_token
        token = body.owner_token or secrets.token_urlsafe(48)
        try:
            store.claim(setup_token=supplied_setup, owner_token=token)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail="invalid setup token") from exc
        except (ValueError, HostConfigConflictError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        _set_owner_cookie(response, token, secure=request.url.scheme == "https")
        return {"claimed": True, "owner_token": token}

    @app.post("/_host/api/session", status_code=status.HTTP_204_NO_CONTENT)
    async def create_session(
        body: SessionRequest, request: Request, response: Response
    ) -> Response:
        if not store.authorize_owner(body.token):
            raise HTTPException(status_code=401, detail="invalid owner token")
        _set_owner_cookie(response, body.token, secure=request.url.scheme == "https")
        return response

    @app.delete("/_host/api/session", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_session(response: Response) -> Response:
        response.delete_cookie(HOST_SESSION_COOKIE, path="/")
        return response

    @app.put("/_host/api/config")
    async def apply_config(
        body: HostConfigInput,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        session: Annotated[str | None, Cookie(alias=HOST_SESSION_COOKIE)] = None,
    ) -> dict[str, Any]:
        require_owner(request, authorization, session)
        try:
            config = await service.apply(body)
        except HostConfigConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (HostActivationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"config": config, "runtime": service.runtime.status}

    @app.post("/_host/api/runtime/restart")
    async def restart_runtime(
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        session: Annotated[str | None, Cookie(alias=HOST_SESSION_COOKIE)] = None,
    ) -> dict[str, str]:
        require_owner(request, authorization, session)
        try:
            await service.restart()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="runtime restart failed") from exc
        return {"status": service.runtime.status}

    @app.get("/_host/api/logs")
    async def logs(
        request: Request,
        after: int = 0,
        authorization: Annotated[str | None, Header()] = None,
        session: Annotated[str | None, Cookie(alias=HOST_SESSION_COOKIE)] = None,
    ) -> dict[str, Any]:
        require_owner(request, authorization, session)
        return {"logs": service.runtime.logs(after=max(0, after))}

    @app.get("/_host/api/logs/stream")
    async def log_stream(
        request: Request,
        after: int = 0,
        authorization: Annotated[str | None, Header()] = None,
        session: Annotated[str | None, Cookie(alias=HOST_SESSION_COOKIE)] = None,
    ) -> StreamingResponse:
        require_owner(request, authorization, session)

        async def events() -> AsyncIterator[str]:
            cursor = max(0, after)
            while not await request.is_disconnected():
                entries = await service.runtime.wait_for_logs(after=cursor)
                if not entries:
                    yield ": keepalive\n\n"
                    continue
                for entry in entries:
                    cursor = entry.sequence
                    yield f"id: {cursor}\ndata: {entry.model_dump_json()}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.api_route(
        "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    )
    async def proxy_runtime(
        path: str,
        request: Request,
        authorization: Annotated[str | None, Header()] = None,
        session: Annotated[str | None, Cookie(alias=HOST_SESSION_COOKIE)] = None,
    ) -> Response:
        endpoint = service.runtime.endpoint
        if service.activating:
            raise HTTPException(status_code=503, detail="OpenTulpa runtime is activating")
        if endpoint is None:
            if request.method == "GET" and not path:
                return RedirectResponse("/_host", status_code=307)
            raise HTTPException(status_code=503, detail="OpenTulpa is not configured")
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower()
            not in _HOP_HEADERS | {"host", "content-length", "authorization", "cookie"}
        }
        token = owner_token(request, authorization, session)
        if token == "__local_owner__" or store.authorize_owner(token):
            active = store.active()
            if active is not None:
                headers["Authorization"] = (
                    f"Bearer {active.internal_runtime_token.get_secret_value()}"
                )
        elif authorization:
            headers["Authorization"] = authorization
        target = f"{endpoint}/{path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        upstream_request = proxy_client.build_request(
            request.method,
            target,
            headers=headers,
            content=request.stream(),
        )
        upstream = await proxy_client.send(upstream_request, stream=True)
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _HOP_HEADERS | {"content-length"}
        }
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream.aclose),
        )

    return app


def _set_owner_cookie(response: Response, token: str, *, secure: bool) -> None:
    response.set_cookie(
        HOST_SESSION_COOKIE,
        token,
        httponly=True,
        secure=secure,
        samesite="strict",
        path="/",
        max_age=60 * 60 * 24 * 30,
    )


__all__ = ["HOST_SESSION_COOKIE", "create_host_app"]
