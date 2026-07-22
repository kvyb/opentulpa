"""Authenticated tenant-scoped v2 integration and connection routes."""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.api.routes.v2_principal import V2Principal, resolve_v2_principal
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.persistence.idempotency import (
    IdempotencyConflictError,
    IdempotencyPendingError,
)

logger = logging.getLogger(__name__)

_INTEGRATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class IntegrationPort(Protocol):
    @property
    def enabled(self) -> bool: ...

    def list_integrations(
        self,
        *,
        tenant_id: str,
        query: str | None,
    ) -> Any: ...

    def connect(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        integration_id: str,
        redirect_url: str | None,
        idempotency_key: str,
    ) -> Any: ...

    def list_connections(
        self,
        *,
        tenant_id: str,
        integration_id: str | None,
    ) -> Any: ...

    def get_connection(self, *, tenant_id: str, connection_id: str) -> Any: ...

    def disconnect(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        idempotency_key: str,
    ) -> Any: ...

    def search_actions(
        self,
        *,
        tenant_id: str,
        query: str,
        integration_id: str | None,
        limit: int,
    ) -> Any: ...

    def get_action(self, *, tenant_id: str, action_name: str) -> Any: ...


class IntegrationConnectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    integration_id: str = Field(min_length=1, max_length=200, pattern=_INTEGRATION_ID_PATTERN)


class IntegrationConnection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    status: str
    integration_id: str
    integration_name: str


class IntegrationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    connected: bool
    connection_id: str | None = None
    connection_status: str | None = None
    requires_authentication: bool = True


def _integration_error(exc: Exception) -> HTTPException:
    logger.warning("v2 integration request failed: exception=%s", type(exc).__name__)
    if isinstance(exc, IdempotencyPendingError | IdempotencyConflictError):
        return HTTPException(status_code=409, detail="idempotency key cannot be reused")
    if isinstance(exc, LookupError | PermissionError):
        return HTTPException(status_code=404, detail="connection not found")
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail="invalid integration request")
    return HTTPException(status_code=502, detail="integration provider request failed")


async def _resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _owned_connections(
    service: IntegrationPort,
    *,
    tenant_id: str,
    integration_id: str | None = None,
) -> list[dict[str, Any]]:
    payload = await _resolve(
        service.list_connections(
            tenant_id=tenant_id,
            integration_id=integration_id,
        )
    )
    raw_items = payload.get("items") if isinstance(payload, dict) else None
    items = raw_items if isinstance(raw_items, list) else []
    return [
        dict(item)
        for item in items
        if isinstance(item, dict) and str(item.get("user_id", "") or "") == tenant_id
    ]


def _public_connection(item: dict[str, Any]) -> IntegrationConnection:
    return IntegrationConnection(
        id=str(item.get("id", "") or ""),
        status=str(item.get("status", "") or ""),
        integration_id=str(item.get("integration_id", "") or ""),
        integration_name=str(item.get("integration_name", "") or ""),
    )


def _public_action(item: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "name": str(item.get("name", "") or ""),
        "title": str(item.get("title", "") or ""),
        "description": str(item.get("description", "") or ""),
        "integration_id": str(item.get("integration_id", "") or ""),
        "integration_name": str(item.get("integration_name", "") or ""),
        "input_schema": item.get("input_schema") if isinstance(item.get("input_schema"), dict) else {},
    }
    sanitized = redact_for_langfuse(payload)
    return dict(sanitized) if isinstance(sanitized, dict) else {}


def _authorization_url(value: Any) -> str | None:
    candidate = str(value or "").strip()
    if not candidate or len(candidate) > 8_192:
        return None
    try:
        parsed = urlsplit(candidate)
    except ValueError:
        return None
    if parsed.scheme != "https" or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return candidate


def register_v2_integration_routes(
    app: FastAPI,
    *,
    get_integration_service: Callable[[], IntegrationPort | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
) -> None:
    """Register public integration CRUD while leaving OAuth callback registration separate."""

    def service_or_503() -> IntegrationPort:
        service = get_integration_service()
        if service is None or not bool(getattr(service, "enabled", False)):
            raise HTTPException(status_code=503, detail="integration service unavailable")
        return service

    @app.get("/v2/integrations")
    async def list_integrations(
        request: Request,
        search: str = Query(default="", max_length=500),
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, list[IntegrationItem]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        service = service_or_503()
        try:
            payload = await _resolve(
                service.list_integrations(
                    tenant_id=principal.tenant_id,
                    query=str(search or "").strip() or None,
                )
            )
        except Exception as exc:
            raise _integration_error(exc) from exc
        raw_items = payload.get("items") if isinstance(payload, dict) else None
        items: list[IntegrationItem] = []
        for raw in raw_items if isinstance(raw_items, list) else []:
            if not isinstance(raw, dict):
                continue
            items.append(
                IntegrationItem(
                    id=str(raw.get("id", "") or ""),
                    name=str(raw.get("name", "") or ""),
                    connected=bool(raw.get("connected", False)),
                    connection_id=str(raw.get("connection_id", "") or "") or None,
                    connection_status=str(raw.get("connection_status", "") or "") or None,
                    requires_authentication=bool(raw.get("requires_authentication", True)),
                )
            )
            if len(items) >= limit:
                break
        return {"integrations": items}

    @app.post("/v2/integrations/connections", status_code=201)
    async def connect_integration(
        body: IntegrationConnectRequest,
        request: Request,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=200),
        ],
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            result = await _resolve(
                service_or_503().connect(
                    tenant_id=principal.tenant_id,
                    actor_id=principal.actor_id,
                    integration_id=body.integration_id,
                    redirect_url=None,
                    idempotency_key=idempotency_key,
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _integration_error(exc) from exc
        authorization_url = _authorization_url(result.get("authorization_url")) or _authorization_url(
            result.get("oauth_url")
        )
        return {
            "connection": {
                "id": str(result.get("connection_id") or result.get("id") or ""),
                "integration_id": body.integration_id,
                "authorization_url": authorization_url,
                "status": str(result.get("status", "") or "pending"),
            }
        }

    @app.get("/v2/integrations/connections")
    async def list_connections(
        request: Request,
        integration_id: Annotated[
            str | None,
            Query(min_length=1, max_length=200, pattern=_INTEGRATION_ID_PATTERN),
        ] = None,
        status: Annotated[list[str] | None, Query()] = None,
    ) -> dict[str, list[IntegrationConnection]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            items = await _owned_connections(
                service_or_503(),
                tenant_id=principal.tenant_id,
                integration_id=integration_id,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _integration_error(exc) from exc
        allowed_statuses = {item.upper() for item in status or []}
        filtered = [
            item
            for item in items
            if not allowed_statuses
            or str(item.get("status", "") or "").upper() in allowed_statuses
        ]
        return {"connections": [_public_connection(item) for item in filtered]}

    @app.delete("/v2/integrations/connections/{connection_id}")
    async def disconnect_integration(
        connection_id: str,
        request: Request,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=200),
        ],
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        service = service_or_503()
        try:
            owned = await _resolve(
                service.get_connection(
                    tenant_id=principal.tenant_id,
                    connection_id=connection_id,
                )
            )
            if (
                not isinstance(owned, dict)
                or str(owned.get("user_id", "") or "") != principal.tenant_id
            ):
                raise HTTPException(status_code=404, detail="connection not found")
            result = await _resolve(
                service.disconnect(
                    tenant_id=principal.tenant_id,
                    connection_id=connection_id,
                    idempotency_key=idempotency_key,
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _integration_error(exc) from exc
        deleted_id = str(result.get("connection_id") or result.get("id") or connection_id)
        return {"deleted": True, "connection_id": deleted_id or connection_id}

    @app.get("/v2/integrations/actions")
    async def search_integration_actions(
        request: Request,
        query: str = Query(default="", max_length=1_000),
        integration_id: str | None = Query(
            default=None,
            min_length=1,
            max_length=200,
            pattern=_INTEGRATION_ID_PATTERN,
        ),
        limit: int = Query(default=20, ge=1, le=50),
    ) -> dict[str, list[dict[str, Any]]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        _ = principal
        if not str(query or "").strip() and not integration_id:
            raise HTTPException(status_code=422, detail="query or integration_id is required")
        try:
            payload = await _resolve(
                service_or_503().search_actions(
                    tenant_id=principal.tenant_id,
                    query=str(query or "").strip(),
                    integration_id=integration_id,
                    limit=limit,
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _integration_error(exc) from exc
        raw_items = payload.get("items") if isinstance(payload, dict) else None
        return {
            "actions": [
                _public_action(item)
                for item in raw_items if isinstance(item, dict)
            ]
            if isinstance(raw_items, list)
            else []
        }

    @app.get("/v2/integrations/actions/{action_name}")
    async def get_integration_action(action_name: str, request: Request) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        _ = principal
        if not re.fullmatch(_INTEGRATION_ID_PATTERN, action_name):
            raise HTTPException(status_code=422, detail="invalid action name")
        try:
            payload = await _resolve(
                service_or_503().get_action(
                    tenant_id=principal.tenant_id,
                    action_name=action_name,
                )
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise _integration_error(exc) from exc
        raw_tool = payload.get("action") if isinstance(payload, dict) else None
        if not isinstance(raw_tool, dict):
            raise HTTPException(status_code=404, detail="integration action not found")
        return {"action": _public_action(raw_tool)}


__all__ = [
    "IntegrationConnectRequest",
    "IntegrationConnection",
    "IntegrationItem",
    "IntegrationPort",
    "register_v2_integration_routes",
]
