"""Authenticated V2 owner notification delivery and acknowledgement routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Path, Query, Request, Response

from opentulpa.api.routes.v2_principal import (
    V2Principal,
    require_v2_scope,
    resolve_v2_principal,
)
from opentulpa.capabilities.credentials import CapabilityAPIScope
from opentulpa.notifications import NotificationNotFoundError, OwnerNotification


class NotificationRouteService(Protocol):
    async def wait(
        self,
        *,
        tenant_id: str,
        consumer_id: str,
        after_id: int = 0,
        limit: int = 100,
        wait_seconds: float = 0,
    ) -> list[OwnerNotification]: ...

    def acknowledge(
        self,
        *,
        tenant_id: str,
        consumer_id: str,
        notification_id: int,
    ) -> bool: ...


def register_v2_notification_routes(
    app: FastAPI,
    *,
    get_notifications: Callable[[], NotificationRouteService | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
) -> None:
    """Expose one durable stream without accepting tenant or consumer identifiers."""

    def service_or_503() -> NotificationRouteService:
        service = get_notifications()
        if service is None:
            raise HTTPException(status_code=503, detail="notification service unavailable")
        return service

    @app.get("/v2/notifications")
    async def list_notifications(
        request: Request,
        after_id: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=100),
        wait_seconds: float = Query(default=0, ge=0, le=30),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.NOTIFICATIONS_READ.value)
        notifications = await service_or_503().wait(
            tenant_id=principal.tenant_id,
            consumer_id=_consumer_id(principal.interface, principal.source_id),
            after_id=after_id,
            limit=limit,
            wait_seconds=wait_seconds,
        )
        return {
            "notifications": [_public_notification(item) for item in notifications],
            "next_after_id": notifications[-1].id if notifications else after_id,
        }

    @app.post(
        "/v2/notifications/{notification_id}/ack",
        status_code=204,
        response_class=Response,
    )
    async def acknowledge_notification(
        request: Request,
        notification_id: int = Path(ge=1),
    ) -> Response:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.NOTIFICATIONS_ACK.value)
        try:
            service_or_503().acknowledge(
                tenant_id=principal.tenant_id,
                consumer_id=_consumer_id(principal.interface, principal.source_id),
                notification_id=notification_id,
            )
        except NotificationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="notification not found") from exc
        return Response(status_code=204)


def _consumer_id(interface: str, source_id: str) -> str:
    value = f"{interface}:{source_id}"
    if len(value) > 300 or any(ord(character) < 32 for character in value):
        raise HTTPException(status_code=401, detail="invalid notification consumer")
    return value


def _public_notification(notification: OwnerNotification) -> dict[str, Any]:
    return notification.model_dump(
        mode="json",
        exclude={"tenant_id", "dedupe_key"},
    )


__all__ = [
    "NotificationRouteService",
    "register_v2_notification_routes",
]
