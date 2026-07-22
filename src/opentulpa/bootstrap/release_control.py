"""Control protocol implemented by every mutable OpenTulpa release."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import logging
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from opentulpa.bootstrap.models import (
    DrainResult,
    IngressEnvelope,
    OutboxEvent,
    ReleaseHealth,
)

logger = logging.getLogger(__name__)
_CONTROL_PATH_RE = re.compile(r"/[A-Za-z0-9._~!$&'()*+,;=:@%/-]{1,200}\Z")
_RELEASE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,99}\Z")

HealthProvider = Callable[[], Mapping[str, bool] | Awaitable[Mapping[str, bool]]]
IngressHandler = Callable[[IngressEnvelope], Awaitable[None]]
EventHandler = Callable[[OutboxEvent], Awaitable[None]]


class ReleaseControlConfigurationError(RuntimeError):
    pass


class DrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float = Field(default=60.0, ge=0, le=300)


class ReleaseActivity:
    """Count requests until their final ASGI response body has been emitted."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def enter(self) -> None:
        async with self._condition:
            self._in_flight += 1

    async def leave(self) -> None:
        async with self._condition:
            self._in_flight = max(0, self._in_flight - 1)
            self._condition.notify_all()

    async def wait_drained(self, *, timeout_seconds: float) -> DrainResult:
        async with self._condition:
            try:
                async with asyncio.timeout(timeout_seconds):
                    await self._condition.wait_for(lambda: self._in_flight == 0)
            except TimeoutError:
                return DrainResult(drained=False, in_flight=self._in_flight)
            return DrainResult(drained=True, in_flight=0)


class ReleaseActivityMiddleware:
    """ASGI middleware that keeps streaming requests in-flight until completion."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        activity: ReleaseActivity,
        excluded_paths: frozenset[str],
        release_id: str,
        lease_epoch: int | None,
        gateway_token: str,
    ) -> None:
        self._app = app
        self._activity = activity
        self._excluded_paths = excluded_paths
        self._release_id = release_id
        self._lease_epoch = str(lease_epoch or "none")
        self._gateway_token = gateway_token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or str(scope.get("path") or "") in self._excluded_paths:
            await self._app(scope, receive, send)
            return
        headers = {
            key.decode("ascii", errors="ignore").casefold(): value.decode(
                "latin-1", errors="ignore"
            )
            for key, value in scope.get("headers", ())
        }
        if (
            not hmac.compare_digest(
                headers.get("x-opentulpa-control-token", ""),
                self._gateway_token,
            )
            or not hmac.compare_digest(
                headers.get("x-opentulpa-release-id", ""),
                self._release_id,
            )
            or not hmac.compare_digest(
                headers.get("x-opentulpa-lease-epoch", ""),
                self._lease_epoch,
            )
        ):
            await Response(status_code=401)(scope, receive, send)
            return
        await self._activity.enter()
        try:
            await self._app(scope, receive, send)
        finally:
            await self._activity.leave()


class ReleaseControlService:
    """Authenticated, loopback-only protocol used by the immutable bootstrap."""

    def __init__(
        self,
        *,
        release_id: str,
        lease_epoch: int | None,
        control_token: str,
        health_path: str = "/_control/v1/health",
        drain_path: str = "/_control/v1/drain",
        ingress_path: str = "/_control/v1/ingress",
        event_path: str = "/_control/v1/events",
        health_provider: HealthProvider,
        ingress_handler: IngressHandler | None = None,
        event_handler: EventHandler | None = None,
        activity: ReleaseActivity | None = None,
    ) -> None:
        if not _RELEASE_ID_RE.fullmatch(release_id):
            raise ValueError("release control ID is invalid")
        if lease_epoch is not None and lease_epoch < 1:
            raise ValueError("release control lease is invalid")
        if not re.fullmatch(r"[A-Za-z0-9_-]{32,200}", control_token):
            raise ValueError("release control token is invalid")
        paths = (health_path, drain_path, ingress_path, event_path)
        if len(set(paths)) != len(paths) or any(
            _CONTROL_PATH_RE.fullmatch(path) is None for path in paths
        ):
            raise ValueError("release control paths are invalid")
        self.release_id = release_id
        self.lease_epoch = lease_epoch
        self._control_token = control_token
        self.health_path = health_path
        self.drain_path = drain_path
        self.ingress_path = ingress_path
        self.event_path = event_path
        self.health_provider = health_provider
        self.ingress_handler = ingress_handler
        self.event_handler = event_handler
        self.activity = activity or ReleaseActivity()

    @classmethod
    def from_environment(
        cls,
        *,
        health_provider: HealthProvider,
        ingress_handler: IngressHandler | None = None,
        event_handler: EventHandler | None = None,
    ) -> ReleaseControlService:
        release_id = str(os.environ.get("OPENTULPA_RELEASE_ID") or "").strip()
        token = str(os.environ.get("OPENTULPA_CONTROL_TOKEN") or "").strip()
        epoch_raw = str(os.environ.get("OPENTULPA_LEASE_EPOCH") or "none").strip()
        try:
            lease_epoch = None if epoch_raw == "none" else int(epoch_raw)
        except ValueError as exc:
            raise ReleaseControlConfigurationError("release lease environment is invalid") from exc
        try:
            return cls(
                release_id=release_id,
                lease_epoch=lease_epoch,
                control_token=token,
                health_path=str(
                    os.environ.get("OPENTULPA_HEALTH_PATH") or "/_control/v1/health"
                ),
                drain_path=str(
                    os.environ.get("OPENTULPA_DRAIN_PATH") or "/_control/v1/drain"
                ),
                ingress_path=str(
                    os.environ.get("OPENTULPA_INGRESS_PATH") or "/_control/v1/ingress"
                ),
                event_path=str(
                    os.environ.get("OPENTULPA_EVENT_PATH") or "/_control/v1/events"
                ),
                health_provider=health_provider,
                ingress_handler=ingress_handler,
                event_handler=event_handler,
            )
        except ValueError as exc:
            raise ReleaseControlConfigurationError("release control environment is invalid") from exc

    async def health(self) -> ReleaseHealth:
        try:
            provided = self.health_provider()
            components = await provided if inspect.isawaitable(provided) else provided
            normalized = {str(name): bool(value) for name, value in components.items()}
        except Exception:
            logger.exception("release health provider failed")
            normalized = {"runtime": False, "agent_api": False}
        required = {"runtime", "agent_api"}
        healthy = required.issubset(normalized) and all(normalized.values())
        return ReleaseHealth(
            healthy=healthy,
            release_id=self.release_id,
            protocol_version=1,
            summary="healthy" if healthy else "release components are degraded",
            components=normalized,
        )

    def authorize(
        self,
        *,
        authorization: str | None,
        release_id: str | None,
        lease_epoch: str | None,
    ) -> None:
        scheme, _, supplied = str(authorization or "").partition(" ")
        expected_epoch = str(self.lease_epoch or "none")
        if (
            scheme.casefold() != "bearer"
            or not hmac.compare_digest(supplied, self._control_token)
            or not hmac.compare_digest(str(release_id or ""), self.release_id)
            or not hmac.compare_digest(str(lease_epoch or ""), expected_epoch)
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid release control credentials are required",
            )

    async def accept_ingress(
        self,
        envelope: IngressEnvelope,
        *,
        idempotency_key: str | None,
    ) -> None:
        if not hmac.compare_digest(str(idempotency_key or ""), envelope.idempotency_key):
            raise HTTPException(status_code=409, detail="ingress idempotency key mismatch")
        if self.ingress_handler is None:
            raise HTTPException(status_code=503, detail="release ingress handler is unavailable")
        await self.ingress_handler(envelope)

    async def accept_event(
        self,
        event: OutboxEvent,
        *,
        idempotency_key: str | None,
    ) -> None:
        if not hmac.compare_digest(str(idempotency_key or ""), event.event_key):
            raise HTTPException(status_code=409, detail="event idempotency key mismatch")
        if self.event_handler is None:
            raise HTTPException(status_code=503, detail="release event handler is unavailable")
        await self.event_handler(event)


def create_release_control_router(service: ReleaseControlService) -> APIRouter:
    async def authorize(
        authorization: Annotated[str | None, Header()] = None,
        x_opentulpa_release_id: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Release-ID"),
        ] = None,
        x_opentulpa_lease_epoch: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Lease-Epoch"),
        ] = None,
    ) -> None:
        service.authorize(
            authorization=authorization,
            release_id=x_opentulpa_release_id,
            lease_epoch=x_opentulpa_lease_epoch,
        )

    router = APIRouter(dependencies=[Depends(authorize)], include_in_schema=False)

    @router.get(service.health_path)
    async def health() -> dict[str, Any]:
        return (await service.health()).model_dump(mode="json")

    @router.post(service.drain_path)
    async def drain(body: DrainRequest) -> dict[str, Any]:
        result = await service.activity.wait_drained(timeout_seconds=body.timeout_seconds)
        return result.model_dump(mode="json")

    @router.post(service.ingress_path, status_code=status.HTTP_204_NO_CONTENT)
    async def ingress(
        body: IngressEnvelope,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> None:
        await service.accept_ingress(body, idempotency_key=idempotency_key)

    @router.post(service.event_path, status_code=status.HTTP_204_NO_CONTENT)
    async def event(
        body: OutboxEvent,
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> None:
        await service.accept_event(body, idempotency_key=idempotency_key)

    return router


def register_release_control_plane(app: FastAPI, service: ReleaseControlService) -> None:
    paths = frozenset(
        {
            service.health_path,
            service.drain_path,
        }
    )
    app.add_middleware(
        ReleaseActivityMiddleware,
        activity=service.activity,
        excluded_paths=paths,
        release_id=service.release_id,
        lease_epoch=service.lease_epoch,
        gateway_token=service._control_token,
    )
    app.include_router(create_release_control_router(service))
    app.state.release_control = service


__all__ = [
    "DrainRequest",
    "EventHandler",
    "HealthProvider",
    "IngressHandler",
    "ReleaseActivity",
    "ReleaseActivityMiddleware",
    "ReleaseControlConfigurationError",
    "ReleaseControlService",
    "create_release_control_router",
    "register_release_control_plane",
]
