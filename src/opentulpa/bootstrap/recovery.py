"""Stable recovery service and FastAPI router independent from mutable releases."""

from __future__ import annotations

import asyncio
import hmac
import logging
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.bootstrap.host import ReleaseHostError
from opentulpa.bootstrap.models import (
    ActivationRecord,
    ReleaseLease,
    ReleaseOrigin,
    ReleaseRecord,
)
from opentulpa.bootstrap.store import BootstrapConflictError
from opentulpa.bootstrap.supervisor import ActivationError, BootstrapSupervisor

logger = logging.getLogger(__name__)


class RecoveryActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = Field(default="Owner requested recovery", max_length=4_000)


class RecoveryActivationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    release: ReleaseRecord
    origin: ReleaseOrigin | None = None
    reason: str = Field(default="Owner requested activation", max_length=4_000)
    start: bool = True


class RecoveryInitialInstallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    release: ReleaseRecord


class RecoveryService:
    """Small owner control surface that remains usable when a release is dead."""

    def __init__(self, supervisor: BootstrapSupervisor) -> None:
        self._supervisor = supervisor
        self._tasks: dict[str, asyncio.Task[ActivationRecord]] = {}

    async def request_activation(
        self,
        *,
        release: ReleaseRecord,
        origin: ReleaseOrigin | None,
        reason: str,
        start: bool,
    ) -> ActivationRecord:
        activation = await self._supervisor.request_activation(
            release,
            origin=origin,
            reason=reason,
        )
        if start:
            self._spawn(activation)
        return activation

    async def install_initial(self, release: ReleaseRecord) -> ReleaseLease:
        return await self._supervisor.install_initial(release)

    def status(self) -> dict[str, Any]:
        store = self._supervisor.store
        state = store.get_state()
        return {
            "schema_version": store.schema_version,
            "state": state.model_dump(mode="json"),
            "releases": [
                {
                    "id": release.id,
                    "candidate_id": release.candidate_id,
                    "source_commit": release.source_commit,
                    "artifact_digest": release.artifact_digest,
                    "created_at": release.created_at,
                }
                for release in store.list_releases(limit=20)
            ],
            "activations": [
                self._public_activation(activation)
                for activation in store.list_activations(limit=50)
            ],
            "pending_reports": len(store.pending_outbox(limit=1_000)),
        }

    async def request_rollback(
        self,
        *,
        reason: str,
        origin: ReleaseOrigin | None = None,
    ) -> ActivationRecord:
        activation = await self._supervisor.request_rollback(origin=origin, reason=reason)
        self._spawn(activation)
        return activation

    def _spawn(self, activation: ActivationRecord) -> None:
        task = asyncio.create_task(
            self._supervisor.activate(activation.id),
            name=f"opentulpa-bootstrap-activation:{activation.id}",
        )
        self._tasks[activation.id] = task
        task.add_done_callback(lambda completed: self._activation_done(activation.id, completed))

    def _activation_done(
        self,
        activation_id: str,
        task: asyncio.Task[ActivationRecord],
    ) -> None:
        self._tasks.pop(activation_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "background activation failed: activation=%s error_type=%s",
                activation_id,
                type(error).__name__,
            )

    async def cancel(self, activation_id: str) -> ActivationRecord:
        return await self._supervisor.cancel(activation_id)

    async def shutdown(self) -> None:
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def get_activation(self, activation_id: str) -> ActivationRecord | None:
        return self._supervisor.store.get_activation(activation_id)

    async def enter_safe_mode(self) -> None:
        await self._supervisor.enter_safe_mode()

    async def restart_last_known_good(self) -> None:
        await self._supervisor.recover_last_known_good()

    async def wait(self, activation_id: str) -> ActivationRecord:
        task = self._tasks.get(activation_id)
        if task is not None:
            return await task
        activation = self._supervisor.store.get_activation(activation_id)
        if activation is None:
            raise ActivationError("activation_missing", "The activation was not found.")
        return activation

    @staticmethod
    def _public_activation(activation: ActivationRecord) -> dict[str, Any]:
        return {
            "id": activation.id,
            "kind": activation.kind,
            "target_release_id": activation.target_release_id,
            "previous_release_id": activation.previous_release_id,
            "status": activation.status,
            "revision": activation.revision,
            "failure_code": activation.failure_code,
            "failure_message": activation.failure_message,
            "created_at": activation.created_at,
            "updated_at": activation.updated_at,
        }


def create_recovery_router(
    service: RecoveryService,
    *,
    recovery_token: str,
) -> APIRouter:
    """Create the immutable recovery routes protected by a separate bearer token."""

    token = str(recovery_token or "").strip()
    if len(token) < 32:
        raise ValueError("recovery token must contain at least 32 characters")

    async def authorize(authorization: str | None = Header(default=None)) -> None:
        scheme, _, supplied = str(authorization or "").partition(" ")
        if scheme.casefold() != "bearer" or not hmac.compare_digest(supplied, token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid recovery credentials are required",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def reject_browser_request(request: Request) -> None:
        browser_headers = {
            name.casefold()
            for name in request.headers
            if name.casefold() in {"origin", "referer"}
            or name.casefold().startswith("sec-fetch-")
        }
        if browser_headers:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="recovery APIs accept only non-browser host clients",
            )

    router = APIRouter()
    protected = APIRouter(
        dependencies=[Depends(reject_browser_request), Depends(authorize)]
    )

    @protected.get("/bootstrap/v1/status")
    async def bootstrap_status() -> dict[str, Any]:
        return service.status()

    @protected.post("/bootstrap/v1/activations", status_code=status.HTTP_202_ACCEPTED)
    async def request_activation(body: RecoveryActivationRequest) -> dict[str, Any]:
        try:
            activation = await service.request_activation(
                release=body.release,
                origin=body.origin,
                reason=body.reason,
                start=body.start,
            )
        except (ActivationError, BootstrapConflictError) as exc:
            message = exc.public_message if isinstance(exc, ActivationError) else str(exc)
            raise HTTPException(status_code=409, detail=message) from exc
        return RecoveryService._public_activation(activation)

    @protected.post("/bootstrap/v1/releases/initial", status_code=status.HTTP_201_CREATED)
    async def install_initial(body: RecoveryInitialInstallRequest) -> dict[str, Any]:
        try:
            lease = await service.install_initial(body.release)
        except BootstrapConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (ActivationError, ReleaseHostError) as exc:
            logger.error("initial release installation failed: error_type=%s", type(exc).__name__)
            raise HTTPException(
                status_code=503,
                detail="the initial release could not be installed",
            ) from exc
        return {
            "release_id": lease.release_id,
            "lease_epoch": lease.epoch,
            "status": lease.status,
        }

    @protected.get("/bootstrap/v1/activations/{activation_id}")
    async def activation_status(activation_id: str) -> dict[str, Any]:
        activation = service.get_activation(activation_id)
        if activation is None:
            raise HTTPException(status_code=404, detail="activation was not found")
        return RecoveryService._public_activation(activation)

    @protected.post("/bootstrap/v1/rollback", status_code=status.HTTP_202_ACCEPTED)
    async def rollback(body: RecoveryActionRequest) -> dict[str, Any]:
        try:
            activation = await service.request_rollback(reason=body.reason)
        except ActivationError as exc:
            raise HTTPException(status_code=409, detail=exc.public_message) from exc
        return RecoveryService._public_activation(activation)

    @protected.post("/bootstrap/v1/activations/{activation_id}/cancel")
    async def cancel_activation(activation_id: str) -> dict[str, Any]:
        try:
            activation = await service.cancel(activation_id)
        except ActivationError as exc:
            raise HTTPException(status_code=409, detail=exc.public_message) from exc
        return RecoveryService._public_activation(activation)

    @protected.post("/bootstrap/v1/safe-mode", status_code=status.HTTP_202_ACCEPTED)
    async def safe_mode() -> dict[str, str]:
        await service.enter_safe_mode()
        return {"status": "safe_mode"}

    @protected.post("/bootstrap/v1/restart", status_code=status.HTTP_202_ACCEPTED)
    async def restart() -> dict[str, str]:
        try:
            await service.restart_last_known_good()
        except ActivationError as exc:
            raise HTTPException(status_code=409, detail=exc.public_message) from exc
        return {"status": "restarted"}

    @router.api_route(
        "/recovery",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    @router.api_route(
        "/recovery/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        include_in_schema=False,
    )
    async def reserve_recovery_path(path: str = "") -> Response:
        del path
        return Response(
            status_code=status.HTTP_404_NOT_FOUND,
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    router.include_router(protected)
    return router


__all__ = [
    "RecoveryActionRequest",
    "RecoveryActivationRequest",
    "RecoveryInitialInstallRequest",
    "RecoveryService",
    "create_recovery_router",
]
