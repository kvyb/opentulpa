"""Authenticated V2 control plane for tenant capability revisions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from opentulpa.api.routes.v2_principal import V2Principal, resolve_v2_principal
from opentulpa.capabilities.revisions import (
    CapabilityRevisionConflictError,
    CapabilityRevisionNotFoundError,
)
from opentulpa.capabilities.service import (
    CapabilityControlService,
    CapabilityEvaluationUnavailableError,
    CapabilityRuntimeUnavailableError,
    CapabilityTestRequiredError,
)
from opentulpa.capabilities.workers import WorkerLifecycleError


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CapabilityTestRequest(_RequestModel):
    revision: int = Field(ge=1)


class CapabilityActivationRequest(_RequestModel):
    revision: int = Field(ge=1)
    expected_generation: int | None = Field(default=None, ge=1)
    config: dict[str, JsonValue] = Field(default_factory=dict, max_length=200)
    secret_handles: dict[str, str] = Field(default_factory=dict, max_length=100)
    refresh_agent_binding: bool = False


class CapabilityRollbackRequest(_RequestModel):
    expected_generation: int = Field(ge=1)
    config: dict[str, JsonValue] | None = Field(default=None, max_length=200)
    secret_handles: dict[str, str] | None = Field(default=None, max_length=100)


def register_v2_capability_routes(
    app: FastAPI,
    *,
    get_capabilities: Callable[[], CapabilityControlService | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
) -> None:
    """Register capability routes without composing them into an application."""

    def service() -> CapabilityControlService:
        value = get_capabilities()
        if value is None:
            raise HTTPException(
                status_code=503,
                detail="capability control plane is unavailable",
            )
        return value

    @app.post("/v2/capabilities/seed-bundled", status_code=status.HTTP_201_CREATED)
    async def seed_bundled(request: Request) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            manifests = service().seed_bundled(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
            )
        except CapabilityRevisionConflictError as exc:
            raise _conflict(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"capabilities": manifests}

    @app.get("/v2/capabilities")
    async def list_capabilities(request: Request) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        return {"capabilities": service().list(tenant_id=principal.tenant_id)}

    @app.get("/v2/capabilities/{capability_name}/revisions")
    async def list_capability_revisions(
        capability_name: str,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        revisions = service().revisions(
            tenant_id=principal.tenant_id,
            capability_name=capability_name,
        )
        if not revisions:
            raise HTTPException(status_code=404, detail="capability not found")
        return {"revisions": revisions}

    @app.get("/v2/capabilities/{capability_name}")
    async def get_capability(
        capability_name: str,
        request: Request,
        revision: int | None = Query(default=None, ge=1),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        value = service().get(
            tenant_id=principal.tenant_id,
            capability_name=capability_name,
            revision=revision,
        )
        if value is None:
            raise HTTPException(status_code=404, detail="capability not found")
        return value

    @app.post("/v2/capabilities/{capability_name}/test")
    async def test_capability(
        capability_name: str,
        body: CapabilityTestRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            result = await service().test(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                capability_name=capability_name,
                revision=body.revision,
            )
        except CapabilityRevisionNotFoundError as exc:
            raise _not_found(exc) from exc
        except CapabilityEvaluationUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"test": result}

    @app.post("/v2/capabilities/{capability_name}/activate")
    async def activate_capability(
        capability_name: str,
        body: CapabilityActivationRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            activation = await service().activate(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                capability_name=capability_name,
                revision=body.revision,
                expected_generation=body.expected_generation,
                config=body.config,
                secret_handles=body.secret_handles,
                refresh_agent_binding=body.refresh_agent_binding,
            )
        except CapabilityRevisionNotFoundError as exc:
            raise _not_found(exc) from exc
        except CapabilityRevisionConflictError as exc:
            raise _conflict(exc) from exc
        except CapabilityTestRequiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CapabilityRuntimeUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except WorkerLifecycleError as exc:
            raise HTTPException(
                status_code=409,
                detail="capability workers did not become healthy",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"activation": activation}

    @app.post("/v2/capabilities/{capability_name}/rollback")
    async def rollback_capability(
        capability_name: str,
        body: CapabilityRollbackRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            activation = await service().rollback(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                capability_name=capability_name,
                expected_generation=body.expected_generation,
                config=body.config,
                secret_handles=body.secret_handles,
            )
        except CapabilityRevisionNotFoundError as exc:
            raise _not_found(exc) from exc
        except CapabilityRevisionConflictError as exc:
            raise _conflict(exc) from exc
        except CapabilityTestRequiredError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except CapabilityRuntimeUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except WorkerLifecycleError as exc:
            raise HTTPException(
                status_code=409,
                detail="capability workers did not become healthy",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"activation": activation}

    @app.delete("/v2/capabilities/{capability_name}")
    async def deactivate_capability(
        capability_name: str,
        request: Request,
        expected_generation: int = Query(ge=1),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            activation = await service().deactivate(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                capability_name=capability_name,
                expected_generation=expected_generation,
            )
        except CapabilityRevisionNotFoundError as exc:
            raise _not_found(exc) from exc
        except CapabilityRevisionConflictError as exc:
            raise _conflict(exc) from exc
        except WorkerLifecycleError as exc:
            raise HTTPException(
                status_code=409,
                detail="capability workers could not be stopped",
            ) from exc
        return {"deactivated": True, "activation": activation}


def _not_found(_: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail="capability not found")


def _conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


__all__ = [
    "CapabilityActivationRequest",
    "CapabilityRollbackRequest",
    "CapabilityTestRequest",
    "register_v2_capability_routes",
]
