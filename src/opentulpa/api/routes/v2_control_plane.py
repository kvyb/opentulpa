"""Authenticated V2 control plane for agent specs, triggers, and secret handles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, JsonValue, SecretStr

from opentulpa.api.routes.v2_principal import V2Principal, resolve_v2_principal
from opentulpa.secrets.models import SecretHandle, SecretScope
from opentulpa.secrets.service import SecretVaultService
from opentulpa.secrets.vault import (
    SecretVaultConflictError,
    SecretVaultNotFoundError,
)
from opentulpa.specs.models import (
    AgentSpec,
    AgentSpecWrite,
    DeliverySpec,
    TriggerSource,
    TriggerSpec,
    TriggerSpecWrite,
)
from opentulpa.specs.protocol import AgentSpecRef, ProtocolSlug
from opentulpa.specs.service import AgentSpecService, TriggerSpecService
from opentulpa.specs.store import SpecConflictError, SpecNotFoundError


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class AgentSpecRevisionRequest(_RequestModel):
    id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    expected_revision: int | None = Field(default=None, ge=1)
    spec: AgentSpecWrite


class LocalAgentSpecRefRequest(_RequestModel):
    spec_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    revision: int = Field(ge=1)


class TriggerSpecWriteRequest(_RequestModel):
    name: str = Field(min_length=1, max_length=200)
    source: TriggerSource
    exposure: str = Field(pattern=r"^(private|external)$")
    agent_spec: LocalAgentSpecRefRequest
    instruction: str = Field(min_length=1, max_length=200_000)
    delivery: DeliverySpec = Field(default_factory=DeliverySpec)
    enabled: bool = True
    source_key: str | None = Field(default=None, min_length=1, max_length=300)
    source_revision: int | None = Field(default=None, ge=1)
    labels: dict[ProtocolSlug, str] = Field(default_factory=dict, max_length=100)

    def to_write(self, *, tenant_id: str) -> TriggerSpecWrite:
        payload: dict[str, JsonValue] = self.model_dump(
            mode="json",
            exclude={"agent_spec"},
        )
        return TriggerSpecWrite.model_validate(
            {
                **payload,
                "agent_spec": AgentSpecRef(
                    tenant_id=tenant_id,
                    spec_id=self.agent_spec.spec_id,
                    revision=self.agent_spec.revision,
                ),
            }
        )


class TriggerSpecRevisionRequest(_RequestModel):
    id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    expected_revision: int | None = Field(default=None, ge=1)
    trigger: TriggerSpecWriteRequest


class ActivateRevisionRequest(_RequestModel):
    revision: int = Field(ge=1)
    expected_active_revision: int | None = Field(default=None, ge=1)


class RollbackRevisionRequest(_RequestModel):
    expected_active_revision: int = Field(ge=1)


class PendingSecretRequest(_RequestModel):
    id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    scopes: tuple[SecretScope, ...] = Field(min_length=1, max_length=100)


class SecretStoreRequest(_RequestModel):
    expected_revision: int = Field(ge=1)
    value: SecretStr
    scopes: tuple[SecretScope, ...] | None = Field(default=None, max_length=100)


class TriggerEventRequest(_RequestModel):
    source_event_id: str = Field(min_length=1, max_length=300)
    event_type: ProtocolSlug
    source: ProtocolSlug
    payload: dict[str, JsonValue] = Field(default_factory=dict)


def _spec_not_found(exc: Exception, *, label: str) -> HTTPException:
    return HTTPException(status_code=404, detail=f"{label} not found")


def _spec_conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


def _secret_not_found(exc: Exception) -> HTTPException:
    return HTTPException(status_code=404, detail="secret handle not found")


def _secret_conflict(exc: Exception) -> HTTPException:
    return HTTPException(status_code=409, detail=str(exc))


def _active_agent_view(service: AgentSpecService, spec: AgentSpec) -> dict[str, Any]:
    active = service.get_active(tenant_id=spec.tenant_id, spec_id=spec.id)
    return {
        "spec": spec,
        "active_revision": active.revision if active is not None else None,
    }


def _active_trigger_view(service: TriggerSpecService, trigger: TriggerSpec) -> dict[str, Any]:
    active = service.get_active(tenant_id=trigger.tenant_id, trigger_id=trigger.id)
    return {
        "trigger": trigger,
        "active_revision": active.revision if active is not None else None,
    }


def register_v2_control_plane_routes(
    app: FastAPI,
    *,
    get_agent_specs: Callable[[], AgentSpecService | None],
    get_trigger_specs: Callable[[], TriggerSpecService | None],
    get_secret_vault: Callable[[], SecretVaultService | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
    on_trigger_changed: Callable[[TriggerSpec], None] | None = None,
    on_trigger_deactivated: Callable[[str, str], None] | None = None,
    dispatch_trigger_event: Callable[..., Awaitable[Any]] | None = None,
) -> None:
    """Register tenant-safe dynamic control-plane routes."""

    def agent_service() -> AgentSpecService:
        service = get_agent_specs()
        if service is None:
            raise HTTPException(status_code=503, detail="AgentSpec control plane is unavailable")
        return service

    def trigger_service() -> TriggerSpecService:
        service = get_trigger_specs()
        if service is None:
            raise HTTPException(status_code=503, detail="TriggerSpec control plane is unavailable")
        return service

    def secret_service() -> SecretVaultService:
        service = get_secret_vault()
        if service is None:
            raise HTTPException(status_code=503, detail="secret vault is unavailable")
        return service

    @app.post("/v2/agent-specs/seed-defaults", status_code=status.HTTP_201_CREATED)
    async def seed_default_agent_specs(request: Request) -> dict[str, list[AgentSpec]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        specs = agent_service().seed_defaults(
            tenant_id=principal.tenant_id,
            actor_id=principal.actor_id,
        )
        return {"specs": specs}

    @app.get("/v2/agent-specs")
    async def list_agent_specs(request: Request) -> dict[str, list[dict[str, Any]]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        service = agent_service()
        return {
            "specs": [
                _active_agent_view(service, spec)
                for spec in service.list_latest(tenant_id=principal.tenant_id)
            ]
        }

    @app.post("/v2/agent-specs")
    async def save_agent_spec(
        body: AgentSpecRevisionRequest,
        request: Request,
        response: Response,
    ) -> dict[str, AgentSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            spec = agent_service().save(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                spec_id=body.id,
                expected_revision=body.expected_revision,
                write=body.spec,
            )
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        if spec.revision == 1:
            response.status_code = status.HTTP_201_CREATED
        return {"spec": spec}

    @app.get("/v2/agent-specs/{spec_id}/revisions")
    async def list_agent_spec_revisions(
        spec_id: str,
        request: Request,
    ) -> dict[str, list[AgentSpec]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        revisions = agent_service().list_revisions(
            tenant_id=principal.tenant_id,
            spec_id=spec_id,
        )
        if not revisions:
            raise HTTPException(status_code=404, detail="AgentSpec not found")
        return {"revisions": revisions}

    @app.post("/v2/agent-specs/{spec_id}/activate")
    async def activate_agent_spec(
        spec_id: str,
        body: ActivateRevisionRequest,
        request: Request,
    ) -> dict[str, AgentSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            spec = agent_service().activate(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                spec_id=spec_id,
                revision=body.revision,
                expected_active_revision=body.expected_active_revision,
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="AgentSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        return {"spec": spec}

    @app.post("/v2/agent-specs/{spec_id}/rollback")
    async def rollback_agent_spec(
        spec_id: str,
        body: RollbackRevisionRequest,
        request: Request,
    ) -> dict[str, AgentSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            spec = agent_service().rollback(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                spec_id=spec_id,
                expected_active_revision=body.expected_active_revision,
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="AgentSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        return {"spec": spec}

    @app.get("/v2/agent-specs/{spec_id}")
    async def get_agent_spec(
        spec_id: str,
        request: Request,
        revision: int | None = Query(default=None, ge=1),
    ) -> dict[str, AgentSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        service = agent_service()
        spec = (
            service.get_revision(
                tenant_id=principal.tenant_id,
                spec_id=spec_id,
                revision=revision,
            )
            if revision is not None
            else service.get_active(tenant_id=principal.tenant_id, spec_id=spec_id)
        )
        if spec is None:
            raise HTTPException(status_code=404, detail="AgentSpec not found")
        return {"spec": spec}

    @app.delete("/v2/agent-specs/{spec_id}")
    async def deactivate_agent_spec(
        spec_id: str,
        request: Request,
        expected_active_revision: int = Query(ge=1),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            ref = agent_service().deactivate(
                tenant_id=principal.tenant_id,
                spec_id=spec_id,
                expected_active_revision=expected_active_revision,
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="AgentSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        return {"deactivated": True, "agent_spec": ref}

    @app.get("/v2/trigger-specs")
    async def list_trigger_specs(request: Request) -> dict[str, list[dict[str, Any]]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        service = trigger_service()
        return {
            "triggers": [
                _active_trigger_view(service, trigger)
                for trigger in service.list_latest(tenant_id=principal.tenant_id)
            ]
        }

    @app.post("/v2/trigger-specs")
    async def save_trigger_spec(
        body: TriggerSpecRevisionRequest,
        request: Request,
        response: Response,
    ) -> dict[str, TriggerSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            trigger = trigger_service().save(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                trigger_id=body.id,
                expected_revision=body.expected_revision,
                write=body.trigger.to_write(tenant_id=principal.tenant_id),
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="AgentSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if trigger.revision == 1:
            response.status_code = status.HTTP_201_CREATED
        return {"trigger": trigger}

    @app.get("/v2/trigger-specs/{trigger_id}/revisions")
    async def list_trigger_spec_revisions(
        trigger_id: str,
        request: Request,
    ) -> dict[str, list[TriggerSpec]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        revisions = trigger_service().list_revisions(
            tenant_id=principal.tenant_id,
            trigger_id=trigger_id,
        )
        if not revisions:
            raise HTTPException(status_code=404, detail="TriggerSpec not found")
        return {"revisions": revisions}

    @app.post("/v2/trigger-specs/{trigger_id}/activate")
    async def activate_trigger_spec(
        trigger_id: str,
        body: ActivateRevisionRequest,
        request: Request,
    ) -> dict[str, TriggerSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            trigger = trigger_service().activate(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                trigger_id=trigger_id,
                revision=body.revision,
                expected_active_revision=body.expected_active_revision,
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="TriggerSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        if on_trigger_changed is not None:
            on_trigger_changed(trigger)
        return {"trigger": trigger}

    @app.post("/v2/trigger-specs/{trigger_id}/rollback")
    async def rollback_trigger_spec(
        trigger_id: str,
        body: RollbackRevisionRequest,
        request: Request,
    ) -> dict[str, TriggerSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            trigger = trigger_service().rollback(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                trigger_id=trigger_id,
                expected_active_revision=body.expected_active_revision,
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="TriggerSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        if on_trigger_changed is not None:
            on_trigger_changed(trigger)
        return {"trigger": trigger}

    @app.post("/v2/trigger-specs/{trigger_id}/events")
    async def dispatch_event_trigger(
        trigger_id: str,
        body: TriggerEventRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        if dispatch_trigger_event is None:
            raise HTTPException(status_code=503, detail="trigger dispatcher is unavailable")
        snapshot = await dispatch_trigger_event(
            tenant_id=principal.tenant_id,
            trigger_id=trigger_id,
            source_event_id=body.source_event_id,
            event_type=body.event_type,
            source=body.source,
            authenticated=True,
            payload=body.payload,
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="active event trigger not found")
        return {
            "run_id": snapshot.run_id,
            "status": snapshot.status,
            "final_text": snapshot.final_text,
            "approval_required": snapshot.status == "interrupted",
        }

    @app.get("/v2/trigger-specs/{trigger_id}")
    async def get_trigger_spec(
        trigger_id: str,
        request: Request,
        revision: int | None = Query(default=None, ge=1),
    ) -> dict[str, TriggerSpec]:
        principal = await resolve_v2_principal(request, resolve_principal)
        service = trigger_service()
        trigger = (
            service.get_revision(
                tenant_id=principal.tenant_id,
                trigger_id=trigger_id,
                revision=revision,
            )
            if revision is not None
            else service.get_active(tenant_id=principal.tenant_id, trigger_id=trigger_id)
        )
        if trigger is None:
            raise HTTPException(status_code=404, detail="TriggerSpec not found")
        return {"trigger": trigger}

    @app.delete("/v2/trigger-specs/{trigger_id}")
    async def deactivate_trigger_spec(
        trigger_id: str,
        request: Request,
        expected_active_revision: int = Query(ge=1),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        try:
            revision = trigger_service().deactivate(
                tenant_id=principal.tenant_id,
                trigger_id=trigger_id,
                expected_active_revision=expected_active_revision,
            )
        except SpecNotFoundError as exc:
            raise _spec_not_found(exc, label="TriggerSpec") from exc
        except SpecConflictError as exc:
            raise _spec_conflict(exc) from exc
        if on_trigger_deactivated is not None:
            on_trigger_deactivated(principal.tenant_id, trigger_id)
        return {
            "deactivated": True,
            "trigger_id": trigger_id,
            "revision": revision,
        }

    @app.get("/v2/secrets")
    async def list_secret_handles(
        request: Request,
        response: Response,
    ) -> dict[str, list[SecretHandle]]:
        principal = await resolve_v2_principal(request, resolve_principal)
        response.headers["Cache-Control"] = "no-store"
        return {"secrets": secret_service().list(tenant_id=principal.tenant_id)}

    @app.post("/v2/secrets/pending", status_code=status.HTTP_201_CREATED)
    async def create_pending_secret(
        body: PendingSecretRequest,
        request: Request,
        response: Response,
    ) -> dict[str, SecretHandle]:
        principal = await resolve_v2_principal(request, resolve_principal)
        response.headers["Cache-Control"] = "no-store"
        try:
            handle = secret_service().create_pending(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                secret_id=body.id,
                name=body.name,
                scopes=body.scopes,
            )
        except SecretVaultConflictError as exc:
            raise _secret_conflict(exc) from exc
        return {"secret": handle}

    @app.get("/v2/secrets/{secret_id}")
    async def get_secret_handle(
        secret_id: str,
        request: Request,
        response: Response,
    ) -> dict[str, SecretHandle]:
        principal = await resolve_v2_principal(request, resolve_principal)
        response.headers["Cache-Control"] = "no-store"
        handle = secret_service().get(tenant_id=principal.tenant_id, secret_id=secret_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="secret handle not found")
        return {"secret": handle}

    @app.put("/v2/secrets/{secret_id}")
    async def store_secret_value(
        secret_id: str,
        body: SecretStoreRequest,
        request: Request,
        response: Response,
    ) -> dict[str, SecretHandle]:
        principal = await resolve_v2_principal(request, resolve_principal)
        response.headers["Cache-Control"] = "no-store"
        try:
            handle = secret_service().store(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                secret_id=secret_id,
                expected_revision=body.expected_revision,
                value=body.value,
                scopes=body.scopes,
            )
        except SecretVaultNotFoundError as exc:
            raise _secret_not_found(exc) from exc
        except SecretVaultConflictError as exc:
            raise _secret_conflict(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid secret value") from exc
        return {"secret": handle}

    @app.delete("/v2/secrets/{secret_id}")
    async def revoke_secret(
        secret_id: str,
        request: Request,
        response: Response,
        expected_revision: Annotated[int, Query(ge=1)],
    ) -> dict[str, SecretHandle]:
        principal = await resolve_v2_principal(request, resolve_principal)
        response.headers["Cache-Control"] = "no-store"
        try:
            handle = secret_service().revoke(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                secret_id=secret_id,
                expected_revision=expected_revision,
            )
        except SecretVaultNotFoundError as exc:
            raise _secret_not_found(exc) from exc
        except SecretVaultConflictError as exc:
            raise _secret_conflict(exc) from exc
        return {"secret": handle}


__all__ = [
    "ActivateRevisionRequest",
    "AgentSpecRevisionRequest",
    "LocalAgentSpecRefRequest",
    "PendingSecretRequest",
    "RollbackRevisionRequest",
    "SecretStoreRequest",
    "TriggerEventRequest",
    "TriggerSpecRevisionRequest",
    "TriggerSpecWriteRequest",
    "register_v2_control_plane_routes",
]
