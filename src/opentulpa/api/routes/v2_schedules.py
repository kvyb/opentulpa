"""V2 tenant-scoped schedule routes."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import ConfigDict, Field

from opentulpa.schedules.models import Schedule, ScheduleWrite
from opentulpa.schedules.service import (
    ScheduleConflictError,
    ScheduleNotFoundError,
    ScheduleService,
)


class SchedulePrincipal(Protocol):
    tenant_id: str
    actor_id: str


class ScheduleSaveRequest(ScheduleWrite):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str | None = Field(default=None, min_length=1, max_length=100)
    expected_revision: int | None = Field(default=None, ge=1)


async def _resolve_principal(
    request: Request,
    resolver: Callable[[Request], SchedulePrincipal | Awaitable[SchedulePrincipal]],
) -> SchedulePrincipal:
    resolved = resolver(request)
    principal = await resolved if inspect.isawaitable(resolved) else resolved
    if not str(getattr(principal, "tenant_id", "") or "").strip():
        raise HTTPException(status_code=401, detail="authenticated tenant is required")
    if not str(getattr(principal, "actor_id", "") or "").strip():
        raise HTTPException(status_code=401, detail="authenticated actor is required")
    return principal


def register_v2_schedule_routes(
    app: FastAPI,
    *,
    get_schedule_service: Callable[[], ScheduleService],
    resolve_principal: Callable[[Request], SchedulePrincipal | Awaitable[SchedulePrincipal]],
) -> None:
    """Register typed schedule CRUD without accepting model-visible tenant IDs."""

    @app.get("/v2/schedules")
    async def list_schedules(request: Request) -> dict[str, list[Schedule]]:
        principal = await _resolve_principal(request, resolve_principal)
        return {
            "schedules": get_schedule_service().list(tenant_id=principal.tenant_id),
        }

    @app.post("/v2/schedules")
    async def save_schedule(
        body: ScheduleSaveRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Schedule]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            schedule = get_schedule_service().save(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                schedule_id=body.id,
                expected_revision=body.expected_revision,
                write=ScheduleWrite.model_validate(
                    body.model_dump(exclude={"id", "expected_revision"})
                ),
            )
        except ScheduleConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if schedule.revision == 1:
            response.status_code = 201
        return {"schedule": schedule}

    @app.delete("/v2/schedules/{schedule_id}")
    async def delete_schedule(
        schedule_id: str,
        request: Request,
        expected_revision: int = Query(ge=1),
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            get_schedule_service().delete(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                schedule_id=schedule_id,
                expected_revision=expected_revision,
            )
        except ScheduleNotFoundError as exc:
            raise HTTPException(status_code=404, detail="schedule not found") from exc
        except ScheduleConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"deleted": True, "schedule_id": schedule_id}


__all__ = ["SchedulePrincipal", "ScheduleSaveRequest", "register_v2_schedule_routes"]
