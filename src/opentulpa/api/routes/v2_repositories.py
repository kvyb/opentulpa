"""Owner-only repository workspace control plane."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field

from opentulpa.api.routes.v2_principal import (
    V2Principal,
    require_v2_scope,
    resolve_v2_principal,
)
from opentulpa.capabilities.credentials import CapabilityAPIScope
from opentulpa.repositories.service import (
    RepositoryPublishError,
    RepositoryWorkspaceConflictError,
    RepositoryWorkspaceError,
    RepositoryWorkspaceNotFoundError,
    RepositoryWorkspaceService,
)


class RepositoryWorkspaceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str = Field(min_length=1, max_length=8_192)
    repository_url: AnyHttpUrl
    base_ref: str = Field(default="main", min_length=1, max_length=300)
    branch: str | None = Field(default=None, min_length=1, max_length=250)
    provider: str = Field(default="auto", pattern=r"^(auto|local|daytona)$")


class RepositoryPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str = Field(min_length=1, max_length=8_192)
    expected_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(default="", max_length=20_000)
    draft: bool = True


def _public_workspace(workspace: Any) -> dict[str, Any]:
    return cast(
        "dict[str, Any]",
        workspace.model_dump(
            mode="json",
            exclude={"tenant_id", "provider_workspace_id"},
        ),
    )


def register_v2_repository_routes(
    app: FastAPI,
    *,
    get_repositories: Callable[[], RepositoryWorkspaceService | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
) -> None:
    def service() -> RepositoryWorkspaceService:
        repositories = get_repositories()
        if repositories is None:
            raise HTTPException(status_code=503, detail="repository workspaces unavailable")
        return repositories

    async def owner(request: Request, *, write: bool = False) -> Any:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(
            principal,
            (
                CapabilityAPIScope.AGENT_RUN_SUBMIT.value
                if write
                else CapabilityAPIScope.AGENT_RUN_REPLAY.value
            ),
        )
        if principal.trust_class != "owner":
            raise HTTPException(status_code=403, detail="owner access required")
        return principal

    @app.post("/v2/repositories/workspaces", status_code=201)
    async def create_workspace(
        body: RepositoryWorkspaceCreate,
        request: Request,
    ) -> dict[str, Any]:
        principal = await owner(request, write=True)
        try:
            workspace = await service().open(
                tenant_id=principal.tenant_id,
                thread_id=body.thread_id,
                repository_url=str(body.repository_url),
                base_ref=body.base_ref,
                branch=body.branch,
                provider=body.provider,
            )
        except RepositoryWorkspaceError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return _public_workspace(workspace)

    @app.get("/v2/repositories/workspaces")
    async def list_workspaces(
        request: Request,
        include_closed: bool = False,
    ) -> dict[str, Any]:
        principal = await owner(request)
        workspaces = await service().list(
            tenant_id=principal.tenant_id,
            include_closed=include_closed,
        )
        return {
            "workspaces": [
                _public_workspace(workspace)
                for workspace in workspaces
            ]
        }

    @app.get("/v2/repositories/workspaces/active")
    async def active_workspace(
        request: Request,
        thread_id: str = Query(min_length=1, max_length=8_192),
    ) -> dict[str, Any]:
        principal = await owner(request)
        workspace = await service().active(
            tenant_id=principal.tenant_id,
            thread_id=thread_id,
        )
        if workspace is None:
            raise HTTPException(status_code=404, detail="no repository workspace is active")
        return _public_workspace(workspace)

    @app.get("/v2/repositories/workspaces/{workspace_id}")
    async def workspace_status(
        workspace_id: str,
        request: Request,
        thread_id: str = Query(min_length=1, max_length=8_192),
    ) -> dict[str, Any]:
        principal = await owner(request)
        try:
            result = await service().status(
                tenant_id=principal.tenant_id,
                thread_id=thread_id,
                workspace_id=workspace_id,
            )
        except RepositoryWorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RepositoryWorkspaceError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        result.pop("tenant_id", None)
        result.pop("provider_workspace_id", None)
        return result

    @app.post("/v2/repositories/workspaces/{workspace_id}/publish")
    async def publish_workspace(
        workspace_id: str,
        body: RepositoryPublishRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await owner(request, write=True)
        try:
            return await service().publish(
                tenant_id=principal.tenant_id,
                thread_id=body.thread_id,
                workspace_id=workspace_id,
                expected_head_sha=body.expected_head_sha,
                title=body.title,
                body=body.body,
                draft=body.draft,
            )
        except RepositoryWorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RepositoryWorkspaceConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RepositoryPublishError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RepositoryWorkspaceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.delete("/v2/repositories/workspaces/{workspace_id}")
    async def close_workspace(
        workspace_id: str,
        request: Request,
        thread_id: str = Query(min_length=1, max_length=8_192),
    ) -> dict[str, Any]:
        principal = await owner(request, write=True)
        try:
            workspace = await service().close(
                tenant_id=principal.tenant_id,
                thread_id=thread_id,
                workspace_id=workspace_id,
            )
        except RepositoryWorkspaceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _public_workspace(workspace)


__all__ = [
    "RepositoryPublishRequest",
    "RepositoryWorkspaceCreate",
    "register_v2_repository_routes",
]
