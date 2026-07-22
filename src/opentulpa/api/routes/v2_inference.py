"""Owner-only V2 APIs for conversation-scoped inference selection."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.api.routes.v2_principal import (
    V2Principal,
    require_v2_scope,
    resolve_v2_principal,
)
from opentulpa.capabilities.credentials import CapabilityAPIScope
from opentulpa.inference.models import InferenceProvider, InferenceSelection
from opentulpa.inference.service import (
    InferenceConflictError,
    InferenceService,
    InferenceUnavailableError,
)


class ThreadInferencePort(Protocol):
    async def get_thread_inference(
        self, *, tenant_id: str, thread_id: str
    ) -> dict[str, Any] | None: ...

    async def update_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        expected_revision: int,
        selection: InferenceSelection | None,
    ) -> dict[str, Any] | None: ...

    async def codex_preference_count(self, tenant_id: str) -> int: ...

    async def reset_codex_preferences(self, tenant_id: str) -> int: ...


class ThreadInferenceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expected_revision: int = Field(ge=0)
    selection: InferenceSelection | None


def register_v2_inference_routes(
    app: FastAPI,
    *,
    get_inference: Callable[[], InferenceService | None],
    get_threads: Callable[[], ThreadInferencePort | None],
    resolve_principal: Callable[[Request], V2Principal | Awaitable[V2Principal]],
) -> None:
    def services() -> tuple[InferenceService, ThreadInferencePort]:
        inference = get_inference()
        threads = get_threads()
        if inference is None or threads is None:
            raise HTTPException(status_code=503, detail="inference service unavailable")
        return inference, threads

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

    @app.get("/v2/inference")
    async def inference_status(request: Request) -> dict[str, Any]:
        principal = await owner(request)
        inference, _ = services()
        return await inference.status(principal.tenant_id)

    @app.get("/v2/inference/models")
    async def inference_models(
        request: Request,
        provider: InferenceProvider,
        query: str = Query(default="", max_length=200),
    ) -> dict[str, Any]:
        principal = await owner(request)
        inference, _ = services()
        try:
            models = await inference.models(principal.tenant_id, provider, query=query)
        except InferenceUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="model discovery is unavailable") from exc
        return {"provider": provider, "models": [model.model_dump(mode="json") for model in models]}

    @app.post("/v2/inference/codex/device-logins", status_code=201)
    async def start_codex_login(request: Request) -> dict[str, Any]:
        principal = await owner(request, write=True)
        inference, _ = services()
        try:
            return await inference.start_device_login(principal.tenant_id)
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Codex login could not be started") from exc

    @app.get("/v2/inference/codex/device-logins/{login_id}")
    async def get_codex_login(login_id: str, request: Request) -> dict[str, Any]:
        principal = await owner(request)
        inference, _ = services()
        result = await inference.get_device_login(principal.tenant_id, login_id)
        if result is None:
            raise HTTPException(status_code=404, detail="device login not found")
        return result

    @app.delete("/v2/inference/codex/device-logins/{login_id}", status_code=204)
    async def cancel_codex_login(login_id: str, request: Request) -> None:
        principal = await owner(request, write=True)
        inference, _ = services()
        if not await inference.cancel_device_login(principal.tenant_id, login_id):
            raise HTTPException(status_code=404, detail="device login not found")

    @app.delete("/v2/inference/codex/credential")
    async def delete_codex_credential(
        request: Request,
        reset_threads: bool = False,
    ) -> dict[str, Any]:
        principal = await owner(request, write=True)
        inference, threads = services()
        count = await threads.codex_preference_count(principal.tenant_id)
        if count and not reset_threads:
            raise HTTPException(
                status_code=409,
                detail="Codex is selected by existing conversations; confirm reset_threads",
            )
        reset = await threads.reset_codex_preferences(principal.tenant_id) if count else 0
        disconnected = inference.delete_credential(principal.tenant_id)
        return {"disconnected": disconnected, "reset_threads": reset}

    @app.get("/v2/agent/threads/{thread_id}/inference")
    async def get_thread_inference(thread_id: str, request: Request) -> dict[str, Any]:
        principal = await owner(request)
        _, threads = services()
        result = await threads.get_thread_inference(
            tenant_id=principal.tenant_id,
            thread_id=thread_id,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="agent thread not found")
        return result

    @app.patch("/v2/agent/threads/{thread_id}/inference")
    async def update_thread_inference(
        thread_id: str,
        body: ThreadInferenceUpdate,
        request: Request,
    ) -> dict[str, Any]:
        principal = await owner(request, write=True)
        _, threads = services()
        try:
            result = await threads.update_thread_inference(
                tenant_id=principal.tenant_id,
                thread_id=thread_id,
                expected_revision=body.expected_revision,
                selection=body.selection,
            )
        except InferenceConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InferenceUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result is None:
            raise HTTPException(status_code=404, detail="agent thread not found")
        return result


__all__ = ["ThreadInferencePort", "ThreadInferenceUpdate", "register_v2_inference_routes"]
