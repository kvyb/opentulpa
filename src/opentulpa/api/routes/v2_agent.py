"""Authenticated v2 streaming routes for Deep Agent runs."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Annotated, Any, Literal, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from opentulpa.api.routes.v2_principal import (
    ResolvedV2Principal,
    V2Principal,
    require_v2_scope,
    resolve_v2_principal,
)
from opentulpa.capabilities.credentials import CapabilityAPIScope
from opentulpa.deep_agent.contracts import (
    AgentApproval,
    AgentRunCapabilityConflictError,
    AgentRunCheckpointConflictError,
    AgentRunEvent,
    AgentRunIdempotencyConflictError,
    AgentRunRequest,
    AgentRunSnapshot,
    ApprovalDecision,
)
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.specs import AgentRunBinding, AgentSpecRef, OriginRef
from opentulpa.tooling.contract import AgentRunContext, AgentRunKind

_CORRELATION_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class AgentPrincipal(V2Principal, Protocol):
    pass


class AgentRunService(Protocol):
    async def open_stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]: ...

    def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]: ...

    async def open_resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]: ...

    def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]: ...

    async def get_run(self, run_id: str) -> AgentRunSnapshot | None: ...

    def events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[AgentRunEvent]: ...

    async def cancel(self, run_id: str) -> AgentRunSnapshot: ...

    async def cancel_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> AgentRunSnapshot | None: ...

    async def create_thread(
        self, *, tenant_id: str, channel: str, title: str | None = None
    ) -> dict[str, Any]: ...

    async def ensure_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        channel: str,
    ) -> None: ...

    async def list_threads(
        self, *, tenant_id: str, cursor: str | None = None, limit: int = 50
    ) -> dict[str, Any]: ...

    async def thread_timeline(
        self, *, tenant_id: str, thread_id: str, cursor: int = 0, limit: int = 30
    ) -> dict[str, Any] | None: ...

    async def update_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None: ...


class AgentRunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    thread_id: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1, max_length=200_000)
    file_ids: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        default_factory=list,
        max_length=20,
    )


class AgentRunSteerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=200_000)
    file_ids: list[Annotated[str, Field(min_length=1, max_length=300)]] = Field(
        default_factory=list,
        max_length=20,
    )


class AgentRunResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    approval_id: str = Field(min_length=1, max_length=300)
    decision: Literal["approve", "edit", "reject"]
    edited_arguments: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_edited_arguments(self) -> AgentRunResumeRequest:
        if self.decision == "edit" and self.edited_arguments is None:
            raise ValueError("edited_arguments are required for edit decisions")
        if self.decision != "edit" and self.edited_arguments is not None:
            raise ValueError("edited_arguments are only allowed for edit decisions")
        return self


class AgentThreadCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=120)


class AgentThreadUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=120)
    archived: bool | None = None

    @model_validator(mode="after")
    def require_change(self) -> AgentThreadUpdateRequest:
        if self.title is None and self.archived is None:
            raise ValueError("title or archived is required")
        return self


def _correlation_id(request: Request) -> str:
    candidate = str(request.headers.get("x-correlation-id", "") or "").strip()
    if _CORRELATION_ID_RE.fullmatch(candidate):
        return candidate
    return f"api_{uuid4().hex}"


def _sse(event: AgentRunEvent) -> str:
    payload = {
        "type": event.type,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "timestamp": event.timestamp,
        "data": redact_for_langfuse(event.data),
    }
    return (
        f"event: {event.type}\n"
        f"id: {event.sequence}\n"
        f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )


async def _stream_sse(events: AsyncIterator[AgentRunEvent]) -> AsyncIterator[str]:
    async for event in events:
        yield _sse(event)


def _streaming_response(events: AsyncIterator[AgentRunEvent]) -> StreamingResponse:
    return StreamingResponse(
        _stream_sse(events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "X-Accel-Buffering": "no",
        },
    )


def _public_approval(approval: AgentApproval) -> dict[str, Any]:
    return {
        "approval_id": str(approval.id),
        "tool_name": str(approval.tool_name),
        "description": str(approval.description),
        "arguments": redact_for_langfuse(approval.arguments),
        "allowed_decisions": list(approval.allowed_decisions),
    }


def _public_snapshot(snapshot: AgentRunSnapshot) -> dict[str, Any]:
    return {
        "run_id": snapshot.run_id,
        "status": snapshot.status,
        "thread_id": snapshot.context.thread_id,
        "final_text": redact_for_langfuse(snapshot.final_text),
        "error": redact_for_langfuse(snapshot.error),
        "pending_approvals": [
            _public_approval(approval)
            for approval in snapshot.approvals
            if approval.status == "pending"
        ],
        "created_at": snapshot.created_at,
        "updated_at": snapshot.updated_at,
        "inference": (
            snapshot.inference_plan.model_dump(mode="json")
            if snapshot.inference_plan is not None
            else None
        ),
    }


async def _owned_snapshot(
    service: AgentRunService,
    *,
    run_id: str,
    principal: ResolvedV2Principal,
) -> AgentRunSnapshot:
    snapshot = await service.get_run(run_id)
    if snapshot is None or snapshot.context.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=404, detail="agent run not found")
    binding = principal.agent_binding
    if binding is not None and (
        snapshot.context.agent_spec != binding.agent_spec
        or snapshot.context.run_kind != binding.run_kind
        or snapshot.context.trust_class != binding.trust_class
    ):
        raise HTTPException(status_code=404, detail="agent run not found")
    return snapshot


def register_v2_agent_routes(
    app: FastAPI,
    *,
    get_agent_service: Callable[[], AgentRunService | None],
    resolve_principal: Callable[[Request], AgentPrincipal | Awaitable[AgentPrincipal]],
    resolve_agent_spec: Callable[[str, str], AgentSpecRef] | None = None,
    secret_ingress: Callable[..., str] | None = None,
) -> None:
    """Register the tenant-safe public API for starting and resuming agent runs."""

    def service_or_503() -> AgentRunService:
        service = get_agent_service()
        if service is None:
            raise HTTPException(status_code=503, detail="agent service unavailable")
        return service

    def run_binding(principal: ResolvedV2Principal) -> AgentRunBinding:
        if principal.agent_binding is not None:
            return principal.agent_binding
        if principal.trust_class != "owner":
            raise HTTPException(status_code=403, detail="agent binding is required")
        agent_spec = (
            resolve_agent_spec(principal.tenant_id, AgentRunKind.OWNER.value)
            if resolve_agent_spec is not None
            else AgentSpecRef(
                tenant_id=principal.tenant_id,
                spec_id="owner",
                revision=1,
            )
        )
        if agent_spec.tenant_id != principal.tenant_id:
            raise HTTPException(status_code=503, detail="invalid owner AgentSpec binding")
        return AgentRunBinding(
            agent_spec=agent_spec,
            run_kind=AgentRunKind.OWNER.value,
            trust_class="owner",
        )

    def build_run_request(
        *,
        principal: ResolvedV2Principal,
        request: Request,
        thread_id: str,
        text: str,
        file_ids: list[str],
        pinned_context: AgentRunContext | None = None,
    ) -> AgentRunRequest:
        binding = run_binding(principal)
        origin = OriginRef(
            interface=principal.interface,
            source_id=principal.source_id,
            conversation_id=principal.conversation_id,
            message_id=principal.message_id,
        )
        context = (
            pinned_context.model_copy(
                update={
                    "actor_id": principal.actor_id,
                    "correlation_id": _correlation_id(request),
                    "origin": origin,
                }
            )
            if pinned_context is not None
            else AgentRunContext(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                thread_id=thread_id,
                channel=principal.channel,
                run_kind=binding.run_kind,
                correlation_id=_correlation_id(request),
                origin=origin,
                agent_spec=binding.agent_spec,
                trust_class=binding.trust_class,
            )
        )
        return AgentRunRequest(
            context=context,
            text=text,
            file_ids=tuple(file_ids),
            idempotency_key=(
                str(request.headers.get("idempotency-key", "") or "").strip() or None
            ),
        )

    @app.post("/v2/agent/threads", status_code=201)
    async def create_agent_thread(
        body: AgentThreadCreateRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_SUBMIT.value)
        return await service_or_503().create_thread(
            tenant_id=principal.tenant_id,
            channel=principal.channel,
            title=body.title,
        )

    @app.put("/v2/agent/threads/{thread_id}")
    async def ensure_agent_thread(thread_id: str, request: Request) -> dict[str, str]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_SUBMIT.value)
        await service_or_503().ensure_thread(
            tenant_id=principal.tenant_id,
            thread_id=thread_id,
            channel=principal.channel,
        )
        return {"thread_id": thread_id}

    @app.get("/v2/agent/threads")
    async def list_agent_threads(
        request: Request,
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=100),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_REPLAY.value)
        return await service_or_503().list_threads(
            tenant_id=principal.tenant_id,
            cursor=cursor,
            limit=limit,
        )

    @app.get("/v2/agent/threads/{thread_id}/timeline")
    async def get_agent_thread_timeline(
        thread_id: str,
        request: Request,
        cursor: int = Query(default=0, ge=0),
        limit: int = Query(default=30, ge=1, le=100),
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_REPLAY.value)
        timeline = await service_or_503().thread_timeline(
            tenant_id=principal.tenant_id,
            thread_id=thread_id,
            cursor=cursor,
            limit=limit,
        )
        if timeline is None:
            raise HTTPException(status_code=404, detail="agent thread not found")
        return timeline

    @app.patch("/v2/agent/threads/{thread_id}")
    async def update_agent_thread(
        thread_id: str,
        body: AgentThreadUpdateRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_SUBMIT.value)
        try:
            updated = await service_or_503().update_thread(
                tenant_id=principal.tenant_id,
                thread_id=thread_id,
                title=body.title,
                archived=body.archived,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if updated is None:
            raise HTTPException(status_code=404, detail="agent thread not found")
        return updated

    @app.post(
        "/v2/agent/runs",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def start_agent_run(
        body: AgentRunCreateRequest,
        request: Request,
    ) -> StreamingResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_SUBMIT.value)
        run_request = build_run_request(
            principal=principal,
            request=request,
            thread_id=body.thread_id,
            text=body.text,
            file_ids=body.file_ids,
        )
        try:
            events = await service_or_503().open_stream(run_request)
        except AgentRunIdempotencyConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="idempotency key belongs to a different agent run request",
            ) from exc
        except AgentRunCheckpointConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="thread has an unresolved agent run",
            ) from exc
        return _streaming_response(events)

    @app.post(
        "/v2/agent/runs/{run_id}/steer",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def steer_agent_run(
        run_id: str,
        body: AgentRunSteerRequest,
        request: Request,
    ) -> StreamingResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_SUBMIT.value)
        service = service_or_503()
        snapshot = await _owned_snapshot(
            service,
            run_id=run_id,
            principal=principal,
        )
        if snapshot.status != "running":
            raise HTTPException(status_code=409, detail="agent run is not active")
        run_request = build_run_request(
            principal=principal,
            request=request,
            thread_id=snapshot.context.thread_id,
            text=body.text,
            file_ids=body.file_ids,
            pinned_context=snapshot.context,
        )
        await service.cancel(run_id)
        try:
            events = await service.open_stream(run_request)
        except AgentRunIdempotencyConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="idempotency key belongs to a different agent run request",
            ) from exc
        except AgentRunCheckpointConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="thread has an unresolved agent run",
            ) from exc
        return _streaming_response(events)

    @app.get("/v2/agent/runs/{run_id}")
    async def get_agent_run(run_id: str, request: Request) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_REPLAY.value)
        snapshot = await _owned_snapshot(
            service_or_503(),
            run_id=run_id,
            principal=principal,
        )
        return _public_snapshot(snapshot)

    @app.get(
        "/v2/agent/runs/{run_id}/events",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def replay_agent_run_events(
        run_id: str,
        request: Request,
        after_sequence: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_REPLAY.value)
        service = service_or_503()
        await _owned_snapshot(service, run_id=run_id, principal=principal)
        header_cursor = str(request.headers.get("last-event-id", "") or "").strip()
        if header_cursor:
            try:
                after_sequence = max(after_sequence, int(header_cursor))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid Last-Event-ID") from exc
        return _streaming_response(service.events(run_id, after_sequence=after_sequence))

    @app.post(
        "/v2/agent/runs/{run_id}/resume",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def resume_agent_run(
        run_id: str,
        body: AgentRunResumeRequest,
        request: Request,
    ) -> StreamingResponse:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, CapabilityAPIScope.AGENT_RUN_RESUME.value)
        service = service_or_503()
        snapshot = await _owned_snapshot(
            service,
            run_id=run_id,
            principal=principal,
        )
        if snapshot.status not in {"interrupted", "resume_pending"}:
            raise HTTPException(status_code=409, detail="agent run is not awaiting approval")
        approval = next(
            (
                item
                for item in snapshot.approvals
                if item.id == body.approval_id
                and (
                    item.status == "pending"
                    or (
                        snapshot.status == "resume_pending"
                        and item.status == body.decision
                        and item.edited_arguments == body.edited_arguments
                    )
                )
            ),
            None,
        )
        if approval is None:
            raise HTTPException(status_code=404, detail="pending approval not found")
        if body.decision not in approval.allowed_decisions:
            raise HTTPException(status_code=409, detail="approval decision is not allowed")
        decision = ApprovalDecision(
            approval_id=body.approval_id,
            decision=body.decision,
            edited_arguments=body.edited_arguments,
        )
        try:
            events = await service.open_resume(run_id, decision)
        except AgentRunCapabilityConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail="approved capability changed; cancel this run and retry",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _streaming_response(events)

    @app.post("/v2/agent/runs/{run_id}/cancel")
    async def cancel_agent_run(run_id: str, request: Request) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, "agent.runs.cancel")
        service = service_or_503()
        await _owned_snapshot(service, run_id=run_id, principal=principal)
        return _public_snapshot(await service.cancel(run_id))

    @app.post("/v2/agent/threads/{thread_id}/cancel")
    async def cancel_agent_thread(thread_id: str, request: Request) -> dict[str, Any]:
        principal = await resolve_v2_principal(request, resolve_principal)
        require_v2_scope(principal, "agent.runs.cancel")
        snapshot = await service_or_503().cancel_thread(
            tenant_id=principal.tenant_id,
            thread_id=thread_id,
        )
        if snapshot is None:
            raise HTTPException(status_code=404, detail="active agent run not found")
        return _public_snapshot(snapshot)


__all__ = [
    "AgentPrincipal",
    "AgentRunCreateRequest",
    "AgentRunResumeRequest",
    "AgentRunSteerRequest",
    "AgentRunService",
    "register_v2_agent_routes",
]
