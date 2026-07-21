"""V2 tenant-scoped intake workflow and revisioned draft routes."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from opentulpa.intake.drafts import (
    IntakeDraftActivationError,
    IntakeDraftConfirmationError,
    IntakeDraftConflictError,
    IntakeDraftNotFoundError,
    IntakeDraftService,
    IntakeDraftValidationError,
)
from opentulpa.persistence.idempotency import IdempotencyConflictError

logger = logging.getLogger(__name__)
_PUBLIC_WORKFLOW_TEST_FAILURE = "Intake workflow test failed. Check server logs."


class IntakePrincipal(Protocol):
    tenant_id: str
    actor_id: str


class _RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IntakeDraftSaveRequest(_RequestModel):
    id: str | None = Field(default=None, min_length=1, max_length=100)
    workflow_id: str | None = Field(default=None, min_length=1, max_length=100)
    expected_revision: int | None = Field(default=None, ge=1)
    payload: dict[str, JsonValue] = Field(min_length=1)


class IntakeDraftPatchRequest(_RequestModel):
    expected_revision: int = Field(ge=1)
    patch: dict[str, JsonValue] = Field(min_length=1)


class IntakeDraftPrepareRequest(_RequestModel):
    expected_revision: int = Field(ge=1)


class IntakeDraftActivateRequest(_RequestModel):
    expected_revision: int = Field(ge=1)
    confirmation_token: str = Field(min_length=32, max_length=500)


class IntakeWorkflowTestRequest(_RequestModel):
    force: bool = True


class IntakeSinkReconcileRequest(_RequestModel):
    effect_revision: int = Field(ge=1)
    decision: Literal["confirm_applied", "retry_no_effect", "reject"]
    reason: str = Field(min_length=1, max_length=2_000)
    provider_result: dict[str, JsonValue] = Field(default_factory=dict)


async def _resolve_principal(
    request: Request,
    resolver: Callable[[Request], IntakePrincipal | Awaitable[IntakePrincipal]],
) -> IntakePrincipal:
    resolved = resolver(request)
    principal = await resolved if inspect.isawaitable(resolved) else resolved
    if not str(getattr(principal, "tenant_id", "") or "").strip():
        raise HTTPException(status_code=401, detail="authenticated tenant is required")
    if not str(getattr(principal, "actor_id", "") or "").strip():
        raise HTTPException(status_code=401, detail="authenticated actor is required")
    return principal


def _draft_error(exc: Exception) -> HTTPException:
    logger.error(
        "intake draft operation failed",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    if isinstance(exc, IntakeDraftNotFoundError):
        return HTTPException(status_code=404, detail="intake draft not found")
    if isinstance(exc, (IntakeDraftConflictError, IntakeDraftConfirmationError)):
        return HTTPException(status_code=409, detail="intake draft conflict")
    if isinstance(exc, IntakeDraftValidationError):
        return HTTPException(status_code=422, detail="intake draft is invalid")
    if isinstance(exc, IntakeDraftActivationError):
        return HTTPException(status_code=502, detail="intake draft activation failed")
    return HTTPException(status_code=500, detail="intake draft operation failed")


def _public_workflow_test_result(result: dict[str, Any]) -> dict[str, Any]:
    if bool(result.get("ok")):
        return {**result, "errors": [], "source_warnings": []}
    return {
        "ok": False,
        "workflow_id": str(result.get("workflow_id") or ""),
        "event_type": str(result.get("event_type") or "test"),
        "processed_conversations": int(result.get("processed_conversations") or 0),
        "matched_conversations": int(result.get("matched_conversations") or 0),
        "results": [],
        "errors": [_PUBLIC_WORKFLOW_TEST_FAILURE],
        "source_warnings": [],
        "summary": _PUBLIC_WORKFLOW_TEST_FAILURE,
    }


def register_v2_intake_routes(
    app: FastAPI,
    *,
    get_draft_service: Callable[[], IntakeDraftService],
    get_intake_workflows: Callable[[], Any],
    resolve_principal: Callable[
        [Request],
        IntakePrincipal | Awaitable[IntakePrincipal],
    ],
    on_workflow_changed: Callable[[dict[str, Any]], None] | None = None,
    on_workflow_deleted: Callable[[str, str], None] | None = None,
) -> None:
    """Register workflow reads/actions and draft-only workflow mutations."""

    @app.get("/v2/intake/workflows")
    async def list_workflows(request: Request) -> dict[str, list[dict[str, Any]]]:
        principal = await _resolve_principal(request, resolve_principal)
        workflows = get_intake_workflows().list_workflows(
            customer_id=principal.tenant_id,
            include_disabled=True,
        )
        return {"workflows": workflows}

    @app.get("/v2/intake/workflows/{workflow_id}")
    async def get_workflow(workflow_id: str, request: Request) -> dict[str, dict[str, Any]]:
        principal = await _resolve_principal(request, resolve_principal)
        workflow = get_intake_workflows().get_workflow(
            customer_id=principal.tenant_id,
            workflow_id=workflow_id,
        )
        if workflow is None:
            raise HTTPException(status_code=404, detail="intake workflow not found")
        return {"workflow": workflow}

    @app.delete("/v2/intake/workflows/{workflow_id}")
    async def delete_workflow(
        workflow_id: str,
        request: Request,
        expected_revision: int = Query(ge=1),
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            result = get_intake_workflows().delete_workflow(
                customer_id=principal.tenant_id,
                workflow_id=workflow_id,
                expected_revision=expected_revision,
            )
        except ValueError as exc:
            logger.error(
                "intake workflow delete conflict",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise HTTPException(
                status_code=409,
                detail="intake workflow revision conflict",
            ) from exc
        if not bool(result.get("deleted", False)):
            raise HTTPException(status_code=404, detail="intake workflow not found")
        if on_workflow_deleted is not None:
            on_workflow_deleted(principal.tenant_id, workflow_id)
        return {"deleted": True, "workflow_id": workflow_id}

    @app.post("/v2/intake/workflows/{workflow_id}/test")
    async def test_workflow(
        workflow_id: str,
        body: IntakeWorkflowTestRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        service = get_intake_workflows()
        if (
            service.get_workflow(
                customer_id=principal.tenant_id,
                workflow_id=workflow_id,
            )
            is None
        ):
            raise HTTPException(status_code=404, detail="intake workflow not found")
        try:
            result = service.run_workflow(
                customer_id=principal.tenant_id,
                workflow_id=workflow_id,
                event_type="test",
                force=body.force,
            )
            resolved = await result if inspect.isawaitable(result) else result
        except Exception as exc:
            logger.error(
                "intake workflow test raised",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            raise HTTPException(status_code=502, detail=_PUBLIC_WORKFLOW_TEST_FAILURE) from exc
        if not isinstance(resolved, dict):
            logger.error("intake workflow test returned a non-object result")
            raise HTTPException(status_code=502, detail=_PUBLIC_WORKFLOW_TEST_FAILURE)
        if not bool(resolved.get("ok")):
            logger.error("intake workflow test failed: %r", resolved)
        return {"result": _public_workflow_test_result(resolved)}

    @app.post("/v2/intake/workflows/{workflow_id}/bookings/{booking_id}/sink/reconcile")
    async def reconcile_sink_effect(
        workflow_id: str,
        booking_id: str,
        body: IntakeSinkReconcileRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            result = get_intake_workflows().reconcile_sink_effect(
                customer_id=principal.tenant_id,
                actor_id=principal.actor_id,
                workflow_id=workflow_id,
                booking_id=booking_id,
                effect_revision=body.effect_revision,
                decision=body.decision,
                reason=body.reason,
                provider_result=dict(body.provider_result),
            )
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="intake sink effect not found") from exc
        except (IdempotencyConflictError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="intake sink effect conflict") from exc
        return {"reconciliation": result}

    @app.get("/v2/intake/drafts")
    async def list_drafts(
        request: Request,
        workflow_id: str | None = Query(default=None, min_length=1, max_length=100),
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        return {
            "drafts": get_draft_service().list(
                tenant_id=principal.tenant_id,
                workflow_id=workflow_id,
            )
        }

    @app.post("/v2/intake/drafts")
    async def save_draft(
        body: IntakeDraftSaveRequest,
        request: Request,
        response: Response,
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            draft = get_draft_service().save(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                draft_id=body.id,
                workflow_id=body.workflow_id,
                expected_revision=body.expected_revision,
                patch=body.payload,
            )
        except Exception as exc:
            raise _draft_error(exc) from exc
        if draft.revision == 1:
            response.status_code = 201
        return {"draft": draft}

    @app.get("/v2/intake/drafts/{draft_id}")
    async def get_draft(draft_id: str, request: Request) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        draft = get_draft_service().get(
            tenant_id=principal.tenant_id,
            draft_id=draft_id,
        )
        if draft is None:
            raise HTTPException(status_code=404, detail="intake draft not found")
        return {"draft": draft}

    @app.patch("/v2/intake/drafts/{draft_id}")
    async def patch_draft(
        draft_id: str,
        body: IntakeDraftPatchRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        service = get_draft_service()
        existing = service.get(tenant_id=principal.tenant_id, draft_id=draft_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="intake draft not found")
        try:
            draft = service.save(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                draft_id=draft_id,
                workflow_id=existing.workflow_id,
                expected_revision=body.expected_revision,
                patch=body.patch,
            )
        except Exception as exc:
            raise _draft_error(exc) from exc
        return {"draft": draft}

    @app.post("/v2/intake/drafts/{draft_id}/prepare")
    async def prepare_draft(
        draft_id: str,
        body: IntakeDraftPrepareRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            prepared = get_draft_service().prepare(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                draft_id=draft_id,
                expected_revision=body.expected_revision,
            )
        except Exception as exc:
            raise _draft_error(exc) from exc
        return {"prepared": prepared}

    @app.post("/v2/intake/drafts/{draft_id}/activate")
    async def activate_draft(
        draft_id: str,
        body: IntakeDraftActivateRequest,
        request: Request,
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            activated = await get_draft_service().activate(
                tenant_id=principal.tenant_id,
                actor_id=principal.actor_id,
                draft_id=draft_id,
                expected_revision=body.expected_revision,
                confirmation_token=body.confirmation_token,
            )
        except Exception as exc:
            raise _draft_error(exc) from exc
        if on_workflow_changed is not None:
            workflow_id = str(activated.draft.workflow_id)
            workflow = get_intake_workflows().get_workflow(
                customer_id=principal.tenant_id,
                workflow_id=workflow_id,
            )
            if isinstance(workflow, dict):
                on_workflow_changed(workflow)
        return {"activated": activated}

    @app.delete("/v2/intake/drafts/{draft_id}")
    async def delete_draft(
        draft_id: str,
        request: Request,
        expected_revision: int = Query(ge=1),
    ) -> dict[str, Any]:
        principal = await _resolve_principal(request, resolve_principal)
        try:
            get_draft_service().delete(
                tenant_id=principal.tenant_id,
                draft_id=draft_id,
                expected_revision=expected_revision,
            )
        except Exception as exc:
            raise _draft_error(exc) from exc
        return {"deleted": True, "draft_id": draft_id}


__all__ = [
    "IntakeDraftActivateRequest",
    "IntakeDraftPatchRequest",
    "IntakeDraftPrepareRequest",
    "IntakeDraftSaveRequest",
    "IntakePrincipal",
    "IntakeSinkReconcileRequest",
    "IntakeWorkflowTestRequest",
    "register_v2_intake_routes",
]
