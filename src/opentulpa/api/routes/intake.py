"""Intake workflow route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_body_customer_id
from opentulpa.api.routes.intake_use_cases import (
    workflow_upsert_kwargs,
    workflow_with_knowledge_files,
)
from opentulpa.api.web_auth import authorized_web_request


def register_intake_workflow_routes(
    app: FastAPI,
    *,
    get_intake_workflows: Callable[[], Any],
    get_workflow_setup_service: Callable[[], Any],
    get_file_vault: Callable[[], Any] | None = None,
    resolve_customer_id: Callable[[str], str] | None = None,
    web_token: str | None = None,
) -> None:
    """Register internal intake workflow endpoints."""

    def _authorized_web_request(request: Request) -> bool:
        return authorized_web_request(request, web_token)

    def _web_customer_id(request: Request) -> str:
        raw = str(request.query_params.get("customer_id", "") or "").strip()
        return resolve_body_customer_id({"customer_id": raw}, resolve_customer_id)

    def _file_vault_or_none() -> Any | None:
        return get_file_vault() if get_file_vault is not None else None

    def _workflow_response(workflow: dict[str, Any]) -> dict[str, Any]:
        return workflow_with_knowledge_files(workflow, file_vault=_file_vault_or_none())

    @app.post("/internal/intake/workflows/upsert")
    async def internal_intake_workflows_upsert(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            workflow = service.upsert_workflow(
                **workflow_upsert_kwargs(body, customer_id=customer_id)
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "workflow": workflow}

    @app.post("/internal/intake/workflows/list")
    async def internal_intake_workflows_list(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        workflows = service.list_workflows(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            include_disabled=bool(body.get("include_disabled", False)),
        )
        return {"ok": True, "workflows": workflows}

    @app.post("/internal/intake/workflows/get")
    async def internal_intake_workflows_get(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        workflow = service.get_workflow(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            workflow_id=str(body.get("workflow_id", "")).strip(),
        )
        if workflow is None:
            return JSONResponse(status_code=404, content={"detail": "workflow not found"})
        return {"ok": True, "workflow": workflow}

    @app.post("/internal/intake/workflows/delete")
    async def internal_intake_workflows_delete(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        result = service.delete_workflow(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            workflow_id=str(body.get("workflow_id", "")).strip(),
        )
        if not bool(result.get("deleted", False)):
            return JSONResponse(status_code=404, content={"detail": "workflow not found"})
        return result

    @app.post("/internal/intake/workflows/run")
    async def internal_intake_workflows_run(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        workflow_id = str(body.get("workflow_id", "")).strip()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        if not workflow_id or not customer_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and workflow_id are required"},
            )
        result = await service.run_workflow(
            customer_id=customer_id,
            workflow_id=workflow_id,
            event_type=str(body.get("event_type", "manual")).strip() or "manual",
            force=bool(body.get("force", False)),
        )
        status_code = 200 if bool(result.get("ok", False)) else 400
        return JSONResponse(status_code=status_code, content=result)

    @app.get("/web/intake/workflows")
    async def web_intake_workflows_list(request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        customer_id = _web_customer_id(request)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        workflows = get_intake_workflows().list_workflows(
            customer_id=customer_id,
            include_disabled=True,
        )
        return {
            "ok": True,
            "workflows": [_workflow_response(workflow) for workflow in workflows],
        }

    @app.get("/web/intake/workflows/{workflow_id}")
    async def web_intake_workflows_get(workflow_id: str, request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        customer_id = _web_customer_id(request)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        workflow = get_intake_workflows().get_workflow(
            customer_id=customer_id,
            workflow_id=str(workflow_id or "").strip(),
        )
        if workflow is None:
            return JSONResponse(status_code=404, content={"detail": "workflow not found"})
        return {"ok": True, "workflow": _workflow_response(workflow)}

    @app.put("/web/intake/workflows/{workflow_id}")
    async def web_intake_workflows_put(workflow_id: str, request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"detail": "workflow payload must be an object"})
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        payload = dict(body)
        payload["workflow_id"] = str(workflow_id or "").strip()
        try:
            workflow = get_intake_workflows().upsert_workflow(
                **workflow_upsert_kwargs(
                    payload,
                    customer_id=customer_id,
                    workflow_id=payload["workflow_id"],
                    default_schedule_on_falsey=True,
                )
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "workflow": _workflow_response(workflow)}

    @app.delete("/web/intake/workflows/{workflow_id}")
    async def web_intake_workflows_delete(workflow_id: str, request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        customer_id = _web_customer_id(request)
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})
        result = get_intake_workflows().delete_workflow(
            customer_id=customer_id,
            workflow_id=str(workflow_id or "").strip(),
        )
        if not bool(result.get("deleted", False)):
            return JSONResponse(status_code=404, content={"detail": "workflow not found"})
        return result

    @app.post("/internal/intake/setup/begin")
    async def internal_intake_setup_begin(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.begin_session(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
                mode=str(body.get("mode", "")).strip(),
                workflow_id=str(body.get("workflow_id", "")).strip() or None,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/get")
    async def internal_intake_setup_get(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        session = service.get_thread_session(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            thread_id=str(body.get("thread_id", "")).strip(),
            include_paused=bool(body.get("include_paused", True)),
        )
        if session is None:
            return JSONResponse(status_code=404, content={"detail": "workflow setup session not found"})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/update")
    async def internal_intake_setup_update(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.update_session(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
                draft_patch=body.get("draft_patch") if isinstance(body.get("draft_patch"), dict) else None,
                scratchpad_patch=body.get("scratchpad_patch") if isinstance(body.get("scratchpad_patch"), dict) else None,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/mark_proposed")
    async def internal_intake_setup_mark_proposed(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.mark_proposed(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/preflight")
    async def internal_intake_setup_preflight(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.preflight_current(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session, "preflight": session.get("preflight", {})}

    @app.post("/internal/intake/setup/propose_current")
    async def internal_intake_setup_propose_current(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.propose_current(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session, "preflight": session.get("preflight", {})}

    @app.post("/internal/intake/setup/confirm_current")
    async def internal_intake_setup_confirm_current(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.confirm_current(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/commit")
    async def internal_intake_setup_commit(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.commit(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/finalize_confirmation")
    async def internal_intake_setup_finalize_confirmation(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.finalize_confirmation(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
                draft_patch=body.get("draft_patch") if isinstance(body.get("draft_patch"), dict) else None,
                scratchpad_patch=body.get("scratchpad_patch") if isinstance(body.get("scratchpad_patch"), dict) else None,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session, "preflight": session.get("preflight", {})}

    @app.post("/internal/intake/setup/pause")
    async def internal_intake_setup_pause(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.pause(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}

    @app.post("/internal/intake/setup/cancel")
    async def internal_intake_setup_cancel(request: Request) -> Any:
        service = get_workflow_setup_service()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            session = service.cancel(
                customer_id=customer_id,
                thread_id=str(body.get("thread_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "session": session}
