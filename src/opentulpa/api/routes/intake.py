"""Intake workflow route registration."""

from __future__ import annotations

import hmac
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_body_customer_id


def register_intake_workflow_routes(
    app: FastAPI,
    *,
    get_intake_workflows: Callable[[], Any],
    get_workflow_setup_service: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
    web_token: str | None = None,
) -> None:
    """Register internal intake workflow endpoints."""

    def _authorized_web_request(request: Request) -> bool:
        expected = str(web_token or "").strip()
        if not expected:
            return False
        header = str(request.headers.get("authorization", "") or "").strip()
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), expected)

    @app.post("/internal/intake/workflows/upsert")
    async def internal_intake_workflows_upsert(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        customer_id = resolve_body_customer_id(body, resolve_customer_id)
        try:
            workflow = service.upsert_workflow(
                customer_id=customer_id,
                workflow_id=str(body.get("workflow_id", "")).strip() or None,
                name=str(body.get("name", "")).strip(),
                channel=str(body.get("channel", "instagram_dm")).strip() or "instagram_dm",
                provider=str(body.get("provider", "composio")).strip() or "composio",
                source_config=body.get("source_config") if isinstance(body.get("source_config"), dict) else None,
                intent_description=str(body.get("intent_description", "")).strip(),
                required_fields=body.get("required_fields") if isinstance(body.get("required_fields"), list) else [],
                field_guidance=body.get("field_guidance") if isinstance(body.get("field_guidance"), dict) else None,
                assistant_instructions=str(body.get("assistant_instructions", "")).strip(),
                business_facts=body.get("business_facts") if isinstance(body.get("business_facts"), dict) else None,
                knowledge_file_ids=body.get("knowledge_file_ids") if isinstance(body.get("knowledge_file_ids"), list) else None,
                sink_type=str(body.get("sink_type", "")).strip(),
                sink_config=body.get("sink_config") if isinstance(body.get("sink_config"), dict) else None,
                schedule=str(body.get("schedule", "*/2 * * * *")).strip() or "*/2 * * * *",
                notify_user=bool(body.get("notify_user", True)),
                enabled=bool(body.get("enabled", True)),
                reply_mode=str(body.get("reply_mode", "auto")).strip() or "auto",
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

    @app.post("/internal/intake/drafts/list")
    async def internal_intake_drafts_list(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        drafts = service.list_drafts(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            workflow_id=str(body.get("workflow_id", "")).strip() or None,
            status=str(body.get("status", "pending")).strip().lower(),
            limit=int(body.get("limit", 50) or 50),
        )
        return {"ok": True, "drafts": drafts}

    @app.post("/internal/intake/drafts/get")
    async def internal_intake_drafts_get(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        draft = service.get_draft(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            draft_id=str(body.get("draft_id", "")).strip(),
        )
        if draft is None:
            return JSONResponse(status_code=404, content={"detail": "draft not found"})
        return {"ok": True, "draft": draft}

    @app.post("/internal/intake/drafts/edit")
    async def internal_intake_drafts_edit(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        try:
            draft = service.edit_draft(
                customer_id=resolve_body_customer_id(body, resolve_customer_id),
                draft_id=str(body.get("draft_id", "")).strip(),
                reply_text=str(body.get("reply_text", "") or ""),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        if draft is None:
            return JSONResponse(status_code=404, content={"detail": "draft not found"})
        return {"ok": True, "draft": draft}

    @app.post("/internal/intake/drafts/discard")
    async def internal_intake_drafts_discard(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        try:
            draft = service.discard_draft(
                customer_id=resolve_body_customer_id(body, resolve_customer_id),
                draft_id=str(body.get("draft_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        if draft is None:
            return JSONResponse(status_code=404, content={"detail": "draft not found"})
        return {"ok": True, "draft": draft}

    @app.post("/internal/intake/drafts/approve")
    async def internal_intake_drafts_approve(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        draft, error = await service.approve_draft(
            customer_id=resolve_body_customer_id(body, resolve_customer_id),
            draft_id=str(body.get("draft_id", "")).strip(),
        )
        if draft is None:
            return JSONResponse(status_code=404, content={"detail": "draft not found"})
        if error is not None:
            return JSONResponse(status_code=400, content={"detail": error, "draft": draft})
        return {"ok": True, "draft": draft}

    @app.post("/web/intake/drafts/list")
    async def web_intake_drafts_list(request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await internal_intake_drafts_list(request)

    @app.post("/web/intake/drafts/edit")
    async def web_intake_drafts_edit(request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await internal_intake_drafts_edit(request)

    @app.post("/web/intake/drafts/discard")
    async def web_intake_drafts_discard(request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await internal_intake_drafts_discard(request)

    @app.post("/web/intake/drafts/approve")
    async def web_intake_drafts_approve(request: Request) -> Any:
        if not _authorized_web_request(request):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        return await internal_intake_drafts_approve(request)

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
