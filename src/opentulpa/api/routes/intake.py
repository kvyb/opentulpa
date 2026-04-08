"""Intake workflow route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_intake_workflow_routes(
    app: FastAPI,
    *,
    get_intake_workflows: Callable[[], Any],
) -> None:
    """Register internal intake workflow endpoints."""

    @app.post("/internal/intake/workflows/upsert")
    async def internal_intake_workflows_upsert(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        try:
            workflow = service.upsert_workflow(
                customer_id=str(body.get("customer_id", "")).strip(),
                workflow_id=str(body.get("workflow_id", "")).strip() or None,
                name=str(body.get("name", "")).strip(),
                channel=str(body.get("channel", "instagram_dm")).strip() or "instagram_dm",
                provider=str(body.get("provider", "composio")).strip() or "composio",
                source_config=body.get("source_config") if isinstance(body.get("source_config"), dict) else None,
                intent_description=str(body.get("intent_description", "")).strip(),
                required_fields=body.get("required_fields") if isinstance(body.get("required_fields"), list) else [],
                field_guidance=body.get("field_guidance") if isinstance(body.get("field_guidance"), dict) else None,
                sink_type=str(body.get("sink_type", "")).strip(),
                sink_config=body.get("sink_config") if isinstance(body.get("sink_config"), dict) else None,
                schedule=str(body.get("schedule", "*/5 * * * *")).strip() or "*/5 * * * *",
                notify_user=bool(body.get("notify_user", True)),
                enabled=bool(body.get("enabled", True)),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "workflow": workflow}

    @app.post("/internal/intake/workflows/list")
    async def internal_intake_workflows_list(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        workflows = service.list_workflows(
            customer_id=str(body.get("customer_id", "")).strip(),
            include_disabled=bool(body.get("include_disabled", False)),
        )
        return {"ok": True, "workflows": workflows}

    @app.post("/internal/intake/workflows/get")
    async def internal_intake_workflows_get(request: Request) -> Any:
        service = get_intake_workflows()
        body = await request.json()
        workflow = service.get_workflow(
            customer_id=str(body.get("customer_id", "")).strip(),
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
            customer_id=str(body.get("customer_id", "")).strip(),
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
        customer_id = str(body.get("customer_id", "")).strip()
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
