"""Task orchestration route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_task_routes(
    app: FastAPI,
    *,
    get_tasks: Callable[[], Any],
) -> None:
    """Register task create/status/event/artifact/control endpoints."""

    @app.post("/internal/tasks/create")
    async def internal_task_create(request: Request) -> Any:
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        goal = str(body.get("goal", "")).strip()
        payload = body.get("payload") or {}
        risk_level = str(body.get("risk_level", "low")).strip() or "low"
        idempotency_key = body.get("idempotency_key")
        if not customer_id or not goal:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and goal are required"}
            )
        if isinstance(payload, dict) and "steps" in payload:
            steps = payload.get("steps")
            if not isinstance(steps, list):
                return JSONResponse(
                    status_code=400, content={"detail": "payload.steps must be a list"}
                )
            bad_idx = next((i for i, step in enumerate(steps) if not isinstance(step, dict)), None)
            if bad_idx is not None:
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": (
                            "payload.steps entries must be objects with a 'type' field "
                            "(e.g. {'type':'run_terminal', ...})."
                        ),
                        "bad_step_index": bad_idx,
                    },
                )
        task = await get_tasks().create_task(
            customer_id=customer_id,
            goal=goal,
            payload=payload,
            risk_level=risk_level,
            idempotency_key=idempotency_key,
        )
        return {"ok": True, "task": task}

    @app.get("/internal/tasks/{task_id}")
    async def internal_task_status(task_id: str) -> Any:
        try:
            task = get_tasks().get_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "task": task}

    @app.get("/internal/tasks/{task_id}/events")
    async def internal_task_events(task_id: str, limit: int = 50, offset: int = 0) -> Any:
        try:
            get_tasks().get_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        events = get_tasks().list_events(task_id, limit=limit, offset=offset)
        return {"ok": True, "events": events}

    @app.get("/internal/tasks/{task_id}/artifacts")
    async def internal_task_artifacts(task_id: str) -> Any:
        try:
            get_tasks().get_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "artifacts": get_tasks().list_task_artifacts(task_id)}

    @app.post("/internal/tasks/{task_id}/relaunch")
    async def internal_task_relaunch(task_id: str, request: Request) -> Any:
        body = await request.json()
        clarification = body.get("clarification")
        trigger_reason = (
            str(body.get("trigger_reason", "user_requested")).strip() or "user_requested"
        )
        try:
            task = await get_tasks().relaunch_task(
                task_id=task_id,
                trigger_reason=trigger_reason,
                clarification=clarification,
            )
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "task": task}

    @app.post("/internal/tasks/{task_id}/cancel")
    async def internal_task_cancel(task_id: str) -> Any:
        try:
            task = await get_tasks().cancel_task(task_id)
        except KeyError:
            return JSONResponse(status_code=404, content={"detail": "task not found"})
        return {"ok": True, "task": task}
