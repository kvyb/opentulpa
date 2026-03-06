"""Scheduler route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.file_helpers import collect_routine_cleanup_paths, normalize_cleanup_paths
from opentulpa.core.ids import new_short_id
from opentulpa.scheduler.models import Routine


def register_scheduler_routes(
    app: FastAPI,
    *,
    get_scheduler: Callable[[], Any],
    delete_file: Callable[..., dict[str, Any]],
) -> None:
    """Register scheduler routine management endpoints."""

    @app.post("/internal/scheduler/routine")
    async def internal_scheduler_add_routine(request: Request) -> Any:
        sched = get_scheduler()
        body = await request.json()
        payload = body.get("payload") if isinstance(body.get("payload"), dict) else {}
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            return JSONResponse(
                status_code=400,
                content={"detail": "payload.instruction is required"},
            )

        rid = str(body.get("id", "")).strip() or new_short_id("rtn")
        routine = Routine(
            id=rid,
            name=body.get("name", "Unnamed"),
            schedule=body.get("schedule", "0 9 * * *"),
            payload=payload,
            enabled=body.get("enabled", True),
            is_cron=body.get("is_cron", True),
        )
        sched.add_routine(routine)
        return {"ok": True, "id": rid}

    @app.get("/internal/scheduler/routines")
    async def internal_scheduler_list_routines(customer_id: str | None = None) -> Any:
        sched = get_scheduler()
        routines = sched.list_routines()
        cid = str(customer_id or "").strip()
        if cid:
            routines = [
                r
                for r in routines
                if str((r.payload or {}).get("customer_id", "")).strip() == cid
            ]
        return {
            "routines": [
                {
                    "id": r.id,
                    "name": r.name,
                    "schedule": r.schedule,
                    "enabled": r.enabled,
                    "is_cron": r.is_cron,
                }
                for r in routines
            ]
        }

    @app.delete("/internal/scheduler/routine/{routine_id}")
    async def internal_scheduler_remove_routine(
        routine_id: str,
        customer_id: str | None = None,
    ) -> Any:
        sched = get_scheduler()
        cid = str(customer_id or "").strip()
        if cid:
            routine = sched.get_routine(routine_id)
            if routine is None:
                return {"ok": False}
            owner = str((routine.payload or {}).get("customer_id", "")).strip()
            if owner != cid:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "routine does not belong to this customer_id"},
                )
        ok = sched.remove_routine(routine_id)
        return {"ok": ok}

    @app.post("/internal/scheduler/routine/delete_with_assets")
    async def internal_scheduler_remove_routine_with_assets(request: Request) -> Any:
        sched = get_scheduler()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        if not customer_id:
            return JSONResponse(status_code=400, content={"detail": "customer_id is required"})

        routine_id = str(body.get("routine_id", "")).strip()
        name = str(body.get("name", "")).strip()
        remove_all_matches = bool(body.get("remove_all_matches", False))
        delete_files = bool(body.get("delete_files", True))
        extra_cleanup_paths = normalize_cleanup_paths(body.get("cleanup_paths"))

        routines = [
            r
            for r in sched.list_routines()
            if str((r.payload or {}).get("customer_id", "")).strip() == customer_id
        ]
        if routine_id:
            matched = [r for r in routines if r.id == routine_id]
        elif name:
            name_cf = name.strip().casefold()
            matched = [r for r in routines if r.name.strip().casefold() == name_cf]
        else:
            return JSONResponse(
                status_code=400,
                content={"detail": "routine_id or name is required"},
            )

        if not matched:
            return JSONResponse(status_code=404, content={"detail": "routine not found"})
        if len(matched) > 1 and not remove_all_matches:
            return JSONResponse(
                status_code=400,
                content={
                    "detail": "multiple routines matched; set remove_all_matches=true",
                    "matched_routines": [
                        {"id": r.id, "name": r.name, "schedule": r.schedule} for r in matched
                    ],
                },
            )

        deleted_routines: list[dict[str, Any]] = []
        failed_routines: list[dict[str, Any]] = []
        deleted_files: list[dict[str, Any]] = []
        failed_files: list[dict[str, Any]] = []

        for routine in matched:
            ok = sched.remove_routine(routine.id)
            if not ok:
                failed_routines.append({"id": routine.id, "name": routine.name, "error": "not found"})
                continue
            deleted_routines.append({"id": routine.id, "name": routine.name})
            if not delete_files:
                continue

            cleanup_paths = collect_routine_cleanup_paths(routine.payload or {})
            cleanup_paths.extend(extra_cleanup_paths)
            seen_paths: set[str] = set()
            unique_paths: list[str] = []
            for path in cleanup_paths:
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                unique_paths.append(path)

            for relative_path in unique_paths:
                try:
                    result = delete_file(relative_path, missing_ok=True)
                    deleted_files.append(
                        {
                            "path": str(result.get("path", relative_path)),
                            "deleted": bool(result.get("deleted", False)),
                            "missing": bool(result.get("missing", False)),
                        }
                    )
                except Exception as exc:
                    failed_files.append({"path": relative_path, "error": str(exc)})

        return {
            "ok": len(deleted_routines) > 0 and len(failed_routines) == 0,
            "deleted_routines": deleted_routines,
            "failed_routines": failed_routines,
            "deleted_files": deleted_files,
            "failed_files": failed_files,
        }
