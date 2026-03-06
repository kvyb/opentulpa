"""Skill store route registration."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_skill_routes(
    app: FastAPI,
    *,
    get_skill_store: Callable[[], Any],
    get_memory: Callable[[], Any],
) -> None:
    """Register internal skill list/get/upsert/delete endpoints."""

    @app.post("/internal/skills/list")
    async def internal_skills_list(request: Request) -> Any:
        store = get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        include_global = bool(body.get("include_global", True))
        include_disabled = bool(body.get("include_disabled", False))
        limit = int(body.get("limit", 200))
        skills = store.list_skills(
            customer_id=customer_id,
            include_global=include_global,
            include_disabled=include_disabled,
            limit=limit,
        )
        return {"ok": True, "skills": skills}

    @app.post("/internal/skills/get")
    async def internal_skills_get(request: Request) -> Any:
        store = get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        name = str(body.get("name", "")).strip()
        include_files = bool(body.get("include_files", True))
        include_global = bool(body.get("include_global", True))
        if not name:
            return JSONResponse(status_code=400, content={"detail": "name is required"})
        skill = store.get_skill(
            customer_id=customer_id,
            name=name,
            include_files=include_files,
            include_global=include_global,
        )
        if skill is None:
            return JSONResponse(status_code=404, content={"detail": "skill not found"})
        return {"ok": True, "skill": skill}

    @app.post("/internal/skills/upsert")
    async def internal_skills_upsert(request: Request) -> Any:
        store = get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        scope = str(body.get("scope", "user")).strip().lower()
        name = str(body.get("name", "")).strip()
        description = str(body.get("description", "")).strip()
        instructions = str(body.get("instructions", "")).strip()
        skill_markdown = str(body.get("skill_markdown", "")).strip()
        source = str(body.get("source", "agent") or "agent")
        supporting_files_raw = body.get("supporting_files")
        supporting_files = (
            supporting_files_raw if isinstance(supporting_files_raw, dict) else None
        )
        if scope == "user" and not customer_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id is required for user skills"}
            )
        if not name:
            return JSONResponse(status_code=400, content={"detail": "name is required"})
        try:
            if not skill_markdown:
                from opentulpa.skills.service import build_skill_markdown

                skill_markdown = build_skill_markdown(
                    name=name,
                    description=description,
                    instructions=instructions,
                )
            skill = store.upsert_skill(
                scope=scope,
                customer_id=customer_id,
                name=name,
                skill_markdown=skill_markdown,
                source=source,
                enabled=True,
                supporting_files=supporting_files,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

        memory = get_memory()
        if memory is not None:
            with suppress(Exception):
                memory.add_text(
                    (
                        "Skill stored for this user: "
                        f"name={skill.get('name')} scope={skill.get('scope')} "
                        f"description={skill.get('description')}"
                    ),
                    user_id=customer_id or "global",
                    metadata={
                        "kind": "user_skill",
                        "skill_name": skill.get("name"),
                        "scope": skill.get("scope"),
                    },
                )
        return {"ok": True, "skill": skill}

    @app.post("/internal/skills/delete")
    async def internal_skills_delete(request: Request) -> Any:
        store = get_skill_store()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        scope = str(body.get("scope", "user")).strip().lower()
        name = str(body.get("name", "")).strip()
        if not name:
            return JSONResponse(status_code=400, content={"detail": "name is required"})
        if scope == "user" and not customer_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id is required for user skills"}
            )
        try:
            deleted = store.delete_skill(scope=scope, customer_id=customer_id, name=name)
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return {"ok": True, "deleted": bool(deleted)}
