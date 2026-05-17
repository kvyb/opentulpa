"""Internal workflow-scoped business knowledge routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.customer_ids import resolve_body_customer_id
from opentulpa.business_knowledge.service import query_result_payload


def register_knowledge_routes(
    app: FastAPI,
    *,
    get_knowledge_service: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register internal business knowledge endpoints."""

    @app.post("/internal/knowledge/index_sources")
    async def internal_knowledge_index_sources(request: Request) -> Any:
        service = get_knowledge_service()
        body = await request.json()
        try:
            return service.index_sources(
                customer_id=resolve_body_customer_id(body, resolve_customer_id),
                scope_type=str(body.get("scope_type", "")).strip(),
                scope_id=str(body.get("scope_id", "")).strip(),
                file_ids=body.get("file_ids") if isinstance(body.get("file_ids"), list) else [],
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/knowledge/query")
    async def internal_knowledge_query(request: Request) -> Any:
        service = get_knowledge_service()
        body = await request.json()
        try:
            result = service.query(
                customer_id=resolve_body_customer_id(body, resolve_customer_id),
                scope_type=str(body.get("scope_type", "")).strip(),
                scope_id=str(body.get("scope_id", "")).strip(),
                query=str(body.get("query", "")).strip(),
                max_extract_chars=int(body.get("max_extract_chars", 3000) or 3000),
                workflow_context=body.get("workflow_context")
                if isinstance(body.get("workflow_context"), dict)
                else None,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        return query_result_payload(result)

    @app.post("/internal/knowledge/preflight")
    async def internal_knowledge_preflight(request: Request) -> Any:
        service = get_knowledge_service()
        body = await request.json()
        try:
            return service.preflight_scope(
                customer_id=resolve_body_customer_id(body, resolve_customer_id),
                scope_type=str(body.get("scope_type", "")).strip(),
                scope_id=str(body.get("scope_id", "")).strip(),
                workflow_goal=str(body.get("workflow_goal", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/knowledge/promote_scope")
    async def internal_knowledge_promote_scope(request: Request) -> Any:
        service = get_knowledge_service()
        body = await request.json()
        try:
            return service.promote_scope(
                customer_id=resolve_body_customer_id(body, resolve_customer_id),
                source_scope_type=str(body.get("source_scope_type", "")).strip(),
                source_scope_id=str(body.get("source_scope_id", "")).strip(),
                target_scope_type=str(body.get("target_scope_type", "")).strip(),
                target_scope_id=str(body.get("target_scope_id", "")).strip(),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
