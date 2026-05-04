"""Internal interactive user-context routes.

These endpoints expose the agent-facing user-context workflow for any uploaded
content type. Documents and spreadsheets can be indexed through local extraction;
PDFs, images, audio, and video are first prepared with the runtime multimodal
summarizer so visible text, speech, visual facts, layout details, and other
retrieval facts become text evidence. The routes then delegate to
``UserContextService`` to index, query, list, reindex, archive, or explicitly
promote selected sources into intake workflow knowledge.

No upload intent is inferred here. If a chat turn does not clearly say what to do
with uploaded files, the agent prompt policy is responsible for asking the user.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def register_user_context_routes(
    app: FastAPI,
    *,
    get_user_context_service: Callable[[], Any],
    get_file_vault: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
) -> None:
    """Register internal user-context endpoints."""

    async def _prepare_user_context_files(customer_id: str, file_ids: list[Any]) -> None:
        runtime = get_agent_runtime()
        if runtime is None or not hasattr(runtime, "summarize_uploaded_blob"):
            return
        vault = get_file_vault()
        for raw_file_id in file_ids[:50]:
            file_id = str(raw_file_id or "").strip()
            if not file_id:
                continue
            record = vault.get_file(customer_id, file_id)
            if not isinstance(record, dict) or not record:
                continue
            filename = str(record.get("original_filename", "") or "").strip()
            mime_type = str(record.get("mime_type", "") or "").strip().lower()
            kind = str(record.get("kind", "") or "").strip().lower()
            lower_name = filename.lower()
            needs_model_processing = (
                kind in {"photo", "video", "video_note", "audio", "voice"}
                or mime_type.startswith(("image/", "video/", "audio/"))
                or mime_type == "application/pdf"
                or lower_name.endswith(".pdf")
            )
            if not needs_model_processing:
                continue
            raw_bytes = vault.read_file_bytes(customer_id, file_id)
            if raw_bytes is None:
                continue
            analysis = await runtime.summarize_uploaded_blob(
                filename=filename or None,
                mime_type=mime_type or None,
                kind=kind or None,
                raw_bytes=raw_bytes,
                caption=str(record.get("caption", "") or "").strip() or None,
                question=(
                    "Prepare this file for durable user_context retrieval. Extract transcript-like "
                    "speech, visible text, visual facts, document layout facts, hooks, offers, claims, "
                    "style cues, and concrete details. Return concise source-grounding notes only."
                ),
            )
            if str(analysis or "").strip():
                vault.set_ai_summary(customer_id, file_id, str(analysis).strip())

    @app.post("/internal/user_context/add_files")
    async def internal_user_context_add_files(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else []
            await _prepare_user_context_files(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
            return service.add_files(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/list_sources")
    async def internal_user_context_list_sources(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            return service.list_sources(
                customer_id=str(body.get("customer_id", "")).strip(),
                include_archived=bool(body.get("include_archived", False)),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/find_sources")
    async def internal_user_context_find_sources(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            return service.find_sources(
                customer_id=str(body.get("customer_id", "")).strip(),
                query=str(body.get("query", "")).strip(),
                limit=int(body.get("limit", 10) or 10),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/query")
    async def internal_user_context_query(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            return service.query(
                customer_id=str(body.get("customer_id", "")).strip(),
                query=str(body.get("query", "")).strip(),
                max_extract_chars=int(body.get("max_extract_chars", 3000) or 3000),
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/reindex")
    async def internal_user_context_reindex(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else None
            if file_ids:
                await _prepare_user_context_files(
                    customer_id=str(body.get("customer_id", "")).strip(),
                    file_ids=file_ids,
                )
            return service.reindex(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/archive_sources")
    async def internal_user_context_archive_sources(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            return service.archive_sources(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=body.get("file_ids") if isinstance(body.get("file_ids"), list) else [],
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/promote_to_intake")
    async def internal_user_context_promote_to_intake(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            return service.promote_to_intake(
                customer_id=str(body.get("customer_id", "")).strip(),
                workflow_id=str(body.get("workflow_id", "")).strip(),
                file_ids=body.get("file_ids") if isinstance(body.get("file_ids"), list) else [],
            )
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
