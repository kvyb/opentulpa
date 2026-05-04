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

import time
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

    def _record_user_context_event(event: str, **fields: Any) -> None:
        runtime = get_agent_runtime()
        if runtime is None:
            return
        recorder = getattr(runtime, "record_observability_event", None)
        if callable(recorder):
            recorder(event=event, **fields)
            return
        logger = getattr(runtime, "log_behavior_event", None)
        if callable(logger):
            logger(event=event, **fields)

    def _warning_count(payload: Any) -> int:
        if not isinstance(payload, dict):
            return 0
        warnings = payload.get("warnings")
        if isinstance(warnings, list):
            return len(warnings)
        indexed = payload.get("indexed")
        if isinstance(indexed, dict):
            return sum(
                len(source.get("warnings") or [])
                for source in indexed.get("sources", [])
                if isinstance(source, dict)
            )
        return 0

    def _source_count(payload: Any) -> int:
        if not isinstance(payload, dict):
            return 0
        sources = payload.get("sources")
        if isinstance(sources, list):
            return len(sources)
        indexed = payload.get("indexed")
        if isinstance(indexed, dict):
            index = indexed.get("index")
            if isinstance(index, dict):
                return int(index.get("source_count") or 0)
        return 0

    async def _prepare_user_context_files(customer_id: str, file_ids: list[Any]) -> dict[str, Any]:
        runtime = get_agent_runtime()
        if runtime is None or not hasattr(runtime, "summarize_uploaded_blob"):
            return {"prepared_count": 0, "failed_count": 0}
        vault = get_file_vault()
        prepared_count = 0
        failed_count = 0
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
            try:
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
            except Exception as exc:
                failed_count += 1
                _record_user_context_event(
                    "user_context.media_prepare_failed",
                    customer_id=customer_id,
                    file_id=file_id,
                    filename=filename,
                    mime_type=mime_type,
                    kind=kind,
                    error=f"{type(exc).__name__}: {exc}"[:500],
                )
                continue
            if str(analysis or "").strip():
                vault.set_ai_summary(customer_id, file_id, str(analysis).strip())
                prepared_count += 1
                _record_user_context_event(
                    "user_context.media_prepare_succeeded",
                    customer_id=customer_id,
                    file_id=file_id,
                    filename=filename,
                    mime_type=mime_type,
                    kind=kind,
                    analysis_chars=len(str(analysis or "")),
                )
            else:
                failed_count += 1
        return {"prepared_count": prepared_count, "failed_count": failed_count}

    @app.post("/internal/user_context/add_files")
    async def internal_user_context_add_files(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            started = time.monotonic()
            file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else []
            prep = await _prepare_user_context_files(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
            result = service.add_files(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
            _record_user_context_event(
                "user_context.add_files",
                customer_id=str(body.get("customer_id", "")).strip(),
                file_count=len(file_ids),
                source_count=_source_count(result),
                warning_count=_warning_count(result),
                prepared_count=int(prep.get("prepared_count") or 0),
                failed_count=int(prep.get("failed_count") or 0),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
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
            started = time.monotonic()
            result = service.query(
                customer_id=str(body.get("customer_id", "")).strip(),
                query=str(body.get("query", "")).strip(),
                max_extract_chars=int(body.get("max_extract_chars", 3000) or 3000),
            )
            _record_user_context_event(
                "user_context.query",
                customer_id=str(body.get("customer_id", "")).strip(),
                ok=bool(result.get("ok")) if isinstance(result, dict) else False,
                source_count=int(result.get("source_count") or 0) if isinstance(result, dict) else 0,
                section_count=int(result.get("section_count") or 0) if isinstance(result, dict) else 0,
                warning_count=_warning_count(result),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/reindex")
    async def internal_user_context_reindex(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            started = time.monotonic()
            file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else None
            prep = {"prepared_count": 0, "failed_count": 0}
            if file_ids:
                prep = await _prepare_user_context_files(
                    customer_id=str(body.get("customer_id", "")).strip(),
                    file_ids=file_ids,
                )
            result = service.reindex(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
            _record_user_context_event(
                "user_context.reindex",
                customer_id=str(body.get("customer_id", "")).strip(),
                file_count=len(file_ids or []),
                source_count=_source_count(result),
                warning_count=_warning_count(result),
                prepared_count=int(prep.get("prepared_count") or 0),
                failed_count=int(prep.get("failed_count") or 0),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/archive_sources")
    async def internal_user_context_archive_sources(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            started = time.monotonic()
            file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else []
            result = service.archive_sources(
                customer_id=str(body.get("customer_id", "")).strip(),
                file_ids=file_ids,
            )
            _record_user_context_event(
                "user_context.archive_sources",
                customer_id=str(body.get("customer_id", "")).strip(),
                file_count=len(file_ids),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/internal/user_context/promote_to_intake")
    async def internal_user_context_promote_to_intake(request: Request) -> Any:
        service = get_user_context_service()
        body = await request.json()
        try:
            started = time.monotonic()
            file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else []
            result = service.promote_to_intake(
                customer_id=str(body.get("customer_id", "")).strip(),
                workflow_id=str(body.get("workflow_id", "")).strip(),
                file_ids=file_ids,
            )
            _record_user_context_event(
                "user_context.promote_to_intake",
                customer_id=str(body.get("customer_id", "")).strip(),
                workflow_id=str(body.get("workflow_id", "")).strip(),
                file_count=len(file_ids),
                source_count=_source_count(result),
                warning_count=_warning_count(result),
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )
            return result
        except Exception as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
