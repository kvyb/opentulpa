"""Authenticated generic chat routes for non-Telegram clients."""

from __future__ import annotations

import asyncio
import json
import mimetypes
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from hmac import compare_digest
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from opentulpa.agent.runtime import STREAM_PROGRESS_PREFIX, STREAM_WAIT_SIGNAL
from opentulpa.api.file_helpers import sanitize_uploaded_file_record
from opentulpa.interfaces.telegram.attachments import (
    _skip_auto_summary_for_upload,
    build_uploaded_files_context,
)
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR, is_within

MAX_WEB_UPLOAD_BYTES = 45_000_000


def _bearer_token(request: Request) -> str:
    header = str(request.headers.get("authorization") or "").strip()
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer":
        return ""
    return token.strip()


def _authorized(request: Request, expected_secret: str) -> bool:
    token = _bearer_token(request)
    return bool(token and compare_digest(token, expected_secret))


def _sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _progress_message(chunk: str) -> str:
    if chunk == STREAM_WAIT_SIGNAL:
        return "Working..."
    if chunk.startswith(STREAM_PROGRESS_PREFIX):
        return chunk.removeprefix(STREAM_PROGRESS_PREFIX).strip() or "Working..."
    return ""


def register_generic_chat_routes(
    app: FastAPI,
    *,
    generic_api_secret: str | None,
    get_agent_runtime: Callable[[], Any],
    get_file_vault: Callable[[], Any],
    get_workflow_setup_service: Callable[[], Any],
) -> None:
    """Register authenticated generic chat endpoints."""

    @app.post("/web/chat/turns")
    async def web_chat_turn(request: Request) -> Any:
        secret = str(generic_api_secret or "").strip()
        if not secret:
            return JSONResponse(
                status_code=503,
                content={"detail": "OPENTULPA_GENERIC_API_SECRET is not configured"},
            )
        if not _authorized(request, secret):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})

        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        thread_id = str(body.get("thread_id", "")).strip()
        text = str(body.get("text", "")).strip()
        raw_file_ids = body.get("file_ids") if isinstance(body.get("file_ids"), list) else []
        file_ids = [str(item).strip() for item in raw_file_ids if str(item).strip()][:20]
        include_pending_context = bool(body.get("include_pending_context", True))
        if not customer_id or not thread_id or not text:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id, thread_id, and text are required"},
            )
        runtime = get_agent_runtime()
        if runtime is None or not (hasattr(runtime, "ainvoke_text") or hasattr(runtime, "astream_text")):
            return JSONResponse(status_code=503, content={"detail": "agent runtime unavailable"})

        return StreamingResponse(
            _stream_turn(
                runtime=runtime,
                workflow_setup_service=get_workflow_setup_service(),
                file_vault=get_file_vault(),
                customer_id=customer_id,
                thread_id=thread_id,
                text=text,
                file_ids=file_ids,
                include_pending_context=include_pending_context,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/web/files/upload")
    async def web_file_upload(request: Request) -> Any:
        secret = str(generic_api_secret or "").strip()
        if not secret:
            return JSONResponse(
                status_code=503,
                content={"detail": "OPENTULPA_GENERIC_API_SECRET is not configured"},
            )
        if not _authorized(request, secret):
            return JSONResponse(status_code=401, content={"detail": "unauthorized"})
        form = await request.form()
        customer_id = str(form.get("customer_id") or "").strip()
        thread_id = str(form.get("thread_id") or "").strip()
        kind = str(form.get("kind") or "document").strip() or "document"
        caption = str(form.get("caption") or "").strip() or None
        upload = form.get("file")
        if not customer_id or not thread_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and thread_id are required"},
            )
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse(status_code=400, content={"detail": "file is required"})
        raw_bytes = await upload.read()
        if not isinstance(raw_bytes, bytes) or not raw_bytes:
            return JSONResponse(status_code=400, content={"detail": "file is empty"})
        if len(raw_bytes) > MAX_WEB_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"detail": "file is too large"})
        filename = str(getattr(upload, "filename", "") or f"{kind}.bin").strip()
        content_type = str(getattr(upload, "content_type", "") or "").strip() or None
        vault = get_file_vault()
        record = vault.ingest_file(
            customer_id=customer_id,
            chat_id=None,
            kind=kind,
            telegram_file_id=None,
            original_filename=filename,
            mime_type=content_type,
            caption=caption,
            raw_bytes=raw_bytes,
        )
        runtime = get_agent_runtime()
        record = await _postprocess_uploaded_file(
            runtime=runtime,
            vault=vault,
            customer_id=customer_id,
            record=record,
            raw_bytes=raw_bytes,
            caption=caption,
        )
        return {
            "ok": True,
            "file": _web_file_metadata(record),
        }

    @app.get("/web/files/{file_id}/metadata")
    async def web_file_metadata(file_id: str, request: Request) -> Any:
        auth = _authorized_web_request(request, generic_api_secret)
        if auth is not None:
            return auth
        customer_id = str(request.query_params.get("customer_id") or "").strip()
        record = get_file_vault().get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        return {"ok": True, "file": _web_file_metadata(record)}

    @app.get("/web/files/{file_id}/content")
    async def web_file_content(file_id: str, request: Request) -> Any:
        auth = _authorized_web_request(request, generic_api_secret)
        if auth is not None:
            return auth
        customer_id = str(request.query_params.get("customer_id") or "").strip()
        vault = get_file_vault()
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})
        filename = str(record.get("original_filename") or "file.bin")
        mime_type = str(record.get("mime_type") or "").strip()
        return Response(
            content=raw_bytes,
            media_type=mime_type or "application/octet-stream",
            headers={"Content-Disposition": _content_disposition(filename)},
        )

    @app.get("/web/local-files/content")
    async def web_local_file_content(request: Request) -> Any:
        auth = _authorized_web_request(request, generic_api_secret)
        if auth is not None:
            return auth
        local_path = str(request.query_params.get("path") or "").strip()
        if not local_path:
            return JSONResponse(status_code=400, content={"detail": "path is required"})
        try:
            target = (TULPA_STUFF_DIR.parent / local_path).resolve()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "invalid path"})
        if not is_within(target, TULPA_STUFF_DIR) or not target.exists() or target.is_dir():
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        raw_bytes = target.read_bytes()
        guessed_mime, _ = mimetypes.guess_type(str(target.name))
        return Response(
            content=raw_bytes,
            media_type=guessed_mime or "application/octet-stream",
            headers={"Content-Disposition": _content_disposition(target.name)},
        )


async def _stream_turn(
    *,
    runtime: Any,
    workflow_setup_service: Any,
    file_vault: Any,
    customer_id: str,
    thread_id: str,
    text: str,
    file_ids: list[str],
    include_pending_context: bool,
) -> AsyncIterator[str]:
    assert customer_id.strip()
    assert thread_id.strip()
    assert text.strip()
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=100)

    async def _send_owner_update(message: str) -> dict[str, Any]:
        safe = str(message or "").strip()
        if not safe:
            return {"ok": False, "sent": False, "reason": "empty_message"}
        await queue.put(("owner_update", {"message": safe}))
        return {"ok": True, "sent": True}

    async def _send_file(file: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(file, dict) or not file:
            return {"ok": False, "sent": False, "reason": "missing_file"}
        await queue.put(("file", {"file": _normalize_file_event(file)}))
        return {"ok": True, "sent": True}

    async def _run_turn() -> None:
        turn_mode = "interactive"
        if workflow_setup_service is not None and hasattr(workflow_setup_service, "thread_status"):
            status = workflow_setup_service.thread_status(
                customer_id=customer_id,
                thread_id=thread_id,
            )
            if str(status.get("status", "") or "").strip().lower() == "active":
                turn_mode = "workflow_setup"

        try:
            if hasattr(runtime, "register_interactive_update_sender"):
                await runtime.register_interactive_update_sender(
                    thread_id=thread_id,
                    sender=_send_owner_update,
                )
            if hasattr(runtime, "register_interactive_file_sender"):
                await runtime.register_interactive_file_sender(
                    thread_id=thread_id,
                    sender=_send_file,
                )
            effective_text = _text_with_uploaded_file_context(
                file_vault=file_vault,
                customer_id=customer_id,
                text=text,
                file_ids=file_ids,
            )
            if turn_mode == "workflow_setup" or not hasattr(runtime, "astream_text"):
                final_text = await runtime.ainvoke_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=effective_text,
                    turn_mode=turn_mode,
                    include_pending_context=include_pending_context,
                )
            else:
                final_text = ""
                async for chunk in runtime.astream_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=effective_text,
                    turn_mode=turn_mode,
                    include_pending_context=include_pending_context,
                ):
                    current = str(chunk or "").strip()
                    if not current:
                        continue
                    progress = _progress_message(current)
                    if progress:
                        await queue.put(("status", {"message": progress}))
                        continue
                    final_text = current
                    await queue.put(("delta", {"text": current}))
            final_text = str(final_text or "").strip()
            if (
                turn_mode == "workflow_setup"
                and workflow_setup_service is not None
                and hasattr(workflow_setup_service, "after_reply")
            ):
                workflow_setup_service.after_reply(
                    customer_id=customer_id,
                    thread_id=thread_id,
                    reply_text=final_text,
                )
            await queue.put(("final", {"text": final_text}))
        except Exception as exc:
            await queue.put(("error", {"message": str(exc), "type": type(exc).__name__}))
        finally:
            if hasattr(runtime, "clear_interactive_update_sender"):
                with suppress(Exception):
                    await runtime.clear_interactive_update_sender(
                        thread_id=thread_id,
                        sender=_send_owner_update,
                    )
            if hasattr(runtime, "clear_interactive_file_sender"):
                with suppress(Exception):
                    await runtime.clear_interactive_file_sender(
                        thread_id=thread_id,
                        sender=_send_file,
                    )
            await queue.put(("done", {}))

    yield _sse("status", {"message": "Starting..."})
    task = asyncio.create_task(_run_turn())
    try:
        while True:
            event, payload = await queue.get()
            if event == "done":
                break
            yield _sse(event, payload)
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


def _authorized_web_request(request: Request, generic_api_secret: str | None) -> JSONResponse | None:
    secret = str(generic_api_secret or "").strip()
    if not secret:
        return JSONResponse(
            status_code=503,
            content={"detail": "OPENTULPA_GENERIC_API_SECRET is not configured"},
        )
    if not _authorized(request, secret):
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return None


async def _postprocess_uploaded_file(
    *,
    runtime: Any,
    vault: Any,
    customer_id: str,
    record: dict[str, Any],
    raw_bytes: bytes,
    caption: str | None,
) -> dict[str, Any]:
    kind = str(record.get("kind") or "").strip()
    filename = str(record.get("original_filename") or "").strip()
    mime_type = str(record.get("mime_type") or "").strip() or None
    if kind == "voice" and runtime is not None and hasattr(runtime, "transcribe_audio_blob"):
        with suppress(Exception):
            transcript = await runtime.transcribe_audio_blob(
                filename=filename or "voice.ogg",
                mime_type=mime_type,
                kind=kind,
                raw_bytes=raw_bytes,
            )
            if str(transcript or "").strip():
                updated = vault.set_ai_summary(customer_id, str(record.get("id") or ""), str(transcript))
                if isinstance(updated, dict):
                    record = updated
    if (
        runtime is not None
        and hasattr(runtime, "summarize_uploaded_blob")
        and not _skip_auto_summary_for_upload(kind=kind, filename=filename, mime_type=mime_type)
    ):
        with suppress(Exception):
            summary = await runtime.summarize_uploaded_blob(
                filename=filename or None,
                mime_type=mime_type,
                kind=kind,
                raw_bytes=raw_bytes,
                caption=caption,
            )
            if str(summary or "").strip():
                updated = vault.set_ai_summary(customer_id, str(record.get("id") or ""), str(summary))
                if isinstance(updated, dict):
                    record = updated
    return record


def _web_file_metadata(record: dict[str, Any]) -> dict[str, Any]:
    clean = sanitize_uploaded_file_record(record, include_excerpt=False)
    file_id = str(clean.get("id") or "").strip()
    if file_id:
        clean["content_path"] = f"/web/files/{quote(file_id)}/content"
        clean["metadata_path"] = f"/web/files/{quote(file_id)}/metadata"
    return clean


def _normalize_file_event(file: dict[str, Any]) -> dict[str, Any]:
    payload = dict(file)
    file_id = str(payload.get("id") or "").strip()
    local_path = str(payload.get("local_path") or "").strip()
    if file_id and not payload.get("content_path"):
        payload["content_path"] = f"/web/files/{quote(file_id)}/content"
    if local_path and not payload.get("content_path"):
        payload["content_path"] = f"/web/local-files/content?path={quote(local_path)}"
    return payload


def _text_with_uploaded_file_context(
    *,
    file_vault: Any,
    customer_id: str,
    text: str,
    file_ids: list[str],
) -> str:
    records = file_vault.get_many(customer_id, file_ids) if file_ids else []
    context = build_uploaded_files_context(records) if records else ""
    if not context:
        return text
    return f"{context}\n\nCurrent user message:\n{text}"


def _content_disposition(filename: str) -> str:
    safe = str(filename or "file.bin").replace('"', "")
    return f'inline; filename="{safe}"'
