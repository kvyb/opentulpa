"""Authenticated generic chat routes for non-Telegram clients."""

from __future__ import annotations

import asyncio
import json
import mimetypes
import time
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from opentulpa.agent.runtime import STREAM_PROGRESS_PREFIX, STREAM_WAIT_SIGNAL, AgentStreamEvent
from opentulpa.api.customer_ids import resolve_customer_id as resolve_customer_id_value
from opentulpa.api.file_helpers import sanitize_uploaded_file_record
from opentulpa.api.web_auth import web_auth_error
from opentulpa.context.uploaded_files import (
    build_uploaded_files_context,
    should_skip_auto_summary_for_upload,
)
from opentulpa.core.shutdown_drain import ShutdownDrainingError
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR, is_within
from opentulpa.web.events import append_web_event

MAX_WEB_UPLOAD_BYTES = 45_000_000


class WebChatTurnRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    customer_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    file_ids: list[str] = Field(default_factory=list, max_length=20)
    include_pending_context: bool = True

    @field_validator("file_ids", mode="before")
    @classmethod
    def _clean_file_ids(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("file_ids must be a list")
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned[:20]


class WebFileResponse(BaseModel):
    ok: bool
    file: dict[str, Any]


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
    web_token: str | None,
    get_agent_runtime: Callable[[], Any],
    get_file_vault: Callable[[], Any],
    get_workflow_setup_service: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
    get_shutdown_drain: Callable[[], Any] | None = None,
) -> None:
    """Register authenticated generic chat endpoints."""

    def _resolve_customer_id(customer_id: str) -> str:
        return resolve_customer_id_value(customer_id, resolve_customer_id)

    @app.post(
        "/web/chat/turns",
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def web_chat_turn(payload: WebChatTurnRequest, request: Request) -> Any:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error

        runtime = get_agent_runtime()
        if runtime is None or not (hasattr(runtime, "ainvoke_text") or hasattr(runtime, "astream_text")):
            return JSONResponse(status_code=503, content={"detail": "agent runtime unavailable"})
        drain = get_shutdown_drain() if get_shutdown_drain is not None else None
        if drain is not None and bool(getattr(drain, "draining", False)):
            return JSONResponse(status_code=503, content={"detail": "instance draining"})

        return StreamingResponse(
            _stream_turn(
                runtime=runtime,
                shutdown_drain=drain,
                workflow_setup_service=get_workflow_setup_service(),
                file_vault=get_file_vault(),
                customer_id=_resolve_customer_id(payload.customer_id),
                thread_id=payload.thread_id,
                text=payload.text,
                file_ids=payload.file_ids,
                include_pending_context=payload.include_pending_context,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/web/files/upload", response_model=WebFileResponse)
    async def web_file_upload(
        request: Request,
        customer_id: Annotated[str, Form(min_length=1)],
        thread_id: Annotated[str, Form(min_length=1)],
        upload: Annotated[UploadFile, File(alias="file")],
        kind: Annotated[str, Form()] = "document",
        caption: Annotated[str | None, Form()] = None,
    ) -> Any:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error
        safe_customer_id = _resolve_customer_id(customer_id)
        safe_thread_id = str(thread_id or "").strip()
        safe_kind = str(kind or "document").strip() or "document"
        safe_caption = str(caption or "").strip() or None
        if not safe_customer_id or not safe_thread_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and thread_id are required"},
            )
        raw_bytes = await upload.read()
        if not isinstance(raw_bytes, bytes) or not raw_bytes:
            return JSONResponse(status_code=400, content={"detail": "file is empty"})
        if len(raw_bytes) > MAX_WEB_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"detail": "file is too large"})
        filename = str(getattr(upload, "filename", "") or f"{safe_kind}.bin").strip()
        content_type = str(getattr(upload, "content_type", "") or "").strip() or None
        vault = get_file_vault()
        record = vault.ingest_file(
            customer_id=safe_customer_id,
            chat_id=None,
            kind=safe_kind,
            telegram_file_id=None,
            original_filename=filename,
            mime_type=content_type,
            caption=safe_caption,
            raw_bytes=raw_bytes,
        )
        runtime = get_agent_runtime()
        record = await _postprocess_uploaded_file(
            runtime=runtime,
            vault=vault,
            customer_id=safe_customer_id,
            record=record,
            raw_bytes=raw_bytes,
            caption=safe_caption,
        )
        return {
            "ok": True,
            "file": _web_file_metadata(record),
        }

    @app.get("/web/files/{file_id}/metadata", response_model=WebFileResponse)
    async def web_file_metadata(
        file_id: str,
        request: Request,
        customer_id: Annotated[str, Query(min_length=1)],
    ) -> Any:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error
        record = get_file_vault().get_file(_resolve_customer_id(customer_id), file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        return {"ok": True, "file": _web_file_metadata(record)}

    @app.get("/web/files/{file_id}/content")
    async def web_file_content(
        file_id: str,
        request: Request,
        customer_id: Annotated[str, Query(min_length=1)],
    ) -> Any:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error
        safe_customer_id = _resolve_customer_id(customer_id)
        vault = get_file_vault()
        record = vault.get_file(safe_customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        raw_bytes = vault.read_file_bytes(safe_customer_id, file_id)
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
    async def web_local_file_content(
        request: Request,
        local_path: Annotated[str, Query(alias="path", min_length=1)],
    ) -> Any:
        auth_error = web_auth_error(request, web_token)
        if auth_error is not None:
            return auth_error
        safe_local_path = local_path.strip()
        if not safe_local_path:
            return JSONResponse(status_code=400, content={"detail": "path is required"})
        try:
            target = (TULPA_STUFF_DIR.parent / safe_local_path).resolve()
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
    shutdown_drain: Any | None = None,
) -> AsyncIterator[str]:
    assert customer_id.strip()
    assert thread_id.strip()
    assert text.strip()
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=100)

    async def _send_owner_update(message: str) -> dict[str, Any]:
        safe = str(message or "").strip()
        if not safe:
            return {"ok": False, "sent": False, "reason": "empty_message"}
        append_web_event(
            customer_id=customer_id,
            thread_id=thread_id,
            source="web",
            kind="owner_update",
            text=safe,
        )
        await queue.put(("owner_update", {"message": safe}))
        return {"ok": True, "sent": True}

    async def _send_file(file: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(file, dict) or not file:
            return {"ok": False, "sent": False, "reason": "missing_file"}
        await queue.put(("file", {"file": _normalize_file_event(file)}))
        return {"ok": True, "sent": True}

    async def _run_turn() -> None:
        turn_context = shutdown_drain.active_turn() if shutdown_drain is not None else None
        turn_context_entered = False
        try:
            if turn_context is not None:
                await turn_context.__aenter__()
                turn_context_entered = True
        except ShutdownDrainingError:
            await queue.put(("error", {"message": "instance draining", "type": "ShutdownDrainingError"}))
            await queue.put(("done", {}))
            return

        append_web_event(
            customer_id=customer_id,
            thread_id=thread_id,
            source="web",
            kind="user_message",
            text=text,
        )
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
            if not hasattr(runtime, "astream_text"):
                final_text = await runtime.ainvoke_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=effective_text,
                    turn_mode=turn_mode,
                    include_pending_context=include_pending_context,
                )
            else:
                final_text = ""
                delta_seq = 0
                async for chunk in runtime.astream_text(
                    thread_id=thread_id,
                    customer_id=customer_id,
                    text=effective_text,
                    turn_mode=turn_mode,
                    include_pending_context=include_pending_context,
                    stream_precommit_seconds=0.0,
                    stream_incremental_deltas=True,
                    stream_status_events=True,
                ):
                    if isinstance(chunk, AgentStreamEvent):
                        if chunk.event in {"reasoning", "status", "tool_call"}:
                            if chunk.event == "tool_call":
                                final_text = ""
                            await queue.put((chunk.event, chunk.payload))
                        continue
                    current = str(chunk or "")
                    if not current:
                        continue
                    progress = _progress_message(current.strip())
                    if progress:
                        final_text = ""
                        await queue.put(("status", {"message": progress}))
                        continue
                    final_text = f"{final_text}{current}"
                    delta_seq += 1
                    await queue.put(
                        (
                            "delta",
                            {
                                "text": current,
                                "append": True,
                                "seq": delta_seq,
                                "server_received_at_ms": int(time.time() * 1000),
                            },
                        )
                    )
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
            if final_text:
                append_web_event(
                    customer_id=customer_id,
                    thread_id=thread_id,
                    source="web",
                    kind="assistant_message",
                    text=final_text,
                )
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
            if turn_context_entered and turn_context is not None:
                await turn_context.__aexit__(None, None, None)
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
        and not should_skip_auto_summary_for_upload(kind=kind, filename=filename, mime_type=mime_type)
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
    customer_id = str(clean.get("customer_id") or "").strip()
    if file_id:
        query = f"?customer_id={quote(customer_id, safe='')}" if customer_id else ""
        clean["content_path"] = f"/web/files/{quote(file_id)}/content{query}"
        clean["metadata_path"] = f"/web/files/{quote(file_id)}/metadata{query}"
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
