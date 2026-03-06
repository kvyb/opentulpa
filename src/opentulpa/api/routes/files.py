"""Internal file-vault route registration."""

from __future__ import annotations

from collections.abc import Callable
import mimetypes
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from opentulpa.api.file_helpers import (
    download_image_from_web_url,
    sanitize_uploaded_file_record,
)
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR, is_within

MAX_LOCAL_SEND_BYTES = 45_000_000


def register_file_routes(
    app: FastAPI,
    *,
    get_file_vault: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_telegram_client: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    telegram_enabled: bool,
) -> None:
    """Register uploaded-file search/get/send/analyze endpoints."""

    @app.post("/internal/files/search")
    async def internal_files_search(request: Request) -> Any:
        vault = get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        query = str(body.get("query", "")).strip()
        limit = int(body.get("limit", 5))
        results = [
            sanitize_uploaded_file_record(r, include_excerpt=False)
            for r in vault.search(customer_id, query=query, limit=limit)
        ]
        return {"ok": True, "results": results}

    @app.post("/internal/files/get")
    async def internal_files_get(request: Request) -> Any:
        vault = get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        file_id = str(body.get("file_id", "")).strip()
        max_excerpt_chars = max(500, min(int(body.get("max_excerpt_chars", 16000)), 60000))
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        return {
            "ok": True,
            "file": sanitize_uploaded_file_record(
                record,
                include_excerpt=True,
                max_excerpt_chars=max_excerpt_chars,
            ),
        }

    @app.post("/internal/files/send")
    async def internal_files_send(request: Request) -> Any:
        vault = get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        file_id = str(body.get("file_id", "")).strip()
        caption_raw = body.get("caption")
        caption = str(caption_raw).strip() if caption_raw is not None else None
        caption = caption or None
        if not telegram_enabled:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if not customer_id or not file_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and file_id are required"}
            )
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})

        chat_id = record.get("chat_id")
        if chat_id is None:
            slots = get_telegram_chat().find_session_slots(customer_id)
            if slots:
                chat_id = slots[0].get("chat_id")
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})

        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})

        sent = await get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(record.get("original_filename", "file.bin")),
            raw_bytes=raw_bytes,
            kind=str(record.get("kind", "document")),
            mime_type=str(record.get("mime_type", "")).strip() or None,
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return JSONResponse(status_code=502, content={"detail": "telegram send failed"})
        return {"ok": True, "file_id": file_id, "chat_id": chat_id}

    @app.post("/internal/files/send_local")
    async def internal_files_send_local(request: Request) -> Any:
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        local_path = str(body.get("path", "")).strip()
        caption_raw = body.get("caption")
        caption = str(caption_raw).strip() if caption_raw is not None else None
        caption = caption or None
        if not telegram_enabled:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if not customer_id or not local_path:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and path are required"}
            )
        try:
            target = (TULPA_STUFF_DIR.parent / local_path).resolve()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "invalid path"})
        if not is_within(target, TULPA_STUFF_DIR):
            return JSONResponse(status_code=400, content={"detail": "path must be under tulpa_stuff/"})
        if not target.exists():
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        if target.is_dir():
            return JSONResponse(status_code=400, content={"detail": "path is a directory"})
        try:
            file_size = int(target.stat().st_size)
        except Exception:
            return JSONResponse(status_code=502, content={"detail": "failed to stat local file"})
        if file_size > MAX_LOCAL_SEND_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "detail": (
                        f"file too large ({file_size} bytes > {MAX_LOCAL_SEND_BYTES} bytes)"
                    )
                },
            )

        chat_id: Any = None
        slots = get_telegram_chat().find_session_slots(customer_id)
        if slots:
            chat_id = slots[0].get("chat_id")
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})
        try:
            raw_bytes = target.read_bytes()
        except Exception:
            return JSONResponse(status_code=502, content={"detail": "failed to read local file"})
        guessed_mime, _ = mimetypes.guess_type(str(target.name))
        sent = await get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(target.name),
            raw_bytes=raw_bytes,
            kind="document",
            mime_type=str(guessed_mime).strip() if guessed_mime else None,
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return JSONResponse(status_code=502, content={"detail": "telegram send failed"})
        return {"ok": True, "path": local_path, "chat_id": chat_id}

    @app.post("/internal/files/send_web_image")
    async def internal_files_send_web_image(request: Request) -> Any:
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        image_url = str(body.get("url", "")).strip()
        caption_raw = body.get("caption")
        caption = str(caption_raw).strip() if caption_raw is not None else None
        caption = caption or None
        max_bytes = int(body.get("max_bytes", 10_000_000))

        if not telegram_enabled:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if not customer_id or not image_url:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and url are required"},
            )

        chat_id: Any = None
        slots = get_telegram_chat().find_session_slots(customer_id)
        if slots:
            chat_id = slots[0].get("chat_id")
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})

        try:
            downloaded = await download_image_from_web_url(image_url, max_bytes=max_bytes)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=502, content={"detail": f"image fetch failed: {exc}"})

        sent = await get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(downloaded["filename"]),
            raw_bytes=downloaded["raw_bytes"],
            kind=(
                "animation"
                if str(downloaded["content_type"]).strip().lower() == "image/gif"
                else "photo"
            ),
            mime_type=str(downloaded["content_type"]),
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return JSONResponse(status_code=502, content={"detail": "telegram send failed"})
        return {
            "ok": True,
            "chat_id": chat_id,
            "url": str(downloaded["final_url"]),
            "mime_type": str(downloaded["content_type"]),
            "size_bytes": int(downloaded["size_bytes"]),
        }

    @app.post("/internal/files/analyze")
    async def internal_files_analyze(request: Request) -> Any:
        vault = get_file_vault()
        body = await request.json()
        customer_id = str(body.get("customer_id", "")).strip()
        file_id = str(body.get("file_id", "")).strip()
        question_raw = body.get("question")
        question = str(question_raw).strip() if question_raw is not None else None
        question = question or None
        if not customer_id or not file_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and file_id are required"}
            )
        agent_runtime = get_agent_runtime()
        if agent_runtime is None or not hasattr(agent_runtime, "analyze_uploaded_file"):
            return JSONResponse(
                status_code=501,
                content={"detail": "agent runtime unavailable for file analysis"},
            )
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})
        try:
            analysis_result = await agent_runtime.analyze_uploaded_file(
                record=record,
                raw_bytes=raw_bytes,
                question=question,
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"detail": f"file analysis failed: {exc}"})

        if not question:
            analysis_text = str(analysis_result.get("analysis", "")).strip()
            if analysis_text:
                updated = vault.set_ai_summary(customer_id, file_id, analysis_text)
                if isinstance(updated, dict):
                    record = updated
        return {
            "ok": True,
            "analysis": str(analysis_result.get("analysis", "")).strip(),
            "file": sanitize_uploaded_file_record(record, include_excerpt=True, max_excerpt_chars=16000),
        }
