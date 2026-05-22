"""Use-case helpers for internal file-vault HTTP routes."""

from __future__ import annotations

import mimetypes
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi.responses import JSONResponse

from opentulpa.agent.knowledge_prep import inspect_uploaded_file_structure
from opentulpa.api.customer_ids import resolve_body_customer_id
from opentulpa.api.file_helpers import (
    download_image_from_web_url,
    sanitize_uploaded_file_record,
)
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR, is_within

MAX_LOCAL_SEND_BYTES = 45_000_000


def _telegram_delivery_payload(**fields: Any) -> dict[str, Any]:
    payload = {
        "ok": True,
        "delivered_to_chat": True,
        "model_instruction": (
            "DELIVERED_TO_CHAT: The file has been sent to Telegram. "
            "Do not call the file-send tool again for this file. "
            "Continue with a short final confirmation only."
        ),
    }
    payload.update(fields)
    return payload


def _chat_delivery_payload(**fields: Any) -> dict[str, Any]:
    payload = {
        "ok": True,
        "delivered_to_chat": True,
        "model_instruction": (
            "DELIVERED_TO_CHAT: The file has been sent to the current chat. "
            "Do not call the file-send tool again for this file. "
            "Continue with a short final confirmation only."
        ),
    }
    payload.update(fields)
    return payload


@dataclass(frozen=True)
class FileRouteUseCases:
    get_file_vault: Callable[[], Any]
    get_telegram_chat: Callable[[], Any]
    get_telegram_client: Callable[[], Any]
    get_agent_runtime: Callable[[], Any]
    telegram_enabled: bool
    resolve_customer_id: Callable[[str], str] | None = None
    tulpa_stuff_dir: Path = TULPA_STUFF_DIR
    download_image: Callable[..., Any] = download_image_from_web_url

    def _customer_id(self, body: dict[str, Any]) -> str:
        return resolve_body_customer_id(body, self.resolve_customer_id)

    def _chat_id_for_customer(self, customer_id: str) -> Any | None:
        slots = self.get_telegram_chat().find_session_slots(customer_id)
        return slots[0].get("chat_id") if slots else None

    async def search(self, body: dict[str, Any]) -> Any:
        vault = self.get_file_vault()
        customer_id = self._customer_id(body)
        query = str(body.get("query", "")).strip()
        limit = int(body.get("limit", 5))
        results = [
            sanitize_uploaded_file_record(r, include_excerpt=False)
            for r in vault.search(customer_id, query=query, limit=limit)
        ]
        return {"ok": True, "results": results}

    async def get(self, body: dict[str, Any]) -> Any:
        vault = self.get_file_vault()
        customer_id = self._customer_id(body)
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

    async def send(self, body: dict[str, Any]) -> Any:
        vault = self.get_file_vault()
        customer_id = self._customer_id(body)
        file_id = str(body.get("file_id", "")).strip()
        caption = _optional_caption(body)
        if not self.telegram_enabled:
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
            chat_id = self._chat_id_for_customer(customer_id)
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})

        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})

        sent = await self.get_telegram_client().send_file(
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
        return _telegram_delivery_payload(file_id=file_id, chat_id=chat_id)

    async def send_local(self, body: dict[str, Any]) -> Any:
        customer_id = self._customer_id(body)
        local_path = str(body.get("path", "")).strip()
        caption = _optional_caption(body)
        if not self.telegram_enabled:
            return JSONResponse(status_code=501, content={"detail": "Telegram not configured"})
        if not customer_id or not local_path:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and path are required"}
            )
        try:
            target = (self.tulpa_stuff_dir.parent / local_path).resolve()
        except Exception:
            return JSONResponse(status_code=400, content={"detail": "invalid path"})
        if not is_within(target, self.tulpa_stuff_dir):
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
                    "detail": f"file too large ({file_size} bytes > {MAX_LOCAL_SEND_BYTES} bytes)"
                },
            )

        chat_id = self._chat_id_for_customer(customer_id)
        if chat_id is None:
            return JSONResponse(status_code=404, content={"detail": "no chat found for customer"})
        try:
            raw_bytes = target.read_bytes()
        except Exception:
            return JSONResponse(status_code=502, content={"detail": "failed to read local file"})
        guessed_mime, _ = mimetypes.guess_type(str(target.name))
        sent = await self.get_telegram_client().send_file(
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
        return _telegram_delivery_payload(path=local_path, chat_id=chat_id)

    async def send_web_image(self, body: dict[str, Any]) -> Any:
        vault = self.get_file_vault()
        customer_id = self._customer_id(body)
        image_url = str(body.get("url", "")).strip()
        caption = _optional_caption(body)
        max_bytes = int(body.get("max_bytes", 10_000_000))

        if not customer_id or not image_url:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and url are required"},
            )

        chat_id = self._chat_id_for_customer(customer_id) if self.telegram_enabled else None
        try:
            downloaded = await self.download_image(image_url, max_bytes=max_bytes)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=502, content={"detail": f"image fetch failed: {exc}"})

        if chat_id is None:
            record = vault.ingest_file(
                customer_id=customer_id,
                chat_id=None,
                kind=_downloaded_image_kind(downloaded),
                telegram_file_id=None,
                original_filename=str(downloaded["filename"]),
                mime_type=str(downloaded["content_type"]),
                caption=caption,
                raw_bytes=downloaded["raw_bytes"],
            )
            emitter = getattr(self.get_agent_runtime(), "emit_interactive_file", None)
            if not callable(emitter):
                return JSONResponse(status_code=501, content={"detail": "file delivery unavailable"})
            emitted = await emitter(file=sanitize_uploaded_file_record(record))
            if not isinstance(emitted, dict) or not emitted.get("sent"):
                reason = str((emitted or {}).get("reason") or "file delivery unavailable")
                return JSONResponse(status_code=501, content={"detail": reason})
            return _chat_delivery_payload(file_id=record.get("id"))

        sent = await self.get_telegram_client().send_file(
            chat_id=chat_id,
            filename=str(downloaded["filename"]),
            raw_bytes=downloaded["raw_bytes"],
            kind=_downloaded_image_kind(downloaded),
            mime_type=str(downloaded["content_type"]),
            caption=caption,
            parse_mode="HTML",
        )
        if not sent:
            return JSONResponse(status_code=502, content={"detail": "telegram send failed"})
        return _telegram_delivery_payload(
            chat_id=chat_id,
            url=str(downloaded["final_url"]),
            mime_type=str(downloaded["content_type"]),
            size_bytes=int(downloaded["size_bytes"]),
        )

    async def analyze(self, body: dict[str, Any]) -> Any:
        vault = self.get_file_vault()
        customer_id = self._customer_id(body)
        file_id = str(body.get("file_id", "")).strip()
        question = _optional_question(body)
        if not customer_id or not file_id:
            return JSONResponse(
                status_code=400, content={"detail": "customer_id and file_id are required"}
            )
        agent_runtime = self.get_agent_runtime()
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
            "file": sanitize_uploaded_file_record(
                record, include_excerpt=True, max_excerpt_chars=16000
            ),
        }

    async def inspect_structure(self, body: dict[str, Any]) -> Any:
        vault = self.get_file_vault()
        customer_id = self._customer_id(body)
        file_id = str(body.get("file_id", "")).strip()
        if not customer_id or not file_id:
            return JSONResponse(
                status_code=400,
                content={"detail": "customer_id and file_id are required"},
            )
        record = vault.get_file(customer_id, file_id)
        if not record:
            return JSONResponse(status_code=404, content={"detail": "file not found"})
        raw_bytes = vault.read_file_bytes(customer_id, file_id)
        if raw_bytes is None:
            return JSONResponse(status_code=404, content={"detail": "stored file bytes not found"})
        inspected = inspect_uploaded_file_structure(
            raw_bytes=raw_bytes,
            filename=str(record.get("original_filename", "") or "file.bin"),
            mime_type=str(record.get("mime_type", "") or ""),
            search_terms=body.get("search_terms"),
        )
        return {
            "ok": True,
            "file": sanitize_uploaded_file_record(record, include_excerpt=False),
            "inspection": inspected,
        }


def _optional_caption(body: dict[str, Any]) -> str | None:
    caption_raw = body.get("caption")
    caption = str(caption_raw).strip() if caption_raw is not None else None
    return caption or None


def _optional_question(body: dict[str, Any]) -> str | None:
    question_raw = body.get("question")
    question = str(question_raw).strip() if question_raw is not None else None
    return question or None


def _downloaded_image_kind(downloaded: dict[str, Any]) -> str:
    return "animation" if str(downloaded["content_type"]).strip().lower() == "image/gif" else "photo"
