"""Telegram API client primitives."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from contextlib import suppress
from typing import Any

import httpx

from opentulpa.interfaces.telegram.formatter import (
    prepare_text_and_mode,
    prepare_text_chunks_and_mode,
)

logger = logging.getLogger(__name__)


def _supports_message_draft(chat_id: int | str) -> bool:
    if isinstance(chat_id, int):
        return chat_id > 0
    text = str(chat_id or "").strip()
    if not text:
        return False
    with suppress(Exception):
        return int(text) > 0
    return False


def _telegram_retry_after_seconds(data: dict[str, Any] | None) -> float | None:
    params = data.get("parameters", {}) if isinstance(data, dict) else {}
    if not isinstance(params, dict):
        return None
    value = params.get("retry_after")
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(min(value, 15))


def _response_json_dict(response: Any) -> dict[str, Any] | None:
    with suppress(Exception):
        data = response.json()
        if isinstance(data, dict):
            return data
    return None


def _resolve_media_send_target(
    *,
    kind: str,
    filename: str,
    mime_type: str | None,
) -> tuple[str, str]:
    safe_kind = str(kind or "").strip().lower()
    safe_name = str(filename or "").strip().lower()
    safe_mime = str(mime_type or "").strip().lower()

    is_gif = safe_mime == "image/gif" or safe_name.endswith(".gif")
    if safe_kind in {"animation", "gif"} or is_gif:
        return "sendAnimation", "animation"
    if safe_kind == "photo" and safe_mime.startswith("image/"):
        return "sendPhoto", "photo"
    return "sendDocument", "document"


class TelegramClient:
    """Thin async client around Telegram Bot API endpoints used by OpenTulpa."""

    def __init__(self, bot_token: str) -> None:
        self.bot_token = str(bot_token or "").strip()
        self._client: Any | None = None

    def _http_client(self) -> Any:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        close = getattr(client, "aclose", None)
        if callable(close):
            with suppress(Exception):
                await close()

    async def _post(self, method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        retryable_http = {408, 429, 500, 502, 503, 504}
        timeout = httpx.Timeout(20.0, connect=8.0, read=15.0)
        max_attempts = 3
        client = self._http_client()
        for attempt in range(max_attempts):
            try:
                r = await client.post(url, json=payload, timeout=timeout)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.4 * (2**attempt))
                    continue
                logger.warning("Telegram API %s transport error: %s", method, exc)
                return None

            if not r.is_success:
                data = _response_json_dict(r)
                # Telegram returns HTTP 400 for no-op edits:
                # "Bad Request: message is not modified".
                # For streaming loader updates this is benign and should not be treated
                # as a hard failure.
                if method == "editMessageText":
                    desc = str((data or {}).get("description", "")).lower()
                    if "message is not modified" in desc:
                        return {"ok": True, "result": {}}
                if r.status_code in retryable_http and attempt < max_attempts - 1:
                    retry_after = _telegram_retry_after_seconds(data)
                    await asyncio.sleep(
                        retry_after if retry_after is not None else 0.4 * (2**attempt)
                    )
                    continue
                if method == "sendChatAction" and r.status_code == 429:
                    logger.info(
                        "Telegram API %s throttled after retries: %s",
                        method,
                        (r.text or "")[:400],
                    )
                    return None
                logger.warning(
                    "Telegram API %s HTTP %s: %s",
                    method,
                    r.status_code,
                    (r.text or "")[:400],
                )
                return None

            try:
                data = r.json()
            except Exception:
                logger.warning("Telegram API %s returned non-JSON body", method)
                return None

            if isinstance(data, dict) and data.get("ok") is True:
                return data

            if attempt < max_attempts - 1:
                retry_after = _telegram_retry_after_seconds(data)
                await asyncio.sleep(
                    retry_after if retry_after is not None else 0.4 * (2**attempt)
                )
                continue

            if method == "sendChatAction" and _telegram_retry_after_seconds(data) is not None:
                logger.info("Telegram API %s throttled after retries.", method)
                return None
            logger.warning("Telegram API %s returned error payload: %s", method, str(data)[:400])
            return None
        return None

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        business_connection_id: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any] | None:
        safe_business_connection_id = str(business_connection_id or "").strip()
        chunks, final_mode = prepare_text_chunks_and_mode(text, parse_mode)
        if not chunks:
            return None
        first_data: dict[str, Any] | None = None
        sent_results: list[dict[str, Any]] = []
        for idx, final_text in enumerate(chunks):
            payload: dict[str, Any] = {"chat_id": chat_id, "text": final_text}
            if final_mode:
                payload["parse_mode"] = final_mode
            if idx == 0 and isinstance(reply_markup, dict):
                payload["reply_markup"] = reply_markup
            if safe_business_connection_id:
                payload["business_connection_id"] = safe_business_connection_id
            if idx == 0 and isinstance(reply_to_message_id, int) and reply_to_message_id > 0:
                payload["reply_parameters"] = {"message_id": reply_to_message_id}
            data = await self._post("sendMessage", payload)
            if not isinstance(data, dict):
                return None
            sent_results.append(data)
            if first_data is None:
                first_data = data
        if first_data is not None and len(sent_results) > 1:
            first_data = dict(first_data)
            first_data["results"] = sent_results
        return first_data

    async def send_message_draft(
        self,
        *,
        chat_id: int | str,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
        parse_mode: str | None = "HTML",
    ) -> bool:
        if not _supports_message_draft(chat_id):
            return False
        final_text, final_mode = prepare_text_and_mode(text, parse_mode)
        if not final_text:
            return False
        safe_draft_id = int(draft_id)
        if safe_draft_id == 0:
            return False
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "draft_id": safe_draft_id,
            "text": final_text,
        }
        if isinstance(message_thread_id, int) and message_thread_id > 0:
            payload["message_thread_id"] = message_thread_id
        if final_mode:
            payload["parse_mode"] = final_mode
        data = await self._post("sendMessageDraft", payload)
        return bool(data)

    async def edit_message_text(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        final_text, final_mode = prepare_text_and_mode(text, parse_mode)
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id, "text": final_text}
        if final_mode:
            payload["parse_mode"] = final_mode
        if isinstance(reply_markup, dict):
            payload["reply_markup"] = reply_markup
        data = await self._post("editMessageText", payload)
        return bool(data)

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        payload: dict[str, Any] = {"chat_id": chat_id, "message_id": message_id}
        if isinstance(reply_markup, dict):
            payload["reply_markup"] = reply_markup
        data = await self._post("editMessageReplyMarkup", payload)
        return bool(data)

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> bool:
        payload: dict[str, Any] = {
            "callback_query_id": str(callback_query_id or "").strip(),
            "show_alert": bool(show_alert),
        }
        if text:
            payload["text"] = str(text).strip()[:180]
        if not payload["callback_query_id"]:
            return False
        data = await self._post("answerCallbackQuery", payload)
        return bool(data)

    async def set_my_commands(
        self,
        *,
        commands: list[dict[str, str]],
        scope: dict[str, Any] | None = None,
    ) -> bool:
        payload: dict[str, Any] = {"commands": commands}
        if isinstance(scope, dict):
            payload["scope"] = scope
        data = await self._post("setMyCommands", payload)
        return bool(data)

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
        business_connection_id: str | None = None,
    ) -> bool:
        safe_action = str(action or "").strip() or "typing"
        payload: dict[str, Any] = {"chat_id": chat_id, "action": safe_action}
        safe_business_connection_id = str(business_connection_id or "").strip()
        if safe_business_connection_id:
            payload["business_connection_id"] = safe_business_connection_id
        data = await self._post("sendChatAction", payload)
        return bool(data)

    async def delete_message(
        self,
        *,
        chat_id: int | str,
        message_id: int,
    ) -> bool:
        data = await self._post(
            "deleteMessage",
            {"chat_id": chat_id, "message_id": int(message_id)},
        )
        return bool(data)

    async def get_me(self) -> dict[str, Any] | None:
        data = await self._post("getMe", {})
        result = data.get("result") if isinstance(data, dict) else None
        return result if isinstance(result, dict) else None

    async def get_webhook_info(self) -> dict[str, Any] | None:
        data = await self._post("getWebhookInfo", {})
        result = data.get("result") if isinstance(data, dict) else None
        return result if isinstance(result, dict) else None

    async def download_file(self, *, file_id: str) -> dict[str, Any] | None:
        info = await self._post("getFile", {"file_id": file_id})
        if not info:
            return None
        result = info.get("result") if isinstance(info, dict) else None
        if not isinstance(result, dict):
            return None
        file_path = str(result.get("file_path", "")).strip()
        if not file_path:
            return None
        file_size = result.get("file_size")
        guessed_mime, _ = mimetypes.guess_type(file_path)
        url = f"https://api.telegram.org/file/bot{self.bot_token}/{file_path}"
        try:
            client = self._http_client()
            resp = await client.get(url, timeout=45.0)
        except Exception:
            return None
        if not resp.is_success:
            return None
        return {
            "file_path": file_path,
            "file_size": int(file_size) if isinstance(file_size, int) else len(resp.content),
            "mime_type": guessed_mime,
            "raw_bytes": resp.content,
        }

    async def send_file(
        self,
        *,
        chat_id: int | str,
        filename: str,
        raw_bytes: bytes,
        kind: str = "document",
        mime_type: str | None = None,
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> bool:
        safe_name = str(filename or "file.bin").strip() or "file.bin"
        method, media_field = _resolve_media_send_target(
            kind=kind,
            filename=safe_name,
            mime_type=mime_type,
        )

        payload: dict[str, Any] = {"chat_id": str(chat_id)}
        if caption:
            final_caption, final_mode = prepare_text_and_mode(caption, parse_mode)
            if final_caption:
                payload["caption"] = final_caption
            if final_mode:
                payload["parse_mode"] = final_mode

        files = {media_field: (safe_name, raw_bytes, mime_type or "application/octet-stream")}
        url = f"https://api.telegram.org/bot{self.bot_token}/{method}"
        try:
            client = self._http_client()
            resp = await client.post(url, data=payload, files=files, timeout=60.0)
        except Exception:
            return False
        if not resp.is_success:
            logger.warning(
                "Telegram API %s HTTP %s: %s",
                method,
                resp.status_code,
                (resp.text or "")[:400],
            )
            return False
        try:
            data = resp.json()
        except Exception:
            logger.warning("Telegram API %s returned non-JSON body", method)
            return False
        ok = bool(isinstance(data, dict) and data.get("ok") is True)
        if not ok:
            logger.warning("Telegram API %s returned error payload: %s", method, str(data)[:400])
        return ok

    async def send_files(
        self,
        *,
        chat_id: int | str,
        files: list[dict[str, Any]],
        caption: str | None = None,
        parse_mode: str | None = None,
    ) -> bool:
        if not files:
            return False
        media: list[dict[str, Any]] = []
        multipart_files: dict[str, tuple[str, bytes, str]] = {}
        for idx, item in enumerate(files):
            filename = str(item.get("filename") or "file.bin").strip() or "file.bin"
            raw_bytes = item.get("raw_bytes")
            if not isinstance(raw_bytes, (bytes, bytearray)):
                continue
            mime_type = str(item.get("mime_type") or "application/octet-stream").strip() or "application/octet-stream"
            attach_name = f"file{idx}"
            media_item: dict[str, Any] = {"type": "document", "media": f"attach://{attach_name}"}
            if idx == 0 and caption:
                final_caption, final_mode = prepare_text_and_mode(caption, parse_mode)
                if final_caption:
                    media_item["caption"] = final_caption
                if final_mode:
                    media_item["parse_mode"] = final_mode
            media.append(media_item)
            multipart_files[attach_name] = (filename, bytes(raw_bytes), mime_type)
        if not media:
            return False
        payload: dict[str, Any] = {"chat_id": str(chat_id), "media": json.dumps(media, ensure_ascii=False)}
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMediaGroup"
        try:
            client = self._http_client()
            resp = await client.post(url, data=payload, files=multipart_files, timeout=120.0)
        except Exception:
            return False
        if not resp.is_success:
            logger.warning(
                "Telegram API sendMediaGroup HTTP %s: %s",
                resp.status_code,
                (resp.text or "")[:400],
            )
            return False
        try:
            data = resp.json()
        except Exception:
            logger.warning("Telegram API sendMediaGroup returned non-JSON body")
            return False
        ok = bool(isinstance(data, dict) and data.get("ok") is True)
        if not ok:
            logger.warning("Telegram API sendMediaGroup returned error payload: %s", str(data)[:400])
        return ok


def parse_telegram_update(body: dict) -> tuple[int | None, int | None, str | None]:
    """Extract (chat_id, user_id, text) from a Telegram update."""
    try:
        message = body.get("message") or body.get("edited_message")
        if not message:
            return None, None, None
        chat_id = message.get("chat", {}).get("id")
        user_id = message.get("from", {}).get("id")
        text = (message.get("text") or message.get("caption") or "").strip()
        return chat_id, user_id, text
    except Exception:
        return None, None, None


def parse_telegram_callback_query(
    body: dict[str, Any],
) -> tuple[str | None, int | None, int | None, str | None, int | None]:
    """Extract callback query metadata: (callback_id, user_id, chat_id, data, message_id)."""
    try:
        callback = body.get("callback_query")
        if not isinstance(callback, dict):
            return None, None, None, None, None
        callback_id = str(callback.get("id", "")).strip() or None
        user_id = callback.get("from", {}).get("id")
        raw_message = callback.get("message")
        message = raw_message if isinstance(raw_message, dict) else {}
        chat_id = message.get("chat", {}).get("id")
        message_id = message.get("message_id")
        data = str(callback.get("data", "")).strip() or None
        return callback_id, user_id, chat_id, data, message_id if isinstance(message_id, int) else None
    except Exception:
        return None, None, None, None, None
