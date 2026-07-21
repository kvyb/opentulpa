"""Small Telegram Bot API client for the standalone interface worker."""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from typing import Any

import httpx


class TelegramAPIError(RuntimeError):
    """Sanitized Telegram transport error that never contains the bot token."""


@dataclass(frozen=True, slots=True)
class TelegramAttachment:
    kind: str
    file_id: str
    filename: str
    mime_type: str | None
    declared_size: int | None


def extract_attachments(message: dict[str, Any]) -> list[TelegramAttachment]:
    attachments: list[TelegramAttachment] = []
    document = message.get("document")
    if isinstance(document, dict):
        file_id = str(document.get("file_id") or "").strip()
        if file_id:
            attachments.append(
                TelegramAttachment(
                    kind="document",
                    file_id=file_id,
                    filename=str(document.get("file_name") or "document.bin").strip(),
                    mime_type=str(document.get("mime_type") or "").strip() or None,
                    declared_size=_positive_int(document.get("file_size")),
                )
            )

    photos = message.get("photo")
    if isinstance(photos, list):
        candidates = [item for item in photos if isinstance(item, dict)]
        chosen = max(candidates, key=lambda item: int(item.get("file_size") or 0), default=None)
        if chosen is not None:
            file_id = str(chosen.get("file_id") or "").strip()
            if file_id:
                unique = str(chosen.get("file_unique_id") or "photo").strip()
                attachments.append(
                    TelegramAttachment(
                        kind="photo",
                        file_id=file_id,
                        filename=f"{unique}.jpg",
                        mime_type="image/jpeg",
                        declared_size=_positive_int(chosen.get("file_size")),
                    )
                )

    extensions = {
        "video": ".mp4",
        "video_note": ".mp4",
        "audio": ".mp3",
        "voice": ".ogg",
    }
    for kind, extension in extensions.items():
        item = message.get(kind)
        if not isinstance(item, dict):
            continue
        file_id = str(item.get("file_id") or "").strip()
        if not file_id:
            continue
        unique = str(item.get("file_unique_id") or kind).strip()
        attachments.append(
            TelegramAttachment(
                kind=kind,
                file_id=file_id,
                filename=str(item.get("file_name") or f"{unique}{extension}").strip(),
                mime_type=str(item.get("mime_type") or "").strip() or None,
                declared_size=_positive_int(item.get("file_size")),
            )
        )
    return attachments


def _positive_int(value: Any) -> int | None:
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None


class TelegramBotAPI:
    """Token-scoped Telegram Bot API client using JSON and bounded downloads."""

    def __init__(
        self,
        *,
        token: str,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.telegram.org",
        max_attachment_bytes: int = 45_000_000,
    ) -> None:
        safe_token = str(token or "").strip()
        if not safe_token:
            raise ValueError("Telegram bot token is required")
        if max_attachment_bytes <= 0:
            raise ValueError("max_attachment_bytes must be positive")
        self._token = safe_token
        self._base_url = str(base_url or "").strip().rstrip("/")
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self._max_attachment_bytes = max_attachment_bytes

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _method_url(self, method: str) -> str:
        return f"{self._base_url}/bot{self._token}/{method}"

    async def _post(
        self,
        method: str,
        payload: dict[str, Any],
        *,
        timeout: float,
    ) -> Any:
        try:
            response = await self._client.post(
                self._method_url(method),
                json=payload,
                timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise TelegramAPIError(f"Telegram {method} transport failed.") from exc
        if not response.is_success:
            raise TelegramAPIError(f"Telegram {method} returned HTTP {response.status_code}.")
        try:
            data = response.json()
        except ValueError as exc:
            raise TelegramAPIError(f"Telegram {method} returned invalid JSON.") from exc
        if not isinstance(data, dict) or data.get("ok") is not True:
            raise TelegramAPIError(f"Telegram {method} rejected the request.")
        return data.get("result")

    async def get_me(self) -> dict[str, Any]:
        result = await self._post("getMe", {}, timeout=20)
        if not isinstance(result, dict) or _positive_int(result.get("id")) is None:
            raise TelegramAPIError("Telegram getMe returned an invalid bot identity.")
        return result

    async def delete_webhook(self) -> None:
        """Ensure long polling is the bot's only Telegram update transport."""

        result = await self._post(
            "deleteWebhook",
            {"drop_pending_updates": False},
            timeout=20,
        )
        if result is not True:
            raise TelegramAPIError("Telegram deleteWebhook returned an invalid result.")

    async def get_updates(self, *, offset: int, timeout_seconds: int) -> list[dict[str, Any]]:
        result = await self._post(
            "getUpdates",
            {
                "offset": max(0, int(offset)),
                "timeout": max(1, min(int(timeout_seconds), 50)),
                "allowed_updates": ["message", "edited_message", "callback_query"],
            },
            timeout=max(15, timeout_seconds + 10),
        )
        if not isinstance(result, list):
            raise TelegramAPIError("Telegram getUpdates returned an invalid update list.")
        updates = [item for item in result if isinstance(item, dict)]
        return sorted(updates, key=lambda item: int(item.get("update_id") or -1))

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        chunks = _text_chunks(text)
        for index, chunk in enumerate(chunks):
            payload: dict[str, Any] = {"chat_id": int(chat_id), "text": chunk}
            if index == 0 and reply_markup is not None:
                payload["reply_markup"] = reply_markup
            await self._post("sendMessage", payload, timeout=30)

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> None:
        await self._post(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query_id,
                "text": text[:180],
                "show_alert": show_alert,
            },
            timeout=20,
        )

    async def download_attachment(self, attachment: TelegramAttachment) -> bytes:
        if (
            attachment.declared_size is not None
            and attachment.declared_size > self._max_attachment_bytes
        ):
            raise TelegramAPIError("Telegram attachment exceeds the configured size limit.")
        result = await self._post(
            "getFile",
            {"file_id": attachment.file_id},
            timeout=20,
        )
        if not isinstance(result, dict):
            raise TelegramAPIError("Telegram getFile returned invalid metadata.")
        file_path = str(result.get("file_path") or "").strip().lstrip("/")
        if not file_path or ".." in file_path.split("/"):
            raise TelegramAPIError("Telegram getFile returned an unsafe path.")
        declared = _positive_int(result.get("file_size"))
        if declared is not None and declared > self._max_attachment_bytes:
            raise TelegramAPIError("Telegram attachment exceeds the configured size limit.")
        try:
            response = await self._client.get(
                f"{self._base_url}/file/bot{self._token}/{file_path}",
                timeout=60,
            )
        except httpx.HTTPError as exc:
            raise TelegramAPIError("Telegram attachment download failed.") from exc
        if not response.is_success:
            raise TelegramAPIError(
                f"Telegram attachment download returned HTTP {response.status_code}."
            )
        if len(response.content) > self._max_attachment_bytes:
            raise TelegramAPIError("Telegram attachment exceeds the configured size limit.")
        return response.content

    @staticmethod
    def inferred_mime_type(attachment: TelegramAttachment) -> str | None:
        if attachment.mime_type:
            return attachment.mime_type
        mime_type, _ = mimetypes.guess_type(attachment.filename)
        return mime_type


def _text_chunks(text: str, *, limit: int = 4_000) -> list[str]:
    remaining = str(text or "").strip()
    if not remaining:
        return ["The operation completed without a message."]
    chunks: list[str] = []
    while len(remaining) > limit:
        split_at = remaining.rfind("\n", 0, limit + 1)
        if split_at < limit // 2:
            split_at = remaining.rfind(" ", 0, limit + 1)
        if split_at < limit // 2:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


__all__ = [
    "TelegramAPIError",
    "TelegramAttachment",
    "TelegramBotAPI",
    "extract_attachments",
]
