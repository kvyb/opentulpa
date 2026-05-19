from __future__ import annotations

import time
from pathlib import Path
from typing import Any


class FakeTelegramClient:
    def __init__(self, _token: str) -> None:
        self.callback_answers: list[dict[str, Any]] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.sent_files: list[dict[str, Any]] = []
        self.edited_messages: list[dict[str, Any]] = []
        self.chat_actions: list[dict[str, Any]] = []
        self.command_menu_calls: list[dict[str, Any]] = []
        self.registered_files: dict[str, dict[str, Any]] = {}
        self.downloaded_files: list[dict[str, Any]] = []
        self._message_id = 10_000

    def register_file(
        self,
        *,
        file_id: str,
        path: Path,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> dict[str, Any]:
        raw_bytes = path.read_bytes()
        record = {
            "file_id": str(file_id or "").strip(),
            "file_path": str(path),
            "filename": str(filename or path.name),
            "mime_type": str(mime_type or "application/octet-stream"),
            "file_size": len(raw_bytes),
            "raw_bytes": raw_bytes,
        }
        self.registered_files[record["file_id"]] = record
        return {k: v for k, v in record.items() if k != "raw_bytes"}

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> bool:
        self.callback_answers.append(
            {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": bool(show_alert),
            }
        )
        return True

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._message_id += 1
        result = {
            "message_id": self._message_id,
            "date": int(time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "text": text,
        }
        safe_business_connection_id = str(kwargs.get("business_connection_id", "") or "").strip()
        if safe_business_connection_id:
            result["business_connection_id"] = safe_business_connection_id
            result["sender_business_bot"] = {"id": "fake-bot"}
        self.sent_messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup or {},
                **kwargs,
                "message_id": self._message_id,
            }
        )
        return {"ok": True, "result": result}

    async def send_message_draft(
        self,
        *,
        chat_id: int | str,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
        parse_mode: str | None = "HTML",
    ) -> bool:
        _ = {
            "chat_id": chat_id,
            "draft_id": draft_id,
            "text": text,
            "message_thread_id": message_thread_id,
            "parse_mode": parse_mode,
        }
        return False

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
        self._message_id += 1
        self.sent_files.append(
            {
                "chat_id": chat_id,
                "filename": str(filename or "file.bin"),
                "size_bytes": len(raw_bytes),
                "kind": str(kind or "document"),
                "mime_type": mime_type,
                "caption": caption,
                "parse_mode": parse_mode,
                "message_id": self._message_id,
            }
        )
        return True

    async def edit_message_text(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        self.edited_messages.append(
            {
                "chat_id": chat_id,
                "message_id": int(message_id),
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup or {},
            }
        )
        return True

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        self.edited_messages.append(
            {
                "chat_id": chat_id,
                "message_id": int(message_id),
                "text": "",
                "parse_mode": None,
                "reply_markup": reply_markup or {},
            }
        )
        return True

    async def send_chat_action(self, *, chat_id: int | str, action: str = "typing") -> bool:
        self.chat_actions.append({"chat_id": chat_id, "action": action})
        return True

    async def set_my_commands(
        self,
        *,
        commands: list[dict[str, str]],
        scope: dict[str, Any] | None = None,
    ) -> bool:
        self.command_menu_calls.append({"commands": commands, "scope": scope or {}})
        return True

    async def download_file(self, *, file_id: str) -> dict[str, Any] | None:
        safe_file_id = str(file_id or "").strip()
        record = self.registered_files.get(safe_file_id)
        self.downloaded_files.append({"file_id": safe_file_id, "found": record is not None})
        if record is None:
            return None
        return {
            "file_path": str(record.get("filename") or record.get("file_path") or safe_file_id),
            "file_size": int(record.get("file_size") or 0),
            "mime_type": str(record.get("mime_type") or "application/octet-stream"),
            "raw_bytes": bytes(record.get("raw_bytes") or b""),
        }

    async def aclose(self) -> None:
        return None
