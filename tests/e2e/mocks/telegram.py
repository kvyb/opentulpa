from __future__ import annotations

import time
from typing import Any


class FakeTelegramClient:
    def __init__(self, _token: str) -> None:
        self.callback_answers: list[dict[str, Any]] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.edited_messages: list[dict[str, Any]] = []
        self.chat_actions: list[dict[str, Any]] = []
        self._message_id = 10_000

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

    async def aclose(self) -> None:
        return None
