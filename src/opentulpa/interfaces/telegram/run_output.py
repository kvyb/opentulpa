"""Bounded, rate-limited Telegram output for one active agent run."""

from __future__ import annotations

import asyncio
from typing import Any

from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.telegram_formatting import TELEGRAM_TEXT_CHAR_LIMIT

_EDIT_MIN_INTERVAL_SECONDS = 1.5
_PREVIEW_CHAR_LIMIT = TELEGRAM_TEXT_CHAR_LIMIT - 300
_PREVIEW_PREFIX = "[Earlier progress omitted]\n\n"
_TOOL_PROGRESS_LABELS = {
    "source_shell": "Working on OpenTulpa source",
    "source_status": "Reviewing OpenTulpa changes",
    "write_todos": "Updating the plan",
}


def _message_id(response: dict[str, Any] | None) -> int | None:
    if not isinstance(response, dict):
        return None
    result = response.get("result")
    value = result.get("message_id") if isinstance(result, dict) else response.get("message_id")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


class TelegramRunOutput:
    """Edit one compact Telegram message while a run is active."""

    def __init__(self, *, client: TelegramClient, chat_id: int) -> None:
        self._client = client
        self._chat_id = chat_id
        self._message_id: int | None = None
        self._rendered = ""
        self._last_edit = 0.0
        self._segment: list[str] = []
        self._latest_segment = ""
        self._pending_text: str | None = None
        self._pending_task: asyncio.Task[None] | None = None
        self._closed = False

    async def delta(self, text: str) -> None:
        value = str(text or "")
        if not value or self._closed:
            return
        self._segment.append(value)
        segment = self._bounded_segment("".join(self._segment))
        self._segment = [segment]
        await self._replace(segment)

    async def tool_started(self, name: str) -> None:
        if self._closed:
            return
        self._finish_segment()
        label = _TOOL_PROGRESS_LABELS.get(name, f"Running {name or 'tool'}")
        text = f"{self._latest_segment}\n\nWorking: {label}..." if self._latest_segment else (
            f"Working: {label}..."
        )
        await self._replace(text)

    async def tool_completed(self) -> None:
        self._finish_segment()

    async def finish(self, text: str) -> None:
        if self._closed:
            return
        await self._cancel_pending()
        final = str(text or "").strip()
        if not final:
            self._finish_segment()
            final = self._latest_segment
        response = await self._client.send_message(
            chat_id=self._chat_id,
            text=final or "The run completed without a message.",
            parse_mode=None,
        )
        if response is not None and self._message_id is not None:
            await self._client.delete_message(
                chat_id=self._chat_id,
                message_id=self._message_id,
            )
        self._closed = True

    async def discard(self) -> None:
        """Remove transient progress once another durable Telegram message replaces it."""

        if self._closed:
            return
        await self._cancel_pending()
        if self._message_id is not None:
            await self._client.delete_message(
                chat_id=self._chat_id,
                message_id=self._message_id,
            )
        self._closed = True

    def _finish_segment(self) -> None:
        segment = self._bounded_segment("".join(self._segment)).strip()
        self._segment.clear()
        if segment:
            self._latest_segment = segment

    async def _replace(self, text: str) -> None:
        preview = self._bounded_preview(text)
        if self._closed or not preview:
            return
        now = asyncio.get_running_loop().time()
        if self._message_id is None:
            if self._rendered:
                return
            await self._apply(preview)
            return
        if now - self._last_edit >= _EDIT_MIN_INTERVAL_SECONDS:
            await self._cancel_pending()
            await self._apply(preview)
            return
        self._pending_text = preview
        if self._pending_task is None or self._pending_task.done():
            delay = _EDIT_MIN_INTERVAL_SECONDS - (now - self._last_edit)
            self._pending_task = asyncio.create_task(self._flush_pending(delay))

    async def _apply(self, preview: str) -> None:
        if self._closed or preview == self._rendered:
            return
        if self._message_id is None:
            response = await self._client.send_message(
                chat_id=self._chat_id,
                text=preview,
                parse_mode=None,
            )
            if response is None:
                return
            self._message_id = _message_id(response)
        else:
            edited = await self._client.edit_message_text(
                chat_id=self._chat_id,
                message_id=self._message_id,
                text=preview,
                parse_mode=None,
            )
            if not edited:
                return
        self._rendered = preview
        self._last_edit = asyncio.get_running_loop().time()

    async def _flush_pending(self, delay: float) -> None:
        try:
            await asyncio.sleep(max(0.0, delay))
            preview = self._pending_text
            self._pending_text = None
            if preview is not None:
                await self._apply(preview)
        finally:
            self._pending_task = None
            preview = self._pending_text
            self._pending_text = None
            if preview is not None and not self._closed:
                await self._replace(preview)

    async def _cancel_pending(self) -> None:
        task = self._pending_task
        self._pending_task = None
        self._pending_text = None
        if task is None or task.done() or task is asyncio.current_task():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    @staticmethod
    def _bounded_preview(text: str) -> str:
        return TelegramRunOutput._bounded_segment(str(text or "").strip())

    @staticmethod
    def _bounded_segment(text: str) -> str:
        preview = str(text or "")
        if len(preview) <= _PREVIEW_CHAR_LIMIT:
            return preview
        available = _PREVIEW_CHAR_LIMIT - len(_PREVIEW_PREFIX)
        return _PREVIEW_PREFIX + preview[-available:].lstrip()


__all__ = ["TelegramRunOutput"]
