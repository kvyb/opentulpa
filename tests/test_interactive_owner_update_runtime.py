from __future__ import annotations

import asyncio
import contextvars
from typing import Any

import pytest

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def _runtime_for_updates() -> OpenTulpaLangGraphRuntime:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._interactive_update_senders_lock = asyncio.Lock()
    runtime._interactive_update_senders = {}
    runtime._interactive_update_sent_keys = {}
    runtime._active_customer_id_ctx = contextvars.ContextVar("test_customer_id", default="")
    runtime._active_thread_id_ctx = contextvars.ContextVar("test_thread_id", default="")
    runtime._active_customer_id = ""
    runtime._active_thread_id = ""
    runtime._behavior_log_enabled = False
    return runtime


@pytest.mark.asyncio
async def test_emit_interactive_update_sends_once_and_dedupes() -> None:
    runtime = _runtime_for_updates()
    runtime.set_active_customer_id("telegram_1")
    runtime.set_active_thread_id("thread_1")
    sent: list[str] = []

    async def _sender(text: str) -> dict[str, Any]:
        sent.append(text)
        return {"sent": True}

    await runtime.register_interactive_update_sender(thread_id="thread_1", sender=_sender)

    first = await runtime.emit_interactive_update(text="Проверяю прайс.", dedupe_key="price")
    second = await runtime.emit_interactive_update(text="Проверяю прайс.", dedupe_key="price")

    assert first == {"ok": True, "sent": True}
    assert second == {"ok": True, "sent": False, "duplicate": True}
    assert sent == ["Проверяю прайс."]


@pytest.mark.asyncio
async def test_emit_interactive_update_stops_after_sender_clear() -> None:
    runtime = _runtime_for_updates()
    runtime.set_active_customer_id("telegram_1")
    runtime.set_active_thread_id("thread_1")

    async def _sender(text: str) -> dict[str, Any]:
        del text
        return {"sent": True}

    await runtime.register_interactive_update_sender(thread_id="thread_1", sender=_sender)
    await runtime.clear_interactive_update_sender(thread_id="thread_1", sender=_sender)

    result = await runtime.emit_interactive_update(text="Проверяю прайс.")

    assert result == {
        "ok": False,
        "sent": False,
        "reason": "interactive_update_unavailable",
    }


@pytest.mark.asyncio
async def test_emit_interactive_update_failed_send_clears_fallback_thread_dedupe() -> None:
    runtime = _runtime_for_updates()
    runtime.set_active_customer_id("telegram_1")
    runtime.set_active_thread_id("thread_1")
    attempts: list[str] = []
    events: list[dict[str, Any]] = []

    async def _sender(text: str) -> dict[str, Any]:
        attempts.append(text)
        return {"sent": False}

    runtime.log_behavior_event = lambda **kwargs: events.append(kwargs)  # type: ignore[assignment]
    await runtime.register_interactive_update_sender(thread_id="thread_1", sender=_sender)

    first = await runtime.emit_interactive_update(text="Проверяю прайс.", dedupe_key="price")
    second = await runtime.emit_interactive_update(text="Проверяю прайс.", dedupe_key="price")

    assert first == {"ok": False, "sent": False, "reason": "send_failed"}
    assert second == {"ok": False, "sent": False, "reason": "send_failed"}
    assert attempts == ["Проверяю прайс.", "Проверяю прайс."]
    assert runtime._interactive_update_sent_keys["thread_1"] == set()
    assert events == []
