from __future__ import annotations

import asyncio

import pytest

from opentulpa.interfaces.telegram import relay as relay_module


class _NeverYieldsRuntime:
    async def astream_text(self, **kwargs):
        while True:
            await asyncio.sleep(10.0)
            yield ""


class _NeverYieldsWithFallbackRuntime:
    async def astream_text(self, **kwargs):
        while True:
            await asyncio.sleep(10.0)
            yield ""

    async def ainvoke_text(self, **kwargs):
        return "Recovered via non-stream fallback."


class _FakeTelegramClient:
    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token
        self.draft_calls: list[tuple[int | str, int, str, str | None, int | None]] = []
        self.message_calls: list[tuple[int | str, str, str | None]] = []
        self.chat_actions: list[tuple[int | str, str]] = []

    async def send_message_draft(
        self,
        *,
        chat_id: int | str,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
        parse_mode: str | None = None,
    ) -> bool:
        self.draft_calls.append((chat_id, draft_id, text, parse_mode, message_thread_id))
        return True

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup=None,
    ) -> bool:
        del reply_markup
        self.message_calls.append((chat_id, text, parse_mode))
        return True

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
    ) -> bool:
        self.chat_actions.append((chat_id, action))
        return True


@pytest.mark.asyncio
async def test_stream_timeout_returns_user_visible_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    # Force timeout path quickly.
    original_wait_for = relay_module.asyncio.wait_for
    calls = {"count": 0}

    async def _fast_timeout(awaitable, timeout):
        # Only force timeout for the stream-token wait path.
        # Let other wait_for usages (e.g. loader stop_event waits) behave normally.
        if asyncio.iscoroutine(awaitable):
            code = getattr(awaitable, "cr_code", None)
            if getattr(code, "co_name", "") == "wait":
                return await original_wait_for(awaitable, timeout)
        calls["count"] += 1
        raise asyncio.TimeoutError()

    monkeypatch.setattr(relay_module.asyncio, "wait_for", _fast_timeout)
    try:
        final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=_NeverYieldsRuntime(),
            thread_id="chat-1",
            customer_id="telegram_1",
            text="hello",
            bot_token="dummy",
            chat_id=1,
        )
    finally:
        monkeypatch.setattr(relay_module.asyncio, "wait_for", original_wait_for)

    assert suppressed is False
    assert isinstance(final, str)
    assert "timed out" in final.lower()
    # One automatic retry is attempted before surfacing timeout.
    assert calls["count"] >= 2
    assert any("timed out" in text.lower() for _, _, text, _, _ in fake_client.draft_calls) or any(
        "timed out" in text.lower() for _, text, _ in fake_client.message_calls
    )
    assert fake_client.chat_actions


@pytest.mark.asyncio
async def test_stream_timeout_uses_non_stream_recovery_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    original_wait_for = relay_module.asyncio.wait_for
    calls = {"count": 0}

    async def _timeout_stream_then_allow(awaitable, timeout):
        if asyncio.iscoroutine(awaitable):
            code = getattr(awaitable, "cr_code", None)
            if getattr(code, "co_name", "") == "wait":
                return await original_wait_for(awaitable, timeout)
        calls["count"] += 1
        if calls["count"] <= 2:
            raise asyncio.TimeoutError()
        return await original_wait_for(awaitable, timeout)

    monkeypatch.setattr(relay_module.asyncio, "wait_for", _timeout_stream_then_allow)
    try:
        final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=_NeverYieldsWithFallbackRuntime(),
            thread_id="chat-1",
            customer_id="telegram_1",
            text="hello",
            bot_token="dummy",
            chat_id=1,
        )
    finally:
        monkeypatch.setattr(relay_module.asyncio, "wait_for", original_wait_for)

    assert suppressed is False
    assert isinstance(final, str)
    assert "recovered via non-stream fallback" in final.lower()
    assert not any("timed out" in text.lower() for _, text, _ in fake_client.message_calls)
