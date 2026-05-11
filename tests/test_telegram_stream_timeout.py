from __future__ import annotations

import asyncio
import logging

import pytest

from opentulpa.interfaces.telegram import relay as relay_module


class _NeverYieldsRuntime:
    def __init__(self, status: str | None = None) -> None:
        self.status = status
        self.status_calls = 0
        self.status_contexts: list[dict] = []

    async def astream_text(self, **kwargs):
        while True:
            await asyncio.sleep(10.0)
            yield ""

    async def generate_status_message(self, **kwargs):
        self.status_calls += 1
        context = kwargs.get("context")
        if isinstance(context, dict):
            self.status_contexts.append(context)
        if not self.status:
            return None
        return {"ok": True, "text": self.status}


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
    runtime = _NeverYieldsRuntime(status="Уточняю детали и скоро отвечу.")

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
            awaitable.close()
        calls["count"] += 1
        raise TimeoutError()

    monkeypatch.setattr(relay_module.asyncio, "wait_for", _fast_timeout)
    try:
        final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=runtime,
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
    assert "reply timeout" in final
    assert runtime.status_calls == 1
    assert runtime.status_contexts
    assert runtime.status_contexts[0]["latest_user_message"] == "hello"
    assert "do not quote" in runtime.status_contexts[0]["latest_user_message_usage"]
    # One automatic retry is attempted before surfacing timeout.
    assert calls["count"] >= 2
    sent_texts = [text for _, text, _ in fake_client.message_calls]
    assert "Уточняю детали и скоро отвечу." in sent_texts
    assert final in sent_texts
    assert fake_client.chat_actions


@pytest.mark.asyncio
async def test_stream_timeout_sends_final_timeout_when_status_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    runtime = _NeverYieldsRuntime(status=None)
    caplog.set_level(logging.WARNING, logger=relay_module.__name__)

    original_wait_for = relay_module.asyncio.wait_for
    calls = {"count": 0}

    async def _fast_timeout(awaitable, timeout):
        if asyncio.iscoroutine(awaitable):
            code = getattr(awaitable, "cr_code", None)
            if getattr(code, "co_name", "") == "wait":
                return await original_wait_for(awaitable, timeout)
            awaitable.close()
        calls["count"] += 1
        raise TimeoutError()

    monkeypatch.setattr(relay_module.asyncio, "wait_for", _fast_timeout)
    try:
        final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=runtime,
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
    assert "reply timeout" in final
    assert runtime.status_calls == 1
    assert [text for _, text, _ in fake_client.message_calls] == [final]
    assert calls["count"] >= 2
    messages = [record.getMessage() for record in caplog.records]
    assert any("telegram.stream status_generation_skipped" in message for message in messages)
    assert any(
        "telegram.stream timeout_without_final_reply" in message
        and "interim_status_sent=False" in message
        and "delivered_any=False" in message
        for message in messages
    )


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
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise TimeoutError()
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
