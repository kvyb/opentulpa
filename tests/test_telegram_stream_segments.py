from __future__ import annotations

import asyncio

import pytest

from opentulpa.agent.runtime import STREAM_PROGRESS_PREFIX, STREAM_WAIT_SIGNAL
from opentulpa.interfaces.telegram import relay as relay_module


class _SegmentedRuntime:
    async def astream_text(self, **kwargs):
        yield "I have access to your inbox. I will check it now."
        yield STREAM_WAIT_SIGNAL
        await asyncio.sleep(0.02)
        yield "I checked your inbox. 3 priority emails found."


class _ToolFirstRuntime:
    async def astream_text(self, **kwargs):
        yield STREAM_WAIT_SIGNAL
        await asyncio.sleep(0.02)
        yield "I checked the inbox. 3 priority emails found."


class _ResultThenWaitRuntime:
    async def astream_text(self, **kwargs):
        yield "Here is the finished result."
        yield STREAM_WAIT_SIGNAL


class _UpdatingProgressRuntime:
    async def astream_text(self, **kwargs):
        yield f"{STREAM_PROGRESS_PREFIX}Searching the web…"
        await asyncio.sleep(0.01)
        yield f"{STREAM_PROGRESS_PREFIX}Fetching a webpage…"
        await asyncio.sleep(0.01)
        yield "Here is the result."


class _RapidChunkRuntime:
    async def astream_text(self, **kwargs):
        yield "H"
        yield "He"
        yield "Hel"
        yield "Hell"
        yield "Hello"
        yield "Hello "
        yield "Hello w"
        yield "Hello wo"
        yield "Hello wor"
        yield "Hello worl"
        yield "Hello world"


class _FakeTelegramClient:
    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token
        self.calls: list[tuple[int | str, str, int | None, str | None]] = []
        self._next_id = 100
        self.chat_actions: list[tuple[int | str, str]] = []
        self.deleted_messages: list[tuple[int | str, int]] = []

    async def upsert_stream_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        message_id: int | None = None,
        parse_mode: str | None = None,
        allow_fallback_send: bool = True,
        reply_markup=None,
    ) -> int | None:
        self.calls.append((chat_id, text, message_id, parse_mode))
        if message_id is None:
            self._next_id += 1
            return self._next_id
        return message_id

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
    ) -> bool:
        self.chat_actions.append((chat_id, action))
        return True

    async def delete_message(self, *, chat_id: int | str, message_id: int) -> bool:
        self.deleted_messages.append((chat_id, message_id))
        return True


@pytest.mark.asyncio
async def test_stream_reuses_same_message_for_new_meaningful_segment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "PROGRESS_MESSAGE_DELAY_SECONDS", 0.0)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_SegmentedRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="check inbox",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "priority emails" in str(final or "").lower()

    assert len(fake_client.calls) >= 2
    assert fake_client.calls[0][2] is None
    assert fake_client.calls[-1][2] is not None
    assert any("working on it" in text.lower() for _, text, _, _ in fake_client.calls)
    assert fake_client.deleted_messages
    assert fake_client.chat_actions


@pytest.mark.asyncio
async def test_stream_wait_signal_creates_separate_progress_message_before_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "PROGRESS_MESSAGE_DELAY_SECONDS", 0.0)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ToolFirstRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="check inbox",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "priority emails" in str(final or "").lower()
    assert fake_client.calls[0][1] == relay_module.PROGRESS_STATUS_TEXT
    assert fake_client.calls[0][2] is None
    assert fake_client.calls[-1][1].startswith("I checked the inbox")
    assert fake_client.calls[-1][2] is None
    assert fake_client.deleted_messages


@pytest.mark.asyncio
async def test_progress_message_is_deleted_when_stream_ends_after_wait_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ResultThenWaitRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="finish",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Here is the finished result."
    assert fake_client.calls[0][1] == "Here is the finished result."
    assert not any(text == relay_module.PROGRESS_STATUS_TEXT for _, text, _, _ in fake_client.calls)
    assert not fake_client.deleted_messages


@pytest.mark.asyncio
async def test_progress_updates_edit_one_message_before_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "PROGRESS_MESSAGE_DELAY_SECONDS", 0.0)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_UpdatingProgressRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="search",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Here is the result."
    assert fake_client.calls[0][1] == "Searching the web…"
    assert fake_client.calls[0][2] is None
    assert fake_client.calls[1][1] == "Fetching a webpage…"
    assert fake_client.calls[1][2] is not None
    assert fake_client.calls[-1][1] == "Here is the result."
    assert fake_client.calls[-1][2] is None
    assert fake_client.deleted_messages


@pytest.mark.asyncio
async def test_stream_throttles_rapid_partial_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "STREAM_EDIT_MIN_INTERVAL_SECONDS", 10.0)
    monkeypatch.setattr(relay_module, "STREAM_EDIT_MIN_CHAR_DELTA", 1000)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_RapidChunkRuntime(),
        thread_id="chat-rapid",
        customer_id="telegram_rapid",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Hello world"
    assert fake_client.calls[0][1] == "H"
    assert fake_client.calls[-1][1] == "Hello world"
    assert len(fake_client.calls) < 11
