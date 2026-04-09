from __future__ import annotations

import asyncio

import pytest

from opentulpa.agent.runtime import STREAM_APPROVAL_HANDOFF_SIGNAL, STREAM_PROGRESS_PREFIX, STREAM_WAIT_SIGNAL
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


class _WordByWordRuntime:
    async def astream_text(self, **kwargs):
        yield "This is a"
        yield "This is a slightly"
        yield "This is a slightly longer"
        yield "This is a slightly longer streamed"
        yield "This is a slightly longer streamed reply"
        yield "This is a slightly longer streamed reply with"
        yield "This is a slightly longer streamed reply with enough"
        yield "This is a slightly longer streamed reply with enough words."


class _PartialThenApprovalRuntime:
    async def astream_text(self, **kwargs):
        yield "I started checking that for you."
        yield STREAM_APPROVAL_HANDOFF_SIGNAL
        yield "This should never be visible."


class _FakeTelegramClient:
    def __init__(self, bot_token: str, *, draft_ok: bool = True) -> None:
        self.bot_token = bot_token
        self.draft_ok = draft_ok
        self.draft_calls: list[tuple[int | str, int, str, str | None, int | None]] = []
        self.message_calls: list[tuple[int | str, str, str | None]] = []
        self.chat_actions: list[tuple[int | str, str]] = []
        self.deleted_messages: list[tuple[int | str, int]] = []

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
        return self.draft_ok

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

    async def delete_message(self, *, chat_id: int | str, message_id: int) -> bool:
        self.deleted_messages.append((chat_id, message_id))
        return True


@pytest.mark.asyncio
async def test_stream_uses_drafts_for_live_partials_and_final_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

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
    assert fake_client.draft_calls
    assert len({draft_id for _, draft_id, _, _, _ in fake_client.draft_calls}) == 1
    assert fake_client.message_calls == [(1, "I checked your inbox. 3 priority emails found.", "HTML")]
    assert fake_client.chat_actions
    assert not fake_client.deleted_messages


@pytest.mark.asyncio
async def test_wait_signal_does_not_emit_visible_progress_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

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
    assert not any("working on it" in text.lower() for _, _, text, _, _ in fake_client.draft_calls)
    assert fake_client.message_calls == [(1, "I checked the inbox. 3 priority emails found.", "HTML")]


@pytest.mark.asyncio
async def test_progress_signals_stay_in_typing_only_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

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
    assert not any("searching the web" in text.lower() for _, _, text, _, _ in fake_client.draft_calls)
    assert not any("fetching a webpage" in text.lower() for _, _, text, _, _ in fake_client.draft_calls)
    assert fake_client.message_calls == [(1, "Here is the result.", "HTML")]


@pytest.mark.asyncio
async def test_stream_throttles_rapid_partial_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "STREAM_EDIT_MIN_INTERVAL_SECONDS", 10.0)
    monkeypatch.setattr(relay_module, "STREAM_EDIT_MIN_CHAR_DELTA", 1000)
    monkeypatch.setattr(relay_module, "STREAM_INITIAL_VISIBLE_MIN_CHARS", 50)
    monkeypatch.setattr(relay_module, "STREAM_INITIAL_VISIBLE_MAX_WAIT_SECONDS", 10.0)

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
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [(1, "Hello world", "HTML")]


@pytest.mark.asyncio
async def test_stream_keeps_meaningful_chunking_for_followups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "STREAM_INITIAL_VISIBLE_MIN_CHARS", 10)
    monkeypatch.setattr(relay_module, "STREAM_INITIAL_VISIBLE_MAX_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(relay_module, "STREAM_EDIT_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(relay_module, "STREAM_FOLLOWUP_VISIBLE_MIN_CHARS", 20)
    monkeypatch.setattr(relay_module, "STREAM_EDIT_MIN_CHAR_DELTA", 200)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_WordByWordRuntime(),
        thread_id="chat-wordy",
        customer_id="telegram_wordy",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "This is a slightly longer streamed reply with enough words."
    assert fake_client.draft_calls[0][2] == "This is a"
    assert fake_client.draft_calls[-1][2] == "This is a slightly longer streamed reply with enough words."
    assert len(fake_client.draft_calls) < 8
    assert fake_client.message_calls == [(1, "This is a slightly longer streamed reply with enough words.", "HTML")]


@pytest.mark.asyncio
async def test_draft_failure_falls_back_to_typing_and_final_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ResultThenWaitRuntime(),
        thread_id="chat-fallback",
        customer_id="telegram_fallback",
        text="finish",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Here is the finished result."
    assert len(fake_client.draft_calls) == 1
    assert fake_client.message_calls == [(1, "Here is the finished result.", "HTML")]
    assert fake_client.chat_actions


@pytest.mark.asyncio
async def test_non_private_chat_bypasses_draft_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ResultThenWaitRuntime(),
        thread_id="chat-group",
        customer_id="telegram_group",
        text="finish",
        bot_token="dummy",
        chat_id=-100123456,
    )

    assert suppressed is False
    assert final == "Here is the finished result."
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [(-100123456, "Here is the finished result.", "HTML")]


@pytest.mark.asyncio
async def test_approval_handoff_stops_draft_streaming_without_final_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "STREAM_INITIAL_VISIBLE_MIN_CHARS", 1)
    monkeypatch.setattr(relay_module, "STREAM_INITIAL_VISIBLE_MAX_WAIT_SECONDS", 0.0)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_PartialThenApprovalRuntime(),
        thread_id="chat-approval",
        customer_id="telegram_approval",
        text="check",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is True
    assert final is None
    assert len(fake_client.draft_calls) == 1
    assert fake_client.message_calls == []
