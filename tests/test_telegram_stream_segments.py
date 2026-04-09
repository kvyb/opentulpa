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


class _DraftThenLongPauseRuntime:
    async def astream_text(self, **kwargs):
        yield "This first visible draft is long enough to publish immediately."
        await asyncio.sleep(4.2)
        yield "This first visible draft is long enough to publish immediately. And here is the completed follow-up chunk."


class _PacedChunkRuntime:
    async def astream_text(self, **kwargs):
        yield "Chunk one."
        await asyncio.sleep(1.0)
        yield "Chunk one. Chunk two."
        await asyncio.sleep(1.0)
        yield "Chunk one. Chunk two. Chunk three."


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
async def test_stream_uses_drafts_for_live_partials_without_separate_final_send(
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
    assert fake_client.message_calls == []
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
    assert fake_client.message_calls == []


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
    assert fake_client.message_calls == []


@pytest.mark.asyncio
async def test_stream_coalesces_rapid_partial_updates_until_final_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

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
    assert [text for _, _, text, _, _ in fake_client.draft_calls] == ["Hello world"]
    assert fake_client.message_calls == []


@pytest.mark.asyncio
async def test_stream_paces_draft_updates_by_time_not_by_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_PacedChunkRuntime(),
        thread_id="chat-wordy",
        customer_id="telegram_wordy",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Chunk one. Chunk two. Chunk three."
    assert [text for _, _, text, _, _ in fake_client.draft_calls] == [
        "Chunk one. Chunk two.",
        "Chunk one. Chunk two. Chunk three.",
    ]
    assert fake_client.message_calls == []


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
async def test_successful_draft_stream_stops_typing_loop_after_first_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_DraftThenLongPauseRuntime(),
        thread_id="chat-draft-stop",
        customer_id="telegram_draft_stop",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "completed follow-up chunk" in str(final or "")
    assert fake_client.draft_calls
    assert 1 <= len(fake_client.chat_actions) <= 2
    assert fake_client.message_calls == []


@pytest.mark.asyncio
async def test_failed_draft_stream_also_stops_typing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_DraftThenLongPauseRuntime(),
        thread_id="chat-draft-fail-stop",
        customer_id="telegram_draft_fail_stop",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "completed follow-up chunk" in str(final or "")
    assert fake_client.draft_calls
    assert 1 <= len(fake_client.chat_actions) <= 2


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
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == []
