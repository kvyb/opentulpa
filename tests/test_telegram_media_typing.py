from __future__ import annotations

import asyncio
from typing import Any

import pytest

from opentulpa.core.shutdown_drain import ShutdownDrain
from opentulpa.interfaces.telegram import chat_service as chat_module


class _FakeStateStore:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.state = initial

    def update(self, mutator: Any) -> Any:
        return mutator(self.state)

    def touch_assistant_message(self, _chat_id: int) -> None:
        return None


class _FakeTelegramClient:
    calls: list[tuple[int | str, str]] = []

    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
    ) -> bool:
        self.calls.append((chat_id, action))
        return True


@pytest.mark.asyncio
async def test_media_ingestion_emits_typing_indicator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "TelegramClient", _FakeTelegramClient)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)
    monkeypatch.setattr(chat_module, "extract_attachments", lambda _message: [object()])

    async def _fake_ingest_attachments(**kwargs: Any) -> list[dict[str, Any]]:
        _ = kwargs
        await asyncio.sleep(0.02)
        return [{"kind": "document", "summary": "ok"}]

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        _ = kwargs
        return "done", False

    monkeypatch.setattr(chat_module, "ingest_attachments", _fake_ingest_attachments)
    monkeypatch.setattr(chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram)

    reply = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 1}, "from": {"id": 100}, "document": {"file_id": "doc1"}}},
        bot_token="123:abc",
        agent_runtime=object(),
        file_vault=object(),
        memory=None,
    )

    assert reply is None
    assert _FakeTelegramClient.calls
    assert all(action == "typing" for _, action in _FakeTelegramClient.calls)


@pytest.mark.asyncio
async def test_telegram_stream_turn_blocks_shutdown_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "get_openai_compatible_api_key_from_env", lambda: "key")
    monkeypatch.setattr(chat_module, "is_user_allowed", lambda **kwargs: True)
    monkeypatch.setattr(chat_module, "extract_attachments", lambda _message: [])

    started = asyncio.Event()
    release = asyncio.Event()

    async def _fake_stream_langgraph_reply_to_telegram(**kwargs: Any) -> tuple[str | None, bool]:
        _ = kwargs
        started.set()
        await release.wait()
        return "done", False

    monkeypatch.setattr(chat_module, "stream_langgraph_reply_to_telegram", _fake_stream_langgraph_reply_to_telegram)
    drain = ShutdownDrain(timeout_seconds=1)
    touch_active_turns: list[int] = []

    def _touch_assistant_message(_chat_id: int) -> None:
        touch_active_turns.append(drain.status().active_turns)

    fake_store.touch_assistant_message = _touch_assistant_message  # type: ignore[method-assign]

    turn_task = asyncio.create_task(
        chat_module.handle_telegram_text(
            body={"message": {"chat": {"id": 1}, "from": {"id": 100}, "text": "hi"}},
            bot_token="123:abc",
            agent_runtime=object(),
            shutdown_drain=drain,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    drain_task = asyncio.create_task(drain.drain())
    await asyncio.sleep(0)
    assert not drain_task.done()

    release.set()

    assert await asyncio.wait_for(turn_task, timeout=1) is None
    assert await asyncio.wait_for(drain_task, timeout=1) is True
    assert drain.status().active_turns == 0
    assert touch_active_turns == [1]
