from __future__ import annotations

import os
from typing import Any

import pytest

from opentulpa.interfaces.telegram import chat_service as chat_module


class _FakeStateStore:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.state = initial

    def update(self, mutator: Any) -> Any:
        return mutator(self.state)


@pytest.mark.asyncio
async def test_fresh_rotates_thread_and_keeps_customer_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_store = _FakeStateStore(
        {
            "admin_user_id": 100,
            "pending_key_by_chat": {"1": "OPENROUTER_API_KEY"},
            "sessions": {
                "1": {
                    "user_id": 100,
                    "customer_id": "telegram_100",
                    "thread_id": "chat-1",
                    "wake_thread_id": "wake_old",
                    "last_user_message_at": "2026-01-01T00:00:00+00:00",
                    "last_assistant_message_at": "2026-01-01T00:00:00+00:00",
                }
            },
        }
    )
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 1}, "from": {"id": 100}, "text": "/fresh"}},
        bot_token=None,
        agent_runtime=None,
    )

    assert isinstance(text, str)
    assert "fresh chat context" in text.lower()
    slot = fake_store.state["sessions"]["1"]
    assert str(slot.get("customer_id")) == "telegram_100"
    assert str(slot.get("thread_id", "")).startswith("chat_")
    assert str(slot.get("wake_thread_id", "")).startswith("wake_")
    assert fake_store.state["pending_key_by_chat"].get("1") is None


@pytest.mark.asyncio
async def test_fresh_creates_session_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_store = _FakeStateStore({"admin_user_id": None, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 7}, "from": {"id": 42}, "text": "/fresh"}},
        bot_token=None,
        agent_runtime=None,
    )

    assert isinstance(text, str)
    slot = fake_store.state["sessions"]["7"]
    assert str(slot.get("customer_id")) == "telegram_42"
    assert str(slot.get("thread_id", "")).startswith("chat_")
    assert str(slot.get("wake_thread_id", "")).startswith("wake_")


@pytest.mark.asyncio
async def test_removed_setup_and_set_commands_do_not_write_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_API_KEY", raising=False)

    setup_text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 1}, "from": {"id": 100}, "text": "/setup"}},
        bot_token=None,
        agent_runtime=None,
    )
    set_text = await chat_module.handle_telegram_text(
        body={
            "message": {
                "chat": {"id": 1},
                "from": {"id": 100},
                "text": "/set OPENAI_COMPATIBLE_API_KEY secret",
            }
        },
        bot_token=None,
        agent_runtime=None,
    )

    assert "OPENAI_COMPATIBLE_API_KEY" in str(setup_text)
    assert "OPENAI_COMPATIBLE_API_KEY" in str(set_text)
    assert os.environ.get("OPENROUTER_API_KEY") is None
    assert os.environ.get("OPENAI_COMPATIBLE_API_KEY") is None
