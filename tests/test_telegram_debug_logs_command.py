from __future__ import annotations

from typing import Any

import pytest

from opentulpa.interfaces.telegram import chat_service as chat_module


class _FakeStateStore:
    def __init__(self, initial: dict[str, Any]) -> None:
        self.state = initial

    def update(self, mutator: Any) -> Any:
        return mutator(self.state)


@pytest.mark.asyncio
async def test_debug_logs_command_sends_log_file_without_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "read_debug_log_bytes", lambda: b"log line 1\nlog line 2\n")

    sent_payload: dict[str, Any] = {}

    class _FakeTelegramClient:
        def __init__(self, bot_token: str) -> None:
            sent_payload["bot_token"] = bot_token

        async def send_file(self, **kwargs: Any) -> bool:
            sent_payload.update(kwargs)
            return True

    monkeypatch.setattr(chat_module, "TelegramClient", _FakeTelegramClient)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 99}, "from": {"id": 100}, "text": "/debug_logs"}},
        bot_token="123:abc",
        agent_runtime=None,
    )

    assert text is None
    assert sent_payload["bot_token"] == "123:abc"
    assert sent_payload["chat_id"] == 99
    assert sent_payload["filename"] == "app.log"
    assert sent_payload["raw_bytes"] == b"log line 1\nlog line 2\n"
    assert sent_payload["mime_type"] == "text/plain"


@pytest.mark.asyncio
async def test_debug_logs_command_reports_missing_log_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "read_debug_log_bytes", lambda: None)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 99}, "from": {"id": 100}, "text": "/debug_logs"}},
        bot_token="123:abc",
        agent_runtime=None,
    )

    assert text == "Debug log file is not available yet."
