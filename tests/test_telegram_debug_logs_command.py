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
async def test_debug_logs_command_sends_last_7_days_archive_without_agent_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(
        chat_module,
        "build_debug_logs_archive_bytes",
        lambda *, lookback_days: ("debug-logs.zip", b"zip-bytes"),
    )

    sent_payloads: list[dict[str, Any]] = []
    sent_groups: list[dict[str, Any]] = []

    class _FakeTelegramClient:
        def __init__(self, bot_token: str) -> None:
            self._bot_token = bot_token

        async def send_file(self, **kwargs: Any) -> bool:
            payload = {"bot_token": self._bot_token}
            payload.update(kwargs)
            sent_payloads.append(payload)
            return True

        async def send_files(self, **kwargs: Any) -> bool:
            payload = {"bot_token": self._bot_token}
            payload.update(kwargs)
            sent_groups.append(payload)
            return True

    monkeypatch.setattr(chat_module, "TelegramClient", _FakeTelegramClient)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 99}, "from": {"id": 100}, "text": "/debug_logs"}},
        bot_token="123:abc",
        agent_runtime=None,
    )

    assert text is None
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["bot_token"] == "123:abc"
    assert sent_payloads[0]["chat_id"] == 99
    assert sent_payloads[0]["filename"] == "debug-logs.zip"
    assert sent_payloads[0]["raw_bytes"] == b"zip-bytes"
    assert sent_payloads[0]["mime_type"] == "application/zip"
    assert sent_payloads[0]["caption"] == "OpenTulpa debug logs dump (last 7 days)"
    assert sent_payloads[0]["parse_mode"] == "HTML"
    assert sent_groups == []


@pytest.mark.asyncio
async def test_debug_logs_command_reports_missing_log_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "build_debug_logs_archive_bytes", lambda *, lookback_days: None)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 99}, "from": {"id": 100}, "text": "/debug_logs"}},
        bot_token="123:abc",
        agent_runtime=None,
    )

    assert text == "Debug log file is not available yet."


@pytest.mark.asyncio
async def test_debug_logs_command_reports_file_send_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(
        chat_module,
        "build_debug_logs_archive_bytes",
        lambda *, lookback_days: ("debug-logs.zip", b"zip-bytes"),
    )

    sent_payloads: list[dict[str, Any]] = []
    sent_groups: list[dict[str, Any]] = []

    class _FakeTelegramClient:
        def __init__(self, bot_token: str) -> None:
            self._bot_token = bot_token

        async def send_file(self, **kwargs: Any) -> bool:
            payload = {"bot_token": self._bot_token}
            payload.update(kwargs)
            sent_payloads.append(payload)
            return False

        async def send_files(self, **kwargs: Any) -> bool:
            payload = {"bot_token": self._bot_token}
            payload.update(kwargs)
            sent_groups.append(payload)
            return True

    monkeypatch.setattr(chat_module, "TelegramClient", _FakeTelegramClient)

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 99}, "from": {"id": 100}, "text": "/debug_logs"}},
        bot_token="123:abc",
        agent_runtime=None,
    )

    assert text == "I couldn't send the debug log files right now."
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["filename"] == "debug-logs.zip"
    assert sent_groups == []
