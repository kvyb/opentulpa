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
    tmp_path,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    first_log = tmp_path / "debug.log"
    first_log.write_bytes(b"log line 1\nlog line 2\n")
    second_log = tmp_path / "agent_behavior.jsonl"
    second_log.write_bytes(b'{"ok":true}\n')
    monkeypatch.setattr(chat_module, "iter_available_debug_log_paths", lambda: [first_log, second_log])

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
    assert sent_payloads == []
    assert len(sent_groups) == 1
    assert sent_groups[0]["bot_token"] == "123:abc"
    assert sent_groups[0]["chat_id"] == 99
    assert sent_groups[0]["caption"] == "OpenTulpa debug logs dump"
    assert sent_groups[0]["parse_mode"] == "HTML"
    assert [item["filename"] for item in sent_groups[0]["files"]] == ["debug.log", "agent_behavior.jsonl"]
    assert sent_groups[0]["files"][0]["raw_bytes"] == b"log line 1\nlog line 2\n"
    assert sent_groups[0]["files"][1]["raw_bytes"] == b'{"ok":true}\n'


@pytest.mark.asyncio
async def test_debug_logs_command_reports_missing_log_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    monkeypatch.setattr(chat_module, "iter_available_debug_log_paths", lambda: [])

    text = await chat_module.handle_telegram_text(
        body={"message": {"chat": {"id": 99}, "from": {"id": 100}, "text": "/debug_logs"}},
        bot_token="123:abc",
        agent_runtime=None,
    )

    assert text == "Debug log file is not available yet."


@pytest.mark.asyncio
async def test_debug_logs_command_sends_single_file_without_media_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    fake_store = _FakeStateStore({"admin_user_id": 100, "pending_key_by_chat": {}, "sessions": {}})
    monkeypatch.setattr(chat_module, "STATE_STORE", fake_store)
    only_log = tmp_path / "debug.log"
    only_log.write_bytes(b"one\n")
    monkeypatch.setattr(chat_module, "iter_available_debug_log_paths", lambda: [only_log])

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
    assert sent_payloads[0]["filename"] == "debug.log"
    assert sent_groups == []
