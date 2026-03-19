from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.application.wake_orchestrator import WakeOrchestrator


class _FakeContextEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def add_event(self, **kwargs: Any) -> int:
        self.events.append(kwargs)
        return len(self.events)


class _FakeTelegramChat:
    def __init__(self) -> None:
        self.touched: list[int] = []
        self.slots: list[dict[str, Any]] = [{"chat_id": 166}]

    async def relay_event(self, **_: Any) -> list[dict[str, Any]]:
        return [{"chat_id": 166, "text": "wake update"}]

    async def relay_task_event(self, **_: Any) -> list[dict[str, Any]]:
        return [{"chat_id": 166, "text": "task update"}]

    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        del customer_id
        return self.slots

    def touch_assistant_message(self, chat_id: int) -> None:
        self.touched.append(int(chat_id))


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> bool:
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
        )
        return True


class _FakeApprovals:
    def __init__(self) -> None:
        self.flush_calls: list[dict[str, str]] = []

    async def flush_deferred_challenges(
        self,
        *,
        origin_interface: str,
        origin_conversation_id: str,
    ) -> int:
        self.flush_calls.append(
            {
                "origin_interface": origin_interface,
                "origin_conversation_id": origin_conversation_id,
            }
        )
        return 1


class _FakeRuntime:
    def __init__(self, result: str = "wake update") -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def ainvoke_text(
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
        **_: Any,
    ) -> str:
        self.calls.append(
            {
                "thread_id": thread_id,
                "customer_id": customer_id,
                "text": text,
                "turn_mode": turn_mode,
                "include_pending_context": include_pending_context,
            }
        )
        return self.result


@pytest.mark.asyncio
async def test_routine_event_flushes_deferred_approval_challenges() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    approvals = _FakeApprovals()
    runtime = _FakeRuntime(result="routine done")

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=lambda: approvals,
    )

    await orchestrator.handle_event(
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "customer_id": "telegram_166",
            "routine_id": "rtn_123",
            "routine_name": "Test Routine",
            "notify_user": True,
            "payload": {
                "customer_id": "telegram_166",
                "notify_user": True,
                "instruction": (
                    "You must run scripts/test_routine.py. First read file tulpa_stuff/input.txt. "
                    "Then write output to tulpa_stuff/output.txt. "
                    "If file read fails, log error and return failure summary."
                ),
            },
        }
    )

    assert approvals.flush_calls == [
        {
            "origin_interface": "telegram",
            "origin_conversation_id": "166",
        }
    ]
    assert client.sent
    assert runtime.calls
    assert runtime.calls[0]["turn_mode"] == "routine_wake"


@pytest.mark.asyncio
async def test_routine_event_silent_mode_still_executes_and_backlogs() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="updated timelog successfully")

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=None,
    )

    await orchestrator.handle_event(
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "customer_id": "telegram_166",
            "routine_id": "rtn_abc",
            "routine_name": "Timelog Updater",
            "notify_user": False,
            "payload": {
                "customer_id": "telegram_166",
                "notify_user": False,
                "instruction": "Append current timestamp to tulpa_stuff/timelog.md",
            },
        }
    )

    assert runtime.calls
    assert runtime.calls[0]["turn_mode"] == "routine_wake"
    assert not client.sent
    assert context_events.events
    queued = context_events.events[-1]
    assert queued["source"] == "routine"
    assert queued["event_type"] == "scheduled"
    payload = queued["payload"]
    assert payload["execution_status"] == "executed"
    assert "updated timelog" in payload["execution_summary"]


@pytest.mark.asyncio
async def test_routine_event_missing_instruction_fails_invalid() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="should not run")

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=None,
    )

    await orchestrator.handle_event(
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "customer_id": "telegram_166",
            "routine_id": "rtn_missing",
            "routine_name": "Broken Routine",
            "notify_user": True,
            "payload": {
                "customer_id": "telegram_166",
                "notify_user": True,
            },
        }
    )

    assert not runtime.calls
    assert not client.sent
    assert context_events.events
    payload = context_events.events[-1]["payload"]
    assert payload["execution_status"] == "invalid"
    assert "missing required instruction" in payload["execution_error"]
