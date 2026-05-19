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
                **_,
            }
        )
        return self.result


class _FakeIntakeWorkflows:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def run_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        event_type: str = "scheduled",
        force: bool = False,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "customer_id": customer_id,
                "workflow_id": workflow_id,
                "event_type": event_type,
                "force": force,
            }
        )
        return self.result


@pytest.mark.asyncio
async def test_routine_event_notifies_and_records_execution() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="routine done")

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
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

    assert client.sent
    assert runtime.calls
    assert runtime.calls[0]["turn_mode"] == "routine_wake"
    assert context_events.events
    payload = context_events.events[-1]["payload"]
    assert payload["routine_id"] == "rtn_123"
    assert payload["execution_status"] == "executed"
    assert payload["execution_summary"] == "routine done"
    assert payload["notification_status"] == "sent"


@pytest.mark.asyncio
async def test_routine_event_notifies_direct_owner_chats_only() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    chat.slots = [
        {"chat_id": 166, "role": "owner"},
        {"chat_id": -1003941778604, "role": "owner"},
        {"chat_id": 9900, "role": "support"},
    ]
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="routine done")

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
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
                "instruction": "Check for AI news and summarize it.",
            },
        }
    )

    assert [item["chat_id"] for item in client.sent] == [166]
    payload = context_events.events[-1]["payload"]
    assert payload["notified_chat_ids"] == [166]


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
    assert payload["notification_status"] == "skipped"
    assert "updated timelog" in payload["execution_summary"]


@pytest.mark.asyncio
async def test_routine_event_uses_compact_literal_chat_wake_prompt() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="done")

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
    )

    await orchestrator.handle_event(
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "customer_id": "telegram_166",
            "routine_id": "rtn_compact",
            "routine_name": "Compact Routine",
            "notify_user": False,
            "payload": {
                "customer_id": "telegram_166",
                "notify_user": False,
                "instruction": "Check for updates and summarize them.",
                "source": "instagram",
                "large_blob": "x" * 3000,
            },
        }
    )

    assert runtime.calls
    call = runtime.calls[0]
    assert call["turn_mode"] == "routine_wake"
    assert call["prompt_mode_override"] == "literal_chat"
    assert call["thread_id"].startswith("routine_rtn_compact_wake_")
    assert "- payload_summary:" in call["text"]
    assert "- payload: {" not in call["text"]
    assert '"customer_id": "telegram_166"' in call["text"]
    assert '"instruction"' not in call["text"]
    assert ("x" * 1500) not in call["text"]


@pytest.mark.asyncio
async def test_intake_workflow_routine_uses_intake_runner_and_skips_runtime() -> None:
    settings = SimpleNamespace(telegram_bot_token="test-token")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="should not run")
    intake = _FakeIntakeWorkflows(
        {
            "ok": True,
            "summary": "Booking saved for Car Wash Intake: contact=alice booking_id=bkg_123 sink=local_csv",
        }
    )

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_intake_workflows=lambda: intake,
    )

    await orchestrator.handle_event(
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "customer_id": "telegram_166",
            "routine_id": "rtn_ig",
            "routine_name": "Car Wash Intake",
            "notify_user": True,
            "payload": {
                "customer_id": "telegram_166",
                "notify_user": True,
                "workflow_type": "intake_workflow",
                "workflow_id": "iwf_123",
                "instruction": "run intake workflow",
            },
        }
    )

    assert intake.calls == [
        {
            "customer_id": "telegram_166",
            "workflow_id": "iwf_123",
            "event_type": "scheduled",
            "force": False,
        }
    ]
    assert runtime.calls == []
    assert client.sent[0]["text"].startswith("Booking saved for Car Wash Intake:")


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
