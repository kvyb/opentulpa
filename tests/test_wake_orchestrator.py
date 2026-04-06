from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from opentulpa.application.wake_orchestrator import WakeOrchestrator
from opentulpa.context.signals import SignalInboxService
from opentulpa.skills.service import SkillStoreService, build_skill_markdown


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


def _mk_signal_skill_store(tmp_path: Path, *, customer_id: str, name: str) -> SkillStoreService:
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    store.upsert_skill(
        scope="user",
        customer_id=customer_id,
        name=name,
        skill_markdown=build_skill_markdown(
            name=name,
            description="Handle manychat contacts.",
            instructions=(
                "Ask concise follow-up questions, capture booking details, and reply naturally."
            ),
        ),
        source="test",
        enabled=True,
    )
    return store


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


@pytest.mark.asyncio
async def test_signal_event_creates_outbox_reply(tmp_path) -> None:
    settings = SimpleNamespace(telegram_bot_token="")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="We are open from 9 to 5.")
    signals = SignalInboxService(db_path=tmp_path / "signals.db")
    skills = _mk_signal_skill_store(tmp_path, customer_id="mc_123", name="manychat-incoming-handler")
    signals.upsert_rule(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        wake_mode="always",
        batch_window_seconds=0,
        auto_reply=True,
        handler_skill_name="manychat-incoming-handler",
        guidance_text="Use business_info.md for answers.",
    )
    signals.ingest_signal(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        text="What are your business hours?",
        dispatch={"conversation_id": "conv_1"},
    )

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=None,
        get_signal_inbox=lambda: signals,
        get_skill_store=lambda: skills,
    )

    await orchestrator.handle_event(
        {
            "type": "signal_event",
            "source": "manychat",
            "customer_id": "mc_123",
            "thread_id": "chat-mc_123",
        }
    )

    assert runtime.calls
    assert runtime.calls[0]["turn_mode"] == "interactive"
    assert "External messages/signals arrived" in runtime.calls[0]["text"]
    assert "handler_skill=manychat-incoming-handler" in runtime.calls[0]["text"]
    outbox = signals.list_outbox(source="manychat")
    assert len(outbox) == 1
    assert outbox[0]["text"] == "We are open from 9 to 5."
    assert outbox[0]["dispatch"]["conversation_id"] == "conv_1"
    assert not context_events.events


@pytest.mark.asyncio
async def test_signal_event_without_wired_handler_playbook_backlogs_and_does_not_reply(tmp_path) -> None:
    settings = SimpleNamespace(telegram_bot_token="")
    context_events = _FakeContextEvents()
    chat = _FakeTelegramChat()
    client = _FakeTelegramClient()
    runtime = _FakeRuntime(result="Should not be used.")
    signals = SignalInboxService(db_path=tmp_path / "signals.db")
    signals.upsert_rule(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        wake_mode="always",
        batch_window_seconds=0,
        auto_reply=True,
        guidance_text="Use business_info.md for answers.",
    )
    signals.ingest_signal(
        source="manychat",
        customer_id="mc_123",
        thread_id="chat-mc_123",
        text="What are your business hours?",
        dispatch={"conversation_id": "conv_1"},
    )

    orchestrator = WakeOrchestrator(
        settings=settings,
        get_context_events=lambda: context_events,
        get_telegram_chat=lambda: chat,
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=None,
        get_signal_inbox=lambda: signals,
        get_skill_store=lambda: _mk_signal_skill_store(
            tmp_path, customer_id="mc_123", name="other-handler"
        ),
    )

    await orchestrator.handle_event(
        {
            "type": "signal_event",
            "source": "manychat",
            "customer_id": "mc_123",
            "thread_id": "chat-mc_123",
        }
    )

    assert not runtime.calls
    assert signals.list_outbox(source="manychat") == []
    assert context_events.events
    assert context_events.events[-1]["event_type"] == "missing_handler_playbook"
