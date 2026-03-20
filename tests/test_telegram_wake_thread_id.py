from __future__ import annotations

from typing import Any

import pytest

from opentulpa.interfaces.telegram import relay as relay_module


class _FakeStateStore:
    def __init__(self, initial_wake_thread_id: Any) -> None:
        self.state: dict[str, Any] = {
            "sessions": {
                "1": {
                    "user_id": 1,
                    "customer_id": "telegram_1",
                    "thread_id": "chat-1",
                    "wake_thread_id": initial_wake_thread_id,
                }
            }
        }

    def update(self, mutator: Any) -> Any:
        return mutator(self.state)


class _Runtime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke_text(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "wake message"


def _find_slots(state_store: _FakeStateStore, customer_id: str) -> list[dict[str, Any]]:
    slot = state_store.state["sessions"]["1"]
    return [
        {
            "chat_id": 1,
            "user_id": slot.get("user_id"),
            "thread_id": slot.get("thread_id"),
            "wake_thread_id": slot.get("wake_thread_id"),
            "customer_id": customer_id,
            "last_user_message_at": "",
            "last_assistant_message_at": "",
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_wake_thread_id", [None, "None", "chat-1", "wake-legacy"])
async def test_relay_event_regenerates_invalid_wake_thread_id(initial_wake_thread_id: Any) -> None:
    state_store = _FakeStateStore(initial_wake_thread_id=initial_wake_thread_id)
    runtime = _Runtime()

    replies = await relay_module.relay_event_via_main_agent(
        customer_id="telegram_1",
        event_label="routine/scheduled",
        payload={"routine_name": "autopost", "payload": {"instruction": "You must send update."}},
        state_store=state_store,
        find_session_slots=lambda cid: _find_slots(state_store, cid),
        agent_runtime=runtime,
    )

    assert replies == [{"chat_id": 1, "text": "wake message"}]
    assert runtime.calls
    used_thread_id = str(runtime.calls[0].get("thread_id", ""))
    assert used_thread_id.startswith("wake_")
    assert runtime.calls[0]["turn_mode"] == "event_notification"
    assert state_store.state["sessions"]["1"]["wake_thread_id"] == used_thread_id


@pytest.mark.asyncio
async def test_relay_event_keeps_existing_wake_thread_id() -> None:
    state_store = _FakeStateStore(initial_wake_thread_id="wake_abcd12")
    runtime = _Runtime()

    replies = await relay_module.relay_event_via_main_agent(
        customer_id="telegram_1",
        event_label="routine/scheduled",
        payload={"routine_name": "autopost", "payload": {"instruction": "You must send update."}},
        state_store=state_store,
        find_session_slots=lambda cid: _find_slots(state_store, cid),
        agent_runtime=runtime,
    )

    assert replies == [{"chat_id": 1, "text": "wake message"}]
    assert runtime.calls
    used_thread_id = str(runtime.calls[0].get("thread_id", ""))
    assert used_thread_id == "wake_abcd12"
    assert runtime.calls[0]["turn_mode"] == "event_notification"
    assert state_store.state["sessions"]["1"]["wake_thread_id"] == "wake_abcd12"
