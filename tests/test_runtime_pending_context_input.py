from __future__ import annotations

from typing import Any

import pytest

from opentulpa.agent.lc_messages import AIMessage, HumanMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.runtime_input import ThreadInputCoordinator


class _CapturingGraph:
    def __init__(self) -> None:
        self.last_state: dict[str, Any] | None = None

    async def ainvoke(self, state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        self.last_state = state
        return {"messages": [AIMessage(content="ok")]}


class _FakeContextEvents:
    def __init__(self) -> None:
        self.cleared: tuple[str, int | None] | None = None

    def list_events(self, customer_id: str, limit: int = 20) -> list[dict[str, Any]]:
        del customer_id, limit
        return [
            {
                "id": 42,
                "source": "approval",
                "event_type": "executed",
                "payload": {
                    "approval_id": "apr_abc",
                    "status": "approved",
                    "raw_prompt": "I want you to scan my telegram Work folder",
                },
            }
        ]

    def clear_events(self, customer_id: str, *, through_id: int | None = None) -> int:
        self.cleared = (customer_id, through_id)
        return 1


@pytest.mark.asyncio
async def test_pending_context_is_not_merged_into_user_message() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    graph = _CapturingGraph()
    events = _FakeContextEvents()

    runtime._graph = graph
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = events
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.expand_link_aliases = lambda **kwargs: str(kwargs.get("text", ""))  # type: ignore[assignment]

    captured_skill_user_text: dict[str, str] = {}

    async def _noop_start() -> None:
        return None

    async def _noop_compact(*, thread_id: str, customer_id: str) -> None:
        del thread_id, customer_id
        return None

    async def _no_pending_lock(*, customer_id: str, thread_id: str) -> bool:
        del customer_id, thread_id
        return False

    async def _capture_skill_state(*, customer_id: str, user_text: str) -> dict[str, Any]:
        del customer_id
        captured_skill_user_text["value"] = user_text
        return {}

    runtime.start = _noop_start  # type: ignore[method-assign]
    runtime._maybe_compact_thread_context = _noop_compact  # type: ignore[method-assign]
    runtime._has_pending_approval_lock = _no_pending_lock  # type: ignore[method-assign]
    runtime._pre_resolve_skill_state = _capture_skill_state  # type: ignore[method-assign]

    user_text = "can you try again?"
    reply = await runtime.ainvoke_text(
        thread_id="chat_test",
        customer_id="telegram_test",
        text=user_text,
        include_pending_context=True,
    )

    assert reply == "ok"
    assert captured_skill_user_text["value"] == user_text
    assert graph.last_state is not None
    model_messages = graph.last_state["messages"]
    assert len(model_messages) == 1
    assert isinstance(model_messages[0], HumanMessage)
    assert model_messages[0].content == user_text
    pending_text = str(graph.last_state.get("pending_context_summary", ""))
    assert "approval_id=apr_abc" in pending_text
    assert "scan my telegram" not in pending_text
    assert "raw_prompt" not in pending_text
    assert events.cleared == ("telegram_test", 42)
