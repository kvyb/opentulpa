from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest

from opentulpa.agent.lc_messages import AIMessage, HumanMessage
from opentulpa.agent.runtime import STREAM_EMPTY_REPLY_FALLBACK, OpenTulpaLangGraphRuntime
from opentulpa.agent.runtime_input import ThreadInputCoordinator


class _NoVisibleOutputGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        # Agent emitted an empty assistant message, which used to end silently.
        yield AIMessage(content=""), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        # Fallback path also fails to provide any visible AI content.
        return {"messages": [HumanMessage(content="user"), AIMessage(content="")]}


class _BufferedRepairGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content="I scheduled it for 22:49."), {"langgraph_node": "agent"}
        yield AIMessage(content="ACTION_CLARIFICATION_REQUIRED"), {"langgraph_node": "tools"}
        yield AIMessage(
            content="I need one clarification before scheduling this."
        ), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


@pytest.mark.asyncio
async def test_astream_text_emits_fallback_when_no_visible_output(tmp_path) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _NoVisibleOutputGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    async def _noop_compact(*, thread_id: str, customer_id: str) -> None:
        del thread_id, customer_id
        return None

    async def _noop_skills(*, customer_id: str, user_text: str) -> dict[str, Any]:
        del customer_id, user_text
        return {}

    runtime.start = _noop_start  # type: ignore[method-assign]
    runtime._maybe_compact_thread_context = _noop_compact  # type: ignore[method-assign]
    runtime._pre_resolve_skill_state = _noop_skills  # type: ignore[method-assign]

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-test",
        customer_id="telegram_test",
        text="hello",
    ):
        chunks.append(chunk)

    assert chunks
    assert chunks[-1] == STREAM_EMPTY_REPLY_FALLBACK

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = {json.loads(line)["event"] for line in lines if line.strip()}
    assert "turn_start" in events
    assert "turn_stream_no_visible_chunks" in events
    assert "turn_stream_fallback_empty" in events
    assert "turn_complete" in events


@pytest.mark.asyncio
async def test_astream_text_logs_no_visible_progress_timeout(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_STREAM_NO_VISIBLE_PROGRESS_SECONDS", "0.001")

    class _SlowNoVisibleOutputGraph:
        async def astream(
            self,
            _state: dict[str, Any],
            *,
            config: dict[str, Any],
            stream_mode: str,
        ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
            del config, stream_mode
            await asyncio.sleep(0.01)
            yield AIMessage(content=""), {"langgraph_node": "agent"}

        async def ainvoke(
            self, _state: dict[str, Any], *, config: dict[str, Any]
        ) -> dict[str, Any]:
            del config
            return {"messages": [HumanMessage(content="user"), AIMessage(content="")]}

    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _SlowNoVisibleOutputGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_timeout.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    async def _noop_compact(*, thread_id: str, customer_id: str) -> None:
        del thread_id, customer_id
        return None

    async def _noop_skills(*, customer_id: str, user_text: str) -> dict[str, Any]:
        del customer_id, user_text
        return {}

    runtime.start = _noop_start  # type: ignore[method-assign]
    runtime._maybe_compact_thread_context = _noop_compact  # type: ignore[method-assign]
    runtime._pre_resolve_skill_state = _noop_skills  # type: ignore[method-assign]

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-timeout",
        customer_id="telegram_timeout",
        text="hello",
    ):
        chunks.append(chunk)

    assert chunks
    assert chunks[-1] == STREAM_EMPTY_REPLY_FALLBACK

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = {json.loads(line)["event"] for line in lines if line.strip()}
    assert "turn_stream_no_visible_progress_timeout" in events
    assert "turn_stream_no_visible_chunks" in events


@pytest.mark.asyncio
async def test_astream_text_holds_early_schedule_claim_until_repair_finishes(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _BufferedRepairGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_precommit.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    async def _noop_compact(*, thread_id: str, customer_id: str) -> None:
        del thread_id, customer_id
        return None

    async def _noop_skills(*, customer_id: str, user_text: str) -> dict[str, Any]:
        del customer_id, user_text
        return {}

    runtime.start = _noop_start  # type: ignore[method-assign]
    runtime._maybe_compact_thread_context = _noop_compact  # type: ignore[method-assign]
    runtime._pre_resolve_skill_state = _noop_skills  # type: ignore[method-assign]

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-precommit",
        customer_id="telegram_precommit",
        text="schedule this",
    ):
        chunks.append(chunk)

    assert chunks == ["I need one clarification before scheduling this."]

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines if line.strip()]
    assert "turn_stream_precommit_discarded" in events
    assert "turn_stream_precommit_flushed" in events
