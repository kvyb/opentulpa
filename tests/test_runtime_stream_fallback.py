from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest

from opentulpa.agent import runtime as runtime_module
from opentulpa.agent.context_compaction import ContextCompactionResult
from opentulpa.agent.lc_messages import AIMessage, HumanMessage
from opentulpa.agent.runtime import (
    STREAM_EMPTY_REPLY_FALLBACK,
    STREAM_PROGRESS_PREFIX,
    AgentStreamEvent,
    OpenTulpaLangGraphRuntime,
)
from opentulpa.agent.runtime_context_provider import RuntimeContextSourceProvider
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


class _StaleFallbackGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content=""), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {
            "messages": [
                HumanMessage(content="old user"),
                AIMessage(content="old assistant reply"),
                HumanMessage(content="current user"),
                AIMessage(content=""),
            ]
        }


def _install_turn_context_stubs(runtime: OpenTulpaLangGraphRuntime) -> None:
    async def _list_available_skills(customer_id: str) -> list[dict[str, Any]]:
        del customer_id
        return []

    async def _load_skill_context_by_names(
        *, customer_id: str, skill_names: list[str]
    ) -> dict[str, Any]:
        del customer_id, skill_names
        return {"skill_names": [], "context": ""}

    runtime._list_available_skills = _list_available_skills  # type: ignore[method-assign]
    runtime._load_skill_context_by_names = _load_skill_context_by_names  # type: ignore[method-assign]
    runtime._context_source_provider = RuntimeContextSourceProvider(runtime)


class _ProvisionalOnlyGraph:
    _TEXT = "I can also search for the ManyChat API documentation to get the exact payload shape."

    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content=self._TEXT), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {
            "final_response_text": self._TEXT,
            "messages": [HumanMessage(content="user"), AIMessage(content=self._TEXT)],
        }


class _ToolThenAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(content="Done checking. 3 priority emails found."), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


class _ProvisionalThenToolThenAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content="Sure! Let me find that for you."), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(content="Here is the final answer."), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


class _DraftThenToolThenAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(
            content="Let me check that for you.",
            tool_calls=[{"id": "call_1", "name": "composio_tool_execute", "args": {"query": "inbox"}}],
        ), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(content="I checked it. 3 priority emails found."), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


class _DraftThenToolThenStreamingAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "memory_search", "args": {"query": "context"}}],
        ), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(content="Hello"), {"langgraph_node": "agent"}
        yield AIMessage(content=" world"), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


class _DraftThenToolThenDraftToolThenAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "memory_search", "args": {"query": "context"}}],
        ), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(
            content="I need to check one more thing.",
            tool_calls=[{"id": "call_2", "name": "skill_get", "args": {"name": "followup"}}],
        ), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running again"), {"langgraph_node": "tools"}
        yield AIMessage(content="Final answer."), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


class _EarlyVisibleThenToolThenAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content="Проверяю подключение Google Sheets."), {"langgraph_node": "agent"}
        yield AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "composio_tool_search", "args": {"query": "append row"}}],
        ), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(content="Готово: Google Sheets подключён, прайс обработан."), {
            "langgraph_node": "agent"
        }

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [HumanMessage(content="user"), AIMessage(content="unused")]}


class _ReasoningThenToolThenAnswerGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "private reasoning must not stream"},
        ), {"langgraph_node": "agent"}
        yield AIMessage(
            content="",
            additional_kwargs={"reasoning_content": "second private reasoning chunk"},
        ), {"langgraph_node": "agent"}
        yield AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "private"}}],
        ), {"langgraph_node": "agent"}
        yield AIMessage(content="tool running"), {"langgraph_node": "tools"}
        yield AIMessage(content="Done."), {"langgraph_node": "agent"}

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

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

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

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

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

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-precommit",
        customer_id="telegram_precommit",
        text="schedule this",
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].startswith(STREAM_PROGRESS_PREFIX)
    assert chunks[1] == "I need one clarification before scheduling this."

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines if line.strip()]
    assert "turn_stream_precommit_discarded" in events
    assert "turn_stream_precommit_flushed" in events


@pytest.mark.asyncio
async def test_astream_text_does_not_reuse_stale_prior_ai_message_in_fallback(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _StaleFallbackGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_stale_fallback.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-stale-fallback",
        customer_id="telegram_stale",
        text="current user",
    ):
        chunks.append(chunk)

    assert chunks == [STREAM_EMPTY_REPLY_FALLBACK]


@pytest.mark.asyncio
async def test_astream_text_discards_provisional_only_reply_and_emits_fallback(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _ProvisionalOnlyGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_provisional_only.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-provisional-only",
        customer_id="telegram_provisional_only",
        text="manychat?",
    ):
        chunks.append(chunk)

    assert chunks == [STREAM_EMPTY_REPLY_FALLBACK]

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines if line.strip()]
    assert "turn_stream_precommit_discarded" in events
    assert "turn_stream_fallback_discarded" in events
    assert "turn_stream_fallback_empty" in events


@pytest.mark.asyncio
async def test_astream_text_emits_wait_signal_before_tool_first_result(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _ToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_tool_first_wait.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-tool-first",
        customer_id="telegram_tool_first",
        text="check inbox",
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].startswith(STREAM_PROGRESS_PREFIX)
    assert chunks[1] == "Done checking. 3 priority emails found."


@pytest.mark.asyncio
async def test_astream_text_drops_provisional_text_before_tool_with_zero_precommit(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _ProvisionalThenToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_zero_precommit_provisional.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-zero-precommit-tool",
        customer_id="telegram_zero_precommit_tool",
        text="find it",
        stream_precommit_seconds=0.0,
    ):
        chunks.append(chunk)

    assert chunks == [f"{STREAM_PROGRESS_PREFIX}Working on it…", "Here is the final answer."]


@pytest.mark.asyncio
async def test_astream_text_holds_agent_draft_when_segment_declares_tool_calls(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _DraftThenToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_tool_declared_buffer.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-tool-declared",
        customer_id="telegram_tool_declared",
        text="check inbox",
    ):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0].startswith(STREAM_PROGRESS_PREFIX)
    assert chunks[1] == "I checked it. 3 priority emails found."


@pytest.mark.asyncio
async def test_astream_text_streams_post_tool_incremental_chunks(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _DraftThenToolThenStreamingAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_post_tool_stream.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-post-tool-stream",
        customer_id="telegram_post_tool_stream",
        text="answer after context",
        stream_precommit_seconds=0.0,
        stream_incremental_deltas=True,
    ):
        chunks.append(chunk)

    assert len(chunks) == 3
    assert chunks[0].startswith(STREAM_PROGRESS_PREFIX)
    assert chunks[1:] == ["Hello", " world"]


@pytest.mark.asyncio
async def test_astream_text_keeps_second_tool_draft_buffered_after_tool_phase(
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _DraftThenToolThenDraftToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_second_tool_draft.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-second-tool-draft",
        customer_id="telegram_second_tool_draft",
        text="answer after two tool phases",
        stream_precommit_seconds=0.0,
        stream_incremental_deltas=True,
    ):
        chunks.append(chunk)

    visible = "\n".join(chunks)
    assert "I need to check one more thing." not in visible
    assert chunks[-1] == "Final answer."


@pytest.mark.asyncio
async def test_astream_text_flushes_post_tool_answer_after_early_visible_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime_module, "STREAM_PRECOMMIT_SECONDS", 0)
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _EarlyVisibleThenToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_post_tool_flush.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-post-tool-flush",
        customer_id="telegram_post_tool_flush",
        text="setup sheets",
    ):
        chunks.append(chunk)

    assert chunks[-1] == "Готово: Google Sheets подключён, прайс обработан."
    assert any(chunk.startswith(STREAM_PROGRESS_PREFIX) for chunk in chunks)

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines if line.strip()]
    assert "turn_stream_buffered_completion_flushed" in events

    assert "turn_stream_precommit_discarded" not in events


@pytest.mark.asyncio
async def test_astream_text_emits_safe_reasoning_and_tool_status_events(tmp_path) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _ReasoningThenToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_safe_status.jsonl"
    runtime._behavior_log_lock = threading.Lock()

    async def _noop_start() -> None:
        return None

    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str | AgentStreamEvent] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-safe-status",
        customer_id="telegram_safe_status",
        text="search",
        stream_status_events=True,
    ):
        chunks.append(chunk)

    events = [chunk for chunk in chunks if isinstance(chunk, AgentStreamEvent)]
    assert [event.event for event in events] == ["reasoning", "tool_call"]
    assert events[0].payload == {"status": "active", "message": "Reasoning..."}
    assert events[1].payload["tool_names"] == ["web_search"]
    assert events[1].payload["tool_call_count"] == 1
    assert "private reasoning" not in json.dumps([event.payload for event in events])
    assert "second private" not in json.dumps([event.payload for event in events])
    assert chunks[-1] == "Done."


@pytest.mark.asyncio
async def test_astream_text_emits_compaction_status_before_compacting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _ReasoningThenToolThenAnswerGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_compaction_status.jsonl"
    runtime._behavior_log_lock = threading.Lock()
    compact_calls: list[tuple[str, str]] = []

    async def _noop_start() -> None:
        return None

    async def _needs_compaction(_runtime: Any, *, thread_id: str) -> bool:
        assert _runtime is runtime
        assert thread_id == "chat-compaction-status"
        return True

    async def _compact(_runtime: Any, *, thread_id: str, customer_id: str) -> ContextCompactionResult:
        assert _runtime is runtime
        compact_calls.append((thread_id, customer_id))
        return ContextCompactionResult(status="compacted", reason="not_needed", attempts=1)

    monkeypatch.setattr(runtime_module, "thread_context_needs_compaction", _needs_compaction)
    monkeypatch.setattr(runtime_module, "compact_thread_context_for_turn", _compact)
    runtime.start = _noop_start  # type: ignore[method-assign]
    _install_turn_context_stubs(runtime)

    chunks: list[str | AgentStreamEvent] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-compaction-status",
        customer_id="telegram_compaction_status",
        text="search",
        stream_status_events=True,
    ):
        chunks.append(chunk)

    events = [chunk for chunk in chunks if isinstance(chunk, AgentStreamEvent)]
    assert events[0] == AgentStreamEvent(
        event="status",
        payload={"status": "active", "message": "Compacting chat history..."},
    )
    assert compact_calls == [("chat-compaction-status", "telegram_compaction_status")]
