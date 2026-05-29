from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest

from opentulpa.agent.lc_messages import AIMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.runtime_context_provider import RuntimeContextSourceProvider
from opentulpa.agent.runtime_input import ThreadInputCoordinator


class _OversizedInvokeGraph:
    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"final_response_text": "A" * 9000}


class _OversizedStreamGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content="B" * 9000), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [AIMessage(content="unused")]}


class _MultiChunkStreamGraph:
    async def astream(
        self,
        _state: dict[str, Any],
        *,
        config: dict[str, Any],
        stream_mode: str,
    ) -> AsyncIterator[tuple[AIMessage, dict[str, str]]]:
        del config, stream_mode
        yield AIMessage(content="Hello"), {"langgraph_node": "agent"}
        yield AIMessage(content=" from"), {"langgraph_node": "agent"}
        yield AIMessage(content=" web."), {"langgraph_node": "agent"}

    async def ainvoke(self, _state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"messages": [AIMessage(content="unused")]}


def _build_runtime(tmp_path, graph: Any) -> OpenTulpaLangGraphRuntime:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = graph
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime._behavior_log_enabled = True
    runtime._behavior_log_path = tmp_path / "agent_behavior_reply_limits.jsonl"
    runtime._behavior_log_lock = threading.Lock()
    runtime._max_user_reply_chars = 4000
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.expand_link_aliases = lambda **kwargs: str(kwargs.get("text", ""))  # type: ignore[assignment]

    async def _noop_start() -> None:
        return None

    async def _list_available_skills(customer_id: str) -> list[dict[str, Any]]:
        del customer_id
        return []

    async def _load_skill_context_by_names(
        *, customer_id: str, skill_names: list[str]
    ) -> dict[str, Any]:
        del customer_id, skill_names
        return {"skill_names": [], "context": ""}

    runtime.start = _noop_start  # type: ignore[method-assign]
    runtime._list_available_skills = _list_available_skills  # type: ignore[method-assign]
    runtime._load_skill_context_by_names = _load_skill_context_by_names  # type: ignore[method-assign]
    runtime._context_source_provider = RuntimeContextSourceProvider(runtime)
    return runtime


@pytest.mark.asyncio
async def test_ainvoke_text_truncates_oversized_reply(tmp_path) -> None:
    runtime = _build_runtime(tmp_path, _OversizedInvokeGraph())

    reply = await runtime.ainvoke_text(
        thread_id="chat-limit",
        customer_id="telegram_limit",
        text="draft the post",
    )

    assert len(reply) <= 4000
    assert reply.endswith("[Response truncated to fit chat limits.]")

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines if line.strip()]
    assert "turn_reply_truncated" in events


@pytest.mark.asyncio
async def test_astream_text_truncates_oversized_stream_reply(tmp_path) -> None:
    runtime = _build_runtime(tmp_path, _OversizedStreamGraph())

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-stream-limit",
        customer_id="telegram_limit",
        text="draft the post",
    ):
        chunks.append(chunk)

    assert chunks
    assert len(chunks[-1]) <= 4000
    assert chunks[-1].endswith("[Response truncated to fit chat limits.]")

    lines = runtime._behavior_log_path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line)["event"] for line in lines if line.strip()]
    assert "turn_stream_reply_truncated" in events


@pytest.mark.asyncio
async def test_astream_text_can_emit_incremental_visible_deltas(tmp_path) -> None:
    runtime = _build_runtime(tmp_path, _MultiChunkStreamGraph())

    chunks: list[str] = []
    async for chunk in runtime.astream_text(
        thread_id="chat-stream-deltas",
        customer_id="telegram_limit",
        text="draft the post",
        stream_precommit_seconds=0.0,
        stream_incremental_deltas=True,
    ):
        chunks.append(chunk)

    assert chunks == ["Hello", " from", " web."]
    assert "".join(chunks) == "Hello from web."
