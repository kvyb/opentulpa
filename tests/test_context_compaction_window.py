from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.agent.context_compaction import (
    _select_split_index,
    _trim_text_to_token_budget,
    compact_thread_context_for_turn,
    persist_rollup_memory,
)
from opentulpa.agent.lc_messages import HumanMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.utils import approx_tokens
from opentulpa.core.config import Settings


def test_trim_text_to_token_budget_respects_limit() -> None:
    raw = "alpha " * 10000
    trimmed = _trim_text_to_token_budget(raw, 500)
    assert trimmed
    assert approx_tokens(trimmed) <= 500


def test_select_split_index_compacts_enough_without_dropping_all() -> None:
    tokens = [1200, 900, 3000, 2200, 800]
    split_idx = _select_split_index(tokens, tokens_to_compact=3500)
    assert split_idx > 0
    assert split_idx < len(tokens)
    assert sum(tokens[:split_idx]) >= 3500


def test_context_compaction_default_threshold_is_20000() -> None:
    assert Settings.model_fields["agent_context_token_limit"].default == 20000
    assert (
        Settings.model_fields["agent_context_compaction_model"].default
        == "google/gemini-3-flash-preview"
    )


def test_runtime_context_token_limit_clamps_at_30000(tmp_path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        context_token_limit=1000000,
    )

    assert runtime._context_token_limit == 30000
    assert runtime._context_short_term_high_tokens == 30000


@dataclass
class _DummyCheckpointer:
    deleted: bool = False

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted = bool(thread_id)


class _DummyGraph:
    def __init__(self, messages: list[Any]) -> None:
        self._messages = list(messages)

    async def aget_state(self, config: dict[str, Any]) -> Any:
        return SimpleNamespace(values={"messages": list(self._messages)})

    async def aupdate_state(self, config: dict[str, Any], values: dict[str, Any]) -> None:
        msgs = values.get("messages", [])
        self._messages = list(msgs) if isinstance(msgs, list) else []


class _DummyModel:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.calls.append(messages)
        # Return intentionally large output; compaction should still cap it.
        return SimpleNamespace(content=("rollup-summary " * 3000))


class _DummyRuntime:
    def __init__(self, messages: list[Any]) -> None:
        self._graph = _DummyGraph(messages)
        self._checkpointer = _DummyCheckpointer()
        self._model = _DummyModel()
        self._context_compaction_model = self._model
        self._context_compaction_model_name = "google/gemini-3-flash-preview"
        self.recursion_limit = 30
        self._context_token_limit = 40000
        self._context_short_term_high_tokens = 40000
        self._context_short_term_low_tokens = 20000
        self._context_recent_tokens = 20000
        self._context_rollup_tokens = 5000
        self._context_compaction_source_tokens = 12000
        self._rollups: dict[str, str] = {}
        self.memory_add_calls: list[dict[str, Any]] = []
        self.memory_persist_started = False
        self.memory_persist_continue: asyncio.Event | None = None

    def _load_thread_rollup(self, thread_id: str) -> str | None:
        return self._rollups.get(thread_id)

    def _save_thread_rollup(self, thread_id: str, rollup: str) -> None:
        self._rollups[thread_id] = str(rollup or "").strip()

    async def _request_with_backoff(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any],
        timeout: float,
        retries: int,
    ) -> dict[str, Any]:
        self.memory_persist_started = True
        self.memory_add_calls.append(
            {
                "method": method,
                "path": path,
                "json_body": json_body,
                "timeout": timeout,
                "retries": retries,
            }
        )
        if self.memory_persist_continue is not None:
            await self.memory_persist_continue.wait()
        return {"ok": True}


@pytest.mark.asyncio
async def test_compact_thread_context_for_turn_enforces_recent_window_and_rollup_cap() -> None:
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2800)) for i in range(70)]
    runtime = _DummyRuntime(messages)

    before_tokens = sum(approx_tokens(f"[user] {str(m.content)}") for m in messages)
    assert before_tokens >= 40000

    result = await compact_thread_context_for_turn(
        runtime,
        thread_id="chat-test",
        customer_id="telegram_1",
    )

    assert result.status == "compacted"
    remaining_messages = runtime._graph._messages
    after_tokens = sum(approx_tokens(f"[user] {str(m.content)}") for m in remaining_messages)
    assert remaining_messages
    assert after_tokens <= 20000
    assert runtime._checkpointer.deleted is True
    assert len(runtime._model.calls) >= 1
    for call in runtime._model.calls:
        assert approx_tokens(str(call[1].content)) <= 20000

    rollup = runtime._rollups.get("chat-test", "")
    assert rollup
    assert approx_tokens(rollup) <= 5000


@pytest.mark.asyncio
async def test_compaction_schedules_rollup_memory_persist_off_hot_path() -> None:
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2800)) for i in range(70)]
    runtime = _DummyRuntime(messages)
    runtime.memory_persist_continue = asyncio.Event()

    result = await compact_thread_context_for_turn(
        runtime,
        thread_id="chat-background",
        customer_id="telegram_1",
    )

    assert result.status == "compacted"
    assert runtime._rollups.get("chat-background")
    assert runtime._checkpointer.deleted is True
    assert runtime._context_compaction_background_tasks

    for _ in range(20):
        if runtime.memory_persist_started:
            break
        await asyncio.sleep(0)
    assert runtime.memory_persist_started is True
    assert runtime.memory_add_calls
    body = runtime.memory_add_calls[0]["json_body"]
    assert body["infer"] is False
    assert body["metadata"] == {
        "kind": "thread_context_rollup",
        "thread_id": "chat-background",
    }

    runtime.memory_persist_continue.set()
    await asyncio.gather(*runtime._context_compaction_background_tasks)
    assert not runtime._context_compaction_background_tasks


@pytest.mark.asyncio
async def test_runtime_shutdown_drains_compaction_memory_tasks_before_teardown() -> None:
    events: list[str] = []

    class _Manager:
        async def shutdown(self) -> None:
            events.append("manager_shutdown")

    async def _background_persist() -> None:
        await asyncio.sleep(0)
        events.append("persist_done")

    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    task = asyncio.create_task(_background_persist())
    runtime._context_compaction_background_tasks = {task}
    runtime._browser_use_local_manager = _Manager()
    runtime._langfuse_tracer = None
    runtime._checkpointer_cm = None
    runtime._checkpointer = object()
    runtime._graph = object()
    runtime._model_with_tools = object()
    runtime._workflow_setup_model_with_tools = object()
    runtime._wake_execution_model_with_tools = object()

    await runtime.shutdown()

    assert events == ["persist_done", "manager_shutdown"]
    assert runtime._context_compaction_background_tasks == set()
    assert runtime._browser_use_local_manager is None
    assert runtime._checkpointer is None
    assert runtime._graph is None


@pytest.mark.asyncio
async def test_persist_rollup_memory_disables_mem0_inference() -> None:
    runtime = _DummyRuntime([])

    await persist_rollup_memory(
        runtime,
        customer_id="telegram_1",
        thread_id="chat-direct",
        rollup="durable summary",
    )

    assert runtime.memory_add_calls
    assert runtime.memory_add_calls[0]["path"] == "/internal/memory/add"
    assert runtime.memory_add_calls[0]["json_body"]["infer"] is False


@pytest.mark.asyncio
async def test_compact_thread_context_for_turn_uses_configured_compaction_model() -> None:
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2800)) for i in range(70)]
    runtime = _DummyRuntime(messages)
    compaction_model = _DummyModel()
    runtime._context_compaction_model = compaction_model
    runtime._context_compaction_model_name = "google/gemini-3-flash-preview"

    result = await compact_thread_context_for_turn(
        runtime,
        thread_id="chat-compaction-model",
        customer_id="telegram_1",
    )

    assert result.status == "compacted"
    assert compaction_model.calls
    assert not runtime._model.calls


@pytest.mark.asyncio
async def test_compact_thread_context_for_turn_chunks_full_removed_context_while_dropping_to_recent_window() -> (
    None
):
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2800)) for i in range(180)]
    runtime = _DummyRuntime(messages)
    runtime._context_short_term_high_tokens = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime._context_rollup_tokens = 2200
    runtime._context_compaction_source_tokens = 12000

    result = await compact_thread_context_for_turn(
        runtime,
        thread_id="chat-huge",
        customer_id="telegram_1",
    )

    assert result.status == "compacted"
    remaining_messages = runtime._graph._messages
    remaining_tokens = sum(approx_tokens(f"[user] {str(m.content)}") for m in remaining_messages)
    assert remaining_messages
    assert remaining_tokens <= 3500
    assert len(runtime._model.calls) > 1

    compaction_inputs = [str(call[1].content) for call in runtime._model.calls]
    assert all("Older conversation segment to fold in:" in item for item in compaction_inputs)
    assert all(approx_tokens(item) <= 15000 for item in compaction_inputs)
    folded_text = "\n".join(compaction_inputs)
    assert "msg_0" in folded_text
    assert "msg_80" in folded_text
    assert "msg_170" in folded_text


@pytest.mark.asyncio
async def test_compact_thread_context_for_turn_noop_inside_hysteresis_window() -> None:
    # ~30k tokens should not trigger compaction with 20k..40k window.
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2200)) for i in range(50)]
    runtime = _DummyRuntime(messages)
    before = list(runtime._graph._messages)
    before_tokens = sum(approx_tokens(f"[user] {str(m.content)}") for m in before)
    assert 20000 < before_tokens < 40000

    result = await compact_thread_context_for_turn(
        runtime,
        thread_id="chat-window",
        customer_id="telegram_1",
    )

    assert result.status == "skipped"
    assert result.reason == "not_needed"
    after = runtime._graph._messages
    assert len(after) == len(before)
    assert runtime._checkpointer.deleted is False
    assert not runtime._rollups.get("chat-window")
