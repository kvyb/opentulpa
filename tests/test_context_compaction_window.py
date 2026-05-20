from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.agent.context_compaction import (
    _select_split_index,
    _trim_text_to_token_budget,
    maybe_compact_thread_context,
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
        self.recursion_limit = 30
        self._context_token_limit = 40000
        self._context_short_term_high_tokens = 40000
        self._context_short_term_low_tokens = 20000
        self._context_recent_tokens = 20000
        self._context_rollup_tokens = 5000
        self._context_compaction_source_tokens = 12000
        self._rollups: dict[str, str] = {}

    def _load_thread_rollup(self, thread_id: str) -> str | None:
        return self._rollups.get(thread_id)

    def _save_thread_rollup(self, thread_id: str, rollup: str) -> None:
        self._rollups[thread_id] = str(rollup or "").strip()


@pytest.mark.asyncio
async def test_maybe_compact_thread_context_enforces_recent_window_and_rollup_cap() -> None:
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2800)) for i in range(70)]
    runtime = _DummyRuntime(messages)

    before_tokens = sum(approx_tokens(f"[user] {str(m.content)}") for m in messages)
    assert before_tokens >= 40000

    await maybe_compact_thread_context(
        runtime,
        thread_id="chat-test",
        customer_id="telegram_1",
    )

    remaining_messages = runtime._graph._messages
    after_tokens = sum(
        approx_tokens(f"[user] {str(m.content)}") for m in remaining_messages
    )
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
async def test_maybe_compact_thread_context_chunks_full_removed_context_while_dropping_to_recent_window() -> None:
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2800)) for i in range(180)]
    runtime = _DummyRuntime(messages)
    runtime._context_short_term_high_tokens = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime._context_rollup_tokens = 2200
    runtime._context_compaction_source_tokens = 12000

    await maybe_compact_thread_context(
        runtime,
        thread_id="chat-huge",
        customer_id="telegram_1",
    )

    remaining_messages = runtime._graph._messages
    remaining_tokens = sum(
        approx_tokens(f"[user] {str(m.content)}") for m in remaining_messages
    )
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
async def test_maybe_compact_thread_context_noop_inside_hysteresis_window() -> None:
    # ~30k tokens should not trigger compaction with 20k..40k window.
    messages = [HumanMessage(content=f"msg_{i} " + ("x" * 2200)) for i in range(50)]
    runtime = _DummyRuntime(messages)
    before = list(runtime._graph._messages)
    before_tokens = sum(approx_tokens(f"[user] {str(m.content)}") for m in before)
    assert 20000 < before_tokens < 40000

    await maybe_compact_thread_context(
        runtime,
        thread_id="chat-window",
        customer_id="telegram_1",
    )

    after = runtime._graph._messages
    assert len(after) == len(before)
    assert runtime._checkpointer.deleted is False
    assert not runtime._rollups.get("chat-window")
