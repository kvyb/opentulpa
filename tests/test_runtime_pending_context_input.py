from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from opentulpa.agent.lc_messages import AIMessage, HumanMessage
from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.runtime_input import ThreadInputCoordinator
from opentulpa.agent.utils import approx_tokens as _approx_tokens


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


class _FakeCheckpointer:
    def __bool__(self) -> bool:
        return False

    def get_next_version(self, current: Any, channel: Any) -> int:
        del current, channel
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


class _GraphModel:
    def __init__(self) -> None:
        self.seen_messages: list[Any] | None = None

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.seen_messages = messages
        return AIMessage(content="ok")


@pytest.mark.asyncio
async def test_agent_reuses_turn_scoped_available_skills_without_relisting() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _GraphModel()
    list_calls = 0
    resolve_calls = 0

    async def _unexpected_list(customer_id: str) -> list[dict[str, Any]]:
        nonlocal list_calls
        list_calls += 1
        del customer_id
        return []

    async def _unexpected_resolve(
        customer_id: str,
        user_text: str,
        *,
        candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        nonlocal resolve_calls
        resolve_calls += 1
        del customer_id, user_text, candidates
        return {"skill_names": [], "context": ""}

    async def _live_time(customer_id: str) -> dict[str, str]:
        del customer_id
        return {
            "server_time_local_iso": "2026-04-02T00:00:00+08:00",
            "server_time_utc_iso": "2026-04-01T16:00:00+00:00",
            "server_utc_offset": "+08:00",
            "user_time_local_iso": "2026-04-02T00:00:00+08:00",
            "user_utc_offset": "+08:00",
            "user_time_source": "profile",
        }

    async def _directive(customer_id: str) -> str | None:
        del customer_id
        return None

    async def _verify_completion_claim(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"usable": True, "mismatch": False, "applies": True}

    runtime._model_with_tools = model
    runtime._checkpointer = InMemorySaver()
    runtime._list_available_skills = _unexpected_list  # type: ignore[method-assign]
    runtime._resolve_skill_context = _unexpected_resolve  # type: ignore[method-assign]
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_thread_rollup = lambda thread_id: None  # type: ignore[assignment]
    runtime._thread_rollup_service = None
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {}
    runtime.verify_completion_claim = _verify_completion_claim  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="use the saved browser skill")],
            "customer_id": "telegram_test",
            "thread_id": "chat_test",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_test",
            "active_skill_query": "use the saved browser skill",
            "active_skill_context": "matched skill context",
            "active_skill_names": ["browser-use-operator"],
            "active_available_skills": [
                {
                    "name": "browser-use-operator",
                    "description": "Use browser steps for dynamic websites.",
                    "scope": "global",
                }
            ],
        },
        config={"configurable": {"thread_id": "chat_test"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "ok"
    assert list_calls == 0
    assert resolve_calls == 0
    assert model.seen_messages is not None
    assert any(
        "browser-use-operator" in str(getattr(msg, "content", ""))
        for msg in model.seen_messages
    )


@pytest.mark.asyncio
async def test_interactive_prompt_injects_memory_grounding_after_stable_prefix() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured: dict[str, Any] = {}

    async def _live_time(customer_id: str) -> dict[str, str]:
        del customer_id
        return {
            "server_time_local_iso": "2026-04-09T10:00:00+08:00",
            "server_time_utc_iso": "2026-04-09T02:00:00+00:00",
            "server_utc_offset": "+08:00",
            "user_time_local_iso": "2026-04-09T10:00:00+08:00",
            "user_utc_offset": "+08:00",
            "user_time_source": "profile",
        }

    async def _directive(customer_id: str) -> str | None:
        del customer_id
        return None

    async def _memory_grounding(**kwargs: Any) -> str:
        del kwargs
        return (
            "Preferences and directives:\n- Be concise and direct.\n\n"
            "Technical or code facts:\n- Telegram bot uses Gemini Flash for media analysis."
        )

    async def _ainvoke_model(model: Any, messages: list[Any], *, stable_prefix_count: int = 0, **kwargs: Any) -> AIMessage:
        del model
        captured["messages"] = messages
        captured["stable_prefix_count"] = stable_prefix_count
        captured["call_context"] = kwargs.get("call_context")
        return AIMessage(content="ok")

    async def _verify_completion_claim(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"usable": True, "mismatch": False, "applies": True}

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.verify_completion_claim = _verify_completion_claim  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="what do you remember about my bot setup?")],
            "customer_id": "telegram_test",
            "thread_id": "chat_test",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_test",
        },
        config={"configurable": {"thread_id": "chat_test"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "ok"
    assert captured["stable_prefix_count"] == 1
    prompt_messages = captured["messages"]
    grounding_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if "Relevant long-term memory grounding" in str(getattr(msg, "content", ""))
    )
    assert grounding_index >= captured["stable_prefix_count"]
    last_human_index = max(
        idx
        for idx, msg in enumerate(prompt_messages)
        if isinstance(msg, HumanMessage)
    )
    assert grounding_index < last_human_index
    assert isinstance(captured["call_context"], dict)
    assert captured["call_context"]["call_site"] == "graph_agent"
    assert "memory_grounding" in captured["call_context"]["prompt_sections"]


def test_memory_grounding_block_stays_compact() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    memories = [
        {"kind": "directive_fact", "text": "Always be concise, direct, and avoid filler.", "score": 0.9},
        {"kind": "life_fact", "text": "Timezone is UTC+8 and works mostly in the afternoon.", "score": 0.8},
        {"kind": "aspirations_fact", "text": "Wants to launch more reliable Telegram and Instagram automation.", "score": 0.7},
        {"kind": "workflow_fact", "text": "Runs an Instagram intake workflow that writes bookings to Google Sheets.", "score": 0.6},
        {"kind": "code_fact", "text": "Main chat model is GLM 5.1 while media and memory use Gemini Flash.", "score": 0.65},
        {"kind": "file_fact", "text": "Uploaded planning PDF is stored in tulpa_stuff/uploads for later recall.", "score": 0.55},
        {"kind": "thread_context_rollup", "text": "Older thread context mentioning long implementation notes and stale discussion that should be deprioritized.", "score": 0.2},
    ]

    block = runtime._build_memory_grounding_block(memories, token_budget=500)

    assert "Preferences and directives:" in block
    assert "Technical or code facts:" in block
    assert _approx_tokens(block) <= 520
