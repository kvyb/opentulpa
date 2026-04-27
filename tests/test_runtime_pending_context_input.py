from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, ToolMessage
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
            "messages": [
                HumanMessage(content="remind me what stack I used before"),
                AIMessage(content="You used Gemini Flash for media analysis."),
                HumanMessage(content="what do you remember about my bot setup?"),
            ],
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
    assert captured["stable_prefix_count"] >= 2
    prompt_messages = captured["messages"]
    older_assistant_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if isinstance(msg, AIMessage) and "Gemini Flash for media analysis" in str(getattr(msg, "content", ""))
    )
    grounding_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if "Relevant long-term memory grounding" in str(getattr(msg, "content", ""))
    )
    assert older_assistant_index < grounding_index
    assert grounding_index < captured["stable_prefix_count"]
    last_human_index = max(
        idx
        for idx, msg in enumerate(prompt_messages)
        if isinstance(msg, HumanMessage)
    )
    assert grounding_index < last_human_index
    assert isinstance(captured["call_context"], dict)
    assert captured["call_context"]["call_site"] == "graph_agent"
    assert "memory_grounding" in captured["call_context"]["prompt_sections"]


@pytest.mark.asyncio
async def test_agent_freezes_live_time_context_across_tool_loop() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_messages: list[list[Any]] = []
    captured_prefix_counts: list[int] = []
    live_time_calls = 0

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {"status": "ok", "result": "done"}

    async def _live_time(customer_id: str) -> dict[str, str]:
        nonlocal live_time_calls
        del customer_id
        live_time_calls += 1
        minute = f"{live_time_calls:02d}"
        return {
            "server_time_local_iso": f"2026-04-09T10:{minute}:00+08:00",
            "server_time_utc_iso": f"2026-04-09T02:{minute}:00+00:00",
            "server_utc_offset": "+08:00",
            "user_time_local_iso": f"2026-04-09T10:{minute}:00+08:00",
            "user_utc_offset": "+08:00",
            "user_time_source": "profile",
        }

    async def _directive(customer_id: str) -> str | None:
        del customer_id
        return None

    async def _memory_grounding(**kwargs: Any) -> str:
        del kwargs
        return ""

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, kwargs
        captured_messages.append(list(messages))
        captured_prefix_counts.append(stable_prefix_count)
        if len(captured_messages) == 1:
            return AIMessage(
                content="Let me run that.",
                tool_calls=[{"id": "call_1", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="Done.")

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
    runtime._has_retrieval_evidence = lambda **kwargs: False  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
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
            "messages": [HumanMessage(content="run the fake tool and then answer")],
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

    assert result["final_response_text"] == "Done."
    assert len(captured_messages) == 2
    assert live_time_calls == 1
    assert captured_prefix_counts[0] >= 2
    assert captured_prefix_counts[1] == captured_prefix_counts[0]

    def _live_time_block(messages: list[Any]) -> str:
        return next(
            str(getattr(msg, "content", ""))
            for msg in messages
            if "Live time context (auto-injected this turn):" in str(getattr(msg, "content", ""))
        )

    first_live_time = _live_time_block(captured_messages[0])
    second_live_time = _live_time_block(captured_messages[1])
    assert "2026-04-09T10:01:00+08:00" in first_live_time
    assert second_live_time == first_live_time
    assert captured_messages[0][:captured_prefix_counts[0]] == captured_messages[1][:captured_prefix_counts[1]]


@pytest.mark.asyncio
async def test_agent_freezes_older_history_projection_and_stale_summary_across_tool_loop() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_messages: list[list[Any]] = []
    captured_prefix_counts: list[int] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {"status": "ok"}

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
        return ""

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, kwargs
        captured_messages.append(list(messages))
        captured_prefix_counts.append(stable_prefix_count)
        if len(captured_messages) == 1:
            return AIMessage(
                content="Let me run that.",
                tool_calls=[{"id": "call_1", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="Done.")

    async def _verify_completion_claim(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"usable": True, "mismatch": False, "applies": True}

    prior_messages: list[Any] = []
    for idx in range(14):
        prior_messages.append(HumanMessage(content=f"Earlier user note {idx}: keep this thread moving."))
        prior_messages.append(AIMessage(content=f"Earlier assistant reply {idx}: acknowledged."))

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._has_retrieval_evidence = lambda **kwargs: False  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
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
            "messages": [*prior_messages, HumanMessage(content="run the fake tool and then answer")],
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

    assert result["final_response_text"] == "Done."
    assert len(captured_messages) == 2
    assert captured_prefix_counts[1] == captured_prefix_counts[0]
    assert captured_messages[0][:captured_prefix_counts[0]] == captured_messages[1][:captured_prefix_counts[1]]

    def _summary_block(messages: list[Any]) -> str:
        return next(
            str(getattr(msg, "content", ""))
            for msg in messages
            if "Compressed older in-thread context." in str(getattr(msg, "content", ""))
        )

    first_summary = _summary_block(captured_messages[0])
    second_summary = _summary_block(captured_messages[1])
    assert first_summary == second_summary


@pytest.mark.asyncio
async def test_deepseek_prompt_uses_only_current_turn_raw_history() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_messages: list[Any] = []

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
        return ""

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_messages.extend(messages)
        return AIMessage(content="Done.")

    async def _verify_completion_claim(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"usable": True, "mismatch": False, "applies": True}

    prior_messages: list[Any] = []
    for idx in range(14):
        prior_messages.append(HumanMessage(content=f"Earlier user note {idx}: old raw chat."))
        prior_messages.append(AIMessage(content=f"Earlier assistant reply {idx}: old raw reply."))

    runtime.model_name = "deepseek/deepseek-v4-pro"
    runtime.openrouter_base_url = "https://openrouter.ai/api/v1"
    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._has_retrieval_evidence = lambda **kwargs: False  # type: ignore[assignment]
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
            "messages": [*prior_messages, HumanMessage(content="current live ask")],
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

    assert result["final_response_text"] == "Done."
    human_texts = [str(getattr(msg, "content", "")) for msg in captured_messages if isinstance(msg, HumanMessage)]
    assistant_texts = [str(getattr(msg, "content", "")) for msg in captured_messages if isinstance(msg, AIMessage)]
    assert "current live ask" in human_texts
    assert not any(text.startswith("Earlier user note") for text in human_texts)
    assert not any(text.startswith("Earlier assistant reply") for text in assistant_texts)


@pytest.mark.asyncio
async def test_deepseek_prompt_keeps_only_latest_tool_segment_raw() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_messages: list[list[Any]] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            return {"status": "ok", "step": args.get("step")}

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
        return ""

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_messages.append(list(messages))
        if len(captured_messages) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "fake_tool", "args": {"step": 1}}],
            )
        if len(captured_messages) == 2:
            return AIMessage(
                content="",
                tool_calls=[{"id": "call_2", "name": "fake_tool", "args": {"step": 2}}],
            )
        return AIMessage(content="Done.")

    async def _verify_completion_claim(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {"usable": True, "mismatch": False, "applies": True}

    runtime.model_name = "deepseek/deepseek-v4-pro"
    runtime.openrouter_base_url = "https://openrouter.ai/api/v1"
    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._has_retrieval_evidence = lambda **kwargs: False  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.verify_completion_claim = _verify_completion_claim  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 10

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="run two tool steps")],
            "customer_id": "telegram_test",
            "thread_id": "chat_test",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_test",
        },
        config={"configurable": {"thread_id": "chat_test"}, "recursion_limit": 10},
    )

    assert result["final_response_text"] == "Done."
    assert len(captured_messages) == 3
    third_prompt = captured_messages[2]
    raw_tool_ids = [
        str(getattr(message, "tool_call_id", "") or "")
        for message in third_prompt
        if isinstance(message, ToolMessage)
    ]
    raw_ai_tool_ids = [
        str(call.get("id", "") or "")
        for message in third_prompt
        if isinstance(message, AIMessage)
        for call in (getattr(message, "tool_calls", []) or [])
        if isinstance(call, dict)
    ]
    assert raw_tool_ids == ["call_2"]
    assert raw_ai_tool_ids == ["call_2"]



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
