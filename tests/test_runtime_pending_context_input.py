from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from opentulpa.agent.graph_builder import build_runtime_graph
from opentulpa.agent.graph_control_tools import register_graph_control_tools
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, ToolMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.runtime_context_provider import RuntimeContextSourceProvider
from opentulpa.agent.runtime_input import ThreadInputCoordinator
from opentulpa.agent.tool_execution_policy import ToolExecutionPolicy
from opentulpa.agent.tools.owner_update_tools import register_owner_update_tools
from opentulpa.agent.turn_context_preparer import (
    build_graph_input,
    format_pending_context,
    pre_resolve_skill_state,
)
from opentulpa.agent.utils import approx_tokens as _approx_tokens


def _ready_setup_session() -> dict[str, Any]:
    return {
        "session_id": "iwsetup_test",
        "customer_id": "telegram_test",
        "thread_id": "chat-workflow-setup-context",
        "status": "active",
        "mode": "create",
        "draft_upsert": {
            "name": "AutoSpa",
            "channel": "telegram_business_dm",
            "provider": "composio",
            "intent_description": "Book car wash leads",
            "required_fields": ["service_name", "vehicle_type", "date", "time", "phone"],
            "knowledge_file_ids": ["file_price"],
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "toolkit": "googlesheets",
                "static_arguments": {
                    "spreadsheetId": "sheet_123",
                    "sheetName": "Bookings",
                },
                "field_mapping": {
                    "service_name": "Service",
                    "vehicle_type": "Vehicle",
                },
            },
        },
        "scratchpad": {
            "last_preflight": {
                "ok": True,
                "status": "ready",
                "errors": [],
                "warnings": [],
                "follow_up_questions": [],
            },
            "knowledge_last_preflight": {"ok": True, "status": "ready"},
        },
        "last_proposed_draft_hash": "",
        "confirmed_draft_hash": "",
        "created_or_updated_workflow_id": "",
    }


class _FakeWorkflowSetupService:
    def __init__(self, session: dict[str, Any] | None) -> None:
        self.session = session

    def get_thread_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        include_paused: bool = True,
    ) -> dict[str, Any] | None:
        del customer_id, thread_id, include_paused
        return self.session


class _CapturingGraph:
    def __init__(self) -> None:
        self.last_state: dict[str, Any] | None = None
        self.states: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del config
        self.last_state = state
        self.states.append(state)
        return {"messages": [AIMessage(content="ok")]}


class _AinvokeStaleMessageGraph:
    async def ainvoke(self, state: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        del state, config
        return {
            "final_response_text": "",
            "messages": [
                HumanMessage(content="old user"),
                AIMessage(content="old assistant reply"),
                HumanMessage(content="current user"),
                AIMessage(content=""),
            ],
        }


class _FakeContextEvents:
    def __init__(self) -> None:
        self.cleared: tuple[str, int | None] | None = None

    def list_events(self, customer_id: str, limit: int = 20) -> list[dict[str, Any]]:
        del customer_id, limit
        return [
            {
                "id": 42,
                "source": "task",
                "event_type": "executed",
                "payload": {
                    "task_id": "task_abc",
                    "status": "executed",
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


class _UnnamedFakeTool:
    async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "args": args}


def _install_prompt_source_stubs(runtime: OpenTulpaLangGraphRuntime) -> None:
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


def _install_minimal_graph_runtime_stubs(
    runtime: OpenTulpaLangGraphRuntime,
    *,
    ainvoke_model: Any,
    behavior_events: list[str] | None = None,
) -> None:
    async def _live_time(customer_id: str) -> dict[str, str]:
        del customer_id
        return {
            "server_time_local_iso": "2026-04-29T12:00:00+08:00",
            "server_time_utc_iso": "2026-04-29T04:00:00+00:00",
            "server_utc_offset": "+08:00",
            "user_time_local_iso": "2026-04-29T12:00:00+08:00",
            "user_utc_offset": "+08:00",
            "user_time_source": "profile",
        }

    async def _directive(customer_id: str) -> str | None:
        del customer_id
        return None

    async def _memory_grounding(**kwargs: Any) -> str:
        del kwargs
        return ""

    async def _list_available_skills(customer_id: str) -> list[dict[str, Any]]:
        del customer_id
        return []

    async def _load_skill_context_by_names(
        *, customer_id: str, skill_names: list[str]
    ) -> dict[str, Any]:
        del customer_id, skill_names
        return {"skill_names": [], "context": ""}

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {}
    runtime.ainvoke_model = ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = (  # type: ignore[assignment]
        (lambda **kwargs: behavior_events.append(str(kwargs.get("event", ""))))
        if behavior_events is not None
        else (lambda **kwargs: None)
    )
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8
    runtime._workflow_setup_service = None
    _install_prompt_source_stubs(runtime)


def test_tool_execution_policy_allows_runtime_tools_without_name_attribute() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._tools = {"tool_group_exec": _UnnamedFakeTool()}

    policy = ToolExecutionPolicy.from_runtime_state(
        runtime=runtime,
        state={
            "customer_id": "telegram_test",
            "thread_id": "chat_test",
            "turn_mode": "interactive",
        },
    )

    assert "tool_group_exec" in policy.allowed_tool_names
    policy.validate_call(call_name="tool_group_exec", customer_scoped_tools=set())


@pytest.mark.asyncio
async def test_graph_keeps_turn_plan_in_state_for_next_agent_step() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_second_pass_messages: list[Any] = []
    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_plan",
                        "name": "turn_plan",
                        "args": {
                            "items": [
                                {
                                    "id": "scope",
                                    "content": "Define research scope",
                                    "status": "completed",
                                },
                                {
                                    "id": "search",
                                    "content": "Gather current sources",
                                    "status": "in_progress",
                                },
                                {
                                    "id": "report",
                                    "content": "Synthesize report",
                                    "status": "pending",
                                },
                            ]
                        },
                    },
                    {
                        "id": "call_plan_merge",
                        "name": "turn_plan",
                        "args": {
                            "merge": True,
                            "items": [
                                {
                                    "id": "search",
                                    "content": "Gather current sources",
                                    "status": "completed",
                                },
                                {
                                    "id": "report",
                                    "content": "Synthesize report",
                                    "status": "in_progress",
                                },
                            ],
                        },
                    },
                ],
            )
        captured_second_pass_messages.extend(messages)
        return AIMessage(content="Done.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = register_graph_control_tools(runtime)

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="research current AI news")],
            "customer_id": "telegram_test",
            "thread_id": "chat_turn_plan_state",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_plan_state",
            "turn_plan": [],
        },
        config={"configurable": {"thread_id": "chat_turn_plan_state"}, "recursion_limit": 8},
    )

    plan_context = next(
        str(getattr(message, "content", ""))
        for message in captured_second_pass_messages
        if "CURRENT_TURN_PLAN" in str(getattr(message, "content", ""))
    )
    assert result["final_response_text"] == "Done."
    assert result["turn_plan"] == [
        {"id": "scope", "content": "Define research scope", "status": "completed"},
        {"id": "search", "content": "Gather current sources", "status": "completed"},
        {"id": "report", "content": "Synthesize report", "status": "in_progress"},
    ]
    assert "CURRENT_TURN_PLAN" in plan_context
    assert "Synthesize report" in plan_context
    assert "[x] scope: Define research scope (completed)" in plan_context
    assert "[x] search: Gather current sources (completed)" in plan_context


@pytest.mark.asyncio
async def test_graph_turn_plan_validation_error_returns_to_model() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_bad_plan",
                        "name": "turn_plan",
                        "args": {
                            "items": [
                                {
                                    "id": "search",
                                    "content": "Gather sources",
                                    "status": "working",
                                }
                            ]
                        },
                    }
                ],
            )
        assert any(
            "status must be one of" in str(getattr(message, "content", "")) for message in messages
        )
        return AIMessage(content="Fixed.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = register_graph_control_tools(runtime)

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="research current AI news")],
            "customer_id": "telegram_test",
            "thread_id": "chat_turn_plan_validation",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_plan_validation",
            "turn_plan": [],
        },
        config={"configurable": {"thread_id": "chat_turn_plan_validation"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Fixed."
    assert result["turn_plan"] == []


@pytest.mark.asyncio
async def test_graph_turn_budget_forces_no_tools_finalizer_after_budget_exhaustion() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    call_sites: list[str] = []
    prompts: list[list[Any]] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {"found": ["one useful result"]}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        call_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        call_sites.append(str((call_context or {}).get("call_site", "")))
        prompts.append(list(messages))
        if calls <= 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": f"call_search_{calls}",
                        "name": "fake_tool",
                        "args": {"query": f"q{calls}"},
                    }
                ],
            )
        assert any("Do not call tools" in str(getattr(message, "content", "")) for message in messages)
        assert any("one useful result" in str(getattr(message, "content", "")) for message in messages)
        return AIMessage(content="Final report from gathered results.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Do long research")],
            "customer_id": "telegram_test",
            "thread_id": "chat_turn_budget_finalizer",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_budget_finalizer",
            "turn_plan": [],
        },
        config={"configurable": {"thread_id": "chat_turn_budget_finalizer"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Final report from gathered results."
    assert call_sites == ["graph_agent", "graph_agent", "graph_turn_budget_finalizer"]
    assert result["turn_budget"]["used_model_calls"] == 2
    assert result["turn_budget"]["finalizer_used"] is True
    second_agent_prompt = "\n".join(str(getattr(message, "content", "")) for message in prompts[1])
    assert "TURN_BUDGET_STATUS" in second_agent_prompt


@pytest.mark.asyncio
async def test_graph_resets_turn_plan_for_new_user_turn() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_prompts: list[list[Any]] = []
    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        captured_prompts.append(messages)
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_plan",
                        "name": "turn_plan",
                        "args": {
                            "items": [
                                {
                                    "id": "search",
                                    "content": "Search current evidence",
                                    "status": "in_progress",
                                }
                            ]
                        },
                    }
                ],
            )
        return AIMessage(content=f"Done {calls}.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = register_graph_control_tools(runtime)

    graph = build_runtime_graph(runtime)
    base_state = {
        "customer_id": "telegram_test",
        "thread_id": "chat_turn_plan_reset",
        "turn_mode": "interactive",
        "turn_status": "running",
        "final_response_text": "",
        "pending_context_summary": "",
        "agent_trace_id": "turn_plan_reset",
        "turn_plan": [],
    }
    first = await graph.ainvoke(
        {**base_state, "messages": [HumanMessage(content="do complex research")]},
        config={"configurable": {"thread_id": "chat_turn_plan_reset"}, "recursion_limit": 13},
    )
    second = await graph.ainvoke(
        {
            **base_state,
            "messages": [HumanMessage(content="simple follow up")],
            "agent_trace_id": "turn_plan_reset_second",
        },
        config={"configurable": {"thread_id": "chat_turn_plan_reset"}, "recursion_limit": 13},
    )

    assert first["final_response_text"] == "Done 2."
    assert second["final_response_text"] == "Done 3."
    second_turn_prompt = "\n".join(
        str(getattr(message, "content", "")) for message in captured_prompts[-1]
    )
    assert "CURRENT_TURN_PLAN" not in second_turn_prompt
    assert second.get("turn_plan") == []


@pytest.mark.asyncio
async def test_graph_does_not_inject_turn_plan_context_outside_interactive() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_messages: list[Any] = []

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_messages.extend(messages)
        return AIMessage(content="Wake done.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="wake task")],
            "customer_id": "telegram_test",
            "thread_id": "routine_wake_plan_block",
            "turn_mode": "routine_wake",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "routine_wake_plan_block",
            "turn_plan": [
                {
                    "id": "search",
                    "content": "This stale plan should not be injected",
                    "status": "in_progress",
                }
            ],
        },
        config={"configurable": {"thread_id": "routine_wake_plan_block"}, "recursion_limit": 8},
    )

    prompt_text = "\n".join(str(getattr(message, "content", "")) for message in captured_messages)
    assert result["final_response_text"] == "Wake done."
    assert "CURRENT_TURN_PLAN" not in prompt_text
    assert "This stale plan should not be injected" not in prompt_text


@pytest.mark.asyncio
async def test_graph_surfaces_tool_call_preamble_as_live_update() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    sequence: list[str] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            sequence.append("tool")
            return {"status": "ok"}

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        assert dedupe_key.startswith("tool_call_preamble:")
        assert thread_id == "chat_tool_preamble"
        sequence.append(f"emit:{text}")
        return {"sent": True, "duplicate": False}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="Черновик заполнен. Запускаю предпроверку:",
                tool_calls=[{"id": "call_preflight", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="Готово.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="работаешь?")],
            "customer_id": "telegram_test",
            "thread_id": "chat_tool_preamble",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_tool_preamble",
        },
        config={"configurable": {"thread_id": "chat_tool_preamble"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово."
    assert sequence == ["emit:Черновик заполнен. Запускаю предпроверку:", "tool"]


@pytest.mark.asyncio
async def test_graph_tool_call_preamble_not_suppressed_by_checkpointed_flag() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    sequence: list[str] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            sequence.append("tool")
            return {"status": "ok"}

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        assert dedupe_key.startswith("tool_call_preamble:")
        assert thread_id == "chat_tool_preamble_repeat"
        sequence.append(f"emit:{text}")
        return {"sent": True, "duplicate": False}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="Starting another tool call.",
                tool_calls=[{"id": "call_repeat", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="Done.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="again")],
            "customer_id": "telegram_test",
            "thread_id": "chat_tool_preamble_repeat",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_tool_preamble_repeat",
            "tool_preamble_update_sent": True,
        },
        config={
            "configurable": {"thread_id": "chat_tool_preamble_repeat"},
            "recursion_limit": 8,
        },
    )

    assert result["final_response_text"] == "Done."
    assert sequence == ["emit:Starting another tool call.", "tool"]


@pytest.mark.asyncio
async def test_graph_tool_execution_sets_thread_context_from_state() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    emitted: list[dict[str, str]] = []
    calls = 0

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        del thread_id
        emitted.append({"text": text, "dedupe_key": dedupe_key})
        return {"sent": True, "duplicate": False}

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_update",
                        "name": "send_owner_update",
                        "args": {"message": "Searching now."},
                    }
                ],
            )
        return AIMessage(content="Done.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = register_owner_update_tools(runtime)
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="search for examples")],
            "customer_id": "telegram_test",
            "thread_id": "chat_owner_context_from_state",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_owner_context_from_state",
        },
        config={
            "configurable": {"thread_id": "chat_owner_context_from_state"},
            "recursion_limit": 13,
        },
    )

    assert result["final_response_text"] == "Done."
    assert emitted == [{"text": "Searching now.", "dedupe_key": ""}]


@pytest.mark.asyncio
async def test_graph_surfaces_tool_call_preamble_for_workflow_setup() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    sequence: list[str] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            sequence.append("tool")
            return {"status": "ok"}

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        assert dedupe_key.startswith("tool_call_preamble:")
        assert thread_id == "chat_workflow_setup"
        sequence.append(f"emit:{text}")
        return {"sent": True, "duplicate": False}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="Черновик заполнен. Запускаю предпроверку:",
                tool_calls=[{"id": "call_preflight", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="Готово.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="работаешь?")],
            "customer_id": "telegram_test",
            "thread_id": "chat_workflow_setup",
            "turn_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_workflow_setup",
        },
        config={"configurable": {"thread_id": "chat_workflow_setup"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово."
    assert sequence == ["emit:Черновик заполнен. Запускаю предпроверку:", "tool"]


@pytest.mark.asyncio
async def test_graph_surfaces_default_progress_for_silent_workflow_setup_tool_call() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    sequence: list[str] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            sequence.append("tool")
            return {"status": "ok"}

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        assert dedupe_key.startswith("tool_call_preamble:")
        assert thread_id == "chat_workflow_setup"
        sequence.append(f"emit:{text}")
        return {"sent": True, "duplicate": False}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_setup",
                        "name": "tool_group_exec",
                        "args": {
                            "group": "intake",
                            "command": "intake_workflow_setup_update",
                            "args_json": {},
                        },
                    }
                ],
            )
        return AIMessage(content="Готово.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"tool_group_exec": _FakeTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="создай workflow")],
            "customer_id": "telegram_test",
            "thread_id": "chat_workflow_setup",
            "turn_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_workflow_setup",
        },
        config={"configurable": {"thread_id": "chat_workflow_setup"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово."
    assert sequence == [
        "emit:I’m setting up the workflow now. I’ll send the proposal or exact blocker when validation finishes.",
        "tool",
    ]


@pytest.mark.asyncio
async def test_graph_finalizes_successful_workflow_delete_when_model_omits_confirmation() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)

    class _DeleteTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {
                "ok": True,
                "deleted": True,
                "workflow_id": "iwf_123",
                "final_response_hint": "Deleted the intake workflow. It is gone now.",
            }

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="I'll look up the workflow and delete it.",
                tool_calls=[
                    {
                        "id": "call_delete",
                        "name": "intake_workflow_delete",
                        "args": {"workflow_id": "iwf_123"},
                    }
                ],
            )
        if calls == 3:
            return AIMessage(content="All set, I deleted that intake workflow.")
        return AIMessage(content="")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"intake_workflow_delete": _DeleteTool()}

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="delete the intake workflow")],
            "customer_id": "telegram_test",
            "thread_id": "chat_delete_workflow",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_delete_workflow",
        },
        config={"configurable": {"thread_id": "chat_delete_workflow"}, "recursion_limit": 8},
    )

    assert calls == 3
    assert result["final_response_text"] == "All set, I deleted that intake workflow."


@pytest.mark.asyncio
async def test_graph_finalizes_ready_workflow_proposal_when_model_omits_summary() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)

    class _ProposeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {
                "last_proposed_draft_hash": "hash_123",
                "draft_upsert": {
                    "name": "AutoSpa",
                    "channel": "telegram_business_dm",
                    "required_fields": ["service_name", "phone"],
                    "sink_type": "google_sheets_composio",
                },
                "preflight": {"ok": True, "status": "ready"},
                "final_response_hint": (
                    "Workflow proposal is ready.\n"
                    "- Name: AutoSpa\n"
                    "- Channel: telegram_business_dm\n"
                    "- Required fields: service_name, phone\n"
                    "- Sink: google_sheets_composio\n"
                    "Confirm/save to activate it, or tell me what to change."
                ),
            }

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_propose",
                        "name": "intake_workflow_setup_propose_current",
                        "args": {},
                    }
                ],
            )
        if calls == 3:
            return AIMessage(
                content=(
                    "I prepared the AutoSpa workflow proposal with Telegram Business DMs, "
                    "service name and phone collection, and Google Sheets as the sink. "
                    "You can confirm it to activate it or tell me what to change."
                )
            )
        return AIMessage(content="")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"intake_workflow_setup_propose_current": _ProposeTool()}

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="propose workflow")],
            "customer_id": "telegram_test",
            "thread_id": "chat_propose_workflow",
            "turn_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_propose_workflow",
        },
        config={"configurable": {"thread_id": "chat_propose_workflow"}, "recursion_limit": 8},
    )

    assert calls == 3
    text = result["final_response_text"]
    assert "I prepared the AutoSpa workflow proposal" in text
    assert "confirm it to activate it" in text
    assert "Workflow proposal is ready" not in text


@pytest.mark.asyncio
async def test_graph_does_not_surface_tool_call_preamble_for_background_turns() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    sequence: list[str] = []

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            sequence.append("tool")
            return {"status": "ok"}

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        del text, dedupe_key, thread_id
        sequence.append("emit")
        return {"sent": True, "duplicate": False}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="Проверяю входящий запрос:",
                tool_calls=[{"id": "call_check", "name": "fake_tool", "args": {}}],
            )
        return AIMessage(content="Готово.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="inbound customer message")],
            "customer_id": "telegram_customer",
            "thread_id": "chat_customer_ingest",
            "turn_mode": "event_notification",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_customer_ingest",
        },
        config={"configurable": {"thread_id": "chat_customer_ingest"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово."
    assert sequence == ["tool"]


@pytest.mark.asyncio
async def test_graph_does_not_duplicate_send_owner_update_preamble() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    emitted: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    class _SendOwnerUpdateTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            tool_calls.append(args)
            return {"sent": True, "duplicate": False}

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        del dedupe_key, thread_id
        emitted.append(text)
        return {"sent": True, "duplicate": False}

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            return AIMessage(
                content="Да, работаю!",
                tool_calls=[
                    {
                        "id": "call_update",
                        "name": "send_owner_update",
                        "args": {"message": "Да, работаю!"},
                    }
                ],
            )
        return AIMessage(content="Готово.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"send_owner_update": _SendOwnerUpdateTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="работаешь?")],
            "customer_id": "telegram_test",
            "thread_id": "chat_owner_update",
            "turn_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_owner_update",
        },
        config={"configurable": {"thread_id": "chat_owner_update"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово."
    assert emitted == []
    assert tool_calls == [{"message": "Да, работаю!"}]


@pytest.mark.asyncio
async def test_graph_adds_loop_limit_instruction_for_final_prose() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    saw_loop_instruction = False

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        nonlocal saw_loop_instruction
        saw_loop_instruction = any(
            "LOOP_LIMIT_APPROACHING" in str(getattr(message, "content", "")) for message in messages
        )
        return AIMessage(content="Current status: I am near the turn limit and stopping tools now.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="keep going")],
            "customer_id": "telegram_test",
            "thread_id": "chat_loop_limit",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_loop_limit",
        },
        config={"configurable": {"thread_id": "chat_loop_limit"}, "recursion_limit": 3},
    )

    assert saw_loop_instruction is True
    assert result["final_response_text"] == (
        "Current status: I am near the turn limit and stopping tools now."
    )


@pytest.mark.asyncio
async def test_graph_blocks_new_tool_calls_when_loop_limit_is_near() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    tool_invoked = False
    model_calls = 0

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            nonlocal tool_invoked
            tool_invoked = True
            return {"status": "ok"}

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal model_calls
        model_calls += 1
        if model_calls > 1:
            return AIMessage(content="I could not run more tools, so here is the current status.")
        return AIMessage(
            content="I will run one more tool:",
            tool_calls=[{"id": "call_more", "name": "fake_tool", "args": {}}],
        )

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="keep going")],
            "customer_id": "telegram_test",
            "thread_id": "chat_loop_limit_tools",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_loop_limit_tools",
        },
        config={"configurable": {"thread_id": "chat_loop_limit_tools"}, "recursion_limit": 5},
    )

    assert tool_invoked is False
    assert result["final_response_text"] == (
        "I could not run more tools, so here is the current status."
    )


@pytest.mark.asyncio
async def test_graph_carries_compact_tool_outcome_context_between_tool_rounds() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    calls = 0
    saw_tool_context = False

    class _FakeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            return {
                "ok": True,
                "answer": "SUV full wash costs 2500 rubles.",
                "headers": {"authorization": "secret"},
            }

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        nonlocal calls, saw_tool_context
        calls += 1
        if calls == 1:
            return AIMessage(
                content="Checking price:",
                tool_calls=[{"id": "call_price", "name": "fake_tool", "args": {}}],
            )
        prompt_text = "\n".join(str(getattr(message, "content", "")) for message in messages)
        saw_tool_context = (
            "Previous tool results" in prompt_text
            and "SUV full wash costs 2500 rubles" in prompt_text
            and "authorization" not in prompt_text
        )
        return AIMessage(content="SUV full wash is 2500 rubles.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._tools = {"fake_tool": _FakeTool()}

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="price for SUV full wash?")],
            "customer_id": "telegram_test",
            "thread_id": "chat_tool_context",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_tool_context",
        },
        config={"configurable": {"thread_id": "chat_tool_context"}, "recursion_limit": 12},
    )

    assert saw_tool_context is True
    assert result["final_response_text"] == "SUV full wash is 2500 rubles."


@pytest.mark.asyncio
async def test_graph_finalize_does_not_reuse_prior_turn_assistant_reply() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    call_count = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(content="first reply")
        return AIMessage(content="")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    graph = build_runtime_graph(runtime)
    config = {"configurable": {"thread_id": "chat-finalize-current-turn"}, "recursion_limit": 8}

    first = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="first user turn")],
            "customer_id": "telegram_test",
            "thread_id": "chat-finalize-current-turn",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_first",
        },
        config=config,
    )
    second = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="second user turn")],
            "customer_id": "telegram_test",
            "thread_id": "chat-finalize-current-turn",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_second",
        },
        config=config,
    )

    assert first["final_response_text"] == "first reply"
    assert second["final_response_text"] == ""


@pytest.mark.asyncio
async def test_graph_keeps_empty_output_without_retry() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    call_count = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(content="")
        return AIMessage(content="Here is the answer.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="answer please")],
            "customer_id": "telegram_test",
            "thread_id": "chat-empty-retry",
            "turn_mode": "interactive",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_empty_repair",
        },
        config={"configurable": {"thread_id": "chat-empty-retry"}, "recursion_limit": 8},
    )
    assert call_count == 1
    assert result["final_response_text"] == ""


def test_pending_context_surfaces_routine_execution_summary() -> None:
    text = format_pending_context(
        [
            {
                "source": "routine",
                "event_type": "scheduled",
                "payload": {
                    "routine_id": "rtn_mtsb77",
                    "routine_name": "daily-ai-oss-briefing",
                    "execution_status": "executed",
                    "execution_summary": "Briefing sent with three fresh stories.",
                    "notification_status": "sent",
                },
            }
        ]
    )

    assert "[routine/scheduled]" in text
    assert "routine_id=rtn_mtsb77" in text
    assert "execution_status=executed" in text
    assert "Briefing sent with three fresh stories" in text
    assert "notification_status=sent" in text


def test_tool_group_exec_progress_label_uses_group_and_command() -> None:
    message = OpenTulpaLangGraphRuntime._describe_tool_calls_for_progress(
        [
            {
                "name": "tool_group_exec",
                "args": {
                    "group": "composio",
                    "command": "GITHUB_LIST_PULL_REQUESTS",
                    "args_json": {"owner": "kvyb"},
                },
            }
        ]
    )

    assert message == "Composio: Github list pull requests…"
    assert "tool group exec" not in message.lower()


def test_batched_tool_group_exec_progress_label_uses_first_two_commands() -> None:
    message = OpenTulpaLangGraphRuntime._describe_tool_calls_for_progress(
        [
            {
                "name": "tool_group_exec",
                "args": {
                    "calls": [
                        {"group": "web", "command": "web_search", "args_json": {"query": "x"}},
                        {
                            "group": "web",
                            "command": "fetch_url_content",
                            "args_json": {"url": "https://example.com"},
                        },
                    ]
                },
            }
        ]
    )

    assert message == "Web: Search, then web: fetch url content…"
    assert "exec exec" not in message.lower()


@pytest.mark.asyncio
async def test_ainvoke_text_does_not_reuse_prior_turn_assistant_reply() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._graph = _AinvokeStaleMessageGraph()
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.0)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
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

    reply = await runtime.ainvoke_text(
        thread_id="chat-ainvoke-stale",
        customer_id="telegram_stale",
        text="current user",
    )

    assert reply == "I ran into an issue and could not produce a final response yet."


@pytest.mark.asyncio
async def test_workflow_setup_empty_no_tool_response_gets_repair_retry() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_prompts: list[list[Any]] = []
    behavior_events: list[str] = []

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_prompts.append(list(messages))
        if len(captured_prompts) == 1:
            return AIMessage(content="")
        return AIMessage(content="Обновил драфт и готов продолжать настройку.")

    _install_minimal_graph_runtime_stubs(
        runtime,
        ainvoke_model=_ainvoke_model,
        behavior_events=behavior_events,
    )
    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Вот spreadsheetId=abc, sheetName=Bookings.")],
            "customer_id": "telegram_test",
            "thread_id": "chat-workflow-setup-repair",
            "turn_mode": "workflow_setup",
            "prompt_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_repair",
        },
        config={"configurable": {"thread_id": "chat-workflow-setup-repair"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Обновил драфт и готов продолжать настройку."
    assert len(captured_prompts) == 2
    assert any(
        "WORKFLOW_SETUP_NO_PROGRESS" in str(getattr(message, "content", ""))
        for message in captured_prompts[1]
    )
    assert "graph.workflow_setup.no_progress_retry" in behavior_events


@pytest.mark.asyncio
async def test_workflow_setup_prompt_injects_authoritative_next_action() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_prompts: list[list[Any]] = []

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_prompts.append(list(messages))
        return AIMessage(content="Готово, показываю предложение.")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._workflow_setup_service = _FakeWorkflowSetupService(_ready_setup_session())
    graph = build_runtime_graph(runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Проверь и предложи workflow.")],
            "customer_id": "telegram_test",
            "thread_id": "chat-workflow-setup-context",
            "turn_mode": "workflow_setup",
            "prompt_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_context",
        },
        config={"configurable": {"thread_id": "chat-workflow-setup-context"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово, показываю предложение."
    prompt_text = "\n\n".join(
        str(getattr(message, "content", "")) for message in captured_prompts[0]
    )
    assert "WORKFLOW_SETUP_CONTROL_CARD" in prompt_text
    assert "draft_status: preflight_ready" in prompt_text
    assert "proposal_status: not_proposed" in prompt_text
    assert "Call intake_workflow_setup_propose_current" in prompt_text
    assert "After a ready preflight, do not re-query knowledge" in prompt_text


@pytest.mark.asyncio
async def test_workflow_setup_control_card_refreshes_after_tool_results() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_prompts: list[list[Any]] = []
    captured_prefix_counts: list[int] = []
    service = _FakeWorkflowSetupService(_ready_setup_session())

    class _FinalizeTool:
        async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
            del args
            completed = dict(_ready_setup_session())
            completed["status"] = "completed"
            completed["created_or_updated_workflow_id"] = "iwf_autospa"
            service.session = completed
            return {
                "ok": True,
                "session": completed,
            }

    calls = 0

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, kwargs
        nonlocal calls
        calls += 1
        captured_prompts.append(list(messages))
        captured_prefix_counts.append(stable_prefix_count)
        if calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{"id": "call_finalize", "name": "fake_finalize", "args": {}}],
            )
        return AIMessage(content="Workflow active with workflow_id=iwf_autospa.")

    async def _emit_update(
        *, text: str, dedupe_key: str = "", thread_id: str | None = None
    ) -> dict[str, bool]:
        del text, dedupe_key, thread_id
        return {"sent": True, "duplicate": False}

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    runtime._workflow_setup_service = service
    runtime._tools = {"fake_finalize": _FinalizeTool()}
    runtime.emit_interactive_update = _emit_update  # type: ignore[method-assign]
    graph = build_runtime_graph(runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Подтверждаю. Сохрани workflow.")],
            "customer_id": "telegram_test",
            "thread_id": "chat-workflow-setup-context",
            "turn_mode": "workflow_setup",
            "prompt_mode": "workflow_setup",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_context_refresh",
        },
        config={"configurable": {"thread_id": "chat-workflow-setup-context"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Workflow active with workflow_id=iwf_autospa."
    assert len(captured_prompts) == 2
    first_prompt = "\n\n".join(
        str(getattr(message, "content", "")) for message in captured_prompts[0]
    )
    second_prompt = "\n\n".join(
        str(getattr(message, "content", "")) for message in captured_prompts[1]
    )
    assert "session_status: active" in first_prompt
    assert "session_status: completed" in second_prompt
    assert "workflow_id=iwf_autospa" in second_prompt
    assert captured_prefix_counts == [4, 4]


@pytest.mark.asyncio
async def test_interactive_turn_promotes_to_workflow_setup_when_session_becomes_active() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_prompts: list[list[Any]] = []
    behavior_events: list[str] = []

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, stable_prefix_count, kwargs
        captured_prompts.append(list(messages))
        return AIMessage(content="Готово, показываю предложение.")

    _install_minimal_graph_runtime_stubs(
        runtime,
        ainvoke_model=_ainvoke_model,
        behavior_events=behavior_events,
    )
    runtime._workflow_setup_service = _FakeWorkflowSetupService(_ready_setup_session())
    graph = build_runtime_graph(runtime)

    result = await graph.ainvoke(
        {
            "messages": [HumanMessage(content="Создай workflow для мойки.")],
            "customer_id": "telegram_test",
            "thread_id": "chat-workflow-setup-context",
            "turn_mode": "interactive",
            "prompt_mode": "execution",
            "turn_status": "running",
            "final_response_text": "",
            "pending_context_summary": "",
            "agent_trace_id": "turn_promote",
        },
        config={"configurable": {"thread_id": "chat-workflow-setup-context"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "Готово, показываю предложение."
    assert result["turn_mode"] == "workflow_setup"
    assert result["prompt_mode"] == "workflow_setup"
    assert "graph.workflow_setup.promoted_turn_mode" in behavior_events
    prompt_text = "\n\n".join(
        str(getattr(message, "content", "")) for message in captured_prompts[0]
    )
    assert "WORKFLOW_SETUP_CONTROL_CARD" in prompt_text
    assert "Call intake_workflow_setup_propose_current" in prompt_text


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

    user_text = "can you try again?"
    reply = await runtime.ainvoke_text(
        thread_id="chat_test",
        customer_id="telegram_test",
        text=user_text,
        include_pending_context=True,
    )

    assert reply == "ok"
    assert graph.last_state is not None
    assert graph.last_state["active_skill_query"] == user_text
    model_messages = graph.last_state["messages"]
    assert len(model_messages) == 1
    assert isinstance(model_messages[0], HumanMessage)
    assert model_messages[0].content == user_text
    pending_text = str(graph.last_state.get("pending_context_summary", ""))
    assert "task_id=task_abc" in pending_text
    assert "scan my telegram" not in pending_text
    assert "raw_prompt" not in pending_text
    assert events.cleared == ("telegram_test", 42)


@pytest.mark.asyncio
async def test_concurrent_input_enqueues_next_turn_not_active_user_message() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    graph = _CapturingGraph()

    runtime._graph = graph
    runtime._thread_inputs = ThreadInputCoordinator(debounce_seconds=0.05)
    runtime._context_events = None
    runtime._link_alias_service = None
    runtime.recursion_limit = 8
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
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

    async def _submit(text: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return await runtime.ainvoke_text(
            thread_id="chat_test",
            customer_id="telegram_test",
            text=text,
            include_pending_context=True,
        )

    first, second = await asyncio.gather(
        _submit("first message", 0.0),
        _submit("are you here?", 0.01),
    )

    assert first == "ok"
    assert second == "ok"
    assert len(graph.states) == 2
    first_messages = graph.states[0]["messages"]
    second_messages = graph.states[1]["messages"]
    assert len(first_messages) == 1
    assert len(second_messages) == 1
    assert isinstance(first_messages[0], HumanMessage)
    assert isinstance(second_messages[0], HumanMessage)
    assert first_messages[0].content == "first message"
    assert second_messages[0].content == "are you here?"


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

    async def _load_skill_context_by_names(
        *, customer_id: str, skill_names: list[str]
    ) -> dict[str, Any]:
        del customer_id, skill_names
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

    runtime._model_with_tools = model
    runtime._checkpointer = InMemorySaver()
    runtime._list_available_skills = _unexpected_list  # type: ignore[method-assign]
    runtime._resolve_skill_context = _unexpected_resolve  # type: ignore[method-assign]
    runtime._load_skill_context_by_names = _load_skill_context_by_names  # type: ignore[method-assign]
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_thread_rollup = lambda thread_id: None  # type: ignore[assignment]
    runtime._thread_rollup_service = None
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {}
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8
    runtime._context_source_provider = RuntimeContextSourceProvider(runtime)

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
        "browser-use-operator" in str(getattr(msg, "content", "")) for msg in model.seen_messages
    )


@pytest.mark.asyncio
async def test_pre_resolve_skill_state_does_not_call_llm_selector() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    selector_calls = 0

    async def _list_available_skills(customer_id: str) -> list[dict[str, Any]]:
        assert customer_id == "telegram_test"
        return [
            {
                "name": "browser-use-operator",
                "description": "Use browser steps for dynamic websites.",
                "scope": "global",
            }
        ]

    async def _selector(**kwargs: Any) -> list[dict[str, Any]]:
        nonlocal selector_calls
        selector_calls += 1
        del kwargs
        return []

    runtime._list_available_skills = _list_available_skills  # type: ignore[method-assign]
    runtime._select_relevant_skills = _selector  # type: ignore[method-assign]
    provider = RuntimeContextSourceProvider(runtime)

    state = await pre_resolve_skill_state(
        provider,
        customer_id="telegram_test",
        user_text="use browser if needed",
        prompt_mode="task_chat",
    )

    assert selector_calls == 0
    assert state["active_skill_names"] == []
    assert state["active_available_skills"][0]["name"] == "browser-use-operator"


@pytest.mark.asyncio
async def test_interactive_prompt_keeps_core_policy_as_stable_prefix() -> None:
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

    async def _ainvoke_model(
        model: Any, messages: list[Any], *, stable_prefix_count: int = 0, **kwargs: Any
    ) -> AIMessage:
        del model
        captured["messages"] = messages
        captured["stable_prefix_count"] = stable_prefix_count
        captured["cacheable_prefix_count"] = kwargs.get("cacheable_prefix_count")
        captured["call_context"] = kwargs.get("call_context")
        return AIMessage(content="ok")

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8
    _install_prompt_source_stubs(runtime)

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
            "turn_plan": [
                {
                    "id": "search",
                    "content": "Search current evidence",
                    "status": "in_progress",
                }
            ],
        },
        config={"configurable": {"thread_id": "chat_test"}, "recursion_limit": 8},
    )

    assert result["final_response_text"] == "ok"
    assert captured["stable_prefix_count"] == 3
    prompt_messages = captured["messages"]
    anchor_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if isinstance(msg, HumanMessage)
        and "OpenTulpa cache anchor v1" in str(getattr(msg, "content", ""))
    )
    assert anchor_index == captured["stable_prefix_count"] - 1
    web_backend_messages = [
        msg
        for msg in prompt_messages
        if "WEB_SEARCH_BACKEND:" in str(getattr(msg, "content", ""))
    ]
    assert len(web_backend_messages) == 1
    current_turn_context_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if "OPENTULPA_CURRENT_TURN_CONTEXT" in str(getattr(msg, "content", ""))
    )
    assert current_turn_context_index >= captured["cacheable_prefix_count"]
    older_assistant_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if isinstance(msg, AIMessage)
        and "Gemini Flash for media analysis" in str(getattr(msg, "content", ""))
    )
    grounding_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if "Relevant long-term memory grounding" in str(getattr(msg, "content", ""))
    )
    assert older_assistant_index < grounding_index
    last_human_index = max(
        idx for idx, msg in enumerate(prompt_messages) if isinstance(msg, HumanMessage)
    )
    assert grounding_index < last_human_index
    plan_index = next(
        idx
        for idx, msg in enumerate(prompt_messages)
        if "CURRENT_TURN_PLAN" in str(getattr(msg, "content", ""))
    )
    assert plan_index >= captured["cacheable_prefix_count"]
    assert isinstance(captured["call_context"], dict)
    assert captured["call_context"]["call_site"] == "graph_agent"
    assert captured["call_context"]["_langfuse_graph_callback_covers_call"] is False
    assert "memory_grounding" in captured["call_context"]["prompt_sections"]
    assert "turn_plan" in captured["call_context"]["prompt_sections"]


@pytest.mark.asyncio
async def test_agent_uses_server_time_tool_guidance_instead_of_live_time_context() -> None:
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

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8
    _install_prompt_source_stubs(runtime)

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
    assert live_time_calls == 0
    assert captured_prefix_counts[0] == 4
    assert captured_prefix_counts[1] == captured_prefix_counts[0]
    anchor_index = next(
        idx
        for idx, msg in enumerate(captured_messages[0])
        if isinstance(msg, HumanMessage)
        and "OpenTulpa cache anchor v1" in str(getattr(msg, "content", ""))
    )
    assert anchor_index == captured_prefix_counts[0] - 1

    def _time_tool_guidance(messages: list[Any]) -> str:
        return next(
            str(getattr(msg, "content", ""))
            for msg in messages
            if 'tool_group_exec(group="memory", command="server_time", args_json={})'
            in str(getattr(msg, "content", ""))
        )

    first_time_guidance = _time_tool_guidance(captured_messages[0])
    second_time_guidance = _time_tool_guidance(captured_messages[1])
    assert "Live time context (auto-injected this turn)" not in first_time_guidance
    assert second_time_guidance == first_time_guidance
    assert (
        captured_messages[0][: captured_prefix_counts[0]]
        == captured_messages[1][: captured_prefix_counts[1]]
    )


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

    prior_messages: list[Any] = []
    for idx in range(14):
        prior_messages.append(
            HumanMessage(content=f"Earlier user note {idx}: keep this thread moving.")
        )
        prior_messages.append(AIMessage(content=f"Earlier assistant reply {idx}: acknowledged."))

    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8
    _install_prompt_source_stubs(runtime)

    graph = build_runtime_graph(runtime)
    result = await graph.ainvoke(
        {
            "messages": [
                *prior_messages,
                HumanMessage(content="run the fake tool and then answer"),
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

    assert result["final_response_text"] == "Done."
    assert len(captured_messages) == 2
    assert captured_prefix_counts[1] == captured_prefix_counts[0]
    assert (
        captured_messages[0][: captured_prefix_counts[0]]
        == captured_messages[1][: captured_prefix_counts[1]]
    )

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
    runtime._tools = {}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 8
    _install_prompt_source_stubs(runtime)

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
    human_texts = [
        str(getattr(msg, "content", ""))
        for msg in captured_messages
        if isinstance(msg, HumanMessage)
    ]
    assistant_texts = [
        str(getattr(msg, "content", "")) for msg in captured_messages if isinstance(msg, AIMessage)
    ]
    assert "current live ask" in human_texts
    assert not any(text.startswith("Earlier user note") for text in human_texts)
    assert not any(text.startswith("Earlier assistant reply") for text in assistant_texts)


@pytest.mark.asyncio
async def test_deepseek_prompt_collapses_completed_tool_segments() -> None:
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

    runtime.model_name = "deepseek/deepseek-v4-pro"
    runtime.openrouter_base_url = "https://openrouter.ai/api/v1"
    runtime._checkpointer = InMemorySaver()
    runtime._model_with_tools = object()
    runtime._thread_rollup_service = None
    runtime._load_active_directive = _directive  # type: ignore[method-assign]
    runtime._load_memory_grounding_context = _memory_grounding  # type: ignore[method-assign]
    runtime._build_live_time_context = _live_time  # type: ignore[method-assign]
    runtime._build_link_alias_context = lambda **kwargs: ""  # type: ignore[assignment]
    runtime._tools = {"fake_tool": _FakeTool()}
    runtime.ainvoke_model = _ainvoke_model  # type: ignore[method-assign]
    runtime.resolve_link_aliases_in_args = lambda **kwargs: kwargs.get("args", {})  # type: ignore[assignment]
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.log_behavior_event = lambda **kwargs: None  # type: ignore[assignment]
    runtime.model_with_tools_for_turn_mode = lambda turn_mode: object()  # type: ignore[assignment]
    runtime._context_token_limit = 12000
    runtime._context_short_term_low_tokens = 3500
    runtime.recursion_limit = 10
    _install_prompt_source_stubs(runtime)

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
    assert raw_tool_ids == []
    assert raw_ai_tool_ids == []
    third_prompt_text = "\n\n".join(
        str(getattr(message, "content", "")) for message in third_prompt
    )
    assert "VERIFIED_TOOL_RESULTS" in third_prompt_text
    assert "fake_tool" in third_prompt_text
    assert "step=1" in third_prompt_text
    assert "step=2" in third_prompt_text


def test_memory_grounding_block_stays_compact() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    memories = [
        {
            "kind": "directive_fact",
            "text": "Always be concise, direct, and avoid filler.",
            "score": 0.9,
        },
        {
            "kind": "life_fact",
            "text": "Timezone is UTC+8 and works mostly in the afternoon.",
            "score": 0.8,
        },
        {
            "kind": "aspirations_fact",
            "text": "Wants to launch more reliable Telegram and Instagram automation.",
            "score": 0.7,
        },
        {
            "kind": "workflow_fact",
            "text": "Runs an Instagram intake workflow that writes bookings to Google Sheets.",
            "score": 0.6,
        },
        {
            "kind": "code_fact",
            "text": "Main chat model is GLM 5.1 while media and memory use Gemini Flash.",
            "score": 0.65,
        },
        {
            "kind": "file_fact",
            "text": "Uploaded planning PDF is stored in tulpa_stuff/uploads for later recall.",
            "score": 0.55,
        },
        {
            "kind": "thread_context_rollup",
            "text": "Older thread context mentioning long implementation notes and stale discussion that should be deprioritized.",
            "score": 0.2,
        },
    ]

    block = runtime._build_memory_grounding_block(memories, token_budget=500)

    assert "Preferences and directives:" in block
    assert "Technical or code facts:" in block
    assert _approx_tokens(block) <= 520


def test_graph_input_preserves_frozen_prompt_state_between_turns() -> None:
    graph_input = build_graph_input(
        user_text="next turn",
        customer_id="telegram_test",
        thread_id="chat_test",
        turn_mode="interactive",
        prompt_mode="task_chat",
        pending_context_summary="",
        trace_id="trace_test",
        skill_state={},
    )

    assert "frozen_prompt_context" not in graph_input
    assert "frozen_history_projection" not in graph_input
    assert graph_input["messages"] == [HumanMessage(content="next turn")]


@pytest.mark.asyncio
async def test_graph_preserves_frozen_history_projection_across_user_turns() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    captured_projection_starts: list[int | None] = []

    async def _ainvoke_model(
        model: Any,
        messages: list[Any],
        *,
        stable_prefix_count: int = 0,
        **kwargs: Any,
    ) -> AIMessage:
        del model, messages, stable_prefix_count, kwargs
        return AIMessage(content="ok")

    _install_minimal_graph_runtime_stubs(runtime, ainvoke_model=_ainvoke_model)
    graph = build_runtime_graph(runtime)
    config = {"configurable": {"thread_id": "chat_projection"}, "recursion_limit": 8}

    first = await graph.ainvoke(
        build_graph_input(
            user_text="first",
            customer_id="telegram_test",
            thread_id="chat_projection",
            turn_mode="interactive",
            prompt_mode="task_chat",
            pending_context_summary="",
            trace_id="trace_first",
            skill_state={},
        ),
        config=config,
    )
    captured_projection_starts.append(
        first.get("frozen_history_projection", {}).get("turn_start_index")
    )

    second = await graph.ainvoke(
        build_graph_input(
            user_text="second",
            customer_id="telegram_test",
            thread_id="chat_projection",
            turn_mode="interactive",
            prompt_mode="task_chat",
            pending_context_summary="",
            trace_id="trace_second",
            skill_state={},
        ),
        config=config,
    )
    captured_projection_starts.append(
        second.get("frozen_history_projection", {}).get("turn_start_index")
    )

    assert captured_projection_starts == [0, 0]
