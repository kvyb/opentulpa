"""Tests for stable/volatile prompt split and provider-agnostic caching extras."""

from __future__ import annotations

import pytest

from opentulpa.agent.graph_builder import (
    _build_connected_composio_toolkits_context,
    _build_late_turn_control_text,
)
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.model_pool import prompt_cache_breakpoint_message_index
from opentulpa.agent.prompt_cache_policy import (
    build_prompt_cache_plan,
    qwen_cache_safe_history_count,
    split_qwen_cacheable_history,
    tail_sized_history_message_count,
)
from opentulpa.agent.prompt_policy import build_system_prompt_message
from opentulpa.agent.prompt_sections import PROMPT_DYNAMIC_BOUNDARY
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _PromptComposio:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def list_connected_accounts(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(dict(kwargs))
        return {
            "items": [
                {"toolkit_slug": "github", "status": "ACTIVE"},
                {"toolkit_slug": "instagram", "status": "ACTIVE"},
                {"toolkit_slug": "github", "status": "ACTIVE"},
                {"toolkit_slug": "googlesheets", "status": "INACTIVE"},
                {"toolkit_slug": "", "status": "ACTIVE"},
            ]
        }


class _PromptComposioRuntime:
    def __init__(self, composio: _PromptComposio | None) -> None:
        self._composio_service = composio

    @property
    def composio_service(self) -> _PromptComposio | None:
        return self._composio_service


def test_prompt_dynamic_boundary_marker_is_single_line_prefix() -> None:
    assert PROMPT_DYNAMIC_BOUNDARY.startswith("[OPENTULPA_PROMPT_DYNAMIC_BOUNDARY]")


def test_full_runtime_policy_retains_hardened_rules() -> None:
    text = str(build_system_prompt_message().content)
    assert "[SECTION A] Core Behavior" in text
    assert "[SECTION B] Scheduling And Routines" in text
    assert "[SECTION C] Tool Selection" in text
    assert "[SECTION D] Claim Discipline And Execution" in text
    assert PROMPT_DYNAMIC_BOUNDARY not in text
    assert "Available via Composio tool for this customer" not in text


@pytest.mark.asyncio
async def test_connected_composio_toolkits_context_is_dynamic_and_cached() -> None:
    composio = _PromptComposio()
    runtime = _PromptComposioRuntime(composio)

    text = await _build_connected_composio_toolkits_context(runtime, "telegram_1")

    assert text.startswith("Available via Composio tool for this customer: github, instagram.")
    assert 'tool_group_exec(group="composio")' in text
    assert "googlesheets" not in text
    assert composio.calls == [
        {
            "customer_id": "telegram_1",
            "statuses": ["ACTIVE"],
            "limit": 20,
        }
    ]

    cached_text = await _build_connected_composio_toolkits_context(runtime, "telegram_1")
    assert cached_text == text
    assert len(composio.calls) == 1


def test_late_turn_control_can_include_connected_composio_toolkits() -> None:
    text = _build_late_turn_control_text(
        customer_id="telegram_1",
    )

    assert PROMPT_DYNAMIC_BOUNDARY in text
    assert 'tool_group_exec(group="memory", command="server_time", args_json={})' in text
    assert "Live time context (auto-injected this turn)" not in text
    assert "Prompt mode:" not in text
    assert "Turn mode:" not in text


def test_model_invoke_extras_empty_when_caching_disabled() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="anthropic/claude-sonnet-4",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )
    assert rt.model_invoke_extras() == {}


def test_model_invoke_extras_anthropic_when_enabled() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="anthropic/claude-sonnet-4",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    assert rt.model_invoke_extras() == {"extra_body": {"cache_control": {"type": "ephemeral"}}}


def test_model_invoke_extras_skips_non_claude_models() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-2.0-flash-001",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    assert rt.model_invoke_extras() == {}
    assert rt.prompt_cache_profile()["strategy"] == "breakpoint"


def test_model_invoke_extras_gemini_3_uses_breakpoints() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    assert rt.model_invoke_extras() == {}
    profile = rt.prompt_cache_profile()
    assert profile["strategy"] == "breakpoint"
    assert profile["supports_breakpoints"] is True


def test_model_invoke_extras_claude_slug_without_anthropic_prefix() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="openrouter/auto-claude-foo",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    assert rt.model_invoke_extras() == {"extra_body": {"cache_control": {"type": "ephemeral"}}}


def test_model_invoke_extras_ttl_1h() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="anthropic/claude-sonnet-4",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
        prompt_cache_ttl_1h=True,
    )
    assert rt.model_invoke_extras() == {
        "extra_body": {"cache_control": {"type": "ephemeral", "ttl": "1h"}}
    }


def test_prompt_cache_profile_openai_is_automatic() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="openai/gpt-5-mini",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    profile = rt.prompt_cache_profile()
    assert profile["strategy"] == "automatic"
    assert profile["supports_top_level"] is False
    assert profile["supports_breakpoints"] is False


def test_prompt_cache_profile_zai_glm_is_automatic() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    profile = rt.prompt_cache_profile()
    assert profile["strategy"] == "automatic"
    assert profile["supports_top_level"] is False
    assert profile["supports_breakpoints"] is False


def test_prompt_cache_profile_qwen_uses_explicit_committed_breakpoint() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="qwen/qwen3.7-max",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )

    profile = rt.prompt_cache_profile()

    assert profile["strategy"] == "explicit_committed_breakpoint"
    assert profile["supports_top_level"] is False
    assert profile["supports_breakpoints"] is True


def test_prepare_messages_for_prompt_cache_wraps_stable_system_message_for_gemini_by_default() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    messages = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="Dynamic user question"),
    ]

    prepared = rt.prepare_messages_for_prompt_cache(messages)

    assert isinstance(prepared[0].content, list)
    stable_block = prepared[0].content[0]
    assert stable_block["type"] == "text"
    assert stable_block["text"] == "Stable system prompt"
    assert stable_block["cache_control"] == {"type": "ephemeral"}
    assert prepared[1].content == "Dynamic user question"


def test_prepare_messages_for_prompt_cache_skips_when_no_stable_system_prefix() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    messages = [HumanMessage(content="Dynamic user question")]

    prepared = rt.prepare_messages_for_prompt_cache(messages)

    assert prepared[0].content == "Dynamic user question"


def test_prepare_messages_for_prompt_cache_prefers_stable_prefix_when_provided() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    messages = [
        SystemMessage(content="Stable system prompt"),
        SystemMessage(content="Stable skills context"),
        HumanMessage(content="Dynamic user question"),
    ]

    prepared = rt.prepare_messages_for_prompt_cache(messages, stable_prefix_count=2)

    assert prepared[0].content == "Stable system prompt"
    assert isinstance(prepared[1].content, list)
    stable_block = prepared[1].content[0]
    assert stable_block["type"] == "text"
    assert stable_block["text"] == "Stable skills context"
    assert stable_block["cache_control"] == {"type": "ephemeral"}
    assert prepared[2].content == "Dynamic user question"


def test_prepare_messages_for_qwen_wraps_latest_cacheable_history_before_current_turn() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="qwen/qwen3.7-max",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    messages = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="OpenTulpa cache anchor v1"),
        AIMessage(content="Prior assistant answer"),
        ToolMessage(content='{"ok": true}', tool_call_id="call_1"),
        HumanMessage(content="Prior user turn"),
        HumanMessage(content="Current user turn"),
    ]

    prepared = rt.prepare_messages_for_prompt_cache(
        messages,
        stable_prefix_count=2,
        cacheable_prefix_count=5,
    )

    assert prepared[0].content == "Stable system prompt"
    assert prepared[1].content == "OpenTulpa cache anchor v1"
    assert prepared[2].content == "Prior assistant answer"
    assert prepared[3].content == '{"ok": true}'
    assert isinstance(prepared[4].content, list)
    cache_block = prepared[4].content[0]
    assert cache_block["type"] == "text"
    assert cache_block["text"] == "Prior user turn"
    assert cache_block["cache_control"] == {"type": "ephemeral"}
    assert prepared[5].content == "Current user turn"


def test_prompt_cache_breakpoint_index_matches_actual_cacheable_message() -> None:
    messages = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="OpenTulpa cache anchor v1"),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_1"}]),
        HumanMessage(content="Current user turn"),
    ]

    index = prompt_cache_breakpoint_message_index(messages, effective_prefix_count=3)

    assert index == 1


def test_qwen_tail_sized_history_count_keeps_only_small_suffix_volatile() -> None:
    assert tail_sized_history_message_count([]) == 0
    assert tail_sized_history_message_count([120, 120]) == 0
    assert tail_sized_history_message_count([1000, 1000, 1000, 150, 120]) == 3
    assert tail_sized_history_message_count([5000, 1000]) == 2


def test_qwen_history_split_keeps_current_user_in_frontier() -> None:
    cacheable, frontier, mode = split_qwen_cacheable_history(
        older_history_messages=[],
        latest_turn_messages=[HumanMessage(content="Current user turn")],
    )

    assert cacheable == []
    assert len(frontier) == 1
    assert mode == "stable_prefix_tail_only"


def test_qwen_history_split_caches_workflow_tool_loop_head() -> None:
    latest_turn = [
        HumanMessage(content="INTERNAL_ONBOARDING_SEED " + ("setup facts " * 1400)),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_1"}]),
        ToolMessage(content="{\"ok\": true, \"result\": \"" + ("draft " * 1200) + "\"}", tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_2"}]),
        ToolMessage(content="{\"ok\": true, \"result\": \"" + ("preflight " * 1200) + "\"}", tool_call_id="call_2"),
    ]

    cacheable, frontier, mode = split_qwen_cacheable_history(
        older_history_messages=[],
        latest_turn_messages=latest_turn,
    )

    assert len(cacheable) == 1
    assert isinstance(cacheable[0], HumanMessage)
    assert str(cacheable[0].content).startswith("QWEN_CACHEABLE_COMMITTED_HISTORY")
    assert "INTERNAL_ONBOARDING_SEED" in str(cacheable[0].content)
    assert frontier == latest_turn[1:]
    assert mode == "committed_tail_sized_history"


def test_qwen_history_split_does_not_end_cacheable_prefix_on_pending_tool_call() -> None:
    latest_turn = [
        HumanMessage(content="INTERNAL_ONBOARDING_SEED. " + ("setup facts " * 900)),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_1"}]),
        ToolMessage(content="{\"ok\": true, \"result\": \"" + ("draft " * 500) + "\"}", tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_2"}]),
    ]

    cacheable, frontier, mode = split_qwen_cacheable_history(
        older_history_messages=[],
        latest_turn_messages=latest_turn,
        target_tail_tokens=0,
    )

    assert isinstance(cacheable[0], HumanMessage)
    assert str(cacheable[0].content).startswith("QWEN_CACHEABLE_COMMITTED_HISTORY")
    assert "internal_onboarding_seed" in str(cacheable[0].content)
    assert frontier == latest_turn[1:]
    assert mode == "committed_tail_sized_history"


def test_qwen_safe_history_count_rolls_back_pending_tool_call() -> None:
    latest_turn = [
        HumanMessage(content="setup facts"),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_1"}]),
        ToolMessage(content='{"ok": true}', tool_call_id="call_1"),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_2"}]),
    ]

    assert qwen_cache_safe_history_count(latest_turn, 4) == 3
    assert qwen_cache_safe_history_count(latest_turn, 2) == 1


def test_prompt_cache_plan_encapsulates_qwen_message_order_and_counts() -> None:
    prefix = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="OpenTulpa cache anchor v1"),
    ]
    older = [HumanMessage(content="older " * 900)]
    latest = [
        HumanMessage(content="INTERNAL_ONBOARDING_SEED. " + ("setup facts " * 900)),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_1"}]),
    ]
    dynamic = [SystemMessage(content="dynamic late")]

    plan = build_prompt_cache_plan(
        prefix_messages=prefix,
        older_history_messages=older,
        frozen_late_messages=[],
        latest_turn_messages=latest,
        dynamic_late_messages=dynamic,
        cache_profile={"strategy": "explicit_committed_breakpoint", "supports_breakpoints": True},
    )

    assert plan.model_messages[:2] == prefix
    assert plan.requested_cacheable_prefix_count == 3
    assert plan.cacheable_prefix_count == 3
    assert plan.cache_breakpoint_index == 2
    assert plan.cacheable_prefix_mode == "committed_tail_sized_history"
    assert len(plan.cacheable_history_messages) == 1
    assert str(plan.cacheable_history_messages[0].content).startswith(
        "QWEN_CACHEABLE_COMMITTED_HISTORY"
    )
    assert str(plan.frontier_history_messages[0].content).startswith(
        "QWEN_VOLATILE_RECENT_HISTORY"
    )
    assert plan.frontier_history_messages[1:] == latest[1:]
    assert plan.model_messages[-1] == dynamic[0]


class _CaptureResponse:
    def __init__(self) -> None:
        self.content = "ok"


class _CaptureModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages: object, **kwargs: object) -> _CaptureResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _CaptureResponse()


class _ProviderRouteCaptureModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages: object, **kwargs: object) -> _CaptureResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _CaptureResponse()


@pytest.mark.asyncio
async def test_ainvoke_model_adds_breakpoint_content_for_gemini() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    model = _CaptureModel()

    await rt.ainvoke_model(
        model,
        [
            SystemMessage(content="Stable system prompt"),
            HumanMessage(content="Dynamic user question"),
        ],
        model_name="google/gemini-3-flash-preview",
    )

    call = model.calls[0]
    assert call["kwargs"] == {}
    sent_messages = call["messages"]
    assert isinstance(sent_messages, list)
    assert sent_messages[0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert sent_messages[1].content == "Dynamic user question"


@pytest.mark.asyncio
async def test_ainvoke_model_keeps_deepseek_v4_pro_reasoning_with_default_medium_effort() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )
    model = _ProviderRouteCaptureModel()

    response = await rt.ainvoke_model(
        model,
        [HumanMessage(content="Dynamic user question")],
        model_name="deepseek/deepseek-v4-pro",
    )

    assert isinstance(response, _CaptureResponse)
    assert len(model.calls) == 1
    assert model.calls[0]["kwargs"] == {}


@pytest.mark.asyncio
async def test_ainvoke_model_can_disable_deepseek_v4_pro_reasoning() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-pro",
        reasoning_effort="",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )
    model = _ProviderRouteCaptureModel()

    response = await rt.ainvoke_model(
        model,
        [HumanMessage(content="Dynamic user question")],
        model_name="deepseek/deepseek-v4-pro",
    )

    assert isinstance(response, _CaptureResponse)
    assert len(model.calls) == 1
    assert model.calls[0]["kwargs"] == {}


def test_model_request_attempts_are_default_off_openrouter() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://example.com/v1",
        model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )

    assert rt._model_request_attempts(model_name="deepseek/deepseek-v4-pro") == [
        {"name": "default", "invoke_extras": {}, "call_context": {}}
    ]


def test_model_request_attempts_are_default_for_deepseek_v4_pro_on_openrouter() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )

    attempts = rt._model_request_attempts(model_name="deepseek/deepseek-v4-pro")

    assert attempts == [{"name": "default", "invoke_extras": {}, "call_context": {}}]


def test_extract_response_usage_fields_normalizes_openrouter_usage() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )

    class _UsageResponse:
        content = "ok"
        usage = {
            "prompt_tokens": 13515,
            "completion_tokens": 46,
            "total_tokens": 13561,
            "prompt_tokens_details": {
                "cached_tokens": 7592,
                "cache_write_tokens": 5923,
            },
            "completion_tokens_details": {
                "reasoning_tokens": 0,
            },
        }

    assert rt.extract_response_usage_fields(_UsageResponse()) == {
        "native_tokens_prompt": 13515,
        "native_tokens_completion": 46,
        "native_tokens_total": 13561,
        "native_tokens_cached": 7592,
        "cache_hit": True,
        "native_tokens_cache_write": 5923,
        "native_tokens_reasoning": 0,
    }


@pytest.mark.asyncio
async def test_ainvoke_model_adds_breakpoint_to_stable_prefix_for_gemini() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    model = _CaptureModel()

    await rt.ainvoke_model(
        model,
        [
            SystemMessage(content="Stable system prompt"),
            SystemMessage(content="Stable skills context"),
            HumanMessage(content="Dynamic user question"),
        ],
        model_name="google/gemini-3-flash-preview",
        stable_prefix_count=2,
    )

    call = model.calls[0]
    assert call["kwargs"] == {}
    sent_messages = call["messages"]
    assert isinstance(sent_messages, list)
    assert sent_messages[1].content[0]["cache_control"] == {"type": "ephemeral"}
    assert sent_messages[2].content == "Dynamic user question"


@pytest.mark.asyncio
async def test_ainvoke_model_adds_top_level_cache_control_for_claude() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="anthropic/claude-sonnet-4",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    model = _CaptureModel()

    await rt.ainvoke_model(
        model,
        [SystemMessage(content="Stable system prompt"), HumanMessage(content="Dynamic user question")],
        model_name="anthropic/claude-sonnet-4",
    )

    call = model.calls[0]
    assert call["kwargs"] == {"extra_body": {"cache_control": {"type": "ephemeral"}}}
