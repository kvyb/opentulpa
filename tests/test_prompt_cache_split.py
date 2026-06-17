"""Tests for stable/volatile prompt split and provider-agnostic caching extras."""

from __future__ import annotations

import pytest

from opentulpa.agent.composio_context import load_connected_composio_toolkits_context
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.model_pool import prompt_cache_breakpoint_message_index
from opentulpa.agent.prompt_cache_policy import (
    build_prompt_cache_plan,
)
from opentulpa.agent.prompt_policy import build_system_prompt_message
from opentulpa.agent.prompt_sections import PROMPT_DYNAMIC_BOUNDARY
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.agent.turn_prompt_builder.frozen_context import (
    _build_late_turn_control_text,
)


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
    cache: dict[str, object] = {}

    text = await load_connected_composio_toolkits_context(
        composio=composio,
        cache=cache,
        customer_id="telegram_1",
    )

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

    cached_text = await load_connected_composio_toolkits_context(
        composio=composio,
        cache=cache,
        customer_id="telegram_1",
    )
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


def test_prompt_cache_profile_zai_glm52_is_automatic() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.2",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    profile = rt.prompt_cache_profile()
    assert profile["strategy"] == "automatic"
    assert profile["supports_top_level"] is False
    assert profile["supports_breakpoints"] is False


def test_prompt_cache_profile_qwen_uses_implicit_stable_prefix() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="qwen/qwen3.7-max",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )

    profile = rt.prompt_cache_profile()

    assert profile["strategy"] == "implicit_stable_prefix"
    assert profile["supports_top_level"] is False
    assert profile["supports_breakpoints"] is False
    assert profile["cache_control"] == {}


def test_prompt_cache_profile_minimax_uses_implicit_stable_prefix() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="minimax/minimax-m3",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )

    profile = rt.prompt_cache_profile()

    assert profile["strategy"] == "implicit_stable_prefix"
    assert profile["supports_top_level"] is False
    assert profile["supports_breakpoints"] is False
    assert profile["cache_control"] == {}


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


def test_prepare_messages_for_qwen_leaves_content_unmarked_for_implicit_cache() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="qwen/qwen3.7-max",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    messages = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="OpenTulpa cache anchor v1. Real conversation messages follow."),
        AIMessage(content="Prior assistant answer"),
        ToolMessage(content='{"ok": true}', tool_call_id="call_1"),
        HumanMessage(content="Prior user turn"),
        HumanMessage(content="Current user turn"),
    ]

    prepared = rt.prepare_messages_for_prompt_cache(
        messages,
        stable_prefix_count=2,
        cacheable_prefix_count=2,
    )

    assert prepared[0].content == "Stable system prompt"
    assert prepared[1].content == "OpenTulpa cache anchor v1. Real conversation messages follow."
    assert prepared[2].content == "Prior assistant answer"
    assert prepared[3].content == '{"ok": true}'
    assert prepared[4].content == "Prior user turn"
    assert prepared[5].content == "Current user turn"


def test_prompt_cache_plan_qwen_implicit_uses_stable_prefix_only() -> None:
    messages = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="OpenTulpa cache anchor v1. Real conversation messages follow."),
        HumanMessage(content="Committed history"),
        HumanMessage(content="Current user turn"),
    ]

    plan = build_prompt_cache_plan(
        prefix_messages=messages[:2],
        older_history_messages=[HumanMessage(content="older " * 900)],
        frozen_late_messages=[],
        latest_turn_messages=messages[2:],
        dynamic_late_messages=[SystemMessage(content="dynamic")],
        cache_profile={"strategy": "implicit_stable_prefix", "supports_breakpoints": False},
    )

    assert plan.cache_breakpoint_index is None
    assert plan.cacheable_prefix_count == 2
    assert plan.cacheable_prefix_mode == "stable_prefix_only"
    assert plan.cacheable_history_messages == []
    assert plan.model_messages == [
        *messages[:2],
        HumanMessage(content="older " * 900),
        *messages[2:],
        SystemMessage(content="dynamic"),
    ]


def test_prompt_cache_breakpoint_index_matches_actual_cacheable_message() -> None:
    messages = [
        SystemMessage(content="Stable system prompt"),
        HumanMessage(content="OpenTulpa cache anchor v1"),
        AIMessage(content="", tool_calls=[{"name": "tool_group_exec", "args": {}, "id": "call_1"}]),
        HumanMessage(content="Current user turn"),
    ]

    index = prompt_cache_breakpoint_message_index(messages, effective_prefix_count=3)

    assert index == 1


def test_prompt_cache_plan_explicit_stable_prefix_marks_only_stable_boundary() -> None:
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
        cache_profile={
            "strategy": "explicit_stable_prefix",
            "supports_breakpoints": True,
            "cache_control": {"type": "ephemeral"},
        },
    )

    assert plan.model_messages[:2] == prefix
    assert plan.requested_cacheable_prefix_count == 2
    assert plan.cacheable_prefix_count == 2
    assert plan.cache_breakpoint_index == 1
    assert plan.cacheable_prefix_mode == "stable_prefix_only"
    assert plan.cacheable_history_messages == []
    assert plan.frontier_history_messages == latest
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


def test_extract_response_usage_fields_normalizes_native_deepseek_cache_usage() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )

    class _UsageResponse:
        content = "ok"
        usage = {
            "prompt_tokens": 10124,
            "completion_tokens": 5,
            "total_tokens": 10129,
            "prompt_cache_hit_tokens": 10112,
            "prompt_cache_miss_tokens": 12,
        }

    assert rt.extract_response_usage_fields(_UsageResponse()) == {
        "native_tokens_prompt": 10124,
        "native_tokens_completion": 5,
        "native_tokens_total": 10129,
        "native_tokens_cached": 10112,
        "cache_hit": True,
        "native_tokens_cache_write": 12,
    }


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


def test_extract_response_usage_fields_normalizes_langchain_stream_usage_metadata() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="qwen/qwen3.7-max",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )

    class _UsageMetadataResponse:
        content = "ok"
        usage_metadata = {
            "input_tokens": 4519,
            "output_tokens": 277,
            "total_tokens": 4796,
            "input_token_details": {"cache_read": 4504},
            "output_token_details": {"reasoning": 271},
        }

    assert rt.extract_response_usage_fields(_UsageMetadataResponse()) == {
        "native_tokens_prompt": 4519,
        "native_tokens_completion": 277,
        "native_tokens_total": 4796,
        "native_tokens_cached": 4504,
        "cache_hit": True,
        "native_tokens_reasoning": 271,
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


@pytest.mark.parametrize("model_name", ["qwen/qwen3.7-max", "minimax/minimax-m3"])
@pytest.mark.asyncio
async def test_ainvoke_model_adds_openrouter_session_id_for_implicit_cache_stickiness(
    model_name: str,
) -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name=model_name,
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=True,
    )
    model = _CaptureModel()

    await rt.ainvoke_model(
        model,
        [
            SystemMessage(content="Stable system prompt"),
            HumanMessage(content="OpenTulpa cache anchor v1. Real conversation messages follow."),
            HumanMessage(content="Dynamic user question"),
        ],
        model_name=model_name,
        stable_prefix_count=2,
        call_context={"thread_id": "thread_1", "customer_id": "cust_1"},
    )

    call = model.calls[0]
    extra_body = call["kwargs"]["extra_body"]
    assert extra_body["session_id"].startswith("opentulpa-")
    assert len(extra_body["session_id"]) == 42


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
