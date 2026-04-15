"""Tests for stable/volatile prompt split and provider-agnostic caching extras."""

from __future__ import annotations

import pytest

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.prompt_policy import build_system_prompt_message
from opentulpa.agent.prompt_sections import PROMPT_DYNAMIC_BOUNDARY
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def test_prompt_dynamic_boundary_marker_is_single_line_prefix() -> None:
    assert PROMPT_DYNAMIC_BOUNDARY.startswith("[OPENTULPA_PROMPT_DYNAMIC_BOUNDARY]")


def test_full_runtime_policy_retains_hardened_rules() -> None:
    text = str(build_system_prompt_message().content)
    assert "[SECTION A] Core Behavior" in text
    assert "[SECTION B] Scheduling And Routines" in text
    assert "[SECTION C] Tool Selection" in text
    assert "[SECTION D] Claim Discipline And Approvals" in text
    assert PROMPT_DYNAMIC_BOUNDARY not in text


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
async def test_ainvoke_model_routes_glm51_with_extended_provider_order() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )
    model = _ProviderRouteCaptureModel()

    response = await rt.ainvoke_model(
        model,
        [HumanMessage(content="Dynamic user question")],
        model_name="z-ai/glm-5.1",
    )

    assert isinstance(response, _CaptureResponse)
    assert len(model.calls) == 1
    provider = model.calls[0]["kwargs"]["extra_body"]["provider"]
    assert provider == {
        "order": ["fireworks", "siliconflow", "friendli", "inceptron", "atlas-cloud"],
        "allow_fallbacks": False,
    }


def test_model_request_attempts_skip_glm51_provider_routing_off_openrouter() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://example.com/v1",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )

    assert rt._model_request_attempts(model_name="z-ai/glm-5.1") == [
        {"name": "default", "invoke_extras": {}, "call_context": {}}
    ]


def test_model_request_attempts_route_glm51_nitro_variants_on_openrouter() -> None:
    rt = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="z-ai/glm-5.1:nitro",
        checkpoint_db_path=".opentulpa/test-prompt-cache.sqlite",
        prompt_caching_enabled=False,
    )

    attempts = rt._model_request_attempts(model_name="z-ai/glm-5.1:nitro")

    assert attempts == [
        {
            "name": "glm51_ordered_providers",
            "invoke_extras": {
                "extra_body": {
                    "provider": {
                        "order": [
                            "fireworks",
                            "siliconflow",
                            "friendli",
                            "inceptron",
                            "atlas-cloud",
                        ],
                        "allow_fallbacks": False,
                    }
                }
            },
            "call_context": {
                "provider_route": "glm51_ordered_providers",
                "provider_order": [
                    "fireworks",
                    "siliconflow",
                    "friendli",
                    "inceptron",
                    "atlas-cloud",
                ],
                "provider_allow_fallbacks": False,
            },
        }
    ]


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
