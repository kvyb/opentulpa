"""Tests for stable/volatile prompt split and provider-agnostic caching extras."""

from __future__ import annotations

from opentulpa.agent.prompt_policy import build_system_prompt_message
from opentulpa.agent.prompt_sections import PROMPT_DYNAMIC_BOUNDARY, build_core_policy_message
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def test_prompt_dynamic_boundary_marker_is_single_line_prefix() -> None:
    assert PROMPT_DYNAMIC_BOUNDARY.startswith("[OPENTULPA_PROMPT_DYNAMIC_BOUNDARY]")


def test_stable_core_policy_is_only_core_message() -> None:
    text = str(build_core_policy_message().content)
    assert "You are OpenTulpa." in text
    assert PROMPT_DYNAMIC_BOUNDARY not in text


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
