from __future__ import annotations

from typing import Any

from langchain_openrouter import ChatOpenRouter

from opentulpa.agent import runtime as runtime_module
from opentulpa.agent.lc_messages import AIMessage, ToolMessage


def test_runtime_passes_reasoning_effort_to_init_chat_model(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        calls.append({"model": model, **kwargs})
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com/v1",
        model_name="openai/gpt-5-mini",
        reasoning_effort="medium",
        wake_classifier_model_name="openai/gpt-5-mini",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert calls
    assert calls[0]["reasoning_effort"] == "medium"
    assert calls[0]["streaming"] is True


def test_runtime_caps_gemini_flash_lite_preview_output_tokens(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        calls.append({"model": model, **kwargs})
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com/v1",
        model_name="google/gemini-3.1-flash-lite-preview",
        wake_classifier_model_name="google/gemini-3.1-flash-lite-preview",
        wake_execution_model_name="google/gemini-3.1-flash-lite-preview",
        telegram_media_model_name="google/gemini-3.1-flash-lite-preview",
        max_completion_tokens=4096,
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert calls
    assert calls[0]["model"] == "google/gemini-3.1-flash-lite-preview"
    assert calls[0]["max_completion_tokens"] == 1000


def test_runtime_defaults_reasoning_effort_medium_for_all_agent_models(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        calls.append({"model": model, **kwargs})
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com/v1",
        model_name="deepseek/deepseek-v4-pro",
        wake_classifier_model_name="google/gemini-3-flash-preview",
        wake_execution_model_name="google/gemini-3-flash-preview",
        telegram_media_model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert calls
    assert all(call["reasoning_effort"] == "medium" for call in calls)
    deepseek_call = next(call for call in calls if call["model"] == "deepseek/deepseek-v4-pro")
    assert "extra_body" not in deepseek_call
    gemini_calls = [call for call in calls if call["model"] == "google/gemini-3-flash-preview"]
    assert gemini_calls
    assert all("extra_body" not in call for call in gemini_calls)


def test_runtime_can_disable_deepseek_v4_pro_thinking_with_empty_reasoning_effort(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        calls.append({"model": model, **kwargs})
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com/v1",
        model_name="deepseek/deepseek-v4-pro",
        reasoning_effort="",
        wake_classifier_model_name="google/gemini-3-flash-preview",
        wake_execution_model_name="google/gemini-3-flash-preview",
        telegram_media_model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    deepseek_call = next(call for call in calls if call["model"] == "deepseek/deepseek-v4-pro")
    assert deepseek_call["extra_body"] == {
        "reasoning": {"effort": "none"},
        "thinking": {"type": "disabled"},
    }
    gemini_calls = [call for call in calls if call["model"] == "google/gemini-3-flash-preview"]
    assert gemini_calls
    assert all("extra_body" not in call for call in gemini_calls)


def test_runtime_keeps_explicit_reasoning_effort_for_deepseek_v4_pro(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        calls.append({"model": model, **kwargs})
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com/v1",
        model_name="deepseek/deepseek-v4-pro",
        reasoning_effort="medium",
        wake_classifier_model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert calls
    assert calls[0]["reasoning_effort"] == "medium"
    assert "extra_body" not in calls[0]
    assert "reasoning" not in calls[0]


def test_runtime_uses_openrouter_adapter_for_deepseek_reasoning(monkeypatch) -> None:
    init_calls: list[dict[str, Any]] = []
    openrouter_calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        init_calls.append({"model": model, **kwargs})
        return object()

    class _FakeChatOpenRouter:
        def __init__(self, **kwargs: Any) -> None:
            openrouter_calls.append(kwargs)

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)
    monkeypatch.setattr(runtime_module, "ChatOpenRouter", _FakeChatOpenRouter)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-pro",
        reasoning_effort="high",
        wake_classifier_model_name="google/gemini-3-flash-preview",
        wake_execution_model_name="google/gemini-3-flash-preview",
        telegram_media_model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert openrouter_calls
    deepseek_call = openrouter_calls[0]
    assert deepseek_call["model"] == "deepseek/deepseek-v4-pro"
    assert deepseek_call["api_key"] == "test-key"
    assert deepseek_call["base_url"] == "https://openrouter.ai/api/v1"
    assert deepseek_call["reasoning"] == {"effort": "high", "exclude": False}
    assert deepseek_call["openrouter_provider"] == {
        "order": ["DeepSeek"],
        "allow_fallbacks": False,
    }
    assert deepseek_call["streaming"] is True
    assert deepseek_call["app_url"] == "https://github.com/kvyb/opentulpa"
    assert deepseek_call["app_title"] == "OpenTulpa"
    assert "reasoning_effort" not in deepseek_call
    assert all(call["model"] != "deepseek/deepseek-v4-pro" for call in init_calls)


def test_runtime_uses_openrouter_adapter_for_qwen_prompt_cache(monkeypatch) -> None:
    init_calls: list[dict[str, Any]] = []
    openrouter_calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        init_calls.append({"model": model, **kwargs})
        return object()

    class _FakeChatOpenRouter:
        def __init__(self, **kwargs: Any) -> None:
            openrouter_calls.append(kwargs)

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)
    monkeypatch.setattr(runtime_module, "ChatOpenRouter", _FakeChatOpenRouter)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="qwen/qwen3.7-max",
        reasoning_effort="medium",
        wake_classifier_model_name="google/gemini-3-flash-preview",
        wake_execution_model_name="google/gemini-3-flash-preview",
        telegram_media_model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert openrouter_calls
    qwen_call = openrouter_calls[0]
    assert qwen_call["model"] == "qwen/qwen3.7-max"
    assert qwen_call["api_key"] == "test-key"
    assert qwen_call["base_url"] == "https://openrouter.ai/api/v1"
    assert qwen_call["streaming"] is True
    assert qwen_call["app_url"] == "https://github.com/kvyb/opentulpa"
    assert qwen_call["app_title"] == "OpenTulpa"
    assert "reasoning" not in qwen_call
    assert "openrouter_provider" not in qwen_call
    assert all(call["model"] != "qwen/qwen3.7-max" for call in init_calls)


def test_runtime_uses_openrouter_adapter_to_disable_deepseek_reasoning(monkeypatch) -> None:
    openrouter_calls: list[dict[str, Any]] = []

    class _FakeChatOpenRouter:
        def __init__(self, **kwargs: Any) -> None:
            openrouter_calls.append(kwargs)

    monkeypatch.setattr(runtime_module, "ChatOpenRouter", _FakeChatOpenRouter)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="deepseek/deepseek-v4-pro",
        reasoning_effort="",
        wake_classifier_model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert openrouter_calls
    assert openrouter_calls[0]["reasoning"] == {"effort": "none", "exclude": False}
    assert openrouter_calls[0]["openrouter_provider"] == {
        "order": ["DeepSeek"],
        "allow_fallbacks": False,
    }


def test_openrouter_deepseek_adapter_replays_reasoning_details_for_tool_turns() -> None:
    model = ChatOpenRouter(
        model="deepseek/deepseek-v4-pro",
        api_key="test-key",
        reasoning={"effort": "high"},
    )
    raw_reasoning_details = [
        {
            "type": "reasoning.text",
            "text": "encrypted-or-provider-reasoning-block",
            "signature": "provider-signature",
        }
    ]
    raw_tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "tulpa_read_file",
            "arguments": '{"path":"price.xlsx"}',
        },
    }
    response = model._create_chat_result(  # type: ignore[attr-defined]
        {
            "id": "resp_test",
            "model": "deepseek/deepseek-v4-pro",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "provider reasoning text",
                        "reasoning_details": raw_reasoning_details,
                        "tool_calls": [raw_tool_call],
                    },
                }
            ],
        }
    )

    assistant_message = response.generations[0].message
    assert isinstance(assistant_message, AIMessage)
    assert assistant_message.additional_kwargs["reasoning_content"] == "provider reasoning text"
    assert assistant_message.additional_kwargs["reasoning_details"] == raw_reasoning_details

    replay_messages, params = model._create_message_dicts(  # type: ignore[attr-defined]
        [
            assistant_message,
            ToolMessage(content='{"status":"ok"}', tool_call_id="call_1"),
        ],
        stop=None,
    )

    assert params["reasoning"] == {"effort": "high"}
    assert replay_messages[0]["reasoning"] == "provider reasoning text"
    assert replay_messages[0]["reasoning_details"] == raw_reasoning_details
    assert replay_messages[0]["tool_calls"][0]["id"] == "call_1"
    assert replay_messages[1]["tool_call_id"] == "call_1"


def test_runtime_sets_openrouter_app_headers_on_model_init(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        calls.append({"model": model, **kwargs})
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)
    monkeypatch.delenv("OPENROUTER_APP_TITLE", raising=False)

    runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://openrouter.ai/api/v1",
        model_name="openai/gpt-5-mini",
        wake_classifier_model_name="openai/gpt-5-mini",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert calls
    assert calls[0]["default_headers"] == {
        "HTTP-Referer": "https://github.com/kvyb/opentulpa",
        "X-OpenRouter-Title": "OpenTulpa",
    }
    assert calls[0]["use_responses_api"] is False
