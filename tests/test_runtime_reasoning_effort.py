from __future__ import annotations

from typing import Any

from opentulpa.agent import runtime as runtime_module


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
        guardrail_classifier_model_name="openai/gpt-5-mini",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    assert calls
    assert calls[0]["reasoning_effort"] == "medium"
