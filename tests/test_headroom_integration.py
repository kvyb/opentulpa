from __future__ import annotations

from typing import Any

from opentulpa.agent import runtime as runtime_module
from opentulpa.integrations.headroom import HeadroomService


def test_headroom_service_keeps_small_results_verbatim() -> None:
    service = HeadroomService(passthrough_token_limit=100, result_token_budget=80)

    result = service.compress_tool_result(
        tool_name="search",
        args={"query": "refund"},
        result={"status": "ok", "count": 2},
        user_text="find the refund count",
    )

    assert result.startswith("{")
    assert '"status":"ok"' in result
    assert '"count":2' in result


def test_headroom_service_compacts_large_results_without_sdk() -> None:
    service = HeadroomService(
        passthrough_token_limit=20,
        result_token_budget=60,
        result_value_char_limit=40,
    )
    service._sdk_checked = True
    service._compress_fn = None

    result = service.compress_tool_result(
        tool_name="web_search",
        args={"query": "refund policy"},
        result={
            "status": "ok",
            "items": [
                {
                    "title": "Refund policy " + ("x" * 180),
                    "snippet": "Detailed explanation " + ("y" * 220),
                }
                for _ in range(8)
            ],
        },
        user_text="what is the refund policy",
    )

    assert result.startswith("status=ok")
    assert "items[0].title=" in result
    assert "x" * 100 not in result
    assert "y" * 100 not in result


def test_headroom_service_uses_sdk_for_large_results() -> None:
    service = HeadroomService(
        passthrough_token_limit=20,
        result_token_budget=220,
        result_value_char_limit=40,
    )
    raw_result = {
        "status": "ok",
        "items": [
            {
                "id": idx,
                "title": "Refund policy " + ("x" * 120),
                "snippet": "Detailed explanation " + ("y" * 180),
            }
            for idx in range(50)
        ],
    }

    result = service.compress_tool_result(
        tool_name="web_search",
        args={"query": "refund policy"},
        result=raw_result,
        user_text="what is the refund policy",
    )

    raw_text = runtime_module._safe_json(raw_result)
    assert result
    assert result != raw_text
    assert len(result) < len(raw_text)
    assert result.startswith("{")


def test_runtime_compress_tool_result_for_model_uses_headroom_service(monkeypatch) -> None:
    def _fake_init_chat_model(model: str | None = None, **kwargs: Any) -> object:
        _ = model
        _ = kwargs
        return object()

    monkeypatch.setattr(runtime_module, "init_chat_model", _fake_init_chat_model)

    runtime = runtime_module.OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="test-key",
        openrouter_base_url="https://example.com/v1",
        model_name="openai/gpt-5-mini",
        wake_classifier_model_name="openai/gpt-5-mini",
        wake_execution_model_name="openai/gpt-5-mini",
        telegram_media_model_name="openai/gpt-5-mini",
        checkpoint_db_path=".opentulpa/test.sqlite",
    )

    result = runtime.compress_tool_result_for_model(
        tool_name="web_search",
        args={"query": "refund policy"},
        result={"items": [{"title": "x" * 400, "snippet": "y" * 400} for _ in range(4)]},
        user_text="summarize the refund policy",
    )

    assert result
    assert result != runtime_module._safe_json(
        {"items": [{"title": "x" * 400, "snippet": "y" * 400} for _ in range(4)]}
    )
    assert "items[0].title=" in result
