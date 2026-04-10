from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from opentulpa.agent.lc_messages import HumanMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.logging.posthog import _extract_openrouter_cost_fields, create_posthog_logger


def test_create_posthog_logger_requires_both_env_values() -> None:
    assert create_posthog_logger(api_key=None, host="https://us.i.posthog.com") is None
    assert create_posthog_logger(api_key="phc_test", host=None) is None
    assert create_posthog_logger(api_key="", host="") is None


def test_create_posthog_logger_builds_callbacks_when_sdk_available(monkeypatch) -> None:
    fake_posthog = types.ModuleType("posthog")
    fake_posthog_ai = types.ModuleType("posthog.ai")
    fake_posthog_langchain = types.ModuleType("posthog.ai.langchain")
    fake_posthog_callbacks = types.ModuleType("posthog.ai.langchain.callbacks")

    class _FakePosthog:
        def __init__(self, api_key: str, host: str) -> None:
            self.api_key = api_key
            self.host = host

        def shutdown(self) -> None:
            return None

    class _FakeCallbackHandler:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    fake_posthog.Posthog = _FakePosthog
    fake_posthog_langchain.CallbackHandler = _FakeCallbackHandler
    fake_posthog_callbacks.ChatGeneration = object
    fake_posthog_callbacks._capture_exception_and_update_properties = lambda *args, **kwargs: args[-1]
    fake_posthog_callbacks._convert_message_to_dict = lambda message: message
    fake_posthog_callbacks._extract_raw_response = lambda generation: generation
    fake_posthog_callbacks._get_http_status = lambda error: 500
    fake_posthog_callbacks._parse_usage = lambda output, provider=None, model=None: types.SimpleNamespace(
        input_tokens=None,
        output_tokens=None,
        cache_write_tokens=None,
        cache_read_tokens=None,
        reasoning_tokens=None,
    )
    fake_posthog_callbacks._stringify_exception = lambda error: str(error)
    fake_posthog_callbacks.sanitize_langchain = lambda value: value
    fake_posthog_callbacks.with_privacy_mode = lambda client, privacy_mode, value: value

    monkeypatch.setitem(sys.modules, "posthog", fake_posthog)
    monkeypatch.setitem(sys.modules, "posthog.ai", fake_posthog_ai)
    monkeypatch.setitem(sys.modules, "posthog.ai.langchain", fake_posthog_langchain)
    monkeypatch.setitem(sys.modules, "posthog.ai.langchain.callbacks", fake_posthog_callbacks)

    logger = create_posthog_logger(api_key="phc_test", host="https://us.i.posthog.com")

    assert logger is not None
    callbacks = logger.build_callbacks(
        distinct_id="telegram_123",
        trace_id="turn_abc",
        properties={"thread_id": "chat_xyz"},
    )

    assert len(callbacks) == 1
    callback = callbacks[0]
    assert callback.kwargs["distinct_id"] == "telegram_123"
    assert callback.kwargs["trace_id"] == "turn_abc"
    assert callback.kwargs["properties"]["thread_id"] == "chat_xyz"


def test_extract_openrouter_cost_fields_reads_usage_cost_from_completion_payload() -> None:
    output = types.SimpleNamespace(
        llm_output={
            "usage": {
                "cost": 0.023471989,
                "cost_details": {
                    "prompt": 0.017,
                    "completion": 0.006471989,
                },
            }
        },
        generations=[],
    )

    fields = _extract_openrouter_cost_fields(output)

    assert fields["$ai_total_cost_usd"] == 0.023471989
    assert fields["openrouter_prompt_cost_usd"] == 0.017
    assert fields["openrouter_completion_cost_usd"] == 0.006471989


class _FakeCallbackLogger:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build_callbacks(self, *, distinct_id: str | None, trace_id: str | None, properties: dict[str, Any] | None) -> list[Any]:
        self.calls.append(
            {
                "distinct_id": distinct_id,
                "trace_id": trace_id,
                "properties": dict(properties or {}),
            }
        )
        return ["posthog-callback"]

    def shutdown(self) -> None:
        return None


class _ConfigurableModel:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def with_config(self, config: dict[str, Any]) -> "_ConfigurableModel":
        self.configs.append(config)
        return self

    async def ainvoke(self, messages: object, **kwargs: object) -> object:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return types.SimpleNamespace(content="ok", tool_calls=[], usage={})


@pytest.mark.asyncio
async def test_prepare_turn_context_adds_posthog_callbacks_to_graph_config() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.recursion_limit = 8
    runtime._posthog_logger = _FakeCallbackLogger()
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.expand_link_aliases = lambda **kwargs: str(kwargs.get("text", ""))  # type: ignore[assignment]
    runtime._build_pending_context_summary = lambda **kwargs: ("", None)  # type: ignore[assignment]

    async def _skill_state(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {}

    runtime._pre_resolve_skill_state = _skill_state  # type: ignore[assignment]

    async def _noop_compact(*, thread_id: str, customer_id: str) -> None:
        del thread_id, customer_id

    async def _no_pending_lock(*, customer_id: str, thread_id: str) -> bool:
        del customer_id, thread_id
        return False

    runtime._maybe_compact_thread_context = _noop_compact  # type: ignore[method-assign]
    runtime._has_pending_approval_lock = _no_pending_lock  # type: ignore[method-assign]

    prepared = await runtime._prepare_turn_context(
        thread_id="chat_test",
        customer_id="telegram_test",
        text="hello",
        turn_mode="interactive",
        include_pending_context=True,
        trace_id="turn_test",
    )

    assert prepared is not None
    assert prepared.config["callbacks"] == ["posthog-callback"]
    assert runtime._posthog_logger.calls[0]["distinct_id"] == "telegram_test"
    assert runtime._posthog_logger.calls[0]["trace_id"] == "turn_test"


@pytest.mark.asyncio
async def test_ainvoke_model_attaches_posthog_callbacks_with_with_config(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._posthog_logger = _FakeCallbackLogger()
    model = _ConfigurableModel()

    await runtime.ainvoke_model(
        model,
        [HumanMessage(content="hi")],
        model_name="google/gemini-3-flash-preview",
        call_context={
            "call_site": "graph_agent",
            "customer_id": "telegram_test",
            "thread_id": "chat_test",
            "trace_id": "turn_test",
            "turn_mode": "interactive",
            "prompt_mode": "literal_chat",
        },
    )

    assert model.configs
    assert model.configs[0]["callbacks"] == ["posthog-callback"]
