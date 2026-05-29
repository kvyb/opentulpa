from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from opentulpa.agent.lc_messages import HumanMessage
from opentulpa.agent.runtime import (
    OpenTulpaLangGraphRuntime,
    _langchain_callback_metadata,
    _tool_schema_trace_fields,
)
from opentulpa.agent.runtime_context_provider import RuntimeContextSourceProvider
from opentulpa.agent.turn_context_preparer import prepare_turn_context
from opentulpa.logging.langfuse import (
    LangfuseTracer,
    create_langfuse_tracer,
    redact_for_langfuse,
)


class _FakeObservation:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.id = "a" * 16
        self.updates: list[dict[str, Any]] = []
        self.ended = False

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


class _FakeObservationContext:
    def __init__(self, client: _FakeLangfuseClient, kwargs: dict[str, Any]) -> None:
        self.client = client
        self.observation = _FakeObservation(kwargs)

    def __enter__(self) -> _FakeObservation:
        self.client.observations.append(self.observation)
        self.client.current_trace_id = (self.observation.kwargs.get("trace_context", {}) or {}).get(
            "trace_id"
        ) or "generated_trace_id"
        return self.observation

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.observation.end()
        self.client.current_trace_id = None
        return False


class _FakeLangfuseClient:
    def __init__(self) -> None:
        self.observations: list[_FakeObservation] = []
        self.flushed = False
        self.shutdown_called = False
        self.current_trace_id: str | None = None

    def create_trace_id(self, *, seed: str) -> str:
        return ("f" * 32) if seed else ""

    def start_as_current_observation(self, **kwargs: Any) -> _FakeObservationContext:
        return _FakeObservationContext(self, kwargs)

    def start_observation(self, **kwargs: Any) -> _FakeObservation:
        observation = _FakeObservation(kwargs)
        self.observations.append(observation)
        return observation

    def get_current_trace_id(self) -> str | None:
        return self.current_trace_id

    def flush(self) -> None:
        self.flushed = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeCallbackHandler:
    instances = 0
    init_kwargs: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).instances += 1
        type(self).init_kwargs.append(dict(kwargs))


def test_create_langfuse_tracer_requires_full_config() -> None:
    assert (
        create_langfuse_tracer(
            public_key=None,
            secret_key="sk",
            base_url="https://cloud.langfuse.com",
        )
        is None
    )
    assert create_langfuse_tracer(public_key="pk", secret_key="sk", base_url=None) is None


def test_create_langfuse_tracer_enabled_with_keys_and_base_url() -> None:
    tracer = create_langfuse_tracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        deployment_tag="test-deploy",
    )

    assert tracer is not None
    assert tracer.enabled is True
    assert tracer.deployment_tag == "test-deploy"
    assert tracer.environment == "test-deploy"


def test_langfuse_environment_defaults_to_railway_service_name(monkeypatch) -> None:
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "OpenTulpa Alpha")

    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=_FakeLangfuseClient(),
    )

    assert tracer.deployment_tag == "OpenTulpa Alpha"
    assert tracer.environment == "opentulpa-alpha"


def test_langfuse_environment_override_is_normalized_and_installed(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    _FakeCallbackHandler.instances = 0
    _FakeCallbackHandler.init_kwargs = []
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        deployment_tag="ignored",
        environment="LANGFUSE Prod!",
        client=_FakeLangfuseClient(),
        callback_handler_cls=_FakeCallbackHandler,
    )

    with tracer.trace_context(
        name="opentulpa.turn.interactive",
        trace_id="turn_1",
        user_id="cust_1",
        session_id="thread_1",
    ):
        callbacks = tracer.build_callbacks(
            user_id="cust_1",
            trace_id="turn_1",
            session_id="thread_1",
            metadata=None,
            tags=None,
        )

    assert callbacks
    assert tracer.environment == "env-langfuse-prod"
    assert os.environ["LANGFUSE_TRACING_ENVIRONMENT"] == "env-langfuse-prod"
    assert _FakeCallbackHandler.init_kwargs[0] == {}


def test_langfuse_callbacks_skip_without_active_root_span() -> None:
    _FakeCallbackHandler.instances = 0
    _FakeCallbackHandler.init_kwargs = []
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=_FakeLangfuseClient(),
        callback_handler_cls=_FakeCallbackHandler,
    )

    callbacks = tracer.build_callbacks(
        user_id="cust_1",
        trace_id="turn_1",
        session_id="thread_1",
        metadata={"call_site": "graph_agent"},
        tags=["interactive"],
    )

    assert callbacks == []
    assert _FakeCallbackHandler.init_kwargs == []


def test_langfuse_callbacks_attach_to_active_root_span() -> None:
    _FakeCallbackHandler.instances = 0
    _FakeCallbackHandler.init_kwargs = []
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=_FakeLangfuseClient(),
        callback_handler_cls=_FakeCallbackHandler,
    )

    with tracer.trace_context(
        name="opentulpa.turn.interactive",
        trace_id="turn_1",
        user_id="cust_1",
        session_id="thread_1",
    ):
        callbacks = tracer.build_callbacks(
            user_id="cust_1",
            trace_id="turn_1",
            session_id="thread_1",
            metadata=None,
            tags=["interactive"],
        )

    assert callbacks
    assert _FakeCallbackHandler.init_kwargs == [{}]


def test_langchain_callback_metadata_stringifies_non_string_values() -> None:
    metadata = _langchain_callback_metadata(
        {
            "call_site": "graph_agent",
            "bound_tool_count": 1,
            "bound_tool_names": ["_trace_lookup_tool"],
            "bound_tool_schema_chars": 123,
            "empty": None,
        }
    )

    assert metadata == {
        "call_site": "graph_agent",
        "bound_tool_count": "1",
        "bound_tool_names": '["_trace_lookup_tool"]',
        "bound_tool_schema_chars": "123",
    }


def test_langfuse_trace_context_uses_active_root_span_and_deployment_tag() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        deployment_tag="carwash-test",
        client=client,
    )

    with tracer.trace_context(
        name="opentulpa.turn.interactive",
        trace_id="turn_123",
        user_id="cust_1",
        session_id="thread_1",
        metadata={"turn_mode": "interactive"},
        tags=["interactive"],
    ):
        pass

    observation = client.observations[0]
    assert "trace_context" not in observation.kwargs
    assert observation.kwargs["metadata"]["deployment_tag"] == "carwash-test"
    assert observation.kwargs["metadata"]["environment"] == "carwash-test"
    assert observation.kwargs["metadata"]["turn_mode"] == "interactive"
    assert observation.kwargs["metadata"]["opentulpa_trace_id"] == "turn_123"
    assert observation.ended is True
    assert "env:carwash-test" in tracer.tags(["interactive"])


def test_record_generation_captures_usage_and_cost() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    tracer.record_generation(
        {
            "model_name": "z-ai/glm-5.1",
            "call_site": "graph_agent",
            "trace_id": "turn_1",
            "prompt_messages": [{"role": "user", "text": "hi"}],
            "response_text": "hello",
            "response_content": "hello",
            "response_tool_calls": [],
            "native_tokens_prompt": 10,
            "native_tokens_completion": 5,
            "native_tokens_total": 15,
            "native_tokens_cached": 3,
            "native_tokens_reasoning": 2,
            "native_cost_prompt_usd": 0.01,
            "native_cost_completion_usd": 0.02,
            "native_cost_usd": 0.03,
        }
    )

    observation = client.observations[0]
    assert observation.kwargs["as_type"] == "generation"
    assert observation.kwargs["model"] == "z-ai/glm-5.1"
    assert observation.kwargs["usage_details"] == {
        "input": 10,
        "output": 5,
        "total": 15,
        "cache_read_input_tokens": 3,
        "reasoning_output_tokens": 2,
    }
    assert observation.kwargs["cost_details"] == {"input": 0.01, "output": 0.02, "total": 0.03}


def test_record_generation_maps_native_deepseek_cache_usage() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    tracer.record_generation(
        {
            "model_name": "deepseek/deepseek-v4-pro",
            "call_site": "graph_agent",
            "trace_id": "turn_1",
            "prompt_messages": [{"role": "user", "text": "hi"}],
            "response_text": "hello",
            "usage": {
                "prompt_tokens": 10124,
                "completion_tokens": 5,
                "total_tokens": 10129,
                "prompt_cache_hit_tokens": 10112,
                "prompt_cache_miss_tokens": 12,
            },
        }
    )

    observation = client.observations[0]
    assert observation.kwargs["usage_details"] == {
        "input": 10124,
        "output": 5,
        "total": 10129,
        "cache_read_input_tokens": 10112,
        "cache_write_input_tokens": 12,
    }


def test_trace_context_rolls_up_child_generation_usage_and_cost() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    with tracer.trace_context(
        name="opentulpa.interactive.turn",
        trace_id="turn_1",
        user_id="cust_1",
        session_id="thread_1",
    ):
        tracer.record_generation(
            {
                "model_name": "model-a",
                "call_site": "graph_agent",
                "trace_id": "turn_1",
                "native_tokens_prompt": 10,
                "native_tokens_completion": 5,
                "native_tokens_total": 15,
                "native_cost_prompt_usd": 0.01,
                "native_cost_completion_usd": 0.02,
                "native_cost_usd": 0.03,
            }
        )
        tracer.record_generation(
            {
                "model_name": "model-b",
                "call_site": "tool_repair",
                "trace_id": "turn_1",
                "native_tokens_prompt": 7,
                "native_tokens_completion": 3,
                "native_tokens_total": 10,
                "native_cost_prompt_usd": 0.004,
                "native_cost_completion_usd": 0.006,
                "native_cost_usd": 0.01,
            }
        )

    root = client.observations[0]
    assert root.kwargs["name"] == "opentulpa.interactive.turn"
    assert root.updates[-1]["usage_details"] == {"input": 17, "output": 8, "total": 25}
    assert root.updates[-1]["cost_details"] == pytest.approx(
        {"input": 0.014, "output": 0.026, "total": 0.04}
    )


def test_record_generation_maps_openrouter_upstream_cost_details() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    tracer.record_generation(
        {
            "model_name": "z-ai/glm-5.1",
            "call_site": "graph_agent",
            "trace_id": "turn_1",
            "prompt_messages": [{"role": "user", "text": "hi"}],
            "response_text": "hello",
            "native_cost_details": {
                "upstream_inference_prompt_cost": 0.004,
                "upstream_inference_completions_cost": 0.006,
                "upstream_inference_cost": 0.01,
            },
        }
    )

    observation = client.observations[0]
    assert observation.kwargs["cost_details"] == {"input": 0.004, "output": 0.006, "total": 0.01}


def test_record_generation_skips_without_trace_context() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    tracer.record_generation(
        {
            "model_name": "z-ai/glm-5.1",
            "call_site": "runtime_model_invoke",
            "prompt_messages": [{"role": "user", "text": "hi"}],
            "response_text": "hello",
        }
    )

    assert client.observations == []


def test_tool_span_captures_status_and_side_effects() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    with tracer.tool_span(
        trace_id="turn_1",
        tool_name="send_message",
        tool_call_id="call_1",
        args={"authorization": "Bearer secret", "text": "hello"},
    ) as span:
        tracer.record_behavior_event(
            {
                "event": "message.sent",
                "customer_id": "cust_1",
                "authorization": "Bearer secret",
            }
        )
        span.set_result({"status": "queued", "token": "secret"}, status="queued")

    observation = client.observations[0]
    assert observation.kwargs["as_type"] == "tool"
    assert observation.kwargs["input"]["authorization"] == "[redacted]"
    update = observation.updates[0]
    assert update["metadata"]["status"] == "queued"
    assert update["metadata"]["side_effect_count"] == 1
    assert update["metadata"]["side_effects"][0]["payload"]["authorization"] == "[redacted]"
    assert update["output"]["token"] == "[redacted]"


def test_tool_span_inherits_active_trace_context() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    with (
        tracer.trace_context(
            name="opentulpa.turn.interactive",
            trace_id="turn_1",
            user_id="cust_1",
            session_id="thread_1",
        ),
        tracer.tool_span(trace_id="turn_1", tool_name="send_message"),
    ):
        pass

    root, tool = client.observations
    assert "trace_context" not in root.kwargs
    assert "trace_context" not in tool.kwargs


def test_tool_span_marks_errors() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )

    with pytest.raises(RuntimeError), tracer.tool_span(trace_id="turn_1", tool_name="broken"):
        raise RuntimeError("boom")

    update = client.observations[0].updates[0]
    assert update["metadata"]["status"] == "error"
    assert update["level"] == "ERROR"


def test_redaction_covers_secrets_and_inline_media() -> None:
    redacted = redact_for_langfuse(
        {
            "Authorization": "Bearer secret",
            "api_key": "secret",
            "password": "secret",
            "image": "data:image/png;base64,AAAA",
            "audio": {"type": "input_audio", "data": "base64-audio"},
        }
    )

    assert redacted["Authorization"] == "[redacted]"
    assert redacted["api_key"] == "[redacted]"
    assert redacted["password"] == "[redacted]"
    assert redacted["image"] == "data:image/png;base64,[redacted]"
    assert redacted["audio"]["data"] == "[redacted-inline-media]"


class _FakeCallbackTracer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.generations: list[dict[str, Any]] = []

    def build_callbacks(
        self,
        *,
        user_id: str | None,
        trace_id: str | None,
        session_id: str | None,
        metadata: dict[str, Any] | None,
        tags: list[str] | None,
    ) -> list[Any]:
        self.calls.append(
            {
                "user_id": user_id,
                "trace_id": trace_id,
                "session_id": session_id,
                "metadata": dict(metadata or {}),
                "tags": list(tags or []),
            }
        )
        return ["langfuse-callback"]

    def record_generation(self, record: dict[str, Any]) -> None:
        self.generations.append(record)


class _ConfigurableModel:
    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []

    def with_config(self, config: dict[str, Any]) -> _ConfigurableModel:
        self.configs.append(config)
        return self

    async def ainvoke(self, messages: object, **kwargs: object) -> object:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return type("Response", (), {"content": "ok", "tool_calls": [], "usage": {}})()


@pytest.mark.asyncio
async def test_prepare_turn_context_adds_langfuse_callbacks_to_graph_config() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.recursion_limit = 8
    runtime._langfuse_tracer = _FakeCallbackTracer()
    runtime._tools = {}
    runtime._context_events = None
    runtime.register_links_from_text = lambda **kwargs: []  # type: ignore[assignment]
    runtime.expand_link_aliases = lambda **kwargs: str(kwargs.get("text", ""))  # type: ignore[assignment]

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
    prepared = await prepare_turn_context(
        runtime.context_source_provider,
        thread_id="chat_test",
        customer_id="telegram_test",
        text="hello",
        turn_mode="interactive",
        include_pending_context=True,
        trace_id="turn_test",
        recursion_limit_override=None,
        forced_skill_names=None,
        prompt_mode_override=None,
        build_langfuse_callbacks=runtime._build_langfuse_callbacks,
        tool_schema_trace_fields=lambda mode: _tool_schema_trace_fields(runtime, mode),
        langchain_callback_metadata=_langchain_callback_metadata,
    )

    assert prepared is not None
    assert prepared.config["callbacks"] == ["langfuse-callback"]
    assert prepared.graph_input["langfuse_graph_callback_attached"] is True
    assert runtime._langfuse_tracer.calls[0]["user_id"] == "telegram_test"
    assert runtime._langfuse_tracer.calls[0]["trace_id"] == "turn_test"
    assert runtime._langfuse_tracer.calls[0]["session_id"] == "chat_test"
    assert len(prepared.config["metadata"]["bound_tool_schema_hash"]) == 64
    assert len(runtime._langfuse_tracer.calls[0]["metadata"]["bound_tool_schema_hash"]) == 64


@pytest.mark.asyncio
async def test_ainvoke_model_attaches_langfuse_callbacks_with_with_config(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._langfuse_tracer = _FakeCallbackTracer()
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
    assert model.configs[0]["callbacks"] == ["langfuse-callback"]
    assert runtime._langfuse_tracer.calls[0]["trace_id"] == "turn_test"
    assert runtime._langfuse_tracer.calls[0]["metadata"]["call_site"] == "graph_agent"
    assert len(runtime._langfuse_tracer.calls[0]["metadata"]["bound_tool_schema_hash"]) == 64
    assert len(model.configs[0]["metadata"]["bound_tool_schema_hash"]) == 64
    assert runtime._langfuse_tracer.generations == []


@pytest.mark.asyncio
async def test_ainvoke_model_skips_langfuse_cloud_when_graph_callback_covers_call(
    tmp_path: Path,
) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._langfuse_tracer = _FakeCallbackTracer()
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"
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
            "_langfuse_graph_callback_covers_call": True,
        },
    )

    assert model.configs == []
    assert runtime._langfuse_tracer.calls == []
    assert runtime._langfuse_tracer.generations == []
    assert runtime._llm_call_trace_path.exists()
