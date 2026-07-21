from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import pytest

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
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        deployment_tag="ignored",
        environment="LANGFUSE Prod!",
        client=_FakeLangfuseClient(),
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


def test_langfuse_callbacks_skip_without_active_root_span() -> None:
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=_FakeLangfuseClient(),
    )

    callbacks = tracer.build_callbacks(
        user_id="cust_1",
        trace_id="turn_1",
        session_id="thread_1",
        metadata={"call_site": "graph_agent"},
        tags=["interactive"],
    )

    assert callbacks == []


def test_langfuse_callbacks_attach_to_active_root_span() -> None:
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=_FakeLangfuseClient(),
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


def test_langfuse_callback_records_timing_without_raw_model_or_tool_content() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )
    prompt_secret = "123456789:telegram-bot-token-value"
    bearer_secret = "bearer-callback-secret"
    password_secret = "callback-password"
    prompt = (
        f"Bot token - {prompt_secret}; "
        f"Authorization: Bearer {bearer_secret}; password is {password_secret}"
    )

    with tracer.trace_context(
        name="opentulpa.turn.interactive",
        trace_id="turn_1",
        user_id="cust_1",
        session_id="thread_1",
        input={"messages": [{"role": "user", "content": prompt}]},
    ):
        callback = tracer.build_callbacks(
            user_id="cust_1",
            trace_id="turn_1",
            session_id="thread_1",
        )[0]
        model_run_id = uuid4()
        callback.on_chat_model_start(
            {"name": "ChatOpenRouter"},
            [[{"role": "user", "content": prompt}]],
            run_id=model_run_id,
        )
        callback.on_llm_end(
            {"content": prompt, "password": password_secret},
            run_id=model_run_id,
        )
        tool_run_id = uuid4()
        callback.on_tool_start(
            {"name": "integration_invoke"},
            f"token={prompt_secret}",
            run_id=tool_run_id,
        )
        callback.on_tool_end(
            {"authorization": f"Bearer {bearer_secret}"},
            run_id=tool_run_id,
        )

    serialized = str(
        [
            {"created": observation.kwargs, "updates": observation.updates}
            for observation in client.observations
        ]
    )
    assert prompt_secret not in serialized
    assert bearer_secret not in serialized
    assert password_secret not in serialized
    assert [observation.kwargs["name"] for observation in client.observations] == [
        "opentulpa.turn.interactive",
        "llm.invoke",
        "tool.invoke",
    ]
    assert client.observations[1].kwargs["metadata"]["content_capture"] == "disabled"
    assert client.observations[2].kwargs["metadata"]["content_capture"] == "disabled"


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


def test_langfuse_trace_context_marks_root_error_without_persisting_message() -> None:
    client = _FakeLangfuseClient()
    tracer = LangfuseTracer(
        public_key="pk",
        secret_key="sk",
        base_url="https://cloud.langfuse.com",
        client=client,
    )
    private_error = "provider token=private-secret from /srv/private/.env"

    with (
        pytest.raises(RuntimeError),
        tracer.trace_context(
            name="opentulpa.turn.interactive",
            trace_id="turn_1",
            user_id="cust_1",
            session_id="thread_1",
        ),
    ):
        raise RuntimeError(private_error)

    observation = client.observations[0]
    error_update = next(update for update in observation.updates if update.get("level") == "ERROR")
    assert error_update["status_message"] == "RuntimeError"
    assert private_error not in str(observation.kwargs)
    assert private_error not in str(observation.updates)


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


def test_redaction_covers_secrets_embedded_in_generic_strings() -> None:
    raw = (
        "curl -H 'Authorization: Bearer header-secret' "
        "'https://example.com/?access_token=url-secret' "
        "OPENAI_API_KEY=environment-secret password='quoted secret'"
    )

    redacted = redact_for_langfuse(raw)

    assert isinstance(redacted, str)
    assert "header-secret" not in redacted
    assert "url-secret" not in redacted
    assert "environment-secret" not in redacted
    assert "quoted secret" not in redacted
    assert redacted.count("[redacted]") >= 4
