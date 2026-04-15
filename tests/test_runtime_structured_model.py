from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _Schema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = False
    reason: str = ""


class _StructuredRunner:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.messages: object | None = None

    async def ainvoke(self, _messages: object, **_: Any) -> object:
        self.messages = _messages
        return self._payload


class _StructuredModel:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.runner: _StructuredRunner | None = None

    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        self.runner = _StructuredRunner(self._payload)
        return self.runner


class _FallbackResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FallbackModel:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: object) -> _FallbackResponse:
        return _FallbackResponse(self._content)


class _BrokenStructuredThenFallbackModel(_FallbackModel):
    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        raise RuntimeError("structured_unavailable")


class _ProviderAwareStructuredRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, _messages: object, **kwargs: Any) -> object:
        self.calls.append({"messages": _messages, "kwargs": kwargs})
        return {"ok": True, "reason": "ordered_route"}


class _ProviderAwareStructuredModel:
    def __init__(self) -> None:
        self.runners: list[_ProviderAwareStructuredRunner] = []

    def with_structured_output(self, _schema: type[BaseModel]) -> _ProviderAwareStructuredRunner:
        runner = _ProviderAwareStructuredRunner()
        self.runners.append(runner)
        return runner


@pytest.mark.asyncio
async def test_invoke_structured_model_prefers_native_structured_output() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _StructuredModel(_Schema(ok=True, reason="native"))

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "native"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_uses_strict_json_fallback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('{"ok": true, "reason": "fallback"}')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fallback"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_accepts_fenced_json_in_fallback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('```json\n{"ok": true, "reason": "fenced"}\n```')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fenced"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_rejects_wrapped_non_json_text() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('prefix {"ok": true, "reason": "x"} suffix')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert parsed is None
    assert isinstance(error, str)
    assert "ValidationError" in error


@pytest.mark.asyncio
async def test_invoke_structured_model_routes_glm51_with_fireworks_then_siliconflow_order() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.openrouter_base_url = "https://openrouter.ai/api/v1"
    runtime.model_name = "z-ai/glm-5.1"
    runtime._prompt_caching_enabled = False
    runtime._prompt_cache_ttl_1h = False
    model = _ProviderAwareStructuredModel()

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
        model_name="z-ai/glm-5.1",
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "ordered_route"
    assert error is None
    assert len(model.runners) == 1
    provider = model.runners[0].calls[0]["kwargs"]["extra_body"]["provider"]
    assert provider == {"order": ["fireworks", "siliconflow"], "allow_fallbacks": False}


@pytest.mark.asyncio
async def test_invoke_structured_model_records_single_llm_call_trace_on_success(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"
    model = _StructuredModel({"ok": True, "reason": "native"})

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
        call_context={
            "call_site": "intake_workflow_decision",
            "trace_id": "intake_trace_test",
            "thread_id": "intake_decision_iwf_conv",
            "customer_id": "telegram_123",
            "turn_mode": "routine_wake",
            "prompt_mode": "structured_intake",
        },
    )

    assert isinstance(parsed, _Schema)
    assert error is None
    records = [
        json.loads(line)
        for line in runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["call_site"] == "intake_workflow_decision"
    assert records[0]["trace_id"] == "intake_trace_test"
    assert "native" in records[0]["response_text"]
    assert records[0]["response_content"]["reason"] == "native"


@pytest.mark.asyncio
async def test_decide_intake_workflow_uses_stronger_policy_prompt() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    model = _StructuredModel(
        {
            "matches_workflow": False,
            "confidence": 0.2,
            "conversation_summary": "unrelated chat",
            "extracted_fields": {},
            "missing_fields": [],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": False,
            "booking_action": "ignore",
            "save_payload": {},
            "reason": "not a booking",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": "interior, exterior, or both"},
            "sink_type": "local_csv",
        },
        conversation={
            "summary": {"conversation_id": "conv_1"},
            "recent_messages": [{"sender_role": "customer", "text": "thanks"}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "reply_execution",
                "error": "Invalid request data provided",
                "prior_decision": {"reply_action": "send_reply"},
            }
        ],
    )

    assert decision["ok"] is True
    assert model.runner is not None
    messages = model.runner.messages
    assert isinstance(messages, list)
    system_text = str(messages[0].content)
    assert isinstance(messages[0].content, list)
    assert messages[0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(messages[1].content, str)
    assert "False positives are worse than ignoring an unrelated DM." in system_text
    assert "If the message is ambiguous" in system_text
    assert "If customer messages conflict, prefer the latest customer-provided value" in system_text
    assert "Ask at most one compact question at a time" in system_text
    assert "When ready_to_save=true, save_payload must contain the merged final field set" in system_text


@pytest.mark.asyncio
async def test_decide_intake_workflow_prefers_compact_structured_path_even_when_tool_runtime_available() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    runtime._graph = object()
    runtime._wake_execution_model_with_tools = object()
    model = _StructuredModel(
        {
            "matches_workflow": True,
            "confidence": 0.95,
            "conversation_summary": "Customer wants a car wash tomorrow at 4pm.",
            "extracted_fields": {"day": "tomorrow"},
            "missing_fields": ["time"],
            "reply_action": "send_reply",
            "reply_text": "What time works best?",
            "ready_to_save": False,
            "booking_action": "create_new_booking",
            "save_payload": {},
            "reason": "Need time before save.",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"
    captured: dict[str, Any] = {"called": False}

    async def _fake_ainvoke_text(**kwargs: Any) -> str:
        captured.update(kwargs)
        captured["called"] = True
        return "{}"

    runtime.ainvoke_text = _fake_ainvoke_text

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": "interior, exterior, or both"},
            "sink_type": "google_sheets_composio",
            "sink_config": {"tool_slug": "GOOGLESHEETS_ADD_ROW", "field_mapping": {"day": "Date"}},
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
            },
            "recent_messages": [{"sender_role": "customer", "text": "Need a wash tomorrow."}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "reply_execution",
                "error": "Invalid request data provided",
                "prior_decision": {"reply_action": "send_reply"},
            }
        ],
    )

    assert decision["ok"] is True
    assert captured["called"] is False
    assert model.runner is not None


@pytest.mark.asyncio
async def test_decide_intake_workflow_escalates_to_tool_runtime_after_structured_failure_with_feedback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    runtime._graph = object()
    runtime._wake_execution_model_with_tools = object()
    runtime._model = _BrokenStructuredThenFallbackModel("not json")
    runtime._wake_execution_model = runtime._model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"
    captured: dict[str, Any] = {}

    async def _fake_ainvoke_text(**kwargs: Any) -> str:
        captured.update(kwargs)
        return (
            '{"matches_workflow": true, "confidence": 0.95, "conversation_summary": '
            '"Customer wants a car wash tomorrow at 4pm.", "extracted_fields": {"day": "tomorrow"}, '
            '"missing_fields": ["time"], "reply_action": "send_reply", "reply_text": "What time works best?", '
            '"ready_to_save": false, "booking_action": "create_new_booking", "save_payload": {}, '
            '"reason": "Need time before save."}'
        )

    runtime.ainvoke_text = _fake_ainvoke_text

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": "interior, exterior, or both"},
            "sink_type": "google_sheets_composio",
            "sink_config": {"tool_slug": "GOOGLESHEETS_ADD_ROW", "field_mapping": {"day": "Date"}},
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
            },
            "recent_messages": [{"sender_role": "customer", "text": "Need a wash tomorrow."}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "reply_execution",
                "error": "Invalid request data provided",
                "prior_decision": {"reply_action": "send_reply"},
            }
        ],
    )

    assert decision["ok"] is True
    assert captured["thread_id"] == "wake_intake_iwf_123_conv_1_msg_1"
    assert captured["turn_mode"] == "routine_wake"
    assert captured["include_pending_context"] is False
    assert captured["prompt_mode_override"] == "literal_chat"
    prompt = str(captured["text"])
    assert "Operate like a real OpenTulpa background execution turn and use tools when needed." in prompt
    assert "composio_tool_search" in prompt
    assert "execution_feedback=" in prompt
    assert "Invalid request data provided" in prompt


@pytest.mark.asyncio
async def test_decide_intake_workflow_compacts_prompt_payload() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    long_text = "x" * 1000
    model = _StructuredModel(
        {
            "matches_workflow": False,
            "confidence": 0.2,
            "conversation_summary": "unrelated chat",
            "extracted_fields": {},
            "missing_fields": [],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": False,
            "booking_action": "ignore",
            "save_payload": {},
            "reason": "not a booking",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"

    recent_messages = [
        {"id": f"m{i}", "created_time": f"2026-04-08T08:0{i}:00+00:00", "sender_role": "customer", "text": long_text}
        for i in range(8)
    ]
    await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": long_text,
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": long_text},
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "tool_slug": "GOOGLESHEETS_ADD_ROW",
                "field_mapping": {"day": "Date"},
                "static_arguments": {"spreadsheet_id": long_text},
            },
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
                "latest_inbound_message_text_preview": long_text,
            },
            "recent_messages": recent_messages,
        },
        active_booking={"booking_id": "bkg_1", "status": "active", "extracted_fields": {"notes": long_text}},
        recent_completed_booking=None,
        execution_feedback=[
            {"phase": "sink_execution", "error": long_text, "prior_decision": {"reply_action": "send_reply"}},
            {"phase": "reply_execution", "error": long_text, "prior_decision": {"reply_action": "send_reply"}},
            {"phase": "ignored", "error": long_text, "prior_decision": {"reply_action": "send_reply"}},
        ],
    )

    assert model.runner is not None
    messages = model.runner.messages
    assert isinstance(messages, list)
    human_text = str(messages[1].content)
    assert human_text.count('"sender_role"') == 6
    assert ('"text": "' + ("x" * 301)) not in human_text
    assert human_text.count('"phase"') == 2


def test_prompt_cache_profile_uses_openrouter_standard_modes() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False

    anth = runtime.prompt_cache_profile(model_name="anthropic/claude-sonnet-4.6")
    gemini = runtime.prompt_cache_profile(model_name="google/gemini-3-flash-preview")
    auto = runtime.prompt_cache_profile(model_name="openai/gpt-4.1")

    assert anth["strategy"] == "top_level"
    assert gemini["strategy"] == "breakpoint"
    assert auto["strategy"] == "automatic"
