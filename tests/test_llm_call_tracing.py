from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _TraceResponse:
    def __init__(self) -> None:
        self.content = "All good."
        self.tool_calls = [{"id": "call_1", "name": "memory_search", "args": {"query": "pricing"}}]
        self.response_metadata = {
            "id": "gen_trace_test",
            "model_provider": "openrouter",
            "model_name": "google/gemini-3-flash-preview",
            "system_fingerprint": "fp_trace_test",
        }
        self.usage = {
            "prompt_tokens": 1234,
            "completion_tokens": 56,
            "total_tokens": 1290,
            "cost": 0.023471989,
            "cost_details": {
                "prompt": 0.017,
                "completion": 0.006471989,
            },
        }


class _TraceModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: object, **kwargs: object) -> _TraceResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _TraceResponse()


class _OpenRouterCostResponse(_TraceResponse):
    def __init__(self) -> None:
        super().__init__()
        self.usage = {
            "prompt_tokens": 100,
            "completion_tokens": 25,
            "total_tokens": 125,
            "cost_details": {
                "upstream_inference_prompt_cost": 0.004,
                "upstream_inference_completions_cost": 0.006,
                "upstream_inference_cost": 0.01,
            },
        }


class _OpenRouterCostModel(_TraceModel):
    async def ainvoke(self, messages: object, **kwargs: object) -> _OpenRouterCostResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _OpenRouterCostResponse()


class _ProviderBodyError(Exception):
    def __init__(self) -> None:
        super().__init__("Provider returned error")
        self.status_code = 400
        self.body = '{"error":{"message":"Thinking mode does not support this tool_choice"}}'
        self.data = {
            "error": {
                "message": "Thinking mode does not support this tool_choice",
                "code": 400,
            }
        }


class _FailingTraceModel:
    async def ainvoke(self, messages: object, **kwargs: object) -> _TraceResponse:
        del messages, kwargs
        raise _ProviderBodyError()


class _FailingStreamTraceModel:
    async def astream(self, messages: object, **kwargs: object) -> Any:
        del messages, kwargs
        raise _ProviderBodyError()
        yield _TraceResponse()


@lc_tool
def _trace_lookup_tool(query: str) -> str:
    """Look up trace fixture data."""

    return f"result for {query}"


@pytest.mark.asyncio
async def test_ainvoke_model_writes_full_llm_call_trace(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        prompt_caching_enabled=True,
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"
    runtime._tools = {"trace_lookup": _trace_lookup_tool}
    model = _TraceModel()

    await runtime.ainvoke_model(
        model,
        [
            SystemMessage(content="Stable system prompt"),
            HumanMessage(content="What do you remember about pricing?"),
        ],
        model_name="google/gemini-3-flash-preview",
        stable_prefix_count=1,
        call_context={
            "call_site": "graph_agent",
            "trace_id": "turn_trace_test",
            "thread_id": "chat_test",
            "customer_id": "telegram_test",
            "turn_mode": "interactive",
            "prompt_mode": "literal_chat",
            "prompt_sections": ["stable_core_policy"],
            "prompt_overhead_tokens": 1900,
            "history_message_count": 2,
            "raw_chat_history_count": 1,
            "raw_tool_history_count": 0,
            "optional_context_messages": 1,
        },
    )

    records = [
        json.loads(line)
        for line in runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    record = records[0]
    assert record["trace_id"] == "turn_trace_test"
    assert record["call_site"] == "graph_agent"
    assert record["model_name"] == "google/gemini-3-flash-preview"
    assert record["stable_prefix_count"] == 1
    assert record["prompt_sections"] == ["stable_core_policy"]
    assert record["native_tokens_prompt"] == 1234
    assert record["native_tokens_completion"] == 56
    assert record["native_cost_usd"] == 0.023471989
    assert record["native_cost_prompt_usd"] == 0.017
    assert record["native_cost_completion_usd"] == 0.006471989
    assert record["openrouter_generation_id"] == "gen_trace_test"
    assert record["response_model_provider"] == "openrouter"
    assert record["response_model_name"] == "google/gemini-3-flash-preview"
    assert record["response_system_fingerprint"] == "fp_trace_test"
    assert record["bound_tool_count"] == 1
    assert record["bound_tool_names"] == ["_trace_lookup_tool"]
    assert len(record["bound_tool_schema_hash"]) == 64
    assert record["bound_tool_schema_chars"] > 0
    assert len(record["prompt_hash"]) == 64
    assert len(record["stable_prefix_hash"]) == 64
    assert record["stable_prefix_chars"] == len("Stable system prompt")
    assert len(record["sticky_first_system_hash"]) == 64
    assert record["sticky_first_system_chars"] == len("Stable system prompt")
    assert len(record["sticky_first_non_system_hash"]) == 64
    assert record["sticky_first_non_system_chars"] == len("What do you remember about pricing?")
    assert record["response_text"] == "All good."
    assert record["response_tool_calls"][0]["name"] == "memory_search"
    assert len(record["prompt_messages"]) == 2
    assert record["prompt_messages"][0]["role"] == "system"
    assert record["prompt_messages"][1]["role"] == "user"


@pytest.mark.asyncio
async def test_ainvoke_model_extracts_openrouter_upstream_cost_details(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"

    await runtime.ainvoke_model(
        _OpenRouterCostModel(),
        [HumanMessage(content="cost please")],
        model_name="google/gemini-3-flash-preview",
        call_context={"call_site": "graph_agent", "trace_id": "turn_trace_test"},
    )

    record = json.loads(runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["native_cost_usd"] == 0.01
    assert record["native_cost_prompt_usd"] == 0.004
    assert record["native_cost_completion_usd"] == 0.006


@pytest.mark.asyncio
async def test_ainvoke_model_traces_provider_error_body(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        behavior_log_path=str(tmp_path / "behavior.jsonl"),
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"

    with pytest.raises(_ProviderBodyError):
        await runtime.ainvoke_model(
            _FailingTraceModel(),
            [HumanMessage(content="break")],
            model_name="deepseek/deepseek-v4-pro",
            call_context={"call_site": "graph_agent", "trace_id": "turn_provider_error"},
        )

    record = json.loads(runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["trace_id"] == "turn_provider_error"
    assert "Provider returned error" in record["error"]
    assert "Thinking mode does not support this tool_choice" in record["provider_error_body"]
    assert "Thinking mode does not support this tool_choice" in record["provider_error_response_data"]

    behavior = [
        json.loads(line)
        for line in (tmp_path / "behavior.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    error_event = next(item for item in behavior if item["event"] == "llm.invoke.error")
    assert error_event["phase"] == "ainvoke"
    assert "Thinking mode does not support this tool_choice" in error_event["provider_error_body"]


@pytest.mark.asyncio
async def test_astream_model_traces_provider_error_body(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="deepseek/deepseek-v4-pro",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"

    with pytest.raises(_ProviderBodyError):
        await runtime.astream_model(
            _FailingStreamTraceModel(),
            [HumanMessage(content="break stream")],
            model_name="deepseek/deepseek-v4-pro",
            call_context={"call_site": "graph_agent", "trace_id": "turn_stream_provider_error"},
        )

    record = json.loads(runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["trace_id"] == "turn_stream_provider_error"
    assert "Provider returned error" in record["error"]
    assert "Thinking mode does not support this tool_choice" in record["provider_error_body"]


@pytest.mark.asyncio
async def test_ainvoke_model_skips_llm_call_trace_when_behavior_log_disabled(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        behavior_log_enabled=False,
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"

    await runtime.ainvoke_model(
        _TraceModel(),
        [HumanMessage(content="do not persist this")],
        model_name="google/gemini-3-flash-preview",
    )

    assert not runtime._llm_call_trace_path.exists()


@pytest.mark.asyncio
async def test_ainvoke_model_redacts_inline_media_from_llm_call_trace(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"
    image_data_url = "data:image/jpeg;base64,/9j/QUJDREVGRw=="
    audio_b64 = "QUJDREVGRw=="

    await runtime.ainvoke_model(
        _TraceModel(),
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": "Analyze this upload."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "mp3"}},
                ]
            )
        ],
        model_name="google/gemini-3-flash-preview",
        call_context={"call_site": "file_analysis"},
    )

    record = json.loads(runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()[-1])
    prompt_message = record["prompt_messages"][0]
    prompt_content = prompt_message["content"]
    serialized_record = json.dumps(record, ensure_ascii=False)

    assert prompt_content[1]["image_url"]["url"] == "data:image/jpeg;base64,[redacted]"
    assert prompt_content[2]["input_audio"]["data"] == "[redacted]"
    assert image_data_url not in serialized_record
    assert audio_b64 not in serialized_record
    assert image_data_url not in prompt_message["text"]
    assert audio_b64 not in prompt_message["text"]


def test_llm_call_trace_keeps_latest_100_records(tmp_path: Path) -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"
    runtime._llm_call_trace_lock = None
    runtime._llm_call_trace_limit = 100

    for idx in range(105):
        runtime._write_llm_call_trace(  # type: ignore[attr-defined]
            {
                "ts": f"2026-04-10T00:00:{idx:02d}Z",
                "trace_id": f"turn_{idx}",
                "call_site": "graph_agent",
                "prompt_messages": [],
                "response_text": "",
            }
        )

    records = [
        json.loads(line)
        for line in runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 100
    assert records[0]["trace_id"] == "turn_5"
    assert records[-1]["trace_id"] == "turn_104"
