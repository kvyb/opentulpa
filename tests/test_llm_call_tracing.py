from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _TraceResponse:
    def __init__(self) -> None:
        self.content = "All good."
        self.tool_calls = [{"id": "call_1", "name": "memory_search", "args": {"query": "pricing"}}]
        self.usage = {
            "prompt_tokens": 1234,
            "completion_tokens": 56,
            "total_tokens": 1290,
        }


class _TraceModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, messages: object, **kwargs: object) -> _TraceResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _TraceResponse()


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
            "prompt_sections": ["stable_core_policy", "style_card"],
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
    assert record["prompt_sections"] == ["stable_core_policy", "style_card"]
    assert record["native_tokens_prompt"] == 1234
    assert record["native_tokens_completion"] == 56
    assert record["response_text"] == "All good."
    assert record["response_tool_calls"][0]["name"] == "memory_search"
    assert len(record["prompt_messages"]) == 2
    assert record["prompt_messages"][0]["role"] == "system"
    assert record["prompt_messages"][1]["role"] == "user"


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


def test_llm_call_trace_inspector_shows_decomposition(tmp_path: Path) -> None:
    trace_path = tmp_path / "llm_call_traces.jsonl"
    trace_path.write_text(
        json.dumps(
            {
                "ts": "2026-04-10T00:00:00Z",
                "trace_id": "turn_show",
                "call_site": "graph_agent",
                "model_name": "z-ai/glm-5.1",
                "thread_id": "chat_test",
                "customer_id": "telegram_test",
                "turn_mode": "interactive",
                "prompt_mode": "literal_chat",
                "stable_prefix_count": 1,
                "prompt_sections": ["stable_core_policy", "style_card"],
                "prompt_overhead_tokens": 1900,
                "history_message_count": 2,
                "raw_chat_history_count": 1,
                "raw_tool_history_count": 0,
                "optional_context_messages": 1,
                "native_tokens_prompt": 1234,
                "native_tokens_completion": 56,
                "prompt_messages": [
                    {"role": "system", "type": "SystemMessage", "approx_tokens": 10, "text": "Stable system prompt"},
                    {"role": "user", "type": "HumanMessage", "approx_tokens": 7, "text": "What do you remember?"},
                ],
                "response_text": "All good.",
                "response_tool_calls": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/inspect_llm_call_traces.py",
            "--path",
            str(trace_path),
            "--trace-id",
            "turn_show",
        ],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "trace_id: turn_show" in result.stdout
    assert "Prompt messages:" in result.stdout
    assert "00. role=system" in result.stdout
    assert "Response text:" in result.stdout
