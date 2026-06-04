from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from opentulpa.telemetry.usage import (
    LocalTraceUsageTelemetryRepository,
    UsageTelemetryQuery,
    UsageTelemetryService,
)


def test_usage_telemetry_projects_and_aggregates_trace_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "llm_call_traces.jsonl"
    _write_jsonl(
        trace_path,
        {
            "ts": "2026-06-04T00:00:00+00:00",
            "customer_id": "telegram_1",
            "thread_id": "thread_a",
            "trace_id": "trace_old",
            "call_site": "graph_agent",
            "turn_mode": "interactive",
            "prompt_mode": "literal_chat",
            "model_name": "google/gemini-3-flash-preview",
            "response_model_provider": "openrouter",
            "response_model_name": "google/gemini-3-flash-preview",
            "openrouter_generation_id": "gen_old",
            "native_tokens_prompt": 100,
            "native_tokens_completion": 50,
            "native_tokens_reasoning": 10,
            "native_tokens_total": 150,
            "native_tokens_cached": 20,
            "native_tokens_cache_write": 5,
            "native_cost_usd": 0.03,
            "native_cost_prompt_usd": 0.01,
            "native_cost_completion_usd": 0.02,
        },
        {
            "ts": "2026-06-04T01:00:00+00:00",
            "customer_id": "telegram_1",
            "thread_id": "thread_a",
            "workflow_id": "iwf_1",
            "trace_id": "trace_new",
            "call_site": "intake_workflow_decision",
            "model_name": "z-ai/glm-5.1",
            "usage": {
                "prompt_tokens": 30,
                "completion_tokens": 7,
                "total_tokens": 37,
                "prompt_tokens_details": {"cached_tokens": 3},
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
        },
        {
            "ts": "2026-06-04T02:00:00+00:00",
            "customer_id": "telegram_2",
            "thread_id": "thread_b",
            "native_tokens_prompt": 999,
            "native_cost_usd": 99,
        },
    )
    service = UsageTelemetryService(
        LocalTraceUsageTelemetryRepository(trace_path=trace_path, max_source_lines=20)
    )

    response = service.query(UsageTelemetryQuery(customer_id="telegram_1", limit=10))

    assert response.source.available is True
    assert response.has_more is False
    assert response.summary.generation_count == 2
    assert response.summary.returned_generation_count == 2
    assert response.summary.input_tokens == 130
    assert response.summary.prompt_tokens == 130
    assert response.summary.output_tokens == 57
    assert response.summary.completion_tokens == 57
    assert response.summary.reasoning_tokens == 12
    assert response.summary.total_tokens == 187
    assert response.summary.cache_read_tokens == 23
    assert response.summary.cache_write_tokens == 5
    assert response.summary.cost_usd == 0.03
    assert response.summary.cost_prompt_usd == 0.01
    assert response.summary.cost_completion_usd == 0.02
    assert [generation.trace_id for generation in response.generations] == [
        "trace_new",
        "trace_old",
    ]
    latest = response.generations[0]
    assert latest.workflow_id == "iwf_1"
    assert latest.model == "z-ai/glm-5.1"
    assert latest.cost_usd is None


def test_usage_telemetry_filters_window_and_reports_limit(tmp_path: Path) -> None:
    trace_path = tmp_path / "llm_call_traces.jsonl"
    _write_jsonl(
        trace_path,
        {
            "ts": "2026-06-04T00:00:00+00:00",
            "customer_id": "telegram_1",
            "thread_id": "thread_a",
            "trace_id": "trace_old",
            "native_tokens_prompt": 10,
        },
        {
            "ts": "2026-06-04T01:00:00+00:00",
            "customer_id": "telegram_1",
            "thread_id": "thread_a",
            "trace_id": "trace_mid",
            "native_tokens_prompt": 20,
        },
        {
            "ts": "2026-06-04T02:00:00+00:00",
            "customer_id": "telegram_1",
            "thread_id": "thread_a",
            "trace_id": "trace_new",
            "native_tokens_prompt": 30,
        },
    )
    service = UsageTelemetryService(
        LocalTraceUsageTelemetryRepository(trace_path=trace_path, max_source_lines=20)
    )

    response = service.query(
        UsageTelemetryQuery(
            customer_id="telegram_1",
            thread_id="thread_a",
            since=datetime(2026, 6, 4, 1, 0),
            until=datetime(2026, 6, 4, 2, 0),
            limit=1,
        )
    )

    assert response.has_more is True
    assert response.summary.generation_count == 2
    assert response.summary.returned_generation_count == 1
    assert response.summary.input_tokens == 50
    assert [generation.trace_id for generation in response.generations] == ["trace_new"]


def test_usage_telemetry_missing_trace_returns_empty_null_totals(tmp_path: Path) -> None:
    service = UsageTelemetryService(
        LocalTraceUsageTelemetryRepository(trace_path=tmp_path / "missing.jsonl")
    )

    response = service.query(UsageTelemetryQuery(customer_id="telegram_1", limit=10))

    assert response.source.available is False
    assert response.summary.generation_count == 0
    assert response.summary.input_tokens is None
    assert response.summary.cost_usd is None
    assert response.generations == []


def _write_jsonl(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
