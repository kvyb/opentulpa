"""Read-only usage telemetry projection from local LLM call traces."""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

LOCAL_TRACE_SOURCE: Literal["local_trace_jsonl"] = "local_trace_jsonl"
DEFAULT_TRACE_MAX_LINES = 1_000
DEFAULT_TRACE_MAX_BYTES = 1_000_000


class UsageTelemetryFields(BaseModel):
    input_tokens: int | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    cost_usd: float | None = None
    cost_prompt_usd: float | None = None
    cost_completion_usd: float | None = None


class UsageTelemetrySummary(UsageTelemetryFields):
    generation_count: int
    returned_generation_count: int
    currency: Literal["USD"] = "USD"


class UsageTelemetryGeneration(UsageTelemetryFields):
    timestamp: datetime | None = None
    customer_id: str
    thread_id: str | None = None
    workflow_id: str | None = None
    trace_id: str | None = None
    call_site: str | None = None
    turn_mode: str | None = None
    prompt_mode: str | None = None
    provider: str | None = None
    model: str | None = None
    generation_id: str | None = None
    source: Literal["local_trace_jsonl"] = LOCAL_TRACE_SOURCE


class UsageTelemetrySource(BaseModel):
    kind: Literal["local_trace_jsonl"] = LOCAL_TRACE_SOURCE
    available: bool
    max_source_lines: int = Field(ge=1)
    max_source_bytes: int = Field(ge=1)
    currency: Literal["USD"] = "USD"


class UsageTelemetryResponse(BaseModel):
    summary: UsageTelemetrySummary
    generations: list[UsageTelemetryGeneration]
    source: UsageTelemetrySource
    has_more: bool


@dataclass(frozen=True)
class UsageTelemetryQuery:
    customer_id: str
    thread_id: str | None = None
    workflow_id: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    limit: int = 100


class LocalTraceUsageTelemetryRepository:
    """Repository for capped local `llm_call_traces.jsonl` reads."""

    def __init__(
        self,
        *,
        trace_path: str | Path | None,
        max_source_lines: int = DEFAULT_TRACE_MAX_LINES,
        max_source_bytes: int = DEFAULT_TRACE_MAX_BYTES,
    ) -> None:
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.max_source_lines = max(1, int(max_source_lines))
        self.max_source_bytes = max(1, int(max_source_bytes))

    @property
    def available(self) -> bool:
        return self.trace_path is not None and self.trace_path.exists() and self.trace_path.is_file()

    def list_records(self) -> list[dict[str, Any]]:
        assert self.max_source_lines > 0
        assert self.max_source_bytes > 0
        if self.trace_path is None or not self.available:
            return []

        records: list[dict[str, Any]] = []
        for line in _tail_lines(
            self.trace_path,
            max_lines=self.max_source_lines,
            max_bytes=self.max_source_bytes,
        ):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records


class UsageTelemetryService:
    def __init__(self, repository: LocalTraceUsageTelemetryRepository) -> None:
        self._repository = repository

    def query(self, query: UsageTelemetryQuery) -> UsageTelemetryResponse:
        assert query.customer_id.strip()
        assert query.limit > 0
        since = _normalize_datetime(query.since)
        until = _normalize_datetime(query.until)
        assert since is None or until is None or since <= until

        matching = [
            _project_record(record)
            for record in self._repository.list_records()
            if _record_matches_query(record, query, since=since, until=until)
        ]
        matching.sort(key=_sort_key, reverse=True)
        returned = matching[: query.limit]
        assert len(returned) <= query.limit
        return UsageTelemetryResponse(
            summary=_summarize(
                matching,
                generation_count=len(matching),
                returned_generation_count=len(returned),
            ),
            generations=returned,
            source=UsageTelemetrySource(
                available=self._repository.available,
                max_source_lines=self._repository.max_source_lines,
                max_source_bytes=self._repository.max_source_bytes,
            ),
            has_more=len(matching) > len(returned),
        )


def _tail_lines(path: Path, *, max_lines: int, max_bytes: int) -> list[str]:
    assert max_lines > 0
    assert max_bytes > 0
    try:
        size = path.stat().st_size
    except OSError:
        return []

    lines: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open("rb") as handle:
            start = max(0, size - max_bytes)
            if start:
                handle.seek(start)
                _ = handle.readline()
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line:
                    lines.append(line)
    except OSError:
        return []
    return list(lines)


def _record_matches_query(
    record: dict[str, Any],
    query: UsageTelemetryQuery,
    *,
    since: datetime | None,
    until: datetime | None,
) -> bool:
    if _text(record.get("customer_id")) != query.customer_id:
        return False
    if query.thread_id and _text(record.get("thread_id")) != query.thread_id:
        return False
    if query.workflow_id and _text(record.get("workflow_id")) != query.workflow_id:
        return False

    timestamp = _datetime_value(record.get("ts"))
    if since is not None and (timestamp is None or timestamp < since):
        return False
    return not (until is not None and (timestamp is None or timestamp > until))


def _project_record(record: dict[str, Any]) -> UsageTelemetryGeneration:
    customer_id = _text(record.get("customer_id"))
    assert customer_id
    tokens = _token_fields(record)
    costs = _cost_fields(record)
    return UsageTelemetryGeneration(
        timestamp=_datetime_value(record.get("ts")),
        customer_id=customer_id,
        thread_id=_none_if_empty(record.get("thread_id")),
        workflow_id=_none_if_empty(record.get("workflow_id")),
        trace_id=_none_if_empty(record.get("trace_id")),
        call_site=_none_if_empty(record.get("call_site")),
        turn_mode=_none_if_empty(record.get("turn_mode")),
        prompt_mode=_none_if_empty(record.get("prompt_mode")),
        provider=_none_if_empty(record.get("response_model_provider")),
        model=_model_name(record),
        generation_id=_none_if_empty(record.get("openrouter_generation_id")),
        input_tokens=tokens["input_tokens"],
        prompt_tokens=tokens["prompt_tokens"],
        output_tokens=tokens["output_tokens"],
        completion_tokens=tokens["completion_tokens"],
        reasoning_tokens=tokens["reasoning_tokens"],
        total_tokens=tokens["total_tokens"],
        cache_read_tokens=tokens["cache_read_tokens"],
        cache_write_tokens=tokens["cache_write_tokens"],
        cost_usd=costs["cost_usd"],
        cost_prompt_usd=costs["cost_prompt_usd"],
        cost_completion_usd=costs["cost_completion_usd"],
    )


def _model_name(record: dict[str, Any]) -> str | None:
    return _none_if_empty(record.get("response_model_name")) or _none_if_empty(
        record.get("model_name")
    )


def _token_fields(record: dict[str, Any]) -> dict[str, int | None]:
    usage = _usage_dict(record)
    prompt_details = _dict_value(usage.get("prompt_tokens_details"))
    completion_details = _dict_value(usage.get("completion_tokens_details"))

    prompt_tokens = _int_field(record, "native_tokens_prompt")
    if prompt_tokens is None:
        prompt_tokens = _int_value(usage.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _int_value(usage.get("input_tokens"))

    completion_tokens = _int_field(record, "native_tokens_completion")
    if completion_tokens is None:
        completion_tokens = _int_value(usage.get("completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _int_value(usage.get("output_tokens"))

    reasoning_tokens = _int_field(record, "native_tokens_reasoning")
    if reasoning_tokens is None:
        reasoning_tokens = _int_value(completion_details.get("reasoning_tokens"))

    total_tokens = _int_field(record, "native_tokens_total")
    if total_tokens is None:
        total_tokens = _int_value(usage.get("total_tokens"))

    cache_read_tokens = _int_field(record, "native_tokens_cached")
    if cache_read_tokens is None:
        cache_read_tokens = _int_value(prompt_details.get("cached_tokens"))
    if cache_read_tokens is None:
        cache_read_tokens = _int_value(usage.get("prompt_cache_hit_tokens"))

    cache_write_tokens = _int_field(record, "native_tokens_cache_write")
    if cache_write_tokens is None:
        cache_write_tokens = _int_value(prompt_details.get("cache_write_tokens"))
    if cache_write_tokens is None:
        cache_write_tokens = _int_value(usage.get("prompt_cache_miss_tokens"))

    return {
        "input_tokens": prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "output_tokens": completion_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": cache_read_tokens,
        "cache_write_tokens": cache_write_tokens,
    }


def _cost_fields(record: dict[str, Any]) -> dict[str, float | None]:
    usage = _usage_dict(record)
    native_cost_details = _dict_value(record.get("native_cost_details"))
    usage_cost_details = _dict_value(usage.get("cost_details"))

    cost_usd = _float_field(record, "native_cost_usd")
    if cost_usd is None:
        cost_usd = _float_value(native_cost_details.get("total"))
    if cost_usd is None:
        cost_usd = _float_value(native_cost_details.get("cost"))
    if cost_usd is None:
        cost_usd = _float_value(usage.get("cost"))
    if cost_usd is None:
        cost_usd = _float_value(usage_cost_details.get("total"))
    if cost_usd is None:
        cost_usd = _float_value(usage_cost_details.get("cost"))

    return {
        "cost_usd": cost_usd,
        "cost_prompt_usd": _cost_part(
            record,
            "native_cost_prompt_usd",
            native_cost_details,
            usage_cost_details,
            "prompt",
            "input",
        ),
        "cost_completion_usd": _cost_part(
            record,
            "native_cost_completion_usd",
            native_cost_details,
            usage_cost_details,
            "completion",
            "completions",
            "output",
        ),
    }


def _cost_part(
    record: dict[str, Any],
    native_field: str,
    native_cost_details: dict[str, Any],
    usage_cost_details: dict[str, Any],
    *detail_keys: str,
) -> float | None:
    value = _float_field(record, native_field)
    if value is not None:
        return value
    for key in detail_keys:
        value = _float_value(native_cost_details.get(key))
        if value is not None:
            return value
    for key in detail_keys:
        value = _float_value(usage_cost_details.get(key))
        if value is not None:
            return value
    return None


def _summarize(
    generations: list[UsageTelemetryGeneration],
    *,
    generation_count: int,
    returned_generation_count: int,
) -> UsageTelemetrySummary:
    assert generation_count >= len(generations)
    assert 0 <= returned_generation_count <= generation_count
    return UsageTelemetrySummary(
        generation_count=generation_count,
        returned_generation_count=returned_generation_count,
        input_tokens=_sum_int(generations, "input_tokens"),
        prompt_tokens=_sum_int(generations, "prompt_tokens"),
        output_tokens=_sum_int(generations, "output_tokens"),
        completion_tokens=_sum_int(generations, "completion_tokens"),
        reasoning_tokens=_sum_int(generations, "reasoning_tokens"),
        total_tokens=_sum_int(generations, "total_tokens"),
        cache_read_tokens=_sum_int(generations, "cache_read_tokens"),
        cache_write_tokens=_sum_int(generations, "cache_write_tokens"),
        cost_usd=_sum_float(generations, "cost_usd"),
        cost_prompt_usd=_sum_float(generations, "cost_prompt_usd"),
        cost_completion_usd=_sum_float(generations, "cost_completion_usd"),
    )


def _usage_dict(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("usage")
    return dict(value) if isinstance(value, dict) else {}


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _sum_int(generations: list[UsageTelemetryGeneration], field_name: str) -> int | None:
    values = [getattr(generation, field_name) for generation in generations]
    ints = [int(value) for value in values if value is not None]
    return sum(ints) if ints else None


def _sum_float(generations: list[UsageTelemetryGeneration], field_name: str) -> float | None:
    values = [getattr(generation, field_name) for generation in generations]
    floats = [float(value) for value in values if value is not None]
    return math.fsum(floats) if floats else None


def _sort_key(generation: UsageTelemetryGeneration) -> tuple[datetime, str, str]:
    return (
        generation.timestamp or datetime.min.replace(tzinfo=UTC),
        generation.trace_id or "",
        generation.call_site or "",
    )


def _int_field(record: dict[str, Any], field_name: str) -> int | None:
    return _int_value(record.get(field_name))


def _float_field(record: dict[str, Any], field_name: str) -> float | None:
    return _float_value(record.get(field_name))


def _int_value(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_value(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _datetime_value(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _normalize_datetime(parsed)


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _none_if_empty(value: Any) -> str | None:
    text = _text(value)
    return text or None


def _text(value: Any) -> str:
    return str(value or "").strip()
