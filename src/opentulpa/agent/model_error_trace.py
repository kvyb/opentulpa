"""Compact provider error details for model traces."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _compact_provider_value(value: Any) -> str:
    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        elif is_dataclass(value):
            value = asdict(value)
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    return str(text or "").strip()[:2000]


def _safe_getattr(value: Any, attr_name: str) -> Any:
    try:
        return getattr(value, attr_name, None)
    except Exception as exc:
        return f"<unavailable {type(exc).__name__}: {exc}>"


def exception_trace_fields(exc: Exception) -> dict[str, str]:
    fields: dict[str, str] = {}
    for attr_name, field_name in (
        ("status_code", "provider_error_status_code"),
        ("response", "provider_error_response"),
        ("data", "provider_error_response_data"),
        ("response_data", "provider_error_response_data"),
        ("body", "provider_error_body"),
        ("message", "provider_error_message"),
        ("code", "provider_error_code"),
    ):
        value = _safe_getattr(exc, attr_name)
        if value is None:
            continue
        text = _compact_provider_value(value)
        if text:
            fields[field_name] = text[:2000]
    response = _safe_getattr(exc, "raw_response") or _safe_getattr(exc, "response")
    if response is not None:
        for attr_name, field_name in (
            ("status_code", "provider_http_status_code"),
            ("text", "provider_http_text"),
        ):
            value = _safe_getattr(response, attr_name)
            if callable(value):
                continue
            text = _compact_provider_value(value)
            if text:
                fields[field_name] = text[:2000]
    return fields


def exception_trace_text(exc: Exception) -> str:
    base = f"{type(exc).__name__}: {exc}"
    fields = exception_trace_fields(exc)
    if not fields:
        return base
    details = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    return f"{base} [{details}]"


def log_invoke_error(
    runtime: Any,
    *,
    exc: Exception,
    model_name: str,
    attempt_context: dict[str, Any],
    phase: str,
) -> tuple[str, dict[str, str]]:
    error_text = exception_trace_text(exc)
    error_fields = exception_trace_fields(exc)
    runtime.log_behavior_event(
        event="llm.invoke.error",
        model_name=model_name,
        call_site=str(attempt_context.get("call_site") or "runtime_model_invoke"),
        trace_id=str(attempt_context.get("trace_id") or ""),
        thread_id=str(attempt_context.get("thread_id") or ""),
        customer_id=str(attempt_context.get("customer_id") or ""),
        provider_attempt_name=str(attempt_context.get("provider_attempt_name") or "default"),
        phase=phase,
        error=error_text,
        **error_fields,
    )
    return error_text, error_fields


def skip_native_structured_output(model_name: str | None) -> bool:
    slug = str(model_name or "").strip().lower()
    return "deepseek" in slug and "v4" in slug
