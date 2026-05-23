"""Compact provider error details for model traces."""

from __future__ import annotations


def exception_trace_fields(exc: Exception) -> dict[str, str]:
    fields: dict[str, str] = {}
    for attr_name, field_name in (
        ("status_code", "provider_error_status_code"),
        ("response", "provider_error_response"),
        ("response_data", "provider_error_response_data"),
        ("body", "provider_error_body"),
        ("message", "provider_error_message"),
        ("code", "provider_error_code"),
    ):
        value = getattr(exc, attr_name, None)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            fields[field_name] = text[:2000]
    response = getattr(exc, "response", None)
    if response is not None:
        for attr_name, field_name in (
            ("status_code", "provider_http_status_code"),
            ("text", "provider_http_text"),
        ):
            value = getattr(response, attr_name, None)
            if callable(value):
                continue
            text = str(value or "").strip()
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


def skip_native_structured_output(model_name: str | None) -> bool:
    slug = str(model_name or "").strip().lower()
    return "deepseek" in slug and "v4" in slug
