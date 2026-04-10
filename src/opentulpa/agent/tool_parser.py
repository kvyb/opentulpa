"""Compact reusable tool payload parsing and serialization helpers."""

from __future__ import annotations

import json
from typing import Any


def _truncate_value(value: str, *, char_limit: int | None) -> str:
    text = " ".join(str(value or "").split()).strip()
    if char_limit is None or char_limit <= 0 or len(text) <= char_limit:
        return text
    return text[: max(1, int(char_limit) - 3)].rstrip() + "..."


def _coerce_jsonish_payload(payload: Any) -> Any:
    if not isinstance(payload, str):
        return payload
    raw = str(payload or "").strip()
    if not raw or raw[0] not in "{[":
        return payload
    try:
        return json.loads(raw)
    except Exception:
        return payload


def _flatten_pairs(
    value: Any,
    *,
    prefix: str,
    out: list[tuple[str, str]],
    value_char_limit: int | None,
) -> None:
    if value in (None, "", [], {}):
        return
    if isinstance(value, dict):
        for key in sorted(value.keys(), key=lambda item: str(item)):
            child = value.get(key)
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten_pairs(
                child,
                prefix=child_prefix,
                out=out,
                value_char_limit=value_char_limit,
            )
        return
    if isinstance(value, list):
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            _flatten_pairs(
                child,
                prefix=child_prefix,
                out=out,
                value_char_limit=value_char_limit,
            )
        return
    normalized = _truncate_value(str(value), char_limit=value_char_limit)
    if not normalized:
        return
    out.append((prefix or "value", normalized))


def compact_tool_payload(
    payload: Any,
    *,
    value_char_limit: int | None = 100,
    pair_delimiter: str = " | ",
) -> str:
    """Flatten JSON-ish tool payloads into a compact stable key=value format."""
    parsed = _coerce_jsonish_payload(payload)
    if isinstance(parsed, (dict, list)):
        pairs: list[tuple[str, str]] = []
        _flatten_pairs(
            parsed,
            prefix="",
            out=pairs,
            value_char_limit=value_char_limit,
        )
        if not pairs:
            return ""
        return pair_delimiter.join(f"{key}={value}" for key, value in pairs)
    return _truncate_value(str(parsed), char_limit=value_char_limit)


def compact_tool_call_record(
    *,
    tool_name: str,
    args: Any,
    result: Any,
    args_value_char_limit: int | None = 100,
    result_value_char_limit: int | None = 100,
) -> str:
    """Format a tool call/result pair in a compact reusable single-line representation."""
    safe_name = str(tool_name or "").strip() or "tool"
    parts = [f"tool={safe_name}"]
    compact_args = compact_tool_payload(args, value_char_limit=args_value_char_limit)
    if compact_args:
        parts.append(f"args[{compact_args}]")
    compact_result = compact_tool_payload(result, value_char_limit=result_value_char_limit)
    if compact_result:
        parts.append(f"result[{compact_result}]")
    return " | ".join(parts)
