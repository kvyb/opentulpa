"""Bounded current-turn tool outcome memory for model prompts."""

from __future__ import annotations

import json
from typing import Any

MAX_TOOL_OUTCOME_ROUNDS = 10
_MAX_RESULT_CHARS = 12000
_MAX_CONTEXT_CHARS = 24000
_MAX_STRING_CHARS = 5000
_MAX_LIST_ITEMS = 40
_NOISE_KEYS = {
    "headers",
    "cookies",
    "stack",
    "traceback",
    "raw",
    "raw_response",
    "request",
    "request_id",
    "debug",
    "debug_info",
    "observability",
    "metadata_json",
}


def _round_id(item: dict[str, Any]) -> int:
    try:
        return int(item.get("round_id", 0) or 0)
    except Exception:
        return 0


def _trim_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    reserve = max(200, max_chars // 2 - 8)
    return f"{text[:reserve]}\n...\n{text[-reserve:]}".strip()


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return _trim_text(value, max_chars=_MAX_STRING_CHARS)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_clean_value(item) for item in value[:_MAX_LIST_ITEMS]]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key or "").strip()
            if not key_text or key_text.startswith("_") or key_text.lower() in _NOISE_KEYS:
                continue
            cleaned[key_text] = _clean_value(item)
        return cleaned
    return _trim_text(value, max_chars=1000)


def compact_tool_result_for_model(*, tool_name: str, result: Any) -> str:
    payload = _clean_value(result)
    try:
        text = json.dumps(payload, ensure_ascii=False, default=str)
    except TypeError:
        text = str(payload)
    prefix = str(tool_name or "tool").strip() or "tool"
    return f"{prefix} result: {_trim_text(text, max_chars=_MAX_RESULT_CHARS)}"


def add_tool_outcomes(left: Any, right: Any) -> list[dict[str, Any]]:
    combined = [
        item
        for group in (left, right)
        if isinstance(group, list)
        for item in group
        if isinstance(item, dict)
    ]
    if not combined:
        return []
    round_ids: list[int] = []
    for item in combined:
        round_id = _round_id(item)
        if round_id > 0 and round_id not in round_ids:
            round_ids.append(round_id)
    keep_rounds = set(round_ids[-MAX_TOOL_OUTCOME_ROUNDS:])
    if keep_rounds:
        return [item for item in combined if _round_id(item) in keep_rounds]
    return combined[-MAX_TOOL_OUTCOME_ROUNDS:]


def next_tool_round_id(outcomes: Any) -> int:
    if not isinstance(outcomes, list):
        return 1
    max_round = 0
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        max_round = max(max_round, _round_id(item))
    return max_round + 1


def build_tool_outcome_context(outcomes: Any) -> str:
    if not isinstance(outcomes, list) or not outcomes:
        return ""
    rounds: dict[int, list[dict[str, Any]]] = {}
    for item in outcomes:
        if not isinstance(item, dict):
            continue
        round_id = _round_id(item)
        if round_id <= 0:
            continue
        rounds.setdefault(round_id, []).append(item)
    if not rounds:
        return ""
    parts = ["Previous tool results in this current turn. Use these as verified context:"]
    for round_id in sorted(rounds)[-MAX_TOOL_OUTCOME_ROUNDS:]:
        parts.append(f"Tool round {round_id}:")
        for item in rounds[round_id]:
            status = str(item.get("status", "") or "").strip() or "unknown"
            name = str(item.get("tool_name", "") or "").strip() or "tool"
            result = str(item.get("result_text") or item.get("error") or "").strip()
            if result:
                parts.append(f"- {name} [{status}]: {_trim_text(result, max_chars=2200)}")
            else:
                parts.append(f"- {name} [{status}]")
    return _trim_text("\n".join(parts), max_chars=_MAX_CONTEXT_CHARS)
