"""Generic fallback user-visible replies from successful tool outcomes."""

from __future__ import annotations

import json
from typing import Any


def fallback_final_text_from_tool_outcomes(outcomes: Any) -> str:
    if not isinstance(outcomes, list):
        return ""
    for outcome in reversed(outcomes):
        if not isinstance(outcome, dict) or outcome.get("status") != "ok":
            continue
        payload = _tool_outcome_payload(outcome)
        direct_hint = _final_response_hint(payload)
        if direct_hint:
            return direct_hint
    return ""


def _tool_outcome_payload(outcome: dict[str, Any]) -> dict[str, Any]:
    raw = str(outcome.get("result_text", "") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _final_response_hint(payload: dict[str, Any]) -> str:
    for key in ("final_response_hint", "user_visible_reply", "confirmation_text"):
        hint = str(payload.get(key, "") or "").strip()
        if hint:
            return hint
    result = payload.get("result")
    if isinstance(result, dict):
        return _final_response_hint(result)
    return ""
