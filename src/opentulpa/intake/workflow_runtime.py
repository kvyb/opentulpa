"""Shared runtime helpers for intake workflow execution."""

from __future__ import annotations

import re
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any

DEFAULT_EDIT_WINDOW = timedelta(hours=2)
INSTAGRAM_STALE_DECISION_REFRESH_ATTEMPTS = 2
MAX_DECISION_RECOVERY_ATTEMPTS = 2
STALE_TERMINAL_STATUSES = {"stale_requeued", "stale_waiting_for_next_poll"}


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def parse_datetime(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text, text.replace("Z", "+00:00")):
        with suppress(ValueError):
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
    return None


def is_older_than(value: Any, *, max_age: timedelta) -> bool:
    parsed = parse_datetime(value)
    if parsed is None:
        return False
    return (utc_now() - parsed) > max_age


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def unique_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


def required_field_is_present(payload: dict[str, Any], field: str) -> bool:
    value = payload.get(field, "")
    if str(value or "").strip():
        return True
    normalized = re.sub(r"[\s_-]+", "", str(field or "").strip().casefold())
    if normalized in {
        "note",
        "notes",
        "comment",
        "comments",
        "примечание",
        "примечания",
        "комментарий",
        "комментарии",
    }:
        return field in payload
    return False


def truthy_config_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    return text in {"1", "true", "yes", "y", "on", "required", "strict"}


def workflow_requires_intent_match(workflow: dict[str, Any]) -> bool:
    source_config = safe_dict(workflow.get("source_config"))
    matching = safe_dict(source_config.get("matching"))
    return any(
        truthy_config_flag(value)
        for value in (
            source_config.get("intent_match_required"),
            source_config.get("strict_intent_matching"),
            source_config.get("filter_by_intent"),
            matching.get("intent_match_required"),
        )
    )


def normalize_source_config(source_config: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(safe_dict(source_config))
    for key in ("intent_match_required", "strict_intent_matching", "filter_by_intent"):
        if key in normalized and not truthy_config_flag(normalized.get(key)):
            normalized.pop(key, None)
    matching = dict(safe_dict(normalized.get("matching")))
    if "intent_match_required" in matching and not truthy_config_flag(
        matching.get("intent_match_required")
    ):
        matching.pop("intent_match_required", None)
    if matching:
        normalized["matching"] = matching
    else:
        normalized.pop("matching", None)
    return normalized
