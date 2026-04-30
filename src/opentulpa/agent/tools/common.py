"""Shared helpers for domain-specific tool modules."""

from __future__ import annotations

import shlex
from typing import Any


def require_customer_id(runtime: Any) -> str:
    getter = getattr(runtime, "get_active_customer_id", None)
    customer_id = ""
    if callable(getter):
        customer_id = str(getter() or "").strip()
    if not customer_id:
        customer_id = str(getattr(runtime, "_active_customer_id", "") or "").strip()
    if not customer_id:
        raise RuntimeError("customer_id is missing in runtime context")
    return customer_id


def require_thread_id(runtime: Any) -> str:
    getter = getattr(runtime, "get_active_thread_id", None)
    thread_id = ""
    if callable(getter):
        thread_id = str(getter() or "").strip()
    if not thread_id:
        thread_id = str(getattr(runtime, "_active_thread_id", "") or "").strip()
    if not thread_id:
        raise RuntimeError("thread_id is missing in runtime context")
    return thread_id


def normalize_cleanup_paths(paths: list[str] | None) -> list[str]:
    if not isinstance(paths, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


_WORKING_DIR_PREFIXES: dict[str, str] = {
    "tulpa_stuff": "tulpa_stuff",
    "integrations": "src/opentulpa/integrations",
    "interfaces": "src/opentulpa/interfaces",
    "tools": "src/opentulpa/tools",
    "skills": "src/opentulpa/skills",
    "opentulpa": "src/opentulpa",
}

_SCHEDULED_ORIGINS = {"scheduled", "schedule", "routine", "wake", "background"}
_SCHEDULED_THREAD_PREFIXES = ("wake_", "wake-", "routine_", "routine-")


def normalize_command_for_working_dir(command: str, working_dir: str) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    prefix = _WORKING_DIR_PREFIXES.get(str(working_dir or "").strip())
    if not prefix:
        return text
    try:
        parts = shlex.split(text)
    except Exception:
        return text
    if len(parts) <= 1:
        return text

    markers = (f"{prefix}/", f"./{prefix}/")

    def _strip_one(token: str) -> str:
        raw = str(token)
        for marker in markers:
            if raw.startswith(marker):
                return raw[len(marker) :]
        if raw.startswith("--") and "=" in raw:
            key, value = raw.split("=", 1)
            for marker in markers:
                if value.startswith(marker):
                    return f"{key}={value[len(marker):]}"
        return raw

    normalized = [parts[0], *(_strip_one(item) for item in parts[1:])]
    return shlex.join(normalized)


def normalize_execution_origin(
    *,
    thread_id: str | None,
    execution_origin: str | None,
) -> str:
    raw_origin = str(execution_origin or "").strip().lower()
    if raw_origin in _SCHEDULED_ORIGINS:
        return "scheduled"
    if raw_origin in {"interactive", "manual", "chat"}:
        return "interactive"
    safe_thread = str(thread_id or "").strip().lower()
    if any(safe_thread.startswith(prefix) for prefix in _SCHEDULED_THREAD_PREFIXES):
        return "scheduled"
    return "interactive"
