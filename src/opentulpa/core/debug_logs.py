"""Helpers for operator-only debug log access."""

from __future__ import annotations

from pathlib import Path

from opentulpa.tasks.sandbox import PROJECT_ROOT


def get_debug_log_path() -> Path:
    return PROJECT_ROOT / ".opentulpa" / "logs" / "app.log"


def read_debug_log_bytes() -> bytes | None:
    path = get_debug_log_path()
    if not path.exists() or not path.is_file():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None
