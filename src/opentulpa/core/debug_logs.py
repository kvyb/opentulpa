"""Helpers for operator-only debug log access."""

from __future__ import annotations

from pathlib import Path

from opentulpa.tasks.sandbox import PROJECT_ROOT


def _debug_log_candidates() -> tuple[Path, ...]:
    return (
        (PROJECT_ROOT / ".cursor" / "debug.log").resolve(),
        (PROJECT_ROOT / ".opentulpa" / "logs" / "app.log").resolve(),
    )


def iter_available_debug_log_paths() -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in _debug_log_candidates():
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file():
            discovered.append(path)
    logs_dir = (PROJECT_ROOT / ".opentulpa" / "logs").resolve()
    if logs_dir.exists() and logs_dir.is_dir():
        for path in sorted(logs_dir.glob("*.log")) + sorted(logs_dir.glob("*.jsonl")):
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def get_debug_log_path() -> Path:
    available = iter_available_debug_log_paths()
    if available:
        return available[0]
    return _debug_log_candidates()[0]


def read_debug_log_bytes() -> bytes | None:
    for path in iter_available_debug_log_paths():
        try:
            return path.read_bytes()
        except Exception:
            continue
    return None
