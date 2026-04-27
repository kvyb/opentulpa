"""Helpers for operator-only debug log access."""

from __future__ import annotations

import io
import sys
import threading
import zipfile
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opentulpa.tasks.sandbox import PROJECT_ROOT

DEFAULT_DEBUG_LOG_LOOKBACK_DAYS = 7
ProcessOutputEventCallback = Callable[[dict[str, Any]], None]
_PROCESS_OUTPUT_EVENT_STATE = threading.local()


class _ProcessOutputTee(io.TextIOBase):
    def __init__(
        self,
        wrapped: Any,
        *,
        stream_name: str,
        project_root: Path,
        event_callback: ProcessOutputEventCallback | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._stream_name = stream_name
        self._project_root = project_root.resolve()
        self._lock = threading.Lock()
        self._event_callback = event_callback

    @property
    def encoding(self) -> str:
        return str(getattr(self._wrapped, "encoding", None) or "utf-8")

    @property
    def errors(self) -> str:
        return str(getattr(self._wrapped, "errors", None) or "replace")

    def fileno(self) -> int:
        return self._wrapped.fileno()

    def isatty(self) -> bool:
        return bool(getattr(self._wrapped, "isatty", lambda: False)())

    def writable(self) -> bool:
        return True

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def flush(self) -> None:
        with suppress(Exception):
            self._wrapped.flush()

    def write(self, text: str) -> int:
        raw = str(text)
        try:
            written = self._wrapped.write(raw)
        except Exception:
            written = len(raw)
        now = datetime.now(UTC)
        self._append_to_server_log(raw, now=now)
        self._emit_output_events(raw, now=now)
        return int(written) if isinstance(written, int) else len(raw)

    def set_event_callback(self, callback: ProcessOutputEventCallback | None) -> None:
        self._event_callback = callback

    def _append_to_server_log(self, text: str, *, now: datetime) -> None:
        if not text:
            return
        log_path = _server_log_path(project_root=self._project_root, now=now)
        prefix = f"{now.isoformat()} [{self._stream_name}] "
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock, log_path.open("a", encoding="utf-8", errors="replace") as handle:
                for chunk in text.splitlines(keepends=True):
                    if chunk:
                        handle.write(prefix)
                        handle.write(chunk)
                        if not chunk.endswith("\n"):
                            handle.write("\n")
        except Exception:
            return

    def _emit_output_events(self, text: str, *, now: datetime) -> None:
        callback = self._event_callback
        if callback is None or not text or bool(getattr(_PROCESS_OUTPUT_EVENT_STATE, "active", False)):
            return
        for chunk in text.splitlines(keepends=True):
            message = chunk.rstrip("\r\n")
            if not message:
                continue
            event = {
                "ts": now.isoformat(),
                "stream": self._stream_name,
                "message": message,
                "project_root": str(self._project_root),
            }
            try:
                _PROCESS_OUTPUT_EVENT_STATE.active = True
                callback(event)
            except Exception:
                return
            finally:
                _PROCESS_OUTPUT_EVENT_STATE.active = False


def _server_log_path(*, project_root: Path, now: datetime | None = None) -> Path:
    stamp = (now or datetime.now(UTC)).astimezone(UTC).date().isoformat()
    return (project_root / ".opentulpa" / "logs" / "server" / f"server-{stamp}.log").resolve()


def install_process_output_log_capture(
    *,
    project_root: Path | None = None,
    event_callback: ProcessOutputEventCallback | None = None,
) -> Path:
    """Tee Python stdout/stderr into a daily server log without hiding console output."""

    root = (project_root or PROJECT_ROOT).resolve()
    if not isinstance(sys.stdout, _ProcessOutputTee):
        sys.stdout = _ProcessOutputTee(
            sys.stdout,
            stream_name="stdout",
            project_root=root,
            event_callback=event_callback,
        )  # type: ignore[assignment]
    else:
        sys.stdout.set_event_callback(event_callback)
    if not isinstance(sys.stderr, _ProcessOutputTee):
        sys.stderr = _ProcessOutputTee(
            sys.stderr,
            stream_name="stderr",
            project_root=root,
            event_callback=event_callback,
        )  # type: ignore[assignment]
    else:
        sys.stderr.set_event_callback(event_callback)
    return _server_log_path(project_root=root)


def configure_process_output_event_callback(
    callback: ProcessOutputEventCallback | None,
) -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, _ProcessOutputTee):
            stream.set_event_callback(callback)


def _debug_log_candidates() -> tuple[Path, ...]:
    return (
        (PROJECT_ROOT / ".cursor" / "debug.log").resolve(),
        (PROJECT_ROOT / ".opentulpa" / "logs" / "app.log").resolve(),
    )


def _within_lookback(path: Path, *, lookback_days: int | None, now: datetime | None = None) -> bool:
    if lookback_days is None:
        return True
    try:
        stat = path.stat()
    except Exception:
        return False
    cutoff = (now or datetime.now(UTC)) - timedelta(days=max(1, int(lookback_days)))
    modified = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
    return modified >= cutoff


def iter_available_debug_log_paths(
    *,
    lookback_days: int | None = None,
    now: datetime | None = None,
) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in _debug_log_candidates():
        if path in seen:
            continue
        seen.add(path)
        if path.exists() and path.is_file() and _within_lookback(path, lookback_days=lookback_days, now=now):
            discovered.append(path)
    logs_dir = (PROJECT_ROOT / ".opentulpa" / "logs").resolve()
    if logs_dir.exists() and logs_dir.is_dir():
        candidates = sorted(logs_dir.rglob("*.log")) + sorted(logs_dir.rglob("*.jsonl"))
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not resolved.is_file():
                continue
            if not _within_lookback(resolved, lookback_days=lookback_days, now=now):
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


def build_debug_logs_archive_bytes(
    *,
    lookback_days: int = DEFAULT_DEBUG_LOG_LOOKBACK_DAYS,
    now: datetime | None = None,
) -> tuple[str, bytes] | None:
    paths = iter_available_debug_log_paths(lookback_days=lookback_days, now=now)
    if not paths:
        return None

    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%d-%H%M%S")
    archive_name = f"opentulpa-debug-logs-last-{lookback_days}-days-{stamp}.zip"
    buffer = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            try:
                raw = path.read_bytes()
            except Exception:
                continue
            try:
                archive_path = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except Exception:
                archive_path = path.name
            archive.writestr(archive_path, raw)
            added += 1
    if added == 0:
        return None
    payload = buffer.getvalue()
    if not payload:
        return None
    return archive_name, payload
