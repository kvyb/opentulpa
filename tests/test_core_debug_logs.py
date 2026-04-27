from __future__ import annotations

import importlib.util
import io
import os
import sys
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _load_debug_logs_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "opentulpa" / "core" / "debug_logs.py"
    spec = importlib.util.spec_from_file_location("opentulpa.core.debug_logs_under_test", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load debug_logs module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_get_debug_log_path_prefers_cursor_file(monkeypatch, tmp_path: Path) -> None:
    debug_logs = _load_debug_logs_module()
    cursor_path = tmp_path / ".cursor" / "debug.log"
    app_path = tmp_path / ".opentulpa" / "logs" / "app.log"
    cursor_path.parent.mkdir(parents=True, exist_ok=True)
    app_path.parent.mkdir(parents=True, exist_ok=True)
    cursor_path.write_text("cursor", encoding="utf-8")
    app_path.write_text("app", encoding="utf-8")

    monkeypatch.setattr(debug_logs, "PROJECT_ROOT", tmp_path)

    assert debug_logs.get_debug_log_path() == cursor_path.resolve()


def test_read_debug_log_bytes_falls_back_to_legacy_app_log(monkeypatch, tmp_path: Path) -> None:
    debug_logs = _load_debug_logs_module()
    app_path = tmp_path / ".opentulpa" / "logs" / "app.log"
    app_path.parent.mkdir(parents=True, exist_ok=True)
    app_path.write_bytes(b"legacy")

    monkeypatch.setattr(debug_logs, "PROJECT_ROOT", tmp_path)

    assert debug_logs.read_debug_log_bytes() == b"legacy"


def test_read_debug_log_bytes_returns_none_when_missing(monkeypatch, tmp_path: Path) -> None:
    debug_logs = _load_debug_logs_module()
    monkeypatch.setattr(debug_logs, "PROJECT_ROOT", tmp_path)

    assert debug_logs.read_debug_log_bytes() is None


def test_iter_available_debug_log_paths_includes_logs_dir_files(
    monkeypatch, tmp_path: Path
) -> None:
    debug_logs = _load_debug_logs_module()
    logs_dir = tmp_path / ".opentulpa" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    behavior_path = logs_dir / "agent_behavior.jsonl"
    behavior_path.write_bytes(b"{}\n")

    monkeypatch.setattr(debug_logs, "PROJECT_ROOT", tmp_path)

    paths = debug_logs.iter_available_debug_log_paths()
    assert behavior_path.resolve() in paths


def test_iter_available_debug_log_paths_filters_to_lookback(
    monkeypatch, tmp_path: Path
) -> None:
    debug_logs = _load_debug_logs_module()
    logs_dir = tmp_path / ".opentulpa" / "logs" / "server"
    logs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    recent_path = logs_dir / "server-2026-04-27.log"
    old_path = logs_dir / "server-2026-04-01.log"
    recent_path.write_text("recent\n", encoding="utf-8")
    old_path.write_text("old\n", encoding="utf-8")
    os.utime(recent_path, (now.timestamp(), now.timestamp()))
    old_time = (now - timedelta(days=20)).timestamp()
    os.utime(old_path, (old_time, old_time))

    monkeypatch.setattr(debug_logs, "PROJECT_ROOT", tmp_path)

    paths = debug_logs.iter_available_debug_log_paths(lookback_days=7, now=now)

    assert recent_path.resolve() in paths
    assert old_path.resolve() not in paths


def test_build_debug_logs_archive_bytes_contains_recent_logs(
    monkeypatch, tmp_path: Path
) -> None:
    debug_logs = _load_debug_logs_module()
    logs_dir = tmp_path / ".opentulpa" / "logs" / "server"
    logs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime(2026, 4, 27, 12, 0, tzinfo=UTC)
    server_path = logs_dir / "server-2026-04-27.log"
    server_path.write_text("server stderr\n", encoding="utf-8")
    os.utime(server_path, (now.timestamp(), now.timestamp()))

    monkeypatch.setattr(debug_logs, "PROJECT_ROOT", tmp_path)

    archive = debug_logs.build_debug_logs_archive_bytes(lookback_days=7, now=now)

    assert archive is not None
    filename, raw_bytes = archive
    assert filename.startswith("opentulpa-debug-logs-last-7-days-")
    with zipfile.ZipFile(io.BytesIO(raw_bytes), mode="r") as zipped:
        assert ".opentulpa/logs/server/server-2026-04-27.log" in zipped.namelist()
        assert zipped.read(".opentulpa/logs/server/server-2026-04-27.log") == b"server stderr\n"


def test_install_process_output_log_capture_tees_stdout_and_stderr(
    monkeypatch, tmp_path: Path
) -> None:
    debug_logs = _load_debug_logs_module()
    stdout = io.StringIO()
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    server_log_path = debug_logs.install_process_output_log_capture(project_root=tmp_path)
    sys.stdout.write("stdout line\n")
    sys.stderr.write("stderr line\n")

    assert stdout.getvalue() == "stdout line\n"
    assert stderr.getvalue() == "stderr line\n"
    text = server_log_path.read_text(encoding="utf-8")
    assert "[stdout] stdout line" in text
    assert "[stderr] stderr line" in text


def test_install_process_output_log_capture_emits_output_events(
    monkeypatch, tmp_path: Path
) -> None:
    debug_logs = _load_debug_logs_module()
    stdout = io.StringIO()
    stderr = io.StringIO()
    events: list[dict[str, object]] = []
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    debug_logs.install_process_output_log_capture(project_root=tmp_path, event_callback=events.append)
    sys.stdout.write("stdout line\n")
    sys.stderr.write("stderr line\n")

    assert [event["stream"] for event in events] == ["stdout", "stderr"]
    assert [event["message"] for event in events] == ["stdout line", "stderr line"]
    assert all(event["project_root"] == str(tmp_path.resolve()) for event in events)


def test_configure_process_output_event_callback_updates_existing_tees(
    monkeypatch, tmp_path: Path
) -> None:
    debug_logs = _load_debug_logs_module()
    stdout = io.StringIO()
    stderr = io.StringIO()
    first_events: list[dict[str, object]] = []
    second_events: list[dict[str, object]] = []
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    debug_logs.install_process_output_log_capture(
        project_root=tmp_path,
        event_callback=first_events.append,
    )
    debug_logs.configure_process_output_event_callback(second_events.append)
    sys.stdout.write("new callback\n")

    assert first_events == []
    assert [event["message"] for event in second_events] == ["new callback"]
