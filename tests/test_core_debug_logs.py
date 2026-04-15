from __future__ import annotations

import importlib.util
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
