from __future__ import annotations

from types import SimpleNamespace

import pytest

from opentulpa.tasks import sandbox


def test_run_terminal_strips_tulpa_stuff_prefix_from_script_path(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "AGENT_VENV_DIR", sandbox.REPO_VENV_DIR)
    captured: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)

    result = sandbox.run_terminal(
        command="python3 tulpa_stuff/tg_login.py",
        working_dir="tulpa_stuff",
        timeout_seconds=20,
    )

    assert result["ok"] is True
    assert captured[0][0] == ["python3", "tg_login.py"]
    assert captured[0][1]["cwd"] == str(sandbox.TULPA_STUFF_DIR)


def test_run_terminal_strips_opentulpa_prefix_from_script_path(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "AGENT_VENV_DIR", sandbox.REPO_VENV_DIR)
    captured: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)

    result = sandbox.run_terminal(
        command="python3 src/opentulpa/integrations/demo.py",
        working_dir="opentulpa",
        timeout_seconds=20,
    )

    assert result["ok"] is True
    assert captured[0][0] == ["python3", "integrations/demo.py"]
    assert captured[0][1]["cwd"] == str(sandbox.PACKAGE_ROOT)


def test_run_terminal_allows_custom_command_names_by_default(monkeypatch) -> None:
    monkeypatch.delenv(sandbox.TERMINAL_COMMAND_ALLOWLIST_ENV, raising=False)
    monkeypatch.setattr(sandbox, "AGENT_VENV_DIR", sandbox.REPO_VENV_DIR)
    captured: list[tuple[list[str], dict[str, object]]] = []

    def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        captured.append((list(args), dict(kwargs)))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)

    result = sandbox.run_terminal(
        command="agent-context query hello --json",
        working_dir="tulpa_stuff",
        timeout_seconds=20,
    )

    assert result["ok"] is True
    assert captured[0][0] == ["agent-context", "query", "hello", "--json"]
    assert captured[0][1]["cwd"] == str(sandbox.TULPA_STUFF_DIR)


def test_run_terminal_rejects_non_allowlisted_command_when_allowlist_configured(monkeypatch) -> None:
    monkeypatch.setenv(sandbox.TERMINAL_COMMAND_ALLOWLIST_ENV, "python3,uv")

    with pytest.raises(PermissionError, match="OPENTULPA_TERMINAL_COMMAND_ALLOWLIST"):
        sandbox.run_terminal(
            command="agent-context query hello --json",
            working_dir="tulpa_stuff",
            timeout_seconds=20,
        )


def test_run_terminal_logs_timeout_before_raising(monkeypatch) -> None:
    monkeypatch.setattr(sandbox, "AGENT_VENV_DIR", sandbox.REPO_VENV_DIR)
    events: list[tuple[str, dict[str, object]]] = []

    def _fake_debug_log(*, hypothesis_id, location, message, data):  # type: ignore[no-untyped-def]
        events.append((message, dict(data)))

    def _fake_run(args, **kwargs):  # type: ignore[no-untyped-def]
        raise sandbox.subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(sandbox, "_debug_log", _fake_debug_log)
    monkeypatch.setattr(sandbox.subprocess, "run", _fake_run)

    with pytest.raises(TimeoutError, match="command timed out"):
        sandbox.run_terminal(
            command="agent-context query hello --json",
            working_dir="tulpa_stuff",
            timeout_seconds=20,
        )

    assert ("terminal_command_timeout", {"working_dir": "tulpa_stuff", "command_bin": "agent-context", "timeout_seconds": 20}) in events
