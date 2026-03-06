from __future__ import annotations

from types import SimpleNamespace

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
