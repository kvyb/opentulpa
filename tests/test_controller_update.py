from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import opentulpa.host.controller_update as controller_update
from opentulpa.host.controller_update import (
    SystemdControllerUpdater,
    run_controller_update,
)


def test_systemd_updater_schedules_private_detached_request(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    updater, state_path, commands = _scheduled_update(tmp_path, monkeypatch)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "scheduled"
    assert state["source_commit"] == "b" * 40
    assert state_path.stat().st_mode & 0o777 == 0o600
    assert "--no-block" in commands[0]
    assert commands[0][-3:] == (
        "-m",
        "opentulpa.host.controller_update",
        str(state_path),
    )
    assert updater.pending_event() is None

    controller_update._update_state(
        state_path,
        status="succeeded",
        controller_generation="2" * 64,
    )
    event = updater.pending_event()
    assert event is not None
    assert event.event_type == "controller.active"
    updater.mark_notified(event)
    assert updater.has_pending_notification() is False


def test_controller_update_activates_reviewed_generation(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _, state_path, _ = _scheduled_update(tmp_path, monkeypatch)
    controller = tmp_path / "install" / "controller"
    targets: list[str] = []

    def run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        if command[0].endswith("/bin/opentulpa"):
            _activate(controller, "2" * 64, "b" * 40)
        return subprocess.CompletedProcess(command, 0, "", "")

    def wait(**kwargs: Any) -> bool:
        targets.append(str(kwargs["target"]))
        return True

    monkeypatch.setattr(controller_update, "_run", run)
    monkeypatch.setattr(controller_update, "_require_exact_source", lambda *args: None)
    monkeypatch.setattr(controller_update, "_wait_for_service", wait)

    assert run_controller_update(state_path) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "succeeded"
    assert state["controller_generation"] == "2" * 64
    assert targets == [f"generations/{'2' * 64}"]


def test_controller_update_restores_previous_generation_when_health_fails(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _, state_path, _ = _scheduled_update(tmp_path, monkeypatch)
    controller = tmp_path / "install" / "controller"

    def run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        if command[0].endswith("/bin/opentulpa"):
            _activate(controller, "2" * 64, "b" * 40)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(controller_update, "_run", run)
    monkeypatch.setattr(controller_update, "_require_exact_source", lambda *args: None)
    monkeypatch.setattr(
        controller_update,
        "_wait_for_service",
        lambda **kwargs: str(kwargs["target"]) == f"generations/{'1' * 64}",
    )

    assert run_controller_update(state_path) == 1
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "rolled_back"
    assert os.readlink(controller / "current") == f"generations/{'1' * 64}"


def test_process_generation_check_uses_launch_arguments_without_resolving_venv_symlinks(
    tmp_path: Path,
) -> None:
    generation = tmp_path / "controller" / "generations" / ("1" * 64)
    process = tmp_path / "proc" / "42"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(
        os.fsencode(generation / "bin" / "python")
        + b"\0"
        + os.fsencode(generation / "bin" / "opentulpa-host")
        + b"\0"
    )

    assert controller_update._process_uses_generation(
        42,
        generation,
        proc_root=tmp_path / "proc",
    )


def _scheduled_update(
    tmp_path: Path,
    monkeypatch: Any,
) -> tuple[SystemdControllerUpdater, Path, list[tuple[str, ...]]]:
    install = tmp_path / "install"
    controller = install / "controller"
    old = controller / "generations" / ("1" * 64)
    new = controller / "generations" / ("2" * 64)
    old.mkdir(parents=True)
    new.mkdir(parents=True)
    (controller / "current").symlink_to(f"generations/{'1' * 64}")
    _metadata(install, "a" * 40, "1" * 64)
    executable = install / "bin" / "opentulpa"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o700)
    source = tmp_path / "source"
    source.mkdir()
    state_path = tmp_path / "control" / "controller-update.json"
    commands: list[tuple[str, ...]] = []

    def run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        commands.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(controller_update, "_run", run)
    updater = SystemdControllerUpdater(
        state_path=state_path,
        source_root=source,
        install_root=install,
        systemd_unit="opentulpa.service",
        health_url="http://127.0.0.1:8000/agent/healthz",
        systemd_run=Path("/usr/bin/systemd-run"),
        systemctl=Path("/usr/bin/systemctl"),
        git=Path("/usr/bin/git"),
        python_executable=Path("/controller/bin/python"),
    )
    assert (
        updater.schedule(
            activation_id="activation-1",
            release_id="release-1",
            source_commit="b" * 40,
            audit={"tenant_id": "owner"},
        )
        == "scheduled"
    )
    return updater, state_path, commands


def _activate(controller: Path, generation: str, commit: str) -> None:
    current = controller / "current"
    current.unlink()
    current.symlink_to(f"generations/{generation}")
    _metadata(controller.parent, commit, generation)


def _metadata(install: Path, source_oid: str, generation: str) -> None:
    path = install / "controller" / "install.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"format_version": 1, "source_oid": source_oid}, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    manifest = path.parent / "generations" / generation / "source-seed-manifest.json"
    manifest.write_text(
        json.dumps(
            {"format_version": 1, "source_commit": source_oid},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    manifest.chmod(0o600)
