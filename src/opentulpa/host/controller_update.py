"""One-shot systemd controller updates for reviewer-approved VPS releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import time
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from opentulpa.evolution.models import EvolutionEvent
from opentulpa.evolution.process import run_bounded_process

_COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
_GENERATION_RE = re.compile(r"^generations/[0-9a-f]{64}$")
_UNIT_RE = re.compile(r"^[A-Za-z0-9_.@-]+\.service$")
_TERMINAL = frozenset({"succeeded", "rolled_back", "failed"})


class ControllerUpdateError(RuntimeError):
    """A trusted controller update could not be scheduled or completed."""


class SystemdControllerUpdater:
    """Start one detached root worker and expose its durable result."""

    def __init__(
        self,
        *,
        state_path: Path,
        source_root: Path,
        install_root: Path,
        systemd_unit: str,
        health_url: str,
        systemd_run: Path,
        systemctl: Path,
        git: Path,
        python_executable: Path,
    ) -> None:
        if _UNIT_RE.fullmatch(systemd_unit) is None:
            raise ControllerUpdateError("OPENTULPA_SYSTEMD_UNIT must name a .service unit")
        if re.fullmatch(r"http://127\.0\.0\.1:[1-9][0-9]{0,4}/agent/healthz", health_url) is None:
            raise ControllerUpdateError("controller health URL must use the local host endpoint")
        self._state_path = state_path.absolute()
        self._source_root = source_root.resolve()
        self._install_root = install_root.resolve()
        self._unit = systemd_unit
        self._health_url = health_url
        self._systemd_run = systemd_run.resolve()
        self._systemctl = systemctl.resolve()
        self._git = git.resolve()
        self._python = python_executable.absolute()

    def schedule(
        self,
        *,
        activation_id: str,
        release_id: str,
        source_commit: str,
        audit: Mapping[str, Any],
    ) -> str:
        commit = _commit(source_commit)
        existing = _read_optional_state(self._state_path)
        if existing is not None:
            if existing.get("activation_id") == activation_id:
                return str(existing["status"])
            if existing.get("notified") is not True:
                raise ControllerUpdateError("the previous controller update result is still pending")
        if _active_source_commit(self._install_root) == commit:
            return "already_active"

        state = {
            "activation_id": activation_id,
            "audit": dict(audit),
            "controller_generation": None,
            "error": None,
            "format_version": 1,
            "git": str(self._git),
            "health_url": self._health_url,
            "install_root": str(self._install_root),
            "notified": False,
            "previous_target": None,
            "release_id": release_id,
            "source_commit": commit,
            "source_root": str(self._source_root),
            "status": "scheduled",
            "systemctl": str(self._systemctl),
            "systemd_unit": self._unit,
        }
        _write_state(self._state_path, state)
        name = "opentulpa-controller-update-" + hashlib.sha256(
            activation_id.encode()
        ).hexdigest()[:16]
        try:
            completed = _run(
                (
                    str(self._systemd_run),
                    "--quiet",
                    "--collect",
                    "--no-block",
                    f"--unit={name}",
                    "--property=Type=exec",
                    "--property=PrivateTmp=yes",
                    "--property=UMask=0077",
                    "--",
                    str(self._python),
                    "-m",
                    "opentulpa.host.controller_update",
                    str(self._state_path),
                ),
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _update_state(self._state_path, status="failed", error=str(exc)[:2_000])
            raise ControllerUpdateError("systemd could not schedule the controller update") from exc
        if completed.returncode != 0:
            error = _process_error("systemd refused the controller update", completed)
            _update_state(self._state_path, status="failed", error=error)
            raise ControllerUpdateError(error)
        return "scheduled"

    def in_progress(self) -> bool:
        state = _read_optional_state(self._state_path)
        return bool(state is not None and state.get("status") in {"scheduled", "running"})

    def has_pending_notification(self) -> bool:
        state = _read_optional_state(self._state_path)
        return bool(state is not None and state.get("notified") is not True)

    def pending_event(self) -> EvolutionEvent | None:
        state = _read_optional_state(self._state_path)
        if state is None or state.get("status") not in _TERMINAL or state.get("notified") is True:
            return None
        status = str(state["status"])
        payload: dict[str, Any] = {
            "status": "active" if status == "succeeded" else status,
            "activation_id": str(state["activation_id"]),
            "source_commit": str(state["source_commit"]),
            "controller_generation": state.get("controller_generation"),
        }
        if state.get("error"):
            payload["error"] = str(state["error"])
        return EvolutionEvent(
            event_key=f"controller:{state['activation_id']}:{status}",
            event_type={
                "succeeded": "controller.active",
                "rolled_back": "controller.rolled_back",
                "failed": "controller.failed",
            }[status],
            release_id=str(state["release_id"]),
            origin=dict(state.get("audit") or {}),
            payload=payload,
        )

    def mark_notified(self, event: EvolutionEvent) -> None:
        state = _load_state(self._state_path)
        if event.event_key == f"controller:{state['activation_id']}:{state['status']}":
            _update_state(self._state_path, notified=True)


def run_controller_update(state_path: Path) -> int:
    """Install, restart, validate, and restore the old generation on failure."""

    state = _load_state(state_path)
    if state["status"] != "scheduled":
        return 0 if state["status"] == "succeeded" else 1
    controller_root = Path(str(state["install_root"])) / "controller"
    previous = _current_target(controller_root)
    target: str | None = None
    _update_state(state_path, status="running", previous_target=previous)
    try:
        source = Path(str(state["source_root"]))
        commit = _commit(str(state["source_commit"]))
        _require_exact_source(str(state["git"]), source, commit)
        updater = Path(str(state["install_root"])) / "bin" / "opentulpa"
        _require_private_executable(updater)
        completed = _run((str(updater), "update", "--source", str(source)), timeout=900)
        if completed.returncode != 0:
            raise ControllerUpdateError(_process_error("controller installer failed", completed))
        target = _current_target(controller_root)
        if target == previous or _active_source_commit(Path(str(state["install_root"]))) != commit:
            raise ControllerUpdateError("installer did not activate the reviewed commit")
        _restart(str(state["systemctl"]), str(state["systemd_unit"]))
        if not _wait_for_service(
            systemctl=str(state["systemctl"]),
            unit=str(state["systemd_unit"]),
            health_url=str(state["health_url"]),
            controller_root=controller_root,
            target=target,
        ):
            raise ControllerUpdateError("updated controller failed host health validation")
        if _current_target(controller_root) != target:
            raise ControllerUpdateError("controller generation changed during validation")
        _update_state(
            state_path,
            status="succeeded",
            controller_generation=target.removeprefix("generations/"),
            error=None,
        )
        return 0
    except Exception as exc:
        error = str(exc).strip()[:2_000] or type(exc).__name__
        try:
            current = _current_target(controller_root)
            if target is not None and current == target:
                _switch_current(controller_root, previous)
                _restart(str(state["systemctl"]), str(state["systemd_unit"]))
            restored = _current_target(controller_root) == previous and _wait_for_service(
                systemctl=str(state["systemctl"]),
                unit=str(state["systemd_unit"]),
                health_url=str(state["health_url"]),
                controller_root=controller_root,
                target=previous,
            )
        except Exception as rollback_exc:
            restored = False
            error = f"{error}; rollback failed: {rollback_exc}"[:2_000]
        _update_state(
            state_path,
            status="rolled_back" if restored else "failed",
            controller_generation=previous.removeprefix("generations/") if restored else None,
            error=error,
        )
        return 1


def _require_exact_source(git: str, source: Path, commit: str) -> None:
    head = _run((git, "-C", str(source), "rev-parse", "HEAD"), timeout=20)
    changes = _run(
        (git, "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"),
        timeout=20,
    )
    if head.returncode or head.stdout.strip() != commit or changes.returncode or changes.stdout.strip():
        raise ControllerUpdateError("controller source no longer matches the clean reviewed commit")


def _restart(systemctl: str, unit: str) -> None:
    completed = _run((systemctl, "restart", unit), timeout=60)
    if completed.returncode:
        raise ControllerUpdateError(_process_error("systemd restart failed", completed))


def _wait_for_service(
    *,
    systemctl: str,
    unit: str,
    health_url: str,
    controller_root: Path,
    target: str,
    timeout: float = 90,
) -> bool:
    deadline = time.monotonic() + timeout
    generation = (controller_root / target).absolute()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            process = _run(
                (systemctl, "show", "--property=MainPID", "--value", unit), timeout=10
            )
            pid = int(process.stdout.strip()) if process.returncode == 0 else 0
            with opener.open(health_url, timeout=min(3, max(0.1, deadline - time.monotonic()))) as response:
                body = response.read(65_537)
            payload = json.loads(body) if len(body) <= 65_536 else None
            if (
                pid > 1
                and _process_uses_generation(pid, generation)
                and isinstance(payload, dict)
                and payload.get("ok") is True
            ):
                return True
        except (AttributeError, OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    return False


def _process_uses_generation(pid: int, generation: Path, *, proc_root: Path = Path("/proc")) -> bool:
    command_line = (proc_root / str(pid) / "cmdline").read_bytes()
    return len(command_line) <= 65_536 and any(
        (argument := Path(os.fsdecode(raw))).is_absolute()
        and argument.absolute().is_relative_to(generation)
        for raw in command_line.split(b"\0")
    )


def _current_target(controller_root: Path) -> str:
    current = controller_root / "current"
    if not current.is_symlink():
        raise ControllerUpdateError("controller/current is not a symbolic link")
    target = os.readlink(current)
    generation = controller_root / target
    if _GENERATION_RE.fullmatch(target) is None or generation.is_symlink() or not generation.is_dir():
        raise ControllerUpdateError("controller/current has an invalid generation target")
    return target


def _switch_current(controller_root: Path, target: str) -> None:
    if _GENERATION_RE.fullmatch(target) is None or not (controller_root / target).is_dir():
        raise ControllerUpdateError("rollback generation is unavailable")
    temporary = controller_root / f".current.rollback.{os.getpid()}"
    os.symlink(target, temporary)
    os.replace(temporary, controller_root / "current")
    descriptor = os.open(controller_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _active_source_commit(install_root: Path) -> str:
    controller = install_root / "controller"
    target = _current_target(controller)
    metadata = _load_private_json(controller / target / "source-seed-manifest.json")
    return _commit(str(metadata.get("source_commit") or ""))


def _commit(value: str) -> str:
    value = value.strip().lower()
    if _COMMIT_RE.fullmatch(value) is None:
        raise ControllerUpdateError("controller source commit is invalid")
    return value


def _require_private_executable(path: Path) -> None:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
        or not metadata.st_mode & stat.S_IXUSR
    ):
        raise ControllerUpdateError("controller update executable is not private and trusted")


def _load_private_json(path: Path) -> dict[str, Any]:
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise ControllerUpdateError(f"controller update file is untrusted: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise ControllerUpdateError(f"controller update file is not canonical: {path}")
    return value


def _load_state(path: Path) -> dict[str, Any]:
    state = _load_private_json(path)
    if state.get("format_version") != 1 or state.get("status") not in {
        "scheduled",
        "running",
        *_TERMINAL,
    }:
        raise ControllerUpdateError("controller update state is unsupported")
    _commit(str(state.get("source_commit") or ""))
    return state


def _read_optional_state(path: Path) -> dict[str, Any] | None:
    try:
        return _load_state(path)
    except FileNotFoundError:
        return None


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}")
    with os.fdopen(
        os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600),
        "wb",
    ) as stream:
        stream.write(_canonical_json(state))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _update_state(path: Path, **changes: Any) -> None:
    state = _load_state(path)
    state.update(changes)
    _write_state(path, state)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _run(command: Sequence[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    result = run_bounded_process(
        command,
        cwd=Path("/"),
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/root"},
        timeout_seconds=timeout,
        max_output_bytes=1_000_000,
    )
    output = result.output.decode("utf-8", errors="replace")
    if result.truncated:
        output += "\n[output truncated]"
    return subprocess.CompletedProcess(tuple(command), result.returncode, output, "")


def _process_error(prefix: str, completed: subprocess.CompletedProcess[str]) -> str:
    detail = (completed.stderr or completed.stdout or "").strip()[:1_500]
    return f"{prefix} ({completed.returncode})" + (f": {detail}" if detail else "")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("state_path")
    args = parser.parse_args()
    raise SystemExit(run_controller_update(Path(args.state_path).absolute()))


if __name__ == "__main__":
    main()


__all__ = ["ControllerUpdateError", "SystemdControllerUpdater", "run_controller_update"]
