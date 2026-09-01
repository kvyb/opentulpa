"""Bounded subprocess execution used at candidate trust boundaries."""

from __future__ import annotations

import ctypes
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BoundedProcessResult:
    returncode: int
    output: bytes
    truncated: bool
    timed_out: bool


def run_bounded_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: float,
    max_output_bytes: int,
    timeout_cleanup: Callable[[], None] | None = None,
    abort_event: threading.Event | None = None,
) -> BoundedProcessResult:
    """Drain stdout continuously while retaining only a fixed-size prefix."""

    if not argv or max_output_bytes < 1 or timeout_seconds <= 0:
        raise ValueError("bounded process configuration is invalid")
    process = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream = process.stdout
    assert stream is not None
    retained = bytearray()
    output_size = 0

    def drain_output() -> None:
        nonlocal output_size
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            output_size += len(chunk)
            remaining = max_output_bytes - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])

    reader = threading.Thread(target=drain_output, name="opentulpa-bounded-output", daemon=True)
    reader.start()
    timed_out = False
    try:
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            if abort_event is not None and abort_event.is_set():
                _kill_process_group(process)
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                _kill_process_group(process)
                if timeout_cleanup is not None:
                    timeout_cleanup()
                break
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        reader.join(timeout=5)
        stream.close()
    return BoundedProcessResult(
        returncode=124 if timed_out else int(process.returncode),
        output=bytes(retained),
        truncated=output_size > len(retained),
        timed_out=timed_out,
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            process.kill()
        except OSError:
            return


def _run_as_subreaper(argv: Sequence[str]) -> int:
    if not argv or not sys.platform.startswith("linux"):
        return 127
    try:
        if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:
            return 127
    except (AttributeError, OSError):
        return 127
    process = subprocess.Popen(list(argv))
    returncode: int | None = None
    while True:
        try:
            pid, status = os.wait()
        except ChildProcessError:
            break
        if pid == process.pid:
            returncode = os.waitstatus_to_exitcode(status)
            process.returncode = returncode
    return returncode if returncode is not None else 127


def _main() -> None:
    argv = sys.argv[1:]
    if argv[:1] == ["--"]:
        argv = argv[1:]
    raise SystemExit(_run_as_subreaper(argv))


if __name__ == "__main__":
    _main()


__all__ = ["BoundedProcessResult", "run_bounded_process"]
