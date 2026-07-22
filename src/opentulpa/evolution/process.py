"""Bounded subprocess execution used at candidate trust boundaries."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
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
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(process)
        if timeout_cleanup is not None:
            timeout_cleanup()
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


__all__ = ["BoundedProcessResult", "run_bounded_process"]
