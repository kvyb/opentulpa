from __future__ import annotations

import os
import sys
from pathlib import Path

from opentulpa.evolution.process import run_bounded_process


def test_bounded_process_drains_without_retaining_unlimited_output(tmp_path: Path) -> None:
    result = run_bounded_process(
        (sys.executable, "-c", "import sys; sys.stdout.write('x' * 2_000_000)"),
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", os.defpath)},
        timeout_seconds=10,
        max_output_bytes=1_024,
    )

    assert result.returncode == 0
    assert result.truncated is True
    assert result.output == b"x" * 1_024


def test_bounded_process_invokes_cleanup_after_timeout(tmp_path: Path) -> None:
    cleanup_calls: list[bool] = []

    result = run_bounded_process(
        (sys.executable, "-c", "import time; time.sleep(30)"),
        cwd=tmp_path,
        env={"PATH": os.environ.get("PATH", os.defpath)},
        timeout_seconds=0.05,
        max_output_bytes=1_024,
        timeout_cleanup=lambda: cleanup_calls.append(True),
    )

    assert result.returncode == 124
    assert result.timed_out is True
    assert cleanup_calls == [True]
