from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_start(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        ["bash", "./start.sh", *args],
        cwd=REPO_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_start_script_help_shows_install_and_runtime_flags() -> None:
    result = _run_start("--help")

    assert result.returncode == 0
    assert "up|install|run" in result.stdout
    assert "--app" in result.stdout
    assert "--manager" in result.stdout
    assert "--browser-use" in result.stdout


def test_start_script_dry_run_direct_app_mode() -> None:
    result = _run_start(
        "--dry-run",
        "--run-only",
        env={"PUBLIC_BASE_URL": "https://example.com"},
    )

    assert result.returncode == 0
    assert "[start] running direct app mode." in result.stdout
    assert "uv run python -m opentulpa" in result.stdout
    assert "scripts/manager.py" not in result.stdout


def test_start_script_dry_run_install_only_skips_browser_use_when_disabled() -> None:
    result = _run_start(
        "--dry-run",
        "--install-only",
        "--app",
        "--no-browser-use",
    )

    assert result.returncode == 0
    assert "[start] uv sync" in result.stdout
    assert "skipping Browser Use Chromium install." in result.stdout
    assert "playwright install chromium" not in result.stdout
