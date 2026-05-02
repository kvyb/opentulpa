from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EMPTY_REQUIRED_ENV = {
    "OPENAI_COMPATIBLE_API_KEY": "",
    "TELEGRAM_BOT_TOKEN": "",
    "TELEGRAM_WEBHOOK_SECRET": "",
    "PUBLIC_BASE_URL": "",
    "OPENTULPA_DATA_ROOT": "",
    "COMPOSIO_API_KEY": "",
    "TELEGRAM_ALLOWED_USERNAMES": "",
    "TELEGRAM_ALLOWED_USER_IDS": "",
}


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
    assert "local|server|install|run|doctor" in result.stdout
    assert "--yes" in result.stdout
    assert "--no-install-uv" in result.stdout
    assert "--browser-use" in result.stdout
    assert "UV_PYTHON=3.12" in result.stdout


def test_start_script_dry_run_server_mode() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env=EMPTY_REQUIRED_ENV,
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" in result.stdout
    assert "OPENAI_COMPATIBLE_API_KEY" in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET" in result.stdout
    assert "PUBLIC_BASE_URL" in result.stdout
    assert "OPENTULPA_DATA_ROOT" in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" in result.stdout
    assert "warning: COMPOSIO_API_KEY is not set" in result.stdout
    assert "[start] running server mode." in result.stdout
    assert "uv run python -m opentulpa" in result.stdout
    assert "scripts/manager.py" not in result.stdout


def test_start_script_dry_run_local_mode() -> None:
    result = _run_start(
        "local",
        "--dry-run",
        env=EMPTY_REQUIRED_ENV,
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for local:" in result.stdout
    assert "TELEGRAM_BOT_TOKEN" in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" in result.stdout
    assert "warning: COMPOSIO_API_KEY is not set" in result.stdout
    assert "OPENTULPA_DATA_ROOT" not in result.stdout
    assert "[start] running local Telegram mode." in result.stdout
    assert "uv run python scripts/manager.py" in result.stdout


def test_start_script_dry_run_install_only_skips_browser_use_when_disabled() -> None:
    result = _run_start(
        "--dry-run",
        "--install-only",
        "--server",
        "--no-browser-use",
    )

    assert result.returncode == 0
    assert "[start] uv sync" in result.stdout
    assert "skipping Browser Use Chromium install." in result.stdout
    assert "playwright install chromium" not in result.stdout


def test_start_script_deprecated_app_alias_maps_to_server_mode() -> None:
    result = _run_start(
        "--dry-run",
        "--run-only",
        "--app",
    )

    assert result.returncode == 0
    assert "--app is deprecated" in result.stderr
    assert "[start] running server mode." in result.stdout
    assert "uv run python -m opentulpa" in result.stdout


def test_start_script_deprecated_manager_alias_maps_to_local_mode() -> None:
    result = _run_start(
        "--dry-run",
        "--run-only",
        "--manager",
    )

    assert result.returncode == 0
    assert "--manager is deprecated" in result.stderr
    assert "[start] running local Telegram mode." in result.stdout
    assert "uv run python scripts/manager.py" in result.stdout


def test_start_script_missing_uv_with_no_install_fails_with_install_command() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        "--no-install-uv",
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode != 0
    assert "uv is required but was not found in PATH" in result.stderr
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in result.stderr


def test_start_script_missing_uv_dry_run_bootstraps_by_default_then_syncs() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={"PATH": "/usr/bin:/bin"},
    )

    assert result.returncode == 0
    assert "uv was not found in PATH; bootstrapping uv." in result.stdout
    assert "curl -LsSf https://astral.sh/uv/install.sh | sh" in result.stdout
    assert "[start] uv sync" in result.stdout
    assert "uv run python -m opentulpa" in result.stdout
