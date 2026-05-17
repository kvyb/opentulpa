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
    "OPENTULPA_WEB_TOKEN": "",
    "COMPOSIO_API_KEY": "",
    "TELEGRAM_ALLOWED_USERNAMES": "",
    "TELEGRAM_ALLOWED_USER_IDS": "",
    "OPENAI_COMPATIBLE_BASE_URL": "https://openrouter.ai/api/v1",
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
        env={**EMPTY_REQUIRED_ENV, "TELEGRAM_BOT_TOKEN": "test-token"},
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" in result.stdout
    assert "OPENAI_COMPATIBLE_API_KEY" in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET" in result.stdout
    assert "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN" in result.stdout
    assert "OPENTULPA_DATA_ROOT" in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" in result.stdout
    assert "warning: COMPOSIO_API_KEY is not set" in result.stdout
    assert "[start] running server mode." in result.stdout
    assert "uv run python -m opentulpa" in result.stdout
    assert "scripts/manager.py" not in result.stdout


def test_start_script_dry_run_server_mode_allows_web_only_without_telegram() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env=EMPTY_REQUIRED_ENV,
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" in result.stdout
    assert "OPENAI_COMPATIBLE_API_KEY" in result.stdout
    assert "OPENTULPA_WEB_TOKEN" in result.stdout
    assert "OPENTULPA_DATA_ROOT" in result.stdout
    assert "TELEGRAM_BOT_TOKEN" not in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET" not in result.stdout
    assert "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN" not in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" not in result.stdout
    assert "server Telegram disabled; web/API startup does not require Telegram env." in result.stdout
    assert "uv run python -m opentulpa" in result.stdout


def test_start_script_dry_run_server_mode_accepts_web_only_env() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_DATA_ROOT": "/tmp/opentulpa-test-data",
            "OPENTULPA_WEB_TOKEN": "test-web-token",
        },
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" not in result.stdout
    assert "TELEGRAM_BOT_TOKEN" not in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET" not in result.stdout
    assert "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN" not in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" not in result.stdout
    assert "server Telegram disabled; web/API startup does not require Telegram env." in result.stdout
    assert "uv run python -m opentulpa" in result.stdout


def test_start_script_doctor_server_web_only_requires_web_token() -> None:
    result = _run_start("doctor", "server", env=EMPTY_REQUIRED_ENV)

    assert result.returncode == 1
    assert "server Telegram disabled; skipping Telegram token and allowlist checks" in result.stdout
    assert "server Telegram disabled; skipping webhook URL/secret checks" in result.stdout
    assert "fail: OPENTULPA_WEB_TOKEN is set" in result.stdout
    assert "TELEGRAM_BOT_TOKEN is set" not in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET is set" not in result.stdout


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


def test_start_script_warns_when_base_url_is_not_openrouter() -> None:
    env = {
        **EMPTY_REQUIRED_ENV,
        "OPENAI_COMPATIBLE_BASE_URL": "https://api.openai.com/v1",
    }
    result = _run_start(
        "server",
        "--dry-run",
        env=env,
    )

    assert result.returncode == 0
    assert "OPENAI_COMPATIBLE_BASE_URL is not OpenRouter" in result.stdout
    assert "opentulpa.config.yaml model settings" in result.stdout
    assert "llm_model" in result.stdout
    assert "wake_execution_model" in result.stdout
    assert "workflow_setup_input_classifier_model" in result.stdout
    assert "memory_llm_model" in result.stdout
    assert "multimodal_llm" in result.stdout
    assert "business_knowledge_oracle_model" in result.stdout
    assert "openai_compatible_embedding_model" in result.stdout
    assert "browser_use_model" in result.stdout


def test_start_script_doctor_warns_for_configured_models_missing_from_catalog(tmp_path: Path) -> None:
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        """#!/usr/bin/env bash
if [[ "$*" == *"/models"* ]]; then
  printf '%s\n' '{"data":[{"id":"z-ai/glm-5.1"}]}'
  exit 0
fi
exit 22
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "OPENAI_COMPATIBLE_API_KEY": "test-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://provider.example/v1",
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_WEBHOOK_SECRET": "test-secret",
        "PUBLIC_BASE_URL": "https://app.example",
        "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
        "TELEGRAM_ALLOWED_USERNAMES": "owner",
        "TELEGRAM_ALLOWED_USER_IDS": "",
        "COMPOSIO_API_KEY": "",
    }

    result = _run_start("doctor", "server", env=env)

    assert result.returncode == 0
    assert "https://provider.example/v1/models did not list configured model(s)" in result.stdout
    assert "memory_llm_model=google/gemini-3-flash-preview" in result.stdout
    assert "multimodal_llm=google/gemini-3.1-flash-lite-preview" in result.stdout
    assert "openai_compatible_embedding_model=openai/text-embedding-3-small" in result.stdout


def test_start_script_run_server_accepts_platform_env_without_dotenv(tmp_path: Path) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    env = {
        "OPENAI_COMPATIBLE_API_KEY": "test-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://openrouter.ai/api/v1",
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_WEBHOOK_SECRET": "test-secret",
        "RAILWAY_PUBLIC_DOMAIN": "opentulpa.example.railway.app",
        "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
        "TELEGRAM_ALLOWED_USERNAMES": "owner",
        "TELEGRAM_ALLOWED_USER_IDS": "",
        "COMPOSIO_API_KEY": "",
    }

    result = subprocess.run(
        ["bash", "./start.sh", "run", "server", "--dry-run"],
        cwd=tmp_path,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert ".env is missing" not in result.stderr
    assert ".env.example was not found" not in result.stderr
    assert "required .env value(s) missing" not in result.stdout
    assert "uv run python -m opentulpa" in result.stdout


def test_start_script_server_accepts_railway_public_domain_fallback() -> None:
    env = {
        **EMPTY_REQUIRED_ENV,
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_WEBHOOK_SECRET": "test-secret",
        "RAILWAY_PUBLIC_DOMAIN": "opentulpa.example.railway.app",
        "TELEGRAM_ALLOWED_USERNAMES": "owner",
    }

    result = _run_start("server", "--dry-run", env=env)

    assert result.returncode == 0
    assert "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN" not in result.stdout
