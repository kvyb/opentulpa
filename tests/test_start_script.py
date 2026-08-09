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
    "OPENTULPA_OWNER_TOKEN": "",
    "OPENTULPA_RECOVERY_TOKEN": "",
    "OPENTULPA_INGRESS_TOKEN": "",
    "OPENTULPA_RELEASE_EGRESS_NETWORK": "",
    "OPENTULPA_RELEASE_BASE_IMAGE": "",
    "SANDBOX_IMAGE": "",
    "COMPOSIO_API_KEY": "",
    "TELEGRAM_ALLOWED_USERNAMES": "",
    "TELEGRAM_ALLOWED_USER_IDS": "",
    "OPENAI_COMPATIBLE_BASE_URL": "https://openrouter.ai/api/v1",
    "OPENTULPA_EXTRAS": "",
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
    assert "serve|local|server|managed|install|run|doctor" in result.stdout
    assert "Start the headless host, Agent API, and configured interfaces" in result.stdout
    assert "--api-key" in result.stdout
    assert "--telegram-bot-token" in result.stdout
    assert "--telegram-user-id" in result.stdout
    assert "--yes" in result.stdout
    assert "--no-install-uv" in result.stdout
    assert "--browser-use" in result.stdout
    assert "UV_PYTHON=3.12" in result.stdout
    assert "OPENTULPA_OPEN_BROWSER=auto|1|0" in result.stdout
    assert "OPENTULPA_RESTART_GRACE_SECONDS=15" in result.stdout


def test_serve_starts_web_and_local_telegram_from_one_command(tmp_path: Path) -> None:
    result = _run_start(
        "serve",
        "--run-only",
        "--dry-run",
        "--api-key",
        "model-secret",
        "--telegram-bot-token",
        "bot-secret",
        "--telegram-user-id",
        "123456789",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
            "OPENTULPA_OPEN_BROWSER": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "required .env value(s) missing" not in result.stdout
    assert "open http://127.0.0.1:8000/ after the server is healthy" in result.stdout
    assert "uv run --no-sync python -m opentulpa.host" in result.stdout
    assert "scripts/manager.py" not in result.stdout
    assert "model-secret" not in result.stdout
    assert "bot-secret" not in result.stdout


def test_default_start_needs_no_model_key_and_launches_setup_host(tmp_path: Path) -> None:
    result = _run_start(
        "--run-only",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
            "OPENTULPA_OPEN_BROWSER": "0",
        },
    )

    assert result.returncode == 0, result.stderr
    assert "can start before model and interface credentials" in result.stdout
    assert "uv run --no-sync python -m opentulpa.host" in result.stdout
    assert "required .env value(s) missing" not in result.stdout


def test_serve_run_only_can_execute_installed_controller_without_uv(tmp_path: Path) -> None:
    executable = tmp_path / "immutable controller" / "opentulpa-host"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    result = _run_start(
        "serve",
        "--run-only",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENTULPA_CONTROLLER_EXECUTABLE": str(executable),
            "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
            "OPENTULPA_OPEN_BROWSER": "0",
            "PATH": "/usr/bin:/bin",
        },
    )

    assert result.returncode == 0, result.stderr
    assert str(executable) in result.stdout
    assert "uv run" not in result.stdout
    assert "uv sync" not in result.stdout


def test_serve_uses_stable_host_when_public_url_is_present(tmp_path: Path) -> None:
    result = _run_start(
        "serve",
        "--run-only",
        "--dry-run",
        "--api-key",
        "model-secret",
        "--telegram-bot-token",
        "bot-secret",
        "--telegram-user-id",
        "123456789",
        "--public-url",
        "https://tulpa.example.com",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "required .env value(s) missing" not in result.stdout
    assert "uv run --no-sync python -m opentulpa.host" in result.stdout
    assert "scripts/manager.py" not in result.stdout
    assert "Telegram webhook secret" not in result.stdout


def test_serve_accepts_bot_token_without_owner_id() -> None:
    result = _run_start(
        "serve",
        "--run-only",
        "--dry-run",
        "--api-key",
        "model-secret",
        "--telegram-bot-token",
        "bot-secret",
        env=EMPTY_REQUIRED_ENV,
    )

    assert result.returncode == 0
    assert "python -m opentulpa.host" in result.stdout


def test_serve_does_not_persist_command_line_secrets_in_dotenv(tmp_path: Path) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    (tmp_path / ".env.example").write_text("# configuration\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; SERVE_MODE=1; DRY_RUN=0; "
            "CLI_API_KEY=model-secret; CLI_TELEGRAM_BOT_TOKEN=bot-secret; "
            "CLI_TELEGRAM_USER_ID=123456789; configure_serve",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "OPENAI_COMPATIBLE_API_KEY": "",
            "TELEGRAM_BOT_TOKEN": "",
            "TELEGRAM_ALLOWED_USER_IDS": "",
            "TELEGRAM_ALLOWED_USERNAMES": "",
            "PUBLIC_BASE_URL": "",
            "RAILWAY_PUBLIC_DOMAIN": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / ".env").exists()
    assert "model-secret" not in result.stdout
    assert "bot-secret" not in result.stdout
    assert "shell history" in result.stderr


def test_container_and_railway_use_direct_immutable_controller() -> None:
    executable = "/opt/opentulpa-install/controller/generations/image/bin/opentulpa-host"
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert f'CMD ["{executable}"]' in dockerfile
    assert "uv build --wheel" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "--only-binary=:all:" in dockerfile
    assert "--no-binary=pysher" in dockerfile
    assert "setuptools==80.9.0" in dockerfile
    assert "--resume-retries 10" in dockerfile
    assert "--mount=type=cache" not in dockerfile
    assert "target=/root/.cache/pip" not in dockerfile
    assert "target=/root/.cache/uv" not in dockerfile
    assert dockerfile.index("uv export --frozen") < dockerfile.index("COPY src ./src")
    assert dockerfile.index("pip download") < dockerfile.index("COPY src ./src")
    assert "uv build --wheel --offline --no-build-isolation" in dockerfile
    assert "uv run" not in dockerfile
    assert "--editable" not in dockerfile
    assert "uv sync" not in dockerfile
    assert "COPY --from=controller-build /usr/local/bin/uv /usr/local/bin/uv" in dockerfile
    assert "test ! -L /usr/local/bin/uv && /usr/local/bin/uv --version" in dockerfile
    assert "_source_seed_sha256" not in dockerfile
    assert "from opentulpa.host.source_seed import source_seed_sha256" in dockerfile
    assert "ENV OPENTULPA_UV_BIN=/usr/local/bin/uv" in dockerfile
    assert "ENV OPENTULPA_SOURCE_ROOT=/app/opentulpa_data/source" in dockerfile
    assert "ENV EVOLUTION_SOURCE_REPOSITORY=https://github.com/kvyb/opentulpa.git" in dockerfile
    assert "ENV OPENTULPA_INSTALL_REF=main" in dockerfile
    assert "ENV OPENTULPA_SOURCE_SEED_ROOT=/opt/opentulpa-source" in dockerfile
    assert "ENV OPENTULPA_TRUSTED_WHEELHOUSE=" in dockerfile
    assert "USER 65532" not in dockerfile
    railway = (REPO_ROOT / "railway.toml").read_text(encoding="utf-8")
    assert f'startCommand = "{executable}"' in railway
    assert 'healthcheckPath = "/healthz"' in railway
    assert "overlapSeconds = 0" in railway
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert f'command: ["{executable}"]' in compose
    assert "HOST: 0.0.0.0" in compose
    assert "OPENTULPA_DATA_ROOT: /app/opentulpa_data" in compose


def test_start_script_stops_verified_existing_opentulpa_server(tmp_path: Path) -> None:
    state_file = tmp_path / "stopped"
    signal_file = tmp_path / "signal"
    fake_lsof = tmp_path / "lsof"
    fake_lsof.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ ! -f \"$STATE_FILE\" ]]; then printf '%s\\n' 43210; fi\n",
        encoding="utf-8",
    )
    fake_lsof.chmod(0o755)
    fake_ps = tmp_path / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' '/tmp/opentulpa/.venv/bin/python3 -m opentulpa'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; "
            "kill() { printf '%s\\n' \"$*\" > \"$SIGNAL_FILE\"; "
            ": > \"$STATE_FILE\"; }; "
            "DRY_RUN=0; PORT=8123; OPENTULPA_RESTART_GRACE_SECONDS=1; "
            "stop_existing_server",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "STATE_FILE": str(state_file),
            "SIGNAL_FILE": str(signal_file),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "stopping existing OpenTulpa server on port 8123" in result.stdout
    assert "existing OpenTulpa server stopped" in result.stdout
    assert signal_file.read_text(encoding="utf-8").strip() == "-TERM 43210"


def test_start_script_refuses_to_stop_unrelated_port_listener(tmp_path: Path) -> None:
    signal_file = tmp_path / "signal"
    fake_lsof = tmp_path / "lsof"
    fake_lsof.write_text("#!/usr/bin/env bash\nprintf '%s\\n' 43210\n", encoding="utf-8")
    fake_lsof.chmod(0o755)
    fake_ps = tmp_path / "ps"
    fake_ps.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' '/usr/bin/python3 -m another_app'\n",
        encoding="utf-8",
    )
    fake_ps.chmod(0o755)

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; "
            "kill() { : > \"$SIGNAL_FILE\"; }; "
            "DRY_RUN=0; PORT=8123; stop_existing_server",
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "SIGNAL_FILE": str(signal_file),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "which is not OpenTulpa; refusing to stop it" in result.stderr
    assert not signal_file.exists()


def test_launcher_env_upsert_replaces_placeholder_and_reloads(tmp_path: Path) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    (tmp_path / ".env.example").write_text(
        "OPENAI_COMPATIBLE_API_KEY=\nUNCHANGED=value\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; upsert_env_value OPENAI_COMPATIBLE_API_KEY test-key; "
            "unset OPENAI_COMPATIBLE_API_KEY; load_dotenv; "
            "test \"$OPENAI_COMPATIBLE_API_KEY\" = test-key",
        ],
        cwd=tmp_path,
        env={**os.environ, "OPENAI_COMPATIBLE_API_KEY": ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    env_file = tmp_path / ".env"
    assert env_file.read_text(encoding="utf-8").count("OPENAI_COMPATIBLE_API_KEY=") == 1
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_local_owner_token_is_private_and_reused(tmp_path: Path) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    (tmp_path / ".env.example").write_text("", encoding="utf-8")
    data_root = tmp_path / "data"
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; MODE=up; HOST=127.0.0.1; "
            "unset OPENTULPA_OWNER_TOKEN PUBLIC_BASE_URL RAILWAY_PUBLIC_DOMAIN; "
            "configure_local_server_defaults server; first=$OPENTULPA_OWNER_TOKEN; "
            "unset OPENTULPA_OWNER_TOKEN; configure_local_server_defaults server; "
            "test \"$first\" = \"$OPENTULPA_OWNER_TOKEN\"",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "OPENTULPA_DATA_ROOT": str(data_root),
            "OPENTULPA_OWNER_TOKEN": "",
            "PUBLIC_BASE_URL": "",
            "RAILWAY_PUBLIC_DOMAIN": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    token_path = data_root / "bootstrap" / "owner.token"
    assert len(token_path.read_text(encoding="utf-8").strip()) == 64
    assert token_path.stat().st_mode & 0o777 == 0o600


def test_local_owner_token_accepts_host_generated_urlsafe_credential(tmp_path: Path) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    (tmp_path / ".env.example").write_text("", encoding="utf-8")
    data_root = tmp_path / "data"
    token_path = data_root / "bootstrap" / "owner.token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text(f"{'urlsafe_owner-token_' * 3}\n", encoding="utf-8")
    token_path.chmod(0o600)

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; MODE=up; HOST=127.0.0.1; "
            "unset OPENTULPA_OWNER_TOKEN PUBLIC_BASE_URL RAILWAY_PUBLIC_DOMAIN; "
            "configure_local_server_defaults server; "
            "test \"$OPENTULPA_OWNER_TOKEN\" = \"urlsafe_owner-token_urlsafe_owner-token_urlsafe_owner-token_\"",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "OPENTULPA_DATA_ROOT": str(data_root),
            "OPENTULPA_OWNER_TOKEN": "",
            "PUBLIC_BASE_URL": "",
            "RAILWAY_PUBLIC_DOMAIN": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


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
    assert "using local data at" in result.stdout
    assert "private generated owner credential" in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" in result.stdout
    assert "warning: COMPOSIO_API_KEY is not set" in result.stdout
    assert "[start] running server mode." in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout
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
    assert "using local data at" in result.stdout
    assert "private generated owner credential" in result.stdout
    assert "TELEGRAM_BOT_TOKEN" not in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET" not in result.stdout
    assert "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN" not in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" not in result.stdout
    assert "server Telegram disabled; Agent API startup does not require Telegram env." in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout


def test_start_script_public_server_does_not_generate_deployment_credentials() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={**EMPTY_REQUIRED_ENV, "PUBLIC_BASE_URL": "https://agent.example"},
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" in result.stdout
    assert "OPENTULPA_OWNER_TOKEN" in result.stdout
    assert "OPENTULPA_DATA_ROOT" in result.stdout
    assert "private generated owner credential" not in result.stdout


def test_start_script_dry_run_server_mode_accepts_web_only_env() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_DATA_ROOT": "/tmp/opentulpa-test-data",
            "OPENTULPA_OWNER_TOKEN": "test-owner-token",
        },
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" not in result.stdout
    assert "TELEGRAM_BOT_TOKEN" not in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET" not in result.stdout
    assert "PUBLIC_BASE_URL or RAILWAY_PUBLIC_DOMAIN" not in result.stdout
    assert "TELEGRAM_ALLOWED_USERNAMES or TELEGRAM_ALLOWED_USER_IDS" not in result.stdout
    assert "server Telegram disabled; Agent API startup does not require Telegram env." in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout


def test_start_script_doctor_server_web_only_requires_web_token() -> None:
    result = _run_start(
        "doctor",
        "server",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENTULPA_CONTAINER_CLI": "opentulpa-test-missing-container-cli",
        },
    )

    assert result.returncode == 1
    assert "server Telegram disabled; skipping Telegram token and allowlist checks" in result.stdout
    assert "server Telegram disabled; skipping webhook URL/secret checks" in result.stdout
    assert "fail: OPENTULPA_OWNER_TOKEN is set" in result.stdout
    assert "TELEGRAM_BOT_TOKEN is set" not in result.stdout
    assert "TELEGRAM_WEBHOOK_SECRET is set" not in result.stdout


def test_start_script_server_with_telegram_still_requires_web_token() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_WEBHOOK_SECRET": "webhook-secret",
            "PUBLIC_BASE_URL": "https://example.test",
            "TELEGRAM_ALLOWED_USER_IDS": "123",
            "OPENTULPA_DATA_ROOT": "/tmp/opentulpa-test-data",
        },
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server: OPENTULPA_OWNER_TOKEN" in result.stdout


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
    assert "uv run --no-sync python scripts/manager.py" in result.stdout


def test_start_script_dry_run_install_only_skips_browser_cloud_extra_when_disabled() -> None:
    result = _run_start(
        "--dry-run",
        "--install-only",
        "--server",
        "--no-browser-use",
    )

    assert result.returncode == 0
    assert "[start] uv sync --frozen --no-dev" in result.stdout
    assert "--extra browser" not in result.stdout
    assert "playwright install chromium" not in result.stdout


def test_start_script_defaults_to_lean_core_dependencies() -> None:
    result = _run_start("install", "--dry-run")

    assert result.returncode == 0
    assert "[start] uv sync --frozen --no-dev" in result.stdout
    assert "--extra bundled" not in result.stdout
    assert "--extra browser" not in result.stdout
    assert (
        "docker build --tag opentulpa-tenant-sandbox:0.1.0 "
        "--file docker/tenant-sandbox.Dockerfile ."
    ) in result.stdout


def test_start_script_defaults_to_stable_host() -> None:
    result = _run_start(
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_DATA_ROOT": "/tmp/opentulpa-test-data",
            "OPENTULPA_OWNER_TOKEN": "test-owner-token",
        },
    )

    assert result.returncode == 0
    assert "[start] running server mode." in result.stdout
    assert "python -m opentulpa.host" in result.stdout
    assert "scripts/manager.py" not in result.stdout


def test_start_script_can_force_open_local_web_after_health() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_OPEN_BROWSER": "1",
        },
    )

    assert result.returncode == 0
    assert "open http://127.0.0.1:8000/ after the server is healthy" in result.stdout


def test_start_script_does_not_open_browser_for_public_server() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_OPEN_BROWSER": "1",
            "PUBLIC_BASE_URL": "https://agent.example",
        },
    )

    assert result.returncode == 0
    assert "open http://127.0.0.1:8000/ after the server is healthy" not in result.stdout


def test_direct_start_without_container_engine_warns_that_readiness_will_fail(
    tmp_path: Path,
) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; unset OPENTULPA_CONTAINER_CLI; "
            "configure_container_engine server; "
            "test \"$DIRECT_ENGINE_AVAILABLE\" = 0",
        ],
        cwd=tmp_path,
        env={**os.environ, "PATH": "/usr/bin:/bin", "OPENTULPA_CONTAINER_CLI": ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (
        "production host readiness will fail unless a sandbox worker is wired" in result.stderr
        or "sandbox worker will use bundled restricted process isolation" in result.stderr
    )


def test_direct_start_reports_restricted_process_sandbox_worker_when_available(
    tmp_path: Path,
) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; "
            "process_repository_sandbox_available() { return 0; }; "
            "warn_without_container_engine",
        ],
        cwd=tmp_path,
        env={**os.environ, "SANDBOX_PROVIDER": "auto"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "sandbox worker will use bundled restricted process isolation" in result.stderr
    assert "shell commands will be unavailable" not in result.stderr


def test_direct_start_recognizes_railway_hosted_sandbox(tmp_path: Path) -> None:
    script = tmp_path / "start.sh"
    script.write_text((REPO_ROOT / "start.sh").read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; unset OPENTULPA_CONTAINER_CLI; "
            "configure_container_engine server; "
            "test \"$DIRECT_ENGINE_AVAILABLE\" = 0",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": "/usr/bin:/bin",
            "OPENTULPA_CONTAINER_CLI": "",
            "SANDBOX_PROVIDER": "railway",
            "RAILWAY_TOKEN": "project-token",
            "OPENTULPA_SANDBOX_RAILWAY_ENVIRONMENT_ID": "environment-id",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "using Railway-hosted sandbox VMs for tenant commands" in result.stdout
    assert "shell commands will be unavailable" not in result.stderr


def test_start_script_install_managed_builds_trusted_runtime_and_evaluator_images() -> None:
    result = _run_start(
        "install",
        "managed",
        "--dry-run",
        env=EMPTY_REQUIRED_ENV,
    )

    assert result.returncode == 0
    assert "docker build --tag opentulpa-runtime-base:0.1.0 --file Dockerfile ." in result.stdout
    assert "--tag opentulpa-evolution:0.1.0" in result.stdout
    assert "--file docker/evolution.Dockerfile ." in result.stdout
    assert (
        "--tag opentulpa-tenant-sandbox:0.1.0 "
        "--file docker/tenant-sandbox.Dockerfile ."
    ) in result.stdout
    assert "uv run --no-sync opentulpa-host" not in result.stdout


def test_start_script_run_managed_uses_live_source_host_without_rebuilding() -> None:
    result = _run_start(
        "run",
        "managed",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_RECOVERY_TOKEN": "r" * 32,
            "OPENTULPA_INGRESS_TOKEN": "i" * 32,
            "OPENTULPA_RELEASE_EGRESS_NETWORK": "restricted-egress",
            "OPENTULPA_RELEASE_BASE_IMAGE": "opentulpa-runtime-base:0.1.0",
            "OPENTULPA_OWNER_TOKEN": "owner-token",
        },
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for managed" not in result.stdout
    assert "running live-source host with managed OCI workers" in result.stdout
    assert "uv run --no-sync opentulpa-host" in result.stdout
    assert "docker build" not in result.stdout


def test_start_script_browser_flag_installs_only_cloud_adapter_dependencies() -> None:
    result = _run_start("install", "--dry-run", "--browser-use")

    assert result.returncode == 0
    assert "uv sync --frozen --no-dev --extra browser" in result.stdout
    assert "--extra bundled" not in result.stdout
    assert "playwright install chromium" not in result.stdout


def test_start_script_retains_selected_optional_adapters() -> None:
    result = _run_start(
        "install",
        "--dry-run",
        env={"OPENTULPA_EXTRAS": "integrations,documents"},
    )

    assert result.returncode == 0
    assert "uv sync --frozen --no-dev --extra integrations --extra documents" in result.stdout


def test_start_script_bakes_selected_adapters_into_managed_runtime() -> None:
    result = _run_start(
        "install",
        "managed",
        "--dry-run",
        env={"OPENTULPA_EXTRAS": "bundled"},
    )

    assert result.returncode == 0
    assert (
        "docker build --build-arg OPENTULPA_EXTRAS=bundled "
        "--tag opentulpa-runtime-base:0.1.0 --file Dockerfile ."
    ) in result.stdout


def test_start_script_rejects_unknown_optional_adapter_bundle() -> None:
    result = _run_start(
        "install",
        "--dry-run",
        env={"OPENTULPA_EXTRAS": "browser,arbitrary"},
    )

    assert result.returncode != 0
    assert "unsupported OPENTULPA_EXTRAS value: arbitrary" in result.stderr


def test_start_script_deprecated_app_alias_maps_to_server_mode() -> None:
    result = _run_start(
        "--dry-run",
        "--run-only",
        "--app",
    )

    assert result.returncode == 0
    assert "--app is deprecated" in result.stderr
    assert "[start] running server mode." in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout


def test_start_script_deprecated_manager_alias_maps_to_local_mode() -> None:
    result = _run_start(
        "--dry-run",
        "--run-only",
        "--manager",
    )

    assert result.returncode == 0
    assert "--manager is deprecated" in result.stderr
    assert "[start] running local Telegram mode." in result.stdout
    assert "uv run --no-sync python scripts/manager.py" in result.stdout


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
    assert "[start] uv sync --frozen --no-dev" in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout


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
    assert "business_knowledge_oracle_model" in result.stdout


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
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ \"$1\" == \"info\" ]]; then printf '%s\\n' '[\"name=rootless\"]'; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    env = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "OPENAI_COMPATIBLE_API_KEY": "test-key",
        "OPENAI_COMPATIBLE_BASE_URL": "https://provider.example/v1",
        "TELEGRAM_BOT_TOKEN": "test-token",
        "TELEGRAM_WEBHOOK_SECRET": "test-secret",
        "PUBLIC_BASE_URL": "https://app.example",
        "OPENTULPA_DATA_ROOT": str(tmp_path / "data"),
        "OPENTULPA_OWNER_TOKEN": "test-owner-token",
        "TELEGRAM_ALLOWED_USERNAMES": "owner",
        "TELEGRAM_ALLOWED_USER_IDS": "",
        "COMPOSIO_API_KEY": "",
    }

    result = _run_start("doctor", "server", env=env)

    assert result.returncode == 0
    assert "https://provider.example/v1/models did not list configured model(s)" in result.stdout
    assert "llm_model=moonshotai/kimi-k3" in result.stdout
    assert (
        "business_knowledge_oracle_model=google/gemini-3.1-flash-lite-preview"
        in result.stdout
    )


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
        "OPENTULPA_OWNER_TOKEN": "test-owner-token",
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
    assert "uv run --no-sync python -m opentulpa" in result.stdout


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
