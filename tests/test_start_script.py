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
    "OPENTULPA_TENANT_ID": "",
    "OPENTULPA_TENANTS_ROOT": "",
    "OPENTULPA_TENANT_WEB_TOKEN": "",
    "OPENTULPA_TENANT_HOST": "",
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
    assert "local|server|managed|tenant|install|run|doctor" in result.stdout
    assert "tenant [TENANT_ID]" in result.stdout
    assert "--yes" in result.stdout
    assert "--no-install-uv" in result.stdout
    assert "--browser-use" in result.stdout
    assert "UV_PYTHON=3.12" in result.stdout
    assert "OPENTULPA_OPEN_BROWSER=auto|1|0" in result.stdout
    assert "OPENTULPA_RESTART_GRACE_SECONDS=15" in result.stdout


def test_tenant_server_command_is_one_isolated_entrypoint(tmp_path: Path) -> None:
    result = _run_start(
        "tenant",
        "acme",
        "--run-only",
        "--port",
        "8101",
        "--public-url",
        "https://acme.example.com",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_TENANTS_ROOT": str(tmp_path / "tenants"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert f"tenant acme: data={tmp_path / 'tenants' / 'acme'} port=8101" in result.stdout
    assert "required .env value(s) missing" not in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout


def test_tenant_server_accepts_identity_from_environment(tmp_path: Path) -> None:
    result = _run_start(
        "tenant",
        "--run-only",
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_TENANT_ID": "owner-two",
            "OPENTULPA_TENANTS_ROOT": str(tmp_path / "tenants"),
        },
    )

    assert result.returncode == 0, result.stderr
    assert "tenant owner-two:" in result.stdout


def test_tenant_server_generates_private_reusable_identity_and_token(tmp_path: Path) -> None:
    data_root = tmp_path / "tenant-data"
    command = (
        "source ./start.sh; "
        "TENANT_ID=acme; TENANT_PORT=8101; TENANT_DATA_ROOT=\"$DATA_ROOT\"; DRY_RUN=0; "
        "unset OPENTULPA_TENANT_WEB_TOKEN; configure_tenant_server; "
        "test \"$OPENTULPA_OWNER_CUSTOMER_ID\" = acme; "
        "test \"$OPENTULPA_DATA_ROOT\" = \"$DATA_ROOT\"; "
        "test \"$PORT\" = 8101; test \"$HOST\" = 0.0.0.0"
    )
    first = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env={**os.environ, "DATA_ROOT": str(data_root), "OPENTULPA_TENANT_HOST": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr
    first_token = (data_root / "bootstrap" / "owner-web.token").read_text(
        encoding="utf-8"
    )
    second = subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env={**os.environ, "DATA_ROOT": str(data_root), "OPENTULPA_TENANT_HOST": ""},
        capture_output=True,
        text=True,
        check=False,
    )

    assert second.returncode == 0, second.stderr
    assert (data_root / "bootstrap" / "tenant-id").read_text(encoding="utf-8") == "acme\n"
    assert len(first_token.strip()) == 64
    assert (data_root / "bootstrap" / "owner-web.token").read_text(
        encoding="utf-8"
    ) == first_token
    assert first_token.strip() not in first.stdout
    assert (data_root / "bootstrap" / "owner-web.token").stat().st_mode & 0o777 == 0o600


def test_tenant_server_refuses_data_root_bound_to_another_tenant(tmp_path: Path) -> None:
    marker = tmp_path / "tenant-data" / "bootstrap" / "tenant-id"
    marker.parent.mkdir(parents=True)
    marker.write_text("first-owner\n", encoding="utf-8")

    result = subprocess.run(
        [
            "bash",
            "-c",
            "source ./start.sh; TENANT_ID=second-owner; "
            "TENANT_DATA_ROOT=\"$DATA_ROOT\"; DRY_RUN=0; configure_tenant_server",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "DATA_ROOT": str(tmp_path / "tenant-data")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "belongs to first-owner, not second-owner" in result.stderr


def test_container_and_railway_use_tenant_entrypoint() -> None:
    assert 'CMD ["./start.sh", "tenant", "--run-only"]' in (
        REPO_ROOT / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert 'startCommand = "./start.sh tenant --run-only"' in (
        REPO_ROOT / "railway.toml"
    ).read_text(encoding="utf-8")
    compose = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'command: ["./start.sh", "tenant", "--run-only"]' in compose
    assert "OPENTULPA_TENANTS_ROOT: /app/opentulpa_data" in compose


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
            "unset OPENTULPA_WEB_TOKEN PUBLIC_BASE_URL RAILWAY_PUBLIC_DOMAIN; "
            "configure_local_server_defaults server; first=$OPENTULPA_WEB_TOKEN; "
            "unset OPENTULPA_WEB_TOKEN; configure_local_server_defaults server; "
            "test \"$first\" = \"$OPENTULPA_WEB_TOKEN\"",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "OPENTULPA_DATA_ROOT": str(data_root),
            "OPENTULPA_WEB_TOKEN": "",
            "PUBLIC_BASE_URL": "",
            "RAILWAY_PUBLIC_DOMAIN": "",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    token_path = data_root / "bootstrap" / "owner-web.token"
    assert len(token_path.read_text(encoding="utf-8").strip()) == 64
    assert token_path.stat().st_mode & 0o777 == 0o600


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
    assert "server Telegram disabled; web/API startup does not require Telegram env." in result.stdout
    assert "uv run --no-sync python -m opentulpa" in result.stdout


def test_start_script_public_server_does_not_generate_deployment_credentials() -> None:
    result = _run_start(
        "server",
        "--dry-run",
        env={**EMPTY_REQUIRED_ENV, "PUBLIC_BASE_URL": "https://agent.example"},
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for server:" in result.stdout
    assert "OPENTULPA_WEB_TOKEN" in result.stdout
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
    assert "uv run --no-sync python -m opentulpa" in result.stdout


def test_start_script_doctor_server_web_only_requires_web_token() -> None:
    result = _run_start("doctor", "server", env=EMPTY_REQUIRED_ENV)

    assert result.returncode == 1
    assert "server Telegram disabled; skipping Telegram token and allowlist checks" in result.stdout
    assert "server Telegram disabled; skipping webhook URL/secret checks" in result.stdout
    assert "fail: OPENTULPA_WEB_TOKEN is set" in result.stdout
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
    assert "required .env value(s) missing for server: OPENTULPA_WEB_TOKEN" in result.stdout


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
    assert "[start] uv sync --no-dev" in result.stdout
    assert "--extra browser" not in result.stdout
    assert "playwright install chromium" not in result.stdout


def test_start_script_defaults_to_lean_core_dependencies() -> None:
    result = _run_start("install", "--dry-run")

    assert result.returncode == 0
    assert "[start] uv sync --no-dev" in result.stdout
    assert "--extra bundled" not in result.stdout
    assert "--extra browser" not in result.stdout
    assert (
        "docker build --tag opentulpa-tenant-sandbox:0.1.0 "
        "--file docker/tenant-sandbox.Dockerfile ."
    ) in result.stdout


def test_start_script_defaults_to_web_server() -> None:
    result = _run_start(
        "--dry-run",
        env={
            **EMPTY_REQUIRED_ENV,
            "OPENAI_COMPATIBLE_API_KEY": "test-key",
            "OPENTULPA_DATA_ROOT": "/tmp/opentulpa-test-data",
            "OPENTULPA_WEB_TOKEN": "test-web-token",
        },
    )

    assert result.returncode == 0
    assert "[start] running server mode." in result.stdout
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


def test_direct_start_without_container_engine_keeps_chat_available(tmp_path: Path) -> None:
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
    assert "chat will start but sandbox shell commands will be unavailable" in result.stderr


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
    assert "uv run --no-sync opentulpa-bootstrap" not in result.stdout


def test_start_script_run_managed_uses_immutable_bootstrap_without_rebuilding() -> None:
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
            "OPENTULPA_WEB_TOKEN": "web-token",
        },
    )

    assert result.returncode == 0
    assert "required .env value(s) missing for managed" not in result.stdout
    assert "running immutable bootstrap with managed OCI releases" in result.stdout
    assert "uv run --no-sync opentulpa-bootstrap" in result.stdout
    assert "docker build" not in result.stdout


def test_start_script_browser_flag_installs_only_cloud_adapter_dependencies() -> None:
    result = _run_start("install", "--dry-run", "--browser-use")

    assert result.returncode == 0
    assert "uv sync --no-dev --extra browser" in result.stdout
    assert "--extra bundled" not in result.stdout
    assert "playwright install chromium" not in result.stdout


def test_start_script_retains_selected_optional_adapters() -> None:
    result = _run_start(
        "install",
        "--dry-run",
        env={"OPENTULPA_EXTRAS": "integrations,documents"},
    )

    assert result.returncode == 0
    assert "uv sync --no-dev --extra integrations --extra documents" in result.stdout


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
    assert "[start] uv sync --no-dev" in result.stdout
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
        "OPENTULPA_WEB_TOKEN": "test-web-token",
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
        "OPENTULPA_WEB_TOKEN": "test-web-token",
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
