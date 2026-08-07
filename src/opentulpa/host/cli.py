"""One-command host launcher and small remote operations client."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
from contextlib import suppress
from importlib import resources
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from pydantic import SecretStr

import opentulpa
from opentulpa.client.config import (
    ClientConfigError,
    Connection,
    clear_connection,
    config_path,
    load_connection,
    normalize_url,
    save_connection,
    update_connection,
)
from opentulpa.client.local_server import (
    LocalServerError,
    ensure_local_server,
    is_loopback_url,
    restart_remembered_local_server,
)
from opentulpa.core.config import get_settings
from opentulpa.evolution.generation_store import GenerationStore
from opentulpa.evolution.models import Release
from opentulpa.host.app import create_host_app
from opentulpa.host.evolution import prepare_live_source_repository
from opentulpa.host.evolution_composition import build_host_evolution_runtime
from opentulpa.host.models import HostConfigInput
from opentulpa.host.paths import HostPaths
from opentulpa.host.runtime import RuntimeGenerationSpec, RuntimeLiveSourceSpec, RuntimeSupervisor
from opentulpa.host.service import HostService
from opentulpa.host.store import HostStore
from opentulpa.sandbox.supervisor import SandboxWorkerSupervisor
from opentulpa.secrets.cipher import AesGcmHostKeyCipher
from opentulpa.secrets.host_key import load_or_create_host_cipher

HOST_GENERATION_DEPENDENCIES = {
    "source_seed": "/opt/opentulpa-source",
    "wheelhouse": "/opt/opentulpa-wheelhouse",
}


class HostInstallRequiredError(RuntimeError):
    """An installed-wheel host has no trusted inputs for its first generation."""

    status = "install_required"


def _private_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise RuntimeError("host credential directory cannot be a symlink")
    if not path.exists():
        value = secrets.token_urlsafe(48)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{value}\n")
            stream.flush()
            os.fsync(stream.fileno())
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError("host credential must be a private regular file")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 32:
        raise RuntimeError("host credential is invalid")
    return value


def _private_pairing_code(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise RuntimeError("host credential directory cannot be a symlink")
    if not path.exists():
        alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
        compact = "".join(secrets.choice(alphabet) for _ in range(15))
        value = "-".join(compact[index : index + 5] for index in range(0, 15, 5))
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{value}\n")
            stream.flush()
            os.fsync(stream.fileno())
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError("host credential must be a private regular file")
    value = path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{16,500}", value) is None:
        raise RuntimeError("host pairing code is invalid")
    return value


def build_host_application() -> tuple[Any, str, str, Path]:
    probation_seconds, probation_probe_interval_seconds = _runtime_probation_settings()
    paths = HostPaths.from_environment()
    paths.provision()
    project_root = _host_application_root()
    settings = get_settings()
    generation_store = GenerationStore(
        paths.generations_root,
        control_root=paths.control_root / "evolution" / "generation-store",
        quarantine_root=paths.control_root / "evolution" / "generation-quarantine",
    )
    generation_store.cleanup_incomplete()
    recovered_generation = _require_installed_host_dependencies(
        project_root,
        settings=settings,
        paths=paths,
        generation_store=generation_store,
    )
    setup_token = _private_pairing_code(paths.control_root / "host-setup.token")
    evolution_token = _private_token(paths.control_root / "evolution.token")
    store = HostStore(
        paths.control_root / "host.db",
        cipher=_load_or_create_control_cipher(paths),
    )
    store.configure_setup_token(setup_token)
    host = str(os.environ.get("HOST") or "127.0.0.1").strip()
    owner_token = str(
        os.environ.get("OPENTULPA_OWNER_TOKEN") or os.environ.get("OPENTULPA_WEB_TOKEN") or ""
    ).strip()
    if not owner_token and host in {"127.0.0.1", "localhost", "::1"}:
        owner_token = _private_token(paths.control_root / "owner.token")
    if owner_token and not store.claimed:
        store.claim(setup_token=setup_token, owner_token=owner_token)

    api_key = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY") or "").strip()
    telegram_token, telegram_id = _telegram_bootstrap_from_environment()
    bootstrap = None
    active = store.active()
    if api_key and active is None:
        bootstrap = HostConfigInput(
            api_key=SecretStr(api_key),
            base_url=settings.openai_compatible_base_url,
            model=settings.llm_model,
            telegram_bot_token=SecretStr(telegram_token) if telegram_token else None,
            telegram_user_id=telegram_id if telegram_token else None,
        )
    elif active is not None and active.telegram_bot_token is None and telegram_token and telegram_id:
        bootstrap = HostConfigInput(
            expected_revision=active.revision,
            base_url=active.base_url,
            model=active.model,
            telegram_bot_token=SecretStr(telegram_token),
            telegram_user_id=telegram_id,
        )
    runtime = RuntimeSupervisor(
        project_root=project_root,
        data_root=paths.product_root,
        application_root=paths.product_root,
        generation_store=generation_store,
        generation_spec=recovered_generation,
        control_path=paths.runtime_control_path,
        child_uid=paths.runtime_uid,
        child_gid=paths.runtime_gid,
        probation_seconds=probation_seconds,
        probation_probe_interval_seconds=probation_probe_interval_seconds,
    )
    if recovered_generation is None:
        live_source = _load_current_live_source(project_root)
        if live_source is not None:
            runtime.set_live_source(live_source)
    sandbox = SandboxWorkerSupervisor(
        project_root=project_root,
        data_root=paths.control_root / "sandbox-host",
        settings=settings,
    )
    runtime.configure_sandbox_worker(
        base_url=sandbox.config.base_url,
        token=sandbox.config.token,
    )
    port = int(os.environ.get("PORT") or 8000)
    evolution = (
        build_host_evolution_runtime(
            runtime=runtime,
            data_root=paths.data_root,
            control_root=paths.control_root,
            product_root=paths.product_root,
            generation_store=generation_store,
            settings=settings,
        )
        if recovered_generation is None
        else None
    )
    if evolution is not None:
        asyncio.run(evolution.prepare())
    if evolution is not None:
        runtime.configure_evolution_control(
            base_url=(f"http://127.0.0.1:{port}/bootstrap/internal/v1/evolution"),
            token=evolution_token,
        )
    service = HostService(
        store=store,
        runtime=runtime,
        bootstrap_config=bootstrap,
        evolution=evolution,
    )
    app = create_host_app(
        store=store,
        service=service,
        local_owner_enabled=host in {"127.0.0.1", "localhost", "::1"},
        setup_token=setup_token,
        evolution_service=evolution.service if evolution is not None else None,
        evolution_token=evolution_token if evolution is not None else None,
        sandbox_supervisor=sandbox,
    )
    return app, host, setup_token, paths.data_root


def _runtime_probation_settings() -> tuple[float, float]:
    return (
        _environment_duration("OPENTULPA_RUNTIME_PROBATION_SECONDS", default=30, allow_zero=True),
        _environment_duration(
            "OPENTULPA_RUNTIME_PROBATION_PROBE_INTERVAL_SECONDS",
            default=1,
            allow_zero=False,
        ),
    )


def _environment_duration(name: str, *, default: float, allow_zero: bool) -> float:
    raw = str(os.environ.get(name) or default).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        requirement = "nonnegative" if allow_zero else "positive"
        raise RuntimeError(f"{name} must be a finite {requirement} number")
    return value


def _host_application_root() -> Path:
    configured_source = str(os.environ.get("OPENTULPA_SOURCE_ROOT") or "").strip()
    if configured_source:
        source = Path(configured_source).expanduser().resolve()
        if not _is_source_checkout(source):
            raise RuntimeError("OPENTULPA_SOURCE_ROOT is not an OpenTulpa source checkout")
        return source

    inferred = Path(__file__).resolve().parents[3]
    if _is_source_checkout(inferred):
        return inferred

    configured_assets = str(os.environ.get("OPENTULPA_INSTALL_ASSETS_ROOT") or "").strip()
    if configured_assets:
        assets = Path(configured_assets).expanduser().resolve()
        if not (assets / "railway_sandbox_bridge" / "bridge.mjs").is_file():
            raise RuntimeError("OPENTULPA_INSTALL_ASSETS_ROOT has no OpenTulpa bridge assets")
        return assets

    package_root = Path(str(resources.files(opentulpa))).resolve()
    if not package_root.is_dir():
        raise RuntimeError("installed OpenTulpa package resources are unavailable")
    return package_root


def _is_source_checkout(root: Path) -> bool:
    return (
        root.is_dir()
        and (root / "pyproject.toml").is_file()
        and (root / "src" / "opentulpa" / "__init__.py").is_file()
    )


def _require_installed_host_dependencies(
    project_root: Path,
    *,
    settings: Any,
    paths: HostPaths,
    generation_store: GenerationStore,
) -> RuntimeGenerationSpec | None:
    if _is_source_checkout(project_root):
        return None
    configured_seed = str(os.environ.get("OPENTULPA_SOURCE_SEED_ROOT") or "").strip()
    configured_wheelhouse = str(os.environ.get("OPENTULPA_TRUSTED_WHEELHOUSE") or "").strip()
    seed = Path(configured_seed or HOST_GENERATION_DEPENDENCIES["source_seed"]).expanduser()
    wheelhouse = Path(
        configured_wheelhouse or HOST_GENERATION_DEPENDENCIES["wheelhouse"]
    ).expanduser()
    if settings.evolution_enabled and seed.is_dir() and wheelhouse.is_dir():
        return None
    recovered = _load_current_generation(paths, generation_store=generation_store)
    if recovered is not None:
        return recovered
    raise HostInstallRequiredError(
        "Installed OpenTulpa has no runnable source checkout or trusted generation inputs. "
        "Run the installer/recovery flow to provide the source seed and offline wheelhouse."
    )


def _load_current_generation(
    paths: HostPaths,
    *,
    generation_store: GenerationStore,
) -> RuntimeGenerationSpec | None:
    pointer = paths.control_root / "evolution" / "current_release.json"
    if not os.path.lexists(pointer):
        return None
    try:
        metadata = pointer.lstat()
        if (
            pointer.is_symlink()
            or not pointer.is_file()
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeError("current release pointer is unsafe")
        release = Release.model_validate_json(pointer.read_bytes())
        if release.metadata.get("artifact_kind") != "python_generation":
            raise RuntimeError("current release is not a Python generation")
        spec = RuntimeGenerationSpec.from_release_metadata(release.metadata)
        if release.artifact_digest != spec.manifest_digest:
            raise RuntimeError("current release manifest provenance is inconsistent")
        installed = generation_store.open(
            spec.generation_id,
            expected_manifest_digest=spec.manifest_digest,
            expected_state_contract_digest=spec.state_contract_digest,
            expected_evaluator_fingerprint=spec.evaluator_fingerprint,
            expected_install_profile=spec.install_profile,
            controller_protocol=spec.controller_protocol,
        )
        if installed.manifest.identity.source_commit != release.source_commit:
            raise RuntimeError("current generation source provenance is inconsistent")
    except Exception as exc:
        raise HostInstallRequiredError(
            "Installed OpenTulpa cannot verify its current generation. "
            "Run the installer/recovery flow."
        ) from exc
    return spec


def _load_current_live_source(project_root: Path) -> RuntimeLiveSourceSpec | None:
    if not (_is_source_checkout(project_root) and (project_root / ".git").exists()):
        return None
    repository = prepare_live_source_repository(project_root)
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostInstallRequiredError(
            "Installed OpenTulpa cannot verify its live source checkout."
        ) from exc
    return RuntimeLiveSourceSpec(source_commit=completed.stdout.strip())


def _load_or_create_control_cipher(paths: HostPaths) -> AesGcmHostKeyCipher:
    if paths.control_root == paths.data_root / "bootstrap":
        return load_or_create_host_cipher(paths.data_root)
    configured = str(os.environ.get("OPENTULPA_SECRET_VAULT_KEY") or "").strip()
    if configured:
        return AesGcmHostKeyCipher.from_base64(configured)
    key_path = paths.control_root / "secret-vault.key"
    if not key_path.exists():
        descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(os.urandom(32))
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            key_path.unlink(missing_ok=True)
            raise
    if key_path.is_symlink() or not key_path.is_file() or key_path.stat().st_mode & 0o077:
        raise RuntimeError("host secret key must be a private regular file")
    key = key_path.read_bytes()
    if len(key) != 32:
        raise RuntimeError("host secret key must contain exactly 32 bytes")
    return AesGcmHostKeyCipher(key)


def _telegram_bootstrap_from_environment() -> tuple[str, int | None]:
    token = str(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    ids = str(os.environ.get("TELEGRAM_ALLOWED_USER_IDS") or "").strip()
    if not ids:
        return token, None
    first = ids.split(",", 1)[0].strip()
    if first.isdigit() and int(first) > 0:
        return token, int(first)
    return token, None


def serve(*, public_url: str | None = None) -> None:
    app, host, setup_token, data_root = build_host_application()
    port = int(os.environ.get("PORT") or 8000)
    origin = _server_origin(host=host, port=port, public_url=public_url)
    print("OpenTulpa is starting.\n")
    print(f"Admin:   {origin}/_host")
    print(f"Connect: opentulpa connect {origin}")
    if host not in {"127.0.0.1", "localhost", "::1"} and not app.state.host_store.claimed:
        print(f"\nPairing code: {setup_token}")
    print(f"\nData: {data_root}", flush=True)
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        ws="none",
        timeout_graceful_shutdown=15,
    )


def _server_origin(*, host: str, port: int, public_url: str | None) -> str:
    candidate = str(public_url or os.environ.get("PUBLIC_BASE_URL") or "").strip()
    if not candidate:
        railway_domain = str(os.environ.get("RAILWAY_PUBLIC_DOMAIN") or "").strip()
        if railway_domain:
            candidate = f"https://{railway_domain}"
    if candidate:
        return normalize_url(candidate)
    if host in {"127.0.0.1", "localhost", "::1"}:
        return f"http://127.0.0.1:{port}"
    hostname = socket.getfqdn().strip() or "SERVER"
    return f"http://{hostname}:{port}"


def _remote(args: argparse.Namespace) -> int:
    if args.command == "connect":
        try:
            url = normalize_url(args.url)
        except ClientConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        supplied = str(args.token or args.pairing_code or "").strip()
        if not supplied and sys.stdin.isatty():
            supplied = getpass.getpass("Owner token or pairing code (blank for local): ").strip()
        token = _connect_credential(url, supplied)
        if token is None:
            return 1
        try:
            connection = save_connection(url, token)
        except ClientConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(
            f"Connected to {connection.url}; credential stored in {connection.credential_storage}."
        )
        if args.no_tui:
            return 0
        _configure_runtime(connection)
        _launch_tui(connection)
        return 0
    try:
        connection = load_connection()
    except ClientConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    headers = _owner_headers(connection.token)
    with httpx.Client(timeout=30, trust_env=False) as client:
        if args.command == "status":
            response = client.get(f"{connection.url}/_host/api/status", headers=headers)
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
            return 0
        if args.command == "logs":
            if args.follow:
                with client.stream(
                    "GET", f"{connection.url}/_host/api/logs/stream", headers=headers
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            entry = json.loads(line[6:])
                            print(f"{entry['stream']:<6} {entry['text']}", flush=True)
                return 0
            response = client.get(f"{connection.url}/_host/api/logs", headers=headers)
            response.raise_for_status()
            for entry in response.json()["logs"]:
                print(f"{entry['stream']:<6} {entry['text']}")
            return 0
        if args.command == "disconnect":
            try:
                clear_connection()
            except ClientConfigError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            print("Forgot the remembered OpenTulpa server and owner credential.")
            return 0
    return 1


def _connect_credential(url: str, supplied: str) -> str | None:
    headers = _owner_headers(supplied)
    try:
        with httpx.Client(timeout=20, trust_env=False) as client:
            response = client.get(f"{url}/_host/api/status", headers=headers)
            response.raise_for_status()
            payload = response.json()
            if bool(payload.get("authenticated")):
                return supplied
            if not bool(payload.get("claimed")) and supplied:
                claim = client.post(
                    f"{url}/_host/api/claim",
                    json={"setup_token": supplied},
                )
                if claim.is_success:
                    owner_token = str(claim.json().get("owner_token") or "").strip()
                    if owner_token:
                        return owner_token
    except (httpx.HTTPError, ValueError) as exc:
        print(f"Could not connect to OpenTulpa: {exc}", file=sys.stderr)
        return None
    print("OpenTulpa rejected the owner token or pairing code.", file=sys.stderr)
    return None


def _owner_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def _interactive_connection() -> Connection:
    try:
        connection = load_connection()
    except ClientConfigError as exc:
        if config_path().exists():
            raise SystemExit(str(exc)) from exc
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise SystemExit("Run `opentulpa connect SERVER_URL` first.") from None
        return _first_connection()
    if is_loopback_url(connection.url):
        try:
            url = ensure_local_server(preferred_url=connection.url)
        except LocalServerError as exc:
            raise SystemExit(str(exc)) from exc
        if url != connection.url:
            connection = save_connection(
                url,
                connection.token,
                thread_id=connection.thread_id,
                last_run_id=connection.last_run_id,
                last_sequence=connection.last_sequence,
            )
        _configure_runtime(connection)
    else:
        try:
            status = _get_host_status(connection)
        except (httpx.HTTPError, ValueError):
            return connection
        if not bool(status.get("configured")):
            _configure_runtime(connection, status=status)
    return connection


def _first_connection() -> Connection:
    print("\nOPENTULPA\n")
    print("  1  Run here")
    print("  2  Connect remotely\n")
    choice = input("Choose [1]: ").strip().casefold()
    if choice in {"", "1", "local", "run", "run here"}:
        print("\nStarting your private OpenTulpa server...")
        try:
            url = ensure_local_server()
        except LocalServerError as exc:
            raise SystemExit(str(exc)) from exc
        token = _connect_credential(url, "")
        if token is None:
            raise SystemExit("The local OpenTulpa server rejected its owner connection.")
        connection = save_connection(url, token)
        _configure_runtime(connection)
        return connection
    if choice not in {"2", "remote", "connect", "connect remotely"}:
        raise SystemExit("Choose 1 to run here or 2 to connect remotely.")
    return _prompt_remote_connection()


def _prompt_remote_connection() -> Connection:
    print("\nConnect to an OpenTulpa server.")
    url = input("Server URL: ").strip()
    credential = getpass.getpass("Owner token or pairing code: ").strip()
    try:
        normalized = normalize_url(url)
    except ClientConfigError as exc:
        raise SystemExit(str(exc)) from exc
    token = _connect_credential(normalized, credential)
    if token is None:
        raise SystemExit(1)
    try:
        connection = save_connection(normalized, token)
    except ClientConfigError as exc:
        raise SystemExit(str(exc)) from exc
    _configure_runtime(connection)
    return connection


def _get_host_status(connection: Connection) -> dict[str, Any]:
    headers = _owner_headers(connection.token)
    with httpx.Client(timeout=10, trust_env=False) as client:
        response = client.get(f"{connection.url}/_host/api/status", headers=headers)
        response.raise_for_status()
        payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("OpenTulpa returned an invalid host status.")
    return payload


def _configure_runtime(
    connection: Connection,
    *,
    status: dict[str, Any] | None = None,
) -> None:
    try:
        payload = status if status is not None else _get_host_status(connection)
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"Could not inspect the OpenTulpa server: {exc}") from exc
    if bool(payload.get("configured")):
        return
    headers = _owner_headers(connection.token)

    print("\nConnect a model. The key is sent only to the encrypted OpenTulpa host.\n")
    api_key = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY") or "").strip()
    if not api_key:
        if not sys.stdin.isatty():
            raise SystemExit("Set OPENAI_COMPATIBLE_API_KEY or run `opentulpa` interactively.")
        api_key = getpass.getpass("Model API key: ").strip()
    if not api_key:
        raise SystemExit("A model API key is required.")
    default_base_url = str(
        os.environ.get("OPENAI_COMPATIBLE_BASE_URL") or "https://openrouter.ai/api/v1"
    ).strip()
    default_model = str(os.environ.get("LLM_MODEL") or "moonshotai/kimi-k3").strip()
    if sys.stdin.isatty():
        base_url = input(f"Endpoint [{default_base_url}]: ").strip() or default_base_url
        model = input(f"Model [{default_model}]: ").strip() or default_model
    else:
        base_url = default_base_url
        model = default_model
    try:
        with httpx.Client(timeout=120, trust_env=False) as client:
            response = client.put(
                f"{connection.url}/_host/api/config",
                headers=headers,
                json={"api_key": api_key, "base_url": base_url, "model": model},
            )
    except httpx.HTTPError as exc:
        raise SystemExit("The OpenTulpa server disconnected during setup.") from exc
    if not response.is_success:
        detail = "OpenTulpa could not activate this model configuration."
        try:
            value = response.json().get("detail")
            if isinstance(value, str) and value.strip():
                detail = value.strip()
        except ValueError:
            pass
        raise SystemExit(detail)
    print("\nOpenTulpa is ready.\n")


def _server_command(args: argparse.Namespace) -> None:
    if not 1 <= args.port <= 65535:
        raise SystemExit("Server port must be between 1 and 65535.")
    if not str(args.host or "").strip() or any(character.isspace() for character in str(args.host)):
        raise SystemExit("Server host is invalid.")
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    if args.public_url:
        try:
            os.environ["PUBLIC_BASE_URL"] = normalize_url(args.public_url)
        except ClientConfigError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        serve(public_url=args.public_url)
    except HostInstallRequiredError as exc:
        raise SystemExit(f"{exc.status}: {exc}") from exc


def _open_tui() -> None:
    try:
        connection = _interactive_connection()
        _launch_tui(connection)
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        return


def _launch_tui(connection: Connection) -> None:
    _migrate_legacy_session_names(connection)
    binary = _ensure_tui_binary()
    connection_read, connection_write = os.pipe()
    state_read, state_write = os.pipe()
    os.set_inheritable(connection_read, True)
    os.set_inheritable(state_write, True)
    environment = os.environ.copy()
    environment["OPENTULPA_CONNECTION_FD"] = str(connection_read)
    environment["OPENTULPA_STATE_FD"] = str(state_write)
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [str(binary)],
            env=environment,
            pass_fds=(connection_read, state_write),
        )
        os.close(connection_read)
        connection_read = -1
        os.close(state_write)
        state_write = -1
        payload = json.dumps(
            {
                "url": connection.url,
                "token": connection.token,
                "thread_id": connection.thread_id,
                "credential_storage": connection.credential_storage,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        with os.fdopen(connection_write, "wb") as stream:
            connection_write = -1
            stream.write(payload)
        process.wait()
        with os.fdopen(state_read, "rb") as stream:
            state_read = -1
            raw_state = stream.read(16_384)
        if raw_state:
            state = json.loads(raw_state)
            thread_id = str(state.get("thread_id") or "").strip()
            if thread_id and thread_id != connection.thread_id:
                update_connection(
                    connection,
                    thread_id=thread_id,
                    last_run_id=None,
                    last_sequence=0,
                )
        if process.returncode:
            raise SystemExit(f"OpenTulpa terminal client exited with status {process.returncode}.")
    except KeyboardInterrupt:
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not launch the OpenTulpa terminal client: {exc}") from exc
    finally:
        for descriptor in (connection_read, connection_write, state_read, state_write):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)


def _ensure_tui_binary() -> Path:
    configured = str(os.environ.get("OPENTULPA_TUI_BINARY") or "").strip()
    if configured:
        binary = Path(configured).expanduser().resolve()
        if binary.is_file() and os.access(binary, os.X_OK) and _tui_protocol(binary) == "2":
            return binary
        raise SystemExit(
            "The configured OpenTulpa terminal client is incompatible. Install the matching "
            "OpenTulpa release."
        )
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64", "AMD64": "x64"}.get(
        platform.machine()
    )
    if system is None or machine is None:
        raise SystemExit("The OpenTulpa terminal client supports macOS and Linux on arm64 or x64.")
    target_name = f"opentulpa-tui-{system}-{machine}"
    project_root = Path(__file__).resolve().parents[3]
    client_root = project_root / "clients" / "tui"
    binary = client_root / "dist" / target_name
    manifest_path = client_root / "dist" / "manifest.json"
    source_digest = _tui_source_digest(client_root)
    if binary.is_file() and os.access(binary, os.X_OK) and source_digest:
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            manifest = {}
        if (
            manifest.get("protocol_version") == 2
            and manifest.get("source_digest") == source_digest
            and _tui_protocol(binary) == "2"
        ):
            return binary
    if not source_digest:
        packaged = _packaged_tui_binary(target_name)
        if packaged is not None:
            return packaged
        for installed_name in ("opentulpa-tui", target_name):
            installed = shutil.which(installed_name)
            if installed:
                installed_binary = Path(installed).resolve()
                if _tui_protocol(installed_binary) != "2":
                    raise SystemExit(
                        "The installed OpenTulpa terminal client is older than this server "
                        "protocol. Upgrade OpenTulpa and try again."
                    )
                return installed_binary
    data_home = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    bun_candidates = [
        data_home / "opentulpa" / "bun" / "bin" / "bun",
        Path(shutil.which("bun") or ""),
    ]
    bun = next(
        (
            str(candidate)
            for candidate in bun_candidates
            if candidate.is_file()
            and os.access(candidate, os.X_OK)
            and subprocess.run(
                [str(candidate), "--version"],
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip()
            == "1.3.14"
        ),
        None,
    )
    if bun is None or not (client_root / "package.json").is_file():
        raise SystemExit(
            "The native terminal client is missing. Run ./install.sh from an OpenTulpa "
            "source checkout to install its pinned build tool, or install a release build "
            "for this platform."
        )
    try:
        subprocess.run(
            [bun, "install", "--frozen-lockfile"],
            cwd=client_root,
            check=True,
        )
        subprocess.run([bun, "run", "build"], cwd=client_root, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit("The native OpenTulpa terminal client could not be built.") from exc
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise SystemExit("The native OpenTulpa terminal client build produced no executable.")
    if _tui_protocol(binary) != "2":
        raise SystemExit("The native OpenTulpa terminal client build is incompatible.")
    return binary


def _packaged_tui_binary(target_name: str) -> Path | None:
    configured_assets = str(os.environ.get("OPENTULPA_INSTALL_ASSETS_ROOT") or "").strip()
    candidates: list[Path] = []
    if configured_assets:
        root = Path(configured_assets).expanduser().resolve()
        candidates.extend((root / "bin" / target_name, root / "tui" / target_name))
    package_root = Path(str(resources.files(opentulpa))).resolve()
    candidates.extend((package_root / "bin" / target_name, package_root / "tui" / target_name))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            if _tui_protocol(candidate) != "2":
                raise SystemExit(
                    "The packaged OpenTulpa terminal client is incompatible with this release."
                )
            return candidate
    return None


def _tui_protocol(binary: Path) -> str:
    try:
        result = subprocess.run(
            [str(binary), "--protocol-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _tui_source_digest(client_root: Path) -> str:
    paths = [
        client_root / "build.ts",
        client_root / "bun.lock",
        client_root / "package.json",
        *(path for path in (client_root / "src").rglob("*") if path.is_file()),
    ]
    if any(not path.is_file() for path in paths):
        return ""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(client_root).as_posix()):
        relative = path.relative_to(client_root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _migrate_legacy_session_names(connection: Connection) -> None:
    path = config_path().with_name("sessions.json")
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        servers = payload.get("servers", {}) if isinstance(payload, dict) else {}
        server = servers.get(hashlib.sha256(connection.url.encode("utf-8")).hexdigest(), {})
        sessions = server.get("sessions", []) if isinstance(server, dict) else []
        names = {
            str(item.get("thread_id")): str(item.get("name"))
            for item in sessions
            if isinstance(item, dict)
            and str(item.get("thread_id") or "").strip()
            and 0 < len(str(item.get("name") or "").strip()) <= 120
        }
        if not names:
            return
        headers = _owner_headers(connection.token)
        with httpx.Client(timeout=10, trust_env=False, headers=headers) as client:
            response = client.get(f"{connection.url}/v2/agent/threads?limit=100")
            if not response.is_success:
                return
            for thread in response.json().get("threads", []):
                thread_id = str(thread.get("thread_id") or "")
                title = names.get(thread_id)
                if title and title != str(thread.get("title") or ""):
                    client.patch(
                        f"{connection.url}/v2/agent/threads/{thread_id}",
                        json={"title": title},
                    )
    except (OSError, ValueError, TypeError, httpx.HTTPError):
        return


def _controller_install_root() -> Path:
    configured = str(os.environ.get("OPENTULPA_INSTALL_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    data_home = str(os.environ.get("XDG_DATA_HOME") or "").strip()
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return (base / "opentulpa" / "install").resolve()


def _private_install_file(path: Path) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"OpenTulpa installation metadata is missing: {path}") from exc
    if (
        path.is_symlink()
        or not path.is_file()
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise RuntimeError(f"OpenTulpa installation file is not private and trusted: {path}")
    return path.read_bytes()


def _update_command(args: argparse.Namespace) -> int:
    install_root = _controller_install_root()
    controller_root = install_root / "controller"
    try:
        raw_metadata = _private_install_file(controller_root / "install.json")
        metadata = json.loads(raw_metadata)
        canonical = (
            json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        if raw_metadata != canonical or int(metadata.get("format_version", 0)) != 1:
            raise RuntimeError("OpenTulpa installation metadata is not canonical or supported")
        source_root = Path(str(metadata["source_root"])).expanduser().resolve()
        repository = str(metadata["repository"])
        ref = str(metadata["ref"])
        managed_source = bool(metadata["managed_source"])
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Cannot update OpenTulpa: {exc}", file=sys.stderr)
        return 1

    installer = controller_root / "installer.sh"
    try:
        _private_install_file(installer)
    except RuntimeError as exc:
        print(f"Cannot update OpenTulpa: {exc}", file=sys.stderr)
        return 1
    if not os.access(installer, os.X_OK):
        print(f"Cannot update OpenTulpa: installer is not executable: {installer}", file=sys.stderr)
        return 1

    requested_source = str(getattr(args, "source", None) or "").strip()
    fetch = bool(getattr(args, "fetch", False))
    if fetch and (requested_source or not managed_source):
        print("Cannot use --fetch with an explicit local source; local sources are never fetched.", file=sys.stderr)
        return 2

    command = [str(installer)]
    if requested_source:
        command.extend(["--source", str(Path(requested_source).expanduser().resolve())])
    elif not managed_source:
        command.extend(["--source", str(source_root)])
    if fetch:
        command.append("--fetch")

    environment = os.environ.copy()
    environment.update(
        {
            "OPENTULPA_INSTALL_ROOT": str(install_root),
            "OPENTULPA_INSTALL_REPOSITORY": repository,
            "OPENTULPA_INSTALL_REF": ref,
        }
    )
    print("Installing a verified OpenTulpa controller generation...", flush=True)
    try:
        completed = subprocess.run(command, env=environment, check=False)  # noqa: S603
    except OSError as exc:
        print(f"OpenTulpa update could not start: {exc}", file=sys.stderr)
        return 1
    if completed.returncode != 0:
        print(
            f"OpenTulpa update failed with status {completed.returncode}; the active controller was unchanged.",
            file=sys.stderr,
        )
        return completed.returncode or 1

    try:
        controller_executable, generation = _active_controller_entrypoint(controller_root)
    except RuntimeError as exc:
        print(f"OpenTulpa update activated invalid controller state: {exc}", file=sys.stderr)
        return 1
    print(f"Activated OpenTulpa controller generation {generation}.")
    if bool(getattr(args, "restart_local_host", False)):
        try:
            restarted_url = restart_remembered_local_server(
                controller_executable=controller_executable,
                controller_generation_id=generation,
            )
        except LocalServerError as exc:
            print(f"Controller updated, but the local host was not restarted: {exc}", file=sys.stderr)
            return 1
        if restarted_url:
            print(f"Restarted the remembered local OpenTulpa host at {restarted_url}.")
        else:
            print("No remembered local OpenTulpa host was running.")
    return 0


def _active_controller_entrypoint(controller_root: Path) -> tuple[Path, str]:
    current = controller_root / "current"
    if not current.is_symlink():
        raise RuntimeError("controller/current is not a symbolic link")
    target = os.readlink(current)
    parts = Path(target).parts
    if len(parts) != 2 or parts[0] != "generations" or re.fullmatch(r"[0-9a-f]{64}", parts[1]) is None:
        raise RuntimeError("controller/current escapes the generation store")
    generation = controller_root / parts[0] / parts[1]
    if generation.is_symlink() or not generation.is_dir():
        raise RuntimeError("active controller generation directory is invalid")
    executable = generation / "bin" / "opentulpa-host"
    try:
        metadata = executable.lstat()
    except OSError as exc:
        raise RuntimeError("active controller host executable is missing") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or not metadata.st_mode & stat.S_IXUSR
    ):
        raise RuntimeError("active controller host executable is invalid")
    return executable.absolute(), parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="opentulpa")
    subparsers = parser.add_subparsers(dest="command")
    server = subparsers.add_parser(
        "server",
        aliases=["serve"],
        help="run a headless OpenTulpa server",
    )
    server.add_argument("--host", default=os.environ.get("HOST") or "0.0.0.0")
    server.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8000))
    server.add_argument("--public-url")
    connect = subparsers.add_parser("connect", help="connect and open the terminal client")
    connect.add_argument("url")
    credential = connect.add_mutually_exclusive_group()
    credential.add_argument("--token")
    credential.add_argument("--pairing-code")
    connect.add_argument("--no-tui", action="store_true", help="save without opening the TUI")
    subparsers.add_parser("status", help="show remote host status")
    logs = subparsers.add_parser("logs", help="show redacted runtime logs")
    logs.add_argument("--follow", action="store_true")
    subparsers.add_parser("disconnect", help="forget the server and owner credential")
    update = subparsers.add_parser("update", help="install and activate a verified controller update")
    update.add_argument("--source", help="import an exact clean local checkout")
    update.add_argument(
        "--fetch",
        action="store_true",
        help="fetch the configured ref for the installer-managed source seed",
    )
    update.add_argument(
        "--restart-local-host",
        action="store_true",
        help="restart only the exact private local host recorded by OpenTulpa",
    )
    args = parser.parse_args()
    if args.command in {"server", "serve"}:
        _server_command(args)
        return
    if args.command is None:
        _open_tui()
        return
    if args.command == "update":
        raise SystemExit(_update_command(args))
    raise SystemExit(_remote(args))


__all__ = ["build_host_application", "main", "serve"]
