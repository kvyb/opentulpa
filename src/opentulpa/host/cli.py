"""One-command host launcher and small remote operations client."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from pydantic import SecretStr

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
)
from opentulpa.core.config import get_settings
from opentulpa.host.app import create_host_app
from opentulpa.host.models import HostConfigInput
from opentulpa.host.runtime import RuntimeSupervisor
from opentulpa.host.service import HostService
from opentulpa.host.store import HostStore
from opentulpa.secrets.host_key import load_or_create_host_cipher


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
    configured_source = str(os.environ.get("OPENTULPA_SOURCE_ROOT") or "").strip()
    project_root = (
        Path(configured_source).expanduser().resolve()
        if configured_source
        else Path(__file__).resolve().parents[3]
    )
    settings = get_settings()
    configured_root = str(os.environ.get("OPENTULPA_DATA_ROOT") or "").strip()
    data_root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path.home() / ".local" / "share" / "opentulpa"
    ).resolve()
    data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    setup_token = _private_pairing_code(data_root / "bootstrap" / "host-setup.token")
    store = HostStore(
        data_root / "bootstrap" / "host.db",
        cipher=load_or_create_host_cipher(data_root),
    )
    store.configure_setup_token(setup_token)
    host = str(os.environ.get("HOST") or "127.0.0.1").strip()
    owner_token = str(os.environ.get("OPENTULPA_OWNER_TOKEN") or "").strip()
    if not owner_token and host in {"127.0.0.1", "localhost", "::1"}:
        owner_token = _private_token(data_root / "bootstrap" / "owner.token")
    if owner_token and not store.claimed:
        store.claim(setup_token=setup_token, owner_token=owner_token)

    api_key = str(os.environ.get("OPENAI_COMPATIBLE_API_KEY") or "").strip()
    telegram_token = str(os.environ.get("TELEGRAM_BOT_TOKEN") or "").strip()
    telegram_ids = str(os.environ.get("TELEGRAM_ALLOWED_USER_IDS") or "").strip()
    telegram_id = None
    if telegram_ids:
        first = telegram_ids.split(",", 1)[0].strip()
        if first.isdigit() and int(first) > 0:
            telegram_id = int(first)
    bootstrap = None
    if api_key and store.active() is None:
        bootstrap = HostConfigInput(
            api_key=SecretStr(api_key),
            base_url=settings.openai_compatible_base_url,
            model=settings.llm_model,
            telegram_bot_token=SecretStr(telegram_token) if telegram_token else None,
            telegram_user_id=telegram_id if telegram_token else None,
        )
    runtime = RuntimeSupervisor(project_root=project_root, data_root=data_root)
    service = HostService(store=store, runtime=runtime, bootstrap_config=bootstrap)
    app = create_host_app(
        store=store,
        service=service,
        local_owner_enabled=host in {"127.0.0.1", "localhost", "::1"},
        setup_token=setup_token,
    )
    return app, host, setup_token, data_root


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
            f"Connected to {connection.url}; credential stored in "
            f"{connection.credential_storage}."
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
    if not str(args.host or "").strip() or any(
        character.isspace() for character in str(args.host)
    ):
        raise SystemExit("Server host is invalid.")
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    if args.public_url:
        try:
            os.environ["PUBLIC_BASE_URL"] = normalize_url(args.public_url)
        except ClientConfigError as exc:
            raise SystemExit(str(exc)) from exc
    serve(public_url=args.public_url)


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
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
        raise SystemExit(f"OpenTulpa terminal client is not executable: {binary}")
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
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    for installed_name in ("opentulpa-tui", target_name):
        installed = shutil.which(installed_name)
        if installed:
            return Path(installed).resolve()
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
    return binary


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
    args = parser.parse_args()
    if args.command in {"server", "serve"}:
        _server_command(args)
        return
    if args.command is None:
        _open_tui()
        return
    raise SystemExit(_remote(args))


__all__ = ["build_host_application", "main", "serve"]
