"""One-command host launcher and small remote operations client."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from pydantic import SecretStr

from opentulpa.client.config import (
    ClientConfigError,
    Connection,
    clear_connection,
    load_connection,
    normalize_url,
    save_connection,
)
from opentulpa.client.tui import run_tui
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


def build_host_application() -> tuple[Any, str, str, Path]:
    project_root = Path(__file__).resolve().parents[3]
    settings = get_settings()
    configured_root = str(os.environ.get("OPENTULPA_DATA_ROOT") or "").strip()
    data_root = (
        Path(configured_root).expanduser()
        if configured_root
        else Path.home() / ".local" / "share" / "opentulpa"
    ).resolve()
    data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    setup_token = _private_token(data_root / "bootstrap" / "host-setup.token")
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


def serve() -> None:
    app, host, setup_token, data_root = build_host_application()
    port = int(os.environ.get("PORT") or 8000)
    print(f"OpenTulpa host: http://{host}:{port}/")
    print(f"Setup and recovery: http://{host}:{port}/_host")
    if host not in {"127.0.0.1", "localhost", "::1"} and not app.state.host_store.claimed:
        print(f"One-time pairing code: {setup_token}")
    print(f"Persistent data: {data_root}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        ws="none",
        timeout_graceful_shutdown=15,
    )


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
        asyncio.run(run_tui(connection))
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
        return load_connection()
    except ClientConfigError:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            raise SystemExit("Run `opentulpa connect SERVER_URL` first.") from None
    print("Connect this terminal to an OpenTulpa server.")
    url = input("Server URL: ").strip()
    credential = getpass.getpass("Owner token or pairing code (blank for local): ").strip()
    try:
        normalized = normalize_url(url)
    except ClientConfigError as exc:
        raise SystemExit(str(exc)) from exc
    token = _connect_credential(normalized, credential)
    if token is None:
        raise SystemExit(1)
    try:
        return save_connection(normalized, token)
    except ClientConfigError as exc:
        raise SystemExit(str(exc)) from exc


def _open_tui() -> None:
    connection = _interactive_connection()
    try:
        asyncio.run(run_tui(connection))
    except KeyboardInterrupt:
        return


def main() -> None:
    parser = argparse.ArgumentParser(prog="opentulpa")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="start the stable host and mutable agent runtime")
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
    if args.command == "serve":
        serve()
        return
    if args.command is None:
        _open_tui()
        return
    raise SystemExit(_remote(args))


__all__ = ["build_host_application", "main", "serve"]
