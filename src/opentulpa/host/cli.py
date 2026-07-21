"""One-command host launcher and small remote operations client."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from pydantic import SecretStr

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
    owner_token = str(os.environ.get("OPENTULPA_WEB_TOKEN") or "").strip()
    if not owner_token and host in {"127.0.0.1", "localhost", "::1"}:
        owner_token = _private_token(data_root / "bootstrap" / "owner-web.token")
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
        print(f"One-time setup token: {setup_token}")
    print(f"Persistent data: {data_root}")
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        ws="none",
        timeout_graceful_shutdown=15,
    )


def _client_config_path() -> Path:
    return Path(
        os.environ.get("OPENTULPA_CLIENT_CONFIG") or "~/.config/opentulpa/client.json"
    ).expanduser()


def _save_client(url: str, token: str) -> None:
    path = _client_config_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump({"url": url.rstrip("/"), "token": token}, stream)
        stream.write("\n")
    os.replace(temporary, path)


def _load_client() -> dict[str, str]:
    path = _client_config_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit("Run `opentulpa connect URL --token TOKEN` first.") from exc
    return {"url": str(value["url"]), "token": str(value["token"])}


def _remote(args: argparse.Namespace) -> int:
    if args.command == "connect":
        config = {"url": args.url.rstrip("/"), "token": args.token}
    else:
        config = _load_client()
    headers = {"Authorization": f"Bearer {config['token']}"}
    with httpx.Client(timeout=30, trust_env=False) as client:
        if args.command == "connect":
            response = client.get(f"{config['url']}/_host/api/status", headers=headers)
            if not response.is_success or not response.json().get("authenticated"):
                print("OpenTulpa rejected the owner token.", file=sys.stderr)
                return 1
            _save_client(config["url"], config["token"])
            print(f"Connected to {config['url']}")
            return 0
        if args.command == "status":
            response = client.get(f"{config['url']}/_host/api/status", headers=headers)
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
            return 0
        if args.command == "logs":
            if args.follow:
                with client.stream(
                    "GET", f"{config['url']}/_host/api/logs/stream", headers=headers
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line.startswith("data: "):
                            entry = json.loads(line[6:])
                            print(f"{entry['stream']:<6} {entry['text']}", flush=True)
                return 0
            response = client.get(f"{config['url']}/_host/api/logs", headers=headers)
            response.raise_for_status()
            for entry in response.json()["logs"]:
                print(f"{entry['stream']:<6} {entry['text']}")
            return 0
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(prog="opentulpa")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("serve", help="start the stable host and mutable agent runtime")
    connect = subparsers.add_parser("connect", help="save a remote owner connection")
    connect.add_argument("url")
    connect.add_argument("--token", required=True)
    subparsers.add_parser("status", help="show remote host status")
    logs = subparsers.add_parser("logs", help="show redacted runtime logs")
    logs.add_argument("--follow", action="store_true")
    args = parser.parse_args()
    if args.command in {None, "serve"}:
        serve()
        return
    raise SystemExit(_remote(args))


__all__ = ["build_host_application", "main", "serve"]
