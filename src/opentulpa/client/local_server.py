"""Lifecycle for the private host started by the local terminal client."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

_STATE_VERSION = 1
_DEFAULT_PORT = 8000


class LocalServerError(RuntimeError):
    """The private local host could not be started or restored."""


@dataclass(frozen=True, slots=True)
class LocalServerState:
    pid: int
    port: int
    url: str
    source_root: str
    started_at: str


def local_data_root() -> Path:
    configured = str(os.environ.get("OPENTULPA_DATA_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    xdg = str(os.environ.get("XDG_DATA_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return (base / "opentulpa").resolve()


def is_loopback_url(url: str) -> bool:
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return False
    return hostname in {"127.0.0.1", "localhost", "::1"}


def ensure_local_server(
    *,
    preferred_url: str | None = None,
    timeout_seconds: float = 45,
) -> str:
    """Return a healthy local host URL, starting its detached process when needed."""

    if preferred_url and is_loopback_url(preferred_url) and _host_ready(preferred_url):
        return preferred_url.rstrip("/")
    state = _load_state()
    if state is not None and _host_ready(state.url):
        return state.url
    if state is not None and _pid_alive(state.pid) and _wait_ready(
        state.url,
        process=None,
        timeout_seconds=5,
    ):
        return state.url

    preferred_port = _url_port(preferred_url) if preferred_url else None
    port = preferred_port or (state.port if state is not None else _DEFAULT_PORT)
    if not _port_available(port):
        candidate_url = f"http://127.0.0.1:{port}"
        if (preferred_url is not None or state is not None) and _host_ready(candidate_url):
            return candidate_url
        port = _free_port()

    root = local_data_root()
    bootstrap = root / "bootstrap"
    bootstrap.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(bootstrap, 0o700)
    log_path = _log_path()
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    source_root = _source_root()
    environment = os.environ.copy()
    environment.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "OPENTULPA_DATA_ROOT": str(root),
            "OPENTULPA_SOURCE_ROOT": str(source_root),
            "OPENTULPA_OPEN_BROWSER": "0",
        }
    )
    try:
        process = subprocess.Popen(  # noqa: S603
            [sys.executable, "-m", "opentulpa.host"],
            cwd=source_root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(descriptor)
    url = f"http://127.0.0.1:{port}"
    _write_state(
        LocalServerState(
            pid=process.pid,
            port=port,
            url=url,
            source_root=str(source_root),
            started_at=datetime.now(UTC).isoformat(),
        )
    )
    if _wait_ready(url, process=process, timeout_seconds=timeout_seconds):
        return url
    if process.poll() is None:
        process.terminate()
    raise LocalServerError(f"OpenTulpa did not start. See {log_path}.")


def _source_root() -> Path:
    configured = str(os.environ.get("OPENTULPA_SOURCE_ROOT") or "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if (candidate / "pyproject.toml").is_file():
            return candidate
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "pyproject.toml").is_file():
        return package_root
    return local_data_root()


def _state_path() -> Path:
    return local_data_root() / "bootstrap" / "local-server.json"


def _log_path() -> Path:
    return local_data_root() / "bootstrap" / "local-server.log"


def _load_state() -> LocalServerState | None:
    path = _state_path()
    try:
        if path.is_symlink() or path.stat().st_mode & 0o077:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0)) != _STATE_VERSION:
            return None
        state = LocalServerState(
            pid=int(payload["pid"]),
            port=int(payload["port"]),
            url=str(payload["url"]),
            source_root=str(payload.get("source_root") or ""),
            started_at=str(payload.get("started_at") or ""),
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if state.pid < 1 or not 1 <= state.port <= 65535 or not is_loopback_url(state.url):
        return None
    return state


def _write_state(state: LocalServerState) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                {
                    "version": _STATE_VERSION,
                    "pid": state.pid,
                    "port": state.port,
                    "url": state.url,
                    "source_root": state.source_root,
                    "started_at": state.started_at,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _host_ready(url: str) -> bool:
    try:
        with httpx.Client(timeout=1, trust_env=False) as client:
            response = client.get(f"{url.rstrip('/')}/healthz")
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    return bool(response.is_success and isinstance(payload, dict) and payload.get("host") == "ready")


def _wait_ready(
    url: str,
    *,
    process: subprocess.Popen[bytes] | None,
    timeout_seconds: float,
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return False
        if _host_ready(url):
            return True
        time.sleep(0.1)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _url_port(url: str | None) -> int | None:
    if not url:
        return None
    try:
        parsed = urlsplit(url)
        return parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None


def _port_available(port: int) -> bool:
    if not 1 <= port <= 65535:
        return False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


__all__ = ["LocalServerError", "ensure_local_server", "is_loopback_url", "local_data_root"]
