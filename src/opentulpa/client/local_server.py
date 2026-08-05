"""Lifecycle for the private host started by the local terminal client."""

from __future__ import annotations

import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

_STATE_VERSION = 4
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
    executable: str = ""
    launch_argv: tuple[str, ...] = ()
    controller_generation_id: str = ""
    process_start_token: str = ""
    process_executable: str = ""
    process_command: str = ""


@dataclass(frozen=True, slots=True)
class _ProcessIdentity:
    start_token: str
    executable: str
    argv: tuple[str, ...]
    command: str


def local_data_root() -> Path:
    configured = str(os.environ.get("OPENTULPA_DATA_ROOT") or "").strip()
    if configured:
        return _safe_absolute_path(Path(configured).expanduser())
    xdg = str(os.environ.get("XDG_DATA_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return _safe_absolute_path(base / "opentulpa")


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
    controller_executable: Path | None = None,
    controller_generation_id: str | None = None,
) -> str:
    """Return a healthy local host URL, starting its detached process when needed."""

    exact_controller_requested = controller_executable is not None
    if (
        not exact_controller_requested
        and preferred_url
        and is_loopback_url(preferred_url)
        and _host_ready(preferred_url)
    ):
        return preferred_url.rstrip("/")
    state = _load_state()
    if not exact_controller_requested and state is not None and _host_ready(state.url):
        return state.url
    if not exact_controller_requested and state is not None and _pid_alive(state.pid) and _wait_ready(
        state.url,
        process=None,
        timeout_seconds=5,
    ):
        return state.url

    preferred_port = _url_port(preferred_url) if preferred_url else None
    port = preferred_port or (state.port if state is not None else _DEFAULT_PORT)
    if not _port_available(port):
        candidate_url = f"http://127.0.0.1:{port}"
        if (
            not exact_controller_requested
            and (preferred_url is not None or state is not None)
            and _host_ready(candidate_url)
        ):
            return candidate_url
        port = _free_port()

    root = local_data_root()
    _secure_private_directory(root)
    _bootstrap_root(create=True)
    log_path = _log_path()
    descriptor = _open_private_append(log_path)
    source_root = _source_root()
    if controller_executable is not None:
        executable, generation_id = _validated_controller_executable(
            controller_executable,
            controller_generation_id,
        )
        command = [executable]
    else:
        executable = str(Path(sys.executable).absolute())
        generation_id = str(controller_generation_id or "")
        command = [executable, "-m", "opentulpa.host"]
    environment = os.environ.copy()
    environment.update(
        {
            "HOST": "127.0.0.1",
            "PORT": str(port),
            "OPENTULPA_DATA_ROOT": str(root),
            "OPENTULPA_OPEN_BROWSER": "0",
        }
    )
    if (source_root / "pyproject.toml").is_file() and (
        source_root / "src" / "opentulpa" / "__init__.py"
    ).is_file():
        environment["OPENTULPA_SOURCE_ROOT"] = str(source_root)
    else:
        environment.pop("OPENTULPA_SOURCE_ROOT", None)
    try:
        process = subprocess.Popen(  # noqa: S603
            command,
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
    try:
        identity = _capture_process_identity(process.pid, process=process)
    except BaseException:
        if process.poll() is None:
            process.terminate()
        raise
    url = f"http://127.0.0.1:{port}"
    _write_state(
        LocalServerState(
            pid=process.pid,
            port=port,
            url=url,
            source_root=str(source_root),
            started_at=datetime.now(UTC).isoformat(),
            executable=executable,
            launch_argv=identity.argv,
            controller_generation_id=generation_id,
            process_start_token=identity.start_token,
            process_executable=identity.executable,
            process_command=identity.command,
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
    configured_assets = str(os.environ.get("OPENTULPA_INSTALL_ASSETS_ROOT") or "").strip()
    if configured_assets:
        assets = Path(configured_assets).expanduser().resolve()
        if assets.is_dir():
            return assets
    return Path(__file__).resolve().parents[1]


def _safe_absolute_path(raw: Path) -> Path:
    path = Path(os.path.abspath(os.fspath(raw)))
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            continue
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise LocalServerError("Local OpenTulpa data has a symbolic-link ancestor.")
    return path


def _secure_private_directory(path: Path) -> Path:
    safe = _safe_absolute_path(path)
    safe.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = safe.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise LocalServerError("Local OpenTulpa data directory is not private and trusted.")
    safe.chmod(0o700)
    return safe


def _bootstrap_root(*, create: bool) -> Path:
    root = local_data_root()
    if create:
        _secure_private_directory(root)
    elif not root.is_dir():
        raise FileNotFoundError(root)
    return _secure_private_directory(root / "bootstrap") if create else _validated_private_directory(
        root / "bootstrap"
    )


def _validated_private_directory(path: Path) -> Path:
    safe = _safe_absolute_path(path)
    metadata = safe.lstat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise LocalServerError("Local OpenTulpa bootstrap directory is not private and trusted.")
    return safe


def _validate_private_regular(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or stat.S_IMODE(metadata.st_mode) & 0o077
    ):
        raise LocalServerError("Local OpenTulpa state file is not private and trusted.")
    return metadata


def _read_private_file(path: Path, *, max_bytes: int) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
            or metadata.st_size > max_bytes
        ):
            raise LocalServerError("Local OpenTulpa state file is not private and trusted.")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read(max_bytes + 1)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_private_append(path: Path) -> int:
    flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise LocalServerError("Local OpenTulpa log is not private and trusted.") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise LocalServerError("Local OpenTulpa log is not private and trusted.")
        os.fchmod(descriptor, 0o600)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _validated_controller_executable(
    executable: Path,
    generation_id: str | None,
) -> tuple[str, str]:
    selected_id = str(generation_id or "").strip().lower()
    if len(selected_id) != 64 or any(value not in "0123456789abcdef" for value in selected_id):
        raise LocalServerError("The new controller generation identity is invalid.")
    path = _safe_absolute_path(executable.expanduser())
    metadata = path.lstat()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
        or not metadata.st_mode & stat.S_IXUSR
        or path.parent.name != "bin"
        or path.parent.parent.name != selected_id
    ):
        raise LocalServerError("The new controller executable is not trusted.")
    return str(path), selected_id


def _state_path() -> Path:
    return local_data_root() / "bootstrap" / "local-server.json"


def _log_path() -> Path:
    return local_data_root() / "bootstrap" / "local-server.log"


def _load_state() -> LocalServerState | None:
    path = _state_path()
    try:
        _bootstrap_root(create=False)
        _validate_private_regular(path)
        payload = json.loads(_read_private_file(path, max_bytes=64 * 1024).decode("utf-8"))
        version = int(payload.get("version", 0))
        if version not in {1, 2, _STATE_VERSION}:
            return None
        raw_argv = payload.get("launch_argv") or []
        if not isinstance(raw_argv, list) or any(not isinstance(value, str) for value in raw_argv):
            return None
        state = LocalServerState(
            pid=int(payload["pid"]),
            port=int(payload["port"]),
            url=str(payload["url"]),
            source_root=str(payload.get("source_root") or ""),
            started_at=str(payload.get("started_at") or ""),
            executable=str(payload.get("executable") or ""),
            launch_argv=tuple(raw_argv),
            controller_generation_id=str(payload.get("controller_generation_id") or ""),
            process_start_token=str(payload.get("process_start_token") or ""),
            process_executable=str(payload.get("process_executable") or ""),
            process_command=str(payload.get("process_command") or ""),
        )
    except (KeyError, OSError, TypeError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    if state.pid < 1 or not 1 <= state.port <= 65535 or not is_loopback_url(state.url):
        return None
    return state


def _write_state(state: LocalServerState) -> None:
    path = _state_path()
    _bootstrap_root(create=True)
    if os.path.lexists(path):
        _validate_private_regular(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
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
                    "executable": state.executable,
                    "launch_argv": list(state.launch_argv),
                    "controller_generation_id": state.controller_generation_id,
                    "process_start_token": state.process_start_token,
                    "process_executable": state.process_executable,
                    "process_command": state.process_command,
                },
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
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


def restart_remembered_local_server(
    *,
    controller_executable: Path,
    controller_generation_id: str,
    timeout_seconds: float = 45,
) -> str | None:
    """Restart only the private local host whose exact process was recorded."""

    state = _load_state()
    if state is None:
        return None
    new_executable, new_generation_id = _validated_controller_executable(
        controller_executable,
        controller_generation_id,
    )
    if not state.executable or not _pid_matches_local_host(state):
        raise LocalServerError("Refusing to restart an unverified local OpenTulpa process.")
    pidfd = _open_verified_pidfd(state)
    if pidfd is None:
        raise LocalServerError(
            "Automatic local host restart requires pidfd support; stop the remembered host manually."
        )
    try:
        sender = signal.__dict__.get("pidfd_send_signal")
        if not callable(sender):
            raise LocalServerError("pidfd signaling became unavailable during restart.")
        sender(pidfd, signal.SIGTERM)
    except OSError as exc:
        raise LocalServerError("The remembered local OpenTulpa process is no longer running.") from exc
    finally:
        os.close(pidfd)
    deadline = time.monotonic() + min(timeout_seconds, 30)
    while time.monotonic() < deadline and _pid_matches_local_host(state):
        time.sleep(0.1)
    if _pid_matches_local_host(state):
        raise LocalServerError("The remembered local OpenTulpa process did not stop cleanly.")
    current = _load_state()
    if current is not None and current == state:
        _validate_private_regular(_state_path())
        _state_path().unlink()
    return ensure_local_server(
        preferred_url=state.url,
        timeout_seconds=timeout_seconds,
        controller_executable=Path(new_executable),
        controller_generation_id=new_generation_id,
    )


def _pid_matches_local_host(state: LocalServerState) -> bool:
    if not state.process_start_token or not state.launch_argv:
        return False
    identity = _read_process_identity(state.pid)
    if identity is None or identity.start_token != state.process_start_token:
        return False
    if identity.argv != state.launch_argv:
        return False
    if state.process_executable and identity.executable != state.process_executable:
        return False
    return not state.process_command or identity.command == state.process_command


def _capture_process_identity(
    pid: int,
    *,
    process: subprocess.Popen[bytes],
) -> _ProcessIdentity:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        identity = _read_process_identity(pid)
        if identity is not None and identity.start_token and identity.argv:
            return identity
        if process.poll() is not None:
            break
        time.sleep(0.01)
    raise LocalServerError("Could not verify the newly started local OpenTulpa process.")


def _read_process_identity(pid: int) -> _ProcessIdentity | None:
    if pid < 1:
        return None
    proc_root = Path(f"/proc/{pid}")
    try:
        stat_payload = (proc_root / "stat").read_text(encoding="ascii")
        command_end = stat_payload.rfind(")")
        if command_end < 0:
            return None
        remaining_fields = stat_payload[command_end + 2 :].split()
        if len(remaining_fields) < 20:
            return None
        arguments = tuple(
                value.decode(errors="surrogateescape")
                for value in (proc_root / "cmdline").read_bytes().split(b"\0")
                if value
        )
        return _ProcessIdentity(
            start_token=f"linux:{remaining_fields[19]}",
            executable=os.readlink(proc_root / "exe"),
            argv=arguments,
            command="",
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError):
        pass
    try:
        started = subprocess.run(  # noqa: S603
            ["ps", "-p", str(pid), "-o", "lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        command = subprocess.run(  # noqa: S603
            ["ps", "-ww", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    started_at = started.stdout.strip()
    command_line = command.stdout.strip()
    if started.returncode != 0 or command.returncode != 0 or not started_at or not command_line:
        return None
    return _ProcessIdentity(
        start_token=f"ps:{started_at}",
        executable="",
        argv=tuple(command_line.split()),
        command=command_line,
    )


def _open_verified_pidfd(state: LocalServerState) -> int | None:
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
    if pidfd_open is None or pidfd_send_signal is None:
        return None
    try:
        descriptor = int(pidfd_open(state.pid, 0))
    except OSError as exc:
        raise LocalServerError("Could not acquire a safe handle for the remembered process.") from exc
    if _pid_matches_local_host(state):
        return descriptor
    os.close(descriptor)
    raise LocalServerError("The remembered local OpenTulpa process identity changed before restart.")


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
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


__all__ = [
    "LocalServerError",
    "ensure_local_server",
    "is_loopback_url",
    "local_data_root",
    "restart_remembered_local_server",
]
