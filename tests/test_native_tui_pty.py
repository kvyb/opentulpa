from __future__ import annotations

import fcntl
import json
import os
import platform
import pty
import select
import signal
import struct
import sys
import termios
import threading
import time
import unittest
import warnings
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

THREAD = {
    "thread_id": "thread-smoke",
    "title": "Terminal smoke test",
    "channel": "web",
    "archived": False,
    "created_at": "2026-01-01T00:00:00Z",
    "updated_at": "2026-01-01T00:00:00Z",
    "last_run_id": None,
    "status": "idle",
    "preview": "",
}
NEW_THREAD = {
    **THREAD,
    "thread_id": "thread-new",
    "title": "New session",
}
THREAD_CREATED = threading.Event()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/v2/agent/threads":
            self._json({"threads": [THREAD], "next_cursor": None})
            return
        if path == "/v2/agent/threads/thread-smoke/timeline":
            self._json({"thread": THREAD, "entries": [], "next_cursor": None})
            return
        if path == "/v2/agent/threads/thread-new/timeline":
            self._json({"thread": NEW_THREAD, "entries": [], "next_cursor": None})
            return
        if path == "/v2/inference":
            self._json(
                {
                    "api_default": {
                        "provider": "api",
                        "model": "moonshotai/kimi-k3",
                        "reasoning_effort": None,
                        "service_tier": None,
                        "fallback_to_api": False,
                    },
                    "codex": {
                        "connected": False,
                        "credential_revision": 0,
                        "experimental": True,
                    },
                }
            )
            return
        if path == "/v2/inference/models":
            self._json(
                {
                    "provider": "api",
                    "models": [
                        {
                            "provider": "api",
                            "id": "moonshotai/kimi-k3",
                            "reasoning_efforts": ["low", "medium", "high"],
                            "default_reasoning_effort": "low",
                            "service_tiers": [],
                            "default_service_tier": None,
                        }
                    ],
                }
            )
            return
        if path in {
            "/v2/agent/threads/thread-smoke/inference",
            "/v2/agent/threads/thread-new/inference",
        }:
            self._json(
                {
                    "revision": 0,
                    "selection": None,
                    "effective": {
                        "provider": "api",
                        "model": "moonshotai/kimi-k3",
                        "reasoning_effort": None,
                        "service_tier": None,
                        "fallback_to_api": False,
                    },
                }
            )
            return
        if path == "/v2/notifications":
            time.sleep(0.1)
            self._json({"notifications": [], "next_after_id": 0})
            return
        self._json({"detail": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.split("?", 1)[0] == "/v2/agent/threads":
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            THREAD_CREATED.set()
            self._json(NEW_THREAD, status=201)
            return
        self._json({"detail": "not found"}, status=404)

    def _json(self, payload: object, *, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_native_tui_pty() -> None:
    binary = _default_binary()
    if binary is None or not binary.is_file():
        raise unittest.SkipTest("native TUI binary has not been built")
    _run_smoke(binary)


def _default_binary() -> Path | None:
    system = {"Darwin": "darwin", "Linux": "linux"}.get(platform.system())
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "x64"}.get(
        platform.machine()
    )
    if system is None or machine is None:
        return None
    return Path(__file__).parents[1] / "clients" / "tui" / "dist" / f"opentulpa-tui-{system}-{machine}"


def _run_smoke(binary: Path) -> None:
    THREAD_CREATED.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    connection_read, connection_write = os.pipe()
    state_read, state_write = os.pipe()
    pid = -1
    master = -1
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded, use of forkpty.*",
                category=DeprecationWarning,
            )
            pid, master = pty.fork()
        if pid == 0:
            os.close(connection_write)
            os.close(state_read)
            os.set_inheritable(connection_read, True)
            os.set_inheritable(state_write, True)
            environment = os.environ.copy()
            environment.update(
                {
                    "OPENTULPA_CONNECTION_FD": str(connection_read),
                    "OPENTULPA_STATE_FD": str(state_write),
                    "TERM": "xterm-256color",
                }
            )
            os.execve(binary, [str(binary)], environment)
        thread.start()
        os.close(connection_read)
        connection_read = -1
        os.close(state_write)
        state_write = -1
        fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 30, 120, 0, 0))
        os.write(
            connection_write,
            json.dumps(
                {
                    "url": f"http://127.0.0.1:{server.server_port}",
                    "token": "smoke-owner-token",
                    "thread_id": THREAD["thread_id"],
                }
            ).encode(),
        )
        os.close(connection_write)
        connection_write = -1

        output = _read_until(master, b"ready")
        assert os.waitpid(pid, os.WNOHANG) == (0, 0)
        os.write(master, b"/mo")
        time.sleep(0.1)
        command_output = _read_until(master, b"Choose the provider and model")
        assert b"/model" in command_output
        os.write(master, b"\r")
        model_output = _read_until(master, b"Server default")
        if b"moonshotai/kimi-k3" not in model_output:
            model_output += _read_until(master, b"moonshotai/kimi-k3")
        assert b"moonshotai/kimi-k3" in model_output
        os.write(master, b"\x1b")
        os.write(master, b"\x0e")
        assert THREAD_CREATED.wait(timeout=3), "ctrl+n did not create a new session"
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pid = -1
        assert b"\x1b[?1049h" in output
        assert b"OpenTulpa" in output
        assert b"ready" in output
    finally:
        server.shutdown()
        server.server_close()
        if pid > 0:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        for descriptor in (master, connection_read, connection_write, state_read, state_write):
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)


def _read_until(descriptor: int, expected: bytes, timeout: float = 8.0) -> bytes:
    deadline = time.monotonic() + timeout
    output = bytearray()
    while time.monotonic() < deadline:
        ready, _, _ = select.select([descriptor], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(descriptor, 65_536)
            except OSError:
                break
            output.extend(chunk)
            cursor_queries = chunk.count(b"\x1b[6n")
            if cursor_queries:
                os.write(descriptor, b"\x1b[1;1R" * cursor_queries)
            if b"\x1b[14t" in chunk:
                os.write(descriptor, b"\x1b[4;600;800t")
            if b"\x1b]10;?\x07" in chunk:
                os.write(descriptor, b"\x1b]10;rgb:ffff/ffff/ffff\x1b\\")
            if b"\x1b]11;?\x07" in chunk:
                os.write(descriptor, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            if expected in output:
                return bytes(output)
    raise AssertionError(f"terminal did not render expected content: {bytes(output)!r}")


if __name__ == "__main__":
    executable = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _default_binary()
    if executable is None or not executable.is_file():
        raise SystemExit("native TUI binary was not found")
    _run_smoke(executable)
    print("native TUI PTY smoke passed")
