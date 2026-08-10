"""Rootful-Linux black-box rehearsal for trusted source activation."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socket import socket
from typing import Any

_HOST_EXECUTABLE = Path("/opt/opentulpa-install/controller/generations/image/bin/opentulpa-host")
_INTERNAL_PREFIX = "/bootstrap/internal/v1/evolution"
_TENANT_ID = "rootful-e2e-tenant"
_OWNER_TOKEN = "rootful-e2e-owner-token-value-000000000000"


class _ModelHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self._json({"data": [{"id": "rootful-e2e-model"}]})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions":
            self._json(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {"content": "ok", "role": "assistant"},
                        }
                    ],
                    "id": "rootful-e2e",
                    "model": "rootful-e2e-model",
                    "object": "chat.completion",
                }
            )
            return
        self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _json(self, payload: object) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _free_port() -> int:
    with closing(socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-OpenTulpa-Evolution-Token"] = token
    request = urllib.request.Request(
        f"{base_url}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            value = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{method} {path} failed with {exc.code}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned a non-object")
    return value


def _wait_for(
    read: Callable[[], Any],
    accept: Callable[[Any], bool],
    *,
    timeout: float,
    label: str,
) -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            last = read()
            if accept(last):
                return last
        except (OSError, RuntimeError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {label}; last value: {last!r}")


def _audit() -> dict[str, str]:
    return {
        "actor_id": "rootful-e2e-owner",
        "channel": "web",
        "correlation_id": "rootful-e2e",
        "run_kind": "owner",
        "tenant_id": _TENANT_ID,
        "thread_id": "rootful-e2e-thread",
    }


def _source(
    base_url: str,
    token: str,
    path: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    return _request(
        base_url,
        "POST",
        f"{_INTERNAL_PREFIX}{path}",
        body={**body, "audit_context": _audit()},
        token=token,
    )


def _status(base_url: str, token: str) -> dict[str, Any]:
    return _source(base_url, token, "/source/status", {})


def _wait_activation(
    base_url: str,
    token: str,
    activation_id: str,
    *,
    expected_status: str,
) -> dict[str, Any]:
    status = _wait_for(
        lambda: _status(base_url, token),
        lambda value: (
            isinstance(value.get("activation"), dict)
            and value["activation"].get("activation_id") == activation_id
            and value["activation"].get("status") in {"active", "failed", "rolled_back"}
        ),
        timeout=300,
        label=f"source activation {activation_id}",
    )
    activation = status["activation"]
    if activation["status"] != expected_status:
        raise RuntimeError(f"source activation ended unexpectedly: {activation!r}")
    return status


def _runtime_record(data_root: Path) -> dict[str, Any]:
    value = json.loads((data_root / "bootstrap" / "runtime-child.json").read_text())
    if not isinstance(value, dict) or value.get("mode") != "live_source":
        raise RuntimeError("runtime ownership record is not live-source bound")
    return value


def _assert_runtime_identity(record: dict[str, Any], source_commit: str) -> None:
    if record.get("source_commit") != source_commit:
        raise RuntimeError("runtime serves the wrong source commit")
    pid = int(record["pid"])
    lines = (Path("/proc") / str(pid) / "status").read_text(encoding="ascii").splitlines()
    status = {
        key: value.strip()
        for line in lines
        if ":" in line
        for key, value in (line.split(":", 1),)
    }
    if status.get("Uid", "").split() != ["65532"] * 4:
        raise RuntimeError("runtime UID boundary is invalid")
    if status.get("Gid", "").split() != ["65532"] * 4:
        raise RuntimeError("runtime GID boundary is invalid")
    if status.get("CapEff") != "0000000000000000" or status.get("NoNewPrivs") != "1":
        raise RuntimeError("runtime process privileges are unsafe")
    if os.getpgid(pid) != pid or int(record["process_group"]) != pid:
        raise RuntimeError("runtime process group identity is invalid")


def _start_host(environment: dict[str, str], log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    log = log_path.open("ab", buffering=0)
    process = subprocess.Popen(  # noqa: S603
        [str(_HOST_EXECUTABLE)],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    return process, log


def _stop_host(process: subprocess.Popen[bytes], log: Any) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    log.close()


def main() -> None:
    if os.name != "posix" or os.geteuid() != 0 or not Path("/proc/self/ns/mnt").is_file():
        raise RuntimeError("this rehearsal requires rootful Linux")
    if not _HOST_EXECUTABLE.is_file():
        raise RuntimeError("the installed OpenTulpa host is unavailable")

    data_root = Path(tempfile.mkdtemp(prefix="opentulpa-rootful-e2e-", dir="/tmp"))
    model = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    model_thread = threading.Thread(target=model.serve_forever, daemon=True)
    model_thread.start()
    base_url = f"http://127.0.0.1:{_free_port()}"
    environment = {
        **os.environ,
        "EVOLUTION_ENABLED": "true",
        "HOST": "127.0.0.1",
        "LLM_MODEL": "rootful-e2e-model",
        "OPENAI_COMPATIBLE_API_KEY": "rootful-e2e-key",
        "OPENAI_COMPATIBLE_BASE_URL": f"http://127.0.0.1:{model.server_address[1]}/v1",
        "OPENTULPA_DATA_ROOT": str(data_root),
        "OPENTULPA_OWNER_CUSTOMER_ID": _TENANT_ID,
        "OPENTULPA_OWNER_TOKEN": _OWNER_TOKEN,
        "OPENTULPA_RUNTIME_PROBATION_PROBE_INTERVAL_SECONDS": "0.1",
        "OPENTULPA_RUNTIME_PROBATION_SECONDS": "1",
        "PORT": base_url.rsplit(":", 1)[1],
    }
    process: subprocess.Popen[bytes] | None = None
    log: Any = None
    log_path = Path("/tmp") / f"{data_root.name}.log"
    try:
        process, log = _start_host(environment, log_path)
        _wait_for(
            lambda: _request(base_url, "GET", "/healthz"),
            lambda value: value.get("ok") is True and value.get("runtime") == "ready",
            timeout=180,
            label="initial runtime",
        )
        token = (data_root / "bootstrap" / "evolution.token").read_text().strip()
        initial = _status(base_url, token)
        initial_release = str(initial["active_release_id"])
        initial_commit = str(initial["active_source_commit"])
        _assert_runtime_identity(_runtime_record(data_root), initial_commit)

        _source(
            base_url,
            token,
            "/source/write",
            {
                "path": "src/opentulpa/e2e_source_marker.py",
                "content": 'MARKER = "trusted-source-e2e"\n',
            },
        )
        queued = _source(
            base_url,
            token,
            "/source/activate",
            {
                "idempotency_key": "rootful-e2e-activate",
                "message": "Rootful trusted source activation",
                "reason": "Rootful E2E",
            },
        )
        active = _wait_activation(
            base_url,
            token,
            str(queued["activation_id"]),
            expected_status="active",
        )
        active_release = str(active["active_release_id"])
        active_commit = str(active["active_source_commit"])
        if active_commit == initial_commit:
            raise RuntimeError("source activation did not change the serving commit")
        _assert_runtime_identity(_runtime_record(data_root), active_commit)

        rollback = _source(
            base_url,
            token,
            "/source/rollback",
            {
                "idempotency_key": "rootful-e2e-rollback",
                "expected_active_release_id": active_release,
                "reason": "Rootful E2E explicit rollback",
            },
        )
        restored = _wait_activation(
            base_url,
            token,
            str(rollback["activation_id"]),
            expected_status="rolled_back",
        )
        if restored["active_release_id"] != initial_release:
            raise RuntimeError("rollback did not restore the initial release")
        _assert_runtime_identity(_runtime_record(data_root), initial_commit)

        _stop_host(process, log)
        process = None
        log = None
        process, log = _start_host(environment, log_path)
        _wait_for(
            lambda: _request(base_url, "GET", "/healthz"),
            lambda value: value.get("ok") is True and value.get("runtime") == "ready",
            timeout=180,
            label="restarted runtime",
        )
        restarted = _status(base_url, token)
        if restarted["active_release_id"] != initial_release:
            raise RuntimeError("host restart lost the durable rollback state")
        _assert_runtime_identity(_runtime_record(data_root), initial_commit)
        print(
            json.dumps(
                {
                    "activated_release_id": active_release,
                    "initial_release_id": initial_release,
                    "restored_release_id": restarted["active_release_id"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as exc:
        if log_path.is_file():
            print(log_path.read_text(encoding="utf-8", errors="replace")[-20_000:])
        raise RuntimeError("rootful self-evolution rehearsal failed") from exc
    finally:
        if process is not None and log is not None:
            _stop_host(process, log)
        model.shutdown()
        model.server_close()
        model_thread.join(timeout=5)
        shutil.rmtree(data_root, ignore_errors=True)


if __name__ == "__main__":
    main()
