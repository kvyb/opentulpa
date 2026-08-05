"""Rootful-Linux black-box rehearsal for immutable self-evolution.

Run this file inside the built OpenTulpa image with the capabilities documented in
``docs/E2E_TESTING.md``. It intentionally is not collected by the default pytest run.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import signal
import sqlite3
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
from typing import Any, cast

_CONTROLLER_ROOT = Path("/opt/opentulpa-install/controller/generations/image")
_HOST_EXECUTABLE = _CONTROLLER_ROOT / "bin/opentulpa-host"
_INTERNAL_PREFIX = "/bootstrap/internal/v1/evolution"
_THREAD_ID = "rootful-e2e-thread"
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
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _free_port() -> int:
    with closing(socket()) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _request_json(
    base_url: str,
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    token: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    data = json.dumps(body, separators=(",", ":")).encode("utf-8") if body is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    if token is not None:
        request_headers["X-OpenTulpa-Evolution-Token"] = token
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:4_000]
        raise RuntimeError(f"{method} {path} failed with {exc.code}: {detail}") from exc
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {path} returned a non-object response")
    return parsed


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


def _audit_context() -> dict[str, str]:
    return {
        "actor_id": "rootful-e2e-owner",
        "channel": "web",
        "correlation_id": "rootful-e2e-correlation",
        "origin": json.dumps(
            {
                "conversation_id": _THREAD_ID,
                "interface": "web",
                "source_id": "rootful-e2e",
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "run_id": "rootful-e2e-run",
        "run_kind": "owner",
        "tenant_id": _TENANT_ID,
        "thread_id": _THREAD_ID,
    }


def _source_call(
    base_url: str,
    token: str,
    path: str,
    body: dict[str, Any],
    *,
    timeout: float = 60,
) -> dict[str, Any]:
    return _request_json(
        base_url,
        "POST",
        f"{_INTERNAL_PREFIX}{path}",
        body={**body, "audit_context": _audit_context()},
        token=token,
        timeout=timeout,
    )


def _source_script(base_url: str, token: str, script: str) -> dict[str, Any]:
    encoded = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return _source_call(
        base_url,
        token,
        "/source/shell",
        {
            "command": f"python -c \"import base64;exec(base64.b64decode('{encoded}'))\"",
            "timeout_seconds": 120,
        },
        timeout=180,
    )


def _source_status(base_url: str, token: str) -> dict[str, Any]:
    return _source_call(base_url, token, "/source/status", {})


def _resolve_source_dependencies(base_url: str, token: str) -> dict[str, Any]:
    status = _source_status(base_url, token)
    candidate_id = str(status.get("candidate_id") or "")
    diff_sha256 = str(status.get("diff_sha256") or "")
    if not status.get("dirty") or not candidate_id or len(diff_sha256) != 64:
        raise RuntimeError(f"dependency proposal is not resolvable: {status!r}")
    resolved = _source_call(
        base_url,
        token,
        "/source/resolve-dependencies",
        {
            "expected_candidate_id": candidate_id,
            "expected_diff_sha256": diff_sha256,
        },
        timeout=1_800,
    )
    required_hashes = (
        "dependency_base_id",
        "dependency_inventory_sha256",
        "dependency_lock_hash",
        "dependency_wheelhouse_sha256",
    )
    if any(len(str(resolved.get(name) or "")) != 64 for name in required_hashes):
        raise RuntimeError(f"dependency resolution metadata is incomplete: {resolved!r}")
    if "uv.lock" not in resolved.get("changed_files", []):
        raise RuntimeError(f"dependency resolution did not install its trusted lock: {resolved!r}")
    return resolved


def _release_source(
    base_url: str,
    token: str,
    *,
    operation: str,
    message: str,
) -> dict[str, Any]:
    status = _source_status(base_url, token)
    candidate_id = str(status.get("candidate_id") or "")
    diff_sha256 = str(status.get("diff_sha256") or "")
    if not status.get("dirty") or not candidate_id or len(diff_sha256) != 64:
        raise RuntimeError(f"source session is not releasable: {status!r}")
    released = _source_call(
        base_url,
        token,
        "/source/release",
        {
            "expected_candidate_id": candidate_id,
            "expected_diff_sha256": diff_sha256,
            "idempotency_key": operation,
            "message": message,
        },
        timeout=1_800,
    )
    promotion = released.get("promotion")
    if not isinstance(promotion, dict) or not promotion.get("id"):
        raise RuntimeError(f"source release did not queue a promotion: {released!r}")
    return promotion


def _promotion(base_url: str, token: str, attempt_id: str) -> dict[str, Any]:
    return _request_json(
        base_url,
        "GET",
        f"{_INTERNAL_PREFIX}/promotions/{attempt_id}",
        token=token,
    )


def _wait_for_promotion(
    base_url: str,
    token: str,
    attempt_id: str,
    *,
    expected: str,
) -> dict[str, Any]:
    if expected not in {"active", "failed"}:
        raise ValueError("expected promotion status is invalid")
    terminal = _wait_for(
        lambda: _promotion(base_url, token, attempt_id),
        lambda value: value.get("status") in {"active", "failed"},
        timeout=300,
        label=f"promotion {attempt_id} to become {expected}",
    )
    if terminal.get("status") != expected:
        raise RuntimeError(
            f"promotion {attempt_id} became {terminal.get('status')!r}, expected {expected!r}: "
            f"{terminal!r}"
        )
    return cast(dict[str, Any], terminal)


def _runtime_record(data_root: Path) -> dict[str, Any]:
    path = data_root / "bootstrap" / "runtime-child.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("mode") != "generation":
        raise RuntimeError("runtime ownership record is not generation-bound")
    return value


def _runtime_url(record: dict[str, Any]) -> str:
    pid = int(record["pid"])
    raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    environment = {
        name.decode("ascii"): value.decode("ascii")
        for entry in raw.split(b"\0")
        if entry
        for name, separator, value in (entry.partition(b"="),)
        if separator
    }
    try:
        port = int(environment["PORT"])
    except (KeyError, ValueError) as exc:
        raise RuntimeError("serving runtime has no exact port binding") from exc
    if not 1 <= port <= 65_535:
        raise RuntimeError("serving runtime port is invalid")
    return f"http://127.0.0.1:{port}"


def _assert_runtime_boundary(data_root: Path, record: dict[str, Any]) -> None:
    pid = int(record["pid"])
    status_lines = (Path("/proc") / str(pid) / "status").read_text(encoding="ascii").splitlines()
    status = {
        key: value.strip()
        for line in status_lines
        if ":" in line
        for key, value in (line.split(":", 1),)
    }
    if status.get("Uid", "").split() != ["65532"] * 4:
        raise RuntimeError(f"runtime UID boundary is invalid: {status.get('Uid')!r}")
    if status.get("Gid", "").split() != ["65532"] * 4:
        raise RuntimeError(f"runtime GID boundary is invalid: {status.get('Gid')!r}")
    if status.get("CapEff") != "0000000000000000":
        raise RuntimeError(f"runtime retained effective capabilities: {status.get('CapEff')!r}")
    if status.get("NoNewPrivs") != "1":
        raise RuntimeError("runtime is not protected by no_new_privs")
    if os.getpgid(pid) != pid or int(record["process_group"]) != pid:
        raise RuntimeError("runtime is not isolated in its recorded process group")

    executable = Path(str(record["executable"]))
    generation_root = executable.parents[2]
    if generation_root.name != record["generation_id"]:
        raise RuntimeError("runtime executable escaped its recorded generation")
    probe = subprocess.run(  # noqa: S603
        [
            "/usr/bin/setpriv",
            "--reuid=65532",
            "--regid=65532",
            "--clear-groups",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--bounding-set=-all",
            "--no-new-privs",
            "/usr/local/bin/python3",
            "-c",
            (
                "import errno,sys;from pathlib import Path;"
                "blocked={errno.EACCES,errno.EPERM,errno.EROFS};"
                "\nfor root in map(Path,sys.argv[1:]):"
                "\n path=root/'.rootful-e2e-runtime-write'"
                "\n try:path.write_bytes(b'unsafe')"
                "\n except OSError as exc:"
                "\n  if exc.errno not in blocked:raise"
                "\n else:raise SystemExit('runtime boundary is writable: '+str(root))"
            ),
            str(data_root / "bootstrap"),
            str(generation_root),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"runtime filesystem boundary failed: stdout={probe.stdout!r} stderr={probe.stderr!r}"
        )


def _notification_rows(data_root: Path) -> list[dict[str, Any]]:
    database = data_root / "product" / ".opentulpa" / "notifications.db"
    if not database.is_file():
        return []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, kind, thread_id, origin_json
            FROM owner_notifications
            WHERE tenant_id = ?
            ORDER BY id ASC
            """,
            (_TENANT_ID,),
        ).fetchall()
    return [dict(row) for row in rows]


def _assert_notification_sequence(data_root: Path) -> list[dict[str, Any]]:
    expected = (
        "evolution.build.preparing",
        "evolution.build.switching",
        "evolution.promotion.active",
        "evolution.build.preparing",
        "evolution.build.switching",
        "evolution.promotion.failed",
        "evolution.build.switching",
        "evolution.rollback.active",
    )
    rows = _wait_for(
        lambda: _notification_rows(data_root),
        lambda value: all(sum(row["kind"] == kind for row in value) >= expected.count(kind) for kind in set(expected)),
        timeout=30,
        label="conversation-bound release notifications",
    )
    filtered = [row for row in rows if row["kind"] in set(expected)]
    kinds = tuple(row["kind"] for row in filtered)
    if kinds != expected:
        raise RuntimeError(f"release notification order changed: {kinds!r}")
    for row in filtered:
        origin = json.loads(str(row["origin_json"]))
        if row["thread_id"] != _THREAD_ID or origin.get("conversation_id") != _THREAD_ID:
            raise RuntimeError("release notification left the originating conversation")
    return filtered


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
    host_port = _free_port()
    model = ThreadingHTTPServer(("127.0.0.1", 0), _ModelHandler)
    model_thread = threading.Thread(target=model.serve_forever, daemon=True)
    model_thread.start()
    model_port = int(model.server_address[1])
    base_url = f"http://127.0.0.1:{host_port}"
    environment = {
        **os.environ,
        "EVOLUTION_ENABLED": "true",
        "HOST": "127.0.0.1",
        "LLM_MODEL": "rootful-e2e-model",
        "OPENAI_COMPATIBLE_API_KEY": "rootful-e2e-key",
        "OPENAI_COMPATIBLE_BASE_URL": f"http://127.0.0.1:{model_port}/v1",
        "OPENTULPA_DATA_ROOT": str(data_root),
        "OPENTULPA_OWNER_CUSTOMER_ID": _TENANT_ID,
        "OPENTULPA_OWNER_TOKEN": _OWNER_TOKEN,
        "OPENTULPA_RUNTIME_PROBATION_PROBE_INTERVAL_SECONDS": "0.1",
        "OPENTULPA_RUNTIME_PROBATION_SECONDS": "1",
        "PORT": str(host_port),
    }
    process: subprocess.Popen[bytes] | None = None
    log: Any = None
    log_path = Path("/tmp") / f"{data_root.name}.log"
    try:
        process, log = _start_host(environment, log_path)
        _wait_for(
            lambda: _request_json(base_url, "GET", "/healthz"),
            lambda value: value.get("ok") is True and value.get("runtime") == "ready",
            timeout=180,
            label="initial host and runtime readiness",
        )
        token_path = data_root / "bootstrap" / "evolution.token"
        token = token_path.read_text(encoding="utf-8").strip()
        initial_status = _source_status(base_url, token)
        if initial_status.get("available") is not True:
            raise RuntimeError(f"rootful source mutation is unavailable: {initial_status!r}")
        initial_release_id = str(initial_status["current_release_id"])
        initial_runtime = _runtime_record(data_root)
        initial_generation_id = str(initial_runtime["generation_id"])
        _assert_runtime_boundary(data_root, initial_runtime)

        dependency_resolution: dict[str, Any] | None = None
        if os.environ.get("OPENTULPA_DEPENDENCY_RESOLVER_IMAGE_DIGEST", "").strip():
            success_script = (
                "from pathlib import Path\n"
                "project_path = Path('pyproject.toml')\n"
                "project = project_path.read_text(encoding='utf-8')\n"
                "dependency_anchor = '    \\\"tree-sitter-bash>=0.25,<0.26\\\",\\n'\n"
                "assert project.count(dependency_anchor) == 1\n"
                "project_path.write_text(\n"
                "    project.replace(\n"
                "        dependency_anchor,\n"
                "        dependency_anchor + '    \\\"tomli-w==1.2.0\\\",\\n',\n"
                "    ),\n"
                "    encoding='utf-8',\n"
                ")\n"
                "app_path = Path('src/opentulpa/api/app.py')\n"
                "source = app_path.read_text(encoding='utf-8')\n"
                "import_anchor = 'from fastapi import FastAPI, Request, Response\\n'\n"
                "content_anchor = '            \\\"status\\\": status,\\n'\n"
                "assert source.count(import_anchor) == 1\n"
                "assert source.count(content_anchor) == 1\n"
                "source = source.replace(import_anchor, 'import tomli_w\\n' + import_anchor)\n"
                "source = source.replace(\n"
                "    content_anchor,\n"
                "    content_anchor\n"
                "    + '            \\\"dependency_probe\\\": '\n"
                "    + 'tomli_w.dumps({\\\"resolved\\\": True}).strip(),\\n',\n"
                ")\n"
                "app_path.write_text(source, encoding='utf-8')\n"
            )
        else:
            success_script = (
                "from pathlib import Path\n"
                "Path('src/opentulpa/e2e_generation_marker.py').write_text("
                "'MARKER = \\\"rootful-phase1-success\\\"\\n', encoding='utf-8')\n"
            )
        success_shell = _source_script(base_url, token, success_script)
        if success_shell.get("exit_code") != 0:
            raise RuntimeError(f"successful candidate edit failed: {success_shell!r}")
        if os.environ.get("OPENTULPA_DEPENDENCY_RESOLVER_IMAGE_DIGEST", "").strip():
            dependency_resolution = _resolve_source_dependencies(base_url, token)
        successful = _release_source(
            base_url,
            token,
            operation="rootful-e2e-success",
            message="Rootful E2E successful generation",
        )
        successful = _wait_for_promotion(
            base_url,
            token,
            str(successful["id"]),
            expected="active",
        )
        if successful.get("status") != "active":
            raise RuntimeError(f"successful generation did not activate: {successful!r}")
        successful_release = successful["release"]
        successful_release_id = str(successful_release["id"])
        successful_generation_id = str(successful_release["metadata"]["generation_id"])
        if successful_generation_id == initial_generation_id:
            raise RuntimeError("successful source edit reused the initial generation identity")
        successful_runtime = _runtime_record(data_root)
        if successful_runtime["generation_id"] != successful_generation_id:
            raise RuntimeError("successful release is not the serving process generation")
        _assert_runtime_boundary(data_root, successful_runtime)
        if dependency_resolution is not None:
            health = _request_json(
                _runtime_url(successful_runtime),
                "GET",
                "/healthz",
            )
            if health.get("dependency_probe") != "resolved = true":
                raise RuntimeError(f"resolved dependency is not active in the runtime: {health!r}")

        failed_shell = _source_script(
            base_url,
            token,
            "from pathlib import Path\n"
            "path = Path('src/opentulpa/api/app.py')\n"
            "source = path.read_text(encoding='utf-8')\n"
            "old = '        ready = app.state.lifecycle_status == \\\"ready\\\"\\n'\n"
            "new = (\n"
            "    '        health_probes = getattr(app.state, \\\"e2e_health_probes\\\", 0) + 1\\n'\n"
            "    '        app.state.e2e_health_probes = health_probes\\n'\n"
            "    '        data_root = os.environ.get(\\\"OPENTULPA_DATA_ROOT\\\")\\n'\n"
            "    '        ready = app.state.lifecycle_status == \\\"ready\\\" and not bool(\\n'\n"
            "    '            health_probes > 1\\n'\n"
            "    '            and data_root\\n'\n"
            "    '            and os.path.isfile(os.path.join(data_root, \\\"e2e-fail-next\\\"))\\n'\n"
            "    '        )\\n'\n"
            ")\n"
            "assert source.count(old) == 1\n"
            "path.write_text(source.replace(old, new), encoding='utf-8')\n",
        )
        if failed_shell.get("exit_code") != 0:
            raise RuntimeError(f"failure candidate edit failed: {failed_shell!r}")
        failure_marker = data_root / "product" / "e2e-fail-next"
        failure_marker.write_text("fail the candidate health probe\n", encoding="utf-8")
        failed = _release_source(
            base_url,
            token,
            operation="rootful-e2e-failure",
            message="Rootful E2E forced health failure",
        )
        failed = _wait_for_promotion(
            base_url,
            token,
            str(failed["id"]),
            expected="failed",
        )
        if failed.get("status") != "failed" or failed.get("failure_code") != "release_unhealthy":
            raise RuntimeError(f"candidate health failure was not contained: {failed!r}")
        status_after_failure = _source_status(base_url, token)
        if status_after_failure.get("current_release_id") != successful_release_id:
            raise RuntimeError("failed candidate changed the active release")
        if _runtime_record(data_root)["generation_id"] != successful_generation_id:
            raise RuntimeError("failed candidate did not restore the exact previous generation")
        _assert_runtime_boundary(data_root, _runtime_record(data_root))
        failure_marker.unlink()

        rollback = _source_call(
            base_url,
            token,
            "/source/rollback",
            {
                "expected_current_release_id": successful_release_id,
                "expected_target_release_id": initial_release_id,
                "idempotency_key": "rootful-e2e-rollback",
                "reason": "Rootful E2E explicit rollback",
            },
        )
        rollback = _wait_for_promotion(
            base_url,
            token,
            str(rollback["id"]),
            expected="active",
        )
        if rollback.get("status") != "active":
            raise RuntimeError(f"explicit rollback did not activate: {rollback!r}")
        rollback_release_id = str(rollback["release"]["id"])
        rollback_runtime = _runtime_record(data_root)
        if rollback_runtime["generation_id"] != initial_generation_id:
            raise RuntimeError("explicit rollback did not restore the initial generation")
        _assert_runtime_boundary(data_root, rollback_runtime)
        if dependency_resolution is not None:
            rollback_health = _request_json(
                _runtime_url(rollback_runtime),
                "GET",
                "/healthz",
            )
            if "dependency_probe" in rollback_health:
                raise RuntimeError(
                    f"rollback retained the resolved dependency behavior: {rollback_health!r}"
                )
        notifications = _assert_notification_sequence(data_root)

        _stop_host(process, log)
        process = None
        log = None
        process, log = _start_host(environment, log_path)
        _wait_for(
            lambda: _request_json(base_url, "GET", "/healthz"),
            lambda value: value.get("ok") is True and value.get("runtime") == "ready",
            timeout=180,
            label="restarted host and runtime readiness",
        )
        restarted = _source_status(base_url, token)
        if restarted.get("current_release_id") != rollback_release_id:
            raise RuntimeError("restart lost the durable rollback release")
        if _runtime_record(data_root)["generation_id"] != initial_generation_id:
            raise RuntimeError("restart did not serve the rolled-back generation")
        notification_kinds = {notification["kind"] for notification in notifications}
        restarted_notifications = [
            row for row in _notification_rows(data_root) if row["kind"] in notification_kinds
        ]
        if restarted_notifications != notifications:
            raise RuntimeError("restart lost or rewrote conversation notifications")

        print(
            json.dumps(
                {
                    "failed_attempt_id": failed["id"],
                    "dependency_base_id": (
                        dependency_resolution["dependency_base_id"]
                        if dependency_resolution is not None
                        else None
                    ),
                    "initial_generation_id": initial_generation_id,
                    "notification_kinds": [row["kind"] for row in notifications],
                    "rollback_release_id": rollback_release_id,
                    "successful_generation_id": successful_generation_id,
                    "successful_release_id": successful_release_id,
                },
                indent=2,
                sort_keys=True,
            )
        )
    except Exception as exc:
        try:
            host_logs = _request_json(
                base_url,
                "GET",
                "/_host/api/logs",
                headers={"Authorization": f"Bearer {_OWNER_TOKEN}"},
            )
        except Exception:
            host_logs = None
        if host_logs is not None:
            print(json.dumps(host_logs, indent=2, sort_keys=True))
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
        log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
