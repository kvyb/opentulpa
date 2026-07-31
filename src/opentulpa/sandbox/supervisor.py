"""Host-owned sandbox worker lifecycle and readiness checks."""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import sys
from dataclasses import dataclass
from pathlib import Path

from opentulpa.core.config import Settings
from opentulpa.sandbox.client import (
    SandboxWorkerCanary,
    SandboxWorkerExecutionProvider,
    SandboxWorkerHealth,
)


def _private_token(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.parent.is_symlink():
        raise RuntimeError("sandbox credential directory cannot be a symlink")
    if not path.exists():
        value = secrets.token_urlsafe(48)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(f"{value}\n")
            stream.flush()
            os.fsync(stream.fileno())
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError("sandbox credential must be a private regular file")
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 32:
        raise RuntimeError("sandbox credential is invalid")
    return value


@dataclass(frozen=True, slots=True)
class SandboxRuntimeConfig:
    base_url: str
    token: str


class SandboxWorkerSupervisor:
    """Start or bind the mandatory sandbox worker for one host process."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        settings: Settings,
    ) -> None:
        self._project_root = project_root.resolve()
        self._data_root = data_root.resolve()
        self._settings = settings
        configured_url = str(os.environ.get("OPENTULPA_SANDBOX_RPC_URL") or "").strip()
        configured_token = str(os.environ.get("OPENTULPA_SANDBOX_RPC_TOKEN") or "").strip()
        if configured_url or configured_token:
            if not configured_url or len(configured_token) < 32:
                raise RuntimeError(
                    "OPENTULPA_SANDBOX_RPC_URL and OPENTULPA_SANDBOX_RPC_TOKEN are both required"
                )
            self._config = SandboxRuntimeConfig(
                base_url=configured_url.rstrip("/"),
                token=configured_token,
            )
            self._managed_process = False
        else:
            token = _private_token(data_root / "bootstrap" / "sandbox-worker.token")
            port = _free_local_port()
            self._config = SandboxRuntimeConfig(
                base_url=f"http://127.0.0.1:{port}/internal/v1/sandbox",
                token=token,
            )
            self._managed_process = True
        self._process: asyncio.subprocess.Process | None = None
        self._client = SandboxWorkerExecutionProvider(
            base_url=self._config.base_url,
            token=self._config.token,
            max_response_bytes=settings.sandbox_max_output_bytes + 65_536,
            max_archive_bytes=settings.railway_sandbox_max_sync_bytes,
            max_archive_entries=settings.sandbox_max_workspace_entries,
            max_file_bytes=settings.sandbox_max_file_bytes,
        )
        self._last_canary: SandboxWorkerCanary | None = None

    @property
    def config(self) -> SandboxRuntimeConfig:
        return self._config

    @property
    def client(self) -> SandboxWorkerExecutionProvider:
        return self._client

    @property
    def last_canary(self) -> SandboxWorkerCanary | None:
        return self._last_canary

    async def start(self) -> None:
        if self._managed_process and self._process is None:
            self._process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "opentulpa.sandbox.worker",
                cwd=self._project_root,
                env=self._worker_environment(),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        await self.require_ready()

    async def require_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + 30
        last: SandboxWorkerCanary | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self._process is not None and self._process.returncode is not None:
                break
            last = await asyncio.to_thread(self._client.canary)
            self._last_canary = last
            if last.ok:
                return
            if last.health is not None and last.health.tier != "unavailable":
                break
            await asyncio.sleep(0.25)
        detail = _canary_summary(last or self._client.canary())
        raise RuntimeError(f"sandbox worker failed readiness: {detail}")

    async def status(self) -> dict[str, object]:
        canary = await asyncio.to_thread(self._client.canary)
        self._last_canary = canary
        return _canary_payload(canary)

    async def shutdown(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()

    def _worker_environment(self) -> dict[str, str]:
        root = self._data_root / "sandbox_worker"
        environment = {
            "HOST": "127.0.0.1",
            "PORT": self._config.base_url.split(":", 2)[2].split("/", 1)[0],
            "PATH": os.environ.get("PATH", os.defpath),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "PYTHONPATH": str(self._project_root / "src"),
            "OPENTULPA_SANDBOX_RPC_TOKEN": self._config.token,
            "OPENTULPA_SANDBOX_WORKER_ROOT": str(root),
            "OPENTULPA_SANDBOX_MAX_OUTPUT_BYTES": str(self._settings.sandbox_max_output_bytes),
            "OPENTULPA_SANDBOX_MAX_ARCHIVE_BYTES": str(
                self._settings.railway_sandbox_max_sync_bytes
            ),
            "OPENTULPA_SANDBOX_MAX_ENTRIES": str(
                self._settings.sandbox_max_workspace_entries
            ),
            "OPENTULPA_SANDBOX_MAX_FILE_BYTES": str(self._settings.sandbox_max_file_bytes),
            "OPENTULPA_SANDBOX_CPU_LIMIT": self._settings.sandbox_cpu_limit,
            "OPENTULPA_SANDBOX_MEMORY_LIMIT": self._settings.sandbox_memory_limit,
            "OPENTULPA_SANDBOX_PID_LIMIT": str(self._settings.sandbox_pid_limit),
            "OPENTULPA_SANDBOX_TIMEOUT_SECONDS": str(self._settings.sandbox_timeout_seconds),
        }
        if str(os.environ.get("OPENTULPA_DEV_ALLOW_NO_SANDBOX") or "").strip():
            environment["OPENTULPA_DEV_ALLOW_NO_SANDBOX"] = str(
                os.environ.get("OPENTULPA_DEV_ALLOW_NO_SANDBOX")
            )
        return environment


def _canary_payload(canary: SandboxWorkerCanary) -> dict[str, object]:
    health: SandboxWorkerHealth | None = canary.health
    return {
        "ok": canary.ok,
        "step": canary.step,
        "tier": health.tier if health is not None else "unavailable",
        "checks": health.checks if health is not None else {},
        "error": canary.error or (health.error if health is not None else None),
    }


def _canary_summary(canary: SandboxWorkerCanary | None) -> str:
    if canary is None:
        return "sandbox worker did not respond"
    payload = _canary_payload(canary)
    checks = payload.get("checks")
    failed = [
        name
        for name, passed in (checks.items() if isinstance(checks, dict) else ())
        if not bool(passed)
    ]
    if failed:
        return f"{payload['step']} failed checks: {', '.join(failed)}"
    return str(payload.get("error") or f"{payload['step']} failed")


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


__all__ = ["SandboxRuntimeConfig", "SandboxWorkerSupervisor"]
