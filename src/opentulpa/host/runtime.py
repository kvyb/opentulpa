"""Child Deep Agents runtime lifecycle owned by the stable host."""

from __future__ import annotations

import asyncio
import os
import re
import socket
import sys
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from opentulpa.host.models import HostConfig
from opentulpa.persistence.tenant_namespace import tenant_namespace_label

_SECRET_LINE = re.compile(
    r"(?i)(api[_-]?key|authorization|bot[_-]?token|secret|password)(\s*[:=]\s*)(\S+)"
)


class RuntimeUnavailableError(RuntimeError):
    """The mutable child runtime did not become healthy."""


class RuntimeLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    sequence: int
    timestamp: datetime
    stream: str
    text: str


@dataclass(slots=True)
class _Child:
    process: asyncio.subprocess.Process
    endpoint: str
    config: HostConfig
    readers: tuple[asyncio.Task[None], ...]


class RuntimeSupervisor:
    """Run exactly one mutable runtime and retain a stable recovery surface."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        startup_timeout_seconds: float = 90,
        shutdown_timeout_seconds: float = 15,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._project_root = project_root.resolve()
        self._data_root = data_root.resolve()
        self._startup_timeout = startup_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2, read=5, write=5, pool=2), trust_env=False
        )
        self._owns_client = client is None
        self._child: _Child | None = None
        self._status = "stopped"
        self._error: str | None = None
        self._logs: deque[RuntimeLogEntry] = deque(maxlen=2_000)
        self._redaction_values: set[str] = set()
        self._sequence = 0
        self._log_changed = asyncio.Condition()
        self._lock = asyncio.Lock()

    @property
    def endpoint(self) -> str | None:
        return self._child.endpoint if self._child is not None else None

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def revision(self) -> int | None:
        return self._child.config.revision if self._child is not None else None

    async def start(self, config: HostConfig) -> None:
        async with self._lock:
            if self._child is not None:
                await self._stop_child(self._child)
                self._child = None
            self._child = await self._spawn(config)

    async def replace(self, config: HostConfig, *, rollback: HostConfig | None) -> None:
        """Replace the child, restoring the previous config when activation fails."""

        async with self._lock:
            previous = self._child
            if previous is not None:
                await self._stop_child(previous)
                self._child = None
            try:
                self._child = await self._spawn(config)
            except Exception:
                if rollback is not None:
                    self._append_log("host", "candidate failed; restoring previous runtime")
                    try:
                        self._child = await self._spawn(rollback)
                    except Exception as rollback_error:
                        self._error = "candidate and rollback runtimes failed to start"
                        self._append_log("host", self._error)
                        raise RuntimeUnavailableError(self._error) from rollback_error
                raise

    async def restart_current(self) -> None:
        child = self._child
        if child is None:
            raise RuntimeUnavailableError("runtime is not configured")
        await self.replace(child.config, rollback=child.config)

    def clear_telegram_identity(self) -> None:
        """Forget a stopped Telegram interface after a committed disconnect."""

        state_path = self._telegram_state_path()
        if state_path.is_symlink():
            raise RuntimeUnavailableError("Telegram state cannot be a symbolic link")
        state_path.unlink(missing_ok=True)

    async def stop(self) -> None:
        async with self._lock:
            child = self._child
            self._child = None
            if child is not None:
                await self._stop_child(child)
            self._status = "stopped"

    async def shutdown(self) -> None:
        await self.stop()
        if self._owns_client:
            await self._client.aclose()

    def logs(self, *, after: int = 0) -> list[RuntimeLogEntry]:
        return [entry for entry in self._logs if entry.sequence > after]

    async def wait_for_logs(self, *, after: int, timeout: float = 15) -> list[RuntimeLogEntry]:
        current = self.logs(after=after)
        if current:
            return current
        async with self._log_changed:
            with suppress(TimeoutError):
                await asyncio.wait_for(self._log_changed.wait(), timeout=timeout)
        return self.logs(after=after)

    async def _spawn(self, config: HostConfig) -> _Child:
        port = self._free_port()
        endpoint = f"http://127.0.0.1:{port}"
        self._status = "starting"
        self._error = None
        self._redaction_values = {
            value
            for value in (
                config.api_key.get_secret_value(),
                config.internal_runtime_token.get_secret_value(),
                config.telegram_bot_token.get_secret_value()
                if config.telegram_bot_token is not None
                else "",
                config.telegram_pairing_code.get_secret_value()
                if config.telegram_pairing_code is not None
                else "",
            )
            if value
        }
        self._append_log("host", f"starting runtime revision {config.revision}")
        self._seed_telegram_identity(config)
        environment = self._child_environment(config, port=port)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "opentulpa",
            cwd=self._project_root,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        readers = tuple(
            asyncio.create_task(self._read_stream(stream, name))
            for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
            if stream is not None
        )
        child = _Child(process=process, endpoint=endpoint, config=config, readers=readers)
        try:
            await self._wait_ready(child)
        except Exception as exc:
            await self._stop_child(child)
            self._status = "failed"
            self._error = self._safe_error(exc)
            self._append_log("host", f"runtime failed: {self._error}")
            raise RuntimeUnavailableError(self._error) from exc
        self._status = "ready"
        self._append_log("host", f"runtime revision {config.revision} is ready")
        return child

    async def _wait_ready(self, child: _Child) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout
        headers = {
            "Authorization": f"Bearer {child.config.internal_runtime_token.get_secret_value()}"
        }
        while asyncio.get_running_loop().time() < deadline:
            if child.process.returncode is not None:
                raise RuntimeUnavailableError(
                    f"runtime exited before readiness with code {child.process.returncode}"
                )
            try:
                health = await self._client.get(f"{child.endpoint}/healthz")
                agent = await self._client.get(f"{child.endpoint}/agent/healthz", headers=headers)
                if health.is_success and agent.is_success:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.25)
        raise RuntimeUnavailableError("runtime readiness timed out")

    async def _stop_child(self, child: _Child) -> None:
        if child.process.returncode is None:
            child.process.terminate()
            try:
                await asyncio.wait_for(child.process.wait(), timeout=self._shutdown_timeout)
            except TimeoutError:
                child.process.kill()
                await child.process.wait()
        for reader in child.readers:
            if not reader.done():
                reader.cancel()
            with suppress(asyncio.CancelledError):
                await reader

    async def _read_stream(self, stream: asyncio.StreamReader, name: str) -> None:
        while line := await stream.readline():
            self._append_log(name, line.decode("utf-8", errors="replace").rstrip())

    def _append_log(self, stream: str, text: str) -> None:
        self._sequence += 1
        safe_text = text
        for value in sorted(self._redaction_values, key=len, reverse=True):
            safe_text = safe_text.replace(value, "[redacted]")
        entry = RuntimeLogEntry(
            sequence=self._sequence,
            timestamp=datetime.now(UTC),
            stream=stream,
            text=_SECRET_LINE.sub(r"\1\2[redacted]", safe_text)[:8_000],
        )
        self._logs.append(entry)

        async def notify() -> None:
            async with self._log_changed:
                self._log_changed.notify_all()

        with suppress(RuntimeError):
            asyncio.get_running_loop().create_task(notify())

    def _child_environment(self, config: HostConfig, *, port: int) -> dict[str, str]:
        environment = os.environ.copy()
        for key in (
            "TELEGRAM_BOT_TOKEN",
            "TELEGRAM_ALLOWED_USER_IDS",
            "TELEGRAM_ALLOWED_USERNAMES",
            "TELEGRAM_WEBHOOK_SECRET",
            "PUBLIC_BASE_URL",
            "RAILWAY_PUBLIC_DOMAIN",
        ):
            environment.pop(key, None)
        environment.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(port),
                "OPENTULPA_DATA_ROOT": str(self._data_root),
                "OPENAI_COMPATIBLE_API_KEY": config.api_key.get_secret_value(),
                "OPENAI_COMPATIBLE_BASE_URL": config.base_url,
                "LLM_MODEL": config.model,
                "OPENTULPA_WEB_TOKEN": config.internal_runtime_token.get_secret_value(),
                "OPENTULPA_OWNER_CUSTOMER_ID": "owner",
                "OPENTULPA_INTERNAL_AGENT_API_URL": f"http://127.0.0.1:{port}",
                "OPENTULPA_DYNAMIC_HOST": "1",
            }
        )
        if config.telegram_pairing_code is not None:
            environment["OPENTULPA_TELEGRAM_PAIRING_CODE"] = (
                config.telegram_pairing_code.get_secret_value()
            )
        return environment

    def _seed_telegram_identity(self, config: HostConfig) -> None:
        if config.telegram_user_id is None:
            return
        from opentulpa.capability_workers.state import TelegramWorkerState

        state_path = self._telegram_state_path()
        state = TelegramWorkerState(state_path)
        existing = state.paired_identity()
        identity = (config.telegram_user_id, config.telegram_user_id)
        if existing is not None and existing != identity:
            raise RuntimeUnavailableError(
                "Telegram is paired to another owner; disconnect it before changing owner ID"
            )
        state.pair(user_id=identity[0], chat_id=identity[1])

    def _telegram_state_path(self) -> Path:
        return (
            self._data_root
            / "deepagents"
            / "capability_state"
            / tenant_namespace_label("owner")
            / "telegram.json"
        )

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _safe_error(error: Exception) -> str:
        text = str(error or "runtime failed").strip()
        return _SECRET_LINE.sub(r"\1\2[redacted]", text)[:1_000]


__all__ = ["RuntimeLogEntry", "RuntimeSupervisor", "RuntimeUnavailableError"]
