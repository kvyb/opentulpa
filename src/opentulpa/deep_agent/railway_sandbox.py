"""Railway-hosted tenant execution with transactional workspace synchronization."""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any

from deepagents.backends.protocol import ExecuteResponse

BridgeRunner = Callable[[dict[str, Any], int], dict[str, Any]]


class RailwaySandboxExecutionError(RuntimeError):
    """Sanitized failure from the local Railway SDK bridge."""


class RailwaySandboxExecutionProvider:
    """Execute commands in Railway VMs without exposing Railway authority to the model."""

    def __init__(
        self,
        *,
        token: str,
        environment_id: str,
        max_output_bytes: int,
        max_workspace_archive_bytes: int,
        max_workspace_entries: int,
        max_file_bytes: int,
        idle_timeout_minutes: int = 30,
        bridge_path: str | Path | None = None,
        node_binary: str = "node",
        runner: BridgeRunner | None = None,
    ) -> None:
        safe_token = str(token or "").strip()
        safe_environment = str(environment_id or "").strip()
        if not safe_token or not safe_environment:
            raise ValueError("Railway sandbox credentials are incomplete")
        if max_output_bytes < 1_024:
            raise ValueError("Railway sandbox output limit is invalid")
        if max_workspace_archive_bytes < 1_024:
            raise ValueError("Railway sandbox workspace archive limit is invalid")
        if max_workspace_entries < 1 or max_file_bytes < 1:
            raise ValueError("Railway sandbox workspace limits are invalid")
        if not 1 <= idle_timeout_minutes <= 120:
            raise ValueError("Railway sandbox idle timeout is invalid")
        default_bridge = (
            Path(__file__).resolve().parents[3]
            / "railway_sandbox_bridge"
            / "bridge.mjs"
        )
        self._token = safe_token
        self._environment_id = safe_environment
        self._max_output_bytes = int(max_output_bytes)
        self._max_workspace_archive_bytes = int(max_workspace_archive_bytes)
        self._max_workspace_entries = int(max_workspace_entries)
        self._max_file_bytes = int(max_file_bytes)
        self._idle_timeout_minutes = int(idle_timeout_minutes)
        self._bridge_path = Path(bridge_path or default_bridge).expanduser().resolve()
        self._node_binary = str(node_binary or "node").strip() or "node"
        self._runner = runner or self._run_bridge
        self._sandbox_ids: dict[str, str] = {}
        self._tenant_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _tenant_lock(self, tenant_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._tenant_locks.setdefault(tenant_id, threading.Lock())

    def _run_bridge(self, request: dict[str, Any], timeout: int) -> dict[str, Any]:
        if not self._bridge_path.is_file():
            raise RailwaySandboxExecutionError("Railway sandbox bridge is unavailable")
        environment = os.environ.copy()
        environment["RAILWAY_TOKEN"] = self._token
        environment["RAILWAY_ENVIRONMENT_ID"] = self._environment_id
        try:
            completed = subprocess.run(
                [self._node_binary, str(self._bridge_path)],
                input=json.dumps(request, separators=(",", ":")).encode(),
                capture_output=True,
                check=False,
                timeout=timeout + 300,
                env=environment,
                cwd=self._bridge_path.parent,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RailwaySandboxExecutionError(
                "Railway sandbox bridge is unavailable"
            ) from exc
        response_limit = (
            self._max_workspace_archive_bytes * 2
            + self._max_output_bytes
            + 1_000_000
        )
        if len(completed.stdout) > response_limit:
            raise RailwaySandboxExecutionError(
                "Railway sandbox bridge response exceeded its limit"
            )
        try:
            result = json.loads(completed.stdout)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RailwaySandboxExecutionError(
                "Railway sandbox bridge returned an invalid response"
            ) from exc
        if completed.returncode != 0 or not isinstance(result, dict) or not result.get("ok"):
            raise RailwaySandboxExecutionError("Railway sandbox execution failed")
        return result

    def _archive_workspace(self, workspace: Path) -> str:
        with tempfile.SpooledTemporaryFile(max_size=1_000_000) as archive_file:
            with tarfile.open(fileobj=archive_file, mode="w:gz") as archive:
                archive.add(workspace, arcname=".", recursive=True)
            size = archive_file.tell()
            if size > self._max_workspace_archive_bytes:
                raise RailwaySandboxExecutionError(
                    "tenant workspace exceeds Railway sandbox synchronization limit"
                )
            archive_file.seek(0)
            return base64.b64encode(archive_file.read()).decode("ascii")

    def _replace_workspace(self, workspace: Path, encoded_archive: str) -> None:
        try:
            raw = base64.b64decode(encoded_archive, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RailwaySandboxExecutionError(
                "Railway sandbox returned an invalid workspace archive"
            ) from exc
        if len(raw) > self._max_workspace_archive_bytes:
            raise RailwaySandboxExecutionError(
                "Railway sandbox workspace archive exceeded its limit"
            )
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
                members = archive.getmembers()
                if len(members) > self._max_workspace_entries:
                    raise RailwaySandboxExecutionError(
                        "Railway sandbox workspace entry limit exceeded"
                    )
                total_size = 0
                for member in members:
                    path = PurePosixPath(member.name)
                    if path.is_absolute() or ".." in path.parts:
                        raise RailwaySandboxExecutionError(
                            "Railway sandbox workspace path is invalid"
                        )
                    if not (member.isdir() or member.isreg()):
                        raise RailwaySandboxExecutionError(
                            "Railway sandbox workspace contains an unsupported entry"
                        )
                    if member.isreg() and member.size > self._max_file_bytes:
                        raise RailwaySandboxExecutionError(
                            "Railway sandbox workspace file limit exceeded"
                        )
                    total_size += member.size
                    if total_size > self._max_workspace_archive_bytes * 8:
                        raise RailwaySandboxExecutionError(
                            "Railway sandbox workspace expanded beyond its limit"
                        )
                if workspace.is_symlink():
                    raise RailwaySandboxExecutionError(
                        "tenant workspace cannot be a symlink"
                    )
                shutil.rmtree(workspace)
                workspace.mkdir(mode=0o700, parents=True)
                archive.extractall(workspace, filter="data")
        except (OSError, tarfile.TarError) as exc:
            raise RailwaySandboxExecutionError(
                "Railway sandbox workspace could not be synchronized"
            ) from exc

    def execute(
        self,
        *,
        tenant_id: str,
        command: str,
        timeout: int,
        workspace: Path | None = None,
    ) -> ExecuteResponse:
        safe_tenant = str(tenant_id or "").strip()
        if not safe_tenant:
            raise ValueError("tenant_id is required")
        with self._tenant_lock(safe_tenant):
            request: dict[str, Any] = {
                "environmentId": self._environment_id,
                "sandboxId": self._sandbox_ids.get(safe_tenant),
                "command": command,
                "timeoutSec": max(1, int(timeout)),
                "idleTimeoutMinutes": self._idle_timeout_minutes,
                "maxOutputBytes": self._max_output_bytes,
                "maxWorkspaceArchiveBytes": self._max_workspace_archive_bytes,
                "workspaceArchive": (
                    self._archive_workspace(workspace) if workspace is not None else None
                ),
            }
            result = self._runner(request, max(1, int(timeout)))
            sandbox_id = str(result.get("sandboxId") or "").strip()
            if not sandbox_id:
                raise RailwaySandboxExecutionError(
                    "Railway sandbox response is missing its identity"
                )
            self._sandbox_ids[safe_tenant] = sandbox_id
            if workspace is not None:
                synchronized = result.get("workspaceSynchronized")
                if not isinstance(synchronized, bool):
                    raise RailwaySandboxExecutionError(
                        "Railway sandbox response is missing its workspace state"
                    )
                if synchronized:
                    encoded_archive = result.get("workspaceArchive")
                    if not isinstance(encoded_archive, str) or not encoded_archive:
                        raise RailwaySandboxExecutionError(
                            "Railway sandbox response is missing its workspace"
                        )
                    self._replace_workspace(workspace, encoded_archive)
            output = result.get("output")
            exit_code = result.get("exitCode")
            truncated = result.get("truncated")
            if (
                not isinstance(output, str)
                or not isinstance(exit_code, int)
                or not isinstance(truncated, bool)
            ):
                raise RailwaySandboxExecutionError(
                    "Railway sandbox returned an invalid execution result"
                )
            return ExecuteResponse(
                output=output,
                exit_code=exit_code,
                truncated=truncated,
            )


__all__ = [
    "RailwaySandboxExecutionError",
    "RailwaySandboxExecutionProvider",
]
