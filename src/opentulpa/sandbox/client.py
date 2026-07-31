"""Host-side client for the mandatory OpenTulpa sandbox worker."""

from __future__ import annotations

import base64
import binascii
import io
import tarfile
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
from deepagents.backends.protocol import ExecuteResponse
from pydantic import BaseModel, ConfigDict, Field


class SandboxWorkerClientError(RuntimeError):
    """Sanitized worker-client failure."""


class SandboxWorkerHealth(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    tier: str
    checks: dict[str, bool]
    error: str | None = None


class SandboxWorkerCanary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ok: bool
    step: str
    health: SandboxWorkerHealth | None = None
    error: str | None = None


class SandboxSecretFileMount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)
    content: str = Field(min_length=1, max_length=200_000)
    env: str | None = Field(default=None, max_length=80)


@dataclass(frozen=True, slots=True)
class _WorkspaceBinding:
    workspace_id: str
    kind: str


class SandboxWorkerExecutionProvider:
    """Execution provider backed by the private sandbox worker API."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        max_response_bytes: int,
        max_archive_bytes: int,
        max_archive_entries: int,
        max_file_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cleaned_url = str(base_url or "").strip().rstrip("/")
        parsed = httpx.URL(cleaned_url)
        if parsed.scheme != "http" or not parsed.host:
            raise ValueError("sandbox worker URL must be an internal HTTP URL")
        safe_token = str(token or "").strip()
        if len(safe_token) < 32:
            raise ValueError("sandbox worker token must contain at least 32 characters")
        if max_response_bytes < 4_096:
            raise ValueError("sandbox worker response limit is invalid")
        if max_archive_bytes < 1_024 or max_archive_entries < 1 or max_file_bytes < 1:
            raise ValueError("sandbox worker archive limits are invalid")
        self._base_url = cleaned_url
        self._headers = {"X-OpenTulpa-Sandbox-Token": safe_token}
        self._max_response_bytes = max_response_bytes
        self._max_archive_bytes = max_archive_bytes
        self._max_archive_entries = max_archive_entries
        self._max_file_bytes = max_file_bytes
        self._transport = transport
        self._bindings: dict[str, _WorkspaceBinding] = {}
        self._bindings_lock = threading.Lock()

    def health(self) -> SandboxWorkerHealth:
        try:
            payload = self._request("GET", "/health", timeout=5).json()
            return SandboxWorkerHealth.model_validate(payload)
        except Exception as exc:
            return SandboxWorkerHealth(
                ok=False,
                tier="unavailable",
                checks={"reachable": False},
                error=_safe_error(exc),
            )

    def canary(self) -> SandboxWorkerCanary:
        health = self.health()
        if not health.ok:
            return SandboxWorkerCanary(ok=False, step="health", health=health, error=health.error)
        workspace_id: str | None = None
        try:
            workspace_id = self._create_workspace(kind="scratch", tenant_id="canary")
            result = self._execute_workspace(
                workspace_id=workspace_id,
                command="printf opentulpa-sandbox-canary",
                timeout=5,
                secret_files=(),
            )
            if result.exit_code != 0 or result.output.strip() != "opentulpa-sandbox-canary":
                return SandboxWorkerCanary(
                    ok=False,
                    step="execute",
                    health=health,
                    error="sandbox canary command failed",
                )
            ssh = self._execute_workspace(
                workspace_id=workspace_id,
                command="ssh -V",
                timeout=5,
                secret_files=(),
            )
            if ssh.exit_code != 0:
                return SandboxWorkerCanary(
                    ok=False,
                    step="ssh",
                    health=health,
                    error="sandbox worker does not provide ssh",
                )
            return SandboxWorkerCanary(ok=True, step="ready", health=health)
        except Exception as exc:
            return SandboxWorkerCanary(
                ok=False,
                step="workspace",
                health=health,
                error=_safe_error(exc),
            )
        finally:
            if workspace_id is not None:
                with suppress(SandboxWorkerClientError):
                    self._delete_workspace(workspace_id)

    def execute(
        self,
        *,
        tenant_id: str,
        command: str,
        timeout: int,
        workspace: Path | None = None,
        cancel_event: threading.Event | None = None,
        secret_files: tuple[SandboxSecretFileMount, ...] = (),
    ) -> ExecuteResponse:
        if cancel_event is not None and cancel_event.is_set():
            return ExecuteResponse(output="sandbox execution was cancelled", exit_code=130, truncated=False)
        if workspace is None:
            workspace_id = self._create_workspace(kind="scratch", tenant_id=tenant_id)
            try:
                return self._execute_workspace(
                    workspace_id=workspace_id,
                    command=command,
                    timeout=timeout,
                    secret_files=secret_files,
                )
            finally:
                self._delete_workspace(workspace_id)
        binding = self._binding_for(tenant_id=tenant_id, kind="scratch")
        self._upload_archive(binding.workspace_id, workspace)
        result = self._execute_workspace(
            workspace_id=binding.workspace_id,
            command=command,
            timeout=timeout,
            secret_files=secret_files,
        )
        self._download_archive(binding.workspace_id, workspace)
        return result

    def _binding_for(self, *, tenant_id: str, kind: str) -> _WorkspaceBinding:
        safe_tenant = str(tenant_id or "").strip()
        if not safe_tenant:
            raise ValueError("tenant_id is required")
        key = f"{kind}:{safe_tenant}"
        with self._bindings_lock:
            existing = self._bindings.get(key)
            if existing is not None:
                return existing
            workspace_id = self._create_workspace(kind=kind, tenant_id=safe_tenant)
            binding = _WorkspaceBinding(workspace_id=workspace_id, kind=kind)
            self._bindings[key] = binding
            return binding

    def _create_workspace(self, *, kind: str, tenant_id: str) -> str:
        response = self._request(
            "POST",
            "/workspaces",
            json={"kind": kind, "tenant_id": tenant_id},
            timeout=10,
        )
        value = str(response.json().get("workspace_id") or "").strip()
        if not value:
            raise SandboxWorkerClientError("sandbox worker returned no workspace id")
        return value

    def _delete_workspace(self, workspace_id: str) -> None:
        self._request("DELETE", f"/workspaces/{workspace_id}", timeout=10)

    def _execute_workspace(
        self,
        *,
        workspace_id: str,
        command: str,
        timeout: int,
        secret_files: tuple[SandboxSecretFileMount, ...],
    ) -> ExecuteResponse:
        response = self._request(
            "POST",
            f"/workspaces/{workspace_id}/execute",
            json={
                "command": command,
                "timeout": timeout,
                "secret_files": [item.model_dump(mode="json") for item in secret_files],
            },
            timeout=timeout + 10,
        )
        payload = response.json()
        return ExecuteResponse(
            output=str(payload.get("output") or ""),
            exit_code=int(payload.get("exit_code") or 0),
            truncated=bool(payload.get("truncated")),
        )

    def _upload_archive(self, workspace_id: str, workspace: Path) -> None:
        encoded = self._archive_workspace(workspace)
        self._request(
            "POST",
            f"/workspaces/{workspace_id}/archive",
            json={"archive": encoded},
            timeout=30,
        )

    def _download_archive(self, workspace_id: str, workspace: Path) -> None:
        response = self._request("GET", f"/workspaces/{workspace_id}/archive", timeout=30)
        encoded = str(response.json().get("archive") or "")
        self._replace_workspace(workspace, encoded)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        timeout: float,
    ) -> httpx.Response:
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=timeout,
                trust_env=False,
                transport=self._transport,
            ) as client:
                response = client.request(
                    method,
                    f"{self._base_url}{path}",
                    headers=self._headers,
                    json=json,
                )
        except httpx.HTTPError as exc:
            raise SandboxWorkerClientError("sandbox worker is unavailable") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise SandboxWorkerClientError("sandbox worker rejected the request")
        if len(response.content) > self._max_response_bytes:
            raise SandboxWorkerClientError("sandbox worker response exceeded its limit")
        return response

    def _archive_workspace(self, workspace: Path) -> str:
        root = workspace.expanduser().resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise SandboxWorkerClientError("sandbox workspace is invalid")
        with io.BytesIO() as buffer:
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                archive.add(root, arcname=".", recursive=True)
            raw = buffer.getvalue()
        if len(raw) > self._max_archive_bytes:
            raise SandboxWorkerClientError("sandbox workspace archive exceeds its limit")
        return base64.b64encode(raw).decode("ascii")

    def _replace_workspace(self, workspace: Path, encoded_archive: str) -> None:
        try:
            raw = base64.b64decode(encoded_archive, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SandboxWorkerClientError("sandbox workspace archive is invalid") from exc
        if len(raw) > self._max_archive_bytes:
            raise SandboxWorkerClientError("sandbox workspace archive exceeds its limit")
        root = workspace.expanduser().resolve(strict=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
            members = archive.getmembers()
            if len(members) > self._max_archive_entries:
                raise SandboxWorkerClientError("sandbox workspace archive entry limit exceeded")
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise SandboxWorkerClientError("sandbox workspace archive path is invalid")
                if not (member.isdir() or member.isreg()):
                    raise SandboxWorkerClientError(
                        "sandbox workspace archive contains unsupported entries"
                    )
                if member.isreg() and member.size > self._max_file_bytes:
                    raise SandboxWorkerClientError("sandbox workspace archive file limit exceeded")
            for child in root.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
            archive.extractall(root, filter="data")


def _safe_error(error: Exception) -> str:
    return str(error or "sandbox worker failed").strip()[:500]


__all__ = [
    "SandboxSecretFileMount",
    "SandboxWorkerCanary",
    "SandboxWorkerClientError",
    "SandboxWorkerExecutionProvider",
    "SandboxWorkerHealth",
]
