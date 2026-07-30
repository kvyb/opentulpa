"""Lease-bound tenant command execution owned by the stable bootstrap host."""

from __future__ import annotations

import asyncio
import hmac
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

import httpx
from deepagents.backends.protocol import ExecuteResponse
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.deep_agent.sandbox import TenantContainerBackend, TenantContainerPolicy


class SandboxExecutionError(RuntimeError):
    """Sanitized failure across the private stable-host sandbox boundary."""


class _SandboxModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SandboxExecuteRequest(_SandboxModel):
    tenant_id: str = Field(min_length=1, max_length=512)
    command: str = Field(min_length=1, max_length=100_000)
    timeout: int = Field(ge=1, le=3_600)


class SandboxExecuteResult(_SandboxModel):
    output: str
    exit_code: int
    truncated: bool


class SandboxExecutionLease(_SandboxModel):
    release_id: str = Field(min_length=1, max_length=100)
    lease_epoch: int = Field(ge=1)


class TenantSandboxExecutionService:
    """Derive tenant roots and execute only through the fixed OCI sandbox policy."""

    def __init__(
        self,
        *,
        workspaces_root: Path,
        allowed_root: Path,
        policy: TenantContainerPolicy,
        container_cli: str,
    ) -> None:
        allowed = allowed_root.expanduser().resolve(strict=True)
        configured = workspaces_root.expanduser()
        if configured.is_symlink():
            raise ValueError("tenant workspaces root cannot be a symlink")
        configured.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = configured.resolve(strict=True)
        if resolved == allowed or not resolved.is_relative_to(allowed):
            raise ValueError("tenant workspaces root escaped release storage")
        self._allowed_root = allowed
        self._workspaces_root = resolved
        self._policy = policy
        self._transition_timeout_seconds = (
            policy.timeout_seconds + policy.cleanup_timeout_seconds
        )
        self._container_cli = str(container_cli or "").strip()
        if not self._container_cli:
            raise ValueError("container_cli is required")
        self._active_lease: SandboxExecutionLease | None = None
        self._admissions_closed = False
        self._inflight: set[asyncio.Task[Any]] = set()
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._lifecycle_lock = asyncio.Lock()
        self._commit_lock = threading.Lock()

    async def reconcile_lease(self, lease: SandboxExecutionLease | None) -> None:
        """Fence admitted commands before changing the exact serving lease."""

        async with self._lifecycle_lock:
            async with self._condition:
                if lease == self._active_lease and not self._admissions_closed:
                    return
                self._admissions_closed = True
                inflight = tuple(self._inflight)

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self._transition_timeout_seconds
            while not self._commit_lock.acquire(blocking=False):
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise SandboxExecutionError(
                        "sandbox lease transition could not fence workspace commits"
                    )
                await asyncio.sleep(min(0.01, remaining))
            try:
                self._active_lease = lease
                for task in inflight:
                    task.cancel()
            finally:
                self._commit_lock.release()

            async with self._condition:
                if self._inflight:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise SandboxExecutionError(
                            "sandbox lease transition did not drain admitted executions"
                        )
                    try:
                        await asyncio.wait_for(
                            self._condition.wait_for(lambda: not self._inflight),
                            timeout=remaining,
                        )
                    except TimeoutError as exc:
                        raise SandboxExecutionError(
                            "sandbox lease transition did not drain admitted executions"
                        ) from exc
                self._admissions_closed = False

    async def execute(
        self,
        *,
        lease: SandboxExecutionLease,
        tenant_id: str,
        command: str,
        timeout: int,
    ) -> ExecuteResponse:
        if (
            self._allowed_root.is_symlink()
            or self._workspaces_root.is_symlink()
            or self._workspaces_root.resolve(strict=True) != self._workspaces_root
            or not self._workspaces_root.is_relative_to(self._allowed_root)
        ):
            raise SandboxExecutionError("tenant workspaces root failed validation")
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - an async service always has a task
            raise SandboxExecutionError("sandbox execution task is unavailable")
        async with self._condition:
            if self._admissions_closed or self._active_lease != lease:
                raise SandboxExecutionError("sandbox execution lease changed")
            self._inflight.add(task)
        try:
            backend = TenantContainerBackend(
                tenant_id=tenant_id,
                workspaces_root=self._workspaces_root,
                policy=self._policy,
                container_cli=self._container_cli,
                persistent_workspace=True,
                commit_authority=lambda: self._authorize_commit(lease),
            )
            return await backend.aexecute(command, timeout=timeout)
        except RuntimeError as exc:
            raise SandboxExecutionError("tenant workspace recovery failed") from exc
        finally:
            async with self._condition:
                self._inflight.discard(task)
                self._condition.notify_all()

    @contextmanager
    def _authorize_commit(self, lease: SandboxExecutionLease) -> Iterator[None]:
        with self._commit_lock:
            if self._admissions_closed or self._active_lease != lease:
                raise SandboxExecutionError("sandbox execution lease changed before commit")
            yield


LeaseAuthorizer = Callable[[str, int, str], Awaitable[None]]


def register_sandbox_execution_api(
    app: FastAPI,
    *,
    service: TenantSandboxExecutionService,
    token: str,
    authorize_lease: LeaseAuthorizer,
    prefix: str = "/bootstrap/internal/v1/sandbox",
) -> None:
    """Register a private command-only endpoint; no OCI or path primitive escapes."""

    expected_token = str(token or "").strip()
    if len(expected_token) < 32:
        raise ValueError("sandbox execution token must contain at least 32 characters")

    async def authorize(
        request: Request,
        supplied: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Sandbox-Token", max_length=500),
        ] = None,
        release_id: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Release-ID", max_length=200),
        ] = None,
        lease_epoch: Annotated[
            int | None,
            Header(alias="X-OpenTulpa-Lease-Epoch", ge=1),
        ] = None,
        control_token: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Control-Token", max_length=500),
        ] = None,
    ) -> None:
        if not hmac.compare_digest(str(supplied or ""), expected_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="valid sandbox execution credentials are required",
            )
        if lease_epoch is None:
            raise HTTPException(status_code=401, detail="active release credentials are required")
        await authorize_lease(
            str(release_id or ""),
            lease_epoch,
            str(control_token or ""),
        )
        request.state.sandbox_execution_lease = SandboxExecutionLease(
            release_id=str(release_id or ""),
            lease_epoch=lease_epoch,
        )

    router = APIRouter(prefix=prefix, dependencies=[Depends(authorize)], include_in_schema=False)

    @router.post("/execute", response_model=SandboxExecuteResult)
    async def execute(body: SandboxExecuteRequest, request: Request) -> dict[str, Any]:
        try:
            result = await service.execute(
                lease=request.state.sandbox_execution_lease,
                tenant_id=body.tenant_id,
                command=body.command,
                timeout=body.timeout,
            )
        except (OSError, SandboxExecutionError, ValueError) as exc:
            raise HTTPException(
                status_code=503,
                detail="tenant sandbox execution is unavailable",
            ) from exc
        return {
            "output": result.output,
            "exit_code": result.exit_code,
            "truncated": result.truncated,
        }

    app.include_router(router)


class SandboxExecutionClient:
    """Private mutable-release adapter; it exposes no host or OCI configuration."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        release_id: str,
        lease_epoch: int,
        control_token: str,
        max_response_bytes: int,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        cleaned_url = str(base_url or "").strip().rstrip("/")
        parsed = urlsplit(cleaned_url)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("sandbox execution URL must be an authenticated HTTP endpoint")
        safe_token = str(token or "").strip()
        safe_release = str(release_id or "").strip()
        safe_control = str(control_token or "").strip()
        if len(safe_token) < 32 or not safe_release or lease_epoch < 1 or len(safe_control) < 32:
            raise ValueError("sandbox execution credentials are invalid")
        if not 4_096 <= max_response_bytes <= 20_000_000:
            raise ValueError("sandbox execution response limit is invalid")
        self._base_url = cleaned_url
        self._headers = {
            "X-OpenTulpa-Sandbox-Token": safe_token,
            "X-OpenTulpa-Release-ID": safe_release,
            "X-OpenTulpa-Lease-Epoch": str(lease_epoch),
            "X-OpenTulpa-Control-Token": safe_control,
        }
        self._max_response_bytes = max_response_bytes
        self._transport = transport

    def execute(
        self,
        *,
        tenant_id: str,
        command: str,
        timeout: int,
        workspace: Path | None = None,
    ) -> ExecuteResponse:
        del workspace
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=httpx.Timeout(connect=5.0, read=timeout + 5.0, write=10.0, pool=5.0),
                trust_env=False,
                transport=self._transport,
            ) as client, client.stream(
                "POST",
                f"{self._base_url}/execute",
                headers=self._headers,
                json={"tenant_id": tenant_id, "command": command, "timeout": timeout},
            ) as response:
                if response.status_code != 200:
                    raise SandboxExecutionError("sandbox execution service rejected the request")
                raw = bytearray()
                for chunk in response.iter_bytes():
                    if len(raw) + len(chunk) > self._max_response_bytes:
                        raise SandboxExecutionError("sandbox execution response exceeded its limit")
                    raw.extend(chunk)
        except httpx.HTTPError as exc:
            raise SandboxExecutionError("sandbox execution service is unavailable") from exc
        try:
            result = SandboxExecuteResult.model_validate_json(raw)
        except ValueError as exc:
            raise SandboxExecutionError("sandbox execution service returned an invalid response") from exc
        return ExecuteResponse(
            output=result.output,
            exit_code=result.exit_code,
            truncated=result.truncated,
        )


__all__ = [
    "SandboxExecutionClient",
    "SandboxExecutionError",
    "SandboxExecutionLease",
    "TenantSandboxExecutionService",
    "register_sandbox_execution_api",
]
