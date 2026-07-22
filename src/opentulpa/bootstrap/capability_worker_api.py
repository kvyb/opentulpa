"""Lease-fenced capability worker authority owned by the stable bootstrap."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from opentulpa.bootstrap.models import ReleaseRecord
from opentulpa.capabilities.models import (
    CapabilityManifest,
    WorkerRuntime,
    WorkerSpec,
    WorkerTransport,
    is_reserved_worker_environment_name,
)
from opentulpa.capabilities.oci_workers import OciCapabilityWorkerHost
from opentulpa.capabilities.workers import (
    WorkerHandle,
    WorkerHost,
    WorkerLaunch,
    WorkerLifecycleError,
)

_WORKER_MODULE = re.compile(r"opentulpa\.capability_workers(?:\.[a-z][a-z0-9_]*)*\Z")
_HANDLE = re.compile(r"capworker_[0-9a-f]{32}\Z")
_OCI_HANDLE = re.compile(r"oci:([0-9a-f]{12,64})\Z")
_MAX_STATE_BYTES = 16 * 1024 * 1024
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class CapabilityWorkerAPIError(RuntimeError):
    """Sanitized failure across the stable capability-worker boundary."""


class _APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CapabilityWorkerLease(_APIModel):
    release_id: str = Field(min_length=1, max_length=100)
    lease_epoch: int = Field(ge=1)


class CapabilityWorkerStartRequest(_APIModel):
    tenant_id: str = Field(min_length=1, max_length=512)
    instance_id: str = Field(min_length=1, max_length=300)
    manifest: CapabilityManifest
    worker: WorkerSpec
    config: dict[str, JsonValue] = Field(default_factory=dict)
    secret_environment: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("config")
    @classmethod
    def bounded_config(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > 1_000_000:
            raise ValueError("capability config exceeded its byte limit")
        return value

    @field_validator("secret_environment")
    @classmethod
    def bounded_secrets(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 100:
            raise ValueError("too many capability secret grants")
        total = 0
        for name, secret in value.items():
            if re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name) is None:
                raise ValueError("capability secret name is invalid")
            if is_reserved_worker_environment_name(name):
                raise ValueError("capability secret cannot override worker runtime environment")
            size = len(secret.encode("utf-8"))
            if size > 65_536:
                raise ValueError("capability secret exceeded its byte limit")
            total += size
        if total > 1_000_000:
            raise ValueError("capability secret grants exceeded their byte limit")
        return value


class CapabilityWorkerResult(_APIModel):
    handle_id: str = Field(pattern=r"^capworker_[0-9a-f]{32}$")
    instance_id: str = Field(min_length=1, max_length=300)
    capability_name: str = Field(min_length=1, max_length=64)
    capability_revision: int = Field(ge=1)
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    worker_name: str = Field(min_length=1, max_length=100)
    endpoint_available: bool = False


class CapabilityWorkerHealthResult(_APIModel):
    healthy: bool


class CapabilityWorkerFenceRequest(_APIModel):
    tenant_id: str = Field(min_length=1, max_length=512)
    capability_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")


class _PersistedHandle(_APIModel):
    id: str
    instance_id: str
    capability_name: str
    capability_revision: int
    manifest_digest: str
    worker_name: str
    endpoint: str | None = None

    def to_handle(self) -> WorkerHandle:
        return WorkerHandle(**self.model_dump(), endpoint_headers={})


class _PersistedWorker(_APIModel):
    external_id: str = Field(pattern=r"^capworker_[0-9a-f]{32}$")
    release_id: str = Field(min_length=1, max_length=100)
    lease_epoch: int = Field(ge=1)
    tenant_id: str = Field(min_length=1, max_length=512)
    original_manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    handle: _PersistedHandle
    worker: WorkerSpec
    ready_path: str | None = None


class _PersistedState(_APIModel):
    version: int = Field(default=1, ge=1, le=1)
    workers: tuple[_PersistedWorker, ...] = ()


@dataclass(slots=True)
class _ProxyOperation:
    deadline: float
    upstream: httpx.Response | None = None
    stream_task: asyncio.Task[Any] | None = None
    released: bool = False


ReleaseLoader = Callable[[str], ReleaseRecord | None]
LeaseAuthorizer = Callable[[str, int, str], Awaitable[None]]


class StableCapabilityWorkerService:
    """Run approved release modules in narrow, state-only OCI containers.

    The active release selects a reviewed module and supplies tenant-scoped config and
    credentials. The stable host, not the release, selects the image, mounts, runtime
    user, resources, and network. No source tree, product database, host secret file,
    container socket, or product ``/workspace`` is mounted into a capability worker.
    """

    def __init__(
        self,
        *,
        host: OciCapabilityWorkerHost,
        release_loader: ReleaseLoader,
        state_path: Path,
        max_proxy_request_bytes: int = 2 * 1024 * 1024,
        max_proxy_response_bytes: int = 8 * 1024 * 1024,
        proxy_timeout_seconds: float = 120,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if not 1_024 <= max_proxy_request_bytes <= 20_000_000:
            raise ValueError("capability proxy request limit is invalid")
        if not 1_024 <= max_proxy_response_bytes <= 100_000_000:
            raise ValueError("capability proxy response limit is invalid")
        if not 1 <= proxy_timeout_seconds <= 3_600:
            raise ValueError("capability proxy timeout is invalid")
        self._host = host
        self._release_loader = release_loader
        configured = state_path.expanduser()
        if configured.is_symlink():
            raise ValueError("capability worker state cannot be a symlink")
        configured.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if configured.parent.is_symlink():
            raise ValueError("capability worker state directory cannot be a symlink")
        self._state_path = configured.resolve()
        self._max_proxy_request_bytes = max_proxy_request_bytes
        self._max_proxy_response_bytes = max_proxy_response_bytes
        self._proxy_timeout_seconds = proxy_timeout_seconds
        self._http = http_client or httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                connect=5,
                read=proxy_timeout_seconds,
                write=30,
                pool=5,
            ),
        )
        self._owns_http = http_client is None
        self._records = self._load_state()
        self._adopted: set[str] = set()
        self._provisional: set[str] = set()
        self._dirty = False
        self._active_lease: CapabilityWorkerLease | None = None
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._lifecycle_lock = asyncio.Lock()
        self._admissions_closed = False
        self._inflight_proxy_operations = 0
        self._proxy_operations: dict[str, _ProxyOperation] = {}

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()
        await self._host.aclose()

    async def start(
        self,
        *,
        lease: CapabilityWorkerLease,
        request: CapabilityWorkerStartRequest,
    ) -> CapabilityWorkerResult:
        release = self._release_loader(lease.release_id)
        if release is None:
            raise CapabilityWorkerAPIError("active release metadata is unavailable")
        worker = self._validate_source_worker(request)
        runtime_worker = worker.model_copy(
            update={
                "runtime": WorkerRuntime.OCI,
                "image": release.artifact_digest,
            }
        )
        runtime_manifest = request.manifest.model_copy(
            update={
                "artifact_digest": release.artifact_digest,
                "workers": tuple(
                    runtime_worker if item == worker else item for item in request.manifest.workers
                ),
            }
        )
        launch = WorkerLaunch(
            tenant_id=request.tenant_id,
            instance_id=request.instance_id,
            manifest=runtime_manifest,
            worker=runtime_worker,
            config=request.config,
            secret_environment=request.secret_environment,
        )
        async with self._exclusive_lifecycle_mutation(lease):
            stale = [
                record
                for record in self._records.values()
                if record.tenant_id == request.tenant_id
                and record.handle.capability_name == request.manifest.name
                and record.handle.worker_name == worker.name
            ]
            self._dirty = True
            for record in stale:
                await self._host.stop(record.handle.to_handle())
                self._records.pop(record.external_id, None)
                self._adopted.discard(record.external_id)
                self._provisional.discard(record.external_id)
            actual = await self._host.start(
                launch,
                release_id=lease.release_id,
                lease_epoch=lease.lease_epoch,
            )
            external_id = f"capworker_{uuid4().hex}"
            public_handle = _PersistedHandle(
                id=actual.id,
                instance_id=request.instance_id,
                capability_name=request.manifest.name,
                capability_revision=request.manifest.revision,
                manifest_digest=request.manifest.content_digest,
                worker_name=worker.name,
                endpoint=actual.endpoint,
            )
            ready_path = self._host.ready_path(actual)
            record = _PersistedWorker(
                external_id=external_id,
                release_id=lease.release_id,
                lease_epoch=lease.lease_epoch,
                tenant_id=request.tenant_id,
                original_manifest_digest=request.manifest.content_digest,
                handle=public_handle,
                worker=runtime_worker,
                ready_path=str(ready_path) if ready_path is not None else None,
            )
            self._records[external_id] = record
            self._adopted.add(external_id)
            self._provisional.add(external_id)
            try:
                self._write_state()
            except BaseException:
                self._dirty = True
                try:
                    await self._host.stop(actual)
                except BaseException as cleanup_error:
                    self._dirty = True
                    raise WorkerLifecycleError(
                        "capability worker persistence cleanup failed"
                    ) from cleanup_error
                self._records.pop(external_id, None)
                self._adopted.discard(external_id)
                self._provisional.discard(external_id)
                raise
            self._provisional.discard(external_id)
            self._dirty = False
        return self._result(record)

    async def reconcile_lease(self, lease: CapabilityWorkerLease | None) -> None:
        """Expose only containers bound to the exact current production lease."""

        async with self._lifecycle_lock, self._condition:
            self._admissions_closed = True
            try:
                await self._drain_proxy_operations_locked()
                await self._reconcile_lease_locked(lease)
            except BaseException:
                self._dirty = True
                raise
            finally:
                self._admissions_closed = False

    async def healthy(self, external_id: str) -> bool:
        record = self._record(external_id)
        return await self._host.healthy(record.handle.to_handle())

    async def stop(self, external_id: str, *, lease: CapabilityWorkerLease) -> None:
        async with self._exclusive_lifecycle_mutation(lease):
            record = self._records.get(external_id)
            if record is None:
                return
            await self._host.stop(record.handle.to_handle())
            self._records.pop(external_id, None)
            self._adopted.discard(external_id)
            self._provisional.discard(external_id)
            try:
                self._write_state()
            except BaseException:
                self._dirty = True
                raise

    async def fence(
        self,
        *,
        lease: CapabilityWorkerLease,
        tenant_id: str,
        capability_name: str,
    ) -> None:
        """Stop orphaned or active generations for one tenant capability."""

        async with self._exclusive_lifecycle_mutation(lease):
            matching = [
                record
                for record in self._records.values()
                if record.tenant_id == tenant_id
                and record.handle.capability_name == capability_name
            ]
            await self._host.fence(
                tenant_id=tenant_id,
                capability_name=capability_name,
            )
            for record in matching:
                self._records.pop(record.external_id, None)
                self._adopted.discard(record.external_id)
                self._provisional.discard(record.external_id)
            if matching:
                try:
                    self._write_state()
                except BaseException:
                    self._dirty = True
                    raise

    async def proxy(
        self,
        external_id: str,
        request: Request,
        *,
        lease: CapabilityWorkerLease,
        suffix: str,
    ) -> Response:
        async with self._condition:
            self._require_active_lease(lease)
            record = self._record(external_id)
            endpoint = record.handle.endpoint
            if endpoint is None:
                raise CapabilityWorkerAPIError("capability worker has no HTTP endpoint")
            operation_id = uuid4().hex
            operation = _ProxyOperation(
                deadline=asyncio.get_running_loop().time() + self._proxy_timeout_seconds
            )
            self._proxy_operations[operation_id] = operation
            self._inflight_proxy_operations += 1

        async def release_operation() -> None:
            async with self._condition:
                self._release_proxy_operation_locked(operation_id, operation)

        upstream: httpx.Response | None = None
        try:
            async with asyncio.timeout_at(operation.deadline):
                body = bytearray()
                async for chunk in request.stream():
                    if len(body) + len(chunk) > self._max_proxy_request_bytes:
                        raise CapabilityWorkerAPIError(
                            "capability worker request exceeded its limit"
                        )
                    body.extend(chunk)
                target = endpoint.rstrip("/")
                if suffix:
                    target = f"{target}/{suffix.lstrip('/')}"
                if request.url.query:
                    target = f"{target}?{request.url.query}"
                headers = {
                    name: value
                    for name, value in request.headers.items()
                    if name.casefold() not in _HOP_BY_HOP
                    and not name.casefold().startswith("x-opentulpa-")
                    and name.casefold() not in {"host", "content-length"}
                }
                upstream = await self._http.send(
                    self._http.build_request(
                        request.method,
                        target,
                        headers=headers,
                        content=bytes(body),
                        timeout=self._proxy_timeout_seconds,
                    ),
                    stream=True,
                )
            async with self._condition:
                if operation.released:
                    abandoned = True
                else:
                    operation.upstream = upstream
                    abandoned = False
            if abandoned:
                await self._close_proxy_responses_bounded((upstream,))
                raise CapabilityWorkerAPIError("capability worker lease changed")
        except TimeoutError as exc:
            try:
                if upstream is not None:
                    await self._close_proxy_responses_bounded((upstream,))
            finally:
                await release_operation()
            raise CapabilityWorkerAPIError(
                "capability worker request exceeded its time limit"
            ) from exc
        except httpx.HTTPError as exc:
            try:
                if upstream is not None:
                    await self._close_proxy_responses_bounded((upstream,))
            finally:
                await release_operation()
            raise CapabilityWorkerAPIError("capability worker endpoint is unavailable") from exc
        except BaseException:
            try:
                if upstream is not None:
                    await self._close_proxy_responses_bounded((upstream,))
            finally:
                await release_operation()
            raise
        try:
            response_headers = {
                name: value
                for name, value in upstream.headers.items()
                if name.casefold() not in _HOP_BY_HOP
                and name.casefold() not in {"content-length", "server"}
            }

            async def stream_body() -> AsyncIterator[bytes]:
                async with self._condition:
                    if operation.released:
                        raise CapabilityWorkerAPIError("capability worker lease changed")
                    operation.stream_task = asyncio.current_task()
                size = 0
                try:
                    async with asyncio.timeout_at(operation.deadline):
                        async for chunk in upstream.aiter_bytes():
                            size += len(chunk)
                            if size > self._max_proxy_response_bytes:
                                raise CapabilityWorkerAPIError(
                                    "capability worker response exceeded its limit"
                                )
                            yield chunk
                except TimeoutError as exc:
                    raise CapabilityWorkerAPIError(
                        "capability worker response exceeded its time limit"
                    ) from exc
                finally:
                    try:
                        await upstream.aclose()
                    finally:
                        await release_operation()

            return StreamingResponse(
                stream_body(),
                status_code=upstream.status_code,
                headers=response_headers,
            )
        except BaseException:
            try:
                await self._close_proxy_responses_bounded((upstream,))
            finally:
                await release_operation()
            raise

    def _validate_source_worker(self, request: CapabilityWorkerStartRequest) -> WorkerSpec:
        worker = request.worker
        if worker not in request.manifest.workers:
            raise CapabilityWorkerAPIError("worker is not part of the supplied manifest")
        if request.manifest.artifact_digest is not None:
            raise CapabilityWorkerAPIError("source capability cannot select an OCI artifact")
        if worker.runtime is not WorkerRuntime.SUBPROCESS or worker.image is not None:
            raise CapabilityWorkerAPIError("source capability must use the release runtime")
        if (
            len(worker.command) != 3
            or worker.command[:2] != ("python", "-m")
            or _WORKER_MODULE.fullmatch(worker.command[2]) is None
        ):
            raise CapabilityWorkerAPIError(
                "capability worker module is outside the reviewed package"
            )
        if worker.transport is WorkerTransport.STDIO and worker.kind.value == "mcp":
            raise CapabilityWorkerAPIError("managed MCP workers require authenticated HTTP")
        declared_secrets = {item.name for item in (*request.manifest.secrets, *worker.secrets)}
        if not set(request.secret_environment).issubset(declared_secrets):
            raise CapabilityWorkerAPIError("capability worker received an undeclared secret")
        return worker

    def _record(self, external_id: str) -> _PersistedWorker:
        if _HANDLE.fullmatch(external_id) is None:
            raise CapabilityWorkerAPIError("capability worker handle is invalid")
        record = self._records.get(external_id)
        active = self._active_lease
        if (
            record is None
            or self._dirty
            or active is None
            or record.release_id != active.release_id
            or record.lease_epoch != active.lease_epoch
            or external_id not in self._adopted
        ):
            raise CapabilityWorkerAPIError("capability worker was not found")
        return record

    async def _reconcile_lease_locked(self, lease: CapabilityWorkerLease | None) -> None:
        if lease == self._active_lease and not self._dirty:
            return
        exact = [
            record
            for record in self._records.values()
            if lease is not None
            and record.external_id not in self._provisional
            and record.release_id == lease.release_id
            and record.lease_epoch == lease.lease_epoch
        ]
        container_ids: dict[str, str] = {}
        for record in exact:
            match = _OCI_HANDLE.fullmatch(record.handle.id)
            if match is None:
                raise CapabilityWorkerAPIError("persisted capability worker handle is invalid")
            container_ids[record.external_id] = match.group(1)
        confirmed = set(
            await self._host.reconcile_managed_workers(
                release_id=lease.release_id if lease is not None else None,
                lease_epoch=lease.lease_epoch if lease is not None else None,
                keep_container_ids=tuple(container_ids.values()),
            )
        )
        retained = {
            external_id
            for external_id, container_id in container_ids.items()
            if container_id in confirmed
        }
        changed = False
        for external_id in tuple(self._records):
            if external_id in retained:
                continue
            self._records.pop(external_id, None)
            self._adopted.discard(external_id)
            self._provisional.discard(external_id)
            changed = True
        if changed or self._dirty:
            self._write_state()
        for external_id in retained:
            if external_id in self._adopted:
                continue
            record = self._records[external_id]
            self._host.adopt(
                handle=record.handle.to_handle(),
                worker=record.worker,
                ready_path=Path(record.ready_path) if record.ready_path else None,
                tenant_id=record.tenant_id,
                release_id=record.release_id,
                lease_epoch=record.lease_epoch,
            )
            self._adopted.add(external_id)
        self._active_lease = lease
        self._dirty = False

    async def _ensure_active_lease_locked(self, lease: CapabilityWorkerLease) -> None:
        if self._dirty and lease == self._active_lease:
            await self._reconcile_lease_locked(lease)
        if self._dirty or self._active_lease != lease:
            raise CapabilityWorkerAPIError("capability worker lease changed")

    async def _drain_proxy_operations_locked(self) -> None:
        if not self._proxy_operations:
            return
        deadline = max(operation.deadline for operation in self._proxy_operations.values())
        try:
            async with asyncio.timeout_at(deadline):
                await self._condition.wait_for(lambda: not self._proxy_operations)
            return
        except TimeoutError:
            pass

        current = asyncio.current_task()
        responses: list[httpx.Response] = []
        for operation_id, operation in tuple(self._proxy_operations.items()):
            task = operation.stream_task
            if task is not None and task is not current and not task.done():
                task.cancel()
            if operation.upstream is not None:
                responses.append(operation.upstream)
            self._release_proxy_operation_locked(operation_id, operation)
        await self._close_proxy_responses_bounded(tuple(responses))

    def _release_proxy_operation_locked(
        self,
        operation_id: str,
        operation: _ProxyOperation,
    ) -> None:
        if operation.released:
            return
        operation.released = True
        self._proxy_operations.pop(operation_id, None)
        self._inflight_proxy_operations -= 1
        self._condition.notify_all()

    async def _close_proxy_responses_bounded(
        self,
        responses: tuple[httpx.Response, ...],
    ) -> None:
        tasks = {asyncio.create_task(response.aclose()) for response in responses}
        if not tasks:
            return
        done, pending = await asyncio.wait(
            tasks,
            timeout=min(1.0, self._proxy_timeout_seconds),
        )
        for task in done:
            with suppress(BaseException):
                task.result()
        for task in pending:
            task.cancel()
            task.add_done_callback(self._consume_background_task_result)

    @staticmethod
    def _consume_background_task_result(task: asyncio.Task[None]) -> None:
        with suppress(BaseException):
            task.result()

    @asynccontextmanager
    async def _exclusive_lifecycle_mutation(
        self,
        lease: CapabilityWorkerLease,
    ) -> AsyncIterator[None]:
        async with self._lifecycle_lock, self._condition:
            if self._active_lease != lease:
                raise CapabilityWorkerAPIError("capability worker lease changed")
            self._admissions_closed = True
            try:
                await self._drain_proxy_operations_locked()
                await self._ensure_active_lease_locked(lease)
                yield
            finally:
                self._admissions_closed = False

    def _require_active_lease(self, lease: CapabilityWorkerLease) -> None:
        if self._admissions_closed or self._dirty or self._active_lease != lease:
            raise CapabilityWorkerAPIError("capability worker lease changed")

    @staticmethod
    def _result(record: _PersistedWorker) -> CapabilityWorkerResult:
        return CapabilityWorkerResult(
            handle_id=record.external_id,
            instance_id=record.handle.instance_id,
            capability_name=record.handle.capability_name,
            capability_revision=record.handle.capability_revision,
            manifest_digest=record.original_manifest_digest,
            worker_name=record.handle.worker_name,
            endpoint_available=record.handle.endpoint is not None,
        )

    def _load_state(self) -> dict[str, _PersistedWorker]:
        if not self._state_path.exists():
            return {}
        if (
            self._state_path.is_symlink()
            or not self._state_path.is_file()
            or self._state_path.stat().st_mode & 0o077
        ):
            raise ValueError("capability worker state must be a regular file")
        if self._state_path.stat().st_size > _MAX_STATE_BYTES:
            raise ValueError("capability worker state exceeded its byte limit")
        try:
            state = _PersistedState.model_validate_json(self._state_path.read_bytes())
        except (OSError, ValidationError) as exc:
            raise ValueError("capability worker state is invalid") from exc
        records = {item.external_id: item for item in state.workers}
        if len(records) != len(state.workers):
            raise ValueError("capability worker state contains duplicate handles")
        return records

    def _write_state(self) -> None:
        state = _PersistedState(workers=tuple(self._records[key] for key in sorted(self._records)))
        temporary = self._state_path.with_name(f".{self._state_path.name}.{uuid4().hex}.tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(state.model_dump_json().encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._state_path)
            directory = os.open(self._state_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def register_capability_worker_api(
    app: FastAPI,
    *,
    service: StableCapabilityWorkerService,
    token: str,
    authorize_lease: LeaseAuthorizer,
    prefix: str = "/bootstrap/internal/v1/capability-workers",
) -> None:
    """Register the lease-fenced lifecycle and HTTP-proxy surface."""

    expected_token = str(token or "").strip()
    if len(expected_token) < 32:
        raise ValueError("capability worker token must contain at least 32 characters")

    async def authorize(
        request: Request,
        supplied: Annotated[
            str | None,
            Header(alias="X-OpenTulpa-Capability-Worker-Token", max_length=500),
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
            raise HTTPException(status_code=401, detail="valid worker credentials are required")
        if lease_epoch is None:
            raise HTTPException(status_code=401, detail="active release credentials are required")
        await authorize_lease(str(release_id or ""), lease_epoch, str(control_token or ""))
        request.state.capability_worker_lease = CapabilityWorkerLease(
            release_id=str(release_id or ""),
            lease_epoch=lease_epoch,
        )

    router = APIRouter(
        prefix=prefix,
        dependencies=[Depends(authorize)],
        include_in_schema=False,
    )

    @router.post("/start", response_model=CapabilityWorkerResult)
    async def start(
        body: CapabilityWorkerStartRequest,
        request: Request,
    ) -> CapabilityWorkerResult:
        try:
            return await service.start(
                lease=request.state.capability_worker_lease,
                request=body,
            )
        except (CapabilityWorkerAPIError, OSError, WorkerLifecycleError, ValueError) as exc:
            raise HTTPException(status_code=503, detail="capability worker start failed") from exc

    @router.get("/{handle_id}/health", response_model=CapabilityWorkerHealthResult)
    async def health(
        handle_id: str,
    ) -> CapabilityWorkerHealthResult:
        try:
            return CapabilityWorkerHealthResult(healthy=await service.healthy(handle_id))
        except (CapabilityWorkerAPIError, OSError, WorkerLifecycleError) as exc:
            raise HTTPException(status_code=404, detail="capability worker was not found") from exc

    @router.post("/fence", status_code=status.HTTP_204_NO_CONTENT)
    async def fence(
        body: CapabilityWorkerFenceRequest,
        request: Request,
    ) -> Response:
        try:
            await service.fence(
                lease=request.state.capability_worker_lease,
                tenant_id=body.tenant_id,
                capability_name=body.capability_name,
            )
        except (CapabilityWorkerAPIError, OSError, WorkerLifecycleError) as exc:
            raise HTTPException(status_code=503, detail="capability worker fence failed") from exc
        return Response(status_code=204)

    @router.delete("/{handle_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def stop(
        handle_id: str,
        request: Request,
    ) -> Response:
        try:
            await service.stop(
                handle_id,
                lease=request.state.capability_worker_lease,
            )
        except (CapabilityWorkerAPIError, OSError, WorkerLifecycleError) as exc:
            raise HTTPException(status_code=503, detail="capability worker stop failed") from exc
        return Response(status_code=204)

    @router.api_route(
        "/{handle_id}/proxy",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    @router.api_route(
        "/{handle_id}/proxy/{suffix:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    )
    async def proxy(
        handle_id: str,
        request: Request,
        suffix: str = "",
    ) -> Response:
        try:
            return await service.proxy(
                handle_id,
                request,
                lease=request.state.capability_worker_lease,
                suffix=suffix,
            )
        except CapabilityWorkerAPIError as exc:
            raise HTTPException(status_code=502, detail="capability worker proxy failed") from exc

    app.include_router(router)


class CapabilityWorkerClient(WorkerHost):
    """Mutable-release client exposing only lease-bound lifecycle operations."""

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        release_id: str,
        lease_epoch: int,
        control_token: str,
        max_response_bytes: int = 1_000_000,
        transport: httpx.AsyncBaseTransport | None = None,
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
            raise ValueError("capability worker URL must be an authenticated HTTP endpoint")
        safe_token = str(token or "").strip()
        safe_release = str(release_id or "").strip()
        safe_control = str(control_token or "").strip()
        if len(safe_token) < 32 or not safe_release or lease_epoch < 1 or len(safe_control) < 32:
            raise ValueError("capability worker credentials are invalid")
        if not 4_096 <= max_response_bytes <= 20_000_000:
            raise ValueError("capability worker response limit is invalid")
        self._base_url = cleaned_url
        self._headers = {
            "X-OpenTulpa-Capability-Worker-Token": safe_token,
            "X-OpenTulpa-Release-ID": safe_release,
            "X-OpenTulpa-Lease-Epoch": str(lease_epoch),
            "X-OpenTulpa-Control-Token": safe_control,
        }
        self._max_response_bytes = max_response_bytes
        self._client = httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
            transport=transport,
            timeout=httpx.Timeout(connect=5, read=130, write=30, pool=5),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        result = await self._request_json(
            "POST",
            "/start",
            body={
                "tenant_id": launch.tenant_id,
                "instance_id": launch.instance_id,
                "manifest": launch.manifest.model_dump(mode="json"),
                "worker": launch.worker.model_dump(mode="json"),
                "config": dict(launch.config),
                "secret_environment": dict(launch.secret_environment),
            },
            model=CapabilityWorkerResult,
        )
        endpoint = None
        endpoint_headers: Mapping[str, str] = {}
        if result.endpoint_available:
            encoded = quote(result.handle_id, safe="")
            endpoint = f"{self._base_url}/{encoded}/proxy"
            endpoint_headers = dict(self._headers)
        return WorkerHandle(
            id=result.handle_id,
            instance_id=result.instance_id,
            capability_name=result.capability_name,
            capability_revision=result.capability_revision,
            manifest_digest=result.manifest_digest,
            worker_name=result.worker_name,
            endpoint=endpoint,
            endpoint_headers=endpoint_headers,
        )

    async def healthy(self, handle: WorkerHandle) -> bool:
        encoded = quote(handle.id, safe="")
        result = await self._request_json(
            "GET",
            f"/{encoded}/health",
            body=None,
            model=CapabilityWorkerHealthResult,
        )
        return bool(result.healthy)

    async def stop(self, handle: WorkerHandle) -> None:
        encoded = quote(handle.id, safe="")
        try:
            response = await self._client.delete(
                f"{self._base_url}/{encoded}",
                headers=self._headers,
            )
        except httpx.HTTPError as exc:
            raise WorkerLifecycleError("stable capability worker service is unavailable") from exc
        if response.status_code != 204:
            raise WorkerLifecycleError("stable capability worker service rejected stop")

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        try:
            response = await self._client.post(
                f"{self._base_url}/fence",
                headers=self._headers,
                json={
                    "tenant_id": tenant_id,
                    "capability_name": capability_name,
                },
            )
        except httpx.HTTPError as exc:
            raise WorkerLifecycleError("stable capability worker service is unavailable") from exc
        if response.status_code != 204:
            raise WorkerLifecycleError("stable capability worker service rejected fence")

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        body: Mapping[str, Any] | None,
        model: type[BaseModel],
    ) -> Any:
        try:
            async with self._client.stream(
                method,
                f"{self._base_url}{path}",
                headers=self._headers,
                json=body,
            ) as response:
                if response.status_code != 200:
                    raise WorkerLifecycleError("stable capability worker service rejected request")
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(raw) + len(chunk) > self._max_response_bytes:
                        raise WorkerLifecycleError("capability worker response exceeded its limit")
                    raw.extend(chunk)
        except httpx.HTTPError as exc:
            raise WorkerLifecycleError("stable capability worker service is unavailable") from exc
        try:
            return model.model_validate_json(raw)
        except ValueError as exc:
            raise WorkerLifecycleError(
                "stable capability worker service returned invalid data"
            ) from exc


__all__ = [
    "CapabilityWorkerAPIError",
    "CapabilityWorkerClient",
    "CapabilityWorkerHealthResult",
    "CapabilityWorkerFenceRequest",
    "CapabilityWorkerLease",
    "CapabilityWorkerResult",
    "CapabilityWorkerStartRequest",
    "StableCapabilityWorkerService",
    "register_capability_worker_api",
]
