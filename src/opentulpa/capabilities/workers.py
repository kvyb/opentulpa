"""Worker lifecycle boundary for dynamic capability implementations."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from opentulpa.capabilities.models import (
    CapabilityManifest,
    WorkerKind,
    WorkerRuntime,
    WorkerSpec,
    WorkerTransport,
    is_reserved_worker_environment_name,
)


class WorkerLifecycleError(RuntimeError):
    """A capability worker could not be started, checked, or stopped safely."""


@dataclass(frozen=True, slots=True)
class WorkerLaunch:
    """Host-only launch input; secret values are deliberately excluded from repr."""

    instance_id: str
    manifest: CapabilityManifest
    worker: WorkerSpec
    tenant_id: str = "default"
    config: Mapping[str, object] = field(default_factory=dict)
    secret_environment: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class WorkerHandle:
    """Opaque, non-secret identifier returned by a worker host."""

    id: str
    instance_id: str
    capability_name: str
    capability_revision: int
    manifest_digest: str
    worker_name: str
    endpoint: str | None = None
    endpoint_headers: Mapping[str, str] = field(default_factory=dict, repr=False)


@dataclass(frozen=True, slots=True)
class CapabilityWorkerSet:
    """All successfully started workers for one capability instance."""

    instance_id: str
    tenant_id: str
    capability_name: str
    capability_revision: int
    manifest_digest: str
    handles: tuple[WorkerHandle, ...]


class WorkerHost(Protocol):
    """Pluggable process/container host with runtime-specific enforcement."""

    async def start(self, launch: WorkerLaunch) -> WorkerHandle: ...

    async def healthy(self, handle: WorkerHandle) -> bool: ...

    async def stop(self, handle: WorkerHandle) -> None: ...


class SubprocessWorkerHost:
    """Minimal trusted host for source-reviewed workers bundled in this release.

    New capability code enters through source evolution and a subsequent release,
    never through a tenant-supplied executable. This host intentionally does not use
    a shell or inherit arbitrary host secrets. Resource and network declarations are
    metadata for these trusted seed workers. This host is a direct-development
    convenience only; managed production routes the same reviewed modules through the
    stable OCI authority, which enforces those boundaries and rollback fencing.
    """

    _INHERITED_ENVIRONMENT = (
        "PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "PYTHONPATH",
        "OPENTULPA_LAUNCH_NONCE",
    )

    def __init__(
        self,
        *,
        cwd: Path | None = None,
        base_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._cwd = cwd.resolve() if cwd is not None else None
        if base_environment is None:
            base_environment = {
                key: value
                for key in self._INHERITED_ENVIRONMENT
                if (value := os.environ.get(key)) is not None
            }
        self._base_environment = dict(base_environment)
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._specs: dict[str, WorkerSpec] = {}
        self._lock = asyncio.Lock()

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        if any(is_reserved_worker_environment_name(name) for name in launch.secret_environment):
            raise WorkerLifecycleError(
                "capability secrets cannot override worker runtime environment"
            )
        if launch.worker.runtime is not WorkerRuntime.SUBPROCESS:
            raise WorkerLifecycleError("subprocess host cannot launch OCI workers")
        ready_directory = None
        ready_path: Path | None = None
        if launch.worker.healthcheck.kind == "ready_file":
            ready_directory = tempfile.TemporaryDirectory(prefix="opentulpa-worker-ready-")
            ready_path = Path(ready_directory.name) / "ready"
        environment = {
            **self._base_environment,
            **dict(launch.secret_environment),
            "OPENTULPA_CAPABILITY_CONFIG": json.dumps(
                dict(launch.config),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "OPENTULPA_CAPABILITY_INSTANCE_ID": launch.instance_id,
            "OPENTULPA_CAPABILITY_NAME": launch.manifest.name,
            "OPENTULPA_CAPABILITY_REVISION": str(launch.manifest.revision),
            "OPENTULPA_WORKER_NAME": launch.worker.name,
        }
        if ready_path is not None:
            environment["OPENTULPA_WORKER_READY_FILE"] = str(ready_path)
        try:
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *launch.worker.command,
                    cwd=self._cwd,
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                ),
                timeout=launch.worker.resources.startup_timeout_seconds,
            )
        except asyncio.CancelledError:
            if ready_directory is not None:
                ready_directory.cleanup()
            raise
        except (OSError, TimeoutError) as exc:
            if ready_directory is not None:
                ready_directory.cleanup()
            raise WorkerLifecycleError(f"worker {launch.worker.name!r} failed to start") from exc
        handle = WorkerHandle(
            id=f"worker_{uuid4().hex}",
            instance_id=launch.instance_id,
            capability_name=launch.manifest.name,
            capability_revision=launch.manifest.revision,
            manifest_digest=launch.manifest.content_digest,
            worker_name=launch.worker.name,
        )
        async with self._lock:
            self._processes[handle.id] = process
            self._specs[handle.id] = launch.worker
        try:
            if ready_path is not None:
                await self._wait_ready(
                    process=process,
                    ready_path=ready_path,
                    timeout_seconds=min(
                        launch.worker.resources.startup_timeout_seconds,
                        launch.worker.healthcheck.timeout_seconds,
                    ),
                    worker_name=launch.worker.name,
                )
            else:
                await asyncio.sleep(0)
                if process.returncode is not None:
                    raise WorkerLifecycleError(
                        f"worker {launch.worker.name!r} exited during startup"
                    )
            return handle
        except asyncio.CancelledError:
            await self.stop(handle)
            raise
        except Exception:
            await self.stop(handle)
            raise
        finally:
            if ready_directory is not None:
                ready_directory.cleanup()

    async def healthy(self, handle: WorkerHandle) -> bool:
        async with self._lock:
            process = self._processes.get(handle.id)
        return process is not None and process.returncode is None

    async def stop(self, handle: WorkerHandle) -> None:
        async with self._lock:
            process = self._processes.pop(handle.id, None)
            spec = self._specs.pop(handle.id, None)
        if process is None:
            return
        if process.returncode is not None:
            await process.wait()
            return
        process.terminate()
        timeout = spec.resources.stop_timeout_seconds if spec is not None else 10
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    async def _wait_ready(
        *,
        process: asyncio.subprocess.Process,
        ready_path: Path,
        timeout_seconds: float,
        worker_name: str,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        expected = str(process.pid)
        while True:
            if process.returncode is not None:
                raise WorkerLifecycleError(f"worker {worker_name!r} exited before readiness")
            try:
                ready = ready_path.read_text(encoding="ascii").strip() == expected
            except (FileNotFoundError, OSError, UnicodeError):
                ready = False
            if ready:
                await asyncio.sleep(0)
                if process.returncode is None:
                    return
                raise WorkerLifecycleError(f"worker {worker_name!r} exited during readiness")
            if loop.time() >= deadline:
                raise WorkerLifecycleError(f"worker {worker_name!r} readiness timed out")
            await asyncio.sleep(0.05)


class CapabilityWorkerManager:
    """Atomically start or stop all workers in one manifest revision."""

    def __init__(self, host: WorkerHost) -> None:
        self._host = host
        self._active: dict[str, CapabilityWorkerSet] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def start(
        self,
        *,
        instance_id: str,
        manifest: CapabilityManifest,
        tenant_id: str = "default",
        config: Mapping[str, object] | None = None,
        secrets: Mapping[str, str] | None = None,
    ) -> CapabilityWorkerSet:
        safe_instance_id = str(instance_id or "").strip()
        if not safe_instance_id:
            raise ValueError("capability instance_id is required")
        safe_config = MappingProxyType(dict(config or {}))
        safe_secrets = dict(secrets or {})
        if any(is_reserved_worker_environment_name(name) for name in safe_secrets):
            raise WorkerLifecycleError(
                "capability secrets cannot override worker runtime environment"
            )
        # Validate serializability before starting the first process.
        try:
            json.dumps(dict(safe_config), ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("capability config must be JSON serializable") from exc

        async with self._lock:
            if self._closed:
                raise WorkerLifecycleError("capability worker manager is closed")
            if safe_instance_id in self._active:
                raise WorkerLifecycleError("capability instance is already running")
            handles: list[WorkerHandle] = []
            remote_fence = getattr(self._host, "fence", None)
            host_interaction_started = False
            safe_tenant = str(tenant_id or "").strip() or "default"
            try:
                for worker in manifest.workers:
                    secret_environment: dict[str, str] = {}
                    missing: list[str] = []
                    for requirement in worker.secrets:
                        value = str(safe_secrets.get(requirement.name, ""))
                        if value:
                            secret_environment[requirement.name] = value
                        elif requirement.required:
                            missing.append(requirement.name)
                    if missing:
                        raise WorkerLifecycleError(
                            f"worker {worker.name!r} is missing required secret grants: "
                            f"{', '.join(sorted(missing))}"
                        )
                    # The MCP adapter owns stdio process lifetime. Starting the same
                    # command here would create an unconnected duplicate whose stdin
                    # and stdout are discarded by the generic worker host.
                    if worker.kind is WorkerKind.MCP and worker.transport is WorkerTransport.STDIO:
                        continue
                    handle: WorkerHandle | None = None
                    try:
                        # A distributed host may launch and durably record a worker
                        # before the acknowledgement is lost. Mark the interaction
                        # before awaiting it so cleanup does not depend on a handle.
                        host_interaction_started = True
                        handle = await self._host.start(
                            WorkerLaunch(
                                instance_id=safe_instance_id,
                                manifest=manifest,
                                worker=worker,
                                tenant_id=safe_tenant,
                                config=safe_config,
                                secret_environment=MappingProxyType(secret_environment),
                            )
                        )
                        handles.append(handle)
                        if not await self._host.healthy(handle):
                            raise WorkerLifecycleError(
                                f"worker {worker.name!r} failed its startup health check"
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        if callable(remote_fence) or worker.required:
                            raise
                        if handle is not None:
                            try:
                                await self._host.stop(handle)
                            except Exception as exc:
                                raise WorkerLifecycleError(
                                    "optional capability worker cleanup failed"
                                ) from exc
                            handles.remove(handle)
                worker_set = CapabilityWorkerSet(
                    instance_id=safe_instance_id,
                    tenant_id=safe_tenant,
                    capability_name=manifest.name,
                    capability_revision=manifest.revision,
                    manifest_digest=manifest.content_digest,
                    handles=tuple(handles),
                )
                self._active[safe_instance_id] = worker_set
                return worker_set
            except BaseException as exc:
                stop_errors: list[BaseException] = []
                for handle in reversed(handles):
                    try:
                        await self._host.stop(handle)
                    except BaseException as stop_error:
                        stop_errors.append(stop_error)
                fence_attempted = False
                fence_error: BaseException | None = None
                if callable(remote_fence) and host_interaction_started:
                    fence_attempted = True
                    try:
                        result = remote_fence(
                            tenant_id=safe_tenant,
                            capability_name=manifest.name,
                        )
                        if inspect.isawaitable(result):
                            await result
                    except BaseException as cleanup_error:
                        fence_error = cleanup_error
                if fence_error is not None or (stop_errors and not fence_attempted):
                    raise WorkerLifecycleError(
                        "capability worker cleanup could not be confirmed"
                    ) from exc
                if not isinstance(exc, Exception):
                    raise
                if isinstance(exc, WorkerLifecycleError):
                    raise
                raise WorkerLifecycleError("capability workers failed to start") from exc

    async def healthy(self, instance_id: str) -> bool:
        async with self._lock:
            worker_set = self._active.get(instance_id)
            if worker_set is None:
                return False
            return all([await self._host.healthy(handle) for handle in worker_set.handles])

    async def stop(self, instance_id: str) -> None:
        async with self._lock:
            worker_set = self._active.get(instance_id)
            if worker_set is None:
                return
            errors: list[Exception] = []
            for handle in reversed(worker_set.handles):
                try:
                    await self._host.stop(handle)
                except Exception as exc:  # pragma: no cover - defensive host boundary
                    errors.append(exc)
            if errors:
                raise WorkerLifecycleError("one or more capability workers failed to stop")
            self._active.pop(instance_id, None)

    def active(self, instance_id: str) -> CapabilityWorkerSet | None:
        return self._active.get(instance_id)

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        """Stop every generation for one tenant capability, including remote orphans."""

        safe_tenant = str(tenant_id or "").strip()
        safe_capability = str(capability_name or "").strip()
        if not safe_tenant or not safe_capability:
            raise ValueError("tenant_id and capability_name are required")
        remote_fence = getattr(self._host, "fence", None)
        if callable(remote_fence):
            result = remote_fence(
                tenant_id=safe_tenant,
                capability_name=safe_capability,
            )
            if inspect.isawaitable(result):
                await result
        async with self._lock:
            matching = tuple(
                worker_set
                for worker_set in self._active.values()
                if worker_set.tenant_id == safe_tenant
                and worker_set.capability_name == safe_capability
            )
            errors: list[Exception] = []
            for worker_set in matching:
                for handle in reversed(worker_set.handles):
                    try:
                        if not callable(remote_fence):
                            await self._host.stop(handle)
                    except Exception as exc:  # pragma: no cover - defensive host boundary
                        errors.append(exc)
                if not errors:
                    self._active.pop(worker_set.instance_id, None)
            if errors:
                raise WorkerLifecycleError("one or more capability generations failed to fence")

    async def aclose(self) -> None:
        """Stop all generations and close host-owned async resources once."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            worker_sets = tuple(reversed(tuple(self._active.values())))
            self._active.clear()
            errors: list[Exception] = []
            for worker_set in worker_sets:
                for handle in reversed(worker_set.handles):
                    try:
                        await self._host.stop(handle)
                    except Exception as exc:  # pragma: no cover - defensive host boundary
                        errors.append(exc)
        close = getattr(self._host, "aclose", None)
        if close is not None:
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # pragma: no cover - defensive host boundary
                errors.append(exc)
        if errors:
            raise WorkerLifecycleError("capability worker host failed to close cleanly")


__all__ = [
    "CapabilityWorkerManager",
    "CapabilityWorkerSet",
    "SubprocessWorkerHost",
    "WorkerHandle",
    "WorkerHost",
    "WorkerLaunch",
    "WorkerLifecycleError",
]
