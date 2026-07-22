"""Rootless OCI host for tenant-generated capability workers."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx

from opentulpa.bootstrap.oci_host import (
    LocalOciCommandRunner,
    OciCommandResult,
    OciCommandRunner,
)
from opentulpa.capabilities.models import (
    WorkerRuntime,
    WorkerSpec,
    WorkerTransport,
    is_reserved_worker_environment_name,
)
from opentulpa.capabilities.workers import (
    WorkerHandle,
    WorkerHost,
    WorkerLaunch,
    WorkerLifecycleError,
)

_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_NETWORK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}\Z")
_ENVIRONMENT_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_RUNTIME_USER = re.compile(r"[0-9]{1,10}:[0-9]{1,10}\Z")
_HOSTNAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}\Z")
_HOST_GATEWAY = re.compile(r"(?:host-gateway|[0-9A-Fa-f:.]{2,64})\Z")
_OWNED_LABEL = "org.opentulpa.capability.managed"
_INSTALLATION_LABEL = "org.opentulpa.capability.installation"
_TENANT_LABEL = "org.opentulpa.capability.tenant"
_CAPABILITY_LABEL = "org.opentulpa.capability.name"
_INSTANCE_LABEL = "org.opentulpa.capability.instance"
_MANIFEST_LABEL = "org.opentulpa.capability.manifest-digest"
_WORKER_LABEL = "org.opentulpa.capability.worker"
_RELEASE_LABEL = "org.opentulpa.capability.release"
_LEASE_LABEL = "org.opentulpa.capability.lease"


@dataclass(frozen=True, slots=True)
class OciCapabilityPolicy:
    container_cli: str = "docker"
    state_root: Path = Path(".opentulpa/deepagents/capability_containers")
    restricted_egress_network: str | None = None
    restricted_allowed_hosts: tuple[str, ...] = ()
    persistent_state_root: Path | None = None
    runtime_user: str = "65532:65532"
    host_gateway_name: str = "host.docker.internal"
    host_gateway_address: str = "host-gateway"

    def __post_init__(self) -> None:
        if Path(self.container_cli).name not in {"docker", "podman"}:
            raise ValueError("capability container runtime must be Docker or Podman")
        if self.restricted_egress_network is not None and not _NETWORK.fullmatch(
            self.restricted_egress_network
        ):
            raise ValueError("capability egress network name is invalid")
        if len(self.restricted_allowed_hosts) != len(set(self.restricted_allowed_hosts)):
            raise ValueError("capability allowed hosts must be unique")
        if _RUNTIME_USER.fullmatch(self.runtime_user) is None:
            raise ValueError("capability runtime user must be numeric uid:gid")
        if _HOSTNAME.fullmatch(self.host_gateway_name) is None:
            raise ValueError("capability host gateway name is invalid")
        if _HOST_GATEWAY.fullmatch(self.host_gateway_address) is None:
            raise ValueError("capability host gateway address is invalid")


@dataclass(frozen=True, slots=True)
class _RunningWorker:
    container_id: str
    container_name: str
    endpoint: str | None
    ready_path: Path | None = None
    tenant_label: str = ""
    capability_label: str = ""
    release_label: str | None = None
    lease_label: str | None = None


class OciCapabilityWorkerHost:
    """Launch digest-pinned workers without repository, data, or socket mounts."""

    def __init__(
        self,
        *,
        policy: OciCapabilityPolicy | None = None,
        runner: OciCommandRunner | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._policy = policy or OciCapabilityPolicy()
        self._state_root = self._policy.state_root.expanduser().resolve()
        self._state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._installation_label = self._label_value(str(self._state_root))
        self._persistent_state_root: Path | None
        configured_state = self._policy.persistent_state_root
        if configured_state is not None:
            configured_state = configured_state.expanduser()
            if configured_state.is_symlink():
                raise ValueError("capability state root cannot be a symlink")
            configured_state.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._persistent_state_root = configured_state.resolve(strict=True)
        else:
            self._persistent_state_root = None
        self._runner = runner or LocalOciCommandRunner(cwd=self._state_root)
        self._http = http_client or httpx.AsyncClient(follow_redirects=False, trust_env=False)
        self._owns_http = http_client is None
        self._running: dict[str, _RunningWorker] = {}
        self._specs: dict[str, WorkerSpec] = {}
        self._ready = False
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def ready_path(self, handle: WorkerHandle) -> Path | None:
        running = self._running.get(handle.id)
        return running.ready_path if running is not None else None

    def adopt(
        self,
        *,
        handle: WorkerHandle,
        worker: WorkerSpec,
        ready_path: Path | None,
        tenant_id: str = "default",
        release_id: str | None = None,
        lease_epoch: int | None = None,
    ) -> None:
        """Restore non-secret lifecycle metadata after a bootstrap process restart."""

        prefix, _, container_id = handle.id.partition(":")
        if prefix != "oci" or _CONTAINER_ID.fullmatch(container_id) is None:
            raise WorkerLifecycleError("persisted OCI capability handle is invalid")
        if ready_path is not None:
            ready_path = ready_path.expanduser().resolve()
            root = self._persistent_state_root
            if root is None or not ready_path.is_relative_to(root):
                raise WorkerLifecycleError("persisted capability readiness path is invalid")
        self._running[handle.id] = _RunningWorker(
            container_id=container_id,
            container_name=self._container_name(handle.instance_id, handle.worker_name),
            endpoint=handle.endpoint,
            ready_path=ready_path,
            tenant_label=self._label_value(tenant_id),
            capability_label=self._label_value(handle.capability_name),
            release_label=(self._label_value(release_id) if release_id is not None else None),
            lease_label=(str(lease_epoch) if lease_epoch is not None else None),
        )
        self._specs[handle.id] = worker

    async def start(
        self,
        launch: WorkerLaunch,
        *,
        release_id: str | None = None,
        lease_epoch: int | None = None,
    ) -> WorkerHandle:
        if any(is_reserved_worker_environment_name(name) for name in launch.secret_environment):
            raise WorkerLifecycleError(
                "capability secrets cannot override worker runtime environment"
            )
        safe_release = str(release_id or "").strip()
        if bool(safe_release) != (lease_epoch is not None) or (
            lease_epoch is not None and lease_epoch < 1
        ):
            raise WorkerLifecycleError("managed capability worker lease is invalid")
        worker = launch.worker
        if worker.runtime is not WorkerRuntime.OCI or worker.image is None:
            raise WorkerLifecycleError("OCI host requires a digest-pinned OCI worker")
        if worker.kind.value == "mcp" and worker.transport is not WorkerTransport.STREAMABLE_HTTP:
            raise WorkerLifecycleError("OCI MCP workers must use streamable HTTP")
        async with self._lock:
            await self._ensure_backend()
            await self._verify_image(worker.image, launch.manifest.artifact_digest)
            network = self._network_for(launch)
            name = self._container_name(launch.instance_id, worker.name)
            tenant_label = self._label_value(launch.tenant_id)
            capability_label = self._label_value(launch.manifest.name)
            release_label = self._label_value(safe_release) if safe_release else None
            lease_label = str(lease_epoch) if lease_epoch is not None else None
            await self._remove(name)
            state_directory = self._state_directory(launch)
            ready_path = self._ready_path(launch, state_directory)
            if ready_path is not None:
                ready_path.unlink(missing_ok=True)
            env_path = self._write_environment(
                launch,
                ready_target=(f"/state/{ready_path.name}" if ready_path is not None else None),
            )
            try:
                argv = [
                    self._policy.container_cli,
                    "run",
                    "--detach",
                    "--name",
                    name,
                    "--init",
                    "--pull",
                    "never",
                    "--restart",
                    "no",
                    "--log-driver",
                    "none",
                    "--read-only",
                    "--network",
                    network,
                    "--ipc",
                    "none",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--cpus",
                    str(worker.resources.cpu),
                    "--memory",
                    f"{worker.resources.memory_mb}m",
                    "--memory-swap",
                    f"{worker.resources.memory_mb}m",
                    "--pids-limit",
                    str(worker.resources.pids),
                    "--ulimit",
                    "nofile=1024:1024",
                    "--ulimit",
                    "core=0:0",
                    "--user",
                    self._policy.runtime_user,
                    "--tmpfs",
                    "/tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777",
                    "--tmpfs",
                    "/run:rw,noexec,nosuid,nodev,size=16m,mode=755",
                    "--env-file",
                    str(env_path),
                    "--label",
                    f"{_OWNED_LABEL}=true",
                    "--label",
                    f"{_INSTALLATION_LABEL}={self._installation_label}",
                    "--label",
                    f"{_TENANT_LABEL}={tenant_label}",
                    "--label",
                    f"{_CAPABILITY_LABEL}={capability_label}",
                    "--label",
                    f"{_INSTANCE_LABEL}={self._label_value(launch.instance_id)}",
                    "--label",
                    f"{_MANIFEST_LABEL}={launch.manifest.content_digest}",
                    "--label",
                    f"{_WORKER_LABEL}={self._label_value(worker.name)}",
                ]
                if release_label is not None and lease_label is not None:
                    argv.extend(
                        (
                            "--label",
                            f"{_RELEASE_LABEL}={release_label}",
                            "--label",
                            f"{_LEASE_LABEL}={lease_label}",
                        )
                    )
                if state_directory is not None:
                    argv.extend(
                        (
                            "--mount",
                            f"type=bind,src={state_directory},dst=/state",
                        )
                    )
                if network != "none":
                    argv.extend(
                        (
                            "--add-host",
                            f"{self._policy.host_gateway_name}:{self._policy.host_gateway_address}",
                        )
                    )
                port = self._worker_port(launch)
                if port is not None:
                    argv.extend(("--publish", f"127.0.0.1::{port}/tcp"))
                argv.extend(
                    (
                        "--entrypoint",
                        worker.command[0],
                        worker.image,
                        *worker.command[1:],
                    )
                )
                try:
                    result = await self._runner.run(
                        argv,
                        timeout_seconds=worker.resources.startup_timeout_seconds,
                        max_output_bytes=worker.resources.max_output_bytes,
                    )
                except BaseException:
                    try:
                        await self._remove(name)
                    except BaseException as cleanup_error:
                        raise WorkerLifecycleError(
                            "OCI capability worker launch cleanup could not be confirmed"
                        ) from cleanup_error
                    raise
            finally:
                env_path.unlink(missing_ok=True)
            container_id = result.output.decode("ascii", errors="ignore").strip().lower()
            if (
                result.returncode != 0
                or result.truncated
                or result.timed_out
                or not _CONTAINER_ID.fullmatch(container_id)
            ):
                await self._remove(name)
                raise WorkerLifecycleError("OCI capability worker failed to start")
            try:
                endpoint = (
                    await self._published_endpoint(
                        container_id,
                        port,
                        path=self._worker_endpoint_path(launch),
                    )
                    if port
                    else None
                )
            except Exception:
                await self._remove(container_id)
                raise
            handle = WorkerHandle(
                id=f"oci:{container_id}",
                instance_id=launch.instance_id,
                capability_name=launch.manifest.name,
                capability_revision=launch.manifest.revision,
                manifest_digest=launch.manifest.content_digest,
                worker_name=worker.name,
                endpoint=endpoint,
                endpoint_headers={},
            )
            self._running[handle.id] = _RunningWorker(
                container_id=container_id,
                container_name=name,
                endpoint=endpoint,
                ready_path=ready_path,
                tenant_label=tenant_label,
                capability_label=capability_label,
                release_label=release_label,
                lease_label=lease_label,
            )
            self._specs[handle.id] = worker
            return handle

    async def healthy(self, handle: WorkerHandle) -> bool:
        running = self._running.get(handle.id)
        if running is None:
            return False
        inspected = await self._runner.run(
            (
                self._policy.container_cli,
                "inspect",
                "--format",
                "{{.State.Running}}",
                running.container_id,
            ),
            timeout_seconds=10,
            max_output_bytes=1_024,
        )
        if inspected.returncode != 0 or inspected.output.strip().lower() != b"true":
            return False
        spec = self._specs.get(handle.id)
        healthcheck = getattr(spec, "healthcheck", None)
        if getattr(healthcheck, "kind", None) == "ready_file":
            timeout = float(getattr(healthcheck, "timeout_seconds", 5))
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            while running.ready_path is None or not running.ready_path.is_file():
                if loop.time() >= deadline:
                    return False
                await asyncio.sleep(0.05)
            return True
        if handle.endpoint is None:
            return True
        timeout = float(getattr(getattr(spec, "resources", None), "startup_timeout_seconds", 5))
        target = str(getattr(healthcheck, "target", "") or "")
        url = f"{handle.endpoint}{target}" if target else handle.endpoint
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            try:
                response = await self._http.get(url, timeout=min(5, timeout))
                if response.status_code < 500:
                    return True
            except httpx.HTTPError:
                pass
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.1)

    async def stop(self, handle: WorkerHandle) -> None:
        async with self._lock:
            running = self._running.get(handle.id)
            spec = self._specs.get(handle.id)
            if running is None:
                return
            await self._remove(
                running.container_id,
                timeout_seconds=(spec.resources.stop_timeout_seconds if spec is not None else 30),
            )
            self._running.pop(handle.id, None)
            self._specs.pop(handle.id, None)

    async def fence(self, *, tenant_id: str, capability_name: str) -> None:
        """Remove every host-owned container for an exact tenant capability."""

        safe_tenant = str(tenant_id or "").strip()
        safe_capability = str(capability_name or "").strip()
        if not safe_tenant or not safe_capability:
            raise ValueError("tenant_id and capability_name are required")
        tenant_label = self._label_value(safe_tenant)
        capability_label = self._label_value(safe_capability)
        async with self._lock:
            discovered = await self._container_ids(
                f"label={_OWNED_LABEL}=true",
                f"label={_INSTALLATION_LABEL}={self._installation_label}",
                f"label={_TENANT_LABEL}={tenant_label}",
                f"label={_CAPABILITY_LABEL}={capability_label}",
            )
            tracked = {
                running.container_id
                for running in self._running.values()
                if running.tenant_label == tenant_label
                and running.capability_label == capability_label
            }
            targets = set(discovered) | tracked
            await self._remove_targets(targets)

    async def reconcile_managed_workers(
        self,
        *,
        release_id: str | None,
        lease_epoch: int | None,
        keep_container_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        """Fence stale and unrecorded managed workers across release lease changes."""

        safe_release = str(release_id or "").strip()
        if bool(safe_release) != (lease_epoch is not None) or (
            lease_epoch is not None and lease_epoch < 1
        ):
            raise ValueError("managed capability worker lease is invalid")
        requested_keep = set(keep_container_ids)
        if any(_CONTAINER_ID.fullmatch(value) is None for value in requested_keep):
            raise WorkerLifecycleError("persisted OCI capability handle is invalid")
        release_label = self._label_value(safe_release) if safe_release else None
        lease_label = str(lease_epoch) if lease_epoch is not None else None
        async with self._lock:
            discovered = set(
                await self._container_ids(
                    f"label={_OWNED_LABEL}=true",
                    f"label={_INSTALLATION_LABEL}={self._installation_label}",
                    f"label={_RELEASE_LABEL}",
                )
            )
            eligible: set[str] = set()
            if release_label is not None and lease_label is not None:
                eligible = set(
                    await self._container_ids(
                        f"label={_OWNED_LABEL}=true",
                        f"label={_INSTALLATION_LABEL}={self._installation_label}",
                        f"label={_RELEASE_LABEL}={release_label}",
                        f"label={_LEASE_LABEL}={lease_label}",
                    )
                )
            keep = requested_keep & eligible
            tracked_stale = {
                running.container_id
                for running in self._running.values()
                if running.release_label is not None
                and (
                    running.release_label != release_label
                    or running.lease_label != lease_label
                    or running.container_id not in keep
                )
            }
            targets = (discovered - keep) | tracked_stale
            await self._remove_targets(targets)
            return tuple(sorted(keep))

    async def _remove_targets(self, targets: set[str]) -> None:
        removed: set[str] = set()
        errors: list[Exception] = []
        for container_id in sorted(targets):
            try:
                await self._remove(container_id)
            except Exception as exc:
                errors.append(exc)
            else:
                removed.add(container_id)
        for handle_id, running in tuple(self._running.items()):
            if running.container_id in removed:
                self._running.pop(handle_id, None)
                self._specs.pop(handle_id, None)
        if errors:
            raise WorkerLifecycleError(
                "one or more OCI capability workers could not be removed"
            ) from errors[0]

    async def _ensure_backend(self) -> None:
        if self._ready:
            return
        engine = Path(self._policy.container_cli).name
        if engine == "docker":
            result = await self._command("info", "--format", "{{json .SecurityOptions}}")
            if "rootless" not in result.output.decode(errors="ignore").casefold():
                raise WorkerLifecycleError("Docker capability host must run rootless")
        else:
            result = await self._command("info", "--format", "{{.Host.Security.Rootless}}")
            if result.output.decode().strip().casefold() != "true":
                raise WorkerLifecycleError("Podman capability host must run rootless")
        self._ready = True

    async def _verify_image(self, image: str, artifact_digest: str | None) -> None:
        expected = str(artifact_digest or "")
        if not _DIGEST.fullmatch(expected) or (
            image != expected and image.rpartition("@")[2] != expected
        ):
            raise WorkerLifecycleError("OCI worker image does not match its artifact digest")
        result = await self._command("image", "inspect", "--format", "{{.Id}}", image)
        if result.output.decode().strip().lower() != expected:
            raise WorkerLifecycleError("local OCI worker image ID does not match the manifest")

    def _network_for(self, launch: WorkerLaunch) -> str:
        policies = (launch.manifest.network, launch.worker.network)
        if all(policy.outbound == "deny" for policy in policies):
            return "none"
        if any(policy.outbound == "tenant_allowlist" for policy in policies):
            raise WorkerLifecycleError("tenant-defined capability egress is not host-enforceable")
        requested = {
            host
            for policy in policies
            if policy.outbound == "allowlist"
            for host in policy.allowed_hosts
        }
        network = self._policy.restricted_egress_network
        if network is None:
            raise WorkerLifecycleError(
                "outbound capability workers require an administrator-controlled egress network"
            )
        allowed = set(self._policy.restricted_allowed_hosts)
        if not allowed or not requested.issubset(allowed):
            raise WorkerLifecycleError("capability requested an undeclared egress destination")
        return network

    def _write_environment(
        self,
        launch: WorkerLaunch,
        *,
        ready_target: str | None = None,
    ) -> Path:
        values = {
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
            **dict(launch.secret_environment),
        }
        if ready_target is not None:
            values["OPENTULPA_WORKER_READY_FILE"] = ready_target
        for name, value in values.items():
            if not _ENVIRONMENT_NAME.fullmatch(name) or any(char in value for char in "\0\r\n"):
                raise WorkerLifecycleError("capability worker environment is invalid")
        path = self._state_root / f".env-{hashlib.sha256(os.urandom(32)).hexdigest()}.tmp"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                for name in sorted(values):
                    stream.write(f"{name}={values[name]}\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    def _state_directory(self, launch: WorkerLaunch) -> Path | None:
        root = self._persistent_state_root
        if root is None:
            return None
        for key, value in launch.config.items():
            if not str(key).endswith("_path") or not isinstance(value, str):
                continue
            path = value.strip()
            if not path.startswith("/state/") or ".." in Path(path).parts:
                raise WorkerLifecycleError("capability state paths must remain below /state")
        digest = hashlib.sha256(
            f"{launch.tenant_id}\0{launch.manifest.name}\0{launch.worker.name}".encode()
        ).hexdigest()
        directory = root / digest
        if directory.is_symlink():
            raise WorkerLifecycleError("capability state directory cannot be a symlink")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise WorkerLifecycleError("capability state directory escaped its root")
        return resolved

    @staticmethod
    def _ready_path(launch: WorkerLaunch, state_directory: Path | None) -> Path | None:
        if launch.worker.healthcheck.kind != "ready_file":
            return None
        if state_directory is None:
            raise WorkerLifecycleError("ready-file workers require isolated persistent state")
        digest = hashlib.sha256(launch.instance_id.encode()).hexdigest()[:24]
        return state_directory / f".ready-{digest}"

    @staticmethod
    def _worker_port(launch: WorkerLaunch) -> int | None:
        endpoint = launch.worker.endpoint
        if launch.worker.transport is not WorkerTransport.STREAMABLE_HTTP:
            return None
        parsed = urlsplit(str(endpoint or ""))
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise WorkerLifecycleError("OCI HTTP worker endpoint must be loopback HTTP")
        try:
            port = parsed.port
        except ValueError as exc:
            raise WorkerLifecycleError("OCI HTTP worker endpoint port is invalid") from exc
        if port is None:
            raise WorkerLifecycleError("OCI HTTP worker endpoint requires an explicit port")
        return port

    @staticmethod
    def _worker_endpoint_path(launch: WorkerLaunch) -> str:
        if launch.worker.transport is not WorkerTransport.STREAMABLE_HTTP:
            return ""
        parsed = urlsplit(str(launch.worker.endpoint or ""))
        path = parsed.path or ""
        if parsed.query or parsed.fragment:
            raise WorkerLifecycleError("OCI HTTP worker endpoint cannot contain query or fragment")
        return path.rstrip("/")

    async def _published_endpoint(self, container_id: str, port: int, *, path: str) -> str:
        for _ in range(20):
            result = await self._runner.run(
                (self._policy.container_cli, "port", container_id, f"{port}/tcp"),
                timeout_seconds=10,
                max_output_bytes=2_048,
            )
            value = result.output.decode("ascii", errors="ignore").strip()
            match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", value)
            if result.returncode == 0 and match is not None:
                published = int(match.group(1))
                if 1 <= published <= 65_535:
                    return f"http://127.0.0.1:{published}{path}"
            await asyncio.sleep(0.05)
        raise WorkerLifecycleError("OCI worker endpoint was not published")

    async def _remove(self, identifier: str, *, timeout_seconds: float = 30) -> None:
        if await self._container_absent(identifier, timeout_seconds=timeout_seconds):
            return
        result = await self._runner.run(
            (self._policy.container_cli, "rm", "--force", identifier),
            timeout_seconds=timeout_seconds,
            max_output_bytes=16_384,
        )
        if result.returncode != 0 or result.truncated or result.timed_out:
            raise WorkerLifecycleError("OCI capability worker removal failed")
        if not await self._container_absent(identifier, timeout_seconds=timeout_seconds):
            raise WorkerLifecycleError("OCI capability worker removal was not confirmed")

    async def _container_absent(self, identifier: str, *, timeout_seconds: float) -> bool:
        filter_name = "id" if _CONTAINER_ID.fullmatch(identifier) is not None else "name"
        return not await self._container_ids(
            f"{filter_name}={identifier}",
            timeout_seconds=timeout_seconds,
        )

    async def _container_ids(
        self,
        *filters: str,
        timeout_seconds: float = 30,
    ) -> tuple[str, ...]:
        argv = [
            self._policy.container_cli,
            "ps",
            "--all",
            "--quiet",
            "--no-trunc",
        ]
        for value in filters:
            argv.extend(("--filter", value))
        result = await self._runner.run(
            argv,
            timeout_seconds=timeout_seconds,
            max_output_bytes=16_384,
        )
        if result.returncode != 0 or result.truncated or result.timed_out:
            raise WorkerLifecycleError("OCI capability worker absence could not be confirmed")
        values = tuple(
            line.strip().decode("ascii", errors="ignore").lower()
            for line in result.output.splitlines()
            if line.strip()
        )
        if any(_CONTAINER_ID.fullmatch(value) is None for value in values):
            raise WorkerLifecycleError("OCI capability worker discovery returned invalid data")
        return tuple(dict.fromkeys(values))

    async def _command(self, *args: str) -> OciCommandResult:
        result = await self._runner.run(
            (self._policy.container_cli, *args),
            timeout_seconds=30,
            max_output_bytes=256_000,
        )
        if result.returncode != 0 or result.truncated or result.timed_out:
            raise WorkerLifecycleError("OCI capability host command failed")
        return result

    def _container_name(self, instance_id: str, worker_name: str) -> str:
        digest = hashlib.sha256(
            f"{self._installation_label}\0{instance_id}\0{worker_name}".encode()
        ).hexdigest()[:24]
        return f"opentulpa-capability-{digest}"

    @staticmethod
    def _label_value(value: str) -> str:
        return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class RoutingWorkerHost:
    """Route trusted seed subprocesses and generated OCI workers by runtime."""

    def __init__(self, *, subprocess: WorkerHost, oci: WorkerHost) -> None:
        self._subprocess = subprocess
        self._oci = oci
        self._owners: dict[str, Literal["subprocess", "oci"]] = {}

    async def start(self, launch: WorkerLaunch) -> WorkerHandle:
        owner: Literal["subprocess", "oci"] = (
            "oci" if launch.worker.runtime is WorkerRuntime.OCI else "subprocess"
        )
        host = self._oci if owner == "oci" else self._subprocess
        handle = await host.start(launch)
        self._owners[handle.id] = owner
        return handle

    async def healthy(self, handle: WorkerHandle) -> bool:
        owner = self._owners.get(handle.id)
        if owner is None:
            return False
        return await (self._oci if owner == "oci" else self._subprocess).healthy(handle)

    async def stop(self, handle: WorkerHandle) -> None:
        owner = self._owners.get(handle.id)
        if owner is None:
            return
        await (self._oci if owner == "oci" else self._subprocess).stop(handle)
        self._owners.pop(handle.id, None)

    async def aclose(self) -> None:
        """Close resources owned by each routed host exactly once."""

        seen: set[int] = set()
        for host in (self._subprocess, self._oci):
            if id(host) in seen:
                continue
            seen.add(id(host))
            close = getattr(host, "aclose", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result


__all__ = [
    "OciCapabilityPolicy",
    "OciCapabilityWorkerHost",
    "RoutingWorkerHost",
]
