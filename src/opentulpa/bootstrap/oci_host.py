"""Rootless OCI implementation of the immutable release host boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import stat
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol, cast
from uuid import uuid4

import httpx

from opentulpa.bootstrap.host import ReleaseHostError
from opentulpa.bootstrap.models import (
    DrainResult,
    PreparedRelease,
    ReleaseHealth,
    ReleaseLaunchContext,
    ReleaseRecord,
    RunningRelease,
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID_RE = re.compile(r"[0-9a-f]{12,64}\Z")
_RESOURCE_RE = re.compile(r"[1-9][0-9]*(?:\.[0-9]+)?(?:[kmgt]i?|b)?\Z", re.I)
_CPU_RE = re.compile(r"(?:0\.[0-9]*[1-9][0-9]*|[1-9][0-9]*(?:\.[0-9]+)?)\Z")
_NETWORK_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,62}\Z")
_ENV_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_RUNTIME_USER_RE = re.compile(r"[0-9]{1,10}:[0-9]{1,10}\Z")
_HOSTNAME_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9.-]{0,252}\Z")
_HOST_GATEWAY_RE = re.compile(r"(?:host-gateway|[0-9a-fA-F:.]{2,64})\Z")
_SAFE_TARGET_RE = re.compile(r"/[A-Za-z0-9._/-]{1,250}\Z")
_RELEASE_LABEL = "org.opentulpa.release.id"
_MODE_LABEL = "org.opentulpa.release.mode"
_LEASE_LABEL = "org.opentulpa.release.lease-epoch"
_ARTIFACT_LABEL = "org.opentulpa.release.artifact-digest"
_SOURCE_LABEL = "org.opentulpa.release.source-commit"
_SOURCE_LAYOUT_LABEL = "org.opentulpa.release.source-layout"
_SOURCE_LAYOUT_VERSION = "full-source-v1"
_PREVIOUS_SOURCE_LAYOUT_VERSION = "capability-workers-manifests-web-assets-v1"
_TRUSTED_SOURCE_LAYOUT_VERSIONS = frozenset(
    {_SOURCE_LAYOUT_VERSION, _PREVIOUS_SOURCE_LAYOUT_VERSION}
)
_NETWORK_LABEL = "org.opentulpa.release.network"
_NETWORK_OWNED_LABEL = "org.opentulpa.release.network-owned"
_PYTHON_IMPORT_ENV_NAMES = frozenset(
    {
        "PYTHONHOME",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "PYTHONPLATLIBDIR",
        "PYTHONSAFEPATH",
        "PYTHONUSERBASE",
    }
)
_RESERVED_ENV_NAMES = frozenset(
    {
        "HOST",
        "PORT",
        "OPENTULPA_CONTROL_TOKEN",
        "OPENTULPA_DATA_ROOT",
        "OPENTULPA_DISABLE_CONSUMERS",
        "OPENTULPA_EVENT_PATH",
        "OPENTULPA_HEALTH_PATH",
        "OPENTULPA_INGRESS_ENABLED",
        "OPENTULPA_INGRESS_PATH",
        "OPENTULPA_LEASE_EPOCH",
        "OPENTULPA_MANAGED_RELEASE",
        "OPENTULPA_RELEASE_ID",
        "OPENTULPA_RELEASE_MODE",
        "OPENTULPA_SECRETS_ENABLED",
        "OPENTULPA_DRAIN_PATH",
    }
) | _PYTHON_IMPORT_ENV_NAMES


@dataclass(frozen=True, slots=True)
class OciCommandResult:
    returncode: int
    output: bytes = b""
    truncated: bool = False
    timed_out: bool = False


class OciCommandRunner(Protocol):
    async def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> OciCommandResult: ...


class LocalOciCommandRunner:
    """Bounded exec-only command runner with no inherited secret environment."""

    def __init__(self, *, cwd: Path) -> None:
        self._cwd = cwd.expanduser().resolve()

    async def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> OciCommandResult:
        if not argv or timeout_seconds <= 0 or max_output_bytes < 1:
            raise ValueError("OCI command runner configuration is invalid")
        environment = {
            key: value
            for key in (
                "PATH",
                "HOME",
                "DOCKER_HOST",
                "DOCKER_CONTEXT",
                "CONTAINER_HOST",
                "XDG_RUNTIME_DIR",
            )
            if (value := os.environ.get(key))
        }
        environment.setdefault("PATH", os.defpath)
        environment.setdefault("HOME", "/tmp")
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=self._cwd,
            env=environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        stream = process.stdout
        assert stream is not None
        retained = bytearray()
        output_size = 0

        async def drain_output() -> None:
            nonlocal output_size
            while chunk := await stream.read(64 * 1_024):
                output_size += len(chunk)
                remaining = max_output_bytes - len(retained)
                if remaining > 0:
                    retained.extend(chunk[:remaining])

        reader = asyncio.create_task(drain_output())
        timed_out = False
        try:
            async with asyncio.timeout(timeout_seconds):
                await process.wait()
        except TimeoutError:
            timed_out = True
            self._kill_process_group(process)
            await process.wait()
        except BaseException:
            self._kill_process_group(process)
            await process.wait()
            await reader
            raise
        await reader
        return OciCommandResult(
            returncode=124 if timed_out else int(process.returncode or 0),
            output=bytes(retained),
            truncated=output_size > len(retained),
            timed_out=timed_out,
        )

    @staticmethod
    def _kill_process_group(process: asyncio.subprocess.Process) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            with suppress(OSError):
                process.kill()


@dataclass(frozen=True, slots=True)
class OciMount:
    source: Path
    target: str
    read_only: bool = True
    production_only: bool = True


@dataclass(frozen=True, slots=True)
class OciReleasePolicy:
    container_cli: str = "docker"
    state_root: Path = Path(".opentulpa/bootstrap")
    network_name: str = "opentulpa-release-internal"
    production_network_name: str | None = None
    host_gateway_name: str = "host.docker.internal"
    host_gateway_address: str = "host-gateway"
    runtime_user: str = "65532:65532"
    data_mount_target: str = "/workspace"
    require_persistent_data_mount: bool = False
    production_environment: tuple[tuple[str, str], ...] = ()
    cpu_limit: str = "1"
    memory_limit: str = "512m"
    pid_limit: int = 128
    stop_timeout_seconds: int = 20
    command_timeout_seconds: int = 30
    health_timeout_seconds: float = 5.0
    max_command_output_bytes: int = 256_000
    max_snapshot_bytes: int = 20 * 1024 * 1024 * 1024
    max_snapshot_entries: int = 1_000_000
    mounts: tuple[OciMount, ...] = ()
    allowed_mount_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        engine = Path(self.container_cli).name
        if engine not in {"docker", "podman"} or "\x00" in self.container_cli:
            raise ValueError("container_cli must be a Docker or Podman executable")
        if not _NETWORK_RE.fullmatch(self.network_name) or len(self.network_name) > 50:
            raise ValueError("OCI release network name is invalid")
        if self.production_network_name is not None and (
            not _NETWORK_RE.fullmatch(self.production_network_name)
            or self.production_network_name in {"host", "none"}
        ):
            raise ValueError("OCI production egress network name is invalid")
        if _HOSTNAME_RE.fullmatch(self.host_gateway_name) is None:
            raise ValueError("OCI host gateway name is invalid")
        if _HOST_GATEWAY_RE.fullmatch(self.host_gateway_address) is None:
            raise ValueError("OCI host gateway address is invalid")
        if _RUNTIME_USER_RE.fullmatch(self.runtime_user) is None:
            raise ValueError("OCI runtime user must be a numeric uid:gid")
        if self.data_mount_target != "/workspace":
            raise ValueError("OCI release data mount target must be /workspace")
        if not _CPU_RE.fullmatch(self.cpu_limit) or float(self.cpu_limit) > 64:
            raise ValueError("OCI release CPU limit is invalid")
        if not _RESOURCE_RE.fullmatch(self.memory_limit):
            raise ValueError("OCI release memory limit is invalid")
        if not 16 <= self.pid_limit <= 4_096:
            raise ValueError("OCI release PID limit must be between 16 and 4096")
        if not 1 <= self.stop_timeout_seconds <= 300:
            raise ValueError("OCI release stop timeout must be between 1 and 300 seconds")
        if not 1 <= self.command_timeout_seconds <= 300:
            raise ValueError("OCI command timeout must be between 1 and 300 seconds")
        if not 0.1 <= self.health_timeout_seconds <= 60:
            raise ValueError("OCI health timeout must be between 0.1 and 60 seconds")
        if self.max_command_output_bytes < 1_024:
            raise ValueError("OCI command output limit is too small")
        if self.max_snapshot_bytes < 1_024 * 1_024:
            raise ValueError("OCI state snapshot byte limit is too small")
        if self.max_snapshot_entries < 100:
            raise ValueError("OCI state snapshot entry limit is too small")
        seen_environment: set[str] = set()
        for name, value in self.production_environment:
            if (
                _ENV_NAME_RE.fullmatch(name) is None
                or name in _RESERVED_ENV_NAMES
                or name in seen_environment
            ):
                raise ValueError("OCI production environment allowlist is invalid")
            if "\x00" in value or "\n" in value or "\r" in value:
                raise ValueError("OCI production environment contains an invalid value")
            seen_environment.add(name)


@dataclass(frozen=True, slots=True)
class _StateFingerprint:
    digest: str
    entries: int
    bytes: int


class RootlessOciReleaseHost:
    """Start approved immutable images with a deny-by-default OCI policy."""

    def __init__(
        self,
        *,
        policy: OciReleasePolicy | None = None,
        runner: OciCommandRunner | None = None,
        http_client: httpx.AsyncClient | None = None,
        release_loader: Callable[[str], ReleaseRecord | None] | None = None,
    ) -> None:
        self._policy = policy or OciReleasePolicy()
        state_root = self._policy.state_root.expanduser().resolve()
        state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._state_root = state_root
        self._runner = runner or LocalOciCommandRunner(cwd=state_root)
        self._http = http_client or httpx.AsyncClient(
            follow_redirects=False,
            timeout=self._policy.health_timeout_seconds,
            trust_env=False,
        )
        self._owns_http = http_client is None
        self._release_loader = release_loader
        self._prepared: dict[str, ReleaseRecord] = {}
        self._releases: dict[str, ReleaseRecord] = {}
        self._running: dict[str, RunningRelease] = {}
        self._instance_networks: dict[str, str] = {}
        self._ready = False
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def prepare(self, release: ReleaseRecord) -> PreparedRelease:
        async with self._lock:
            await self._ensure_backend()
            if not _DIGEST_RE.fullmatch(release.artifact_digest):
                raise ReleaseHostError("release artifact must be an immutable SHA-256 image")
            result = await self._command(
                "image",
                "inspect",
                release.artifact_digest,
                "--format",
                "{{json .}}",
            )
            document = self._json_object(result.output, label="OCI image inspection")
            image_id = str(document.get("Id") or "").lower()
            if image_id != release.artifact_digest:
                raise ReleaseHostError("local OCI image ID does not match the approved artifact")
            config = document.get("Config")
            labels = config.get("Labels") if isinstance(config, dict) else None
            labels = labels if isinstance(labels, dict) else {}
            volumes = config.get("Volumes") if isinstance(config, dict) else None
            if isinstance(volumes, dict) and volumes:
                raise ReleaseHostError("OCI images with implicit writable volumes are not allowed")
            required = {
                "org.opentulpa.release.manifest-digest": release.manifest_digest,
                "org.opentulpa.release.source-commit": release.source_commit,
                "org.opentulpa.release.protocol-version": str(release.protocol_version),
            }
            source_layout = str(labels.get(_SOURCE_LAYOUT_LABEL) or "")
            if (
                any(str(labels.get(key) or "") != value for key, value in required.items())
                or source_layout not in _TRUSTED_SOURCE_LAYOUT_VERSIONS
            ):
                raise ReleaseHostError("OCI image labels do not match the approved release")
            self._validate_mounts()
            token = f"oci:{release.id}:{release.artifact_digest}"
            self._prepared[token] = release
            self._releases[release.id] = release
            return PreparedRelease(
                release_id=release.id,
                artifact_digest=release.artifact_digest,
                token=token,
            )

    async def start(
        self,
        prepared: PreparedRelease,
        context: ReleaseLaunchContext,
    ) -> RunningRelease:
        async with self._lock:
            release = self._prepared.get(prepared.token)
            if (
                release is None
                or release.id != prepared.release_id
                or release.artifact_digest != prepared.artifact_digest
            ):
                raise ReleaseHostError("prepared release token is invalid or expired")
            name_fragment = re.sub(r"[^a-zA-Z0-9_.-]+", "-", release.id)[:40]
            container_name = f"opentulpa-{name_fragment}-{context.mode}-{uuid4().hex[:10]}"
            mount_options = [
                self._validated_mount_option(mount)
                for mount in self._policy.mounts
                if not mount.production_only or context.secrets_enabled
            ]
            owned_network: str | None = None
            if context.mode == "staging":
                network_name = f"{self._policy.network_name}-{uuid4().hex[:10]}"
                await self._create_internal_network(network_name)
                owned_network = network_name
            else:
                network_name = str(self._policy.production_network_name or "").strip()
                if not network_name:
                    raise ReleaseHostError(
                        "production release egress network must be configured explicitly"
                    )
                await self._require_egress_network(network_name)
            lease_label = str(context.lease_epoch) if context.lease_epoch is not None else "none"
            control_token = secrets.token_urlsafe(32)
            environment = self._release_environment(
                release,
                context=context,
                lease_label=lease_label,
                control_token=control_token,
            )
            environment_file = self._write_environment_file(environment)
            argv = [
                self._policy.container_cli,
                "run",
                "--detach",
                "--name",
                container_name,
                "--init",
                "--pull",
                "never",
                "--restart",
                "no",
                "--read-only",
                "--workdir",
                "/app",
                "--network",
                network_name,
                "--ipc",
                "none",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--cpus",
                self._policy.cpu_limit,
                "--memory",
                self._policy.memory_limit,
                "--memory-swap",
                self._policy.memory_limit,
                "--pids-limit",
                str(self._policy.pid_limit),
                "--ulimit",
                "nofile=1024:1024",
                "--ulimit",
                "core=0:0",
                "--user",
                self._policy.runtime_user,
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777",
                "--tmpfs",
                "/run:rw,noexec,nosuid,nodev,size=32m,mode=755",
                "--publish",
                f"127.0.0.1::{release.control_port}/tcp",
                "--label",
                f"{_RELEASE_LABEL}={release.id}",
                "--label",
                f"{_MODE_LABEL}={context.mode}",
                "--label",
                f"{_LEASE_LABEL}={lease_label}",
                "--label",
                f"{_ARTIFACT_LABEL}={release.artifact_digest}",
                "--label",
                f"{_SOURCE_LABEL}={release.source_commit}",
                "--label",
                f"{_NETWORK_LABEL}={network_name}",
                "--label",
                f"{_NETWORK_OWNED_LABEL}={str(owned_network is not None).lower()}",
                "--env-file",
                str(environment_file),
            ]
            if context.mode == "staging":
                runtime_uid, runtime_gid = self._policy.runtime_user.split(":", 1)
                argv.extend(
                    (
                        "--tmpfs",
                        "/workspace:rw,nosuid,nodev,size=512m,mode=700,"
                        f"uid={runtime_uid},gid={runtime_gid}",
                    )
                )
            else:
                argv.extend(
                    (
                        "--add-host",
                        f"{self._policy.host_gateway_name}:{self._policy.host_gateway_address}",
                    )
                )
            for option in mount_options:
                argv.extend(("--mount", option))
            argv.extend(
                (
                    "--entrypoint",
                    release.entrypoint[0],
                    release.artifact_digest,
                    *release.entrypoint[1:],
                )
            )
            try:
                result = await self._runner.run(
                    argv,
                    timeout_seconds=self._policy.command_timeout_seconds,
                    max_output_bytes=self._policy.max_command_output_bytes,
                )
            except BaseException:
                if owned_network is not None:
                    await self._remove_network(owned_network)
                raise
            finally:
                environment_file.unlink(missing_ok=True)
            container_id = result.output.decode("ascii", errors="ignore").strip().lower()
            if (
                result.returncode != 0
                or result.truncated
                or result.timed_out
                or not _CONTAINER_ID_RE.fullmatch(container_id)
            ):
                await self._force_remove(container_name, network_name=owned_network)
                raise ReleaseHostError("OCI release container did not start")
            try:
                endpoint = await self._published_endpoint(container_id, release.control_port)
            except Exception:
                await self._force_remove(container_id, network_name=owned_network)
                raise
            running = RunningRelease(
                release_id=release.id,
                instance_id=container_id,
                mode=context.mode,
                lease_epoch=context.lease_epoch,
                endpoint=endpoint,
                control_token=control_token,
            )
            self._running[container_id] = running
            if owned_network is not None:
                self._instance_networks[container_id] = owned_network
            return running

    async def probe(self, running: RunningRelease) -> ReleaseHealth:
        release = self._required_release(running.release_id)
        endpoint = self._required_endpoint(running)
        try:
            status_code, payload = await self._request_json(
                "GET",
                f"{endpoint}{release.health_path}",
                timeout_seconds=self._policy.health_timeout_seconds,
                headers=self._control_headers(running),
            )
            protocol_version = int(payload.get("protocol_version") or 0)
        except (httpx.HTTPError, ValueError, TypeError, ReleaseHostError):
            return self._unhealthy(release.id, "Release health endpoint was unavailable.")
        if status_code != 200:
            return self._unhealthy(release.id, "Release health endpoint rejected the probe.")
        components_raw = payload.get("components")
        components = (
            {str(key): bool(value) for key, value in components_raw.items()}
            if isinstance(components_raw, dict)
            else {}
        )
        return ReleaseHealth(
            healthy=bool(payload.get("healthy", payload.get("status") == "ok")),
            release_id=str(payload.get("release_id") or ""),
            protocol_version=protocol_version,
            summary=str(payload.get("summary") or "")[:2_000],
            components=components,
        )

    async def drain(
        self,
        running: RunningRelease,
        *,
        timeout_seconds: float,
    ) -> DrainResult:
        release = self._required_release(running.release_id)
        endpoint = self._required_endpoint(running)
        try:
            status_code, payload = await self._request_json(
                "POST",
                f"{endpoint}{release.drain_path}",
                timeout_seconds=max(self._policy.health_timeout_seconds, timeout_seconds + 1),
                json_body={"timeout_seconds": max(0.0, float(timeout_seconds))},
                headers=self._control_headers(running),
            )
            in_flight = max(0, int(payload.get("in_flight", 0)))
        except (httpx.HTTPError, ValueError, TypeError, ReleaseHostError):
            return DrainResult(drained=False, in_flight=1)
        if status_code != 200:
            return DrainResult(drained=False, in_flight=1)
        return DrainResult(
            drained=bool(payload.get("drained", False)),
            in_flight=in_flight,
        )

    async def stop(self, running: RunningRelease) -> None:
        async with self._lock:
            network_name = self._instance_networks.get(running.instance_id)
            stop = await self._runner.run(
                (
                    self._policy.container_cli,
                    "stop",
                    "--time",
                    str(self._policy.stop_timeout_seconds),
                    running.instance_id,
                ),
                timeout_seconds=self._policy.stop_timeout_seconds + 10,
                max_output_bytes=self._policy.max_command_output_bytes,
            )
            remove = await self._runner.run(
                (self._policy.container_cli, "rm", "--force", running.instance_id),
                timeout_seconds=self._policy.command_timeout_seconds,
                max_output_bytes=self._policy.max_command_output_bytes,
            )
            self._running.pop(running.instance_id, None)
            self._instance_networks.pop(running.instance_id, None)
            network_removed = True
            if network_name is not None:
                network_removed = await self._remove_network(network_name)
            if (remove.returncode != 0 and stop.returncode != 0) or not network_removed:
                raise ReleaseHostError("OCI release container could not be stopped")

    async def snapshot_state(self, activation_id: str) -> None:
        """Snapshot only release-coupled worker state, never shared product data."""

        async with self._lock:
            workspace = self._release_state_directory()
            snapshot = self._snapshot_path(activation_id)
            await asyncio.to_thread(
                self._create_state_snapshot,
                activation_id,
                workspace,
                snapshot,
            )

    async def restore_state(self, activation_id: str) -> bool:
        """Restore release-coupled worker state without rewinding product data."""

        async with self._lock:
            workspace = self._release_state_directory()
            snapshot = self._snapshot_path(activation_id)
            if not snapshot.exists():
                return False
            await asyncio.to_thread(
                self._restore_state_snapshot,
                activation_id,
                workspace,
                snapshot,
            )
            return True

    async def discard_state_snapshot(self, activation_id: str) -> None:
        async with self._lock:
            snapshot = self._snapshot_path(activation_id)
            await asyncio.to_thread(shutil.rmtree, snapshot, True)

    async def discover(
        self,
        release_id: str,
        *,
        mode: str = "production",
    ) -> RunningRelease | None:
        async with self._lock:
            await self._ensure_backend()
            if mode not in {"staging", "production"}:
                raise ReleaseHostError("OCI release mode is invalid")
            for running in self._running.values():
                if running.release_id == release_id and running.mode == mode:
                    return running
            result = await self._runner.run(
                (
                    self._policy.container_cli,
                    "ps",
                    "--filter",
                    f"label={_RELEASE_LABEL}={release_id}",
                    "--filter",
                    f"label={_MODE_LABEL}={mode}",
                    "--format",
                    "{{.ID}}",
                ),
                timeout_seconds=self._policy.command_timeout_seconds,
                max_output_bytes=self._policy.max_command_output_bytes,
            )
            if result.returncode != 0 or result.truncated:
                raise ReleaseHostError("OCI running release discovery failed")
            identifiers = [line.strip().lower() for line in result.output.decode().splitlines() if line]
            if not identifiers:
                return None
            if len(identifiers) != 1 or not _CONTAINER_ID_RE.fullmatch(identifiers[0]):
                raise ReleaseHostError("OCI discovery found ambiguous release containers")
            container_id = identifiers[0]
            inspect_result = await self._command(
                "inspect",
                container_id,
                "--format",
                "{{json .}}",
            )
            document = self._json_object(inspect_result.output, label="OCI container inspection")
            config = document.get("Config")
            labels = config.get("Labels") if isinstance(config, dict) else None
            labels = labels if isinstance(labels, dict) else {}
            environment = config.get("Env") if isinstance(config, dict) else None
            if labels.get(_RELEASE_LABEL) != release_id or labels.get(_MODE_LABEL) != mode:
                raise ReleaseHostError("OCI discovered container labels do not match")
            lease_raw = str(labels.get(_LEASE_LABEL) or "none")
            release = self._required_release(release_id)
            if (
                str(document.get("Image") or "").lower() != release.artifact_digest
                or labels.get(_ARTIFACT_LABEL) != release.artifact_digest
                or labels.get(_SOURCE_LABEL) != release.source_commit
            ):
                raise ReleaseHostError("OCI discovered container artifact does not match")
            try:
                lease_epoch = None if lease_raw == "none" else int(lease_raw)
            except ValueError as exc:
                raise ReleaseHostError("OCI discovered container lease is invalid") from exc
            network_name = str(labels.get(_NETWORK_LABEL) or "")
            network_owned = str(labels.get(_NETWORK_OWNED_LABEL) or "").casefold() == "true"
            if mode == "staging":
                if not network_owned or not self._is_instance_network(network_name):
                    raise ReleaseHostError("OCI discovered staging network is invalid")
                await self._require_internal_network(network_name)
            else:
                if network_owned or network_name != self._policy.production_network_name:
                    raise ReleaseHostError("OCI discovered production network is invalid")
                await self._require_egress_network(network_name)
            control_token = self._environment_value(environment, "OPENTULPA_CONTROL_TOKEN")
            if not re.fullmatch(r"[A-Za-z0-9_-]{32,200}", control_token):
                raise ReleaseHostError("OCI discovered container control token is invalid")
            endpoint = await self._published_endpoint(container_id, release.control_port)
            running = RunningRelease(
                release_id=release_id,
                instance_id=container_id,
                mode=cast(Literal["staging", "production"], mode),
                lease_epoch=lease_epoch,
                endpoint=endpoint,
                control_token=control_token,
            )
            self._running[container_id] = running
            if network_owned:
                self._instance_networks[container_id] = network_name
            return running

    def _release_state_directory(self) -> Path:
        candidates = tuple(
            mount
            for mount in self._policy.mounts
            if mount.target == self._policy.data_mount_target
            and not mount.read_only
            and mount.production_only
        )
        if len(candidates) != 1:
            raise ReleaseHostError("OCI release requires one writable persistent workspace")
        self._validated_mount_option(candidates[0])
        raw = candidates[0].source.expanduser()
        if raw.is_symlink():
            raise ReleaseHostError("OCI persistent workspace cannot be a symlink")
        workspace = raw.resolve(strict=True)
        if not workspace.is_dir():
            raise ReleaseHostError("OCI persistent workspace must be a directory")
        if (
            workspace == self._state_root
            or workspace.is_relative_to(self._state_root)
            or self._state_root.is_relative_to(workspace)
        ):
            raise ReleaseHostError("OCI workspace and bootstrap state roots must be disjoint")
        state = workspace / ".opentulpa" / "deepagents" / "capability_state"
        if state.is_symlink():
            raise ReleaseHostError("release capability state cannot be a symlink")
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved_state = state.resolve(strict=True)
        if not resolved_state.is_dir() or not resolved_state.is_relative_to(workspace):
            raise ReleaseHostError("release capability state escaped the persistent workspace")
        return resolved_state

    def _snapshot_path(self, activation_id: str) -> Path:
        safe_id = str(activation_id or "")
        if (
            not safe_id
            or len(safe_id) > 100
            or any(ord(character) < 32 or ord(character) == 127 for character in safe_id)
        ):
            raise ReleaseHostError("state snapshot activation identity is invalid")
        root = self._state_root / "workspace-snapshots"
        if root.is_symlink():
            raise ReleaseHostError("state snapshot root cannot be a symlink")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        resolved = root.resolve(strict=True)
        if resolved.parent != self._state_root:
            raise ReleaseHostError("state snapshot root escaped bootstrap storage")
        digest = hashlib.sha256(safe_id.encode("utf-8")).hexdigest()
        return resolved / digest

    def _create_state_snapshot(
        self,
        activation_id: str,
        workspace: Path,
        snapshot: Path,
    ) -> None:
        if snapshot.is_symlink():
            raise ReleaseHostError("state snapshot cannot be a symlink")
        if snapshot.exists():
            self._load_state_snapshot(activation_id, workspace, snapshot)
            return
        temporary = snapshot.parent / f".{snapshot.name}.tmp-{uuid4().hex}"
        temporary.mkdir(mode=0o700)
        try:
            source = _fingerprint_state_tree(
                workspace,
                max_bytes=self._policy.max_snapshot_bytes,
                max_entries=self._policy.max_snapshot_entries,
            )
            data = temporary / "data"
            shutil.copytree(
                workspace,
                data,
                symlinks=True,
                copy_function=shutil.copy2,
            )
            copied = _fingerprint_state_tree(
                data,
                max_bytes=self._policy.max_snapshot_bytes,
                max_entries=self._policy.max_snapshot_entries,
            )
            if copied != source:
                raise ReleaseHostError("persistent workspace changed while snapshotting")
            manifest = {
                "activation_id": activation_id,
                "bytes": source.bytes,
                "digest": source.digest,
                "entries": source.entries,
                "version": 1,
                "workspace": hashlib.sha256(str(workspace).encode("utf-8")).hexdigest(),
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )
            manifest_path.chmod(0o600)
            if snapshot.exists():
                self._load_state_snapshot(activation_id, workspace, snapshot)
                return
            os.replace(temporary, snapshot)
        except ReleaseHostError:
            raise
        except (OSError, ValueError) as exc:
            raise ReleaseHostError("persistent workspace snapshot failed") from exc
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def _restore_state_snapshot(
        self,
        activation_id: str,
        workspace: Path,
        snapshot: Path,
    ) -> None:
        expected = self._load_state_snapshot(activation_id, workspace, snapshot)
        data = snapshot / "data"
        staging = workspace.parent / f".{workspace.name}.restore-{uuid4().hex}"
        backup = workspace.parent / f".{workspace.name}.failed-{uuid4().hex}"
        try:
            shutil.copytree(
                data,
                staging,
                symlinks=True,
                copy_function=shutil.copy2,
            )
            copied = _fingerprint_state_tree(
                staging,
                max_bytes=self._policy.max_snapshot_bytes,
                max_entries=self._policy.max_snapshot_entries,
            )
            if copied != expected:
                raise ReleaseHostError("state snapshot copy failed integrity verification")
            os.replace(workspace, backup)
            try:
                os.replace(staging, workspace)
            except BaseException:
                os.replace(backup, workspace)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        except ReleaseHostError:
            raise
        except (OSError, ValueError) as exc:
            raise ReleaseHostError("persistent workspace restore failed") from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not workspace.exists():
                with suppress(OSError):
                    os.replace(backup, workspace)

    def _load_state_snapshot(
        self,
        activation_id: str,
        workspace: Path,
        snapshot: Path,
    ) -> _StateFingerprint:
        if snapshot.is_symlink() or not snapshot.is_dir():
            raise ReleaseHostError("state snapshot storage is invalid")
        manifest_path = snapshot / "manifest.json"
        data = snapshot / "data"
        if manifest_path.is_symlink() or data.is_symlink() or not data.is_dir():
            raise ReleaseHostError("state snapshot storage is invalid")
        try:
            document = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ReleaseHostError("state snapshot manifest is invalid") from exc
        workspace_digest = hashlib.sha256(str(workspace).encode("utf-8")).hexdigest()
        if (
            not isinstance(document, dict)
            or set(document)
            != {"activation_id", "bytes", "digest", "entries", "version", "workspace"}
            or document.get("version") != 1
            or document.get("activation_id") != activation_id
            or document.get("workspace") != workspace_digest
            or not isinstance(document.get("bytes"), int)
            or not isinstance(document.get("entries"), int)
            or not re.fullmatch(r"[0-9a-f]{64}", str(document.get("digest") or ""))
        ):
            raise ReleaseHostError("state snapshot manifest is invalid")
        expected = _StateFingerprint(
            digest=str(document["digest"]),
            entries=int(document["entries"]),
            bytes=int(document["bytes"]),
        )
        actual = _fingerprint_state_tree(
            data,
            max_bytes=self._policy.max_snapshot_bytes,
            max_entries=self._policy.max_snapshot_entries,
        )
        if actual != expected:
            raise ReleaseHostError("state snapshot failed integrity verification")
        return expected

    async def _ensure_backend(self) -> None:
        if self._ready:
            return
        engine = Path(self._policy.container_cli).name
        if engine == "docker":
            result = await self._command("info", "--format", "{{json .SecurityOptions}}")
            if "rootless" not in result.output.decode("utf-8", errors="ignore").casefold():
                raise ReleaseHostError("Docker engine is not operating in rootless mode")
        else:
            result = await self._command("info", "--format", "{{.Host.Security.Rootless}}")
            if result.output.decode("ascii", errors="ignore").strip().casefold() != "true":
                raise ReleaseHostError("Podman engine is not operating in rootless mode")
        self._ready = True

    async def _create_internal_network(self, network_name: str) -> None:
        if not self._is_instance_network(network_name):
            raise ReleaseHostError("OCI release network name is invalid")
        created = await self._command("network", "create", "--internal", network_name)
        network_id = created.output.decode("ascii", errors="ignore").strip().lower()
        if not _CONTAINER_ID_RE.fullmatch(network_id):
            await self._remove_network(network_name)
            raise ReleaseHostError("internal OCI release network was not created")
        try:
            await self._require_internal_network(network_name)
        except BaseException:
            await self._remove_network(network_name)
            raise

    async def _require_internal_network(self, network_name: str) -> None:
        result = await self._command(
            "network",
            "inspect",
            network_name,
            "--format",
            "{{.Internal}}",
        )
        if result.output.decode("ascii", errors="ignore").strip().casefold() != "true":
            raise ReleaseHostError("OCI release network is not internal-only")

    async def _require_egress_network(self, network_name: str) -> None:
        if _NETWORK_RE.fullmatch(network_name) is None:
            raise ReleaseHostError("OCI production egress network is invalid")
        result = await self._command(
            "network",
            "inspect",
            network_name,
            "--format",
            "{{.Internal}}",
        )
        if result.output.decode("ascii", errors="ignore").strip().casefold() != "false":
            raise ReleaseHostError("OCI production network must permit declared egress")

    async def _remove_network(self, network_name: str) -> bool:
        if not self._is_instance_network(network_name):
            return False
        result = await self._runner.run(
            (self._policy.container_cli, "network", "rm", network_name),
            timeout_seconds=self._policy.command_timeout_seconds,
            max_output_bytes=self._policy.max_command_output_bytes,
        )
        return result.returncode == 0

    def _is_instance_network(self, network_name: str) -> bool:
        return re.fullmatch(rf"{re.escape(self._policy.network_name)}-[0-9a-f]{{10}}", network_name) is not None

    def _validate_mounts(self) -> None:
        for mount in self._policy.mounts:
            self._validated_mount_option(mount)
        if self._policy.require_persistent_data_mount and not any(
            mount.target == self._policy.data_mount_target
            and not mount.read_only
            and mount.production_only
            for mount in self._policy.mounts
        ):
            raise ReleaseHostError("production release requires a writable data mount")

    def _validated_mount_option(self, mount: OciMount) -> str:
        roots = tuple(
            root.expanduser().resolve(strict=True) for root in self._policy.allowed_mount_roots
        )
        source_raw = mount.source.expanduser()
        if source_raw.is_symlink() or "," in str(source_raw):
            raise ReleaseHostError("OCI release mount source is invalid")
        source = source_raw.resolve(strict=True)
        source_mode = source.stat().st_mode
        if stat.S_ISSOCK(source_mode) or source.name in {
            ".env",
            "docker.sock",
            "podman.sock",
            "containerd.sock",
        }:
            raise ReleaseHostError("OCI release mount source is forbidden")
        if source.is_dir() and (source / ".git").exists():
            raise ReleaseHostError("OCI release source repositories cannot be mounted")
        if not roots or not any(source == root or source.is_relative_to(root) for root in roots):
            raise ReleaseHostError("OCI release mount escaped the allowlist")
        if not _SAFE_TARGET_RE.fullmatch(mount.target) or "," in mount.target:
            raise ReleaseHostError("OCI release mount target is invalid")
        target = PurePosixPath(mount.target)
        if (
            target.root != "/"
            or target.as_posix() != mount.target
            or ".." in target.parts
            or not (
                mount.target == "/workspace"
                or mount.target.startswith("/workspace/")
            )
        ):
            raise ReleaseHostError("OCI release mount target is reserved")
        if not mount.read_only and not (
            mount.target == "/workspace" or mount.target.startswith("/workspace/")
        ):
            raise ReleaseHostError("writable OCI mounts are restricted to /workspace")
        if not mount.read_only and not mount.production_only:
            raise ReleaseHostError("writable OCI mounts must be production-only")
        if not mount.production_only:
            raise ReleaseHostError("OCI release mounts must be production-only")
        return (
            f"type=bind,src={source},dst={mount.target}"
            + (",readonly" if mount.read_only else "")
        )

    async def _published_endpoint(self, container_id: str, port: int) -> str:
        for attempt in range(20):
            result = await self._runner.run(
                (
                    self._policy.container_cli,
                    "port",
                    container_id,
                    f"{port}/tcp",
                ),
                timeout_seconds=self._policy.command_timeout_seconds,
                max_output_bytes=2_048,
            )
            value = result.output.decode("ascii", errors="ignore").strip()
            match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", value)
            if result.returncode == 0 and match is not None:
                published = int(match.group(1))
                if 1 <= published <= 65_535:
                    return f"http://127.0.0.1:{published}"
            if attempt < 19:
                await asyncio.sleep(0.05)
        raise ReleaseHostError("OCI release did not publish a loopback control port")

    def _release_environment(
        self,
        release: ReleaseRecord,
        *,
        context: ReleaseLaunchContext,
        lease_label: str,
        control_token: str,
    ) -> dict[str, str]:
        environment = {
            "HOST": "0.0.0.0",
            "PORT": str(release.control_port),
            "OPENTULPA_RELEASE_ID": release.id,
            "OPENTULPA_RELEASE_MODE": context.mode,
            "OPENTULPA_LEASE_EPOCH": lease_label,
            "OPENTULPA_CONTROL_TOKEN": control_token,
            "OPENTULPA_INGRESS_ENABLED": str(context.ingress_enabled).lower(),
            "OPENTULPA_SECRETS_ENABLED": str(context.secrets_enabled).lower(),
            "OPENTULPA_HEALTH_PATH": release.health_path,
            "OPENTULPA_DRAIN_PATH": release.drain_path,
            "OPENTULPA_INGRESS_PATH": release.ingress_path,
            "OPENTULPA_EVENT_PATH": release.event_path,
            "OPENTULPA_DATA_ROOT": self._policy.data_mount_target,
            "OPENTULPA_MANAGED_RELEASE": "true",
            # The candidate source is an immutable image layer. Never import from the
            # writable workspace or implicitly prepend the working directory.
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": "/app/src",
            "PYTHONSAFEPATH": "1",
        }
        if context.mode == "staging":
            environment.update(
                {
                    "OPENAI_COMPATIBLE_API_KEY": "staging-disabled",
                    "OPENAI_COMPATIBLE_BASE_URL": "http://127.0.0.1:9/v1",
                    "OPENTULPA_DISABLE_CONSUMERS": "true",
                    "EVOLUTION_ENABLED": "false",
                }
            )
        else:
            environment.update(dict(self._policy.production_environment))
            environment.setdefault("EVOLUTION_ENABLED", "false")
        return environment

    def _write_environment_file(self, environment: dict[str, str]) -> Path:
        root = self._state_root / "environment"
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = root / f"release-{uuid4().hex}.env"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                for name, value in sorted(environment.items()):
                    if _ENV_NAME_RE.fullmatch(name) is None or any(
                        character in value for character in ("\x00", "\n", "\r")
                    ):
                        raise ReleaseHostError("OCI release environment is invalid")
                    stream.write(f"{name}={value}\n")
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path

    async def _force_remove(self, identifier: str, *, network_name: str | None = None) -> None:
        try:
            await self._runner.run(
                (self._policy.container_cli, "rm", "--force", identifier),
                timeout_seconds=self._policy.command_timeout_seconds,
                max_output_bytes=self._policy.max_command_output_bytes,
            )
        finally:
            if network_name is not None:
                await self._remove_network(network_name)

    async def _command(self, *args: str) -> OciCommandResult:
        result = await self._runner.run(
            (self._policy.container_cli, *args),
            timeout_seconds=self._policy.command_timeout_seconds,
            max_output_bytes=self._policy.max_command_output_bytes,
        )
        if result.returncode != 0 or result.truncated or result.timed_out:
            raise ReleaseHostError("OCI command failed")
        return result

    def _required_release(self, release_id: str) -> ReleaseRecord:
        release = self._releases.get(release_id)
        if release is None and self._release_loader is not None:
            release = self._release_loader(release_id)
            if release is not None and release.id == release_id:
                self._releases[release_id] = release
        if release is None:
            raise ReleaseHostError("persisted release metadata is unavailable")
        return release

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        timeout_seconds: float,
        json_body: dict[str, float] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        retained = bytearray()
        async with self._http.stream(
            method,
            url,
            json=json_body,
            headers=headers,
            timeout=timeout_seconds,
        ) as response:
            async for chunk in response.aiter_bytes():
                if len(retained) + len(chunk) > 65_536:
                    raise ReleaseHostError("release control response exceeded its byte limit")
                retained.extend(chunk)
            status_code = response.status_code
        try:
            payload = json.loads(retained) if retained else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseHostError("release control response was invalid") from exc
        if not isinstance(payload, dict):
            raise ReleaseHostError("release control response was not an object")
        return status_code, payload

    @staticmethod
    def _control_headers(running: RunningRelease) -> dict[str, str]:
        headers = {
            "X-OpenTulpa-Release-ID": running.release_id,
            "X-OpenTulpa-Lease-Epoch": str(running.lease_epoch or "none"),
        }
        if running.control_token is not None:
            headers["Authorization"] = f"Bearer {running.control_token}"
        return headers

    @staticmethod
    def _environment_value(environment: object, name: str) -> str:
        if not isinstance(environment, list):
            return ""
        prefix = f"{name}="
        matches = [
            value.removeprefix(prefix)
            for value in environment
            if isinstance(value, str) and value.startswith(prefix)
        ]
        return matches[0] if len(matches) == 1 else ""

    @staticmethod
    def _required_endpoint(running: RunningRelease) -> str:
        if running.endpoint is None:
            raise ReleaseHostError("running release has no loopback endpoint")
        return running.endpoint

    @staticmethod
    def _json_object(raw: bytes, *, label: str) -> dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseHostError(f"{label} was invalid") from exc
        if not isinstance(value, dict):
            raise ReleaseHostError(f"{label} was not an object")
        return value

    @staticmethod
    def _unhealthy(release_id: str, summary: str) -> ReleaseHealth:
        return ReleaseHealth(
            healthy=False,
            release_id=release_id,
            summary=summary,
            components={"runtime": False, "agent_api": False},
        )


def _fingerprint_state_tree(
    root: Path,
    *,
    max_bytes: int,
    max_entries: int,
) -> _StateFingerprint:
    if root.is_symlink() or not root.is_dir():
        raise ReleaseHostError("state snapshot tree root is invalid")
    digest = hashlib.sha256()
    entry_count = 0
    byte_count = 0
    pending: list[tuple[Path, PurePosixPath]] = [(root, PurePosixPath())]
    try:
        while pending:
            directory, prefix = pending.pop()
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
            children: list[tuple[Path, PurePosixPath]] = []
            for entry in entries:
                entry_count += 1
                if entry_count > max_entries:
                    raise ReleaseHostError("persistent workspace exceeds snapshot entry limit")
                relative = prefix / entry.name
                encoded_path = relative.as_posix().encode("utf-8")
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISDIR(mode):
                    digest.update(b"D\0" + encoded_path + b"\0")
                    digest.update(str(stat.S_IMODE(mode)).encode("ascii") + b"\0")
                    children.append((Path(entry.path), relative))
                    continue
                if stat.S_ISLNK(mode):
                    target = os.readlink(entry.path)
                    digest.update(b"L\0" + encoded_path + b"\0" + os.fsencode(target) + b"\0")
                    continue
                if not stat.S_ISREG(mode):
                    raise ReleaseHostError("persistent workspace contains a special file")
                byte_count += metadata.st_size
                if byte_count > max_bytes:
                    raise ReleaseHostError("persistent workspace exceeds snapshot byte limit")
                digest.update(b"F\0" + encoded_path + b"\0")
                digest.update(
                    f"{stat.S_IMODE(mode)}:{metadata.st_size}\0".encode("ascii")
                )
                flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(entry.path, flags)
                with os.fdopen(descriptor, "rb") as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode) or opened.st_size != metadata.st_size:
                        raise ReleaseHostError("persistent workspace changed while snapshotting")
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                digest.update(b"\0")
            pending.extend(reversed(children))
    except ReleaseHostError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ReleaseHostError("persistent workspace could not be fingerprinted") from exc
    return _StateFingerprint(
        digest=digest.hexdigest(),
        entries=entry_count,
        bytes=byte_count,
    )


__all__ = [
    "LocalOciCommandRunner",
    "OciCommandResult",
    "OciCommandRunner",
    "OciMount",
    "OciReleasePolicy",
    "RootlessOciReleaseHost",
]
