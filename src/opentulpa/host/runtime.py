"""Child Deep Agents runtime lifecycle owned by the stable host."""

from __future__ import annotations

import asyncio
import ctypes
import fcntl
import inspect
import json
import logging
import math
import os
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import httpx
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from opentulpa.evolution.generation_store import GenerationStore, InstalledGeneration
from opentulpa.evolution.models import EvolutionEvent
from opentulpa.host.models import HostConfig

logger = logging.getLogger(__name__)

_SECRET_LINE = re.compile(
    r"(?i)(api[_-]?key|authorization|bot[_-]?token|secret|password)(\s*[:=]\s*)(\S+)"
)
_GENERATION_ID_PATTERN = r"^[0-9a-f]{64}$"
_DIGEST_PATTERN = r"^sha256:[0-9a-f]{64}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_PROFILE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
_OWNERSHIP_MAX_BYTES = 64 * 1024
_PLATFORM_ENVIRONMENT_ALLOWLIST = frozenset(
    {
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
    }
)
_TRUSTED_SYSTEM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_SETPRIV_PATH = Path("/usr/bin/setpriv")
_DEDICATED_RUNTIME_UID = 65532
_DEDICATED_RUNTIME_GID = 65532


class RuntimeUnavailableError(RuntimeError):
    """The mutable child runtime did not become healthy."""


class _ChildExitedBeforeReadyError(RuntimeUnavailableError):
    """The nonce-bound candidate exited before it could satisfy readiness."""


class _RuntimeProbeError(RuntimeUnavailableError):
    """A nonce-bound runtime did not satisfy one strict health probe."""


class RuntimeGenerationSpec(BaseModel):
    """Controller-held provenance required to launch one immutable generation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
        str_strip_whitespace=True,
    )

    generation_id: str = Field(pattern=_GENERATION_ID_PATTERN)
    expected_manifest_digest: str = Field(
        pattern=_DIGEST_PATTERN,
        validation_alias=AliasChoices("expected_manifest_digest", "manifest_digest"),
    )
    expected_state_contract_digest: str = Field(
        pattern=_SHA256_PATTERN,
        validation_alias=AliasChoices(
            "expected_state_contract_digest",
            "state_contract_digest",
            "state_contract_sha256",
        ),
    )
    expected_evaluator_fingerprint: str = Field(
        pattern=_DIGEST_PATTERN,
        validation_alias=AliasChoices(
            "expected_evaluator_fingerprint",
            "evaluator_fingerprint",
        ),
    )
    expected_install_profile: str = Field(
        min_length=1,
        max_length=100,
        pattern=_PROFILE_PATTERN,
        validation_alias=AliasChoices("expected_install_profile", "install_profile"),
    )
    controller_protocol: int = Field(ge=1)

    @property
    def manifest_digest(self) -> str:
        return self.expected_manifest_digest

    @property
    def state_contract_digest(self) -> str:
        return self.expected_state_contract_digest

    @property
    def evaluator_fingerprint(self) -> str:
        return self.expected_evaluator_fingerprint

    @property
    def install_profile(self) -> str:
        return self.expected_install_profile

    @classmethod
    def from_release_metadata(cls, metadata: Mapping[str, object]) -> Self:
        """Extract only trusted generation provenance from persisted release metadata."""

        image_reference = metadata.get("image_reference")
        generation_id = metadata.get("generation_id")
        if generation_id is None and isinstance(image_reference, str):
            prefix = "python-generation:"
            generation_id = image_reference.removeprefix(prefix) if image_reference.startswith(prefix) else None
        return cls.model_validate(
            {
                "generation_id": generation_id,
                "expected_manifest_digest": metadata.get("expected_manifest_digest")
                or metadata.get("manifest_digest"),
                "expected_state_contract_digest": metadata.get(
                    "expected_state_contract_digest"
                )
                or metadata.get("state_contract_digest")
                or metadata.get("state_contract_sha256"),
                "expected_evaluator_fingerprint": metadata.get(
                    "expected_evaluator_fingerprint"
                )
                or metadata.get("evaluator_fingerprint"),
                "expected_install_profile": metadata.get("expected_install_profile")
                or metadata.get("install_profile"),
                "controller_protocol": metadata.get("controller_protocol"),
            }
        )


class RuntimeLogEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    stream_id: str
    sequence: int
    timestamp: datetime
    stream: str
    text: str


@dataclass(frozen=True, slots=True)
class RuntimeProcessIdentity:
    """Observed OS identity used to prove a recorded child before fencing it."""

    pid: int
    process_group: int
    executable: Path
    argv: tuple[str, ...]
    parent_pid: int | None = None
    process_birth: str | None = None
    launch_nonce: str | None = None


@dataclass(frozen=True, slots=True)
class _LinuxProcessMetadata:
    parent_pid: int
    process_group: int
    process_birth: str
    proc_uid: int
    status_uids: tuple[int, int, int, int]

    def owned_by(self, uid: int) -> bool:
        return self.proc_uid == uid and all(value == uid for value in self.status_uids)


ProcessInspector = Callable[[int], RuntimeProcessIdentity | None]
ProcessFencer = Callable[
    [RuntimeProcessIdentity, signal.Signals],
    Awaitable[None] | None,
]
DescendantInspector = Callable[[int, str], tuple[RuntimeProcessIdentity, ...]]
ProcessSignaler = Callable[[int, int], None]


class _OwnershipRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format_version: Literal[1] = 1
    pid: int = Field(ge=1)
    process_group: int = Field(ge=1)
    host_pid: int = Field(ge=1)
    host_birth: str = Field(min_length=1, max_length=200)
    mode: Literal["generation", "legacy"]
    generation_id: str | None = Field(default=None, pattern=_GENERATION_ID_PATTERN)
    legacy_source_root: str | None = None
    launch_nonce: str = Field(min_length=16, max_length=200)
    process_birth: str = Field(min_length=1, max_length=200)
    executable: str = Field(min_length=1, max_length=4_096)
    argv: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _coherent_identity(self) -> Self:
        if (self.mode == "generation") != (self.generation_id is not None):
            raise ValueError("generation ownership identity is incomplete")
        if (self.mode == "legacy") != (self.legacy_source_root is not None):
            raise ValueError("legacy ownership identity is incomplete")
        if not Path(self.executable).is_absolute() or "\x00" in self.executable:
            raise ValueError("owned executable must be an absolute path")
        if self.legacy_source_root is not None and (
            not Path(self.legacy_source_root).is_absolute() or "\x00" in self.legacy_source_root
        ):
            raise ValueError("owned legacy source root must be absolute")
        if any(not value or "\x00" in value or len(value) > 4_096 for value in self.argv):
            raise ValueError("owned command is invalid")
        return self


class _LaunchIntent(BaseModel):
    """Fail-closed marker covering the unavoidable spawn-to-PID-record gap."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format_version: Literal[1] = 1
    host_pid: int = Field(ge=1)
    mode: Literal["generation", "legacy"]
    generation_id: str | None = Field(default=None, pattern=_GENERATION_ID_PATTERN)
    legacy_source_root: str | None = None
    launch_nonce: str = Field(min_length=16, max_length=200)
    executable: str = Field(min_length=1, max_length=4_096)
    argv: tuple[str, ...] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _coherent_identity(self) -> Self:
        if (self.mode == "generation") != (self.generation_id is not None):
            raise ValueError("generation launch intent is incomplete")
        if (self.mode == "legacy") != (self.legacy_source_root is not None):
            raise ValueError("legacy launch intent is incomplete")
        if not Path(self.executable).is_absolute() or "\x00" in self.executable:
            raise ValueError("launch intent executable must be absolute")
        if any(not value or "\x00" in value or len(value) > 4_096 for value in self.argv):
            raise ValueError("launch intent command is invalid")
        return self


@dataclass(frozen=True, slots=True)
class _LaunchTarget:
    generation: RuntimeGenerationSpec | None
    project_root: Path | None

    @classmethod
    def for_generation(cls, generation: RuntimeGenerationSpec) -> _LaunchTarget:
        return cls(generation=generation, project_root=None)

    @classmethod
    def for_source(cls, project_root: Path) -> _LaunchTarget:
        return cls(generation=None, project_root=project_root)


@dataclass(slots=True)
class _Child:
    process: asyncio.subprocess.Process
    endpoint: str
    config: HostConfig
    project_root: Path | None
    generation: RuntimeGenerationSpec | None
    installed_generation: InstalledGeneration | None
    launch_nonce: str
    process_group: int
    process_birth: str
    executable: Path
    argv: tuple[str, ...]
    readers: tuple[asyncio.Task[None], ...]
    watcher: asyncio.Task[None] | None = None
    requested_stop: bool = False

    @property
    def generation_id(self) -> str | None:
        return self.generation.generation_id if self.generation is not None else None

    @property
    def target(self) -> _LaunchTarget:
        if self.generation is not None:
            return _LaunchTarget.for_generation(self.generation)
        if self.project_root is None:
            raise RuntimeError("legacy child has no source identity")
        return _LaunchTarget.for_source(self.project_root)


class RuntimeSupervisor:
    """Run one child with UID isolation on root Linux and integrity-only mode elsewhere."""

    def __init__(
        self,
        *,
        project_root: Path,
        data_root: Path,
        application_root: Path | None = None,
        generation_store: GenerationStore | None = None,
        generation_spec: RuntimeGenerationSpec | Mapping[str, object] | None = None,
        control_path: Path | None = None,
        legacy_releases_root: Path | None = None,
        startup_timeout_seconds: float = 90,
        shutdown_timeout_seconds: float = 15,
        probation_seconds: float = 0,
        probation_probe_interval_seconds: float = 1,
        strict_generation_readiness: bool = True,
        max_unexpected_restarts: int = 3,
        restart_backoff_seconds: float = 0.25,
        max_restart_backoff_seconds: float = 5,
        child_uid: int | None = None,
        child_gid: int | None = None,
        apply_child_identity_to_legacy: bool = False,
        process_inspector: ProcessInspector | None = None,
        process_fencer: ProcessFencer | None = None,
        descendant_inspector: DescendantInspector | None = None,
        process_signaler: ProcessSignaler | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if startup_timeout_seconds <= 0 or shutdown_timeout_seconds <= 0:
            raise ValueError("runtime timeouts must be positive")
        if not math.isfinite(probation_seconds) or probation_seconds < 0:
            raise ValueError("runtime probation must be finite and nonnegative")
        if (
            not math.isfinite(probation_probe_interval_seconds)
            or probation_probe_interval_seconds <= 0
        ):
            raise ValueError("runtime probation probe interval must be finite and positive")
        if max_unexpected_restarts < 0:
            raise ValueError("maximum unexpected restarts cannot be negative")
        if restart_backoff_seconds < 0 or max_restart_backoff_seconds < 0:
            raise ValueError("runtime restart backoff cannot be negative")
        if max_restart_backoff_seconds < restart_backoff_seconds:
            raise ValueError("maximum restart backoff is too small")
        if (child_uid is None) != (child_gid is None):
            raise ValueError("runtime child UID and GID must be configured together")
        self._validate_identity_value(child_uid, label="child UID")
        self._validate_identity_value(child_gid, label="child GID")
        identity_explicit = child_uid is not None
        if identity_explicit:
            if os.name != "posix":
                raise ValueError("runtime child identity is only supported on POSIX")
            assert child_uid is not None and child_gid is not None
            if child_uid in {0, os.geteuid()} or child_gid in {0, os.getegid()}:
                raise ValueError("runtime child identity must differ from controller and root")
            if os.geteuid() != 0:
                raise ValueError("controller lacks authority to assume runtime child identity")
        elif sys.platform.startswith("linux") and os.geteuid() == 0:
            child_uid = _DEDICATED_RUNTIME_UID
            child_gid = _DEDICATED_RUNTIME_GID

        self._project_root = project_root.expanduser().resolve()
        bridge_path = self._project_root / "railway_sandbox_bridge" / "bridge.mjs"
        self._railway_sandbox_bridge = bridge_path.resolve() if bridge_path.is_file() else None
        self._data_root = data_root.expanduser().resolve()
        self._application_root = (
            application_root.expanduser().resolve()
            if application_root is not None
            else self._data_root
        )
        self._generation_store = generation_store
        self._selected_generation = (
            self._coerce_generation_spec(generation_spec) if generation_spec is not None else None
        )
        if self._selected_generation is not None and generation_store is None:
            raise ValueError("a generation store is required for a selected generation")
        self._control_path = (
            control_path.expanduser().absolute()
            if control_path is not None
            else self._data_root / "bootstrap" / "runtime-child.json"
        )
        self._legacy_releases_root = (
            legacy_releases_root.expanduser().absolute()
            if legacy_releases_root is not None
            else None
        )
        self._intent_path = self._control_path.with_name(f".{self._control_path.name}.intent")
        self._owner_lock_path = self._control_path.with_name("runtime-owner.lock")
        self._startup_timeout = startup_timeout_seconds
        self._shutdown_timeout = shutdown_timeout_seconds
        self._probation_seconds = probation_seconds
        self._probation_probe_interval = probation_probe_interval_seconds
        self._strict_generation_readiness = strict_generation_readiness
        self._max_unexpected_restarts = max_unexpected_restarts
        self._restart_backoff = restart_backoff_seconds
        self._max_restart_backoff = max_restart_backoff_seconds
        self._child_uid = child_uid
        self._child_gid = child_gid
        self._apply_child_identity_to_legacy = apply_child_identity_to_legacy
        self._process_inspector = process_inspector or self._inspect_process
        self._process_fencer = process_fencer
        self._uses_default_descendant_inspector = descendant_inspector is None
        self._descendant_inspector = descendant_inspector or self._inspect_linux_descendants
        self._process_signaler = process_signaler or os.kill
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2, read=5, write=5, pool=2), trust_env=False
        )
        self._owns_client = client is None
        self._child: _Child | None = None
        self._status = "stopped"
        self._error: str | None = None
        self._logs: deque[RuntimeLogEntry] = deque(maxlen=2_000)
        self._redaction_values: set[str] = set()
        self._log_stream_id = secrets.token_urlsafe(12)
        self._sequence = 0
        self._log_changed = asyncio.Condition()
        self._lock = asyncio.Lock()
        self._evolution_url: str | None = None
        self._evolution_token: str | None = None
        self._sandbox_url: str | None = None
        self._sandbox_token: str | None = None
        self._ownership_checked = False
        self._owner_lock_descriptor: int | None = None
        self._selection_version = 0
        self._desired_running = False
        self._desired_config: HostConfig | None = None
        self._desired_target: _LaunchTarget | None = None
        self._unexpected_restarts = 0
        self._operation_task: asyncio.Task[Any] | None = None
        self._watcher_task: asyncio.Task[None] | None = None
        self._probation_rollback_task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._isolation_mode_logged = False
        self._subreaper_attempted = False
        self._subreaper_enabled = False

    @property
    def endpoint(self) -> str | None:
        child = self._child
        if (
            self.status not in {"ready", "probation"}
            or child is None
            or child.requested_stop
            or child.process.returncode is not None
        ):
            return None
        return child.endpoint

    @property
    def status(self) -> str:
        child = self._child
        if self._status in {"ready", "probation"} and (
            child is None or child.process.returncode is not None
        ):
            return "failed"
        return self._status

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def log_stream_id(self) -> str:
        return self._log_stream_id

    @property
    def revision(self) -> int | None:
        return self._child.config.revision if self._child is not None else None

    @property
    def project_root(self) -> Path:
        return self._project_root

    @property
    def generation_id(self) -> str | None:
        child = self._child
        if child is not None and child.generation_id is not None:
            return child.generation_id
        if self._selected_generation is not None:
            return self._selected_generation.generation_id
        return None

    @property
    def generation(self) -> RuntimeGenerationSpec | None:
        return self._selected_generation

    @property
    def control_path(self) -> Path:
        return self._control_path

    def set_generation(self, generation: RuntimeGenerationSpec | Mapping[str, object]) -> None:
        """Select an exact verified generation before the runtime starts."""

        if self._generation_store is None:
            raise RuntimeUnavailableError("generation storage is not configured")
        if (
            self._child is not None
            or self._desired_running
            or self._operation_task is not None
            or self._watcher_task is not None
            or self._status not in {"stopped", "failed"}
        ):
            raise RuntimeUnavailableError("cannot change generation while runtime is running")
        self._selected_generation = self._coerce_generation_spec(generation)
        self._desired_config = None
        self._desired_target = None
        self._unexpected_restarts = 0

    def set_project_root(self, project_root: Path) -> None:
        """Explicitly select legacy source-overlay compatibility before startup."""

        if (
            self._child is not None
            or self._desired_running
            or self._operation_task is not None
            or self._watcher_task is not None
            or self._status not in {"stopped", "failed"}
        ):
            raise RuntimeUnavailableError("cannot change source while runtime is running")
        self._project_root = self._validated_project_root(project_root)
        self._selected_generation = None
        self._desired_config = None
        self._desired_target = None
        self._unexpected_restarts = 0

    def configure_evolution_control(self, *, base_url: str, token: str) -> None:
        if not self._is_fully_stopped():
            raise RuntimeUnavailableError(
                "cannot change evolution control during a runtime transition"
            )
        cleaned_url = str(base_url or "").strip().rstrip("/")
        cleaned_token = str(token or "").strip()
        if not cleaned_url.startswith("http://127.0.0.1:") or len(cleaned_token) < 32:
            raise ValueError("evolution control configuration is invalid")
        self._evolution_url = cleaned_url
        self._evolution_token = cleaned_token

    def configure_sandbox_worker(self, *, base_url: str, token: str) -> None:
        if not self._is_fully_stopped():
            raise RuntimeUnavailableError(
                "cannot change sandbox worker during a runtime transition"
            )
        cleaned_url = str(base_url or "").strip().rstrip("/")
        cleaned_token = str(token or "").strip()
        if not cleaned_url.startswith("http://") or len(cleaned_token) < 32:
            raise ValueError("sandbox worker configuration is invalid")
        self._sandbox_url = cleaned_url
        self._sandbox_token = cleaned_token

    async def start(self, config: HostConfig) -> None:
        async with self._lock:
            try:
                self._claim_operation()
                await self._ensure_controller_ownership()
                self._require_launch_safe()
                target = self._selected_target()
                previous = self._child
                try:
                    await self._preflight_target(target)
                except Exception as exc:
                    self._error = self._safe_error(exc)
                    if previous is None:
                        self._status = "failed"
                    raise
                previous_config = previous.config if previous is not None else None
                previous_target = previous.target if previous is not None else None
                if previous is not None:
                    self._status = "draining"
                    await self._stop_child(previous)
                    self._child = None
                self._begin_selection(config, target)
                try:
                    child = await self._spawn_target(config, target)
                except asyncio.CancelledError:
                    await self._restore_after_cancellation(previous_config, previous_target)
                    raise
                except Exception:
                    self._desired_running = False
                    raise
                self._adopt_child(child)
            finally:
                self._release_operation()

    async def replace(self, config: HostConfig, *, rollback: HostConfig | None) -> None:
        """Replace configuration without changing the exact selected runtime identity."""

        async with self._lock:
            try:
                self._claim_operation()
                await self._ensure_controller_ownership()
                self._require_launch_safe()
                target = self._current_target()
                await self._preflight_target(target)
                await self._replace_config_locked(
                    config, rollback=rollback, target=target, probation=True
                )
            finally:
                self._release_operation()

    async def replace_generation(
        self,
        generation: RuntimeGenerationSpec | Mapping[str, object],
        *,
        rollback: RuntimeGenerationSpec | Mapping[str, object] | None = None,
    ) -> None:
        """Activate one exact generation and restore an exact prior target on failure."""

        if self._generation_store is None:
            raise RuntimeUnavailableError("generation storage is not configured")
        candidate_spec = self._coerce_generation_spec(generation)
        rollback_spec = self._coerce_generation_spec(rollback) if rollback is not None else None
        async with self._lock:
            try:
                self._claim_operation()
                await self._ensure_controller_ownership()
                self._require_launch_safe()
                config = self._current_config()
                previous_target = self._current_target()
                if rollback_spec is not None and previous_target.generation != rollback_spec:
                    raise RuntimeUnavailableError(
                        "rollback generation does not match the captured previous target"
                    )
                await self._replace_target_locked(
                    config,
                    candidate=_LaunchTarget.for_generation(candidate_spec),
                    previous=previous_target,
                )
            finally:
                self._release_operation()

    async def restart_current(self) -> None:
        async with self._lock:
            try:
                self._claim_operation()
                await self._ensure_controller_ownership()
                self._require_launch_safe()
                config = self._current_config()
                target = self._current_target()
                await self._preflight_target(target)
                await self._replace_config_locked(config, rollback=config, target=target)
            finally:
                self._release_operation()

    async def replace_source(self, project_root: Path) -> None:
        """Activate a legacy source overlay and restore the exact previous target on failure."""

        candidate_root = self._validated_project_root(project_root)
        async with self._lock:
            try:
                self._claim_operation()
                await self._ensure_controller_ownership()
                self._require_launch_safe()
                await self._replace_target_locked(
                    self._current_config(),
                    candidate=_LaunchTarget.for_source(candidate_root),
                    previous=self._current_target(),
                )
            finally:
                self._release_operation()

    async def _replace_target_locked(
        self,
        config: HostConfig,
        *,
        candidate: _LaunchTarget,
        previous: _LaunchTarget,
    ) -> None:
        await self._preflight_target(candidate)
        old_child = self._child
        previous_config = old_child.config if old_child is not None else config
        if old_child is not None:
            self._status = "draining"
            await self._stop_child(old_child)
            self._child = None
        self._select_target(candidate)
        self._begin_selection(config, candidate)
        try:
            child = await self._spawn_target(config, candidate)
        except asyncio.CancelledError:
            await self._restore_after_cancellation(previous_config, previous)
            raise
        except Exception:
            if self._status == "recovery_required":
                self._desired_running = False
                raise
            self._append_log("host", "runtime candidate failed; restoring exact previous target")
            self._select_target(previous)
            self._begin_selection(previous_config, previous)
            self._status = "rolling_back"
            try:
                restoration = asyncio.create_task(self._spawn_target(previous_config, previous))
                restored, cancelled = await self._await_failure_restoration(restoration)
            except Exception as rollback_error:
                self._desired_running = False
                self._status = "failed"
                self._error = "candidate and exact rollback target failed to start"
                self._append_log("host", self._error)
                raise RuntimeUnavailableError(self._error) from rollback_error
            self._adopt_child(restored)
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        await self._adopt_replacement_candidate(
            child,
            previous_config=previous_config,
            previous_target=previous,
        )

    async def stop(self) -> None:
        self._stop_requested = True
        try:
            await self._cancel_lifecycle_tasks()
            async with self._lock:
                self._selection_version += 1
                self._desired_running = False
                child = self._child
                if child is None and (
                    os.path.lexists(self._control_path)
                    or os.path.lexists(self._intent_path)
                ):
                    self._status = "recovery_required"
                    self._error = self._error or "runtime ownership remains unresolved"
                    raise RuntimeUnavailableError(self._error)
                self._status = "stopping" if child is not None else "stopped"
                if child is not None:
                    await self._stop_child(child)
                    self._child = None
                self._status = "stopped"
                self._error = None
                self._desired_config = None
                self._desired_target = None
                self._unexpected_restarts = 0
        finally:
            self._stop_requested = False

    async def shutdown(self) -> None:
        await self.stop()
        if self._child is not None or self._status == "recovery_required":
            raise RuntimeUnavailableError("runtime shutdown could not prove child containment")
        if self._owns_client:
            await self._client.aclose()
        self._release_owner_lock()

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

    async def deliver_evolution_event(self, event: EvolutionEvent) -> None:
        """Deliver an event only to the nonce-bound child currently serving traffic."""

        if self._status == "probation":
            child = self._require_event_delivery_child()
            await self._deliver_evolution_event(child, event)
            return
        async with self._lock:
            child = self._require_event_delivery_child()
            await self._deliver_evolution_event(child, event)

    def _require_event_delivery_child(self) -> _Child:
        child = self._child
        if (
            self._status not in {"ready", "probation"}
            or child is None
            or child.requested_stop
            or child.process.returncode is not None
        ):
            raise RuntimeUnavailableError("serving runtime is unavailable for event delivery")
        return child

    async def _deliver_evolution_event(self, child: _Child, event: EvolutionEvent) -> None:
        try:
            response = await self._client.post(
                f"{child.endpoint}/_runtime/evolution-events",
                headers={
                    "Authorization": "Bearer " + child.config.internal_runtime_token.get_secret_value(),
                    "X-OpenTulpa-Launch-Nonce": child.launch_nonce,
                },
                content=event.model_dump_json(),
            )
        except httpx.HTTPError as exc:
            raise RuntimeUnavailableError(
                "serving runtime could not receive the evolution event"
            ) from exc
        if response.status_code != 204:
            raise RuntimeUnavailableError("serving runtime rejected the evolution event")

    async def _replace_config_locked(
        self,
        config: HostConfig,
        *,
        rollback: HostConfig | None,
        target: _LaunchTarget,
        probation: bool = False,
    ) -> None:
        previous = self._child
        previous_config = previous.config if previous is not None else self._desired_config
        previous_target = previous.target if previous is not None else self._desired_target
        if rollback is not None and rollback != previous_config:
            raise RuntimeUnavailableError(
                "rollback configuration does not match the captured previous configuration"
            )
        if previous is not None:
            self._status = "draining"
            await self._stop_child(previous)
            self._child = None
        self._begin_selection(config, target)
        try:
            candidate = await self._spawn_target(config, target)
        except asyncio.CancelledError:
            await self._restore_after_cancellation(previous_config, previous_target)
            raise
        except Exception:
            if self._status == "recovery_required":
                self._desired_running = False
                raise
            if previous_config is None or previous_target is None:
                self._desired_running = False
                raise
            self._append_log("host", "candidate failed; restoring previous runtime")
            self._select_target(previous_target)
            self._begin_selection(previous_config, previous_target)
            self._status = "rolling_back"
            try:
                restoration = asyncio.create_task(self._spawn_target(previous_config, previous_target))
                restored, cancelled = await self._await_failure_restoration(restoration)
            except Exception as rollback_error:
                self._desired_running = False
                self._status = "failed"
                self._error = "candidate and rollback runtimes failed to start"
                self._append_log("host", self._error)
                raise RuntimeUnavailableError(self._error) from rollback_error
            self._adopt_child(restored)
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        if probation:
            await self._adopt_replacement_candidate(
                candidate,
                previous_config=previous_config,
                previous_target=previous_target,
            )
        else:
            self._adopt_child(candidate)

    async def _spawn_target(self, config: HostConfig, target: _LaunchTarget) -> _Child:
        if target.generation is not None:
            return await self._spawn(config, generation_spec=target.generation)
        if target.project_root is None:
            raise RuntimeUnavailableError("runtime source identity is unavailable")
        return await self._spawn(config, project_root=target.project_root)

    async def _spawn(
        self,
        config: HostConfig,
        *,
        project_root: Path | None = None,
        generation_spec: RuntimeGenerationSpec | None = None,
    ) -> _Child:
        attempts = 2 if generation_spec is not None and self._strict_generation_readiness else 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._spawn_attempt(
                    config,
                    project_root=project_root,
                    generation_spec=generation_spec,
                )
            except _ChildExitedBeforeReadyError:
                if attempt >= attempts:
                    raise
                self._append_log(
                    "host",
                    "nonce-bound generation exited before readiness; retrying with a new port",
                )
        raise RuntimeUnavailableError("runtime launch attempts were exhausted")

    async def _spawn_attempt(
        self,
        config: HostConfig,
        *,
        project_root: Path | None,
        generation_spec: RuntimeGenerationSpec | None,
    ) -> _Child:
        port = self._free_port()
        endpoint = f"http://127.0.0.1:{port}"
        launch_nonce = secrets.token_urlsafe(24)
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
                self._evolution_token or "",
                self._sandbox_token or "",
            )
            if value
        }
        self._append_log("host", f"starting runtime revision {config.revision}")

        child: _Child | None = None
        process: asyncio.subprocess.Process | None = None
        intent_written = False
        try:
            if generation_spec is not None:
                installed = await self._open_generation(generation_spec)
                source_root = None
                cwd = self._generation_cwd(installed)
                argv = installed.entrypoint_argv
                if not argv or not Path(argv[0]).is_absolute():
                    raise RuntimeUnavailableError("generation entrypoint is not absolute")
                executable = installed.interpreter_path
            else:
                installed = None
                source_root = self._validated_project_root(project_root or self._project_root)
                cwd = source_root
                argv = (sys.executable, "-m", "opentulpa")
                executable = Path(sys.executable).resolve()

            environment = self._child_environment(
                config,
                port=port,
                project_root=source_root,
                installed_generation=installed,
                launch_nonce=launch_nonce,
            )
            spawn_options: dict[str, Any] = {
                "cwd": cwd,
                "env": environment,
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if os.name == "posix":
                spawn_options["start_new_session"] = True
            spawn_argv = self._child_spawn_argv(
                tuple(argv),
                generation=generation_spec is not None,
            )
            self._write_launch_intent(
                generation=generation_spec,
                project_root=source_root,
                launch_nonce=launch_nonce,
                executable=executable,
                argv=tuple(argv),
            )
            intent_written = True
            if generation_spec is not None:
                store = self._generation_store
                if store is None:
                    raise RuntimeUnavailableError("generation storage is not configured")
                assert installed is not None
                with store.locked():
                    final_generation = self._open_generation_locked(store, generation_spec)
                    if (
                        final_generation.entrypoint_argv != installed.entrypoint_argv
                        or final_generation.interpreter_path != installed.interpreter_path
                    ):
                        raise RuntimeUnavailableError(
                            "generation identity changed during final verification"
                        )
                    process = await asyncio.create_subprocess_exec(*spawn_argv, **spawn_options)
            else:
                process = await asyncio.create_subprocess_exec(*spawn_argv, **spawn_options)
            process_birth = self._capture_process_birth(process.pid)
            readers = tuple(
                asyncio.create_task(self._read_stream(stream, name))
                for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr"))
                if stream is not None
            )
            child = _Child(
                process=process,
                endpoint=endpoint,
                config=config,
                project_root=source_root,
                generation=generation_spec,
                installed_generation=installed,
                launch_nonce=launch_nonce,
                process_group=process.pid,
                process_birth=process_birth,
                executable=executable,
                argv=tuple(argv),
                readers=readers,
            )
            self._write_ownership_record(child)
            self._remove_launch_intent(launch_nonce)
            intent_written = False
            await self._wait_ready(child)
        except asyncio.CancelledError:
            if child is None and process is not None:
                child = _Child(
                    process=process,
                    endpoint=endpoint,
                    config=config,
                    project_root=source_root,
                    generation=generation_spec,
                    installed_generation=installed,
                    launch_nonce=launch_nonce,
                    process_group=process.pid,
                    process_birth="unverified",
                    executable=executable,
                    argv=tuple(argv),
                    readers=(),
                )
            if child is not None:
                await self._shield_candidate_cleanup(child)
                intent_written = False
            elif intent_written:
                self._status = "recovery_required"
                self._error = "runtime spawn was cancelled before child identity was captured"
            raise
        except Exception as exc:
            cleanup_error: Exception | None = None
            if child is None and process is not None:
                child = _Child(
                    process=process,
                    endpoint=endpoint,
                    config=config,
                    project_root=source_root,
                    generation=generation_spec,
                    installed_generation=installed,
                    launch_nonce=launch_nonce,
                    process_group=process.pid,
                    process_birth="unverified",
                    executable=executable,
                    argv=tuple(argv),
                    readers=(),
                )
            if child is not None:
                try:
                    await self._stop_child(child)
                    intent_written = False
                except Exception as stop_error:
                    cleanup_error = stop_error
            elif intent_written:
                self._remove_launch_intent(launch_nonce)
                intent_written = False
            if cleanup_error is not None:
                self._status = "recovery_required"
                self._error = self._safe_error(cleanup_error)
                raise RuntimeUnavailableError(self._error) from cleanup_error
            self._status = "failed"
            self._error = self._safe_error(exc)
            self._append_log("host", f"runtime failed: {self._error}")
            logger.error("child runtime failed: %s", self._error)
            if isinstance(exc, RuntimeUnavailableError):
                raise
            raise RuntimeUnavailableError(self._error) from exc
        self._status = "ready"
        identity = (
            f" generation {child.generation_id}" if child.generation_id is not None else ""
        )
        self._append_log("host", f"runtime revision {config.revision}{identity} is ready")
        return child

    async def _open_generation(self, spec: RuntimeGenerationSpec) -> InstalledGeneration:
        store = self._generation_store
        if store is None:
            raise RuntimeUnavailableError("generation storage is not configured")
        return await asyncio.to_thread(self._open_generation_locked, store, spec)

    @staticmethod
    def _open_generation_locked(
        store: GenerationStore,
        spec: RuntimeGenerationSpec,
    ) -> InstalledGeneration:
        return store.open(
            spec.generation_id,
            expected_manifest_digest=spec.expected_manifest_digest,
            expected_state_contract_digest=spec.expected_state_contract_digest,
            expected_evaluator_fingerprint=spec.expected_evaluator_fingerprint,
            expected_install_profile=spec.expected_install_profile,
            controller_protocol=spec.controller_protocol,
        )

    async def _wait_ready(self, child: _Child) -> None:
        deadline = asyncio.get_running_loop().time() + self._startup_timeout
        while asyncio.get_running_loop().time() < deadline:
            try:
                await self._probe_child_readiness(child)
                return
            except _ChildExitedBeforeReadyError:
                raise
            except _RuntimeProbeError:
                pass
            await asyncio.sleep(0.25)
        raise RuntimeUnavailableError("runtime readiness timed out")

    async def _probe_child_readiness(self, child: _Child) -> None:
        if child.process.returncode is not None:
            raise _ChildExitedBeforeReadyError(
                f"runtime exited before readiness with code {child.process.returncode}"
            )
        headers = {
            "Authorization": f"Bearer {child.config.internal_runtime_token.get_secret_value()}"
        }
        try:
            identity_matches = True
            if child.generation is None or self._strict_generation_readiness:
                identity = await self._client.get(
                    f"{child.endpoint}/_runtime/identity",
                    headers={"X-OpenTulpa-Launch-Nonce": child.launch_nonce},
                )
                identity_matches = identity.is_success and self._ready_identity_matches(
                    child, identity
                )
            health = await self._client.get(f"{child.endpoint}/healthz")
            agent = await self._client.get(f"{child.endpoint}/agent/healthz", headers=headers)
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise _RuntimeProbeError("runtime health probe failed") from exc
        if child.process.returncode is not None:
            raise _ChildExitedBeforeReadyError(
                f"runtime exited before readiness with code {child.process.returncode}"
            )
        if not identity_matches:
            raise _RuntimeProbeError("runtime identity probe did not match")
        if not health.is_success or not agent.is_success:
            raise _RuntimeProbeError("runtime health probe was unhealthy")

    def _ready_identity_matches(self, child: _Child, response: httpx.Response) -> bool:
        if child.generation is not None and not self._strict_generation_readiness:
            return True
        payload = response.json()
        return (
            isinstance(payload, dict)
            and payload.get("generation_id") == child.generation_id
            and payload.get("launch_nonce") == child.launch_nonce
        )

    async def _adopt_replacement_candidate(
        self,
        child: _Child,
        *,
        previous_config: HostConfig | None,
        previous_target: _LaunchTarget | None,
    ) -> None:
        self._adopt_child(child)
        if self._probation_seconds == 0:
            return
        self._status = "probation"
        self._append_log("host", "runtime candidate entered live probation")
        try:
            await self._wait_probation(child)
        except asyncio.CancelledError:
            await self._shield_candidate_cleanup(child)
            if self._child is child:
                self._child = None
            await self._restore_after_cancellation(previous_config, previous_target)
            raise
        except Exception as exc:
            await self._run_probation_rollback(
                child,
                previous_config=previous_config,
                previous_target=previous_target,
            )
            raise RuntimeUnavailableError(
                "runtime candidate failed probation; the exact previous runtime was restored"
            ) from exc
        self._status = "ready"
        self._append_log("host", "runtime candidate passed live probation")

    async def _run_probation_rollback(
        self,
        child: _Child,
        *,
        previous_config: HostConfig | None,
        previous_target: _LaunchTarget | None,
    ) -> None:
        active = self._probation_rollback_task
        if active is not None and not active.done():
            self._status = "recovery_required"
            self._desired_running = False
            raise RuntimeUnavailableError("another probation rollback is already active")
        rollback = asyncio.create_task(
            self._rollback_failed_probation(
                child,
                previous_config=previous_config,
                previous_target=previous_target,
            )
        )
        self._probation_rollback_task = rollback
        cancelled = False
        try:
            while not rollback.done():
                try:
                    await asyncio.shield(rollback)
                except asyncio.CancelledError:
                    cancelled = True
            rollback.result()
        finally:
            if self._probation_rollback_task is rollback:
                self._probation_rollback_task = None
        if cancelled:
            raise asyncio.CancelledError from None

    async def _wait_probation(self, child: _Child) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._probation_seconds
        while True:
            await self._probe_child_readiness(child)
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.sleep(min(self._probation_probe_interval, remaining))

    async def _rollback_failed_probation(
        self,
        child: _Child,
        *,
        previous_config: HostConfig | None,
        previous_target: _LaunchTarget | None,
    ) -> None:
        try:
            await self._stop_child(child)
        except Exception as containment_error:
            self._desired_running = False
            raise RuntimeUnavailableError(
                "runtime candidate failed probation and containment could not be proven"
            ) from containment_error
        if self._child is child:
            self._child = None
        if previous_config is None or previous_target is None:
            self._desired_running = False
            self._status = "failed"
            self._error = "runtime candidate failed probation"
            raise RuntimeUnavailableError(self._error)
        self._select_target(previous_target)
        self._begin_selection(previous_config, previous_target)
        self._status = "rolling_back"
        try:
            restored = await self._spawn_target(previous_config, previous_target)
        except Exception as rollback_error:
            self._desired_running = False
            if self._status != "recovery_required":
                self._status = "failed"
                self._error = "runtime candidate failed probation and exact rollback failed"
            raise RuntimeUnavailableError(self._error) from rollback_error
        self._adopt_child(restored)

    def _adopt_child(self, child: _Child) -> None:
        self._child = child
        watcher = asyncio.create_task(self._watch_child_exit(child))
        child.watcher = watcher
        self._watcher_task = watcher

    async def _watch_child_exit(self, child: _Child) -> None:
        try:
            return_code = await child.process.wait()
            async with self._lock:
                if child.requested_stop or self._child is not child:
                    return
                self._status = "fencing"
                try:
                    await self._terminate_child_process_group(child)
                except Exception as exc:
                    self._status = "recovery_required"
                    self._desired_running = False
                    self._error = "runtime descendants could not be fenced after leader exit"
                    self._append_log("host", self._error)
                    logger.error("runtime descendant fencing failed: %s", self._safe_error(exc))
                    return
                await self._close_readers(child)
                self._remove_ownership_record(child.launch_nonce)
                self._remove_launch_intent(child.launch_nonce)
                self._error = f"runtime exited unexpectedly with code {return_code}"
                self._append_log("host", self._error)
                version = self._selection_version
                config = child.config
                target = child.target
                self._child = None
                if not self._desired_running or self._max_unexpected_restarts == 0:
                    self._status = "failed"
                    return
                self._status = "restarting"

            while True:
                if self._unexpected_restarts >= self._max_unexpected_restarts:
                    async with self._lock:
                        if self._selection_version == version and self._child is None:
                            self._desired_running = False
                            self._status = "failed"
                            self._error = "runtime exhausted its unexpected-exit restart budget"
                            self._append_log("host", self._error)
                    return
                attempt = self._unexpected_restarts + 1
                delay = min(
                    self._restart_backoff * (2 ** (attempt - 1)),
                    self._max_restart_backoff,
                )
                if delay:
                    await asyncio.sleep(delay)
                async with self._lock:
                    if (
                        self._selection_version != version
                        or not self._desired_running
                        or self._child is not None
                    ):
                        return
                    self._unexpected_restarts = attempt
                    self._status = "restarting"
                    self._append_log(
                        "host",
                        f"restarting runtime after unexpected exit ({attempt}/"
                        f"{self._max_unexpected_restarts})",
                    )
                    try:
                        replacement = await self._spawn_target(config, target)
                    except Exception:
                        if self._status == "recovery_required":
                            self._desired_running = False
                            return
                        if attempt >= self._max_unexpected_restarts:
                            self._desired_running = False
                            self._status = "failed"
                            self._error = "runtime exhausted its unexpected-exit restart budget"
                            self._append_log("host", self._error)
                            return
                        self._status = "restarting"
                        continue
                    self._adopt_child(replacement)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("runtime exit watcher failed: %s", self._safe_error(exc))
            async with self._lock:
                if self._child is child:
                    self._desired_running = False
                    self._status = "recovery_required"
                    self._error = "runtime exit watcher could not prove containment"
        finally:
            if self._watcher_task is asyncio.current_task():
                self._watcher_task = None

    async def _stop_child(self, child: _Child) -> None:
        child.requested_stop = True
        terminated = False
        try:
            await self._terminate_child_process_group(child)
            terminated = True
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status = "recovery_required"
            self._error = self._safe_error(exc)
            if self._child is None:
                self._child = child
            raise
        finally:
            watcher = child.watcher
            if watcher is not None and watcher is not asyncio.current_task() and not watcher.done():
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
            await self._close_readers(child)
            if terminated:
                self._remove_ownership_record(child.launch_nonce)
                self._remove_launch_intent(child.launch_nonce)

    async def _terminate_child_process_group(self, child: _Child) -> None:
        if sys.platform.startswith("linux"):
            await self._terminate_linux_child(child)
            return
        await self._terminate_non_linux_child(child)

    async def _terminate_linux_child(self, child: _Child) -> None:
        if not self._subreaper_enabled:
            raise RuntimeUnavailableError("Linux descendant containment is not enabled")
        leader = self._child_process_identity(child)
        if child.process.returncode is None:
            await self._signal_verified_identity(
                leader,
                signal.SIGTERM,
                fallback=child.process.terminate,
            )
        term_signaled: set[tuple[int, str | None]] = set()
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout
        while asyncio.get_running_loop().time() < deadline:
            descendants = self._owned_descendants(child)
            new_descendants = tuple(
                descendant
                for descendant in descendants
                if (descendant.pid, descendant.process_birth) not in term_signaled
            )
            await self._signal_verified_descendants(new_descendants, signal.SIGTERM)
            term_signaled.update(
                (descendant.pid, descendant.process_birth) for descendant in new_descendants
            )
            self._reap_descendants(descendants)
            if child.process.returncode is not None and not self._owned_descendants(child):
                return
            await asyncio.sleep(0.05)

        if child.process.returncode is None:
            await self._signal_verified_identity(
                leader,
                signal.SIGKILL,
                fallback=child.process.kill,
            )
        kill_deadline = asyncio.get_running_loop().time() + self._shutdown_timeout
        while asyncio.get_running_loop().time() < kill_deadline:
            descendants = self._owned_descendants(child)
            await self._signal_verified_descendants(descendants, signal.SIGKILL)
            self._reap_descendants(descendants)
            if child.process.returncode is not None and not self._owned_descendants(child):
                return
            await asyncio.sleep(0.05)
        if child.process.returncode is None or self._owned_descendants(child):
            raise RuntimeUnavailableError("runtime descendants did not exit after containment kill")

    async def _terminate_non_linux_child(self, child: _Child) -> None:
        if os.name != "posix":
            if child.process.returncode is None:
                child.process.terminate()
            try:
                await asyncio.wait_for(child.process.wait(), timeout=self._shutdown_timeout)
                return
            except TimeoutError:
                child.process.kill()
                await asyncio.wait_for(child.process.wait(), timeout=self._shutdown_timeout)
                return
        if child.process.returncode is not None:
            raise RuntimeUnavailableError(
                "non-Linux runtime leader exited before group containment was proven"
            )
        try:
            actual_group = os.getpgid(child.process.pid)
        except ProcessLookupError as exc:
            raise RuntimeUnavailableError("runtime leader identity became ambiguous") from exc
        if actual_group != child.process_group or child.process_group != child.process.pid:
            raise RuntimeUnavailableError("runtime process group ownership is ambiguous")
        self._verify_non_linux_leader(child, label="group signal")
        os.killpg(child.process_group, signal.SIGTERM)
        try:
            await asyncio.wait_for(child.process.wait(), timeout=self._shutdown_timeout)
        except TimeoutError as exc:
            if child.process.returncode is not None:
                raise RuntimeUnavailableError(
                    "non-Linux leader exited before escalation could be bound"
                ) from exc
            try:
                current_group = os.getpgid(child.process.pid)
            except ProcessLookupError as lookup_error:
                raise RuntimeUnavailableError(
                    "runtime leader disappeared before escalation"
                ) from lookup_error
            if current_group != child.process_group:
                raise RuntimeUnavailableError(
                    "runtime process group changed before escalation"
                ) from exc
            try:
                self._verify_non_linux_leader(child, label="group escalation")
            except RuntimeUnavailableError as identity_error:
                raise identity_error from exc
            os.killpg(child.process_group, signal.SIGKILL)
            await asyncio.wait_for(child.process.wait(), timeout=self._shutdown_timeout)

    def _verify_non_linux_leader(self, child: _Child, *, label: str) -> None:
        if child.process.returncode is not None:
            raise RuntimeUnavailableError(f"runtime leader exited before {label}")
        observed_birth = self._capture_process_birth(child.process.pid)
        if observed_birth != child.process_birth:
            raise RuntimeUnavailableError(f"runtime leader identity changed before {label}")
        if child.process.returncode is not None:
            raise RuntimeUnavailableError(f"runtime leader exited before {label}")

    @staticmethod
    def _child_process_identity(child: _Child) -> RuntimeProcessIdentity:
        return RuntimeProcessIdentity(
            pid=child.process.pid,
            process_group=child.process_group,
            executable=child.executable,
            argv=child.argv,
            process_birth=child.process_birth,
            launch_nonce=child.launch_nonce,
        )

    def _owned_descendants(self, child: _Child) -> tuple[RuntimeProcessIdentity, ...]:
        if not sys.platform.startswith("linux"):
            return ()
        if not self._subreaper_enabled:
            raise RuntimeUnavailableError("Linux descendant containment is not enabled")
        try:
            descendants = self._descendant_inspector(child.process.pid, child.launch_nonce)
        except Exception as exc:
            raise RuntimeUnavailableError("runtime descendants could not be enumerated") from exc
        for descendant in descendants:
            if (
                descendant.pid == child.process.pid
                or descendant.process_birth is None
                or descendant.launch_nonce != child.launch_nonce
            ):
                raise RuntimeUnavailableError("runtime descendant identity is ambiguous")
        return tuple(sorted(descendants, key=lambda item: item.pid))

    async def _signal_verified_descendants(
        self,
        descendants: tuple[RuntimeProcessIdentity, ...],
        selected_signal: signal.Signals,
    ) -> None:
        for expected in descendants:
            await self._signal_verified_identity(expected, selected_signal)

    async def _signal_verified_identity(
        self,
        expected: RuntimeProcessIdentity,
        selected_signal: signal.Signals,
        *,
        fallback: Callable[[], None] | None = None,
    ) -> bool:
        observed = self._process_inspector(expected.pid)
        if observed is None:
            return False
        if not self._same_process_identity(expected, observed):
            raise RuntimeUnavailableError("runtime process changed before fencing")
        if self._process_fencer is not None:
            final_observation = self._process_inspector(expected.pid)
            if final_observation is None:
                return False
            if not self._same_process_identity(expected, final_observation):
                raise RuntimeUnavailableError("runtime process changed before fencing")
            result = self._process_fencer(expected, selected_signal)
            if inspect.isawaitable(result):
                await result
            return True
        pidfd_open = getattr(os, "pidfd_open", None)
        pidfd_send_signal = getattr(signal, "pidfd_send_signal", None)
        if callable(pidfd_open) and callable(pidfd_send_signal):
            try:
                descriptor = pidfd_open(expected.pid, 0)
            except ProcessLookupError:
                return False
            except OSError as exc:
                raise RuntimeUnavailableError("runtime pidfd could not be opened") from exc
            try:
                final_observation = self._process_inspector(expected.pid)
                if final_observation is None:
                    return False
                if not self._same_process_identity(expected, final_observation):
                    raise RuntimeUnavailableError("runtime process changed before pidfd signal")
                try:
                    pidfd_send_signal(descriptor, selected_signal, None, 0)
                except ProcessLookupError:
                    return False
                except OSError as exc:
                    raise RuntimeUnavailableError("runtime pidfd signal failed") from exc
                return True
            finally:
                os.close(descriptor)
        final_observation = self._process_inspector(expected.pid)
        if final_observation is None:
            return False
        if not self._same_process_identity(expected, final_observation):
            raise RuntimeUnavailableError("runtime process changed before PID signal")
        try:
            if fallback is not None:
                fallback()
            else:
                self._process_signaler(expected.pid, selected_signal)
        except ProcessLookupError:
            return False
        except OSError as exc:
            raise RuntimeUnavailableError("runtime PID signal failed") from exc
        return True

    @staticmethod
    def _same_process_identity(
        expected: RuntimeProcessIdentity,
        observed: RuntimeProcessIdentity,
    ) -> bool:
        return (
            expected.pid == observed.pid
            and expected.process_birth is not None
            and expected.process_birth == observed.process_birth
            and expected.launch_nonce is not None
            and expected.launch_nonce == observed.launch_nonce
            and expected.executable.resolve() == observed.executable.resolve()
            and expected.argv == observed.argv
        )

    @staticmethod
    def _reap_descendants(descendants: tuple[RuntimeProcessIdentity, ...]) -> None:
        for descendant in descendants:
            with suppress(ChildProcessError, ProcessLookupError):
                os.waitpid(descendant.pid, os.WNOHANG)

    async def _close_readers(self, child: _Child) -> None:
        for reader in child.readers:
            if not reader.done():
                reader.cancel()
        for reader in child.readers:
            with suppress(asyncio.CancelledError, Exception):
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
            stream_id=self._log_stream_id,
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

    def _child_environment(
        self,
        config: HostConfig,
        *,
        port: int,
        project_root: Path | None = None,
        installed_generation: InstalledGeneration | None = None,
        launch_nonce: str | None = None,
    ) -> dict[str, str]:
        source_root = project_root or self._project_root
        inherited = os.environ
        environment = {
            key: value
            for key in _PLATFORM_ENVIRONMENT_ALLOWLIST
            if (value := inherited.get(key)) is not None and "\x00" not in value
        }
        host_port = str(inherited.get("PORT") or "").strip()
        internal_agent_api_url = str(
            inherited.get("OPENTULPA_INTERNAL_AGENT_API_URL") or ""
        ).strip() or f"http://127.0.0.1:{host_port or port}"
        owner_customer_id = (
            str(inherited.get("OPENTULPA_OWNER_CUSTOMER_ID") or "").strip() or "owner"
        )
        executable_bin = (
            installed_generation.interpreter_path.parent
            if installed_generation is not None
            else Path(sys.executable).resolve().parent
        )
        environment.update(
            {
                "HOST": "127.0.0.1",
                "PORT": str(port),
                "OPENTULPA_DATA_ROOT": str(self._data_root),
                "OPENAI_COMPATIBLE_API_KEY": config.api_key.get_secret_value(),
                "OPENAI_COMPATIBLE_BASE_URL": config.base_url,
                "LLM_MODEL": config.model,
                "OPENTULPA_OWNER_TOKEN": config.internal_runtime_token.get_secret_value(),
                "OPENTULPA_OWNER_CUSTOMER_ID": owner_customer_id,
                "OPENTULPA_INTERNAL_AGENT_API_URL": internal_agent_api_url,
                "OPENTULPA_DYNAMIC_HOST": "1",
                "HOME": str(self._application_root),
                "PATH": f"{executable_bin}:{_TRUSTED_SYSTEM_PATH}",
            }
        )
        if installed_generation is None:
            environment["PYTHONPATH"] = str(source_root / "src")
        else:
            manifest = installed_generation.manifest
            environment.update(
                {
                    "OPENTULPA_APPLICATION_ROOT": str(self._application_root),
                    "OPENTULPA_GENERATION_ID": installed_generation.generation_id,
                    "OPENTULPA_GENERATION_MANIFEST_DIGEST": installed_generation.manifest_digest,
                    "OPENTULPA_GENERATION_SOURCE_COMMIT": manifest.identity.source_commit,
                    "OPENTULPA_GENERATION_SOURCE_TREE_SHA256": (
                        manifest.identity.source_tree_sha256
                    ),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                }
            )
        if launch_nonce is not None:
            environment["OPENTULPA_LAUNCH_NONCE"] = launch_nonce
        if self._railway_sandbox_bridge is not None:
            environment["OPENTULPA_RAILWAY_SANDBOX_BRIDGE_PATH"] = str(
                self._railway_sandbox_bridge
            )
        if self._evolution_url is not None and self._evolution_token is not None:
            environment.update(
                {
                    "EVOLUTION_ENABLED": "true",
                    "OPENTULPA_BOOTSTRAP_EVOLUTION_URL": self._evolution_url,
                    "OPENTULPA_BOOTSTRAP_EVOLUTION_TOKEN": self._evolution_token,
                }
            )
        if self._sandbox_url is not None and self._sandbox_token is not None:
            environment.update(
                {
                    "OPENTULPA_SANDBOX_RPC_URL": self._sandbox_url,
                    "OPENTULPA_SANDBOX_RPC_TOKEN": self._sandbox_token,
                }
            )
        if config.telegram_pairing_code is not None:
            environment["OPENTULPA_TELEGRAM_PAIRING_CODE"] = (
                config.telegram_pairing_code.get_secret_value()
            )
        environment["OPENTULPA_TELEGRAM_OWNER_ID"] = (
            str(config.telegram_user_id) if config.telegram_user_id is not None else ""
        )
        return environment

    def _generation_cwd(self, installed: InstalledGeneration) -> Path:
        root = self._application_root
        for label, external_root in (
            ("application", root),
            ("data", self._data_root),
        ):
            if self._is_relative_to(external_root, installed.path):
                raise RuntimeUnavailableError(
                    f"generation {label} root cannot be inside the generation"
                )
            if self._looks_like_source_checkout(self._project_root) and self._is_relative_to(
                external_root, self._project_root
            ):
                raise RuntimeUnavailableError(
                    f"generation {label} root cannot be inside a checkout"
                )
        if root.is_symlink() or not root.is_dir():
            raise RuntimeUnavailableError("generation application root is unavailable")
        self._require_root_usable(root, label="application")
        self._require_root_usable(self._data_root, label="data")
        return root.resolve(strict=True)

    def _child_spawn_argv(
        self,
        argv: tuple[str, ...],
        *,
        generation: bool,
    ) -> tuple[str, ...]:
        if not generation and not self._apply_child_identity_to_legacy:
            return argv
        uid = self._child_uid
        gid = self._child_gid
        if uid is None and gid is None:
            return argv
        if os.name != "posix":
            raise RuntimeUnavailableError("runtime child identity requires POSIX")
        euid = os.geteuid()
        egid = os.getegid()
        if uid is not None and uid != euid and euid != 0:
            raise RuntimeUnavailableError("controller cannot assume the configured child UID")
        if gid is not None and gid != egid and euid != 0:
            raise RuntimeUnavailableError("controller cannot assume the configured child GID")
        if uid is None or gid is None:
            raise RuntimeUnavailableError("runtime child identity is incomplete")
        setpriv = self._identity_switch_executable()
        return (
            str(setpriv),
            f"--reuid={uid}",
            f"--regid={gid}",
            "--clear-groups",
            "--inh-caps=-all",
            "--ambient-caps=-all",
            "--bounding-set=-all",
            "--no-new-privs",
            "--",
            *argv,
        )

    @staticmethod
    def _identity_switch_executable() -> Path:
        path = _SETPRIV_PATH
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeUnavailableError("trusted runtime identity switch is unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != 0
            or stat.S_IMODE(metadata.st_mode) & 0o022
            or not os.access(path, os.X_OK)
        ):
            raise RuntimeUnavailableError("trusted runtime identity switch is unsafe")
        return path

    async def _preflight_target(self, target: _LaunchTarget) -> None:
        if sys.platform.startswith("linux"):
            self._enable_child_subreaper()
        if target.generation is None:
            if target.project_root is None:
                raise RuntimeUnavailableError("runtime source identity is unavailable")
            source_root = self._validated_project_root(target.project_root)
            if self._apply_child_identity_to_legacy:
                allowed = self._legacy_releases_root
                if allowed is None or not self._is_relative_to(source_root, allowed):
                    raise RuntimeUnavailableError(
                        "isolated legacy runtime requires a trusted materialized release"
                    )
                self._log_isolation_mode()
                self._require_legacy_source_safe(source_root, allowed)
            return
        try:
            installed = await self._open_generation(target.generation)
        except Exception as exc:
            raise RuntimeUnavailableError(self._safe_error(exc)) from exc
        self._log_isolation_mode()
        self._generation_cwd(installed)
        self._require_child_cannot_write_controller_or_generation(installed)

    def _enable_child_subreaper(self) -> None:
        if self._subreaper_attempted:
            if not self._subreaper_enabled:
                raise RuntimeUnavailableError("Linux child subreaper containment is unavailable")
            return
        self._subreaper_attempted = True
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            enabled = ctypes.c_int()
            set_result = libc.prctl(36, 1, 0, 0, 0)  # PR_SET_CHILD_SUBREAPER
            get_result = libc.prctl(37, ctypes.byref(enabled), 0, 0, 0)  # PR_GET_CHILD_SUBREAPER
        except (AttributeError, OSError) as exc:
            raise RuntimeUnavailableError("Linux child subreaper containment is unavailable") from exc
        if set_result != 0 or get_result != 0 or enabled.value != 1:
            error_number = ctypes.get_errno()
            raise RuntimeUnavailableError(
                f"Linux child subreaper containment failed with errno {error_number}"
            )
        self._subreaper_enabled = True

    def _require_child_cannot_write_controller_or_generation(
        self,
        installed: InstalledGeneration,
    ) -> None:
        store = self._generation_store
        if store is None:
            raise RuntimeUnavailableError("generation storage is not configured")
        protected_paths = [self._control_path.parent, store.root]
        current = installed.path
        while current != store.root:
            protected_paths.append(current)
            parent = current.parent
            if parent == current or not self._is_relative_to(parent, store.root):
                raise RuntimeUnavailableError("generation path escaped its protected store root")
            current = parent
        uid = self._child_uid
        gid = self._child_gid
        for path in dict.fromkeys(protected_paths):
            try:
                metadata = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeUnavailableError("protected runtime root is unavailable") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeUnavailableError("protected runtime root is unsafe")
            if uid is not None and gid is not None:
                writable = self._mode_allows(metadata, uid=uid, gid=gid, required=0o2)
            else:
                # Same-UID mode is integrity-only; broad write bits are still rejected.
                writable = bool(stat.S_IMODE(metadata.st_mode) & 0o022)
            if writable:
                raise RuntimeUnavailableError(
                    "runtime child identity can write a protected controller or generation root"
                )

    def _require_legacy_source_safe(self, root: Path, allowed_root: Path) -> None:
        try:
            allowed_metadata = allowed_root.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeUnavailableError("legacy runtime source store is unavailable") from exc
        if (
            stat.S_ISLNK(allowed_metadata.st_mode)
            or not stat.S_ISDIR(allowed_metadata.st_mode)
            or allowed_metadata.st_uid != os.geteuid()
            or stat.S_IMODE(allowed_metadata.st_mode) != 0o711
            or root.parent != allowed_root
        ):
            raise RuntimeUnavailableError("legacy runtime source store is unsafe")
        entries = 0
        total_bytes = 0
        pending = [root]
        while pending:
            path = pending.pop()
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise RuntimeUnavailableError("legacy runtime source is unavailable") from exc
            if stat.S_ISLNK(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise RuntimeUnavailableError("legacy runtime source ownership is unsafe")
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISDIR(metadata.st_mode):
                if mode != 0o555:
                    raise RuntimeUnavailableError("legacy runtime source directory is writable")
                children = tuple(Path(entry.path) for entry in os.scandir(path))
                entries += len(children)
                if entries > 100_000:
                    raise RuntimeUnavailableError("legacy runtime source has too many entries")
                pending.extend(children)
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or mode not in {0o444, 0o555}
            ):
                raise RuntimeUnavailableError("legacy runtime source file is unsafe")
            total_bytes += metadata.st_size
            if total_bytes > 512 * 1024 * 1024:
                raise RuntimeUnavailableError("legacy runtime source exceeds its size limit")
        uid = self._child_uid
        gid = self._child_gid
        if uid is not None and gid is not None:
            if self._mode_allows(allowed_metadata, uid=uid, gid=gid, required=0o2):
                raise RuntimeUnavailableError("runtime child can replace legacy releases")
            if not self._mode_allows(allowed_metadata, uid=uid, gid=gid, required=0o1):
                raise RuntimeUnavailableError("runtime child cannot traverse legacy releases")

    def _require_root_usable(self, root: Path, *, label: str) -> None:
        try:
            metadata = root.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeUnavailableError(f"generation {label} root is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeUnavailableError(f"generation {label} root is unsafe")
        uid = self._child_uid
        gid = self._child_gid
        if uid is None or gid is None:
            if not os.access(root, os.R_OK | os.W_OK | os.X_OK):
                raise RuntimeUnavailableError(
                    f"generation {label} root is not usable in integrity-only mode"
                )
            return
        if not self._mode_allows(metadata, uid=uid, gid=gid, required=0o7):
            raise RuntimeUnavailableError(
                f"generation {label} root is not usable by the runtime child identity"
            )
        parent = root.parent
        while parent != root:
            parent_metadata = parent.stat(follow_symlinks=False)
            if not self._mode_allows(parent_metadata, uid=uid, gid=gid, required=0o1):
                raise RuntimeUnavailableError(
                    f"generation {label} root is not traversable by the runtime child identity"
                )
            root = parent
            parent = root.parent

    @staticmethod
    def _mode_allows(metadata: os.stat_result, *, uid: int, gid: int, required: int) -> bool:
        mode = stat.S_IMODE(metadata.st_mode)
        granted = (
            (mode >> 6) & 0o7
            if metadata.st_uid == uid
            else (mode >> 3) & 0o7
            if metadata.st_gid == gid
            else mode & 0o7
        )
        return granted & required == required

    def _log_isolation_mode(self) -> None:
        if self._isolation_mode_logged:
            return
        if self._child_uid is None:
            self._append_log(
                "host",
                "generation runtime is using integrity-only same-UID mode; "
                "child-UID isolation is unavailable on this controller",
            )
        else:
            self._append_log(
                "host",
                f"generation runtime will use dedicated UID/GID "
                f"{self._child_uid}/{self._child_gid}",
            )
        self._isolation_mode_logged = True

    def _begin_selection(self, config: HostConfig, target: _LaunchTarget) -> None:
        self._selection_version += 1
        self._desired_running = True
        self._desired_config = config
        self._desired_target = target
        self._unexpected_restarts = 0

    def _is_fully_stopped(self) -> bool:
        return (
            self._status == "stopped"
            and self._child is None
            and not self._desired_running
            and self._operation_task is None
            and self._watcher_task is None
            and self._probation_rollback_task is None
        )

    def _claim_operation(self) -> None:
        task = asyncio.current_task()
        if task is None:
            raise RuntimeUnavailableError("runtime lifecycle task is unavailable")
        if self._operation_task is not None and self._operation_task is not task:
            raise RuntimeUnavailableError("another runtime lifecycle operation is active")
        self._operation_task = task
        self._stop_requested = False

    def _release_operation(self) -> None:
        if self._operation_task is asyncio.current_task():
            self._operation_task = None

    async def _cancel_lifecycle_tasks(self) -> None:
        current = asyncio.current_task()
        failure: Exception | None = None
        for task in (self._operation_task, self._watcher_task):
            if task is None or task is current or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                failure = failure or exc
        if self._watcher_task is not None and self._watcher_task.done():
            self._watcher_task = None
        if failure is not None:
            self._status = "recovery_required"
            self._error = self._safe_error(failure)
            raise RuntimeUnavailableError(
                self._error or "runtime lifecycle cleanup could not prove containment"
            ) from failure

    def _require_launch_safe(self) -> None:
        if self._status == "recovery_required":
            raise RuntimeUnavailableError(
                "runtime recovery is required before another child can be launched"
            )

    async def _restore_after_cancellation(
        self,
        config: HostConfig | None,
        target: _LaunchTarget | None,
    ) -> None:
        if (
            config is None
            or target is None
            or self._stop_requested
            or self._status == "recovery_required"
        ):
            self._desired_running = False
            if self._status != "recovery_required":
                self._status = "stopped" if self._stop_requested else "failed"
            return
        self._append_log("host", "runtime activation cancelled; restoring exact previous runtime")
        self._select_target(target)
        self._begin_selection(config, target)
        self._status = "rolling_back"
        restoration = asyncio.create_task(self._spawn_target(config, target))
        try:
            restored = await self._await_restoration(restoration)
        except asyncio.CancelledError:
            self._desired_running = False
            if self._child is None and self._status != "recovery_required":
                self._status = "stopped"
            return
        except Exception as exc:
            self._desired_running = False
            if self._status != "recovery_required":
                self._status = "failed"
                self._error = "cancelled activation could not restore the previous runtime"
            self._append_log(
                "host",
                self._error or "cancelled activation could not restore the previous runtime",
            )
            logger.error("runtime cancellation rollback failed: %s", self._safe_error(exc))
            return
        self._adopt_child(restored)

    async def _shield_candidate_cleanup(self, child: _Child) -> None:
        cleanup = asyncio.create_task(self._stop_child(child))
        await self._await_shielded(cleanup)

    @staticmethod
    async def _await_shielded(task: asyncio.Task[Any]) -> Any:
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if task.done():
                    return task.result()

    async def _await_restoration(self, task: asyncio.Task[_Child]) -> _Child:
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                if not self._stop_requested:
                    if task.done():
                        return task.result()
                    continue
                task.cancel()
                try:
                    restored = await self._await_shielded(task)
                except asyncio.CancelledError:
                    pass
                else:
                    self._adopt_child(restored)
                raise

    async def _await_failure_restoration(
        self,
        task: asyncio.Task[_Child],
    ) -> tuple[_Child, bool]:
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not self._stop_requested:
                    cancelled = True
                    continue
                task.cancel()
                try:
                    restored = await self._await_shielded(task)
                except asyncio.CancelledError:
                    pass
                else:
                    self._adopt_child(restored)
                raise
        return task.result(), cancelled

    async def _ensure_controller_ownership(self) -> None:
        self._acquire_owner_lock()
        await self._ensure_orphan_fenced()

    def _acquire_owner_lock(self) -> None:
        if self._owner_lock_descriptor is not None:
            return
        parent = self._secure_control_parent()
        descriptor: int | None = None
        try:
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(self._owner_lock_path, flags, 0o600)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                raise RuntimeUnavailableError("runtime owner lock is unsafe")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeUnavailableError("another runtime controller owns this child slot") from exc
            self._owner_lock_descriptor = descriptor
            descriptor = None
            self._fsync_directory(parent)
        except RuntimeUnavailableError:
            raise
        except OSError as exc:
            raise RuntimeUnavailableError("runtime owner lock could not be acquired") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _release_owner_lock(self) -> None:
        descriptor = self._owner_lock_descriptor
        self._owner_lock_descriptor = None
        if descriptor is None:
            return
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _selected_target(self) -> _LaunchTarget:
        if self._selected_generation is not None:
            return _LaunchTarget.for_generation(self._selected_generation)
        return _LaunchTarget.for_source(self._validated_project_root(self._project_root))

    def _current_target(self) -> _LaunchTarget:
        if self._child is not None:
            return self._child.target
        if self._desired_target is not None:
            return self._desired_target
        return self._selected_target()

    def _current_config(self) -> HostConfig:
        if self._child is not None:
            return self._child.config
        if self._desired_config is not None:
            return self._desired_config
        raise RuntimeUnavailableError("runtime is not configured")

    def _select_target(self, target: _LaunchTarget) -> None:
        if target.generation is not None:
            self._selected_generation = target.generation
            return
        if target.project_root is None:
            raise RuntimeUnavailableError("runtime source identity is unavailable")
        self._selected_generation = None
        self._project_root = target.project_root

    async def _ensure_orphan_fenced(self) -> None:
        if self._ownership_checked:
            return
        record = self._read_ownership_record()
        intent = self._read_launch_intent()
        if intent is not None and record is None:
            self._status = "recovery_required"
            raise RuntimeUnavailableError(
                "an incomplete runtime launch intent requires manual recovery"
            )
        if intent is not None and record is not None and intent.launch_nonce != record.launch_nonce:
            self._status = "recovery_required"
            raise RuntimeUnavailableError("runtime launch intent and ownership record disagree")
        if record is None:
            self._ownership_checked = True
            return
        if sys.platform.startswith("linux"):
            self._enable_child_subreaper()
        try:
            observed_host_birth = self._capture_process_birth(record.host_pid)
        except RuntimeUnavailableError:
            observed_host_birth = None
        if (
            record.host_pid != os.getpid()
            and observed_host_birth is not None
            and observed_host_birth == record.host_birth
        ):
            self._status = "recovery_required"
            raise RuntimeUnavailableError("runtime child is owned by a live controller")
        try:
            identity = self._process_inspector(record.pid)
        except Exception as exc:
            self._status = "recovery_required"
            raise RuntimeUnavailableError("recorded runtime process identity is ambiguous") from exc
        if identity is None:
            if not sys.platform.startswith("linux"):
                self._status = "recovery_required"
                raise RuntimeUnavailableError(
                    "recorded runtime leader is absent; descendant ownership is ambiguous"
                )
            try:
                descendants = self._recovered_descendants(record)
                if descendants:
                    await self._fence_recovered_descendants(record, descendants)
            except Exception as exc:
                self._status = "recovery_required"
                raise RuntimeUnavailableError(
                    "recorded runtime descendants could not be fenced"
                ) from exc
            self._remove_ownership_record(record.launch_nonce)
            self._remove_launch_intent(record.launch_nonce)
            self._ownership_checked = True
            return
        if not self._ownership_identity_matches(record, identity):
            self._status = "recovery_required"
            raise RuntimeUnavailableError("recorded runtime process identity is ambiguous")
        try:
            final_identity = self._process_inspector(record.pid)
        except Exception as exc:
            self._status = "recovery_required"
            raise RuntimeUnavailableError("recorded runtime process identity is ambiguous") from exc
        if final_identity is None or not self._ownership_identity_matches(record, final_identity):
            self._status = "recovery_required"
            raise RuntimeUnavailableError("recorded runtime process changed before fencing")
        try:
            await self._signal_verified_identity(final_identity, signal.SIGTERM)
            term_signaled: set[tuple[int, str | None]] = set()
            deadline = asyncio.get_running_loop().time() + self._shutdown_timeout
            while asyncio.get_running_loop().time() < deadline:
                descendants = self._recovered_descendants(record)
                new_descendants = tuple(
                    descendant
                    for descendant in descendants
                    if (descendant.pid, descendant.process_birth) not in term_signaled
                )
                await self._signal_verified_descendants(new_descendants, signal.SIGTERM)
                term_signaled.update(
                    (descendant.pid, descendant.process_birth) for descendant in new_descendants
                )
                self._reap_descendants(descendants)
                leader = self._process_inspector(record.pid)
                if leader is not None and not self._same_process_identity(final_identity, leader):
                    raise RuntimeUnavailableError("recovered runtime leader identity changed")
                if leader is None and not self._recovered_descendants(record):
                    break
                await asyncio.sleep(0.05)

            leader = self._process_inspector(record.pid)
            if leader is not None:
                if not self._same_process_identity(final_identity, leader):
                    raise RuntimeUnavailableError("recovered runtime leader identity changed")
                await self._signal_verified_identity(final_identity, signal.SIGKILL)
            kill_deadline = asyncio.get_running_loop().time() + self._shutdown_timeout
            descendants = self._recovered_descendants(record)
            while (leader is not None or descendants) and (
                asyncio.get_running_loop().time() < kill_deadline
            ):
                await self._signal_verified_descendants(descendants, signal.SIGKILL)
                self._reap_descendants(descendants)
                await asyncio.sleep(0.05)
                leader = self._process_inspector(record.pid)
                if leader is not None and not self._same_process_identity(final_identity, leader):
                    raise RuntimeUnavailableError("recovered runtime leader identity changed")
                descendants = self._recovered_descendants(record)
            if leader is not None or descendants:
                raise RuntimeUnavailableError("recovered runtime processes did not exit")
        except Exception as exc:
            self._status = "recovery_required"
            raise RuntimeUnavailableError("recorded runtime process could not be fenced") from exc
        self._remove_ownership_record(record.launch_nonce)
        self._remove_launch_intent(record.launch_nonce)
        self._ownership_checked = True

    async def _fence_recovered_descendants(
        self,
        record: _OwnershipRecord,
        descendants: tuple[RuntimeProcessIdentity, ...],
    ) -> None:
        term_signaled: set[tuple[int, str | None]] = set()
        deadline = asyncio.get_running_loop().time() + self._shutdown_timeout
        while descendants and asyncio.get_running_loop().time() < deadline:
            new_descendants = tuple(
                descendant
                for descendant in descendants
                if (descendant.pid, descendant.process_birth) not in term_signaled
            )
            await self._signal_verified_descendants(new_descendants, signal.SIGTERM)
            term_signaled.update(
                (descendant.pid, descendant.process_birth) for descendant in new_descendants
            )
            self._reap_descendants(descendants)
            await asyncio.sleep(0.05)
            descendants = self._recovered_descendants(record)
        if not descendants:
            return
        kill_deadline = asyncio.get_running_loop().time() + self._shutdown_timeout
        while descendants and asyncio.get_running_loop().time() < kill_deadline:
            await self._signal_verified_descendants(descendants, signal.SIGKILL)
            self._reap_descendants(descendants)
            await asyncio.sleep(0.05)
            descendants = self._recovered_descendants(record)
        if descendants:
            raise RuntimeUnavailableError("recovered runtime descendants did not exit")

    def _recovered_descendants(
        self,
        record: _OwnershipRecord,
    ) -> tuple[RuntimeProcessIdentity, ...]:
        if not sys.platform.startswith("linux"):
            return ()
        try:
            if self._uses_default_descendant_inspector:
                descendants = self._inspect_linux_descendants(
                    record.pid,
                    record.launch_nonce,
                    expected_executable=Path(record.executable),
                    expected_argv=record.argv,
                )
            else:
                descendants = self._descendant_inspector(record.pid, record.launch_nonce)
        except Exception as exc:
            raise RuntimeUnavailableError("recovered runtime descendants are ambiguous") from exc
        for descendant in descendants:
            if descendant.process_birth is None or descendant.launch_nonce != record.launch_nonce:
                raise RuntimeUnavailableError("recovered runtime descendant identity is ambiguous")
        return tuple(sorted(descendants, key=lambda item: item.pid))

    def _ownership_identity_matches(
        self,
        record: _OwnershipRecord,
        identity: RuntimeProcessIdentity,
    ) -> bool:
        if identity.pid != record.pid or identity.process_group != record.process_group:
            return False
        if identity.executable.resolve() != Path(record.executable).resolve():
            return False
        observed = identity.argv
        expected = record.argv
        if observed != expected and observed != (record.executable, *expected):
            return False
        return (
            identity.process_birth is not None
            and identity.process_birth == record.process_birth
            and identity.launch_nonce is not None
            and identity.launch_nonce == record.launch_nonce
        )

    def _write_ownership_record(self, child: _Child) -> None:
        record = _OwnershipRecord(
            pid=child.process.pid,
            process_group=child.process_group,
            host_pid=os.getpid(),
            host_birth=self._capture_process_birth(os.getpid()),
            mode="generation" if child.generation is not None else "legacy",
            generation_id=child.generation_id,
            legacy_source_root=str(child.project_root) if child.project_root is not None else None,
            launch_nonce=child.launch_nonce,
            process_birth=child.process_birth,
            executable=str(child.executable),
            argv=child.argv,
        )
        payload = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._atomic_write_control_file(self._control_path, payload, label="ownership record")

    def _read_ownership_record(self) -> _OwnershipRecord | None:
        if not os.path.lexists(self._control_path):
            return None
        payload = self._read_private_control_file(
            self._control_path,
            label="runtime ownership record",
        )
        try:
            return _OwnershipRecord.model_validate_json(payload)
        except Exception as exc:
            raise RuntimeUnavailableError("runtime ownership record is invalid") from exc

    def _write_launch_intent(
        self,
        *,
        generation: RuntimeGenerationSpec | None,
        project_root: Path | None,
        launch_nonce: str,
        executable: Path,
        argv: tuple[str, ...],
    ) -> None:
        intent = _LaunchIntent(
            host_pid=os.getpid(),
            mode="generation" if generation is not None else "legacy",
            generation_id=generation.generation_id if generation is not None else None,
            legacy_source_root=str(project_root) if project_root is not None else None,
            launch_nonce=launch_nonce,
            executable=str(executable),
            argv=argv,
        )
        payload = json.dumps(
            intent.model_dump(mode="json"),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        self._atomic_write_control_file(self._intent_path, payload, label="launch intent")

    def _read_launch_intent(self) -> _LaunchIntent | None:
        if not os.path.lexists(self._intent_path):
            return None
        payload = self._read_private_control_file(
            self._intent_path,
            label="runtime launch intent",
        )
        try:
            return _LaunchIntent.model_validate_json(payload)
        except Exception as exc:
            raise RuntimeUnavailableError("runtime launch intent is invalid") from exc

    def _atomic_write_control_file(self, path: Path, payload: bytes, *, label: str) -> None:
        parent = self._secure_control_parent()
        temporary = parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
        descriptor: int | None = None
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(temporary, flags, 0o600)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(descriptor, remaining)
                if written < 1:
                    raise OSError(f"runtime {label} write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            self._fsync_directory(parent)
        except OSError as exc:
            raise RuntimeUnavailableError(f"runtime {label} could not be persisted") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(OSError):
                temporary.unlink()

    def _read_private_control_file(self, path: Path, *, label: str) -> bytes:
        self._secure_control_parent()
        descriptor: int | None = None
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
                or metadata.st_size < 2
                or metadata.st_size > _OWNERSHIP_MAX_BYTES
            ):
                raise RuntimeUnavailableError(f"{label} is unsafe")
            chunks: list[bytes] = []
            remaining = metadata.st_size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 16 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            payload = b"".join(chunks)
            if len(payload) != metadata.st_size:
                raise RuntimeUnavailableError(f"{label} changed while reading")
            return payload
        except RuntimeUnavailableError:
            raise
        except Exception as exc:
            raise RuntimeUnavailableError(f"{label} is invalid") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def _remove_ownership_record(self, launch_nonce: str) -> None:
        if not os.path.lexists(self._control_path):
            return
        record = self._read_ownership_record()
        if record is None or record.launch_nonce != launch_nonce:
            return
        try:
            self._control_path.unlink()
            self._fsync_directory(self._control_path.parent)
        except OSError as exc:
            raise RuntimeUnavailableError("runtime ownership record could not be cleared") from exc

    def _remove_launch_intent(self, launch_nonce: str) -> None:
        if not os.path.lexists(self._intent_path):
            return
        intent = self._read_launch_intent()
        if intent is None or intent.launch_nonce != launch_nonce:
            return
        try:
            self._intent_path.unlink()
            self._fsync_directory(self._intent_path.parent)
        except OSError as exc:
            raise RuntimeUnavailableError("runtime launch intent could not be cleared") from exc

    def _secure_control_parent(self) -> Path:
        parent = self._control_path.parent
        if self._control_path.name in {"", ".", ".."}:
            raise RuntimeUnavailableError("runtime control path is invalid")
        current = Path(parent.anchor)
        for component in parent.parts[1:]:
            current /= component
            if os.path.lexists(current) and current.is_symlink():
                raise RuntimeUnavailableError("runtime control path has a symbolic-link ancestor")
        existed = parent.exists()
        try:
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not existed:
                parent.chmod(0o700)
            metadata = parent.lstat()
        except OSError as exc:
            raise RuntimeUnavailableError("runtime control directory is unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise RuntimeUnavailableError("runtime control directory is not private")
        return parent

    def _inspect_linux_descendants(
        self,
        leader_pid: int,
        launch_nonce: str,
        *,
        expected_executable: Path | None = None,
        expected_argv: tuple[str, ...] | None = None,
    ) -> tuple[RuntimeProcessIdentity, ...]:
        if not sys.platform.startswith("linux"):
            return ()
        processes = self._linux_process_table()
        descendants: set[int] = set()
        parents = {leader_pid}
        while True:
            discovered = {
                pid
                for pid, metadata in processes.items()
                if metadata.parent_pid in parents | descendants
            } - descendants - {leader_pid}
            if not discovered:
                break
            descendants.update(discovered)
        group_members = {
            pid
            for pid, metadata in processes.items()
            if metadata.process_group == leader_pid and pid != leader_pid
        }
        adopted = {
            pid
            for pid, metadata in processes.items()
            if metadata.parent_pid == os.getpid() and pid != leader_pid
        }
        root_isolated = os.geteuid() == 0
        runtime_uid = self._child_uid if root_isolated else os.geteuid()
        if runtime_uid is None:
            raise RuntimeUnavailableError("runtime child identity is unavailable")
        # Root production reserves a UID for the runtime. Non-root recovery scans
        # same-UID processes and accepts only those retaining the private launch nonce.
        runtime_owned = {
            pid
            for pid, metadata in processes.items()
            if pid != leader_pid and metadata.owned_by(runtime_uid)
        }
        candidates = descendants | group_members | adopted | runtime_owned
        identities: list[RuntimeProcessIdentity] = []
        for pid in sorted(candidates):
            metadata = processes[pid]
            structurally_bound = pid in descendants or pid in group_members
            if structurally_bound and not metadata.owned_by(runtime_uid):
                raise RuntimeUnavailableError("runtime descendant ownership changed")
            if not structurally_bound and pid not in runtime_owned:
                continue
            identity = self._process_inspector(pid)
            if identity is None:
                continue
            if (
                identity.parent_pid != metadata.parent_pid
                or identity.process_group != metadata.process_group
                or identity.process_birth != metadata.process_birth
            ):
                raise RuntimeUnavailableError("runtime descendant changed during inspection")
            if identity.launch_nonce != launch_nonce:
                if (
                    not root_isolated
                    and expected_executable is not None
                    and expected_argv is not None
                    and self._runtime_command_matches(
                        identity,
                        expected_executable=expected_executable,
                        expected_argv=expected_argv,
                    )
                ):
                    raise RuntimeUnavailableError(
                        "a possible runtime descendant removed its launch identity"
                    )
                if structurally_bound or (root_isolated and pid in runtime_owned):
                    raise RuntimeUnavailableError(
                        "a runtime descendant removed its launch identity"
                    )
                continue
            identities.append(identity)
        return tuple(identities)

    @staticmethod
    def _runtime_command_matches(
        identity: RuntimeProcessIdentity,
        *,
        expected_executable: Path,
        expected_argv: tuple[str, ...],
    ) -> bool:
        try:
            executable_matches = identity.executable.resolve() == expected_executable.resolve()
        except OSError:
            return False
        return executable_matches and identity.argv in {
            expected_argv,
            (str(expected_executable), *expected_argv),
        }

    @staticmethod
    def _linux_process_table() -> dict[int, _LinuxProcessMetadata]:
        processes: dict[int, _LinuxProcessMetadata] = {}
        try:
            entries = tuple(Path("/proc").iterdir())
        except OSError as exc:
            raise RuntimeUnavailableError("Linux process table could not be enumerated") from exc
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                parent_pid, process_group, process_birth = RuntimeSupervisor._linux_process_metadata(entry)
                proc_uid, status_uids = RuntimeSupervisor._linux_process_uids(entry)
            except FileNotFoundError:
                continue
            except (OSError, RuntimeUnavailableError) as exc:
                raise RuntimeUnavailableError(
                    "Linux process table containment could not be proven"
                ) from exc
            processes[int(entry.name)] = _LinuxProcessMetadata(
                parent_pid=parent_pid,
                process_group=process_group,
                process_birth=process_birth,
                proc_uid=proc_uid,
                status_uids=status_uids,
            )
        return processes

    @staticmethod
    def _linux_process_uids(proc_root: Path) -> tuple[int, tuple[int, int, int, int]]:
        metadata = proc_root.stat(follow_symlinks=False)
        raw_status = (proc_root / "status").read_text(encoding="ascii")
        uid_line = next((line for line in raw_status.splitlines() if line.startswith("Uid:")), "")
        values = uid_line.removeprefix("Uid:").split()
        if len(values) != 4 or any(not value.isdigit() for value in values):
            raise RuntimeUnavailableError("runtime process UID status is invalid")
        parsed = tuple(int(value) for value in values)
        return metadata.st_uid, (parsed[0], parsed[1], parsed[2], parsed[3])

    @staticmethod
    def _inspect_process(pid: int) -> RuntimeProcessIdentity | None:
        if not RuntimeSupervisor._pid_alive(pid):
            return None
        proc_root = Path("/proc") / str(pid)
        if not proc_root.is_dir():
            raise RuntimeUnavailableError("safe runtime process inspection is unavailable")
        try:
            raw_argv = (proc_root / "cmdline").read_bytes()
            raw_environment = (proc_root / "environ").read_bytes()
            parent_pid, process_group, process_birth = (
                RuntimeSupervisor._linux_process_metadata(proc_root)
            )
        except OSError as exc:
            if not RuntimeSupervisor._pid_alive(pid):
                return None
            raise RuntimeUnavailableError("recorded runtime process could not be inspected") from exc
        argv = tuple(value.decode("utf-8", errors="surrogateescape") for value in raw_argv.split(b"\0") if value)
        if not argv:
            raise RuntimeUnavailableError("recorded runtime process command is unavailable")
        try:
            executable = RuntimeSupervisor._linux_process_executable(proc_root, argv)
        except OSError as exc:
            if not RuntimeSupervisor._pid_alive(pid):
                return None
            raise RuntimeUnavailableError("recorded runtime executable could not be inspected") from exc
        launch_nonce = None
        for entry in raw_environment.split(b"\0"):
            if entry.startswith(b"OPENTULPA_LAUNCH_NONCE="):
                launch_nonce = entry.partition(b"=")[2].decode("ascii", errors="strict")
                break
        return RuntimeProcessIdentity(
            pid=pid,
            process_group=process_group,
            executable=executable,
            argv=argv,
            parent_pid=parent_pid,
            process_birth=process_birth,
            launch_nonce=launch_nonce,
        )

    @staticmethod
    def _linux_process_executable(proc_root: Path, argv: tuple[str, ...]) -> Path:
        try:
            return (proc_root / "exe").resolve(strict=True)
        except PermissionError:
            # Linux may hide /proc/<pid>/exe after a trusted UID drop without
            # SYS_PTRACE. The private nonce, birth, UID, group, and exact argv
            # still bind this process; requiring SYS_PTRACE would be broader.
            return Path(argv[0])

    @staticmethod
    def _capture_process_birth(pid: int) -> str:
        if sys.platform.startswith("linux"):
            proc_root = Path("/proc") / str(pid)
            try:
                return RuntimeSupervisor._linux_process_birth(proc_root)
            except OSError as exc:
                raise RuntimeUnavailableError("runtime process birth could not be captured") from exc
        if sys.platform == "darwin":
            try:
                result = subprocess.run(
                    ["/bin/ps", "-p", str(pid), "-o", "lstart="],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise RuntimeUnavailableError("runtime process birth could not be captured") from exc
            started = result.stdout.strip()
            if not started:
                raise RuntimeUnavailableError("runtime process birth could not be captured")
            return f"darwin:{started}"
        raise RuntimeUnavailableError("runtime process birth inspection is unsupported")

    @staticmethod
    def _linux_process_birth(proc_root: Path) -> str:
        return RuntimeSupervisor._linux_process_metadata(proc_root)[2]

    @staticmethod
    def _linux_process_metadata(proc_root: Path) -> tuple[int, int, str]:
        raw_stat = (proc_root / "stat").read_text(encoding="ascii")
        close_paren = raw_stat.rfind(")")
        fields = raw_stat[close_paren + 2 :].split() if close_paren >= 0 else []
        if (
            len(fields) <= 19
            or not fields[1].isdigit()
            or not fields[2].isdigit()
            or not fields[19].isdigit()
        ):
            raise RuntimeUnavailableError("runtime process start time is invalid")
        return int(fields[1]), int(fields[2]), f"linux:{fields[19]}"

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    @staticmethod
    def _safe_error(error: Exception) -> str:
        text = str(error or "runtime failed").strip()
        return _SECRET_LINE.sub(r"\1\2[redacted]", text)[:1_000]

    @staticmethod
    def _validated_project_root(project_root: Path) -> Path:
        root = project_root.expanduser()
        if root.is_symlink() or not root.is_dir():
            raise RuntimeUnavailableError("runtime source root is unavailable")
        resolved = root.resolve(strict=True)
        if not (resolved / "src" / "opentulpa" / "__init__.py").is_file():
            raise RuntimeUnavailableError("runtime source root is invalid")
        return resolved

    @staticmethod
    def _coerce_generation_spec(
        value: RuntimeGenerationSpec | Mapping[str, object],
    ) -> RuntimeGenerationSpec:
        if isinstance(value, RuntimeGenerationSpec):
            return value
        return RuntimeGenerationSpec.model_validate(value)

    @staticmethod
    def _validate_identity_value(value: int | None, *, label: str) -> None:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{label} must be a non-negative integer")

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @staticmethod
    def _looks_like_source_checkout(root: Path) -> bool:
        return (root / "src" / "opentulpa" / "__init__.py").is_file()

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
        except ValueError:
            return False
        return True

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "DescendantInspector",
    "ProcessFencer",
    "ProcessInspector",
    "ProcessSignaler",
    "RuntimeGenerationSpec",
    "RuntimeLogEntry",
    "RuntimeProcessIdentity",
    "RuntimeSupervisor",
    "RuntimeUnavailableError",
]
