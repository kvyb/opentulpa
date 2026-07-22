"""Tenant-scoped control plane for immutable capability revisions."""

from __future__ import annotations

import asyncio
import builtins
import hashlib
import inspect
import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from opentulpa.capabilities.bundled import BUNDLED_CAPABILITY_TEMPLATES
from opentulpa.capabilities.models import (
    AgentInterfaceBinding,
    CapabilityActivation,
    CapabilityManifest,
    CapabilitySecretBinding,
    CapabilityTestCheck,
    CapabilityTestResult,
    CapabilityTestStatus,
    SecretRequirement,
    SecretSource,
    WorkerKind,
    WorkerRuntime,
    WorkerTransport,
    canonical_json_digest,
)
from opentulpa.capabilities.revisions import (
    CapabilityRevisionConflictError,
    CapabilityRevisionNotFoundError,
    CapabilityRevisionStore,
)
from opentulpa.capabilities.workers import (
    CapabilityWorkerManager,
    WorkerLifecycleError,
)
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.specs import AgentRunBinding

logger = logging.getLogger(__name__)

_MAX_RELEASE_ACTIVATION_STATE_BYTES = 16 * 1024 * 1024


class _ReleaseSeedActivation(BaseModel):
    """Release-coupled seed values restored without rewinding product tables."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    namespace: str = Field(min_length=1, max_length=200)
    capability_name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    manifest_body_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    config: dict[str, Any] = Field(default_factory=dict)
    secret_handles: dict[str, str] = Field(default_factory=dict)
    secret_bindings: dict[str, CapabilitySecretBinding] = Field(default_factory=dict)
    agent_binding: AgentRunBinding | None = None


class _ReleaseSeedActivationState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    manifests: dict[str, str] = Field(default_factory=dict)
    activations: tuple[_ReleaseSeedActivation, ...] = Field(default=(), max_length=10_000)


@dataclass(frozen=True, slots=True)
class _ReleaseStateLookup:
    configured: bool
    manifests: dict[str, str]
    activations: dict[tuple[str, str], _ReleaseSeedActivation]


class CapabilityControlError(RuntimeError):
    """Base error for safe capability lifecycle transitions."""


class CapabilityEvaluationUnavailableError(CapabilityControlError):
    """No isolated evaluator is configured."""


class CapabilityTestRequiredError(CapabilityControlError):
    """Activation was attempted without a passing digest-bound test."""


class CapabilityRuntimeUnavailableError(CapabilityControlError):
    """A worker or secret host required for activation is unavailable."""


class CapabilityEvaluator(Protocol):
    """Isolated evaluator port; implementations must not run candidates on the host."""

    async def evaluate(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> Sequence[CapabilityTestCheck]: ...


class CapabilitySecretResolver(Protocol):
    """Resolve tenant-owned handles to ephemeral worker environment grants."""

    async def bind(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
        secret_handles: Mapping[str, str],
    ) -> Mapping[str, CapabilitySecretBinding]: ...

    async def bind_agent_run(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> AgentRunBinding | None: ...

    async def resolve(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        secret_bindings: Mapping[str, CapabilitySecretBinding],
        agent_binding: AgentRunBinding | None,
    ) -> Mapping[str, str]: ...

    async def revoke(self, *, tenant_id: str, instance_id: str) -> None: ...


class CapabilityToolHost(Protocol):
    """Publish exact MCP worker tools into the tenant agent registry."""

    async def start(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        config: Mapping[str, object],
        secrets: Mapping[str, str],
        worker_endpoints: Mapping[str, str],
        worker_endpoint_headers: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None: ...

    async def replace(
        self,
        *,
        tenant_id: str,
        previous_instance_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        config: Mapping[str, object],
        secrets: Mapping[str, str],
        worker_endpoints: Mapping[str, str],
        worker_endpoint_headers: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None: ...

    async def stop(self, instance_id: str) -> None: ...


class CapabilityControlService:
    """Coordinate revision, test, secret, worker, and activation boundaries."""

    def __init__(
        self,
        *,
        revisions: CapabilityRevisionStore,
        evaluator: CapabilityEvaluator | None = None,
        workers: CapabilityWorkerManager | None = None,
        tool_host: CapabilityToolHost | None = None,
        secret_resolver: CapabilitySecretResolver | None = None,
        bundled: Sequence[CapabilityManifest] = BUNDLED_CAPABILITY_TEMPLATES,
        config_defaults: Callable[[str, CapabilityManifest], Mapping[str, Any]] | None = None,
        blocked_capabilities: Sequence[str] = (),
        release_state_path: Path | None = None,
    ) -> None:
        self._revisions = revisions
        self._evaluator = evaluator
        self._workers = workers
        self._tool_host = tool_host
        self._secret_resolver = secret_resolver
        self._bundled = tuple(bundled)
        self._config_defaults = config_defaults
        self._release_state_path = (
            release_state_path.expanduser().resolve() if release_state_path is not None else None
        )
        self._blocked_capabilities = frozenset(
            str(name or "").strip().lower() for name in blocked_capabilities
        )
        if "" in self._blocked_capabilities:
            raise ValueError("blocked capability names must be non-empty")
        self._lifecycle_lock = asyncio.Lock()
        self._started = False
        self._startup_errors: set[tuple[str, str]] = set()
        self._secret_refresh_tasks: set[asyncio.Task[Any]] = set()

    @property
    def started(self) -> bool:
        return self._started

    async def healthy(self) -> bool:
        """Check persisted activation state and every active worker generation."""

        if not self._started or self._startup_errors or self._revisions.list_all_deactivating():
            return False
        active_values = self._revisions.list_all_active()
        for active in active_values:
            if active.capability_name in self._blocked_capabilities:
                continue
            manifest = self._revisions.get(
                namespace=active.namespace,
                capability_name=active.capability_name,
                revision=active.revision,
            )
            if manifest is None:
                return False
            if not manifest.workers:
                continue
            if self._workers is None:
                return False
            try:
                if not await self._workers.healthy(_instance_id(active)):
                    return False
            except Exception:
                logger.exception(
                    "Capability worker liveness check failed: tenant=%s capability=%s",
                    active.namespace,
                    active.capability_name,
                )
                return False
        return True

    def notify_secret_changed(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
    ) -> None:
        """Schedule reconciliation after a vault revision changes."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error(
                "Capability secret change could not be scheduled outside an event loop: "
                "tenant=%s secret=%s",
                tenant_id,
                secret_id,
            )
            return
        task = loop.create_task(
            self.refresh_secret_bindings(
                tenant_id=tenant_id,
                actor_id=actor_id,
                secret_id=secret_id,
            ),
            name=f"capability-secret-refresh:{tenant_id}:{secret_id}",
        )
        self._secret_refresh_tasks.add(task)
        task.add_done_callback(self._secret_refresh_done)

    async def wait_for_secret_refresh(self) -> None:
        """Wait until all already scheduled secret reconciliations finish."""

        while self._secret_refresh_tasks:
            await asyncio.gather(
                *tuple(self._secret_refresh_tasks),
                return_exceptions=True,
            )

    def _secret_refresh_done(self, task: asyncio.Task[Any]) -> None:
        self._secret_refresh_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Capability secret reconciliation task failed: %s",
                type(error).__name__,
            )

    async def refresh_secret_bindings(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
    ) -> tuple[CapabilityActivation, ...]:
        """Restart every generation bound to a rotated tenant secret."""

        refreshed: list[CapabilityActivation] = []
        async with self._lifecycle_lock:
            active_values = self._revisions.list_active(namespace=tenant_id)
            for active in active_values:
                if active.capability_name in self._blocked_capabilities:
                    continue
                if secret_id not in active.secret_handles.values():
                    continue
                manifest = self._revisions.get(
                    namespace=tenant_id,
                    capability_name=active.capability_name,
                    revision=active.revision,
                )
                if manifest is None:
                    continue
                try:
                    bindings = await self._bind_secret_handles(
                        tenant_id=tenant_id,
                        manifest=manifest,
                        secret_handles=active.secret_handles,
                    )
                    if bindings == active.secret_bindings:
                        continue
                    refreshed.append(
                        await self._activate_locked(
                            tenant_id=tenant_id,
                            actor_id=actor_id,
                            capability_name=active.capability_name,
                            revision=active.revision,
                            expected_generation=active.generation,
                            config=active.config,
                            secret_handles=active.secret_handles,
                            preserve_agent_binding=True,
                        )
                    )
                except Exception:
                    logger.exception(
                        "Capability was disabled after secret reconciliation failed: "
                        "tenant=%s capability=%s secret=%s",
                        tenant_id,
                        active.capability_name,
                        secret_id,
                    )
                    current = self._revisions.active(
                        namespace=tenant_id,
                        capability_name=active.capability_name,
                    )
                    if current is not None and current.generation != active.generation:
                        refreshed.append(current)
                        self._startup_errors.add((tenant_id, active.capability_name))
                        continue
                    await self._disable_active_locked(active)
                    self._startup_errors.add((tenant_id, active.capability_name))
            return tuple(refreshed)

    async def start(self) -> None:
        """Restore persisted worker activations after a process restart."""

        async with self._lifecycle_lock:
            if self._started:
                return
            await self._reconcile_deactivations_locked()
            await self._reconcile_legacy_agent_bindings_locked()
            await self._reconcile_seed_activations_locked()
            self._started = True
            for active in self._revisions.list_all_active():
                if active.capability_name in self._blocked_capabilities:
                    try:
                        await self._disable_active_locked(active)
                        logger.info(
                            "Host-disabled capability was deactivated: tenant=%s capability=%s",
                            active.namespace,
                            active.capability_name,
                        )
                    except Exception:
                        self._startup_errors.add((active.namespace, active.capability_name))
                        logger.exception(
                            "Host-disabled capability could not be deactivated: "
                            "tenant=%s capability=%s",
                            active.namespace,
                            active.capability_name,
                        )
                    continue
                manifest = self._revisions.get(
                    namespace=active.namespace,
                    capability_name=active.capability_name,
                    revision=active.revision,
                )
                if manifest is None or not manifest.workers:
                    continue
                try:
                    _validate_activation_agent_binding(active, manifest)
                except CapabilityRuntimeUnavailableError:
                    try:
                        await self._disable_active_locked(active)
                        logger.error(
                            "Legacy or invalid interface generation was fenced: "
                            "tenant=%s capability=%s generation=%s",
                            active.namespace,
                            active.capability_name,
                            active.generation,
                        )
                    except Exception:
                        self._startup_errors.add((active.namespace, active.capability_name))
                        logger.exception(
                            "Invalid interface generation could not be fenced: "
                            "tenant=%s capability=%s generation=%s",
                            active.namespace,
                            active.capability_name,
                            active.generation,
                        )
                    continue
                try:
                    _validate_config(manifest.config_schema, active.config)
                    bindings = await self._bind_secret_handles(
                        tenant_id=active.namespace,
                        manifest=manifest,
                        secret_handles=active.secret_handles,
                    )
                    if bindings != active.secret_bindings:
                        await self._activate_locked(
                            tenant_id=active.namespace,
                            actor_id="bootstrap",
                            capability_name=active.capability_name,
                            revision=active.revision,
                            expected_generation=active.generation,
                            config=active.config,
                            secret_handles=active.secret_handles,
                            allow_current_seed_without_attestation=True,
                            preserve_agent_binding=True,
                        )
                    else:
                        await self._start_workers(
                            tenant_id=active.namespace,
                            actor_id="bootstrap",
                            manifest=manifest,
                            instance_id=_instance_id(active),
                            config=active.config,
                            secret_bindings=active.secret_bindings,
                            agent_binding=active.agent_binding,
                        )
                except Exception:
                    self._startup_errors.add((active.namespace, active.capability_name))
                    logger.exception(
                        "Capability workers could not be restored: tenant=%s capability=%s",
                        active.namespace,
                        active.capability_name,
                    )

    async def shutdown(self) -> None:
        await self.wait_for_secret_refresh()
        async with self._lifecycle_lock:
            if not self._started:
                if self._workers is not None:
                    await self._workers.aclose()
                return
            for transition in reversed(self._revisions.list_all_deactivating()):
                try:
                    await self._quiesce_generation(transition)
                except Exception:
                    logger.exception(
                        "Pending capability deactivation could not be quiesced at shutdown: "
                        "tenant=%s capability=%s",
                        transition.namespace,
                        transition.capability_name,
                    )
            for active in reversed(self._revisions.list_all_active()):
                instance_id = _instance_id(active)
                if self._tool_host is not None:
                    try:
                        await self._tool_host.stop(instance_id)
                    except Exception:
                        logger.exception(
                            "Capability tool shutdown failed: tenant=%s capability=%s",
                            active.namespace,
                            active.capability_name,
                        )
                if self._workers is not None:
                    try:
                        await self._workers.stop(instance_id)
                    except WorkerLifecycleError:
                        logger.exception(
                            "Capability worker shutdown failed: tenant=%s capability=%s",
                            active.namespace,
                            active.capability_name,
                        )
                try:
                    await self._revoke_worker_credentials(
                        tenant_id=active.namespace,
                        instance_id=instance_id,
                    )
                except Exception:
                    logger.exception(
                        "Capability credential shutdown revocation failed: tenant=%s capability=%s",
                        active.namespace,
                        active.capability_name,
                    )
            if self._workers is not None:
                try:
                    await self._workers.aclose()
                except WorkerLifecycleError:
                    logger.exception("Capability worker host shutdown failed")
            self._started = False

    def list(self, *, tenant_id: str) -> list[dict[str, Any]]:
        return [
            self._view(tenant_id=tenant_id, manifest=manifest)
            for manifest in self._revisions.list_latest(namespace=tenant_id)
        ]

    def get(
        self,
        *,
        tenant_id: str,
        capability_name: str,
        revision: int | None = None,
    ) -> dict[str, Any] | None:
        if revision is None:
            values = self._revisions.list(
                namespace=tenant_id,
                capability_name=capability_name,
            )
            manifest = values[-1] if values else None
        else:
            manifest = self._revisions.get(
                namespace=tenant_id,
                capability_name=capability_name,
                revision=revision,
            )
        return self._view(tenant_id=tenant_id, manifest=manifest) if manifest else None

    def revisions(
        self,
        *,
        tenant_id: str,
        capability_name: str,
    ) -> builtins.list[CapabilityManifest]:
        return self._revisions.list(
            namespace=tenant_id,
            capability_name=capability_name,
        )

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        manifest: CapabilityManifest,
        expected_latest_revision: int | None,
    ) -> CapabilityManifest:
        _identity(tenant_id, "tenant_id")
        _identity(actor_id, "actor_id")
        if manifest.seed:
            raise ValueError("the capability seed flag is reserved for bundled templates")
        if manifest.module is not None:
            raise ValueError("tenant capabilities must run out of process")
        if manifest.artifact_digest is None:
            raise ValueError("tenant capabilities require a content-addressed artifact digest")
        unsafe_workers = [
            worker.name for worker in manifest.workers if worker.runtime is not WorkerRuntime.OCI
        ]
        if unsafe_workers:
            raise ValueError("tenant capabilities must use digest-pinned OCI workers")
        invalid_mcp_workers = [
            worker.name
            for worker in manifest.workers
            if worker.kind is WorkerKind.MCP
            and worker.transport is not WorkerTransport.STREAMABLE_HTTP
        ]
        if invalid_mcp_workers:
            raise ValueError("tenant MCP workers must use streamable HTTP")
        mismatched_images = [
            worker.name
            for worker in manifest.workers
            if worker.image is None or worker.image.rpartition("@")[2] != manifest.artifact_digest
        ]
        if mismatched_images:
            raise ValueError("tenant worker images must match the capability artifact digest")
        return self._revisions.append(
            namespace=tenant_id,
            manifest=manifest,
            expected_latest_revision=expected_latest_revision,
        )

    def seed_bundled(
        self,
        *,
        tenant_id: str,
        actor_id: str,
    ) -> tuple[CapabilityManifest, ...]:
        """Idempotently copy bundled templates into one tenant revision archive."""

        _identity(tenant_id, "tenant_id")
        _identity(actor_id, "actor_id")
        planned: list[tuple[CapabilityManifest, CapabilityManifest | None]] = []
        current_values: dict[str, CapabilityManifest] = {}
        for template in self._bundled:
            values = self._revisions.list(
                namespace=tenant_id,
                capability_name=template.name,
            )
            latest = values[-1] if values else None
            if latest is not None and _manifest_body_digest(latest) == _manifest_body_digest(
                template
            ):
                current_values[template.name] = latest
                continue
            if latest is not None and not latest.seed:
                raise CapabilityRevisionConflictError(
                    f"tenant capability {template.name!r} is not a bundled seed"
                )
            planned.append((template, latest))
        for template, latest in planned:
            revision = 1 if latest is None else latest.revision + 1
            current_values[template.name] = self._revisions.append(
                namespace=tenant_id,
                manifest=template.model_copy(update={"revision": revision}),
                expected_latest_revision=latest.revision if latest else None,
            )
        return tuple(current_values[template.name] for template in self._bundled)

    async def test(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        revision: int,
    ) -> CapabilityTestResult:
        _identity(actor_id, "actor_id")
        manifest = self._require_manifest(tenant_id, capability_name, revision)
        if self._evaluator is None:
            raise CapabilityEvaluationUnavailableError(
                "an isolated capability evaluator is not configured"
            )
        try:
            raw_checks = tuple(
                await self._evaluator.evaluate(
                    tenant_id=tenant_id,
                    manifest=manifest,
                )
            )
            checks = tuple(_sanitize_check(check) for check in raw_checks)
            if not checks:
                checks = (
                    CapabilityTestCheck(
                        name="evaluator",
                        status=CapabilityTestStatus.FAILED,
                        message="The evaluator returned no checks.",
                    ),
                )
        except Exception as exc:
            logger.warning(
                "Capability evaluation failed: tenant=%s capability=%s revision=%s exception=%s",
                tenant_id,
                capability_name,
                revision,
                type(exc).__name__,
            )
            checks = (
                CapabilityTestCheck(
                    name="evaluator",
                    status=CapabilityTestStatus.FAILED,
                    message="The isolated capability evaluation could not be completed.",
                ),
            )
        status = (
            CapabilityTestStatus.FAILED
            if any(check.status is CapabilityTestStatus.FAILED for check in checks)
            else CapabilityTestStatus.PASSED
        )
        result = CapabilityTestResult(
            namespace=tenant_id,
            capability_name=manifest.name,
            revision=manifest.revision,
            manifest_digest=manifest.content_digest,
            status=status,
            checks=checks,
            tested_at=datetime.now(UTC).isoformat(),
        )
        return self._revisions.record_test(result)

    async def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        revision: int,
        expected_generation: int | None,
        config: Mapping[str, Any] | None = None,
        secret_handles: Mapping[str, str] | None = None,
        refresh_agent_binding: bool = False,
    ) -> CapabilityActivation:
        async with self._lifecycle_lock:
            return await self._activate_locked(
                tenant_id=tenant_id,
                actor_id=actor_id,
                capability_name=capability_name,
                revision=revision,
                expected_generation=expected_generation,
                config=config,
                secret_handles=secret_handles,
                refresh_agent_binding=refresh_agent_binding,
            )

    async def rollback(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        expected_generation: int,
        config: Mapping[str, Any] | None = None,
        secret_handles: Mapping[str, str] | None = None,
    ) -> CapabilityActivation:
        async with self._lifecycle_lock:
            active = self._revisions.active(
                namespace=tenant_id,
                capability_name=capability_name,
            )
            if active is None:
                raise CapabilityRevisionNotFoundError(
                    f"capability {capability_name!r} is not active"
                )
            if active.generation != expected_generation:
                raise CapabilityRevisionConflictError(
                    f"expected activation generation {expected_generation!r}, "
                    f"found {active.generation!r}"
                )
            revisions = self._revisions.list(
                namespace=tenant_id,
                capability_name=capability_name,
            )
            target = next(
                (
                    manifest
                    for manifest in reversed(revisions)
                    if manifest.revision < active.revision
                ),
                None,
            )
            if target is None:
                raise CapabilityRevisionNotFoundError(
                    f"capability {capability_name!r} has no earlier revision"
                )
            return await self._activate_locked(
                tenant_id=tenant_id,
                actor_id=actor_id,
                capability_name=capability_name,
                revision=target.revision,
                expected_generation=expected_generation,
                config=config if config is not None else active.config,
                secret_handles=(
                    secret_handles if secret_handles is not None else active.secret_handles
                ),
                preserve_agent_binding=True,
            )

    async def deactivate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        expected_generation: int,
    ) -> CapabilityActivation:
        _identity(actor_id, "actor_id")
        async with self._lifecycle_lock:
            active = self._revisions.active(
                namespace=tenant_id,
                capability_name=capability_name,
            )
            if active is None:
                transition = self._revisions.deactivating(
                    namespace=tenant_id,
                    capability_name=capability_name,
                )
                if transition is None:
                    completed = self._revisions.inactive(
                        namespace=tenant_id,
                        capability_name=capability_name,
                    )
                    if completed is not None and completed.generation == expected_generation:
                        return completed
                    if completed is not None:
                        raise CapabilityRevisionConflictError(
                            f"expected activation generation {expected_generation!r}, "
                            f"found {completed.generation!r}"
                        )
                    raise CapabilityRevisionNotFoundError(
                        f"capability {capability_name!r} is not active"
                    )
                active = transition
            if active.generation != expected_generation:
                raise CapabilityRevisionConflictError(
                    f"expected activation generation {expected_generation!r}, "
                    f"found {active.generation!r}"
                )
            if (
                self._revisions.deactivating(
                    namespace=tenant_id,
                    capability_name=capability_name,
                )
                is None
            ):
                active = self._revisions.begin_deactivation(
                    namespace=tenant_id,
                    capability_name=capability_name,
                    expected_generation=expected_generation,
                )
            try:
                return await self._finish_deactivation_locked(
                    active,
                    persist_release_state=True,
                )
            except Exception as shutdown_error:
                try:
                    await self._restore_deactivation_locked(
                        active,
                        actor_id=actor_id,
                    )
                except Exception as restore_error:
                    self._startup_errors.add((tenant_id, capability_name))
                    logger.exception(
                        "Capability deactivation compensation failed and remains pending: "
                        "tenant=%s capability=%s generation=%s error=%s",
                        tenant_id,
                        capability_name,
                        expected_generation,
                        type(restore_error).__name__,
                    )
                    raise CapabilityRuntimeUnavailableError(
                        "capability deactivation is pending restart reconciliation"
                    ) from shutdown_error
                raise

    async def _activate_locked(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        revision: int,
        expected_generation: int | None,
        config: Mapping[str, Any] | None,
        secret_handles: Mapping[str, str] | None,
        allow_current_seed_without_attestation: bool = False,
        refresh_agent_binding: bool = False,
        preserve_agent_binding: bool = False,
    ) -> CapabilityActivation:
        _identity(actor_id, "actor_id")
        self._assert_not_blocked(capability_name)
        manifest = self._require_manifest(tenant_id, capability_name, revision)
        current_seed = manifest.seed and self._is_current_seed(manifest)
        if manifest.seed and not current_seed:
            raise CapabilityRuntimeUnavailableError(
                "the capability seed revision is not bundled with the current release"
            )
        attestation = self._revisions.test_result(
            namespace=tenant_id,
            capability_name=capability_name,
            revision=revision,
        )
        if not (allow_current_seed_without_attestation and current_seed) and (
            attestation is None
            or attestation.status is not CapabilityTestStatus.PASSED
            or attestation.manifest_digest != manifest.content_digest
        ):
            raise CapabilityTestRequiredError(
                "the exact capability revision must pass isolated tests before activation"
            )
        safe_config = (
            dict(self._config_defaults(tenant_id, manifest))
            if self._config_defaults is not None
            else {}
        )
        safe_config.update(dict(config or {}))
        _validate_config(manifest.config_schema, safe_config)
        safe_handles = _secret_handle_bindings(secret_handles or {})
        requirements = _secret_requirements(manifest)
        tenant_requirements = {
            name: requirement
            for name, requirement in requirements.items()
            if requirement.source is SecretSource.TENANT_HANDLE
        }
        unknown_handles = set(safe_handles).difference(tenant_requirements)
        if unknown_handles:
            raise ValueError(
                f"unknown capability secret handles: {', '.join(sorted(unknown_handles))}"
            )
        missing_handles = sorted(
            name
            for name, requirement in tenant_requirements.items()
            if requirement.required and name not in safe_handles
        )
        if missing_handles:
            raise ValueError(f"missing capability secret handles: {', '.join(missing_handles)}")
        safe_bindings = await self._bind_secret_handles(
            tenant_id=tenant_id,
            manifest=manifest,
            secret_handles=safe_handles,
        )
        current = self._revisions.active(
            namespace=tenant_id,
            capability_name=capability_name,
        )
        if (
            self._revisions.deactivating(
                namespace=tenant_id,
                capability_name=capability_name,
            )
            is not None
        ):
            raise CapabilityRevisionConflictError("capability deactivation is in progress")
        current_generation = current.generation if current else None
        if current_generation != expected_generation:
            raise CapabilityRevisionConflictError(
                f"expected activation generation {expected_generation!r}, "
                f"found {current_generation!r}"
            )
        if refresh_agent_binding and preserve_agent_binding:
            raise ValueError("agent binding cannot be refreshed and preserved together")
        declared_agent_binding = _declared_agent_binding(manifest)
        if refresh_agent_binding and declared_agent_binding is None:
            raise ValueError("capability does not declare an Agent API interface binding")
        if current is not None:
            current_manifest = self._require_manifest(
                tenant_id,
                current.capability_name,
                current.revision,
            )
            try:
                _validate_activation_agent_binding(current, current_manifest)
            except CapabilityRuntimeUnavailableError:
                await self._disable_active_locked(current)
                raise CapabilityRuntimeUnavailableError(
                    "the previous interface generation lacked a durable agent binding and "
                    "was fenced; activate again against the inactive generation"
                ) from None
        if (
            current is not None
            and current.revision == revision
            and current.config == safe_config
            and current.secret_handles == safe_handles
            and current.secret_bindings == safe_bindings
            and not refresh_agent_binding
        ):
            self._persist_release_seed_activations()
            return current
        if preserve_agent_binding:
            if current is None:
                raise CapabilityRuntimeUnavailableError(
                    "there is no prior generation binding to preserve"
                )
            safe_agent_binding = current.agent_binding
        else:
            safe_agent_binding = await self._bind_agent_run(
                tenant_id=tenant_id,
                manifest=manifest,
            )
        _validate_agent_binding(
            tenant_id=tenant_id,
            manifest=manifest,
            agent_binding=safe_agent_binding,
        )
        generation = self._revisions.next_generation(
            namespace=tenant_id,
            capability_name=capability_name,
        )
        instance_id = _next_instance_id(
            tenant_id=tenant_id,
            capability_name=capability_name,
            generation=generation,
        )
        worker_started = False
        tool_host_changed = False
        previous_instance_id = _instance_id(current) if current is not None else None
        if current is not None and self._workers is not None:
            assert previous_instance_id is not None
            await self._workers.stop(previous_instance_id)
        if manifest.workers:
            try:
                await self._start_workers(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    instance_id=instance_id,
                    manifest=manifest,
                    config=safe_config,
                    secret_bindings=safe_bindings,
                    agent_binding=safe_agent_binding,
                    replace_instance_id=previous_instance_id,
                )
                worker_started = True
                tool_host_changed = True
            except Exception:
                if current is not None:
                    await self._restore_previous_generation(
                        active=current,
                        manifest=self._require_manifest(
                            tenant_id,
                            current.capability_name,
                            current.revision,
                        ),
                        actor_id=actor_id,
                        tools_were_replaced=False,
                    )
                raise
        elif current is not None and self._tool_host is not None:
            assert previous_instance_id is not None
            await self._tool_host.stop(previous_instance_id)
            tool_host_changed = True
        try:
            activated = self._revisions.activate(
                namespace=tenant_id,
                capability_name=capability_name,
                revision=revision,
                expected_generation=expected_generation,
                config=safe_config,
                secret_handles=safe_handles,
                secret_bindings=safe_bindings,
                agent_binding=safe_agent_binding,
            )
        except Exception:
            try:
                if worker_started and self._workers is not None:
                    await self._workers.stop(instance_id)
                if current is not None:
                    await self._restore_previous_generation(
                        active=current,
                        manifest=self._require_manifest(
                            tenant_id,
                            current.capability_name,
                            current.revision,
                        ),
                        actor_id=actor_id,
                        tools_were_replaced=tool_host_changed,
                        replaced_instance_id=(instance_id if worker_started else None),
                    )
                elif worker_started and self._tool_host is not None:
                    await self._tool_host.stop(instance_id)
            finally:
                await self._revoke_worker_credentials(
                    tenant_id=tenant_id,
                    instance_id=instance_id,
                )
            raise
        if current is not None:
            assert previous_instance_id is not None
            await self._revoke_worker_credentials(
                tenant_id=tenant_id,
                instance_id=previous_instance_id,
            )
        self._persist_release_seed_activations()
        self._startup_errors.discard((tenant_id, capability_name))
        return activated

    async def _reconcile_deactivations_locked(self) -> None:
        """Complete shutdowns whose durable transition survived a process exit."""

        for transition in tuple(self._revisions.list_all_deactivating()):
            try:
                await self._finish_deactivation_locked(
                    transition,
                    persist_release_state=True,
                )
            except Exception:
                self._startup_errors.add((transition.namespace, transition.capability_name))
                logger.exception(
                    "Capability deactivation could not be reconciled: "
                    "tenant=%s capability=%s generation=%s",
                    transition.namespace,
                    transition.capability_name,
                    transition.generation,
                )

    async def _reconcile_legacy_agent_bindings_locked(self) -> None:
        """Fence pre-upgrade interface generations with ambiguous authority."""

        for active in tuple(self._revisions.list_all_active()):
            manifest = self._revisions.get(
                namespace=active.namespace,
                capability_name=active.capability_name,
                revision=active.revision,
            )
            if manifest is None:
                continue
            try:
                _validate_activation_agent_binding(active, manifest)
            except CapabilityRuntimeUnavailableError:
                try:
                    await self._disable_active_locked(active)
                    logger.error(
                        "Legacy or invalid interface generation was fenced: "
                        "tenant=%s capability=%s generation=%s",
                        active.namespace,
                        active.capability_name,
                        active.generation,
                    )
                except Exception:
                    self._startup_errors.add((active.namespace, active.capability_name))
                    logger.exception(
                        "Invalid interface generation could not be fenced: "
                        "tenant=%s capability=%s generation=%s",
                        active.namespace,
                        active.capability_name,
                        active.generation,
                    )

    async def _finish_deactivation_locked(
        self,
        transition: CapabilityActivation,
        *,
        persist_release_state: bool,
    ) -> CapabilityActivation:
        """Quiesce one hidden generation, then commit its inactive tombstone."""

        await self._quiesce_generation(transition)
        if persist_release_state:
            # The transitional generation is already excluded from active queries.
            # Persist the desired release state before committing the tombstone.
            self._persist_release_seed_activations()
        removed = self._revisions.deactivate(
            namespace=transition.namespace,
            capability_name=transition.capability_name,
            expected_generation=transition.generation,
        )
        self._startup_errors.discard((transition.namespace, transition.capability_name))
        return removed

    async def _quiesce_generation(self, activation: CapabilityActivation) -> None:
        """Remove every exposure of an exact generation in retry-safe order."""

        instance_id = _instance_id(activation)
        errors: list[Exception] = []
        if self._tool_host is not None:
            try:
                await self._tool_host.stop(instance_id)
            except Exception as exc:
                errors.append(exc)
        if self._workers is not None:
            try:
                await self._workers.stop(instance_id)
            except Exception as exc:
                errors.append(exc)
            try:
                await self._workers.fence(
                    tenant_id=activation.namespace,
                    capability_name=activation.capability_name,
                )
            except Exception as exc:
                errors.append(exc)
        try:
            await self._revoke_worker_credentials(
                tenant_id=activation.namespace,
                instance_id=instance_id,
            )
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise errors[0]

    async def _restore_deactivation_locked(
        self,
        transition: CapabilityActivation,
        *,
        actor_id: str,
    ) -> None:
        """Compensate a failed shutdown without changing the generation identity."""

        manifest = self._require_manifest(
            transition.namespace,
            transition.capability_name,
            transition.revision,
        )
        instance_id = _instance_id(transition)
        # Normalize partial stop/revoke outcomes before recreating the exact generation.
        await self._quiesce_generation(transition)
        if manifest.workers:
            await self._start_workers(
                tenant_id=transition.namespace,
                actor_id=actor_id,
                instance_id=instance_id,
                manifest=manifest,
                config=transition.config,
                secret_bindings=transition.secret_bindings,
                agent_binding=transition.agent_binding,
            )
        try:
            self._persist_release_seed_activations(
                restoring=(transition,),
            )
            self._revisions.cancel_deactivation(
                namespace=transition.namespace,
                capability_name=transition.capability_name,
                expected_generation=transition.generation,
            )
        except Exception:
            # A transition that remains durable must never retain runtime exposure.
            try:
                await self._quiesce_generation(transition)
            except Exception:
                logger.exception(
                    "Capability compensation cleanup failed: tenant=%s capability=%s generation=%s",
                    transition.namespace,
                    transition.capability_name,
                    transition.generation,
                )
            raise
        self._startup_errors.discard((transition.namespace, transition.capability_name))

    async def _reconcile_seed_activations_locked(self) -> None:
        """Fence persisted seed pointers to manifests in this exact release."""

        current_by_name = {manifest.name: manifest for manifest in self._bundled}
        release_state = self._load_release_seed_activations()
        for active in tuple(self._revisions.list_all_active()):
            manifest = self._revisions.get(
                namespace=active.namespace,
                capability_name=active.capability_name,
                revision=active.revision,
            )
            if manifest is None or not manifest.seed:
                continue
            current = current_by_name.get(active.capability_name)
            if current is None:
                await self._disable_active_locked(active, persist_release_state=False)
                logger.info(
                    "Candidate-only seed capability was disabled during release recovery: "
                    "tenant=%s capability=%s",
                    active.namespace,
                    active.capability_name,
                )
                continue
            current_digest = _manifest_body_digest(current)
            release_state_matches = (
                release_state.configured
                and release_state.manifests.get(current.name) == current_digest
            )
            restored = (
                release_state.activations.get((active.namespace, active.capability_name))
                if release_state_matches
                else None
            )
            if release_state_matches and restored is None:
                await self._disable_active_locked(active, persist_release_state=False)
                logger.info(
                    "Seed capability was disabled to match restored release state: "
                    "tenant=%s capability=%s",
                    active.namespace,
                    active.capability_name,
                )
                continue
            target = next(
                (
                    revision
                    for revision in reversed(
                        self._revisions.list(
                            namespace=active.namespace,
                            capability_name=active.capability_name,
                        )
                    )
                    if revision.seed and _manifest_body_digest(revision) == current_digest
                ),
                None,
            )
            if target is None:
                if restored is not None:
                    raise CapabilityRuntimeUnavailableError(
                        "the restored release seed revision is unavailable"
                    )
                await self._disable_active_locked(active, persist_release_state=False)
                logger.error(
                    "Current release seed revision was unavailable and the capability was "
                    "disabled: tenant=%s capability=%s",
                    active.namespace,
                    active.capability_name,
                )
                continue
            desired_config = active.config
            desired_handles = active.secret_handles
            desired_bindings = active.secret_bindings
            desired_agent_binding = active.agent_binding
            if restored is not None:
                _validate_release_seed_activation(restored, current)
                desired_config = restored.config
                desired_handles = restored.secret_handles
                desired_bindings = restored.secret_bindings
                desired_agent_binding = restored.agent_binding
            if (
                active.revision == target.revision
                and active.config == desired_config
                and active.secret_handles == desired_handles
                and active.secret_bindings == desired_bindings
                and active.agent_binding == desired_agent_binding
            ):
                continue
            # A release rollback creates a new generation. Fence and revoke the
            # previously persisted generation before moving the durable pointer.
            await self._quiesce_generation(active)
            self._revisions.activate(
                namespace=active.namespace,
                capability_name=active.capability_name,
                revision=target.revision,
                expected_generation=active.generation,
                config=dict(desired_config),
                secret_handles=dict(desired_handles),
                secret_bindings=dict(desired_bindings),
                agent_binding=desired_agent_binding,
            )

        for key, restored in release_state.activations.items():
            namespace, capability_name = key
            if (
                self._revisions.deactivating(
                    namespace=namespace,
                    capability_name=capability_name,
                )
                is not None
            ):
                continue
            if (
                self._revisions.active(
                    namespace=namespace,
                    capability_name=capability_name,
                )
                is not None
            ):
                continue
            current = current_by_name.get(capability_name)
            if current is None or release_state.manifests.get(
                capability_name
            ) != _manifest_body_digest(current):
                continue
            try:
                _validate_release_seed_activation(restored, current)
            except CapabilityRuntimeUnavailableError:
                logger.error(
                    "Legacy release activation without a durable agent binding was ignored: "
                    "tenant=%s capability=%s",
                    namespace,
                    capability_name,
                )
                continue
            target = next(
                (
                    revision
                    for revision in reversed(
                        self._revisions.list(
                            namespace=namespace,
                            capability_name=capability_name,
                        )
                    )
                    if revision.seed
                    and _manifest_body_digest(revision) == restored.manifest_body_digest
                ),
                None,
            )
            if target is None:
                raise CapabilityRuntimeUnavailableError(
                    "the restored release seed revision is unavailable"
                )
            self._revisions.activate(
                namespace=namespace,
                capability_name=capability_name,
                revision=target.revision,
                expected_generation=None,
                config=dict(restored.config),
                secret_handles=dict(restored.secret_handles),
                secret_bindings=dict(restored.secret_bindings),
                agent_binding=restored.agent_binding,
            )
        self._persist_release_seed_activations()

    def _load_release_seed_activations(self) -> _ReleaseStateLookup:
        path = self._release_state_path
        if path is None or not path.exists():
            return _ReleaseStateLookup(configured=False, manifests={}, activations={})
        if path.is_symlink() or not path.is_file():
            raise CapabilityRuntimeUnavailableError(
                "the release-coupled capability state is invalid"
            )
        try:
            if path.stat().st_size > _MAX_RELEASE_ACTIVATION_STATE_BYTES:
                raise ValueError("release capability state is too large")
            state = _ReleaseSeedActivationState.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            manifests = dict(state.manifests)
            if any(
                not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                for name, digest in manifests.items()
            ):
                raise ValueError("release manifest binding is invalid")
            activations: dict[tuple[str, str], _ReleaseSeedActivation] = {}
            for activation in state.activations:
                key = (activation.namespace, activation.capability_name)
                if key in activations:
                    raise ValueError("release capability activation is duplicated")
                if manifests.get(activation.capability_name) != activation.manifest_body_digest:
                    raise ValueError("release capability activation is not manifest-bound")
                activations[key] = activation
        except (OSError, UnicodeError, ValidationError, TypeError, ValueError) as exc:
            raise CapabilityRuntimeUnavailableError(
                "the release-coupled capability state is invalid"
            ) from exc
        return _ReleaseStateLookup(
            configured=True,
            manifests=manifests,
            activations=activations,
        )

    def _persist_release_seed_activations(
        self,
        *,
        restoring: Sequence[CapabilityActivation] = (),
    ) -> None:
        path = self._release_state_path
        if path is None:
            return
        manifests = {
            manifest.name: _manifest_body_digest(manifest)
            for manifest in self._bundled
            if manifest.seed
        }
        activations: list[_ReleaseSeedActivation] = []
        candidates = {
            (active.namespace, active.capability_name): active
            for active in self._revisions.list_all_active()
        }
        for active in restoring:
            candidates[(active.namespace, active.capability_name)] = active
        for active in candidates.values():
            manifest = self._revisions.get(
                namespace=active.namespace,
                capability_name=active.capability_name,
                revision=active.revision,
            )
            body_digest = _manifest_body_digest(manifest) if manifest is not None else ""
            if (
                manifest is None
                or not manifest.seed
                or manifests.get(active.capability_name) != body_digest
            ):
                continue
            activations.append(
                _ReleaseSeedActivation(
                    namespace=active.namespace,
                    capability_name=active.capability_name,
                    manifest_body_digest=body_digest,
                    config=active.config,
                    secret_handles=active.secret_handles,
                    secret_bindings=active.secret_bindings,
                    agent_binding=active.agent_binding,
                )
            )
        state = _ReleaseSeedActivationState(
            manifests=manifests,
            activations=tuple(
                sorted(
                    activations,
                    key=lambda item: (item.namespace, item.capability_name),
                )
            ),
        )
        try:
            payload = json.dumps(
                state.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            if len(payload) > _MAX_RELEASE_ACTIVATION_STATE_BYTES:
                raise ValueError("release capability state is too large")
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise OSError("release capability state path is invalid")
            temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
        except (OSError, TypeError, ValueError) as exc:
            raise CapabilityRuntimeUnavailableError(
                "the release-coupled capability state could not be persisted"
            ) from exc

    def _is_current_seed(self, manifest: CapabilityManifest) -> bool:
        return any(
            current.name == manifest.name
            and _manifest_body_digest(current) == _manifest_body_digest(manifest)
            for current in self._bundled
        )

    async def _start_workers(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        config: Mapping[str, Any],
        secret_bindings: Mapping[str, CapabilitySecretBinding],
        agent_binding: AgentRunBinding | None,
        replace_instance_id: str | None = None,
        publish_tools: bool = True,
    ) -> None:
        _validate_agent_binding(
            tenant_id=tenant_id,
            manifest=manifest,
            agent_binding=agent_binding,
        )
        if self._workers is None:
            raise CapabilityRuntimeUnavailableError("a capability worker host is not configured")
        if any(worker.kind is WorkerKind.MCP for worker in manifest.workers) and (
            self._tool_host is None
        ):
            raise CapabilityRuntimeUnavailableError("an MCP capability tool host is not configured")
        try:
            requirements = _secret_requirements(manifest)
            secrets: Mapping[str, str] = {}
            if secret_bindings or requirements:
                if self._secret_resolver is None:
                    raise CapabilityRuntimeUnavailableError(
                        "a capability secret resolver is not configured"
                    )
                secrets = await self._secret_resolver.resolve(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    instance_id=instance_id,
                    manifest=manifest,
                    secret_bindings=secret_bindings,
                    agent_binding=agent_binding,
                )
                missing_values = sorted(
                    name
                    for name, requirement in requirements.items()
                    if requirement.required and not str(secrets.get(name) or "")
                )
                if missing_values:
                    raise CapabilityRuntimeUnavailableError(
                        "one or more required capability secret grants are unavailable"
                    )
            worker_set = await self._workers.start(
                instance_id=instance_id,
                manifest=manifest,
                tenant_id=tenant_id,
                config=config,
                secrets=secrets,
            )
            if self._tool_host is not None and publish_tools:
                endpoint_map = {
                    handle.worker_name: handle.endpoint
                    for handle in worker_set.handles
                    if handle.endpoint is not None
                }
                endpoint_headers = {
                    handle.worker_name: dict(handle.endpoint_headers)
                    for handle in worker_set.handles
                    if handle.endpoint_headers
                }
                replace = getattr(self._tool_host, "replace", None)
                if replace_instance_id is not None and callable(replace):
                    await replace(
                        tenant_id=tenant_id,
                        previous_instance_id=replace_instance_id,
                        instance_id=instance_id,
                        manifest=manifest,
                        config=config,
                        secrets=secrets,
                        worker_endpoints=endpoint_map,
                        worker_endpoint_headers=endpoint_headers,
                    )
                else:
                    if replace_instance_id is not None:
                        await self._tool_host.stop(replace_instance_id)
                    await self._tool_host.start(
                        tenant_id=tenant_id,
                        instance_id=instance_id,
                        manifest=manifest,
                        config=config,
                        secrets=secrets,
                        worker_endpoints=endpoint_map,
                        worker_endpoint_headers=endpoint_headers,
                    )
        except Exception:
            if self._tool_host is not None and publish_tools:
                await self._tool_host.stop(instance_id)
            if self._workers is not None:
                await self._workers.stop(instance_id)
            await self._revoke_worker_credentials(
                tenant_id=tenant_id,
                instance_id=instance_id,
            )
            raise

    async def _restore_previous_generation(
        self,
        *,
        active: CapabilityActivation,
        manifest: CapabilityManifest,
        actor_id: str,
        tools_were_replaced: bool,
        replaced_instance_id: str | None = None,
    ) -> None:
        """Restore the fenced generation before surfacing activation failure."""

        previous_instance_id = _instance_id(active)
        await self._start_workers(
            tenant_id=active.namespace,
            actor_id=actor_id,
            instance_id=previous_instance_id,
            manifest=manifest,
            config=active.config,
            secret_bindings=active.secret_bindings,
            agent_binding=active.agent_binding,
            publish_tools=False,
        )
        if self._tool_host is None:
            return
        worker_set = self._workers.active(previous_instance_id) if self._workers else None
        handles = worker_set.handles if worker_set is not None else ()
        secrets: Mapping[str, str] = {}
        if _secret_requirements(manifest):
            if self._secret_resolver is None:
                raise CapabilityRuntimeUnavailableError(
                    "a capability secret resolver is not configured"
                )
            secrets = await self._secret_resolver.resolve(
                tenant_id=active.namespace,
                actor_id=actor_id,
                instance_id=previous_instance_id,
                manifest=manifest,
                secret_bindings=active.secret_bindings,
                agent_binding=active.agent_binding,
            )
        replace = getattr(self._tool_host, "replace", None)
        if callable(replace) and replaced_instance_id is not None:
            await replace(
                tenant_id=active.namespace,
                previous_instance_id=replaced_instance_id,
                instance_id=previous_instance_id,
                manifest=manifest,
                config=active.config,
                secrets=secrets,
                worker_endpoints={
                    handle.worker_name: handle.endpoint
                    for handle in handles
                    if handle.endpoint is not None
                },
                worker_endpoint_headers={
                    handle.worker_name: dict(handle.endpoint_headers)
                    for handle in handles
                    if handle.endpoint_headers
                },
            )
        else:
            if tools_were_replaced and replaced_instance_id is not None:
                await self._tool_host.stop(replaced_instance_id)
            await self._tool_host.stop(previous_instance_id)
            await self._tool_host.start(
                tenant_id=active.namespace,
                instance_id=previous_instance_id,
                manifest=manifest,
                config=active.config,
                secrets=secrets,
                worker_endpoints={
                    handle.worker_name: handle.endpoint
                    for handle in handles
                    if handle.endpoint is not None
                },
                worker_endpoint_headers={
                    handle.worker_name: dict(handle.endpoint_headers)
                    for handle in handles
                    if handle.endpoint_headers
                },
            )

    async def _revoke_worker_credentials(self, *, tenant_id: str, instance_id: str) -> None:
        if self._secret_resolver is None:
            return
        revoke = getattr(self._secret_resolver, "revoke", None)
        if revoke is None:
            return
        result = revoke(tenant_id=tenant_id, instance_id=instance_id)
        if inspect.isawaitable(result):
            await result

    async def _bind_secret_handles(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
        secret_handles: Mapping[str, str],
    ) -> dict[str, CapabilitySecretBinding]:
        if not secret_handles:
            return {}
        if self._secret_resolver is None:
            raise CapabilityRuntimeUnavailableError(
                "a capability secret resolver is not configured"
            )
        bind = getattr(self._secret_resolver, "bind", None)
        if bind is None:
            raise CapabilityRuntimeUnavailableError(
                "the capability secret resolver cannot bind handle revisions"
            )
        raw_bindings = bind(
            tenant_id=tenant_id,
            manifest=manifest,
            secret_handles=secret_handles,
        )
        if inspect.isawaitable(raw_bindings):
            raw_bindings = await raw_bindings
        bindings = dict(raw_bindings)
        if set(bindings) != set(secret_handles):
            raise CapabilityRuntimeUnavailableError(
                "the capability secret resolver returned incomplete bindings"
            )
        requirements = _secret_requirements(manifest)
        for name, binding in bindings.items():
            requirement = requirements.get(name)
            if (
                not isinstance(binding, CapabilitySecretBinding)
                or binding.handle_id != secret_handles[name]
                or requirement is None
                or requirement.source is not SecretSource.TENANT_HANDLE
                or binding.scopes != requirement.scopes
            ):
                raise CapabilityRuntimeUnavailableError(
                    "the capability secret resolver returned invalid bindings"
                )
        return bindings

    async def _bind_agent_run(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> AgentRunBinding | None:
        if _declared_agent_binding(manifest) is None:
            return None
        if self._secret_resolver is None:
            raise CapabilityRuntimeUnavailableError(
                "a capability secret resolver is not configured"
            )
        bind = getattr(self._secret_resolver, "bind_agent_run", None)
        if bind is None:
            raise CapabilityRuntimeUnavailableError(
                "the capability secret resolver cannot bind AgentSpec revisions"
            )
        raw_binding = bind(tenant_id=tenant_id, manifest=manifest)
        if inspect.isawaitable(raw_binding):
            raw_binding = await raw_binding
        if raw_binding is not None and not isinstance(raw_binding, AgentRunBinding):
            raise CapabilityRuntimeUnavailableError(
                "the capability secret resolver returned an invalid agent binding"
            )
        _validate_agent_binding(
            tenant_id=tenant_id,
            manifest=manifest,
            agent_binding=raw_binding,
        )
        return raw_binding

    async def _disable_active_locked(
        self,
        active: CapabilityActivation,
        *,
        persist_release_state: bool = True,
    ) -> None:
        transition = self._revisions.begin_deactivation(
            namespace=active.namespace,
            capability_name=active.capability_name,
            expected_generation=active.generation,
        )
        try:
            await self._finish_deactivation_locked(
                transition,
                persist_release_state=persist_release_state,
            )
        except Exception:
            self._startup_errors.add((active.namespace, active.capability_name))
            raise

    def _assert_not_blocked(self, capability_name: str) -> None:
        if str(capability_name or "").strip().lower() in self._blocked_capabilities:
            raise CapabilityRuntimeUnavailableError(
                f"capability {capability_name!r} is disabled by the host composition"
            )

    def _require_manifest(
        self,
        tenant_id: str,
        capability_name: str,
        revision: int,
    ) -> CapabilityManifest:
        manifest = self._revisions.get(
            namespace=tenant_id,
            capability_name=capability_name,
            revision=revision,
        )
        if manifest is None:
            raise CapabilityRevisionNotFoundError(
                f"capability {capability_name!r} revision {revision} does not exist"
            )
        return manifest

    def _view(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> dict[str, Any]:
        return {
            "manifest": manifest,
            "activation": self._revisions.active(
                namespace=tenant_id,
                capability_name=manifest.name,
            ),
            "test": self._revisions.test_result(
                namespace=tenant_id,
                capability_name=manifest.name,
                revision=manifest.revision,
            ),
        }


def _identity(value: str, label: str) -> str:
    safe = str(value or "").strip()
    if not safe:
        raise ValueError(f"{label} is required")
    return safe


def _manifest_body_digest(manifest: CapabilityManifest) -> str:
    return canonical_json_digest(manifest.model_dump(mode="json", exclude={"revision"}))


def _validate_release_seed_activation(
    activation: _ReleaseSeedActivation,
    manifest: CapabilityManifest,
) -> None:
    try:
        if (
            not manifest.seed
            or activation.capability_name != manifest.name
            or activation.manifest_body_digest != _manifest_body_digest(manifest)
        ):
            raise ValueError("release activation manifest binding is invalid")
        _validate_config(manifest.config_schema, activation.config)
        handles = _secret_handle_bindings(activation.secret_handles)
        if handles != activation.secret_handles or set(handles) != set(activation.secret_bindings):
            raise ValueError("release activation secret bindings are invalid")
        requirements = _secret_requirements(manifest)
        tenant_requirements = {
            name: requirement
            for name, requirement in requirements.items()
            if requirement.source is SecretSource.TENANT_HANDLE
        }
        if set(handles).difference(tenant_requirements):
            raise ValueError("release activation secret handles are invalid")
        if any(
            requirement.required and name not in handles
            for name, requirement in tenant_requirements.items()
        ):
            raise ValueError("release activation required secret handles are missing")
        for name, binding in activation.secret_bindings.items():
            requirement = tenant_requirements[name]
            if binding.handle_id != handles[name] or binding.scopes != requirement.scopes:
                raise ValueError("release activation secret binding contract changed")
        _validate_agent_binding(
            tenant_id=activation.namespace,
            manifest=manifest,
            agent_binding=activation.agent_binding,
        )
    except (CapabilityRuntimeUnavailableError, KeyError, TypeError, ValueError) as exc:
        raise CapabilityRuntimeUnavailableError(
            "the restored release capability activation is invalid"
        ) from exc


def _sanitize_check(check: CapabilityTestCheck) -> CapabilityTestCheck:
    if not isinstance(check, CapabilityTestCheck):
        raise TypeError("capability evaluator returned an invalid check")
    redacted = redact_for_langfuse(check.message)
    safe_name = redact_for_langfuse(check.name)
    return check.model_copy(
        update={
            "name": str(safe_name or "check")[:200],
            "message": str(redacted or "")[:1_000],
        }
    )


def _secret_requirements(
    manifest: CapabilityManifest,
) -> dict[str, SecretRequirement]:
    requirements: dict[str, SecretRequirement] = {}
    for requirement in (
        *manifest.secrets,
        *(item for worker in manifest.workers for item in worker.secrets),
    ):
        existing = requirements.get(requirement.name)
        if existing is None or (requirement.required and not existing.required):
            requirements[requirement.name] = requirement
    return requirements


def _declared_agent_binding(
    manifest: CapabilityManifest,
) -> AgentInterfaceBinding | None:
    bindings = {
        worker.agent_binding
        for worker in manifest.workers
        if any(
            secret.name == "OPENTULPA_AGENT_API_TOKEN" and secret.source is SecretSource.ISSUED
            for secret in worker.secrets
        )
    }
    if not bindings:
        return None
    if None in bindings or len(bindings) != 1:
        raise CapabilityRuntimeUnavailableError(
            "interface workers do not declare one consistent agent binding"
        )
    binding = next(iter(bindings))
    assert binding is not None
    return binding


def _validate_agent_binding(
    *,
    tenant_id: str,
    manifest: CapabilityManifest,
    agent_binding: AgentRunBinding | None,
) -> None:
    declared = _declared_agent_binding(manifest)
    if declared is None:
        if agent_binding is not None:
            raise CapabilityRuntimeUnavailableError(
                "capability without Agent API access has an agent binding"
            )
        return
    if agent_binding is None:
        raise CapabilityRuntimeUnavailableError(
            "interface generation is missing its durable agent binding"
        )
    if (
        agent_binding.agent_spec.tenant_id != tenant_id
        or agent_binding.agent_spec.spec_id != declared.agent_spec_id
        or agent_binding.run_kind != declared.run_kind
        or agent_binding.trust_class != declared.trust_class
    ):
        raise CapabilityRuntimeUnavailableError(
            "interface generation agent binding does not match its reviewed manifest"
        )


def _validate_activation_agent_binding(
    activation: CapabilityActivation,
    manifest: CapabilityManifest,
) -> None:
    _validate_agent_binding(
        tenant_id=activation.namespace,
        manifest=manifest,
        agent_binding=activation.agent_binding,
    )


def _secret_handle_bindings(value: Mapping[str, str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for raw_name, raw_handle in value.items():
        name = str(raw_name or "").strip()
        handle = str(raw_handle or "").strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{0,127}", name):
            raise ValueError("capability secret environment names are invalid")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", handle):
            raise ValueError("capability secret handle identifiers are invalid")
        bindings[name] = handle
    return bindings


def _next_instance_id(*, tenant_id: str, capability_name: str, generation: int) -> str:
    tenant_hash = hashlib.sha256(tenant_id.encode()).hexdigest()[:16]
    name_hash = hashlib.sha256(capability_name.encode()).hexdigest()[:8]
    return f"cap_{tenant_hash}_{capability_name[:20]}_{name_hash}_g{generation}"


def _instance_id(activation: CapabilityActivation) -> str:
    return _next_instance_id(
        tenant_id=activation.namespace,
        capability_name=activation.capability_name,
        generation=activation.generation,
    )


def _validate_config(schema: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Validate the bounded object subset used by bundled capability manifests."""

    try:
        json.dumps(config, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("capability config must contain only JSON values") from exc
    _reject_secret_config(config)
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("capability config schema properties are invalid")
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError("capability config schema required fields are invalid")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"missing capability config fields: {', '.join(sorted(missing))}")
    if schema.get("additionalProperties") is False:
        unknown = set(config).difference(properties)
        if unknown:
            raise ValueError(f"unknown capability config fields: {', '.join(sorted(unknown))}")
    for name, value in config.items():
        property_schema = properties.get(name)
        if isinstance(property_schema, dict) and not _matches_type(value, property_schema):
            raise ValueError(f"capability config field {name!r} has an invalid value")


def _matches_type(value: Any, schema: Mapping[str, Any]) -> bool:
    if "enum" in schema and value not in schema["enum"]:
        return False
    expected = schema.get("type")
    if expected == "string":
        valid = isinstance(value, str)
    elif expected == "integer":
        valid = isinstance(value, int) and not isinstance(value, bool)
    elif expected == "number":
        valid = isinstance(value, int | float) and not isinstance(value, bool)
    elif expected == "boolean":
        valid = isinstance(value, bool)
    elif expected == "array":
        valid = isinstance(value, list)
    elif expected == "object":
        valid = isinstance(value, dict)
    elif expected == "null":
        valid = value is None
    else:
        valid = True
    if not valid:
        return False
    if isinstance(value, int | float) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False
        if "maximum" in schema and value > schema["maximum"]:
            return False
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False
    return True


def _reject_secret_config(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if re.search(r"(?:api_?key|token|secret|password|credential|private_?key)", normalized):
                raise ValueError("capability secrets must be supplied through secret handles")
            _reject_secret_config(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secret_config(item)


__all__ = [
    "CapabilityControlError",
    "CapabilityControlService",
    "CapabilityEvaluationUnavailableError",
    "CapabilityEvaluator",
    "CapabilityRuntimeUnavailableError",
    "CapabilitySecretResolver",
    "CapabilityToolHost",
    "CapabilityTestRequiredError",
]
