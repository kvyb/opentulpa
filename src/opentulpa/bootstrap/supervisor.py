"""Immutable outer supervisor for staged release activation and rollback."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from opentulpa.bootstrap.host import ReleaseHost
from opentulpa.bootstrap.models import (
    ActivationKind,
    ActivationRecord,
    ActivationStatus,
    OutboxEvent,
    ReleaseHealth,
    ReleaseLaunchContext,
    ReleaseLease,
    ReleaseOrigin,
    ReleaseRecord,
    RunningRelease,
)
from opentulpa.bootstrap.store import BootstrapConflictError, BootstrapStore, LeaseFenceError

logger = logging.getLogger(__name__)


class ActivationError(RuntimeError):
    """Sanitized failure safe for the recovery API and owner thread."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "activation_failed")[:100]
        self.public_message = str(message or "Release activation failed.")[:2_000]


class OutboxSink(Protocol):
    async def deliver(self, event: OutboxEvent) -> None: ...


class InMemoryOutboxSink:
    def __init__(self) -> None:
        self.events: list[OutboxEvent] = []

    async def deliver(self, event: OutboxEvent) -> None:
        self.events.append(event)


@dataclass(frozen=True, slots=True)
class SupervisorPolicy:
    drain_timeout_seconds: float = 60.0
    stage_probe_attempts: int = 1
    production_probe_attempts: int = 3
    probe_interval_seconds: float = 5.0
    probation_seconds: float = 600.0
    probation_probe_interval_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 1 <= self.stage_probe_attempts <= 20:
            raise ValueError("stage_probe_attempts must be between 1 and 20")
        if not 1 <= self.production_probe_attempts <= 20:
            raise ValueError("production_probe_attempts must be between 1 and 20")
        for label, value in (
            ("drain_timeout_seconds", self.drain_timeout_seconds),
            ("probe_interval_seconds", self.probe_interval_seconds),
            ("probation_seconds", self.probation_seconds),
            ("probation_probe_interval_seconds", self.probation_probe_interval_seconds),
        ):
            if value < 0 or value > 86_400:
                raise ValueError(f"{label} must be between 0 and 86400")
        if self.probation_seconds > 0 and self.probation_probe_interval_seconds <= 0:
            raise ValueError("positive probation requires a positive probe interval")


class BootstrapSupervisor:
    """Own the one production lease while mutable releases remain replaceable."""

    def __init__(
        self,
        *,
        store: BootstrapStore,
        host: ReleaseHost,
        policy: SupervisorPolicy | None = None,
        outbox_sink: OutboxSink | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._store = store
        self._host = host
        self._policy = policy or SupervisorPolicy()
        self._outbox_sink = outbox_sink
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep
        self._operation_lock = asyncio.Lock()
        self._active_handle: RunningRelease | None = None
        self._lease_change_hook: Callable[[ReleaseLease | None], Awaitable[None]] | None = None
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def store(self) -> BootstrapStore:
        return self._store

    @property
    def host(self) -> ReleaseHost:
        return self._host

    def set_lease_change_hook(
        self,
        hook: Callable[[ReleaseLease | None], Awaitable[None]],
    ) -> None:
        if self._lease_change_hook is not None and self._lease_change_hook is not hook:
            raise RuntimeError("bootstrap lease change hook is already configured")
        self._lease_change_hook = hook

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("supervisor clock must return an aware datetime")
        return value.astimezone(UTC)

    async def start(self) -> None:
        if self._started:
            await self._notify_current_lease()
            return
        async with self._operation_lock:
            await self._reconcile()
            await self._notify_current_lease()
            self._started = True
        await self.flush_outbox()

    async def install_initial(self, release: ReleaseRecord) -> ReleaseLease:
        """Install the first trusted release and prove it healthy before ingress."""

        self._require_started()
        async with self._operation_lock:
            self._store.add_release(release)
            lease = self._store.install_initial_lease(release.id)
            running: RunningRelease | None = None
            try:
                await self._notify_lease_change(lease)
                prepared = await self._host.prepare(release)
                self._validate_prepared(release, prepared.release_id, prepared.artifact_digest)
                running = await self._host.start(
                    prepared,
                    ReleaseLaunchContext(
                        mode="production",
                        lease_epoch=lease.epoch,
                        secrets_enabled=True,
                        ingress_enabled=False,
                    ),
                )
                self._validate_running(
                    running,
                    release_id=release.id,
                    mode="production",
                    lease_epoch=lease.epoch,
                )
                await self._require_healthy(
                    running,
                    attempts=self._policy.production_probe_attempts,
                    code="initial_release_unhealthy",
                )
                self._active_handle = running
                self._store.resume_ingress()
                await self._publish(
                    event_key=f"release:{release.id}:installed",
                    event_type="release.active",
                    origin=None,
                    payload={"release_id": release.id, "initial": True},
                )
            except Exception as exc:
                if running is not None and not await self._stop_quietly(running):
                    self._active_handle = running
                    await self._enter_safe_mode()
                    raise ActivationError(
                        "initial_release_containment_failed",
                        "The initial release failed and could not be contained.",
                    ) from exc
                await self._enter_safe_mode()
                raise
        await self.flush_outbox()
        return lease

    async def request_activation(
        self,
        release: ReleaseRecord,
        *,
        origin: ReleaseOrigin | None,
        reason: str = "Owner approved",
        kind: ActivationKind = ActivationKind.DEPLOY,
        activation_id: str | None = None,
    ) -> ActivationRecord:
        self._require_started()
        self._store.add_release(release)
        state = self._store.get_state()
        if state.serving_release_id is None or state.last_known_good_release_id is None:
            raise ActivationError("no_active_release", "No active release is available for rollback.")
        if release.id == state.serving_release_id:
            raise ActivationError("already_active", "The requested release is already serving.")
        activation_values = {
            "kind": kind,
            "target_release_id": release.id,
            "previous_release_id": state.serving_release_id,
            "origin": origin,
            "reason": reason,
        }
        if activation_id is not None:
            activation_values["id"] = activation_id
        activation = ActivationRecord.model_validate(activation_values)
        self._store.create_activation(activation)
        await self._publish_activation(activation, "activation.queued")
        await self.flush_outbox()
        return activation

    async def request_rollback(
        self,
        *,
        origin: ReleaseOrigin | None,
        reason: str = "Owner requested rollback",
        activation_id: str | None = None,
    ) -> ActivationRecord:
        self._require_started()
        state = self._store.get_state()
        target_id = state.previous_release_id
        if target_id is None or state.serving_release_id is None:
            raise ActivationError("no_rollback_target", "No previous release is available.")
        target = self._store.get_release(target_id)
        if target is None:
            raise ActivationError("rollback_target_missing", "The previous release is unavailable.")
        return await self.request_activation(
            target,
            origin=origin,
            reason=reason,
            kind=ActivationKind.ROLLBACK,
            activation_id=activation_id,
        )

    async def activate(self, activation_id: str) -> ActivationRecord:
        """Run one persisted activation through staging, cutover, and probation."""

        self._require_started()
        async with self._operation_lock:
            activation = self._required_activation(activation_id)
            if activation.status is not ActivationStatus.QUEUED:
                if activation.status in {
                    ActivationStatus.ACTIVE,
                    ActivationStatus.FAILED,
                    ActivationStatus.ROLLED_BACK,
                    ActivationStatus.CANCELLED,
                }:
                    return activation
                raise BootstrapConflictError("activation is already in progress")
            terminal = await self._run_activation(activation)
        await self.flush_outbox()
        return terminal

    async def cancel(self, activation_id: str) -> ActivationRecord:
        self._require_started()
        async with self._operation_lock:
            activation = self._required_activation(activation_id)
            if activation.status is ActivationStatus.CANCELLED:
                return activation
            if activation.status is not ActivationStatus.QUEUED:
                raise ActivationError(
                    "activation_not_cancellable",
                    "Only a queued activation can be cancelled safely.",
                )
            cancelled = self._store.transition_activation(
                activation.id,
                expected=ActivationStatus.QUEUED,
                target=ActivationStatus.CANCELLED,
            )
            await self._publish_activation(cancelled, "activation.cancelled")
        await self.flush_outbox()
        return cancelled

    async def enter_safe_mode(self) -> None:
        self._require_started()
        async with self._operation_lock:
            if self._active_handle is not None and await self._stop_quietly(
                self._active_handle
            ):
                self._active_handle = None
            await self._enter_safe_mode()
            await self._publish(
                event_key=f"bootstrap:safe-mode:{self._now().isoformat()}",
                event_type="bootstrap.safe_mode",
                origin=None,
                payload={"status": "safe_mode"},
            )
        await self.flush_outbox()

    async def recover_last_known_good(self) -> ReleaseLease:
        """Start the retained last-known-good release from recovery safe mode."""

        self._require_started()
        async with self._operation_lock:
            state = self._store.get_state()
            release_id = state.last_known_good_release_id
            if release_id is None:
                raise ActivationError(
                    "no_last_known_good",
                    "No last-known-good release is available.",
                )
            release = self._store.get_release(release_id)
            if release is None:
                raise ActivationError(
                    "last_known_good_missing",
                    "The last-known-good release artifact is unavailable.",
                )
            if self._active_handle is not None:
                if not await self._stop_quietly(self._active_handle):
                    await self._enter_safe_mode()
                    raise ActivationError(
                        "recovery_containment_failed",
                        "The existing release could not be contained; safe mode remains active.",
                    )
                self._active_handle = None
            recovery_release_ids = {
                candidate_id
                for candidate_id in (
                    state.serving_release_id,
                    state.last_known_good_release_id,
                    state.previous_release_id,
                    *(
                        activation.target_release_id
                        for activation in self._store.list_activations(limit=1_000)
                    ),
                )
                if candidate_id is not None
            }
            for candidate_id in recovery_release_ids:
                for mode in ("staging", "production"):
                    running_candidate = await self._host.discover(candidate_id, mode=mode)
                    if running_candidate is not None and not await self._stop_quietly(
                        running_candidate
                    ):
                        await self._enter_safe_mode()
                        raise ActivationError(
                            "recovery_containment_failed",
                            "An existing release could not be contained; safe mode remains active.",
                        )
            running: RunningRelease | None = None
            lease = self._store.begin_restore_lease(
                release_id=release.id,
                activation_id=None,
            )
            try:
                await self._notify_lease_change(lease)
                prepared = await self._host.prepare(release)
                self._validate_prepared(release, prepared.release_id, prepared.artifact_digest)
                running = await self._host.start(
                    prepared,
                    ReleaseLaunchContext(
                        mode="production",
                        lease_epoch=lease.epoch,
                        secrets_enabled=True,
                        ingress_enabled=False,
                    ),
                )
                self._validate_running(
                    running,
                    release_id=release.id,
                    mode="production",
                    lease_epoch=lease.epoch,
                )
                await self._require_healthy(
                    running,
                    attempts=self._policy.production_probe_attempts,
                    code="recovery_unhealthy",
                )
                self._active_handle = running
                self._store.complete_recovery(
                    release_id=release.id,
                    lease=lease,
                    previous_release_id=state.previous_release_id,
                )
                await self._publish(
                    event_key=f"release:{release.id}:recovered:{lease.epoch}",
                    event_type="release.recovered",
                    origin=None,
                    payload={"release_id": release.id, "lease_epoch": lease.epoch},
                )
            except Exception as exc:
                contained = running is None or await self._stop_quietly(running)
                self._active_handle = None if contained else running
                await self._enter_safe_mode()
                error = self._public_error(exc)
                if not contained:
                    error = ActivationError(
                        "recovery_containment_failed",
                        "The failed recovery process could not be contained.",
                    )
                raise ActivationError(error.code, error.public_message) from exc
        await self.flush_outbox()
        return lease

    async def flush_outbox(self, *, limit: int = 100) -> int:
        sink = self._outbox_sink
        if sink is None:
            return 0
        delivered = 0
        for event in self._store.pending_outbox(limit=limit):
            try:
                await sink.deliver(event)
            except Exception:
                logger.exception("bootstrap outbox delivery failed: event=%s", event.id)
                self._store.mark_outbox_attempt(event.id, delivered=False)
                continue
            self._store.mark_outbox_attempt(event.id, delivered=True)
            delivered += 1
        return delivered

    async def _run_activation(self, activation: ActivationRecord) -> ActivationRecord:
        release = self._store.get_release(activation.target_release_id)
        if release is None:
            return await self._fail_before_cutover(
                activation,
                code="release_missing",
                message="The candidate release artifact is unavailable.",
            )

        current = self._store.transition_activation(
            activation.id,
            expected=ActivationStatus.QUEUED,
            target=ActivationStatus.PREPARING,
        )
        await self._publish_activation(current, "activation.started")
        prepared = None
        staging: RunningRelease | None = None
        cutover_started = False
        state_snapshot_created = False
        production: RunningRelease | None = None
        lease: ReleaseLease | None = None
        try:
            prepared = await self._host.prepare(release)
            self._validate_prepared(release, prepared.release_id, prepared.artifact_digest)
            staging = await self._host.start(
                prepared,
                ReleaseLaunchContext(mode="staging"),
            )
            self._validate_running(
                staging,
                release_id=release.id,
                mode="staging",
                lease_epoch=None,
            )
            await self._require_healthy(
                staging,
                attempts=self._policy.stage_probe_attempts,
                code="staging_unhealthy",
            )
            if not await self._stop_quietly(staging):
                raise ActivationError(
                    "staging_containment_failed",
                    "The staging release could not be contained.",
                )
            staging = None
            current = self._store.transition_activation(
                current.id,
                expected=ActivationStatus.PREPARING,
                target=ActivationStatus.STAGED,
            )
            await self._publish_activation(current, "activation.staged")

            current = self._store.transition_activation(
                current.id,
                expected=ActivationStatus.STAGED,
                target=ActivationStatus.DRAINING,
            )
            self._store.pause_ingress()
            old = await self._active_production_handle(current.previous_release_id)
            if old is None:
                raise ActivationError(
                    "active_release_missing",
                    "The current release process could not be found; cutover was not attempted.",
                )
            drain = await self._host.drain(
                old,
                timeout_seconds=self._policy.drain_timeout_seconds,
            )
            if not drain.drained:
                raise ActivationError(
                    "drain_timeout",
                    f"The current release still has {drain.in_flight} in-flight operation(s).",
                )

            current = self._store.transition_activation(
                current.id,
                expected=ActivationStatus.DRAINING,
                target=ActivationStatus.STARTING,
            )
            if not await self._stop_quietly(old):
                raise ActivationError(
                    "active_containment_failed",
                    "The current release could not be contained; cutover was not attempted.",
                )
            self._active_handle = None
            cutover_started = True
            await self._host.snapshot_state(current.id)
            state_snapshot_created = True
            lease = self._store.begin_cutover(current)
            await self._notify_lease_change(lease)
            production = await self._host.start(
                prepared,
                ReleaseLaunchContext(
                    mode="production",
                    lease_epoch=lease.epoch,
                    secrets_enabled=True,
                    ingress_enabled=False,
                ),
            )
            self._validate_running(
                production,
                release_id=release.id,
                mode="production",
                lease_epoch=lease.epoch,
            )
            self._active_handle = production
            current = self._store.transition_activation(
                current.id,
                expected=ActivationStatus.STARTING,
                target=ActivationStatus.VERIFYING,
                lease_epoch=lease.epoch,
            )
            await self._require_healthy(
                production,
                attempts=self._policy.production_probe_attempts,
                code="production_unhealthy",
            )

            probation_ends_at = self._now() + timedelta(seconds=self._policy.probation_seconds)
            current = self._store.commit_probation(
                current.id,
                lease=lease,
                probation_ends_at=probation_ends_at,
            )
            await self._publish_activation(current, "activation.probation")
            await self._monitor_probation(production)
            current = self._store.commit_activation_success(current.id, lease=lease)
            await self._publish_activation(current, "release.active")
            await self._discard_state_snapshot_quietly(current.id)
            return current
        except Exception as exc:
            error = self._public_error(exc)
            await self._collect_runtime_diagnostics(production or staging)
            if staging is not None and not await self._stop_quietly(staging):
                await self._enter_safe_mode()
                return await self._fail_after_cutover(
                    current,
                    code="staging_containment_failed",
                    message="The staging release could not be contained; safe mode is active.",
                )
            if not cutover_started:
                return await self._fail_before_cutover(
                    current,
                    code=error.code,
                    message=error.public_message,
                )
            return await self._automatic_rollback(
                current,
                production=production,
                state_snapshot_created=state_snapshot_created,
                code=error.code,
                message=error.public_message,
            )

    async def _automatic_rollback(
        self,
        activation: ActivationRecord,
        *,
        production: RunningRelease | None,
        state_snapshot_created: bool,
        code: str,
        message: str,
    ) -> ActivationRecord:
        if production is not None and not await self._stop_quietly(production):
            self._active_handle = production
            await self._enter_safe_mode()
            return await self._fail_after_cutover(
                activation,
                code="candidate_containment_failed",
                message=(
                    "The candidate could not be contained, so the previous release "
                    "was not started. Safe mode is active."
                ),
            )
        self._active_handle = None
        previous_id = activation.previous_release_id
        if previous_id is None:
            await self._enter_safe_mode()
            return await self._fail_after_cutover(
                activation,
                code=code,
                message=f"{message} No previous release was available; safe mode is active.",
            )
        previous = self._store.get_release(previous_id)
        if previous is None:
            await self._enter_safe_mode()
            return await self._fail_after_cutover(
                activation,
                code=code,
                message=f"{message} The previous release was unavailable; safe mode is active.",
            )

        rolling = self._store.transition_activation(
            activation.id,
            expected=activation.status,
            target=ActivationStatus.ROLLING_BACK,
        )
        await self._publish_activation(rolling, "rollback.started")
        restored: RunningRelease | None = None
        try:
            if state_snapshot_created and not await self._host.restore_state(rolling.id):
                raise ActivationError(
                    "rollback_state_missing",
                    "The pre-candidate capability state snapshot was unavailable.",
                )
            restore_lease = self._store.begin_restore_lease(
                release_id=previous.id,
                activation_id=rolling.id,
            )
            await self._notify_lease_change(restore_lease)
            prepared = await self._host.prepare(previous)
            self._validate_prepared(previous, prepared.release_id, prepared.artifact_digest)
            restored = await self._host.start(
                prepared,
                ReleaseLaunchContext(
                    mode="production",
                    lease_epoch=restore_lease.epoch,
                    secrets_enabled=True,
                    ingress_enabled=False,
                ),
            )
            self._validate_running(
                restored,
                release_id=previous.id,
                mode="production",
                lease_epoch=restore_lease.epoch,
            )
            await self._require_healthy(
                restored,
                attempts=self._policy.production_probe_attempts,
                code="rollback_unhealthy",
            )
            self._active_handle = restored
            completed = self._store.commit_rollback(
                rolling.id,
                restored_release_id=previous.id,
                failed_release_id=activation.target_release_id,
                lease=restore_lease,
                failure_code=code,
                failure_message=message,
            )
            await self._publish_activation(completed, "activation.failed")
            await self._publish_activation(completed, "rollback.completed")
            await self._discard_state_snapshot_quietly(completed.id)
            return completed
        except Exception:
            logger.exception("automatic release rollback failed: activation=%s", activation.id)
            await self._collect_runtime_diagnostics(restored)
            contained = restored is None or await self._stop_quietly(restored)
            self._active_handle = None if contained else restored
            await self._enter_safe_mode()
            return await self._fail_after_cutover(
                rolling,
                code="rollback_failed" if contained else "rollback_containment_failed",
                message=(
                    "The candidate failed and the previous release could not be restored. "
                    "Safe mode is active."
                    if contained
                    else "The failed rollback process could not be contained. Safe mode is active."
                ),
            )

    async def _fail_before_cutover(
        self,
        activation: ActivationRecord,
        *,
        code: str,
        message: str,
    ) -> ActivationRecord:
        state = self._store.get_state()
        if state.serving_release_id is not None and state.active_lease_epoch is not None:
            with suppress(BootstrapConflictError):
                self._store.resume_ingress()
        failed = self._store.transition_activation(
            activation.id,
            expected=activation.status,
            target=ActivationStatus.FAILED,
            failure_code=code,
            failure_message=message,
        )
        await self._publish_activation(failed, "activation.failed")
        return failed

    async def _fail_after_cutover(
        self,
        activation: ActivationRecord,
        *,
        code: str,
        message: str,
    ) -> ActivationRecord:
        failed = self._store.transition_activation(
            activation.id,
            expected=activation.status,
            target=ActivationStatus.FAILED,
            failure_code=code,
            failure_message=message,
        )
        await self._publish_activation(failed, "activation.failed")
        return failed

    async def _monitor_probation(self, running: RunningRelease) -> None:
        seconds = self._policy.probation_seconds
        if seconds <= 0:
            await self._require_healthy(running, attempts=1, code="probation_unhealthy")
            return
        interval = self._policy.probation_probe_interval_seconds
        attempts = max(1, math.ceil(seconds / interval))
        elapsed = 0.0
        for _ in range(attempts):
            wait = min(interval, max(0.0, seconds - elapsed))
            if wait:
                await self._sleep(wait)
                elapsed += wait
            await self._require_healthy(running, attempts=1, code="probation_unhealthy")

    async def _require_healthy(
        self,
        running: RunningRelease,
        *,
        attempts: int,
        code: str,
    ) -> None:
        for attempt in range(attempts):
            report = await self._host.probe(running)
            self._validate_health(running, report, code=code)
            if attempt + 1 < attempts and self._policy.probe_interval_seconds:
                await self._sleep(self._policy.probe_interval_seconds)

    @staticmethod
    def _validate_health(
        running: RunningRelease,
        report: ReleaseHealth,
        *,
        code: str,
    ) -> None:
        if report.release_id != running.release_id or report.protocol_version != 1:
            raise ActivationError(
                "control_protocol_mismatch",
                "The release did not identify itself with the required control protocol.",
            )
        required_components = {"runtime", "agent_api"}
        if not required_components.issubset(report.components):
            raise ActivationError(
                "health_contract_incomplete",
                "The release omitted required runtime or Agent API health checks.",
            )
        if not report.healthy or any(not value for value in report.components.values()):
            raise ActivationError(code, "The release failed its required health checks.")

    @staticmethod
    def _validate_prepared(
        release: ReleaseRecord,
        release_id: str,
        artifact_digest: str,
    ) -> None:
        if release_id != release.id or artifact_digest != release.artifact_digest:
            raise ActivationError(
                "prepared_artifact_mismatch",
                "The prepared artifact does not match the approved release.",
            )

    @staticmethod
    def _validate_running(
        running: RunningRelease,
        *,
        release_id: str,
        mode: str,
        lease_epoch: int | None,
    ) -> None:
        if (
            running.release_id != release_id
            or running.mode != mode
            or running.lease_epoch != lease_epoch
        ):
            raise ActivationError(
                "running_release_mismatch",
                "The host started a process that does not match the approved release context.",
            )

    async def _active_production_handle(
        self,
        release_id: str | None,
    ) -> RunningRelease | None:
        if release_id is None:
            return None
        if self._active_handle is not None and self._active_handle.release_id == release_id:
            return self._active_handle
        discovered = await self._host.discover(release_id, mode="production")
        if discovered is not None:
            self._active_handle = discovered
        return discovered

    async def _reconcile(self) -> None:
        """Fail closed after a bootstrap crash and restore last-known-good if needed."""

        state = self._store.get_state()
        incomplete = self._store.incomplete_activations()
        candidate_production_found: set[str] = set()
        for activation in incomplete:
            try:
                staging = await self._host.discover(
                    activation.target_release_id,
                    mode="staging",
                )
                if staging is not None and not await self._stop_quietly(staging):
                    raise RuntimeError("staging release containment failed")
            except Exception:
                logger.exception(
                    "bootstrap could not clean an orphaned staging release: activation=%s",
                    activation.id,
                )
                await self._enter_safe_mode()
                await self._publish(
                    event_key=f"bootstrap:reconcile:staging-stop-failed:{activation.id}",
                    event_type="bootstrap.safe_mode",
                    origin=None,
                    payload={"failure_code": "staging_stop_failed"},
                )
                return
            if activation.status in {
                ActivationStatus.STARTING,
                ActivationStatus.VERIFYING,
                ActivationStatus.PROBATION,
                ActivationStatus.ROLLING_BACK,
            }:
                try:
                    production = await self._host.discover(
                        activation.target_release_id,
                        mode="production",
                    )
                    if production is not None:
                        candidate_production_found.add(activation.id)
                        if not await self._stop_quietly(production):
                            raise RuntimeError("candidate release containment failed")
                except Exception:
                    logger.exception(
                        "bootstrap could not stop an incomplete candidate release: activation=%s",
                        activation.id,
                    )
                    await self._enter_safe_mode()
                    await self._publish(
                        event_key=f"bootstrap:reconcile:candidate-stop-failed:{activation.id}",
                        event_type="bootstrap.safe_mode",
                        origin=None,
                        payload={"failure_code": "candidate_stop_failed"},
                    )
                    return
        if state.safe_mode:
            release_ids = {
                release_id
                for release_id in (
                    state.serving_release_id,
                    state.last_known_good_release_id,
                    state.previous_release_id,
                    *(activation.target_release_id for activation in incomplete),
                    *(
                        activation.target_release_id
                        for activation in self._store.list_activations(limit=1_000)
                    ),
                )
                if release_id is not None
            }
            for release_id in release_ids:
                for mode in ("staging", "production"):
                    try:
                        running = await self._host.discover(release_id, mode=mode)
                        if running is not None and not await self._stop_quietly(running):
                            raise RuntimeError("safe-mode release containment failed")
                    except Exception:
                        logger.exception(
                            "bootstrap could not clean a release while entering safe mode: "
                            "release=%s mode=%s",
                            release_id,
                            mode,
                        )
                        await self._publish(
                            event_key=(
                                f"bootstrap:reconcile:safe-mode-stop-failed:{release_id}:{mode}"
                            ),
                            event_type="bootstrap.safe_mode",
                            origin=None,
                            payload={"failure_code": "safe_mode_stop_failed"},
                        )
                        return
        cutover_incomplete = any(
            activation.status
            in {
                ActivationStatus.STARTING,
                ActivationStatus.VERIFYING,
                ActivationStatus.PROBATION,
                ActivationStatus.ROLLING_BACK,
            }
            for activation in incomplete
        )
        needs_restore = (
            not state.safe_mode
            and state.last_known_good_release_id is not None
            and (
                state.serving_release_id != state.last_known_good_release_id
                or cutover_incomplete
            )
        )
        if needs_restore:
            last_good_id = state.last_known_good_release_id
            assert last_good_id is not None
            last_good = self._store.get_release(last_good_id)
            if last_good is None:
                await self._enter_safe_mode()
                await self._publish(
                    event_key="bootstrap:reconcile:last-known-good-missing",
                    event_type="bootstrap.safe_mode",
                    origin=None,
                    payload={"failure_code": "last_known_good_missing"},
                )
                return
            possible_running = {
                release_id
                for release_id in (state.serving_release_id, last_good.id)
                if release_id is not None
            }
            try:
                for release_id in possible_running:
                    running = await self._host.discover(release_id, mode="production")
                    if running is not None and not await self._stop_quietly(running):
                        raise RuntimeError("production release containment failed")
            except Exception:
                logger.exception("bootstrap could not contain production releases before recovery")
                await self._enter_safe_mode()
                await self._publish(
                    event_key="bootstrap:reconcile:production-stop-failed",
                    event_type="bootstrap.safe_mode",
                    origin=None,
                    payload={"failure_code": "production_stop_failed"},
                )
                return
            restore_candidates = tuple(
                activation
                for activation in incomplete
                if activation.status
                in {
                    ActivationStatus.STARTING,
                    ActivationStatus.VERIFYING,
                    ActivationStatus.PROBATION,
                    ActivationStatus.ROLLING_BACK,
                }
                and (
                    activation.status is not ActivationStatus.STARTING
                    or state.active_activation_id == activation.id
                    or activation.id in candidate_production_found
                )
            )
            if len(restore_candidates) > 1:
                await self._enter_safe_mode()
                await self._publish(
                    event_key="bootstrap:reconcile:ambiguous-state-snapshot",
                    event_type="bootstrap.safe_mode",
                    origin=None,
                    payload={"failure_code": "state_snapshot_ambiguous"},
                )
                return
            if restore_candidates:
                restore_activation = restore_candidates[0]
                try:
                    snapshot_restored = await self._host.restore_state(restore_activation.id)
                except Exception:
                    logger.exception(
                        "bootstrap could not restore capability state snapshot: activation=%s",
                        restore_activation.id,
                    )
                    snapshot_restored = False
                if not snapshot_restored:
                    await self._enter_safe_mode()
                    await self._publish(
                        event_key=(
                            "bootstrap:reconcile:state-snapshot-missing:"
                            f"{restore_activation.id}"
                        ),
                        event_type="bootstrap.safe_mode",
                        origin=None,
                        payload={"failure_code": "state_snapshot_missing"},
                    )
                    return
            restore_lease = self._store.begin_restore_lease(
                release_id=last_good.id,
                activation_id=None,
            )
            await self._notify_lease_change(restore_lease)
            restored: RunningRelease | None = None
            try:
                prepared = await self._host.prepare(last_good)
                self._validate_prepared(
                    last_good,
                    prepared.release_id,
                    prepared.artifact_digest,
                )
                restored = await self._host.start(
                    prepared,
                    ReleaseLaunchContext(
                        mode="production",
                        lease_epoch=restore_lease.epoch,
                        secrets_enabled=True,
                        ingress_enabled=False,
                    ),
                )
                self._validate_running(
                    restored,
                    release_id=last_good.id,
                    mode="production",
                    lease_epoch=restore_lease.epoch,
                )
                await self._require_healthy(restored, attempts=1, code="recovery_unhealthy")
                self._active_handle = restored
                self._store.complete_recovery(
                    release_id=last_good.id,
                    lease=restore_lease,
                    previous_release_id=state.serving_release_id,
                )
            except Exception:
                logger.exception("bootstrap could not restore last-known-good release")
                contained = restored is None or await self._stop_quietly(restored)
                self._active_handle = None if contained else restored
                await self._enter_safe_mode()
                await self._publish(
                    event_key="bootstrap:reconcile:restore-failed",
                    event_type="bootstrap.safe_mode",
                    origin=None,
                    payload={
                        "failure_code": (
                            "recovery_failed" if contained else "recovery_containment_failed"
                        )
                    },
                )
                return
        elif state.serving_release_id is not None:
            running = await self._host.discover(state.serving_release_id, mode="production")
            if running is not None:
                self._active_handle = running
            else:
                serving = self._store.get_release(state.serving_release_id)
                if serving is None:
                    await self._enter_safe_mode()
                    await self._publish(
                        event_key="bootstrap:reconcile:serving-release-missing",
                        event_type="bootstrap.safe_mode",
                        origin=None,
                        payload={"failure_code": "serving_release_missing"},
                    )
                    return
                recovery_lease = self._store.begin_restore_lease(
                    release_id=serving.id,
                    activation_id=None,
                )
                await self._notify_lease_change(recovery_lease)
                running = None
                try:
                    prepared = await self._host.prepare(serving)
                    self._validate_prepared(
                        serving,
                        prepared.release_id,
                        prepared.artifact_digest,
                    )
                    running = await self._host.start(
                        prepared,
                        ReleaseLaunchContext(
                            mode="production",
                            lease_epoch=recovery_lease.epoch,
                            secrets_enabled=True,
                            ingress_enabled=False,
                        ),
                    )
                    self._validate_running(
                        running,
                        release_id=serving.id,
                        mode="production",
                        lease_epoch=recovery_lease.epoch,
                    )
                    await self._require_healthy(
                        running,
                        attempts=1,
                        code="recovery_unhealthy",
                    )
                    self._active_handle = running
                    self._store.complete_recovery(
                        release_id=serving.id,
                        lease=recovery_lease,
                        previous_release_id=state.previous_release_id,
                    )
                except Exception:
                    logger.exception("bootstrap could not restart the serving release")
                    contained = running is None or await self._stop_quietly(running)
                    self._active_handle = None if contained else running
                    await self._enter_safe_mode()
                    await self._publish(
                        event_key="bootstrap:reconcile:restart-failed",
                        event_type="bootstrap.safe_mode",
                        origin=None,
                        payload={
                            "failure_code": (
                                "recovery_failed" if contained else "recovery_containment_failed"
                            )
                        },
                    )
                    return

        for activation in incomplete:
            current = self._required_activation(activation.id)
            if current.status in {
                ActivationStatus.ROLLING_BACK,
                ActivationStatus.PROBATION,
                ActivationStatus.VERIFYING,
                ActivationStatus.STARTING,
            } and self._store.get_state().last_known_good_release_id == current.previous_release_id:
                if current.status is not ActivationStatus.ROLLING_BACK:
                    current = self._store.transition_activation(
                        current.id,
                        expected=current.status,
                        target=ActivationStatus.ROLLING_BACK,
                    )
                restored_state = self._store.get_state()
                restored_id = restored_state.last_known_good_release_id
                epoch = restored_state.active_lease_epoch
                if restored_id is not None and epoch is not None:
                    lease = self._store.assert_active_lease(restored_id, epoch)
                    completed = self._store.commit_rollback(
                        current.id,
                        restored_release_id=restored_id,
                        failed_release_id=current.target_release_id,
                        lease=lease,
                        failure_code="bootstrap_restarted",
                        failure_message="The bootstrap restarted during activation and restored the previous release.",
                    )
                    await self._publish_activation(completed, "activation.failed")
                    await self._publish_activation(completed, "rollback.completed")
                    await self._discard_state_snapshot_quietly(completed.id)
                    continue
            if current.status not in {
                ActivationStatus.FAILED,
                ActivationStatus.ROLLED_BACK,
                ActivationStatus.ACTIVE,
                ActivationStatus.CANCELLED,
            }:
                failed = self._store.transition_activation(
                    current.id,
                    expected=current.status,
                    target=ActivationStatus.FAILED,
                    failure_code="bootstrap_restarted",
                    failure_message="The bootstrap restarted before activation completed; the serving release was retained.",
                )
                await self._publish_activation(failed, "activation.failed")
                await self._discard_state_snapshot_quietly(failed.id)
        reconciled = self._store.get_state()
        if (
            not reconciled.safe_mode
            and reconciled.ingress_paused
            and reconciled.serving_release_id is not None
            and reconciled.active_lease_epoch is not None
        ):
            self._store.resume_ingress()

    async def _notify_current_lease(self) -> None:
        state = self._store.get_state()
        lease: ReleaseLease | None = None
        if (
            not state.safe_mode
            and state.serving_release_id is not None
            and state.active_lease_epoch is not None
        ):
            with suppress(LeaseFenceError):
                lease = self._store.assert_active_lease(
                    state.serving_release_id,
                    state.active_lease_epoch,
                )
        await self._notify_lease_change(lease)

    async def _notify_lease_change(self, lease: ReleaseLease | None) -> None:
        hook = self._lease_change_hook
        if hook is not None:
            await hook(lease)

    async def _enter_safe_mode(self) -> None:
        self._store.enter_safe_mode()
        await self._notify_lease_change(None)

    async def _publish_activation(self, activation: ActivationRecord, event_type: str) -> None:
        await self._publish(
            event_key=f"activation:{activation.id}:{event_type}",
            event_type=event_type,
            origin=activation.origin,
            payload={
                "activation_id": activation.id,
                "kind": activation.kind.value,
                "status": activation.status.value,
                "target_release_id": activation.target_release_id,
                "previous_release_id": activation.previous_release_id,
                "failure_code": activation.failure_code,
                "failure_message": activation.failure_message,
            },
        )

    async def _publish(
        self,
        *,
        event_key: str,
        event_type: str,
        origin: ReleaseOrigin | None,
        payload: dict[str, object],
    ) -> None:
        self._store.append_outbox(
            OutboxEvent.model_validate(
                {
                    "event_key": event_key,
                    "event_type": event_type,
                    "origin": origin,
                    "payload": payload,
                }
            )
        )

    async def _stop_quietly(self, running: RunningRelease) -> bool:
        try:
            contained = await self._host.contain(running)
        except Exception:
            logger.exception("release containment could not be verified: release=%s", running.release_id)
            return False
        if not contained:
            logger.error("release remained discoverable after containment: release=%s", running.release_id)
        return contained

    async def _collect_runtime_diagnostics(self, running: RunningRelease | None) -> None:
        if running is None:
            return
        try:
            logs = await self._host.collect_logs(running)
        except Exception:
            logger.exception("release diagnostics unavailable: release=%s", running.release_id)
            return
        if logs:
            logger.error("release diagnostics: release=%s\n%s", running.release_id, logs)

    async def _discard_state_snapshot_quietly(self, activation_id: str) -> None:
        try:
            await self._host.discard_state_snapshot(activation_id)
        except Exception:
            logger.exception(
                "release capability state snapshot cleanup failed: activation=%s",
                activation_id,
            )

    def _required_activation(self, activation_id: str) -> ActivationRecord:
        activation = self._store.get_activation(activation_id)
        if activation is None:
            raise ActivationError("activation_missing", "The activation was not found.")
        return activation

    @staticmethod
    def _public_error(exc: Exception) -> ActivationError:
        if isinstance(exc, ActivationError):
            return exc
        if isinstance(exc, BootstrapConflictError):
            return ActivationError("activation_conflict", "Activation state changed unexpectedly.")
        logger.exception("release activation failed", exc_info=exc)
        return ActivationError("activation_failed", "The release could not be activated.")

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("bootstrap supervisor has not been started")


__all__ = [
    "ActivationError",
    "BootstrapSupervisor",
    "InMemoryOutboxSink",
    "OutboxSink",
    "SupervisorPolicy",
]
