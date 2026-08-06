"""Host boundary for starting mutable releases without giving them host authority."""

from __future__ import annotations

import asyncio
import copy
from collections import defaultdict, deque
from collections.abc import Iterable
from contextlib import suppress
from typing import Protocol
from uuid import uuid4

from opentulpa.bootstrap.models import (
    DrainResult,
    PreparedRelease,
    ReleaseHealth,
    ReleaseLaunchContext,
    ReleaseRecord,
    RunningRelease,
)


class ReleaseHostError(RuntimeError):
    pass


class ReleaseHost(Protocol):
    """Deployment-specific process/container operations owned by the bootstrap."""

    async def prepare(self, release: ReleaseRecord) -> PreparedRelease: ...

    async def start(
        self,
        prepared: PreparedRelease,
        context: ReleaseLaunchContext,
    ) -> RunningRelease: ...

    async def probe(self, running: RunningRelease) -> ReleaseHealth: ...

    async def drain(
        self,
        running: RunningRelease,
        *,
        timeout_seconds: float,
    ) -> DrainResult: ...

    async def stop(self, running: RunningRelease) -> None: ...

    async def contain(self, running: RunningRelease, *, attempts: int = 3) -> bool: ...

    async def snapshot_state(self, activation_id: str) -> None: ...

    async def restore_state(self, activation_id: str) -> bool: ...

    async def discard_state_snapshot(self, activation_id: str) -> None: ...

    async def discover(
        self,
        release_id: str,
        *,
        mode: str = "production",
    ) -> RunningRelease | None: ...

    async def collect_logs(
        self,
        running: RunningRelease,
        *,
        max_bytes: int = 64 * 1024,
    ) -> str: ...


class InMemoryReleaseHost:
    """Safe deterministic host for tests; it never executes candidate code."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        control_token: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, str | int | None]] = []
        self._running: dict[str, RunningRelease] = {}
        self._health: dict[tuple[str, str], deque[bool]] = defaultdict(deque)
        self._drain: dict[str, DrainResult] = {}
        self._prepare_failures: set[str] = set()
        self._start_failures: set[tuple[str, str]] = set()
        self._endpoint = endpoint
        self._control_token = control_token or ("t" * 32 if endpoint is not None else None)
        self.capability_state: dict[str, object] = {}
        self.product_state: dict[str, object] = {}
        self._state_snapshots: dict[str, dict[str, object]] = {}
        self._start_state_mutations: dict[tuple[str, str], dict[str, object]] = {}
        self._start_product_mutations: dict[tuple[str, str], dict[str, object]] = {}
        self._lock = asyncio.Lock()

    def health_sequence(
        self,
        release_id: str,
        *,
        mode: str,
        values: Iterable[bool],
    ) -> None:
        self._health[(release_id, mode)] = deque(bool(value) for value in values)

    def drain_result(self, release_id: str, *, drained: bool, in_flight: int = 0) -> None:
        self._drain[release_id] = DrainResult(drained=drained, in_flight=in_flight)

    def fail_prepare(self, release_id: str) -> None:
        self._prepare_failures.add(release_id)

    def fail_start(self, release_id: str, *, mode: str = "production") -> None:
        self._start_failures.add((release_id, mode))

    def mutate_state_on_start(
        self,
        release_id: str,
        *,
        values: dict[str, object],
        mode: str = "production",
    ) -> None:
        self._start_state_mutations[(release_id, mode)] = copy.deepcopy(values)

    def mutate_product_state_on_start(
        self,
        release_id: str,
        *,
        values: dict[str, object],
        mode: str = "production",
    ) -> None:
        self._start_product_mutations[(release_id, mode)] = copy.deepcopy(values)

    async def prepare(self, release: ReleaseRecord) -> PreparedRelease:
        async with self._lock:
            self.calls.append(("prepare", release.id, None))
            if release.id in self._prepare_failures:
                raise ReleaseHostError("release preparation failed")
            return PreparedRelease(
                release_id=release.id,
                artifact_digest=release.artifact_digest,
                token=f"prepared:{release.id}:{release.artifact_digest}",
            )

    async def start(
        self,
        prepared: PreparedRelease,
        context: ReleaseLaunchContext,
    ) -> RunningRelease:
        async with self._lock:
            self.calls.append(("start", prepared.release_id, context.mode))
            key = (prepared.release_id, context.mode)
            if key in self._start_failures:
                self._start_failures.remove(key)
                raise ReleaseHostError("release start failed")
            mutation = self._start_state_mutations.get(key)
            if mutation is not None:
                self.capability_state.update(copy.deepcopy(mutation))
            product_mutation = self._start_product_mutations.get(key)
            if product_mutation is not None:
                self.product_state.update(copy.deepcopy(product_mutation))
            running = RunningRelease(
                release_id=prepared.release_id,
                instance_id=f"instance_{uuid4().hex}",
                mode=context.mode,
                lease_epoch=context.lease_epoch,
                endpoint=self._endpoint,
                control_token=self._control_token,
            )
            self._running[running.instance_id] = running
            return running

    async def probe(self, running: RunningRelease) -> ReleaseHealth:
        async with self._lock:
            self.calls.append(("probe", running.release_id, running.mode))
            sequence = self._health[(running.release_id, running.mode)]
            healthy = sequence.popleft() if sequence else True
            return ReleaseHealth(
                healthy=healthy,
                release_id=running.release_id,
                summary="healthy" if healthy else "configured unhealthy test release",
                components={"runtime": healthy, "agent_api": healthy},
            )

    async def drain(
        self,
        running: RunningRelease,
        *,
        timeout_seconds: float,
    ) -> DrainResult:
        del timeout_seconds
        async with self._lock:
            self.calls.append(("drain", running.release_id, running.lease_epoch))
            return self._drain.get(running.release_id, DrainResult(drained=True))

    async def stop(self, running: RunningRelease) -> None:
        async with self._lock:
            self.calls.append(("stop", running.release_id, running.mode))
            self._running.pop(running.instance_id, None)

    async def contain(self, running: RunningRelease, *, attempts: int = 3) -> bool:
        target = running
        for _ in range(attempts):
            with suppress(Exception):
                await self.stop(target)
            surviving = await self.discover(target.release_id, mode=target.mode)
            if surviving is None:
                return True
            target = surviving
        return False

    async def collect_logs(self, running: RunningRelease, *, max_bytes: int = 64 * 1024) -> str:
        del max_bytes
        async with self._lock:
            self.calls.append(("logs", running.release_id, running.mode))
        return ""

    async def snapshot_state(self, activation_id: str) -> None:
        async with self._lock:
            self.calls.append(("snapshot", activation_id, None))
            self._state_snapshots.setdefault(
                activation_id,
                copy.deepcopy(self.capability_state),
            )

    async def restore_state(self, activation_id: str) -> bool:
        async with self._lock:
            self.calls.append(("restore", activation_id, None))
            snapshot = self._state_snapshots.get(activation_id)
            if snapshot is None:
                return False
            self.capability_state.clear()
            self.capability_state.update(copy.deepcopy(snapshot))
            return True

    async def discard_state_snapshot(self, activation_id: str) -> None:
        async with self._lock:
            self.calls.append(("discard_snapshot", activation_id, None))
            self._state_snapshots.pop(activation_id, None)

    async def discover(
        self,
        release_id: str,
        *,
        mode: str = "production",
    ) -> RunningRelease | None:
        async with self._lock:
            for running in self._running.values():
                if running.release_id == release_id and running.mode == mode:
                    return running
        return None


__all__ = ["InMemoryReleaseHost", "ReleaseHost", "ReleaseHostError"]
