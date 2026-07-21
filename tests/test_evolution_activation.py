from __future__ import annotations

from pathlib import Path

import pytest

from opentulpa.bootstrap.host import InMemoryReleaseHost
from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.bootstrap.store import BootstrapStore
from opentulpa.bootstrap.supervisor import BootstrapSupervisor, SupervisorPolicy
from opentulpa.evolution.activation import (
    BootstrapReleaseActivator,
    ReleaseActivationStatus,
)


def _release(name: str, character: str) -> ReleaseRecord:
    return ReleaseRecord(
        id=f"release_{name}",
        candidate_id=f"candidate_{name}",
        source_commit=character * 40,
        artifact_digest=f"sha256:{character * 64}",
        manifest_digest=f"sha256:{character * 64}",
        entrypoint=("python", "-m", "opentulpa"),
    )


def _origin() -> ReleaseOrigin:
    return ReleaseOrigin(
        tenant_id="tenant_1",
        actor_id="owner_1",
        thread_id="thread_1",
        channel="web",
        correlation_id="correlation_1",
    )


def _supervisor(tmp_path: Path, host: InMemoryReleaseHost) -> BootstrapSupervisor:
    return BootstrapSupervisor(
        store=BootstrapStore(tmp_path / "bootstrap.db"),
        host=host,
        policy=SupervisorPolicy(
            drain_timeout_seconds=1,
            stage_probe_attempts=1,
            production_probe_attempts=1,
            probe_interval_seconds=0,
            probation_seconds=0,
            probation_probe_interval_seconds=1,
        ),
    )


@pytest.mark.asyncio
async def test_bootstrap_activator_waits_for_terminal_active_and_is_idempotent(
    tmp_path: Path,
) -> None:
    host = InMemoryReleaseHost()
    supervisor = _supervisor(tmp_path, host)
    await supervisor.start()
    await supervisor.install_initial(_release("blue", "a"))
    green = _release("green", "b")
    activator = BootstrapReleaseActivator(supervisor)

    first = await activator.activate(
        green,
        activation_id="promotion_green",
        origin=_origin(),
        reason="Owner approved",
        rollback=False,
    )
    repeated = await activator.activate(
        green,
        activation_id="promotion_green",
        origin=_origin(),
        reason="Owner approved",
        rollback=False,
    )

    assert first.status is ReleaseActivationStatus.ACTIVE
    assert repeated == first
    assert supervisor.store.get_state().serving_release_id == green.id
    assert [item.id for item in supervisor.store.list_activations()] == ["promotion_green"]


@pytest.mark.asyncio
async def test_bootstrap_activator_returns_persisted_automatic_rollback_failure(
    tmp_path: Path,
) -> None:
    host = InMemoryReleaseHost()
    supervisor = _supervisor(tmp_path, host)
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await supervisor.install_initial(blue)
    host.health_sequence(green.id, mode="production", values=(True, False))
    activator = BootstrapReleaseActivator(supervisor)

    result = await activator.activate(
        green,
        activation_id="promotion_green",
        origin=_origin(),
        reason="Owner approved",
        rollback=False,
    )

    assert result.status is ReleaseActivationStatus.ROLLED_BACK
    assert result.failure_code == "probation_unhealthy"
    assert supervisor.store.get_state().serving_release_id == blue.id


@pytest.mark.asyncio
async def test_bootstrap_activator_uses_verified_bootstrap_rollback(tmp_path: Path) -> None:
    host = InMemoryReleaseHost()
    supervisor = _supervisor(tmp_path, host)
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await supervisor.install_initial(blue)
    activator = BootstrapReleaseActivator(supervisor)
    assert (
        await activator.activate(
            green,
            activation_id="promotion_green",
            origin=_origin(),
            reason="Owner approved",
            rollback=False,
        )
    ).status is ReleaseActivationStatus.ACTIVE

    rolled_back = await activator.activate(
        blue,
        activation_id="rollback_blue",
        origin=_origin(),
        reason="Owner requested rollback",
        rollback=True,
    )

    assert rolled_back.status is ReleaseActivationStatus.ACTIVE
    assert supervisor.store.get_state().serving_release_id == blue.id
