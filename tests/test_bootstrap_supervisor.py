from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path

import pytest

from opentulpa.bootstrap.host import InMemoryReleaseHost
from opentulpa.bootstrap.models import (
    ActivationKind,
    ActivationStatus,
    IngressEnvelope,
    ReleaseLaunchContext,
    ReleaseOrigin,
    ReleaseRecord,
)
from opentulpa.bootstrap.store import BootstrapConflictError, BootstrapStore, LeaseFenceError
from opentulpa.bootstrap.supervisor import (
    BootstrapSupervisor,
    InMemoryOutboxSink,
    SupervisorPolicy,
)


def _release(name: str, character: str) -> ReleaseRecord:
    return ReleaseRecord(
        id=f"release_{name}",
        candidate_id=f"candidate_{name}",
        source_commit=character * 40,
        artifact_digest=f"sha256:{character * 64}",
        manifest_digest=f"sha256:{character * 64}",
        entrypoint=("python", "-m", f"release_{name}"),
    )


def _origin() -> ReleaseOrigin:
    return ReleaseOrigin(
        tenant_id="tenant_1",
        actor_id="owner_1",
        thread_id="thread_1",
        run_id="run_1",
        channel="web",
        correlation_id="correlation_1",
    )


def _policy(*, probation_seconds: float = 0) -> SupervisorPolicy:
    return SupervisorPolicy(
        drain_timeout_seconds=1,
        stage_probe_attempts=1,
        production_probe_attempts=1,
        probe_interval_seconds=0,
        probation_seconds=probation_seconds,
        probation_probe_interval_seconds=1,
    )


@pytest.mark.asyncio
async def test_lease_change_hook_runs_after_old_stop_and_before_new_start(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    observed: list[tuple[str | None, bool, bool]] = []
    blue = _release("blue", "a")
    green = _release("green", "b")

    async def lease_changed(lease) -> None:  # type: ignore[no-untyped-def]
        observed.append(
            (
                lease.release_id if lease is not None else None,
                await host.discover(blue.id) is not None,
                await host.discover(green.id) is not None,
            )
        )

    supervisor.set_lease_change_hook(lease_changed)
    await supervisor.start()
    await supervisor.install_initial(blue)
    queued = await supervisor.request_activation(green, origin=_origin())
    await supervisor.activate(queued.id)

    assert observed == [
        (None, False, False),
        (blue.id, False, False),
        (green.id, False, False),
    ]


@pytest.mark.asyncio
async def test_staged_activation_rotates_lease_requeues_ingress_and_becomes_active(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    sink = InMemoryOutboxSink()
    supervisor = BootstrapSupervisor(
        store=store,
        host=host,
        policy=_policy(),
        outbox_sink=sink,
    )
    await supervisor.start()
    blue = _release("blue", "a")
    old_lease = await supervisor.install_initial(blue)
    envelope = store.enqueue_ingress(
        IngressEnvelope(
            tenant_id="tenant_1",
            thread_id="thread_1",
            channel="telegram",
            idempotency_key="update:1",
            payload={"text": "during deployment"},
        )
    )
    assert store.claim_ingress(
        release_id=blue.id,
        lease_epoch=old_lease.epoch,
    )[0].id == envelope.id

    green = _release("green", "b")
    queued = await supervisor.request_activation(green, origin=_origin())
    active = await supervisor.activate(queued.id)

    assert active.status is ActivationStatus.ACTIVE
    state = store.get_state()
    assert state.serving_release_id == green.id
    assert state.last_known_good_release_id == green.id
    assert state.previous_release_id == blue.id
    assert state.ingress_paused is False
    assert state.active_lease_epoch is not None
    with pytest.raises(LeaseFenceError):
        store.assert_active_lease(blue.id, old_lease.epoch)
    reclaimed = store.claim_ingress(
        release_id=green.id,
        lease_epoch=state.active_lease_epoch,
    )
    assert reclaimed[0].id == envelope.id
    assert reclaimed[0].attempt_count == 2
    assert ("start", green.id, "staging") in host.calls
    assert ("start", green.id, "production") in host.calls
    assert any(event.event_type == "release.active" for event in sink.events)
    assert all(event.status == "delivered" for event in store.pending_outbox())


@pytest.mark.asyncio
async def test_probation_failure_automatically_restores_previous_release(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    sink = InMemoryOutboxSink()
    supervisor = BootstrapSupervisor(
        store=store,
        host=host,
        policy=_policy(),
        outbox_sink=sink,
    )
    await supervisor.start()
    blue = _release("blue", "a")
    await supervisor.install_initial(blue)
    green = _release("green", "b")
    host.capability_state["telegram_cursor"] = 10
    host.product_state["message"] = "before probation"
    host.mutate_state_on_start(green.id, values={"telegram_cursor": 99})
    host.mutate_product_state_on_start(
        green.id,
        values={"message": "written during probation"},
    )
    host.health_sequence(green.id, mode="production", values=(True, False))

    queued = await supervisor.request_activation(green, origin=_origin())
    result = await supervisor.activate(queued.id)

    assert result.status is ActivationStatus.ROLLED_BACK
    assert result.failure_code == "probation_unhealthy"
    state = store.get_state()
    assert state.serving_release_id == blue.id
    assert state.last_known_good_release_id == blue.id
    assert state.previous_release_id == green.id
    assert state.ingress_paused is False
    assert host.capability_state == {"telegram_cursor": 10}
    assert host.product_state == {"message": "written during probation"}
    assert ("restore", queued.id, None) in host.calls
    assert ("discard_snapshot", queued.id, None) in host.calls
    assert any(event.event_type == "activation.failed" for event in sink.events)
    assert any(event.event_type == "rollback.completed" for event in sink.events)


@pytest.mark.asyncio
async def test_failed_candidate_stop_enters_safe_mode_without_starting_rollback(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await supervisor.install_initial(blue)
    host.health_sequence(green.id, mode="production", values=(True, False))
    original_stop = host.stop

    async def fail_green_production_stop(running) -> None:  # type: ignore[no-untyped-def]
        if running.release_id == green.id and running.mode == "production":
            raise RuntimeError("configured containment failure")
        await original_stop(running)

    host.stop = fail_green_production_stop  # type: ignore[method-assign]

    queued = await supervisor.request_activation(green, origin=_origin())
    result = await supervisor.activate(queued.id)

    assert result.status is ActivationStatus.FAILED
    assert result.failure_code == "candidate_containment_failed"
    assert store.get_state().safe_mode is True
    assert [call for call in host.calls if call == ("start", blue.id, "production")] == [
        ("start", blue.id, "production")
    ]


@pytest.mark.asyncio
async def test_failed_restored_release_probe_stops_started_rollback_process(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    host.health_sequence(blue.id, mode="production", values=(True, False))
    await supervisor.install_initial(blue)
    host.health_sequence(green.id, mode="production", values=(True, False))

    queued = await supervisor.request_activation(green, origin=_origin())
    result = await supervisor.activate(queued.id)

    assert result.status is ActivationStatus.FAILED
    assert result.failure_code == "rollback_failed"
    assert store.get_state().safe_mode is True
    assert await host.discover(blue.id, mode="production") is None
    assert host.calls.count(("stop", blue.id, "production")) == 2


@pytest.mark.asyncio
async def test_staging_failure_keeps_blue_and_does_not_rotate_lease(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    blue = _release("blue", "a")
    lease = await supervisor.install_initial(blue)
    green = _release("green", "b")
    host.health_sequence(green.id, mode="staging", values=(False,))

    queued = await supervisor.request_activation(green, origin=_origin())
    result = await supervisor.activate(queued.id)

    assert result.status is ActivationStatus.FAILED
    assert result.failure_code == "staging_unhealthy"
    state = store.get_state()
    assert state.serving_release_id == blue.id
    assert state.active_lease_epoch == lease.epoch
    assert store.assert_active_lease(blue.id, lease.epoch).status == "active"
    assert ("start", green.id, "production") not in host.calls


@pytest.mark.asyncio
async def test_drain_failure_resumes_old_release_ingress(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await supervisor.install_initial(blue)
    host.drain_result(blue.id, drained=False, in_flight=2)

    queued = await supervisor.request_activation(green, origin=_origin())
    failed = await supervisor.activate(queued.id)

    assert failed.status is ActivationStatus.FAILED
    assert failed.failure_code == "drain_timeout"
    state = store.get_state()
    assert state.serving_release_id == blue.id
    assert state.ingress_paused is False


@pytest.mark.asyncio
async def test_only_one_nonterminal_activation_can_be_queued(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    await supervisor.install_initial(_release("blue", "a"))
    await supervisor.request_activation(_release("green", "b"), origin=_origin())

    with pytest.raises(BootstrapConflictError, match="already in progress"):
        await supervisor.request_activation(_release("red", "c"), origin=_origin())


@pytest.mark.asyncio
async def test_manual_rollback_is_a_normal_verified_activation(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    supervisor = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await supervisor.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await supervisor.install_initial(blue)
    deploy = await supervisor.request_activation(green, origin=_origin())
    assert (await supervisor.activate(deploy.id)).status is ActivationStatus.ACTIVE

    rollback = await supervisor.request_rollback(origin=_origin(), reason="Telegram broke")
    result = await supervisor.activate(rollback.id)

    assert result.kind is ActivationKind.ROLLBACK
    assert result.status is ActivationStatus.ACTIVE
    assert store.get_state().serving_release_id == blue.id
    assert store.get_state().previous_release_id == green.id


@pytest.mark.asyncio
async def test_restart_during_probation_restores_last_known_good(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    entered_sleep = asyncio.Event()
    never_finish = asyncio.Event()

    async def crashable_sleep(_: float) -> None:
        entered_sleep.set()
        await never_finish.wait()

    first = BootstrapSupervisor(
        store=store,
        host=host,
        policy=_policy(probation_seconds=60),
        sleep=crashable_sleep,
    )
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    host.capability_state["telegram_cursor"] = 10
    host.product_state["message"] = "before probation"
    host.mutate_state_on_start(green.id, values={"telegram_cursor": 99})
    host.mutate_product_state_on_start(
        green.id,
        values={"message": "written during probation"},
    )
    queued = await first.request_activation(green, origin=_origin())
    task = asyncio.create_task(first.activate(queued.id))
    await entered_sleep.wait()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert store.get_state().serving_release_id == green.id
    assert store.get_state().last_known_good_release_id == blue.id

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    recovered = store.get_activation(queued.id)
    assert recovered is not None
    assert recovered.status is ActivationStatus.ROLLED_BACK
    assert recovered.failure_code == "bootstrap_restarted"
    assert store.get_state().serving_release_id == blue.id
    assert store.get_state().last_known_good_release_id == blue.id
    assert host.capability_state == {"telegram_cursor": 10}
    assert host.product_state == {"message": "written during probation"}
    assert ("restore", queued.id, None) in host.calls
    assert ("discard_snapshot", queued.id, None) in host.calls


@pytest.mark.parametrize("crash_status", [ActivationStatus.QUEUED, ActivationStatus.STAGED])
@pytest.mark.asyncio
async def test_restart_before_drain_retains_last_known_good(
    tmp_path: Path,
    crash_status: ActivationStatus,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    queued = await first.request_activation(green, origin=_origin())
    if crash_status is ActivationStatus.STAGED:
        preparing = store.transition_activation(
            queued.id,
            expected=ActivationStatus.QUEUED,
            target=ActivationStatus.PREPARING,
        )
        store.transition_activation(
            preparing.id,
            expected=ActivationStatus.PREPARING,
            target=ActivationStatus.STAGED,
        )

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    recovered = store.get_activation(queued.id)
    assert recovered is not None
    assert recovered.status is ActivationStatus.FAILED
    assert recovered.failure_code == "bootstrap_restarted"
    state = store.get_state()
    assert state.serving_release_id == blue.id
    assert state.last_known_good_release_id == blue.id
    assert state.ingress_paused is False


@pytest.mark.parametrize(
    "crash_status",
    [ActivationStatus.VERIFYING, ActivationStatus.ROLLING_BACK],
)
@pytest.mark.asyncio
async def test_restart_after_candidate_start_restores_last_known_good(
    tmp_path: Path,
    crash_status: ActivationStatus,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    queued = await first.request_activation(green, origin=_origin())
    preparing = store.transition_activation(
        queued.id,
        expected=ActivationStatus.QUEUED,
        target=ActivationStatus.PREPARING,
    )
    staged = store.transition_activation(
        preparing.id,
        expected=ActivationStatus.PREPARING,
        target=ActivationStatus.STAGED,
    )
    draining = store.transition_activation(
        staged.id,
        expected=ActivationStatus.STAGED,
        target=ActivationStatus.DRAINING,
    )
    starting = store.transition_activation(
        draining.id,
        expected=ActivationStatus.DRAINING,
        target=ActivationStatus.STARTING,
    )
    await host.snapshot_state(starting.id)
    green_lease = store.begin_cutover(starting)
    await host.start(
        await host.prepare(green),
        ReleaseLaunchContext(
            mode="production",
            lease_epoch=green_lease.epoch,
            secrets_enabled=True,
            ingress_enabled=False,
        ),
    )
    current = store.transition_activation(
        starting.id,
        expected=ActivationStatus.STARTING,
        target=ActivationStatus.VERIFYING,
        lease_epoch=green_lease.epoch,
    )
    if crash_status is ActivationStatus.ROLLING_BACK:
        store.transition_activation(
            current.id,
            expected=ActivationStatus.VERIFYING,
            target=ActivationStatus.ROLLING_BACK,
        )

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    recovered = store.get_activation(queued.id)
    assert recovered is not None
    assert recovered.status is ActivationStatus.ROLLED_BACK
    assert recovered.failure_code == "bootstrap_restarted"
    state = store.get_state()
    assert state.serving_release_id == blue.id
    assert state.last_known_good_release_id == blue.id
    assert state.ingress_paused is False
    assert await host.discover(green.id, mode="production") is None
    assert ("restore", queued.id, None) in host.calls


@pytest.mark.asyncio
async def test_restart_after_lease_rotation_but_before_green_start_restores_blue(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    queued = await first.request_activation(green, origin=_origin())
    preparing = store.transition_activation(
        queued.id,
        expected=ActivationStatus.QUEUED,
        target=ActivationStatus.PREPARING,
    )
    staged = store.transition_activation(
        preparing.id,
        expected=ActivationStatus.PREPARING,
        target=ActivationStatus.STAGED,
    )
    draining = store.transition_activation(
        staged.id,
        expected=ActivationStatus.STAGED,
        target=ActivationStatus.DRAINING,
    )
    starting = store.transition_activation(
        draining.id,
        expected=ActivationStatus.DRAINING,
        target=ActivationStatus.STARTING,
    )
    await host.snapshot_state(starting.id)
    green_lease = store.begin_cutover(starting)
    assert store.get_state().serving_release_id == blue.id
    assert store.get_state().active_lease_epoch == green_lease.epoch

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    recovered = store.get_activation(queued.id)
    assert recovered is not None
    assert recovered.status is ActivationStatus.ROLLED_BACK
    assert store.get_state().serving_release_id == blue.id
    assert store.get_state().active_lease_epoch != green_lease.epoch
    with pytest.raises(LeaseFenceError):
        store.assert_active_lease(green.id, green_lease.epoch)


@pytest.mark.asyncio
async def test_restart_after_cutover_fails_closed_without_capability_state_snapshot(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    queued = await first.request_activation(green, origin=_origin())
    preparing = store.transition_activation(
        queued.id,
        expected=ActivationStatus.QUEUED,
        target=ActivationStatus.PREPARING,
    )
    staged = store.transition_activation(
        preparing.id,
        expected=ActivationStatus.PREPARING,
        target=ActivationStatus.STAGED,
    )
    draining = store.transition_activation(
        staged.id,
        expected=ActivationStatus.STAGED,
        target=ActivationStatus.DRAINING,
    )
    starting = store.transition_activation(
        draining.id,
        expected=ActivationStatus.DRAINING,
        target=ActivationStatus.STARTING,
    )
    store.begin_cutover(starting)

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    state = store.get_state()
    assert state.safe_mode is True
    assert state.ingress_paused is True
    assert ("restore", queued.id, None) in host.calls


@pytest.mark.asyncio
async def test_restart_cleans_orphaned_staging_container(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    queued = await first.request_activation(green, origin=_origin())
    store.transition_activation(
        queued.id,
        expected=ActivationStatus.QUEUED,
        target=ActivationStatus.PREPARING,
    )
    await host.start(
        await host.prepare(green),
        ReleaseLaunchContext(mode="staging"),
    )

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    assert await host.discover(green.id, mode="staging") is None
    assert ("stop", green.id, "staging") in host.calls
    failed = store.get_activation(queued.id)
    assert failed is not None
    assert failed.status is ActivationStatus.FAILED


@pytest.mark.asyncio
async def test_restart_resumes_ingress_paused_before_drain(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    green = _release("green", "b")
    await first.install_initial(blue)
    queued = await first.request_activation(green, origin=_origin())
    preparing = store.transition_activation(
        queued.id,
        expected=ActivationStatus.QUEUED,
        target=ActivationStatus.PREPARING,
    )
    staged = store.transition_activation(
        preparing.id,
        expected=ActivationStatus.PREPARING,
        target=ActivationStatus.STAGED,
    )
    store.transition_activation(
        staged.id,
        expected=ActivationStatus.STAGED,
        target=ActivationStatus.DRAINING,
    )
    store.pause_ingress()

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    assert store.get_state().serving_release_id == blue.id
    assert store.get_state().ingress_paused is False
    failed = store.get_activation(queued.id)
    assert failed is not None
    assert failed.status is ActivationStatus.FAILED


@pytest.mark.asyncio
async def test_explicit_safe_mode_survives_bootstrap_restart(tmp_path: Path) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    first = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await first.start()
    blue = _release("blue", "a")
    await first.install_initial(blue)
    await first.enter_safe_mode()

    restarted = BootstrapSupervisor(store=store, host=host, policy=_policy())
    await restarted.start()

    state = store.get_state()
    assert state.safe_mode is True
    assert state.serving_release_id is None
    assert state.ingress_paused is True
