from __future__ import annotations

import stat
from pathlib import Path

import pytest

from opentulpa.bootstrap.models import IngressEnvelope, OutboxEvent, ReleaseRecord
from opentulpa.bootstrap.store import BootstrapConflictError, BootstrapStore, LeaseFenceError


def _release(name: str, character: str) -> ReleaseRecord:
    return ReleaseRecord(
        id=f"release_{name}",
        candidate_id=f"candidate_{name}",
        source_commit=character * 40,
        artifact_digest=f"sha256:{character * 64}",
        manifest_digest=f"sha256:{character * 64}",
        entrypoint=("python", "-m", f"release_{name}"),
    )


def test_bootstrap_store_migrates_idempotently_and_fences_stale_leases(tmp_path: Path) -> None:
    path = tmp_path / "bootstrap.db"
    blue = _release("blue", "a")
    store = BootstrapStore(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    store.add_release(blue)
    first = store.install_initial_lease(blue.id)
    store.resume_ingress()

    assert store.schema_version == 1
    assert store.assert_active_lease(blue.id, first.epoch) == first

    reopened = BootstrapStore(path)
    assert reopened.schema_version == 1
    assert reopened.get_release(blue.id) == blue
    assert reopened.get_state().serving_release_id == blue.id

    reopened.enter_safe_mode()
    with pytest.raises(LeaseFenceError, match="stale"):
        reopened.assert_active_lease(blue.id, first.epoch)


def test_ingress_is_idempotent_requeued_on_lease_rotation_and_completed_once(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    blue = _release("blue", "a")
    store.add_release(blue)
    first = store.install_initial_lease(blue.id)
    store.resume_ingress()
    envelope = IngressEnvelope(
        tenant_id="tenant_1",
        thread_id="thread_1",
        channel="telegram",
        idempotency_key="update:42",
        payload={"update_id": 42, "text": "hello"},
    )

    assert store.enqueue_ingress(envelope) == envelope
    assert store.enqueue_ingress(envelope.model_copy(update={"id": "ignored_duplicate"})) == envelope
    claimed = store.claim_ingress(
        release_id=blue.id,
        lease_epoch=first.epoch,
    )
    assert [item.id for item in claimed] == [envelope.id]
    assert claimed[0].attempt_count == 1

    store.enter_safe_mode()
    replacement = store.begin_restore_lease(
        release_id=blue.id,
        activation_id=None,
    )
    store.complete_recovery(
        release_id=blue.id,
        lease=replacement,
        previous_release_id=None,
    )
    with pytest.raises(LeaseFenceError):
        store.complete_ingress(
            envelope.id,
            release_id=blue.id,
            lease_epoch=first.epoch,
        )

    reclaimed = store.claim_ingress(
        release_id=blue.id,
        lease_epoch=replacement.epoch,
    )
    assert reclaimed[0].attempt_count == 2
    completed = store.complete_ingress(
        envelope.id,
        release_id=blue.id,
        lease_epoch=replacement.epoch,
    )
    assert completed.status == "processed"
    assert store.complete_ingress(
        envelope.id,
        release_id=blue.id,
        lease_epoch=replacement.epoch,
    ) == completed


def test_outbox_is_durable_and_event_keys_cannot_change_payload(tmp_path: Path) -> None:
    path = tmp_path / "bootstrap.db"
    store = BootstrapStore(path)
    event = OutboxEvent(
        event_key="activation:a:failed",
        event_type="activation.failed",
        payload={"activation_id": "a", "failure_code": "staging_unhealthy"},
    )

    assert store.append_outbox(event) == event
    assert BootstrapStore(path).pending_outbox() == [event]
    with pytest.raises(BootstrapConflictError, match="another payload"):
        store.append_outbox(event.model_copy(update={"payload": {"changed": True}}))

    attempted = store.mark_outbox_attempt(event.id, delivered=False)
    assert attempted.attempt_count == 1
    delivered = store.mark_outbox_attempt(event.id, delivered=True)
    assert delivered.status == "delivered"
    assert delivered.attempt_count == 2
    assert BootstrapStore(path).pending_outbox() == []


def test_failed_ingress_delivery_can_only_be_requeued_by_its_active_lease(
    tmp_path: Path,
) -> None:
    store = BootstrapStore(tmp_path / "bootstrap.db")
    blue = _release("blue", "a")
    store.add_release(blue)
    lease = store.install_initial_lease(blue.id)
    store.resume_ingress()
    envelope = store.enqueue_ingress(
        IngressEnvelope(
            tenant_id="tenant_1",
            thread_id="thread_1",
            channel="telegram",
            idempotency_key="update:99",
            payload={"text": "retry me"},
        )
    )
    assert store.get_ingress(envelope.id) == envelope
    claimed = store.claim_ingress(release_id=blue.id, lease_epoch=lease.epoch)[0]

    with pytest.raises(LeaseFenceError):
        store.requeue_ingress_claim(
            claimed.id,
            release_id=blue.id,
            lease_epoch=lease.epoch + 1,
        )
    requeued = store.requeue_ingress_claim(
        claimed.id,
        release_id=blue.id,
        lease_epoch=lease.epoch,
    )

    assert requeued.status == "pending"
    assert requeued.claimed_epoch is None
    assert requeued.attempt_count == 1
