import sqlite3
from pathlib import Path

import pytest

from opentulpa.persistence.idempotency import (
    IdempotencyConflictError,
    IdempotencyPendingError,
    IdempotencyStore,
)


def test_external_effect_is_replayed_and_mismatches_fail_closed(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "effects.sqlite3")
    claim = store.claim(
        tenant_id="a",
        idempotency_key="key-1",
        operation="send",
        arguments={"to": "x"},
    )
    assert claim.created
    with pytest.raises(IdempotencyPendingError):
        store.claim(
            tenant_id="a",
            idempotency_key="key-1",
            operation="send",
            arguments={"to": "x"},
        )
    store.complete(
        tenant_id="a",
        idempotency_key="key-1",
        result={"message_id": "1"},
    )
    replay = store.claim(
        tenant_id="a",
        idempotency_key="key-1",
        operation="send",
        arguments={"to": "x"},
    )
    assert replay.result == {"message_id": "1"}
    with pytest.raises(IdempotencyConflictError):
        store.claim(
            tenant_id="a",
            idempotency_key="key-1",
            operation="send",
            arguments={"to": "different"},
        )


def test_effect_keys_are_tenant_scoped(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "effects.sqlite3")
    for tenant in ("a", "b"):
        assert store.claim(
            tenant_id=tenant,
            idempotency_key="same",
            operation="send",
            arguments={},
        ).created


def test_explicit_no_effect_can_reconcile_and_retry_matching_pending_claim(
    tmp_path: Path,
) -> None:
    store = IdempotencyStore(tmp_path / "effects.sqlite3")
    arguments = {"booking_id": "booking-1"}
    assert store.claim(
        tenant_id="a",
        idempotency_key="sink-1",
        operation="upsert",
        arguments=arguments,
    ).created

    store.reconcile_pending(
        tenant_id="a",
        idempotency_key="sink-1",
        decision="retry_no_effect",
        actor_id="owner-1",
        reason="provider confirmed the write was not applied",
    )

    assert store.claim(
        tenant_id="a",
        idempotency_key="sink-1",
        operation="upsert",
        arguments=arguments,
    ).created
    with sqlite3.connect(tmp_path / "effects.sqlite3") as connection:
        audit = connection.execute(
            """
            SELECT tenant_id, idempotency_key, decision, actor_id, reason
            FROM external_effect_reconciliations
            """
        ).fetchone()
    assert audit == (
        "a",
        "sink-1",
        "retry_no_effect",
        "owner-1",
        "provider confirmed the write was not applied",
    )


def test_confirm_applied_replays_result_and_cannot_cross_tenants(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "effects.sqlite3")
    arguments = {"booking_id": "booking-1"}
    assert store.claim(
        tenant_id="tenant-a",
        idempotency_key="sink-1",
        operation="upsert",
        arguments=arguments,
    ).created

    with pytest.raises(LookupError):
        store.reconcile_pending(
            tenant_id="tenant-b",
            idempotency_key="sink-1",
            decision="confirm_applied",
            actor_id="owner-b",
            reason="not this tenant's effect",
            result={"external_id": "wrong"},
        )

    store.reconcile_pending(
        tenant_id="tenant-a",
        idempotency_key="sink-1",
        decision="confirm_applied",
        actor_id="owner-a",
        reason="verified in the provider",
        result={"external_id": "record-1"},
    )
    replay = store.claim(
        tenant_id="tenant-a",
        idempotency_key="sink-1",
        operation="upsert",
        arguments=arguments,
    )
    assert replay.result == {"external_id": "record-1"}


@pytest.mark.asyncio
async def test_execute_calls_provider_once_and_replays(tmp_path: Path) -> None:
    store = IdempotencyStore(tmp_path / "effects.sqlite3")
    calls = 0

    async def invoke() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"external_id": "x"}

    first = await store.execute(
        tenant_id="a",
        operation="send",
        idempotency_key="send-1",
        request_hash="hash",
        invoke=invoke,
    )
    second = await store.execute(
        tenant_id="a",
        operation="send",
        idempotency_key="send-1",
        request_hash="hash",
        invoke=invoke,
    )
    assert first == second == {"external_id": "x"}
    assert calls == 1
