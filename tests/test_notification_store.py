from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from opentulpa.notifications import (
    NotificationApproval,
    NotificationDedupeConflictError,
    NotificationOrigin,
    NotificationService,
    NotificationStore,
    NotificationWrite,
)


def _payload(text: str = "Background work completed.") -> NotificationWrite:
    return NotificationWrite(
        kind="run.completed",
        text=text,
        status="completed",
        thread_id="thread-1",
        run_id="run-1",
        origin=NotificationOrigin(
            interface="web",
            source_id="owner-web",
            correlation_id="correlation-1",
        ),
    )


def test_notifications_are_monotonic_deduplicated_and_acknowledged_per_consumer(
    tmp_path: Path,
) -> None:
    current = datetime(2026, 7, 20, tzinfo=UTC)
    store = NotificationStore(tmp_path / "notifications.db")
    service = NotificationService(store, clock=lambda: current)

    first = service.publish(
        tenant_id="tenant-a",
        dedupe_key="trigger:daily:run-1",
        notification=_payload(),
    )
    replay = service.publish(
        tenant_id="tenant-a",
        dedupe_key="trigger:daily:run-1",
        notification=_payload(),
    )
    current += timedelta(seconds=1)
    second = service.publish(
        tenant_id="tenant-a",
        dedupe_key="trigger:daily:run-2",
        notification=_payload("Second result."),
    )
    other_tenant = service.publish(
        tenant_id="tenant-b",
        dedupe_key="trigger:daily:run-1",
        notification=_payload("Private tenant result."),
    )

    assert replay == first
    assert first.id < second.id < other_tenant.id
    assert [item.id for item in store.list_unacked(
        tenant_id="tenant-a", consumer_id="web:owner", after_id=0
    )] == [first.id, second.id]

    assert service.acknowledge(
        tenant_id="tenant-a",
        consumer_id="web:owner",
        notification_id=first.id,
    )
    assert not service.acknowledge(
        tenant_id="tenant-a",
        consumer_id="web:owner",
        notification_id=first.id,
    )
    assert [item.id for item in store.list_unacked(
        tenant_id="tenant-a", consumer_id="web:owner", after_id=0
    )] == [second.id]
    assert [item.id for item in store.list_unacked(
        tenant_id="tenant-a", consumer_id="telegram:instance", after_id=0
    )] == [first.id, second.id]
    assert [item.id for item in store.list_unacked(
        tenant_id="tenant-b", consumer_id="web:owner", after_id=0
    )] == [other_tenant.id]

    restarted = NotificationStore(tmp_path / "notifications.db")
    assert [item.id for item in restarted.list_unacked(
        tenant_id="tenant-a", consumer_id="web:owner", after_id=0
    )] == [second.id]


def test_dedupe_key_cannot_hide_a_different_notification(tmp_path: Path) -> None:
    service = NotificationService(NotificationStore(tmp_path / "notifications.db"))
    service.publish(
        tenant_id="tenant-a",
        dedupe_key="same-key",
        notification=_payload("Original."),
    )

    with pytest.raises(NotificationDedupeConflictError):
        service.publish(
            tenant_id="tenant-a",
            dedupe_key="same-key",
            notification=_payload("Different."),
        )


@pytest.mark.asyncio
async def test_long_poll_observes_a_later_publish(tmp_path: Path) -> None:
    service = NotificationService(
        NotificationStore(tmp_path / "notifications.db"),
        poll_interval_seconds=0.005,
    )
    waiting = asyncio.create_task(
        service.wait(
            tenant_id="tenant-a",
            consumer_id="web:owner",
            wait_seconds=0.5,
        )
    )
    await asyncio.sleep(0.02)
    published = service.publish(
        tenant_id="tenant-a",
        dedupe_key="later",
        notification=NotificationWrite(
            kind="approval.required",
            text="Approval is waiting.",
            status="interrupted",
            run_id="run-waiting",
            approvals=(
                NotificationApproval(
                    approval_id="approval-1",
                    tool_name="integration_invoke",
                    description="Send an email.",
                    allowed_decisions=("approve", "reject"),
                ),
            ),
        ),
    )

    assert await waiting == [published]
