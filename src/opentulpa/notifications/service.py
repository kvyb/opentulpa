"""Small application service for durable owner notifications."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

from opentulpa.notifications.models import NotificationWrite, OwnerNotification
from opentulpa.notifications.store import NotificationStore


class NotificationService:
    """Publish once, wait without a second event loop, and acknowledge per interface."""

    def __init__(
        self,
        store: NotificationStore,
        *,
        clock: Callable[[], datetime] | None = None,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("notification poll interval must be positive")
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))
        self._poll_interval_seconds = poll_interval_seconds

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("notification clock must return an aware datetime")
        return value.astimezone(UTC)

    def publish(
        self,
        *,
        tenant_id: str,
        dedupe_key: str,
        notification: NotificationWrite,
    ) -> OwnerNotification:
        stored, _ = self._store.append(
            tenant_id=tenant_id,
            dedupe_key=dedupe_key,
            payload=notification,
            created_at=self._now(),
        )
        return stored

    async def wait(
        self,
        *,
        tenant_id: str,
        consumer_id: str,
        after_id: int = 0,
        limit: int = 100,
        wait_seconds: float = 0,
    ) -> list[OwnerNotification]:
        timeout = max(0.0, min(float(wait_seconds), 30.0))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            notifications = self._store.list_unacked(
                tenant_id=tenant_id,
                consumer_id=consumer_id,
                after_id=after_id,
                limit=limit,
            )
            if notifications or loop.time() >= deadline:
                return notifications
            await asyncio.sleep(
                min(self._poll_interval_seconds, max(0.0, deadline - loop.time()))
            )

    def acknowledge(
        self,
        *,
        tenant_id: str,
        consumer_id: str,
        notification_id: int,
    ) -> bool:
        return self._store.acknowledge(
            tenant_id=tenant_id,
            consumer_id=consumer_id,
            notification_id=notification_id,
            acked_at=self._now(),
        )


__all__ = ["NotificationService"]
