"""Tenant-scoped SQLite persistence for the owner notification stream."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from opentulpa.notifications.models import NotificationWrite, OwnerNotification
from opentulpa.persistence.sqlite import connect_sqlite


class NotificationDedupeConflictError(RuntimeError):
    """A dedupe key was reused with a different immutable payload."""


class NotificationNotFoundError(KeyError):
    """A notification does not exist inside the authenticated tenant."""


class NotificationStore:
    """Append-only notifications with per-interface delivery acknowledgements."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS owner_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    dedupe_key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    thread_id TEXT,
                    run_id TEXT,
                    origin_json TEXT,
                    approvals_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE (tenant_id, dedupe_key)
                );

                CREATE INDEX IF NOT EXISTS idx_owner_notifications_tenant_id
                ON owner_notifications (tenant_id, id);

                CREATE TABLE IF NOT EXISTS owner_notification_acks (
                    tenant_id TEXT NOT NULL,
                    consumer_id TEXT NOT NULL,
                    notification_id INTEGER NOT NULL,
                    acked_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, consumer_id, notification_id),
                    FOREIGN KEY (notification_id) REFERENCES owner_notifications (id)
                );

                CREATE INDEX IF NOT EXISTS idx_owner_notification_acks_consumer
                ON owner_notification_acks (tenant_id, consumer_id, notification_id);
                """
            )
            conn.commit()

    def append(
        self,
        *,
        tenant_id: str,
        dedupe_key: str,
        payload: NotificationWrite,
        created_at: datetime,
    ) -> tuple[OwnerNotification, bool]:
        tenant = _identity(tenant_id, label="tenant_id", max_length=200)
        key = _identity(dedupe_key, label="dedupe_key", max_length=300)
        timestamp = _aware_iso(created_at)
        payload_json = payload.model_dump(mode="json")
        canonical = json.dumps(
            payload_json,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT * FROM owner_notifications
                WHERE tenant_id = ? AND dedupe_key = ?
                """,
                (tenant, key),
            ).fetchone()
            if existing is not None:
                if not hmac.compare_digest(str(existing["payload_hash"]), payload_hash):
                    conn.rollback()
                    raise NotificationDedupeConflictError(
                        "notification dedupe key is bound to a different payload"
                    )
                conn.commit()
                return self._notification(existing), False
            cursor = conn.execute(
                """
                INSERT INTO owner_notifications (
                    tenant_id, dedupe_key, payload_hash, kind, text, status,
                    thread_id, run_id, origin_json, approvals_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant,
                    key,
                    payload_hash,
                    payload.kind,
                    payload.text,
                    payload.status,
                    payload.thread_id,
                    payload.run_id,
                    (
                        json.dumps(
                            payload.origin.model_dump(mode="json"),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        if payload.origin is not None
                        else None
                    ),
                    json.dumps(
                        [item.model_dump(mode="json") for item in payload.approvals],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    timestamp,
                ),
            )
            notification_id = int(cursor.lastrowid or 0)
            row = conn.execute(
                "SELECT * FROM owner_notifications WHERE id = ?",
                (notification_id,),
            ).fetchone()
            conn.commit()
        if row is None:  # pragma: no cover - guarded by the insert transaction
            raise RuntimeError("notification insert did not return a row")
        return self._notification(row), True

    def get(self, *, tenant_id: str, notification_id: int) -> OwnerNotification:
        tenant = _identity(tenant_id, label="tenant_id", max_length=200)
        identifier = _positive_id(notification_id)
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM owner_notifications WHERE tenant_id = ? AND id = ?",
                (tenant, identifier),
            ).fetchone()
        if row is None:
            raise NotificationNotFoundError(identifier)
        return self._notification(row)

    def list_unacked(
        self,
        *,
        tenant_id: str,
        consumer_id: str,
        after_id: int = 0,
        limit: int = 100,
    ) -> list[OwnerNotification]:
        tenant = _identity(tenant_id, label="tenant_id", max_length=200)
        consumer = _identity(consumer_id, label="consumer_id", max_length=300)
        cursor = max(0, int(after_id))
        safe_limit = max(1, min(int(limit), 100))
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT notification.*
                FROM owner_notifications AS notification
                LEFT JOIN owner_notification_acks AS ack
                  ON ack.tenant_id = notification.tenant_id
                 AND ack.consumer_id = ?
                 AND ack.notification_id = notification.id
                WHERE notification.tenant_id = ?
                  AND notification.id > ?
                  AND ack.notification_id IS NULL
                ORDER BY notification.id ASC
                LIMIT ?
                """,
                (consumer, tenant, cursor, safe_limit),
            ).fetchall()
        return [self._notification(row) for row in rows]

    def acknowledge(
        self,
        *,
        tenant_id: str,
        consumer_id: str,
        notification_id: int,
        acked_at: datetime,
    ) -> bool:
        tenant = _identity(tenant_id, label="tenant_id", max_length=200)
        consumer = _identity(consumer_id, label="consumer_id", max_length=300)
        identifier = _positive_id(notification_id)
        timestamp = _aware_iso(acked_at)
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM owner_notifications WHERE tenant_id = ? AND id = ?",
                (tenant, identifier),
            ).fetchone()
            if owned is None:
                conn.rollback()
                raise NotificationNotFoundError(identifier)
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO owner_notification_acks (
                    tenant_id, consumer_id, notification_id, acked_at
                ) VALUES (?, ?, ?, ?)
                """,
                (tenant, consumer, identifier, timestamp),
            )
            conn.commit()
        return cursor.rowcount == 1

    @staticmethod
    def _notification(row: sqlite3.Row) -> OwnerNotification:
        return OwnerNotification.model_validate(
            {
                "id": row["id"],
                "tenant_id": row["tenant_id"],
                "dedupe_key": row["dedupe_key"],
                "kind": row["kind"],
                "text": row["text"],
                "status": row["status"],
                "thread_id": row["thread_id"],
                "run_id": row["run_id"],
                "origin": json.loads(str(row["origin_json"])) if row["origin_json"] else None,
                "approvals": json.loads(str(row["approvals_json"])),
                "created_at": row["created_at"],
            }
        )


def _identity(value: str, *, label: str, max_length: int) -> str:
    safe = str(value or "").strip()
    if not safe or len(safe) > max_length or any(ord(char) < 32 for char in safe):
        raise ValueError(f"{label} is invalid")
    return safe


def _positive_id(value: int) -> int:
    identifier = int(value)
    if identifier < 1:
        raise ValueError("notification_id must be positive")
    return identifier


def _aware_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("notification timestamp must include a UTC offset")
    return value.isoformat()


__all__ = [
    "NotificationDedupeConflictError",
    "NotificationNotFoundError",
    "NotificationStore",
]
