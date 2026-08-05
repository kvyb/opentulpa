"""Crash-safe SQLite state for release activation and gateway delivery."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Callable, Iterable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ValidationError

from opentulpa.bootstrap.models import (
    TERMINAL_ACTIVATION_STATUSES,
    ActivationRecord,
    ActivationStatus,
    BootstrapState,
    IngressEnvelope,
    OutboxEvent,
    ReleaseLease,
    ReleaseRecord,
    utc_now,
)
from opentulpa.evolution.release_provenance import ReleaseArtifactProvenance

_ModelT = TypeVar("_ModelT", bound=BaseModel)
_SCHEMA_VERSION = 1


class BootstrapStoreError(RuntimeError):
    pass


class BootstrapConflictError(BootstrapStoreError):
    pass


class BootstrapCorruptionError(BootstrapStoreError):
    pass


class LeaseFenceError(BootstrapStoreError):
    pass


_ALLOWED_TRANSITIONS: dict[ActivationStatus, frozenset[ActivationStatus]] = {
    ActivationStatus.QUEUED: frozenset(
        {ActivationStatus.PREPARING, ActivationStatus.CANCELLED, ActivationStatus.FAILED}
    ),
    ActivationStatus.PREPARING: frozenset(
        {ActivationStatus.STAGED, ActivationStatus.CANCELLED, ActivationStatus.FAILED}
    ),
    ActivationStatus.STAGED: frozenset(
        {ActivationStatus.DRAINING, ActivationStatus.CANCELLED, ActivationStatus.FAILED}
    ),
    ActivationStatus.DRAINING: frozenset(
        {ActivationStatus.STARTING, ActivationStatus.FAILED, ActivationStatus.ROLLING_BACK}
    ),
    ActivationStatus.STARTING: frozenset(
        {ActivationStatus.VERIFYING, ActivationStatus.FAILED, ActivationStatus.ROLLING_BACK}
    ),
    ActivationStatus.VERIFYING: frozenset(
        {ActivationStatus.PROBATION, ActivationStatus.FAILED, ActivationStatus.ROLLING_BACK}
    ),
    ActivationStatus.PROBATION: frozenset(
        {ActivationStatus.ACTIVE, ActivationStatus.FAILED, ActivationStatus.ROLLING_BACK}
    ),
    ActivationStatus.ACTIVE: frozenset({ActivationStatus.ROLLING_BACK}),
    ActivationStatus.ROLLING_BACK: frozenset(
        {ActivationStatus.ROLLED_BACK, ActivationStatus.FAILED}
    ),
    ActivationStatus.FAILED: frozenset(),
    ActivationStatus.ROLLED_BACK: frozenset(),
    ActivationStatus.CANCELLED: frozenset(),
}


class BootstrapStore:
    """Single-host control database independent from mutable release state."""

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._clock = clock or utc_now
        self._lock = threading.RLock()
        self._migrate()

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("bootstrap clock must return an aware datetime")
        return value.astimezone(UTC)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        for path in (
            self.path,
            self.path.with_name(f"{self.path.name}-wal"),
            self.path.with_name(f"{self.path.name}-shm"),
        ):
            try:
                os.chmod(path, 0o600)
            except FileNotFoundError:
                continue
        return connection

    def _migrate(self) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS bootstrap_schema_version (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        version INTEGER NOT NULL CHECK (version >= 0),
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS bootstrap_releases (
                        id TEXT PRIMARY KEY,
                        artifact_digest TEXT NOT NULL,
                        source_commit TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS bootstrap_activations (
                        id TEXT PRIMARY KEY,
                        target_release_id TEXT NOT NULL,
                        previous_release_id TEXT,
                        status TEXT NOT NULL,
                        revision INTEGER NOT NULL CHECK (revision >= 1),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (target_release_id)
                            REFERENCES bootstrap_releases(id) ON DELETE RESTRICT,
                        FOREIGN KEY (previous_release_id)
                            REFERENCES bootstrap_releases(id) ON DELETE RESTRICT
                    );

                    CREATE INDEX IF NOT EXISTS idx_bootstrap_activations_status_updated
                    ON bootstrap_activations (status, updated_at DESC, id);

                    CREATE TABLE IF NOT EXISTS bootstrap_state (
                        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                        payload_json TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS bootstrap_leases (
                        epoch INTEGER PRIMARY KEY AUTOINCREMENT,
                        release_id TEXT NOT NULL,
                        activation_id TEXT,
                        status TEXT NOT NULL CHECK (status IN ('active', 'revoked')),
                        issued_at TEXT NOT NULL,
                        revoked_at TEXT,
                        payload_json TEXT NOT NULL,
                        FOREIGN KEY (release_id)
                            REFERENCES bootstrap_releases(id) ON DELETE RESTRICT,
                        FOREIGN KEY (activation_id)
                            REFERENCES bootstrap_activations(id) ON DELETE RESTRICT
                    );

                    CREATE UNIQUE INDEX IF NOT EXISTS idx_bootstrap_one_active_lease
                    ON bootstrap_leases (status) WHERE status = 'active';

                    CREATE TABLE IF NOT EXISTS bootstrap_ingress (
                        id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        channel TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (
                            status IN ('pending', 'claimed', 'processed')
                        ),
                        claimed_epoch INTEGER,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        UNIQUE (tenant_id, channel, idempotency_key)
                    );

                    CREATE INDEX IF NOT EXISTS idx_bootstrap_ingress_delivery
                    ON bootstrap_ingress (status, created_at ASC, id);

                    CREATE TABLE IF NOT EXISTS bootstrap_outbox (
                        id TEXT PRIMARY KEY,
                        event_key TEXT NOT NULL UNIQUE,
                        event_type TEXT NOT NULL,
                        status TEXT NOT NULL CHECK (status IN ('pending', 'delivered')),
                        attempt_count INTEGER NOT NULL CHECK (attempt_count >= 0),
                        created_at TEXT NOT NULL,
                        delivered_at TEXT,
                        payload_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_bootstrap_outbox_delivery
                    ON bootstrap_outbox (status, created_at ASC, id);
                    """
                )
                row = connection.execute(
                    "SELECT version FROM bootstrap_schema_version WHERE singleton = 1"
                ).fetchone()
                if row is not None and int(row["version"]) > _SCHEMA_VERSION:
                    raise BootstrapStoreError(
                        "bootstrap database was created by a newer schema version"
                    )
                now = self._now().isoformat()
                connection.execute(
                    """
                    INSERT INTO bootstrap_schema_version (singleton, version, updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(singleton) DO UPDATE SET
                        version = excluded.version,
                        updated_at = excluded.updated_at
                    """,
                    (_SCHEMA_VERSION, now),
                )
                state = BootstrapState(updated_at=self._now())
                connection.execute(
                    "INSERT OR IGNORE INTO bootstrap_state (singleton, payload_json) VALUES (1, ?)",
                    (self._json(state),),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @property
    def schema_version(self) -> int:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT version FROM bootstrap_schema_version WHERE singleton = 1"
            ).fetchone()
        if row is None:
            raise BootstrapCorruptionError("bootstrap schema version is missing")
        return int(row["version"])

    def add_release(self, release: ReleaseRecord) -> ReleaseRecord:
        payload = self._json(release)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT payload_json FROM bootstrap_releases WHERE id = ?", (release.id,)
                ).fetchone()
                if existing is not None:
                    current = self._model(existing["payload_json"], ReleaseRecord)
                    if current != release:
                        raise BootstrapConflictError("release id is bound to another artifact")
                    connection.commit()
                    return current
                digest_owner = connection.execute(
                    "SELECT id FROM bootstrap_releases WHERE artifact_digest = ?",
                    (release.artifact_digest,),
                ).fetchone()
                if digest_owner is not None and str(digest_owner["id"]) != release.id:
                    raise BootstrapConflictError("artifact digest is bound to another release")
                connection.execute(
                    """
                    INSERT INTO bootstrap_releases (
                        id, artifact_digest, source_commit, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        release.id,
                        release.artifact_digest,
                        release.source_commit,
                        release.created_at.isoformat(),
                        payload,
                    ),
                )
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
        return release

    def add_release_alias(
        self,
        release: ReleaseRecord,
        *,
        artifact_release_id: str,
    ) -> ReleaseRecord:
        """Persist a new release identity for one exact previously trusted artifact."""

        source_id = self._identifier(artifact_release_id, field="artifact_release_id")
        payload = self._json(release)
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT payload_json FROM bootstrap_releases WHERE id = ?", (release.id,)
                ).fetchone()
                if existing is not None:
                    current = self._model(existing["payload_json"], ReleaseRecord)
                    if current != release:
                        raise BootstrapConflictError("release id is bound to another artifact")
                    connection.commit()
                    return current
                source_row = connection.execute(
                    "SELECT payload_json FROM bootstrap_releases WHERE id = ?", (source_id,)
                ).fetchone()
                if source_row is None:
                    raise BootstrapConflictError("release alias source is unavailable")
                source = self._model(source_row["payload_json"], ReleaseRecord)
                if not self._same_release_artifact(source, release):
                    raise BootstrapConflictError("release alias changed artifact provenance")
                connection.execute(
                    """
                    INSERT INTO bootstrap_releases (
                        id, artifact_digest, source_commit, created_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        release.id,
                        release.artifact_digest,
                        release.source_commit,
                        release.created_at.isoformat(),
                        payload,
                    ),
                )
                connection.commit()
                return release
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _same_release_artifact(first: ReleaseRecord, second: ReleaseRecord) -> bool:
        try:
            first_provenance = ReleaseArtifactProvenance.from_values(
                source_commit=first.source_commit,
                artifact_digest=first.artifact_digest,
                manifest_digest=first.manifest_digest,
                entrypoint=first.entrypoint,
                metadata=first.metadata,
            )
            second_provenance = ReleaseArtifactProvenance.from_values(
                source_commit=second.source_commit,
                artifact_digest=second.artifact_digest,
                manifest_digest=second.manifest_digest,
                entrypoint=second.entrypoint,
                metadata=second.metadata,
            )
        except (ValidationError, ValueError):
            return False
        fields_match = (
            first.candidate_id == second.candidate_id
            and first.protocol_version == second.protocol_version
            and first.agent_api_version == second.agent_api_version
            and first.control_api_version == second.control_api_version
            and first.control_port == second.control_port
            and first.health_path == second.health_path
            and first.drain_path == second.drain_path
            and first.ingress_path == second.ingress_path
            and first.event_path == second.event_path
        )
        return fields_match and first_provenance == second_provenance

    def get_release(self, release_id: str) -> ReleaseRecord | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM bootstrap_releases WHERE id = ?",
                (self._identifier(release_id, "release_id"),),
            ).fetchone()
        return self._model(row["payload_json"], ReleaseRecord) if row is not None else None

    def list_releases(self, *, limit: int = 100) -> list[ReleaseRecord]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM bootstrap_releases
                ORDER BY created_at DESC, id ASC LIMIT ?
                """,
                (max(1, min(int(limit), 1_000)),),
            ).fetchall()
        return [self._model(row["payload_json"], ReleaseRecord) for row in rows]

    def create_activation(self, activation: ActivationRecord) -> ActivationRecord:
        if activation.status is not ActivationStatus.QUEUED or activation.revision != 1:
            raise ValueError("new activation must be queued at revision 1")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                active = connection.execute(
                    """
                    SELECT id FROM bootstrap_activations
                    WHERE status NOT IN ('active', 'failed', 'rolled_back', 'cancelled')
                    LIMIT 1
                    """
                ).fetchone()
                if active is not None:
                    raise BootstrapConflictError("another activation is already in progress")
                if self._release_row(connection, activation.target_release_id) is None:
                    raise BootstrapConflictError("activation target release does not exist")
                if (
                    activation.previous_release_id is not None
                    and self._release_row(connection, activation.previous_release_id) is None
                ):
                    raise BootstrapConflictError("activation previous release does not exist")
                connection.execute(
                    """
                    INSERT INTO bootstrap_activations (
                        id, target_release_id, previous_release_id, status, revision,
                        created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        activation.id,
                        activation.target_release_id,
                        activation.previous_release_id,
                        activation.status.value,
                        activation.revision,
                        activation.created_at.isoformat(),
                        activation.updated_at.isoformat(),
                        self._json(activation),
                    ),
                )
                connection.commit()
                return activation
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise BootstrapConflictError("activation already exists") from exc
            except BaseException:
                connection.rollback()
                raise

    def get_activation(self, activation_id: str) -> ActivationRecord | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM bootstrap_activations WHERE id = ?",
                (self._identifier(activation_id, "activation_id"),),
            ).fetchone()
        return self._model(row["payload_json"], ActivationRecord) if row is not None else None

    def list_activations(
        self,
        *,
        statuses: Iterable[ActivationStatus] | None = None,
        limit: int = 100,
    ) -> list[ActivationRecord]:
        safe_limit = max(1, min(int(limit), 1_000))
        selected = tuple(statuses or ())
        with self._lock, closing(self._connect()) as connection:
            if selected:
                placeholders = ",".join("?" for _ in selected)
                rows = connection.execute(
                    f"""
                    SELECT payload_json FROM bootstrap_activations
                    WHERE status IN ({placeholders})
                    ORDER BY created_at DESC, id ASC LIMIT ?
                    """,
                    (*[status.value for status in selected], safe_limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT payload_json FROM bootstrap_activations
                    ORDER BY created_at DESC, id ASC LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        return [self._model(row["payload_json"], ActivationRecord) for row in rows]

    def incomplete_activations(self) -> list[ActivationRecord]:
        statuses = tuple(status for status in ActivationStatus if status not in TERMINAL_ACTIVATION_STATUSES)
        return self.list_activations(statuses=statuses, limit=1_000)

    def transition_activation(
        self,
        activation_id: str,
        *,
        expected: ActivationStatus | Iterable[ActivationStatus],
        target: ActivationStatus,
        lease_epoch: int | None = None,
        probation_ends_at: datetime | None = None,
        failure_code: str | None = None,
        failure_message: str | None = None,
    ) -> ActivationRecord:
        expected_set = (
            {expected} if isinstance(expected, ActivationStatus) else set(expected)
        )
        if not expected_set:
            raise ValueError("at least one expected activation status is required")
        if any(target not in _ALLOWED_TRANSITIONS[status] for status in expected_set):
            raise ValueError("activation transition is not allowed")
        if target in {ActivationStatus.FAILED, ActivationStatus.ROLLED_BACK}:
            if not failure_code or not failure_message:
                raise ValueError("terminal failure transition requires sanitized failure details")
        elif failure_code is not None or failure_message is not None:
            raise ValueError("failure details are only valid on failed rollback transitions")

        safe_id = self._identifier(activation_id, "activation_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT payload_json FROM bootstrap_activations WHERE id = ?", (safe_id,)
                ).fetchone()
                if row is None:
                    raise BootstrapConflictError("activation does not exist")
                current = self._model(row["payload_json"], ActivationRecord)
                if current.status not in expected_set:
                    raise BootstrapConflictError(
                        f"activation is {current.status.value}, expected "
                        f"{sorted(status.value for status in expected_set)}"
                    )
                now = max(self._now(), current.updated_at)
                updated = current.model_copy(
                    update={
                        "status": target,
                        "revision": current.revision + 1,
                        "lease_epoch": lease_epoch if lease_epoch is not None else current.lease_epoch,
                        "probation_ends_at": probation_ends_at,
                        "failure_code": failure_code,
                        "failure_message": failure_message,
                        "updated_at": now,
                    }
                )
                cursor = connection.execute(
                    """
                    UPDATE bootstrap_activations
                    SET status = ?, revision = ?, updated_at = ?, payload_json = ?
                    WHERE id = ? AND revision = ? AND status = ?
                    """,
                    (
                        updated.status.value,
                        updated.revision,
                        updated.updated_at.isoformat(),
                        self._json(updated),
                        current.id,
                        current.revision,
                        current.status.value,
                    ),
                )
                if cursor.rowcount != 1:
                    raise BootstrapConflictError("activation changed during transition")
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def get_state(self) -> BootstrapState:
        with self._lock, closing(self._connect()) as connection:
            return self._state_in_transaction(connection)

    def install_initial_lease(self, release_id: str) -> ReleaseLease:
        safe_release = self._identifier(release_id, "release_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._state_in_transaction(connection)
                if (
                    state.serving_release_id is not None
                    or state.last_known_good_release_id is not None
                ):
                    raise BootstrapConflictError("an initial release is already installed")
                if self._release_row(connection, safe_release) is None:
                    raise BootstrapConflictError("initial release does not exist")
                lease = self._issue_lease(connection, safe_release, activation_id=None)
                self._write_state(
                    connection,
                    state.model_copy(
                        update={
                            "serving_release_id": safe_release,
                            "last_known_good_release_id": safe_release,
                            "active_lease_epoch": lease.epoch,
                            "ingress_paused": True,
                            "safe_mode": False,
                            "updated_at": self._now(),
                        }
                    ),
                )
                connection.commit()
                return lease
            except BaseException:
                connection.rollback()
                raise

    def begin_cutover(self, activation: ActivationRecord) -> ReleaseLease:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._state_in_transaction(connection)
                if state.serving_release_id != activation.previous_release_id:
                    raise BootstrapConflictError("serving release changed before cutover")
                self._revoke_active_leases(connection)
                self._requeue_claimed_ingress(connection)
                lease = self._issue_lease(
                    connection,
                    activation.target_release_id,
                    activation_id=activation.id,
                )
                self._write_state(
                    connection,
                    state.model_copy(
                        update={
                            "active_activation_id": activation.id,
                            "active_lease_epoch": lease.epoch,
                            "ingress_paused": True,
                            "safe_mode": False,
                            "updated_at": self._now(),
                        }
                    ),
                )
                connection.commit()
                return lease
            except BaseException:
                connection.rollback()
                raise

    def commit_probation(
        self,
        activation_id: str,
        *,
        lease: ReleaseLease,
        probation_ends_at: datetime,
    ) -> ActivationRecord:
        """Atomically expose green traffic and record its probation state."""

        safe_id = self._identifier(activation_id, "activation_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._activation_in_transaction(connection, safe_id)
                if current.status is not ActivationStatus.VERIFYING:
                    raise BootstrapConflictError("activation is not ready for probation")
                self._require_lease_in_transaction(
                    connection, current.target_release_id, lease.epoch
                )
                state = self._state_in_transaction(connection)
                if state.active_activation_id != current.id:
                    raise BootstrapConflictError("another activation owns cutover")
                updated = current.model_copy(
                    update={
                        "status": ActivationStatus.PROBATION,
                        "revision": current.revision + 1,
                        "lease_epoch": lease.epoch,
                        "probation_ends_at": probation_ends_at,
                        "updated_at": self._now(),
                    }
                )
                self._write_activation(connection, current=current, updated=updated)
                self._write_state(
                    connection,
                    state.model_copy(
                        update={
                            "serving_release_id": current.target_release_id,
                            "previous_release_id": current.previous_release_id,
                            "active_lease_epoch": lease.epoch,
                            "ingress_paused": False,
                            "safe_mode": False,
                            "updated_at": self._now(),
                        }
                    ),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def commit_activation_success(
        self,
        activation_id: str,
        *,
        lease: ReleaseLease,
    ) -> ActivationRecord:
        """Atomically promote a probationary release to last-known-good."""

        safe_id = self._identifier(activation_id, "activation_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._activation_in_transaction(connection, safe_id)
                if current.status is not ActivationStatus.PROBATION:
                    raise BootstrapConflictError("activation is not in probation")
                self._require_lease_in_transaction(
                    connection, current.target_release_id, lease.epoch
                )
                state = self._state_in_transaction(connection)
                if state.active_activation_id != current.id:
                    raise BootstrapConflictError("another activation owns completion")
                updated = current.model_copy(
                    update={
                        "status": ActivationStatus.ACTIVE,
                        "revision": current.revision + 1,
                        "updated_at": self._now(),
                    }
                )
                self._write_activation(connection, current=current, updated=updated)
                self._write_state(
                    connection,
                    state.model_copy(
                        update={
                            "serving_release_id": current.target_release_id,
                            "last_known_good_release_id": current.target_release_id,
                            "previous_release_id": current.previous_release_id,
                            "active_lease_epoch": lease.epoch,
                            "ingress_paused": False,
                            "safe_mode": False,
                            "updated_at": self._now(),
                        }
                    ),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def begin_restore_lease(
        self,
        *,
        release_id: str,
        activation_id: str | None,
    ) -> ReleaseLease:
        """Fence failed green and issue a paused lease to last-known-good blue."""

        safe_release = self._identifier(release_id, "release_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if self._release_row(connection, safe_release) is None:
                    raise BootstrapConflictError("restore release does not exist")
                state = self._state_in_transaction(connection)
                self._revoke_active_leases(connection)
                self._requeue_claimed_ingress(connection)
                lease = self._issue_lease(
                    connection,
                    safe_release,
                    activation_id=activation_id,
                )
                self._write_state(
                    connection,
                    state.model_copy(
                        update={
                            "active_activation_id": activation_id,
                            "active_lease_epoch": lease.epoch,
                            "ingress_paused": True,
                            "safe_mode": False,
                            "updated_at": self._now(),
                        }
                    ),
                )
                connection.commit()
                return lease
            except BaseException:
                connection.rollback()
                raise

    def commit_rollback(
        self,
        activation_id: str,
        *,
        restored_release_id: str,
        failed_release_id: str,
        lease: ReleaseLease,
        failure_code: str,
        failure_message: str,
    ) -> ActivationRecord:
        """Atomically resume blue and make the failed activation terminal."""

        safe_id = self._identifier(activation_id, "activation_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                current = self._activation_in_transaction(connection, safe_id)
                if current.status is not ActivationStatus.ROLLING_BACK:
                    raise BootstrapConflictError("activation is not rolling back")
                self._require_lease_in_transaction(connection, restored_release_id, lease.epoch)
                state = self._state_in_transaction(connection)
                updated = current.model_copy(
                    update={
                        "status": ActivationStatus.ROLLED_BACK,
                        "revision": current.revision + 1,
                        "failure_code": failure_code,
                        "failure_message": failure_message,
                        "updated_at": self._now(),
                    }
                )
                self._write_activation(connection, current=current, updated=updated)
                self._write_state(
                    connection,
                    state.model_copy(
                        update={
                            "serving_release_id": restored_release_id,
                            "last_known_good_release_id": restored_release_id,
                            "previous_release_id": failed_release_id,
                            "active_activation_id": current.id,
                            "active_lease_epoch": lease.epoch,
                            "ingress_paused": False,
                            "safe_mode": False,
                            "updated_at": self._now(),
                        }
                    ),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def complete_recovery(
        self,
        *,
        release_id: str,
        lease: ReleaseLease,
        previous_release_id: str | None,
    ) -> BootstrapState:
        """Commit a restarted last-known-good process after bootstrap recovery."""

        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_lease_in_transaction(connection, release_id, lease.epoch)
                state = self._state_in_transaction(connection)
                updated = state.model_copy(
                    update={
                        "serving_release_id": release_id,
                        "last_known_good_release_id": release_id,
                        "previous_release_id": previous_release_id,
                        "active_activation_id": None,
                        "active_lease_epoch": lease.epoch,
                        "ingress_paused": False,
                        "safe_mode": False,
                        "updated_at": self._now(),
                    }
                )
                self._write_state(connection, updated)
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def resume_ingress(self) -> BootstrapState:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._state_in_transaction(connection)
                if state.serving_release_id is None or state.active_lease_epoch is None:
                    raise BootstrapConflictError("cannot resume ingress without a serving lease")
                updated = state.model_copy(
                    update={"ingress_paused": False, "safe_mode": False, "updated_at": self._now()}
                )
                self._write_state(connection, updated)
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def pause_ingress(self) -> BootstrapState:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._state_in_transaction(connection)
                if state.serving_release_id is None or state.active_lease_epoch is None:
                    raise BootstrapConflictError("cannot pause ingress without a serving lease")
                updated = state.model_copy(
                    update={"ingress_paused": True, "updated_at": self._now()}
                )
                self._write_state(connection, updated)
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def enter_safe_mode(self) -> BootstrapState:
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                state = self._state_in_transaction(connection)
                self._revoke_active_leases(connection)
                self._requeue_claimed_ingress(connection)
                updated = state.model_copy(
                    update={
                        "serving_release_id": None,
                        "active_activation_id": None,
                        "active_lease_epoch": None,
                        "ingress_paused": True,
                        "safe_mode": True,
                        "updated_at": self._now(),
                    }
                )
                self._write_state(connection, updated)
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def assert_active_lease(self, release_id: str, epoch: int) -> ReleaseLease:
        with self._lock, closing(self._connect()) as connection:
            return self._require_lease_in_transaction(
                connection,
                self._identifier(release_id, "release_id"),
                int(epoch),
            )

    def enqueue_ingress(self, envelope: IngressEnvelope) -> IngressEnvelope:
        if envelope.status != "pending" or envelope.claimed_epoch is not None:
            raise ValueError("new ingress must be pending and unclaimed")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """
                    SELECT payload_json FROM bootstrap_ingress
                    WHERE tenant_id = ? AND channel = ? AND idempotency_key = ?
                    """,
                    (envelope.tenant_id, envelope.channel, envelope.idempotency_key),
                ).fetchone()
                if existing is not None:
                    current = self._model(existing["payload_json"], IngressEnvelope)
                    if current.payload != envelope.payload or current.thread_id != envelope.thread_id:
                        raise BootstrapConflictError(
                            "ingress idempotency key is bound to another payload"
                        )
                    connection.commit()
                    return current
                connection.execute(
                    """
                    INSERT INTO bootstrap_ingress (
                        id, tenant_id, channel, idempotency_key, status, claimed_epoch,
                        created_at, updated_at, payload_json
                    ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)
                    """,
                    (
                        envelope.id,
                        envelope.tenant_id,
                        envelope.channel,
                        envelope.idempotency_key,
                        envelope.status,
                        envelope.created_at.isoformat(),
                        envelope.updated_at.isoformat(),
                        self._json(envelope),
                    ),
                )
                connection.commit()
                return envelope
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise BootstrapConflictError("ingress already exists") from exc
            except BaseException:
                connection.rollback()
                raise

    def claim_ingress(
        self,
        *,
        release_id: str,
        lease_epoch: int,
        limit: int = 10,
    ) -> list[IngressEnvelope]:
        safe_limit = max(1, min(int(limit), 100))
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_lease_in_transaction(connection, release_id, lease_epoch)
                state = self._state_in_transaction(connection)
                if state.ingress_paused:
                    connection.commit()
                    return []
                rows = connection.execute(
                    """
                    SELECT payload_json FROM bootstrap_ingress
                    WHERE status = 'pending'
                    ORDER BY created_at ASC, id ASC LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
                claimed: list[IngressEnvelope] = []
                now = self._now()
                for row in rows:
                    current = self._model(row["payload_json"], IngressEnvelope)
                    updated = current.model_copy(
                        update={
                            "status": "claimed",
                            "claimed_epoch": int(lease_epoch),
                            "attempt_count": current.attempt_count + 1,
                            "updated_at": now,
                        }
                    )
                    connection.execute(
                        """
                        UPDATE bootstrap_ingress
                        SET status = 'claimed', claimed_epoch = ?, updated_at = ?, payload_json = ?
                        WHERE id = ? AND status = 'pending'
                        """,
                        (lease_epoch, now.isoformat(), self._json(updated), current.id),
                    )
                    claimed.append(updated)
                connection.commit()
                return claimed
            except BaseException:
                connection.rollback()
                raise

    def get_ingress(self, ingress_id: str) -> IngressEnvelope | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT payload_json FROM bootstrap_ingress WHERE id = ?",
                (self._identifier(ingress_id, "ingress_id"),),
            ).fetchone()
        return self._model(row["payload_json"], IngressEnvelope) if row is not None else None

    def requeue_ingress_claim(
        self,
        ingress_id: str,
        *,
        release_id: str,
        lease_epoch: int,
    ) -> IngressEnvelope:
        """Release a failed delivery claim without permitting a stale worker to replay it."""

        safe_id = self._identifier(ingress_id, "ingress_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_lease_in_transaction(connection, release_id, lease_epoch)
                row = connection.execute(
                    "SELECT payload_json FROM bootstrap_ingress WHERE id = ?", (safe_id,)
                ).fetchone()
                if row is None:
                    raise BootstrapConflictError("ingress does not exist")
                current = self._model(row["payload_json"], IngressEnvelope)
                if current.status != "claimed" or current.claimed_epoch != int(lease_epoch):
                    raise LeaseFenceError("ingress claim belongs to another lease")
                updated = current.model_copy(
                    update={
                        "status": "pending",
                        "claimed_epoch": None,
                        "updated_at": self._now(),
                    }
                )
                connection.execute(
                    """
                    UPDATE bootstrap_ingress
                    SET status = 'pending', claimed_epoch = NULL, updated_at = ?, payload_json = ?
                    WHERE id = ? AND status = 'claimed' AND claimed_epoch = ?
                    """,
                    (updated.updated_at.isoformat(), self._json(updated), safe_id, lease_epoch),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def complete_ingress(
        self,
        ingress_id: str,
        *,
        release_id: str,
        lease_epoch: int,
    ) -> IngressEnvelope:
        safe_id = self._identifier(ingress_id, "ingress_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._require_lease_in_transaction(connection, release_id, lease_epoch)
                row = connection.execute(
                    "SELECT payload_json FROM bootstrap_ingress WHERE id = ?", (safe_id,)
                ).fetchone()
                if row is None:
                    raise BootstrapConflictError("ingress does not exist")
                current = self._model(row["payload_json"], IngressEnvelope)
                if current.status == "processed":
                    connection.commit()
                    return current
                if current.status != "claimed" or current.claimed_epoch != int(lease_epoch):
                    raise LeaseFenceError("ingress claim belongs to another lease")
                updated = current.model_copy(
                    update={"status": "processed", "updated_at": self._now()}
                )
                connection.execute(
                    """
                    UPDATE bootstrap_ingress
                    SET status = 'processed', updated_at = ?, payload_json = ?
                    WHERE id = ? AND status = 'claimed' AND claimed_epoch = ?
                    """,
                    (updated.updated_at.isoformat(), self._json(updated), safe_id, lease_epoch),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def append_outbox(self, event: OutboxEvent) -> OutboxEvent:
        if event.status != "pending":
            raise ValueError("new outbox event must be pending")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    "SELECT payload_json FROM bootstrap_outbox WHERE event_key = ?",
                    (event.event_key,),
                ).fetchone()
                if existing is not None:
                    current = self._model(existing["payload_json"], OutboxEvent)
                    if current.event_type != event.event_type or current.payload != event.payload:
                        raise BootstrapConflictError(
                            "outbox event key is bound to another payload"
                        )
                    connection.commit()
                    return current
                connection.execute(
                    """
                    INSERT INTO bootstrap_outbox (
                        id, event_key, event_type, status, attempt_count,
                        created_at, delivered_at, payload_json
                    ) VALUES (?, ?, ?, 'pending', 0, ?, NULL, ?)
                    """,
                    (
                        event.id,
                        event.event_key,
                        event.event_type,
                        event.created_at.isoformat(),
                        self._json(event),
                    ),
                )
                connection.commit()
                return event
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise BootstrapConflictError("outbox event already exists") from exc
            except BaseException:
                connection.rollback()
                raise

    def pending_outbox(self, *, limit: int = 100) -> list[OutboxEvent]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT payload_json FROM bootstrap_outbox
                WHERE status = 'pending'
                ORDER BY created_at ASC, id ASC LIMIT ?
                """,
                (max(1, min(int(limit), 1_000)),),
            ).fetchall()
        return [self._model(row["payload_json"], OutboxEvent) for row in rows]

    def mark_outbox_attempt(self, event_id: str, *, delivered: bool) -> OutboxEvent:
        safe_id = self._identifier(event_id, "event_id")
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT payload_json FROM bootstrap_outbox WHERE id = ?", (safe_id,)
                ).fetchone()
                if row is None:
                    raise BootstrapConflictError("outbox event does not exist")
                current = self._model(row["payload_json"], OutboxEvent)
                if current.status == "delivered":
                    connection.commit()
                    return current
                delivered_at = self._now() if delivered else None
                updated = current.model_copy(
                    update={
                        "status": "delivered" if delivered else "pending",
                        "attempt_count": current.attempt_count + 1,
                        "delivered_at": delivered_at,
                    }
                )
                connection.execute(
                    """
                    UPDATE bootstrap_outbox
                    SET status = ?, attempt_count = ?, delivered_at = ?, payload_json = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (
                        updated.status,
                        updated.attempt_count,
                        delivered_at.isoformat() if delivered_at is not None else None,
                        self._json(updated),
                        safe_id,
                    ),
                )
                connection.commit()
                return updated
            except BaseException:
                connection.rollback()
                raise

    def _issue_lease(
        self,
        connection: sqlite3.Connection,
        release_id: str,
        *,
        activation_id: str | None,
    ) -> ReleaseLease:
        now = self._now()
        cursor = connection.execute(
            """
            INSERT INTO bootstrap_leases (
                release_id, activation_id, status, issued_at, revoked_at, payload_json
            ) VALUES (?, ?, 'active', ?, NULL, '{}')
            """,
            (release_id, activation_id, now.isoformat()),
        )
        if cursor.lastrowid is None:
            raise BootstrapCorruptionError("SQLite did not allocate a lease epoch")
        lease = ReleaseLease(
            epoch=cursor.lastrowid,
            release_id=release_id,
            activation_id=activation_id,
            issued_at=now,
        )
        connection.execute(
            "UPDATE bootstrap_leases SET payload_json = ? WHERE epoch = ?",
            (self._json(lease), lease.epoch),
        )
        return lease

    def _revoke_active_leases(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT epoch, payload_json FROM bootstrap_leases WHERE status = 'active'"
        ).fetchall()
        now = self._now()
        for row in rows:
            current = self._model(row["payload_json"], ReleaseLease)
            updated = current.model_copy(update={"status": "revoked", "revoked_at": now})
            connection.execute(
                """
                UPDATE bootstrap_leases
                SET status = 'revoked', revoked_at = ?, payload_json = ?
                WHERE epoch = ? AND status = 'active'
                """,
                (now.isoformat(), self._json(updated), current.epoch),
            )

    def _requeue_claimed_ingress(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            "SELECT payload_json FROM bootstrap_ingress WHERE status = 'claimed'"
        ).fetchall()
        now = self._now()
        for row in rows:
            current = self._model(row["payload_json"], IngressEnvelope)
            updated = current.model_copy(
                update={"status": "pending", "claimed_epoch": None, "updated_at": now}
            )
            connection.execute(
                """
                UPDATE bootstrap_ingress
                SET status = 'pending', claimed_epoch = NULL, updated_at = ?, payload_json = ?
                WHERE id = ? AND status = 'claimed'
                """,
                (now.isoformat(), self._json(updated), current.id),
            )

    def _require_lease_in_transaction(
        self,
        connection: sqlite3.Connection,
        release_id: str,
        epoch: int,
    ) -> ReleaseLease:
        row = connection.execute(
            "SELECT payload_json FROM bootstrap_leases WHERE epoch = ?", (int(epoch),)
        ).fetchone()
        if row is None:
            raise LeaseFenceError("release lease does not exist")
        lease = self._model(row["payload_json"], ReleaseLease)
        state = self._state_in_transaction(connection)
        if (
            lease.status != "active"
            or lease.release_id != release_id
            or state.active_lease_epoch != lease.epoch
        ):
            raise LeaseFenceError("release lease is stale")
        return lease

    def _state_in_transaction(self, connection: sqlite3.Connection) -> BootstrapState:
        row = connection.execute(
            "SELECT payload_json FROM bootstrap_state WHERE singleton = 1"
        ).fetchone()
        if row is None:
            raise BootstrapCorruptionError("bootstrap state row is missing")
        return self._model(row["payload_json"], BootstrapState)

    def _activation_in_transaction(
        self,
        connection: sqlite3.Connection,
        activation_id: str,
    ) -> ActivationRecord:
        row = connection.execute(
            "SELECT payload_json FROM bootstrap_activations WHERE id = ?",
            (activation_id,),
        ).fetchone()
        if row is None:
            raise BootstrapConflictError("activation does not exist")
        return self._model(row["payload_json"], ActivationRecord)

    def _write_activation(
        self,
        connection: sqlite3.Connection,
        *,
        current: ActivationRecord,
        updated: ActivationRecord,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE bootstrap_activations
            SET status = ?, revision = ?, updated_at = ?, payload_json = ?
            WHERE id = ? AND revision = ? AND status = ?
            """,
            (
                updated.status.value,
                updated.revision,
                updated.updated_at.isoformat(),
                self._json(updated),
                current.id,
                current.revision,
                current.status.value,
            ),
        )
        if cursor.rowcount != 1:
            raise BootstrapConflictError("activation changed during update")

    def _write_state(self, connection: sqlite3.Connection, state: BootstrapState) -> None:
        connection.execute(
            "UPDATE bootstrap_state SET payload_json = ? WHERE singleton = 1",
            (self._json(state),),
        )

    @staticmethod
    def _release_row(connection: sqlite3.Connection, release_id: str) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT payload_json FROM bootstrap_releases WHERE id = ?", (release_id,)
        ).fetchone()
        return cast("sqlite3.Row | None", row)

    @staticmethod
    def _identifier(value: str, field: str) -> str:
        cleaned = str(value or "").strip()
        if not cleaned:
            raise ValueError(f"{field} is required")
        if len(cleaned) > 300:
            raise ValueError(f"{field} is too long")
        return cleaned

    @staticmethod
    def _json(value: BaseModel) -> str:
        return json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @staticmethod
    def _model(raw: Any, model: type[_ModelT]) -> _ModelT:
        try:
            return model.model_validate(json.loads(str(raw)))
        except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
            raise BootstrapCorruptionError(f"invalid persisted {model.__name__}") from exc


__all__ = [
    "BootstrapConflictError",
    "BootstrapCorruptionError",
    "BootstrapStore",
    "BootstrapStoreError",
    "LeaseFenceError",
]
