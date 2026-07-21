"""Tenant-scoped immutable AgentSpec and TriggerSpec SQLite stores."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.specs.models import AgentSpec, AgentSpecWrite, TriggerSpec, TriggerSpecWrite
from opentulpa.specs.protocol import AgentSpecRef


class SpecConflictError(RuntimeError):
    """A latest-revision or active-reference compare-and-swap failed."""


class SpecNotFoundError(KeyError):
    """A tenant-owned immutable revision does not exist."""


def _canonical(value: object) -> str:
    if not hasattr(value, "model_dump"):
        raise TypeError("spec value must be a Pydantic model")
    payload = value.model_dump(mode="json")  # type: ignore[attr-defined]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _digest(payload_json: str) -> str:
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


class AgentSpecStore:
    """Append AgentSpec revisions and activate them through an explicit CAS."""

    def __init__(
        self,
        db_path: Path,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock or (lambda: datetime.now(UTC))
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = connect_sqlite(self.db_path, wal=True)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("spec store clock must return an aware datetime")
        return value.astimezone(UTC)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agent_spec_revisions (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id, revision)
                );

                CREATE TABLE IF NOT EXISTS agent_spec_active_refs (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id),
                    FOREIGN KEY (tenant_id, id, revision)
                        REFERENCES agent_spec_revisions (tenant_id, id, revision)
                );

                CREATE INDEX IF NOT EXISTS idx_agent_specs_tenant_latest
                ON agent_spec_revisions (tenant_id, id, revision DESC);
                """
            )
            conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> AgentSpec:
        payload = json.loads(str(row["payload_json"]))
        return AgentSpec.model_validate(
            {
                **payload,
                "tenant_id": row["tenant_id"],
                "id": row["id"],
                "revision": row["revision"],
                "content_digest": row["content_digest"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
            }
        )

    def create_revision(
        self,
        *,
        tenant_id: str,
        spec_id: str,
        write: AgentSpecWrite,
        expected_revision: int | None,
        created_by: str,
    ) -> AgentSpec:
        payload_json = _canonical(write)
        content_digest = _digest(payload_json)
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                """
                SELECT * FROM agent_spec_revisions
                WHERE tenant_id = ? AND id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, spec_id),
            ).fetchone()
            if latest is None:
                if expected_revision is not None:
                    conn.rollback()
                    raise SpecConflictError("AgentSpec does not exist at the expected revision")
                revision = 1
            else:
                current = self._row(latest)
                if current.content_digest == content_digest:
                    conn.rollback()
                    return current
                if expected_revision != current.revision:
                    conn.rollback()
                    raise SpecConflictError(
                        f"AgentSpec revision is {current.revision}, expected {expected_revision}"
                    )
                revision = current.revision + 1
            spec = AgentSpec(
                tenant_id=tenant_id,
                id=spec_id,
                revision=revision,
                content_digest=content_digest,
                created_at=now,
                created_by=created_by,
                **write.model_dump(),
            )
            conn.execute(
                """
                INSERT INTO agent_spec_revisions (
                    tenant_id, id, revision, payload_json, content_digest,
                    created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    spec.tenant_id,
                    spec.id,
                    spec.revision,
                    payload_json,
                    spec.content_digest,
                    spec.created_at.isoformat(),
                    spec.created_by,
                ),
            )
            conn.commit()
            return spec

    def get_revision(self, ref: AgentSpecRef) -> AgentSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_spec_revisions
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (ref.tenant_id, ref.spec_id, ref.revision),
            ).fetchone()
        return self._row(row) if row is not None else None

    def get_latest(self, *, tenant_id: str, spec_id: str) -> AgentSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_spec_revisions
                WHERE tenant_id = ? AND id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, spec_id),
            ).fetchone()
        return self._row(row) if row is not None else None

    def get_active_ref(self, *, tenant_id: str, spec_id: str) -> AgentSpecRef | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT revision FROM agent_spec_active_refs
                WHERE tenant_id = ? AND id = ?
                """,
                (tenant_id, spec_id),
            ).fetchone()
        if row is None:
            return None
        return AgentSpecRef(tenant_id=tenant_id, spec_id=spec_id, revision=row["revision"])

    def get_active(self, *, tenant_id: str, spec_id: str) -> AgentSpec | None:
        ref = self.get_active_ref(tenant_id=tenant_id, spec_id=spec_id)
        return self.get_revision(ref) if ref is not None else None

    def activate(
        self,
        ref: AgentSpecRef,
        *,
        expected_active_revision: int | None,
        updated_by: str,
    ) -> AgentSpecRef:
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            target = conn.execute(
                """
                SELECT 1 FROM agent_spec_revisions
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (ref.tenant_id, ref.spec_id, ref.revision),
            ).fetchone()
            if target is None:
                conn.rollback()
                raise SpecNotFoundError(ref)
            active = conn.execute(
                """
                SELECT revision FROM agent_spec_active_refs
                WHERE tenant_id = ? AND id = ?
                """,
                (ref.tenant_id, ref.spec_id),
            ).fetchone()
            if active is None:
                if expected_active_revision is not None:
                    conn.rollback()
                    raise SpecConflictError("AgentSpec has no active revision")
                conn.execute(
                    """
                    INSERT INTO agent_spec_active_refs (
                        tenant_id, id, revision, updated_at, updated_by
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ref.tenant_id,
                        ref.spec_id,
                        ref.revision,
                        now.isoformat(),
                        updated_by,
                    ),
                )
            else:
                current = int(active["revision"])
                if current == ref.revision:
                    conn.rollback()
                    return ref
                if expected_active_revision != current:
                    conn.rollback()
                    raise SpecConflictError(
                        f"active AgentSpec revision is {current}, "
                        f"expected {expected_active_revision}"
                    )
                cursor = conn.execute(
                    """
                    UPDATE agent_spec_active_refs
                    SET revision = ?, updated_at = ?, updated_by = ?
                    WHERE tenant_id = ? AND id = ? AND revision = ?
                    """,
                    (
                        ref.revision,
                        now.isoformat(),
                        updated_by,
                        ref.tenant_id,
                        ref.spec_id,
                        current,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise SpecConflictError("active AgentSpec changed during activation")
            conn.commit()
        return ref

    def list_active(self, *, tenant_id: str) -> list[AgentSpec]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT revisions.* FROM agent_spec_active_refs AS active
                JOIN agent_spec_revisions AS revisions
                  ON revisions.tenant_id = active.tenant_id
                 AND revisions.id = active.id
                 AND revisions.revision = active.revision
                WHERE active.tenant_id = ?
                ORDER BY revisions.id ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_latest(self, *, tenant_id: str) -> list[AgentSpec]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT revisions.* FROM agent_spec_revisions AS revisions
                JOIN (
                    SELECT id, MAX(revision) AS revision
                    FROM agent_spec_revisions
                    WHERE tenant_id = ?
                    GROUP BY id
                ) AS latest
                  ON latest.id = revisions.id
                 AND latest.revision = revisions.revision
                WHERE revisions.tenant_id = ?
                ORDER BY revisions.id ASC
                """,
                (tenant_id, tenant_id),
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_revisions(self, *, tenant_id: str, spec_id: str) -> list[AgentSpec]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM agent_spec_revisions
                WHERE tenant_id = ? AND id = ?
                ORDER BY revision DESC
                """,
                (tenant_id, spec_id),
            ).fetchall()
        return [self._row(row) for row in rows]

    def previous_revision(
        self,
        *,
        tenant_id: str,
        spec_id: str,
        before_revision: int,
    ) -> AgentSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM agent_spec_revisions
                WHERE tenant_id = ? AND id = ? AND revision < ?
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, spec_id, before_revision),
            ).fetchone()
        return self._row(row) if row is not None else None

    def deactivate(
        self,
        *,
        tenant_id: str,
        spec_id: str,
        expected_active_revision: int,
    ) -> AgentSpecRef:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                """
                SELECT revision FROM agent_spec_active_refs
                WHERE tenant_id = ? AND id = ?
                """,
                (tenant_id, spec_id),
            ).fetchone()
            if active is None:
                conn.rollback()
                raise SpecNotFoundError(spec_id)
            current = int(active["revision"])
            if current != expected_active_revision:
                conn.rollback()
                raise SpecConflictError(
                    f"active AgentSpec revision is {current}, "
                    f"expected {expected_active_revision}"
                )
            cursor = conn.execute(
                """
                DELETE FROM agent_spec_active_refs
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (tenant_id, spec_id, current),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise SpecConflictError("active AgentSpec changed during deactivation")
            conn.commit()
        return AgentSpecRef(tenant_id=tenant_id, spec_id=spec_id, revision=current)


class TriggerSpecStore:
    """Append TriggerSpec revisions and bind each one to an exact AgentSpec."""

    def __init__(
        self,
        db_path: Path,
        *,
        agent_specs: AgentSpecStore,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._agent_specs = agent_specs
        self._clock = clock or (lambda: datetime.now(UTC))
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = connect_sqlite(self.db_path, wal=True)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trigger store clock must return an aware datetime")
        return value.astimezone(UTC)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trigger_spec_revisions (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    payload_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id, revision)
                );

                CREATE TABLE IF NOT EXISTS trigger_spec_active_refs (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    updated_at TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id),
                    FOREIGN KEY (tenant_id, id, revision)
                        REFERENCES trigger_spec_revisions (tenant_id, id, revision)
                );

                CREATE INDEX IF NOT EXISTS idx_trigger_specs_tenant_latest
                ON trigger_spec_revisions (tenant_id, id, revision DESC);
                """
            )
            conn.commit()

    @staticmethod
    def _row(row: sqlite3.Row) -> TriggerSpec:
        payload = json.loads(str(row["payload_json"]))
        return TriggerSpec.model_validate(
            {
                **payload,
                "tenant_id": row["tenant_id"],
                "id": row["id"],
                "revision": row["revision"],
                "content_digest": row["content_digest"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
            }
        )

    def _validate_target(self, *, tenant_id: str, write: TriggerSpecWrite) -> None:
        if write.agent_spec.tenant_id != tenant_id:
            raise ValueError("trigger and AgentSpec must belong to the same tenant")
        target = self._agent_specs.get_revision(write.agent_spec)
        if target is None:
            raise SpecNotFoundError(write.agent_spec)
        if target.isolation != write.exposure:
            raise ValueError("trigger exposure must match the target AgentSpec isolation")

    def create_revision(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        write: TriggerSpecWrite,
        expected_revision: int | None,
        created_by: str,
    ) -> TriggerSpec:
        self._validate_target(tenant_id=tenant_id, write=write)
        payload_json = _canonical(write)
        content_digest = _digest(payload_json)
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            latest = conn.execute(
                """
                SELECT * FROM trigger_spec_revisions
                WHERE tenant_id = ? AND id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, trigger_id),
            ).fetchone()
            if latest is None:
                if expected_revision is not None:
                    conn.rollback()
                    raise SpecConflictError("TriggerSpec does not exist at the expected revision")
                revision = 1
            else:
                current = self._row(latest)
                if current.content_digest == content_digest:
                    conn.rollback()
                    return current
                if expected_revision != current.revision:
                    conn.rollback()
                    raise SpecConflictError(
                        f"TriggerSpec revision is {current.revision}, expected {expected_revision}"
                    )
                revision = current.revision + 1
            trigger = TriggerSpec(
                tenant_id=tenant_id,
                id=trigger_id,
                revision=revision,
                content_digest=content_digest,
                created_at=now,
                created_by=created_by,
                **write.model_dump(),
            )
            conn.execute(
                """
                INSERT INTO trigger_spec_revisions (
                    tenant_id, id, revision, payload_json, content_digest,
                    created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trigger.tenant_id,
                    trigger.id,
                    trigger.revision,
                    payload_json,
                    trigger.content_digest,
                    trigger.created_at.isoformat(),
                    trigger.created_by,
                ),
            )
            conn.commit()
            return trigger

    def get_latest(self, *, tenant_id: str, trigger_id: str) -> TriggerSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM trigger_spec_revisions
                WHERE tenant_id = ? AND id = ?
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, trigger_id),
            ).fetchone()
        return self._row(row) if row is not None else None

    def get_revision(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        revision: int,
    ) -> TriggerSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM trigger_spec_revisions
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (tenant_id, trigger_id, revision),
            ).fetchone()
        return self._row(row) if row is not None else None

    def get_active(self, *, tenant_id: str, trigger_id: str) -> TriggerSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT revisions.* FROM trigger_spec_active_refs AS active
                JOIN trigger_spec_revisions AS revisions
                  ON revisions.tenant_id = active.tenant_id
                 AND revisions.id = active.id
                 AND revisions.revision = active.revision
                WHERE active.tenant_id = ? AND active.id = ?
                """,
                (tenant_id, trigger_id),
            ).fetchone()
        return self._row(row) if row is not None else None

    def activate(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        revision: int,
        expected_active_revision: int | None,
        updated_by: str,
    ) -> TriggerSpec:
        now = self._now()
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            target_row = conn.execute(
                """
                SELECT * FROM trigger_spec_revisions
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (tenant_id, trigger_id, revision),
            ).fetchone()
            if target_row is None:
                conn.rollback()
                raise SpecNotFoundError(trigger_id)
            target = self._row(target_row)
            active = conn.execute(
                """
                SELECT revision FROM trigger_spec_active_refs
                WHERE tenant_id = ? AND id = ?
                """,
                (tenant_id, trigger_id),
            ).fetchone()
            if active is None:
                if expected_active_revision is not None:
                    conn.rollback()
                    raise SpecConflictError("TriggerSpec has no active revision")
                conn.execute(
                    """
                    INSERT INTO trigger_spec_active_refs (
                        tenant_id, id, revision, updated_at, updated_by
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (tenant_id, trigger_id, revision, now.isoformat(), updated_by),
                )
            else:
                current = int(active["revision"])
                if current == revision:
                    conn.rollback()
                    return target
                if expected_active_revision != current:
                    conn.rollback()
                    raise SpecConflictError(
                        f"active TriggerSpec revision is {current}, "
                        f"expected {expected_active_revision}"
                    )
                cursor = conn.execute(
                    """
                    UPDATE trigger_spec_active_refs
                    SET revision = ?, updated_at = ?, updated_by = ?
                    WHERE tenant_id = ? AND id = ? AND revision = ?
                    """,
                    (
                        revision,
                        now.isoformat(),
                        updated_by,
                        tenant_id,
                        trigger_id,
                        current,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise SpecConflictError("active TriggerSpec changed during activation")
            conn.commit()
            return target

    def list_active(self, *, tenant_id: str) -> list[TriggerSpec]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT revisions.* FROM trigger_spec_active_refs AS active
                JOIN trigger_spec_revisions AS revisions
                  ON revisions.tenant_id = active.tenant_id
                 AND revisions.id = active.id
                 AND revisions.revision = active.revision
                WHERE active.tenant_id = ?
                ORDER BY revisions.id ASC
                """,
                (tenant_id,),
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_tenant_ids(self) -> list[str]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT tenant_id FROM trigger_spec_active_refs
                ORDER BY tenant_id ASC
                """
            ).fetchall()
        return [str(row["tenant_id"]) for row in rows]

    def get_active_revision(self, *, tenant_id: str, trigger_id: str) -> int | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT revision FROM trigger_spec_active_refs
                WHERE tenant_id = ? AND id = ?
                """,
                (tenant_id, trigger_id),
            ).fetchone()
        return int(row["revision"]) if row is not None else None

    def list_latest(self, *, tenant_id: str) -> list[TriggerSpec]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT revisions.* FROM trigger_spec_revisions AS revisions
                JOIN (
                    SELECT id, MAX(revision) AS revision
                    FROM trigger_spec_revisions
                    WHERE tenant_id = ?
                    GROUP BY id
                ) AS latest
                  ON latest.id = revisions.id
                 AND latest.revision = revisions.revision
                WHERE revisions.tenant_id = ?
                ORDER BY revisions.id ASC
                """,
                (tenant_id, tenant_id),
            ).fetchall()
        return [self._row(row) for row in rows]

    def list_revisions(self, *, tenant_id: str, trigger_id: str) -> list[TriggerSpec]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM trigger_spec_revisions
                WHERE tenant_id = ? AND id = ?
                ORDER BY revision DESC
                """,
                (tenant_id, trigger_id),
            ).fetchall()
        return [self._row(row) for row in rows]

    def previous_revision(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        before_revision: int,
    ) -> TriggerSpec | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM trigger_spec_revisions
                WHERE tenant_id = ? AND id = ? AND revision < ?
                ORDER BY revision DESC LIMIT 1
                """,
                (tenant_id, trigger_id, before_revision),
            ).fetchone()
        return self._row(row) if row is not None else None

    def deactivate(
        self,
        *,
        tenant_id: str,
        trigger_id: str,
        expected_active_revision: int,
    ) -> int:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            active = conn.execute(
                """
                SELECT revision FROM trigger_spec_active_refs
                WHERE tenant_id = ? AND id = ?
                """,
                (tenant_id, trigger_id),
            ).fetchone()
            if active is None:
                conn.rollback()
                raise SpecNotFoundError(trigger_id)
            current = int(active["revision"])
            if current != expected_active_revision:
                conn.rollback()
                raise SpecConflictError(
                    f"active TriggerSpec revision is {current}, "
                    f"expected {expected_active_revision}"
                )
            cursor = conn.execute(
                """
                DELETE FROM trigger_spec_active_refs
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (tenant_id, trigger_id, current),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise SpecConflictError("active TriggerSpec changed during deactivation")
            conn.commit()
        return current


__all__ = [
    "AgentSpecStore",
    "SpecConflictError",
    "SpecNotFoundError",
    "TriggerSpecStore",
]
