"""Immutable capability revisions with compare-and-swap activation."""

from __future__ import annotations

import builtins
import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from opentulpa.capabilities.models import (
    CapabilityActivation,
    CapabilityActivationState,
    CapabilityManifest,
    CapabilitySecretBinding,
    CapabilityTestCheck,
    CapabilityTestResult,
    CapabilityTestStatus,
)
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.specs import AgentRunBinding


class CapabilityRevisionError(RuntimeError):
    """Base error for capability revision storage."""


class CapabilityRevisionConflictError(CapabilityRevisionError):
    """A revision or activation generation changed concurrently."""


class CapabilityRevisionNotFoundError(CapabilityRevisionError):
    """The requested immutable capability revision does not exist."""


class CapabilityRevisionCorruptionError(CapabilityRevisionError):
    """Persisted capability data no longer matches its digest or schema."""


class CapabilityRevisionStore:
    """SQLite revision archive; payload rows are append-only."""

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
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS capability_revisions (
                    namespace TEXT NOT NULL,
                    capability_name TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    manifest_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, capability_name, revision)
                );

                CREATE TABLE IF NOT EXISTS capability_activations (
                    namespace TEXT NOT NULL,
                    capability_name TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    manifest_digest TEXT NOT NULL,
                    generation INTEGER NOT NULL CHECK (generation >= 1),
                    activated_at TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    secret_handles_json TEXT NOT NULL DEFAULT '{}',
                    secret_bindings_json TEXT NOT NULL DEFAULT '{}',
                    agent_binding_json TEXT,
                    lifecycle_state TEXT NOT NULL DEFAULT 'active' CHECK (
                        lifecycle_state IN ('active', 'deactivating', 'inactive')
                    ),
                    PRIMARY KEY (namespace, capability_name),
                    FOREIGN KEY (namespace, capability_name, revision)
                        REFERENCES capability_revisions (
                            namespace, capability_name, revision
                        ) ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS capability_test_results (
                    namespace TEXT NOT NULL,
                    capability_name TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    manifest_digest TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('passed', 'failed')),
                    checks_json TEXT NOT NULL,
                    tested_at TEXT NOT NULL,
                    PRIMARY KEY (namespace, capability_name, revision),
                    FOREIGN KEY (namespace, capability_name, revision)
                        REFERENCES capability_revisions (
                            namespace, capability_name, revision
                        ) ON DELETE CASCADE
                );
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(capability_activations)").fetchall()
            }
            if "config_json" not in columns:
                conn.execute(
                    "ALTER TABLE capability_activations "
                    "ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "secret_handles_json" not in columns:
                conn.execute(
                    "ALTER TABLE capability_activations "
                    "ADD COLUMN secret_handles_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "secret_bindings_json" not in columns:
                conn.execute(
                    "ALTER TABLE capability_activations "
                    "ADD COLUMN secret_bindings_json TEXT NOT NULL DEFAULT '{}'"
                )
            if "agent_binding_json" not in columns:
                conn.execute(
                    "ALTER TABLE capability_activations ADD COLUMN agent_binding_json TEXT"
                )
            if "lifecycle_state" not in columns:
                conn.execute(
                    "ALTER TABLE capability_activations "
                    "ADD COLUMN lifecycle_state TEXT NOT NULL DEFAULT 'active' CHECK ("
                    "lifecycle_state IN ('active', 'deactivating', 'inactive'))"
                )
            conn.commit()

    @staticmethod
    def _namespace(value: str) -> str:
        safe = str(value or "").strip()
        if not safe or len(safe) > 200:
            raise ValueError("capability namespace must contain 1 to 200 characters")
        return safe

    @staticmethod
    def _serialize(manifest: CapabilityManifest) -> str:
        return json.dumps(
            manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _manifest_from_row(row: sqlite3.Row) -> CapabilityManifest:
        try:
            manifest = CapabilityManifest.model_validate_json(str(row["payload_json"]))
        except ValidationError as exc:
            raise CapabilityRevisionCorruptionError(
                "stored capability manifest is invalid"
            ) from exc
        if manifest.content_digest != str(row["manifest_digest"]):
            raise CapabilityRevisionCorruptionError(
                "stored capability manifest digest does not match"
            )
        return manifest

    @staticmethod
    def _activation_from_row(row: sqlite3.Row) -> CapabilityActivation:
        try:
            return CapabilityActivation(
                namespace=str(row["namespace"]),
                capability_name=str(row["capability_name"]),
                revision=int(row["revision"]),
                manifest_digest=str(row["manifest_digest"]),
                generation=int(row["generation"]),
                activated_at=str(row["activated_at"]),
                config=dict(json.loads(str(row["config_json"]))),
                secret_handles=dict(json.loads(str(row["secret_handles_json"]))),
                secret_bindings={
                    str(name): CapabilitySecretBinding.model_validate(binding)
                    for name, binding in dict(json.loads(str(row["secret_bindings_json"]))).items()
                },
                agent_binding=(
                    AgentRunBinding.model_validate_json(str(row["agent_binding_json"]))
                    if row["agent_binding_json"] is not None
                    else None
                ),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise CapabilityRevisionCorruptionError(
                "stored capability activation is invalid"
            ) from exc

    def append(
        self,
        *,
        namespace: str,
        manifest: CapabilityManifest,
        expected_latest_revision: int | None,
    ) -> CapabilityManifest:
        """Append exactly the next revision; existing rows are never replaced."""

        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT MAX(revision) AS revision
                FROM capability_revisions
                WHERE namespace = ? AND capability_name = ?
                """,
                (safe_namespace, manifest.name),
            ).fetchone()
            current = (
                int(row["revision"]) if row is not None and row["revision"] is not None else None
            )
            if current != expected_latest_revision:
                conn.rollback()
                raise CapabilityRevisionConflictError(
                    f"expected latest revision {expected_latest_revision!r}, found {current!r}"
                )
            expected_new = 1 if current is None else current + 1
            if manifest.revision != expected_new:
                conn.rollback()
                raise CapabilityRevisionConflictError(
                    f"new manifest must be revision {expected_new}, got {manifest.revision}"
                )
            try:
                conn.execute(
                    """
                    INSERT INTO capability_revisions (
                        namespace, capability_name, revision, manifest_digest,
                        payload_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        safe_namespace,
                        manifest.name,
                        manifest.revision,
                        manifest.content_digest,
                        self._serialize(manifest),
                        self._clock().astimezone(UTC).isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise CapabilityRevisionConflictError("capability revision already exists") from exc
            conn.commit()
        return manifest

    def get(
        self,
        *,
        namespace: str,
        capability_name: str,
        revision: int,
    ) -> CapabilityManifest | None:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM capability_revisions
                WHERE namespace = ? AND capability_name = ? AND revision = ?
                """,
                (safe_namespace, capability_name, revision),
            ).fetchone()
        return self._manifest_from_row(row) if row is not None else None

    def list(
        self,
        *,
        namespace: str,
        capability_name: str,
    ) -> list[CapabilityManifest]:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM capability_revisions
                WHERE namespace = ? AND capability_name = ?
                ORDER BY revision ASC
                """,
                (safe_namespace, capability_name),
            ).fetchall()
        return [self._manifest_from_row(row) for row in rows]

    def list_latest(self, *, namespace: str) -> builtins.list[CapabilityManifest]:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT revisions.* FROM capability_revisions AS revisions
                JOIN (
                    SELECT capability_name, MAX(revision) AS revision
                    FROM capability_revisions
                    WHERE namespace = ?
                    GROUP BY capability_name
                ) AS latest
                  ON latest.capability_name = revisions.capability_name
                 AND latest.revision = revisions.revision
                WHERE revisions.namespace = ?
                ORDER BY revisions.capability_name ASC
                """,
                (safe_namespace, safe_namespace),
            ).fetchall()
        return [self._manifest_from_row(row) for row in rows]

    def active(
        self,
        *,
        namespace: str,
        capability_name: str,
    ) -> CapabilityActivation | None:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND capability_name = ? AND lifecycle_state = ?
                """,
                (
                    safe_namespace,
                    capability_name,
                    CapabilityActivationState.ACTIVE.value,
                ),
            ).fetchone()
        return self._activation_from_row(row) if row is not None else None

    def deactivating(
        self,
        *,
        namespace: str,
        capability_name: str,
    ) -> CapabilityActivation | None:
        """Return the exact generation currently committed to shutdown."""

        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND capability_name = ? AND lifecycle_state = ?
                """,
                (
                    safe_namespace,
                    capability_name,
                    CapabilityActivationState.DEACTIVATING.value,
                ),
            ).fetchone()
        return self._activation_from_row(row) if row is not None else None

    def inactive(
        self,
        *,
        namespace: str,
        capability_name: str,
    ) -> CapabilityActivation | None:
        """Return the last completed shutdown tombstone, if any."""

        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND capability_name = ? AND lifecycle_state = ?
                """,
                (
                    safe_namespace,
                    capability_name,
                    CapabilityActivationState.INACTIVE.value,
                ),
            ).fetchone()
        return self._activation_from_row(row) if row is not None else None

    def list_active(self, *, namespace: str) -> builtins.list[CapabilityActivation]:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND lifecycle_state = ?
                ORDER BY capability_name ASC
                """,
                (safe_namespace, CapabilityActivationState.ACTIVE.value),
            ).fetchall()
        return [self._activation_from_row(row) for row in rows]

    def list_all_active(self) -> builtins.list[CapabilityActivation]:
        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE lifecycle_state = ?
                ORDER BY namespace ASC, capability_name ASC
                """,
                (CapabilityActivationState.ACTIVE.value,),
            ).fetchall()
        return [self._activation_from_row(row) for row in rows]

    def list_all_deactivating(self) -> builtins.list[CapabilityActivation]:
        """List durable shutdown transitions for startup reconciliation."""

        with closing(self._conn()) as conn:
            rows = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE lifecycle_state = ?
                ORDER BY namespace ASC, capability_name ASC
                """,
                (CapabilityActivationState.DEACTIVATING.value,),
            ).fetchall()
        return [self._activation_from_row(row) for row in rows]

    def next_generation(self, *, namespace: str, capability_name: str) -> int:
        """Return the generation reserved by the next successful activation."""

        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT generation, lifecycle_state FROM capability_activations
                WHERE namespace = ? AND capability_name = ?
                """,
                (safe_namespace, capability_name),
            ).fetchone()
        if row is None:
            return 1
        if str(row["lifecycle_state"]) == CapabilityActivationState.DEACTIVATING.value:
            raise CapabilityRevisionConflictError("capability deactivation is in progress")
        return int(row["generation"]) + 1

    def activate(
        self,
        *,
        namespace: str,
        capability_name: str,
        revision: int,
        expected_generation: int | None,
        config: dict[str, object] | None = None,
        secret_handles: dict[str, str] | None = None,
        secret_bindings: dict[str, CapabilitySecretBinding] | None = None,
        agent_binding: AgentRunBinding | None = None,
    ) -> CapabilityActivation:
        """CAS the active pointer; activating the current revision is idempotent."""

        safe_namespace = self._namespace(namespace)
        if agent_binding is not None and agent_binding.agent_spec.tenant_id != safe_namespace:
            raise ValueError("capability agent binding belongs to a different tenant")
        safe_config = dict(config or {})
        safe_secret_handles = dict(secret_handles or {})
        safe_secret_bindings = dict(secret_bindings or {})
        try:
            config_json = json.dumps(
                safe_config,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handles_json = json.dumps(
                safe_secret_handles,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            bindings_json = json.dumps(
                {
                    name: binding.model_dump(mode="json")
                    for name, binding in safe_secret_bindings.items()
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            agent_binding_json = (
                json.dumps(
                    agent_binding.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                if agent_binding is not None
                else None
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("capability activation values must be JSON serializable") from exc
        with closing(self._conn()) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN IMMEDIATE")
            manifest_row = conn.execute(
                """
                SELECT * FROM capability_revisions
                WHERE namespace = ? AND capability_name = ? AND revision = ?
                """,
                (safe_namespace, capability_name, revision),
            ).fetchone()
            if manifest_row is None:
                conn.rollback()
                raise CapabilityRevisionNotFoundError(
                    f"capability {capability_name!r} revision {revision} does not exist"
                )
            manifest = self._manifest_from_row(manifest_row)
            current_row = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND capability_name = ?
                """,
                (safe_namespace, capability_name),
            ).fetchone()
            current = self._activation_from_row(current_row) if current_row is not None else None
            lifecycle_state = (
                str(current_row["lifecycle_state"]) if current_row is not None else None
            )
            if lifecycle_state == CapabilityActivationState.DEACTIVATING.value:
                conn.rollback()
                raise CapabilityRevisionConflictError("capability deactivation is in progress")
            current_generation = (
                current.generation
                if current is not None and lifecycle_state == CapabilityActivationState.ACTIVE.value
                else None
            )
            if current_generation != expected_generation:
                conn.rollback()
                raise CapabilityRevisionConflictError(
                    f"expected activation generation {expected_generation!r}, "
                    f"found {current_generation!r}"
                )
            if (
                current is not None
                and lifecycle_state == CapabilityActivationState.ACTIVE.value
                and current.revision == revision
                and current.config == safe_config
                and current.secret_handles == safe_secret_handles
                and current.secret_bindings == safe_secret_bindings
                and current.agent_binding == agent_binding
            ):
                conn.rollback()
                return current

            generation = 1 if current is None else current.generation + 1
            activated_at = self._clock().astimezone(UTC).isoformat()
            if current is None:
                conn.execute(
                    """
                    INSERT INTO capability_activations (
                        namespace, capability_name, revision, manifest_digest,
                        generation, activated_at, config_json, secret_handles_json,
                        secret_bindings_json, agent_binding_json, lifecycle_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        safe_namespace,
                        capability_name,
                        revision,
                        manifest.content_digest,
                        generation,
                        activated_at,
                        config_json,
                        handles_json,
                        bindings_json,
                        agent_binding_json,
                        CapabilityActivationState.ACTIVE.value,
                    ),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE capability_activations
                    SET revision = ?, manifest_digest = ?, generation = ?, activated_at = ?,
                        config_json = ?, secret_handles_json = ?, secret_bindings_json = ?,
                        agent_binding_json = ?, lifecycle_state = ?
                    WHERE namespace = ? AND capability_name = ? AND generation = ?
                        AND lifecycle_state = ?
                    """,
                    (
                        revision,
                        manifest.content_digest,
                        generation,
                        activated_at,
                        config_json,
                        handles_json,
                        bindings_json,
                        agent_binding_json,
                        CapabilityActivationState.ACTIVE.value,
                        safe_namespace,
                        capability_name,
                        current.generation,
                        lifecycle_state,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise CapabilityRevisionConflictError(
                        "capability activation changed concurrently"
                    )
            conn.commit()
        return CapabilityActivation(
            namespace=safe_namespace,
            capability_name=manifest.name,
            revision=manifest.revision,
            manifest_digest=manifest.content_digest,
            generation=generation,
            activated_at=activated_at,
            config=safe_config,
            secret_handles=safe_secret_handles,
            secret_bindings=safe_secret_bindings,
            agent_binding=agent_binding,
        )

    def begin_deactivation(
        self,
        *,
        namespace: str,
        capability_name: str,
        expected_generation: int,
    ) -> CapabilityActivation:
        """Hide an exact active generation before any runtime side effect."""

        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND capability_name = ?
                """,
                (safe_namespace, capability_name),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise CapabilityRevisionNotFoundError(
                    f"capability {capability_name!r} is not active"
                )
            active = self._activation_from_row(row)
            if active.generation != expected_generation:
                conn.rollback()
                raise CapabilityRevisionConflictError(
                    f"expected activation generation {expected_generation!r}, "
                    f"found {active.generation!r}"
                )
            lifecycle_state = str(row["lifecycle_state"])
            if lifecycle_state == CapabilityActivationState.DEACTIVATING.value:
                conn.rollback()
                return active
            if lifecycle_state != CapabilityActivationState.ACTIVE.value:
                conn.rollback()
                raise CapabilityRevisionNotFoundError(
                    f"capability {capability_name!r} is not active"
                )
            cursor = conn.execute(
                """
                UPDATE capability_activations
                SET lifecycle_state = ?
                WHERE namespace = ? AND capability_name = ? AND generation = ?
                    AND lifecycle_state = ?
                """,
                (
                    CapabilityActivationState.DEACTIVATING.value,
                    safe_namespace,
                    capability_name,
                    expected_generation,
                    CapabilityActivationState.ACTIVE.value,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise CapabilityRevisionConflictError("capability activation changed concurrently")
            conn.commit()
        return active

    def cancel_deactivation(
        self,
        *,
        namespace: str,
        capability_name: str,
        expected_generation: int,
    ) -> CapabilityActivation:
        """Restore visibility only after the exact generation is operational again."""

        return self._change_deactivation_state(
            namespace=namespace,
            capability_name=capability_name,
            expected_generation=expected_generation,
            target=CapabilityActivationState.ACTIVE,
        )

    def deactivate(
        self,
        *,
        namespace: str,
        capability_name: str,
        expected_generation: int,
    ) -> CapabilityActivation:
        """Commit a prepared shutdown while retaining an idempotency tombstone."""

        return self._change_deactivation_state(
            namespace=namespace,
            capability_name=capability_name,
            expected_generation=expected_generation,
            target=CapabilityActivationState.INACTIVE,
        )

    def _change_deactivation_state(
        self,
        *,
        namespace: str,
        capability_name: str,
        expected_generation: int,
        target: CapabilityActivationState,
    ) -> CapabilityActivation:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT * FROM capability_activations
                WHERE namespace = ? AND capability_name = ?
                """,
                (safe_namespace, capability_name),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise CapabilityRevisionNotFoundError(
                    f"capability {capability_name!r} has no lifecycle record"
                )
            active = self._activation_from_row(row)
            if active.generation != expected_generation:
                conn.rollback()
                raise CapabilityRevisionConflictError(
                    f"expected activation generation {expected_generation!r}, "
                    f"found {active.generation!r}"
                )
            lifecycle_state = str(row["lifecycle_state"])
            if lifecycle_state == target.value:
                conn.rollback()
                return active
            if lifecycle_state != CapabilityActivationState.DEACTIVATING.value:
                conn.rollback()
                raise CapabilityRevisionConflictError(
                    "capability generation is not in a deactivation transition"
                )
            cursor = conn.execute(
                """
                UPDATE capability_activations
                SET lifecycle_state = ?
                WHERE namespace = ? AND capability_name = ? AND generation = ?
                    AND lifecycle_state = ?
                """,
                (
                    target.value,
                    safe_namespace,
                    capability_name,
                    expected_generation,
                    CapabilityActivationState.DEACTIVATING.value,
                ),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise CapabilityRevisionConflictError("capability activation changed concurrently")
            conn.commit()
        return active

    def record_test(self, result: CapabilityTestResult) -> CapabilityTestResult:
        """Persist the latest sanitized test attestation for one revision."""

        safe_namespace = self._namespace(result.namespace)
        with closing(self._conn()) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            row = conn.execute(
                """
                SELECT manifest_digest FROM capability_revisions
                WHERE namespace = ? AND capability_name = ? AND revision = ?
                """,
                (safe_namespace, result.capability_name, result.revision),
            ).fetchone()
            if row is None:
                raise CapabilityRevisionNotFoundError(
                    f"capability {result.capability_name!r} revision "
                    f"{result.revision} does not exist"
                )
            if str(row["manifest_digest"]) != result.manifest_digest:
                raise CapabilityRevisionConflictError(
                    "capability test digest does not match the stored revision"
                )
            conn.execute(
                """
                INSERT INTO capability_test_results (
                    namespace, capability_name, revision, manifest_digest,
                    status, checks_json, tested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (namespace, capability_name, revision) DO UPDATE SET
                    manifest_digest = excluded.manifest_digest,
                    status = excluded.status,
                    checks_json = excluded.checks_json,
                    tested_at = excluded.tested_at
                """,
                (
                    safe_namespace,
                    result.capability_name,
                    result.revision,
                    result.manifest_digest,
                    result.status.value,
                    json.dumps(
                        [check.model_dump(mode="json") for check in result.checks],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    result.tested_at,
                ),
            )
            conn.commit()
        return result

    def test_result(
        self,
        *,
        namespace: str,
        capability_name: str,
        revision: int,
    ) -> CapabilityTestResult | None:
        safe_namespace = self._namespace(namespace)
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT * FROM capability_test_results
                WHERE namespace = ? AND capability_name = ? AND revision = ?
                """,
                (safe_namespace, capability_name, revision),
            ).fetchone()
        if row is None:
            return None
        try:
            checks = tuple(
                CapabilityTestCheck.model_validate(item)
                for item in json.loads(str(row["checks_json"]))
            )
            return CapabilityTestResult(
                namespace=str(row["namespace"]),
                capability_name=str(row["capability_name"]),
                revision=int(row["revision"]),
                manifest_digest=str(row["manifest_digest"]),
                status=CapabilityTestStatus(str(row["status"])),
                checks=checks,
                tested_at=str(row["tested_at"]),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise CapabilityRevisionCorruptionError(
                "stored capability test result is invalid"
            ) from exc


__all__ = [
    "CapabilityRevisionConflictError",
    "CapabilityRevisionCorruptionError",
    "CapabilityRevisionError",
    "CapabilityRevisionNotFoundError",
    "CapabilityRevisionStore",
]
