"""SQLite repository for revisioned intake workflow drafts."""

from __future__ import annotations

import hmac
import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from opentulpa.intake.drafts.models import IntakeDraft
from opentulpa.persistence.sqlite import connect_sqlite


class IntakeDraftConflictError(RuntimeError):
    """A draft revision or lifecycle precondition no longer matches."""


class IntakeDraftNotFoundError(KeyError):
    """The tenant-owned draft does not exist."""


class IntakeDraftConfirmationError(IntakeDraftConflictError):
    """A confirmation token does not authorize the prepared proposal."""


class IntakeDraftStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path)

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            # Active workflow replacement attaches this database to the workflow
            # database. Rollback journals are required for SQLite's atomic
            # multi-database super-journal commit.
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS intake_drafts (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    revision INTEGER NOT NULL CHECK (revision >= 1),
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    proposal_json TEXT,
                    proposal_hash TEXT,
                    prepared_revision INTEGER,
                    confirmation_token_hash TEXT,
                    activation_attempt_id TEXT,
                    created_by_actor_id TEXT NOT NULL,
                    updated_by_actor_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    prepared_at TEXT,
                    activated_at TEXT,
                    PRIMARY KEY (tenant_id, id)
                );

                CREATE INDEX IF NOT EXISTS idx_intake_drafts_tenant_updated
                ON intake_drafts (tenant_id, updated_at DESC, id);

                CREATE INDEX IF NOT EXISTS idx_intake_drafts_tenant_workflow
                ON intake_drafts (tenant_id, workflow_id, updated_at DESC);
                """
            )
            conn.commit()

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def _row_to_draft(cls, row: sqlite3.Row) -> IntakeDraft:
        proposal = json.loads(str(row["proposal_json"])) if row["proposal_json"] else None
        return IntakeDraft.model_validate(
            {
                "id": row["id"],
                "tenant_id": row["tenant_id"],
                "workflow_id": row["workflow_id"],
                "revision": row["revision"],
                "status": row["status"],
                "payload": json.loads(str(row["payload_json"])),
                "proposal": proposal,
                "proposal_hash": row["proposal_hash"],
                "prepared_revision": row["prepared_revision"],
                "created_by_actor_id": row["created_by_actor_id"],
                "updated_by_actor_id": row["updated_by_actor_id"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "prepared_at": row["prepared_at"],
                "activated_at": row["activated_at"],
            }
        )

    def get(self, *, tenant_id: str, draft_id: str) -> IntakeDraft | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM intake_drafts WHERE tenant_id = ? AND id = ?",
                (tenant_id, draft_id),
            ).fetchone()
        return self._row_to_draft(row) if row is not None else None

    def list(self, *, tenant_id: str, workflow_id: str | None = None) -> list[IntakeDraft]:
        with closing(self._conn()) as conn:
            if workflow_id:
                rows = conn.execute(
                    """
                    SELECT * FROM intake_drafts
                    WHERE tenant_id = ? AND workflow_id = ?
                    ORDER BY updated_at DESC, id
                    """,
                    (tenant_id, workflow_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM intake_drafts
                    WHERE tenant_id = ?
                    ORDER BY updated_at DESC, id
                    """,
                    (tenant_id,),
                ).fetchall()
        return [self._row_to_draft(row) for row in rows]

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        workflow_id: str,
        patch: dict[str, Any],
        expected_revision: int | None,
        now: datetime,
    ) -> IntakeDraft:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM intake_drafts WHERE tenant_id = ? AND id = ?",
                (tenant_id, draft_id),
            ).fetchone()
            if row is None:
                if expected_revision is not None:
                    conn.rollback()
                    raise IntakeDraftConflictError(
                        f"draft {draft_id!r} does not exist at revision {expected_revision}"
                    )
                revision = 1
                payload = dict(patch)
                conn.execute(
                    """
                    INSERT INTO intake_drafts (
                        tenant_id, id, workflow_id, revision, status, payload_json,
                        proposal_json, proposal_hash, prepared_revision,
                        confirmation_token_hash, activation_attempt_id,
                        created_by_actor_id, updated_by_actor_id,
                        created_at, updated_at, prepared_at, activated_at
                    ) VALUES (?, ?, ?, ?, 'editing', ?, NULL, NULL, NULL, NULL,
                              NULL, ?, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        tenant_id,
                        draft_id,
                        workflow_id,
                        revision,
                        self._json(payload),
                        actor_id,
                        actor_id,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
            else:
                if str(row["status"]) == "activating":
                    conn.rollback()
                    raise IntakeDraftConflictError("draft activation is in progress")
                current_revision = int(row["revision"])
                if expected_revision is None:
                    conn.rollback()
                    raise IntakeDraftConflictError(
                        "expected_revision is required when patching a draft"
                    )
                if current_revision != expected_revision:
                    conn.rollback()
                    raise IntakeDraftConflictError(
                        f"expected revision {expected_revision}, found {current_revision}"
                    )
                if str(row["workflow_id"]) != workflow_id:
                    conn.rollback()
                    raise IntakeDraftConflictError("draft workflow_id is immutable")
                payload = json.loads(str(row["payload_json"]))
                payload.update(patch)
                revision = current_revision + 1
                cursor = conn.execute(
                    """
                    UPDATE intake_drafts SET
                        revision = ?, status = 'editing', payload_json = ?,
                        proposal_json = NULL, proposal_hash = NULL,
                        prepared_revision = NULL, confirmation_token_hash = NULL,
                        activation_attempt_id = NULL, updated_by_actor_id = ?,
                        updated_at = ?, prepared_at = NULL, activated_at = NULL
                    WHERE tenant_id = ? AND id = ? AND revision = ?
                    """,
                    (
                        revision,
                        self._json(payload),
                        actor_id,
                        now.isoformat(),
                        tenant_id,
                        draft_id,
                        expected_revision,
                    ),
                )
                if cursor.rowcount != 1:
                    conn.rollback()
                    raise IntakeDraftConflictError("draft changed during patch")
            conn.commit()
        draft = self.get(tenant_id=tenant_id, draft_id=draft_id)
        if draft is None:  # pragma: no cover - committed row must exist
            raise RuntimeError("saved intake draft disappeared")
        return draft

    def prepare(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        proposal: dict[str, Any],
        proposal_hash: str,
        confirmation_token_hash: str,
        now: datetime,
    ) -> IntakeDraft:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT revision, status FROM intake_drafts WHERE tenant_id = ? AND id = ?",
                (tenant_id, draft_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise IntakeDraftNotFoundError(draft_id)
            current_revision = int(row["revision"])
            if current_revision != expected_revision:
                conn.rollback()
                raise IntakeDraftConflictError(
                    f"expected revision {expected_revision}, found {current_revision}"
                )
            if str(row["status"]) == "activating":
                conn.rollback()
                raise IntakeDraftConflictError("draft activation is in progress")
            conn.execute(
                """
                UPDATE intake_drafts SET
                    status = 'prepared', proposal_json = ?, proposal_hash = ?,
                    prepared_revision = ?, confirmation_token_hash = ?,
                    activation_attempt_id = NULL, updated_by_actor_id = ?,
                    updated_at = ?, prepared_at = ?, activated_at = NULL
                WHERE tenant_id = ? AND id = ? AND revision = ?
                """,
                (
                    self._json(proposal),
                    proposal_hash,
                    expected_revision,
                    confirmation_token_hash,
                    actor_id,
                    now.isoformat(),
                    now.isoformat(),
                    tenant_id,
                    draft_id,
                    expected_revision,
                ),
            )
            conn.commit()
        draft = self.get(tenant_id=tenant_id, draft_id=draft_id)
        if draft is None:  # pragma: no cover
            raise RuntimeError("prepared intake draft disappeared")
        return draft

    def claim_activation(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token_hash: str,
        activation_attempt_id: str,
        now: datetime,
    ) -> IntakeDraft:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            draft = self.claim_activation_in_transaction(
                conn,
                schema="main",
                tenant_id=tenant_id,
                actor_id=actor_id,
                draft_id=draft_id,
                expected_revision=expected_revision,
                confirmation_token_hash=confirmation_token_hash,
                activation_attempt_id=activation_attempt_id,
                now=now,
            )
            conn.commit()
        return draft

    @staticmethod
    def _table(schema: str) -> str:
        if schema not in {"main", "intake_drafts_db"}:
            raise ValueError("unsupported intake draft database schema")
        return f"{schema}.intake_drafts"

    def claim_activation_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        schema: str,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token_hash: str,
        activation_attempt_id: str,
        now: datetime,
    ) -> IntakeDraft:
        """Claim a prepared draft using the caller's open transaction."""

        table = self._table(schema)
        row = conn.execute(
            f"SELECT * FROM {table} WHERE tenant_id = ? AND id = ?",  # noqa: S608
            (tenant_id, draft_id),
        ).fetchone()
        if row is None:
            raise IntakeDraftNotFoundError(draft_id)
        current_revision = int(row["revision"])
        if current_revision != expected_revision:
            raise IntakeDraftConflictError(
                f"expected revision {expected_revision}, found {current_revision}"
            )
        if str(row["status"]) != "prepared":
            raise IntakeDraftConfirmationError("draft is not prepared for activation")
        if int(row["prepared_revision"] or 0) != expected_revision:
            raise IntakeDraftConfirmationError(
                "prepared proposal does not match the draft revision"
            )
        stored_hash = str(row["confirmation_token_hash"] or "")
        if not stored_hash or not hmac.compare_digest(
            stored_hash,
            confirmation_token_hash,
        ):
            raise IntakeDraftConfirmationError("confirmation token is invalid")
        cursor = conn.execute(  # noqa: S608
            f"""
            UPDATE {table} SET
                status = 'activating', activation_attempt_id = ?,
                updated_by_actor_id = ?, updated_at = ?
            WHERE tenant_id = ? AND id = ? AND revision = ?
              AND status = 'prepared'
            """,
            (
                activation_attempt_id,
                actor_id,
                now.isoformat(),
                tenant_id,
                draft_id,
                expected_revision,
            ),
        )
        if cursor.rowcount != 1:
            raise IntakeDraftConflictError("draft changed during activation claim")
        claimed = conn.execute(
            f"SELECT * FROM {table} WHERE tenant_id = ? AND id = ?",  # noqa: S608
            (tenant_id, draft_id),
        ).fetchone()
        if claimed is None:  # pragma: no cover - transaction still owns the row
            raise RuntimeError("claimed intake draft disappeared")
        return self._row_to_draft(claimed)

    def release_activation(
        self,
        *,
        tenant_id: str,
        draft_id: str,
        activation_attempt_id: str,
        now: datetime,
    ) -> None:
        with closing(self._conn()) as conn:
            conn.execute(
                """
                UPDATE intake_drafts SET
                    status = 'prepared', activation_attempt_id = NULL, updated_at = ?
                WHERE tenant_id = ? AND id = ? AND status = 'activating'
                  AND activation_attempt_id = ?
                """,
                (now.isoformat(), tenant_id, draft_id, activation_attempt_id),
            )
            conn.commit()

    def finish_activation(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        activation_attempt_id: str,
        now: datetime,
    ) -> IntakeDraft:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            draft = self.finish_activation_in_transaction(
                conn,
                schema="main",
                tenant_id=tenant_id,
                actor_id=actor_id,
                draft_id=draft_id,
                expected_revision=expected_revision,
                activation_attempt_id=activation_attempt_id,
                now=now,
            )
            conn.commit()
        return draft

    def finish_activation_in_transaction(
        self,
        conn: sqlite3.Connection,
        *,
        schema: str,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        activation_attempt_id: str,
        now: datetime,
    ) -> IntakeDraft:
        """Consume a claimed confirmation using the caller's transaction."""

        table = self._table(schema)
        cursor = conn.execute(  # noqa: S608
            f"""
            UPDATE {table} SET
                status = 'activated', confirmation_token_hash = NULL,
                activation_attempt_id = NULL, updated_by_actor_id = ?,
                updated_at = ?, activated_at = ?
            WHERE tenant_id = ? AND id = ? AND revision = ?
              AND status = 'activating' AND activation_attempt_id = ?
            """,
            (
                actor_id,
                now.isoformat(),
                now.isoformat(),
                tenant_id,
                draft_id,
                expected_revision,
                activation_attempt_id,
            ),
        )
        if cursor.rowcount != 1:
            raise IntakeDraftConflictError("draft activation claim was lost")
        row = conn.execute(
            f"SELECT * FROM {table} WHERE tenant_id = ? AND id = ?",  # noqa: S608
            (tenant_id, draft_id),
        ).fetchone()
        if row is None:  # pragma: no cover - transaction still owns the row
            raise RuntimeError("activated intake draft disappeared")
        return self._row_to_draft(row)

    def delete(
        self,
        *,
        tenant_id: str,
        draft_id: str,
        expected_revision: int,
    ) -> None:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT revision, status FROM intake_drafts WHERE tenant_id = ? AND id = ?",
                (tenant_id, draft_id),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise IntakeDraftNotFoundError(draft_id)
            if str(row["status"]) == "activating":
                conn.rollback()
                raise IntakeDraftConflictError("draft activation is in progress")
            revision = int(row["revision"])
            if revision != expected_revision:
                conn.rollback()
                raise IntakeDraftConflictError(
                    f"expected revision {expected_revision}, found {revision}"
                )
            conn.execute(
                "DELETE FROM intake_drafts WHERE tenant_id = ? AND id = ? AND revision = ?",
                (tenant_id, draft_id, expected_revision),
            )
            conn.commit()


__all__ = [
    "IntakeDraftConfirmationError",
    "IntakeDraftConflictError",
    "IntakeDraftNotFoundError",
    "IntakeDraftStore",
]
