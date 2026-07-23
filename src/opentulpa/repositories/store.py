"""SQLite persistence for repository workspaces and thread bindings."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import datetime
from pathlib import Path

from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.repositories.models import RepositoryWorkspace


class RepositoryWorkspaceConflictError(RuntimeError):
    pass


class RepositoryWorkspaceNotFoundError(KeyError):
    pass


class RepositoryWorkspaceStore:
    """Keep workspace ownership and active-thread routing durable."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = connect_sqlite(self.db_path, wal=True)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS repository_workspaces (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    repository_url TEXT NOT NULL,
                    provider TEXT NOT NULL CHECK (provider IN ('local', 'daytona')),
                    provider_workspace_id TEXT,
                    base_ref TEXT NOT NULL,
                    base_sha TEXT,
                    branch TEXT NOT NULL,
                    head_sha TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('creating', 'ready', 'stopped', 'failed', 'published')
                    ),
                    last_error TEXT,
                    pull_request_url TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_repository_workspaces_tenant_activity
                ON repository_workspaces (tenant_id, last_used_at DESC);

                CREATE TABLE IF NOT EXISTS thread_repository_bindings (
                    tenant_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    bound_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, thread_id),
                    FOREIGN KEY (workspace_id) REFERENCES repository_workspaces (id)
                );
                """
            )
            conn.commit()

    @staticmethod
    def _workspace(row: sqlite3.Row) -> RepositoryWorkspace:
        return RepositoryWorkspace.model_validate(dict(row))

    def create(self, workspace: RepositoryWorkspace) -> RepositoryWorkspace:
        data = workspace.model_dump(mode="json")
        with closing(self._conn()) as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO repository_workspaces (
                        id, tenant_id, repository_url, provider, provider_workspace_id,
                        base_ref, base_sha, branch, head_sha, status, last_error,
                        pull_request_url, created_at, updated_at, last_used_at
                    ) VALUES (
                        :id, :tenant_id, :repository_url, :provider, :provider_workspace_id,
                        :base_ref, :base_sha, :branch, :head_sha, :status, :last_error,
                        :pull_request_url, :created_at, :updated_at, :last_used_at
                    )
                    """,
                    data,
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                raise RepositoryWorkspaceConflictError("repository workspace already exists") from exc
        return workspace

    def update(self, workspace: RepositoryWorkspace) -> RepositoryWorkspace:
        data = workspace.model_dump(mode="json")
        with closing(self._conn()) as conn:
            cursor = conn.execute(
                """
                UPDATE repository_workspaces SET
                    provider_workspace_id=:provider_workspace_id,
                    base_sha=:base_sha,
                    head_sha=:head_sha,
                    status=:status,
                    last_error=:last_error,
                    pull_request_url=:pull_request_url,
                    updated_at=:updated_at,
                    last_used_at=:last_used_at
                WHERE id=:id AND tenant_id=:tenant_id
                """,
                data,
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise RepositoryWorkspaceNotFoundError(workspace.id)
            conn.commit()
        return workspace

    def get(self, *, tenant_id: str, workspace_id: str) -> RepositoryWorkspace | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM repository_workspaces WHERE tenant_id=? AND id=?",
                (tenant_id, workspace_id),
            ).fetchone()
        return self._workspace(row) if row is not None else None

    def get_any(self, workspace_id: str) -> RepositoryWorkspace | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                "SELECT * FROM repository_workspaces WHERE id=?",
                (workspace_id,),
            ).fetchone()
        return self._workspace(row) if row is not None else None

    def list(
        self,
        *,
        tenant_id: str,
        statuses: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[RepositoryWorkspace]:
        parameters: list[object] = [tenant_id]
        where = "tenant_id=?"
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where += f" AND status IN ({placeholders})"
            parameters.extend(str(status) for status in statuses)
        parameters.append(max(1, min(int(limit), 500)))
        with closing(self._conn()) as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM repository_workspaces
                WHERE {where}
                ORDER BY last_used_at DESC, id ASC
                LIMIT ?
                """,  # noqa: S608 - placeholders are generated, not user-controlled.
                parameters,
            ).fetchall()
        return [self._workspace(row) for row in rows]

    def bind(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str,
        bound_at: datetime,
    ) -> None:
        with closing(self._conn()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            owned = conn.execute(
                "SELECT 1 FROM repository_workspaces WHERE tenant_id=? AND id=?",
                (tenant_id, workspace_id),
            ).fetchone()
            if owned is None:
                conn.rollback()
                raise RepositoryWorkspaceNotFoundError(workspace_id)
            conn.execute(
                """
                INSERT INTO thread_repository_bindings (
                    tenant_id, thread_id, workspace_id, bound_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT (tenant_id, thread_id) DO UPDATE SET
                    workspace_id=excluded.workspace_id,
                    bound_at=excluded.bound_at
                """,
                (tenant_id, thread_id, workspace_id, bound_at.isoformat()),
            )
            conn.commit()

    def unbind(self, *, tenant_id: str, thread_id: str) -> None:
        with closing(self._conn()) as conn:
            conn.execute(
                "DELETE FROM thread_repository_bindings WHERE tenant_id=? AND thread_id=?",
                (tenant_id, thread_id),
            )
            conn.commit()

    def active(self, *, tenant_id: str, thread_id: str) -> RepositoryWorkspace | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT workspace.* FROM thread_repository_bindings AS binding
                JOIN repository_workspaces AS workspace
                  ON workspace.id = binding.workspace_id
                WHERE binding.tenant_id=? AND binding.thread_id=?
                  AND workspace.tenant_id=binding.tenant_id
                """,
                (tenant_id, thread_id),
            ).fetchone()
        return self._workspace(row) if row is not None else None


__all__ = [
    "RepositoryWorkspaceConflictError",
    "RepositoryWorkspaceNotFoundError",
    "RepositoryWorkspaceStore",
]
