"""Durable SQLite persistence for MCP audit and idempotency boundaries."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from opentulpa.mcp.models import MCPAuditEvent, MCPBrokerResult
from opentulpa.persistence.sqlite import connect_sqlite


class SQLiteMCPAuditSink:
    """Append-only MCP audit events that survive runtime restarts."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(
            self.db_path,
            synchronous_normal=False,
            wal=True,
        )

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mcp_audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    audit_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    capability_name TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_mcp_audit_tenant_sequence
                ON mcp_audit_events (tenant_id, sequence);

                CREATE INDEX IF NOT EXISTS idx_mcp_audit_id_sequence
                ON mcp_audit_events (audit_id, sequence);
                """
            )
            conn.commit()
        self.db_path.chmod(0o600)

    async def record(self, event: MCPAuditEvent) -> None:
        await asyncio.to_thread(self._record, event)

    def _record(self, event: MCPAuditEvent) -> None:
        payload = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with closing(self._conn()) as conn:
            conn.execute(
                """
                INSERT INTO mcp_audit_events (
                    audit_id, tenant_id, capability_name, tool_name, outcome, event_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.audit_id,
                    event.tenant_id,
                    event.capability_name,
                    event.tool_name,
                    event.outcome,
                    payload,
                ),
            )
            conn.commit()

    async def list_events(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 1_000,
    ) -> tuple[MCPAuditEvent, ...]:
        """Read persisted events in commit order for diagnostics and tests."""

        safe_limit = max(1, min(int(limit), 10_000))
        return await asyncio.to_thread(self._list_events, tenant_id, safe_limit)

    def _list_events(
        self,
        tenant_id: str | None,
        limit: int,
    ) -> tuple[MCPAuditEvent, ...]:
        query = "SELECT event_json FROM mcp_audit_events"
        parameters: tuple[object, ...]
        if tenant_id is None:
            query += " ORDER BY sequence ASC LIMIT ?"
            parameters = (limit,)
        else:
            query += " WHERE tenant_id = ? ORDER BY sequence ASC LIMIT ?"
            parameters = (str(tenant_id), limit)
        with closing(self._conn()) as conn:
            rows = conn.execute(query, parameters).fetchall()
        return tuple(
            MCPAuditEvent.model_validate_json(str(row["event_json"])) for row in rows
        )


class SQLiteMCPIdempotencyStore:
    """Persist completed MCP results for safe replay after a runtime restart."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(
            self.db_path,
            synchronous_normal=False,
            wal=True,
        )

    def _init_db(self) -> None:
        with closing(self._conn()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS mcp_idempotency_results (
                    idempotency_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    )
                );
                """
            )
            conn.commit()
        self.db_path.chmod(0o600)

    async def get(self, key: str) -> MCPBrokerResult | None:
        return await asyncio.to_thread(self._get, key)

    def _get(self, key: str) -> MCPBrokerResult | None:
        with closing(self._conn()) as conn:
            row = conn.execute(
                """
                SELECT result_json
                FROM mcp_idempotency_results
                WHERE idempotency_key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return MCPBrokerResult.model_validate_json(str(row["result_json"]))

    async def put(self, key: str, result: MCPBrokerResult) -> None:
        await asyncio.to_thread(self._put, key, result)

    def _put(self, key: str, result: MCPBrokerResult) -> None:
        payload = json.dumps(
            result.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with closing(self._conn()) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO mcp_idempotency_results (
                    idempotency_key, result_json
                ) VALUES (?, ?)
                """,
                (key, payload),
            )
            conn.commit()


__all__ = ["SQLiteMCPAuditSink", "SQLiteMCPIdempotencyStore"]
