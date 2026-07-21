"""Durable idempotency records for external side effects."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from opentulpa.persistence.sqlite import connect_sqlite


class IdempotencyConflictError(RuntimeError):
    pass


class IdempotencyPendingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IdempotencyClaim:
    created: bool
    result: dict[str, Any] | None = None


class IdempotencyStore:
    """Claim an effect key before calling a provider and replay committed results."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path.expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with connect_sqlite(self._db_path, wal=True) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS external_effects (
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, idempotency_key)
                );
                CREATE INDEX IF NOT EXISTS idx_external_effects_operation
                ON external_effects (tenant_id, operation, updated_at DESC);
                CREATE TABLE IF NOT EXISTS external_effect_reconciliations (
                    reconciliation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK (
                        decision IN ('confirm_applied', 'retry_no_effect', 'reject')
                    ),
                    actor_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def request_hash(operation: str, arguments: dict[str, Any]) -> str:
        encoded = json.dumps(
            {"operation": operation, "arguments": arguments},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def claim(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        operation: str,
        arguments: dict[str, Any],
    ) -> IdempotencyClaim:
        tenant = self._required(tenant_id, "tenant_id")
        key = self._required(idempotency_key, "idempotency_key")
        safe_operation = self._required(operation, "operation")
        digest = self.request_hash(safe_operation, arguments)
        now = datetime.now(UTC).isoformat()
        with connect_sqlite(self._db_path, wal=True) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT operation, request_hash, status, result_json
                FROM external_effects
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant, key),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO external_effects (
                        tenant_id, idempotency_key, operation, request_hash, status,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (tenant, key, safe_operation, digest, now, now),
                )
                conn.commit()
                return IdempotencyClaim(created=True)
            if str(row["operation"]) != safe_operation or str(row["request_hash"]) != digest:
                conn.rollback()
                raise IdempotencyConflictError(
                    "idempotency key was already used for a different request"
                )
            status = str(row["status"])
            if status == "pending":
                conn.rollback()
                raise IdempotencyPendingError(
                    "the external effect is pending or its outcome is indeterminate"
                )
            if status == "failed":
                conn.rollback()
                raise IdempotencyConflictError(
                    "the previous external effect attempt failed; use a new idempotency key"
                )
            result = json.loads(str(row["result_json"] or "{}"))
            conn.commit()
            return IdempotencyClaim(
                created=False,
                result=result if isinstance(result, dict) else {},
            )

    async def execute(
        self,
        *,
        tenant_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        invoke: Callable[[], Any],
    ) -> Any:
        claim = self.claim(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            operation=operation,
            arguments={"request_hash": str(request_hash or "")},
        )
        if not claim.created:
            return claim.result
        try:
            pending = invoke()
            value = await pending if inspect.isawaitable(pending) else pending
        except Exception as exc:
            self.fail(
                tenant_id=tenant_id,
                idempotency_key=idempotency_key,
                code=type(exc).__name__.lower(),
            )
            raise
        serialized = json.loads(
            json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        )
        result = serialized if isinstance(serialized, dict) else {"value": serialized}
        self.complete(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            result=result,
        )
        return value

    def complete(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        result: dict[str, Any],
    ) -> None:
        self._finish(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            status="completed",
            result=result,
        )

    def fail(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        code: str,
    ) -> None:
        self._finish(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            status="failed",
            result={"error": {"code": str(code or "external_effect_failed")[:100]}},
        )

    def reconcile_pending(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        decision: Literal["confirm_applied", "retry_no_effect", "reject"],
        actor_id: str,
        reason: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Resolve an indeterminate effect only from an explicit owner/application decision."""

        tenant = self._required(tenant_id, "tenant_id")
        key = self._required(idempotency_key, "idempotency_key")
        actor = self._required(actor_id, "actor_id")
        safe_reason = str(reason or "").strip()
        if not safe_reason or len(safe_reason) > 2_000:
            raise ValueError("reason is required and must not exceed 2000 characters")
        if decision not in {"confirm_applied", "retry_no_effect", "reject"}:
            raise ValueError("unsupported reconciliation decision")
        now = datetime.now(UTC).isoformat()
        with connect_sqlite(self._db_path, wal=True) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT status, result_json
                FROM external_effects
                WHERE tenant_id = ? AND idempotency_key = ?
                """,
                (tenant, key),
            ).fetchone()
            if row is None:
                conn.rollback()
                raise LookupError("idempotency claim was not found")
            status = str(row["status"])
            if status != "pending":
                if decision == "confirm_applied" and status == "completed":
                    conn.commit()
                    return
                conn.rollback()
                raise IdempotencyConflictError("idempotency claim is not pending")
            if decision == "retry_no_effect":
                conn.execute(
                    "DELETE FROM external_effects WHERE tenant_id = ? AND idempotency_key = ?",
                    (tenant, key),
                )
            else:
                payload = json.dumps(
                    result
                    if decision == "confirm_applied"
                    else {"error": {"code": "owner_rejected_effect"}},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
                conn.execute(
                    """
                    UPDATE external_effects
                    SET status = ?, result_json = ?, updated_at = ?
                    WHERE tenant_id = ? AND idempotency_key = ? AND status = 'pending'
                    """,
                    (
                        "completed" if decision == "confirm_applied" else "failed",
                        payload,
                        now,
                        tenant,
                        key,
                    ),
                )
            conn.execute(
                """
                INSERT INTO external_effect_reconciliations (
                    tenant_id, idempotency_key, decision, actor_id, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (tenant, key, decision, actor, safe_reason, now),
            )
            conn.commit()

    def _finish(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        status: Literal["completed", "failed"],
        result: dict[str, Any],
    ) -> None:
        tenant = self._required(tenant_id, "tenant_id")
        key = self._required(idempotency_key, "idempotency_key")
        payload = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        with connect_sqlite(self._db_path, wal=True) as conn:
            cursor = conn.execute(
                """
                UPDATE external_effects
                SET status = ?, result_json = ?, updated_at = ?
                WHERE tenant_id = ? AND idempotency_key = ? AND status = 'pending'
                """,
                (status, payload, datetime.now(UTC).isoformat(), tenant, key),
            )
            if cursor.rowcount != 1:
                conn.rollback()
                raise IdempotencyConflictError("idempotency claim is not pending")
            conn.commit()

    @staticmethod
    def _required(value: str, field: str) -> str:
        safe = str(value or "").strip()
        if not safe:
            raise ValueError(f"{field} is required")
        if len(safe) > 300:
            raise ValueError(f"{field} is too long")
        return safe


__all__ = [
    "IdempotencyClaim",
    "IdempotencyConflictError",
    "IdempotencyPendingError",
    "IdempotencyStore",
]
