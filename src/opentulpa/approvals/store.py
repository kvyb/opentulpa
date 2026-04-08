"""Durable storage for pending approvals."""

from __future__ import annotations

import json
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from opentulpa.approvals.models import ApprovalRecord


class PendingApprovalStore:
    """SQLite-backed lifecycle store for single-use approval challenges."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS pending_approvals (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    origin_interface TEXT NOT NULL,
                    origin_user_id TEXT NOT NULL,
                    origin_conversation_id TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    action_args_json TEXT NOT NULL,
                    recipient_scope TEXT NOT NULL,
                    impact_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    decided_at TEXT,
                    executed_at TEXT,
                    decision_actor_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pending_approvals_customer
                    ON pending_approvals(customer_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_pending_approvals_status_expires
                    ON pending_approvals(status, expires_at ASC);
                """
            )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    def utc_now(self) -> datetime:
        return self._utc_now()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            id=str(row["id"]),
            customer_id=str(row["customer_id"]),
            thread_id=str(row["thread_id"]),
            origin_interface=str(row["origin_interface"]),
            origin_user_id=str(row["origin_user_id"]),
            origin_conversation_id=str(row["origin_conversation_id"]),
            action_name=str(row["action_name"]),
            action_args_json=str(row["action_args_json"]),
            recipient_scope=str(row["recipient_scope"]),  # type: ignore[arg-type]
            impact_type=str(row["impact_type"]),  # type: ignore[arg-type]
            summary=str(row["summary"]),
            reason=str(row["reason"]),
            confidence=float(row["confidence"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            created_at=str(row["created_at"]),
            expires_at=str(row["expires_at"]),
            decided_at=str(row["decided_at"]) if row["decided_at"] else None,
            executed_at=str(row["executed_at"]) if row["executed_at"] else None,
            decision_actor_id=str(row["decision_actor_id"]) if row["decision_actor_id"] else None,
        )

    def expire_due(self) -> int:
        now_iso = self._utc_now_iso()
        with self._conn() as conn:
            cur = conn.execute(
                """
                UPDATE pending_approvals
                SET status='expired', decided_at=?
                WHERE status='pending' AND expires_at<=?
                """,
                (now_iso, now_iso),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def get(self, approval_id: str) -> ApprovalRecord | None:
        self.expire_due()
        aid = str(approval_id or "").strip()
        if not aid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM pending_approvals
                WHERE id=?
                """,
                (aid,),
            ).fetchone()
        if not row:
            return None
        return self._row_to_record(row)

    def list_pending_for_origin(
        self,
        *,
        origin_interface: str,
        origin_user_id: str,
        origin_conversation_id: str,
        limit: int = 5,
    ) -> list[ApprovalRecord]:
        self.expire_due()
        iface = str(origin_interface or "").strip()
        uid = str(origin_user_id or "").strip()
        conv = str(origin_conversation_id or "").strip()
        if not iface or not uid or not conv:
            return []
        safe_limit = max(1, min(int(limit), 20))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM pending_approvals
                WHERE origin_interface=? AND origin_user_id=? AND origin_conversation_id=? AND status='pending'
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (iface, uid, conv, safe_limit),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def has_pending_for_customer_thread(
        self,
        *,
        customer_id: str,
        thread_id: str,
    ) -> bool:
        self.expire_due()
        cid = str(customer_id or "").strip()
        tid = str(thread_id or "").strip()
        if not cid or not tid:
            return False
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM pending_approvals
                WHERE customer_id=? AND thread_id=? AND status='pending'
                LIMIT 1
                """,
                (cid, tid),
            ).fetchone()
        return bool(row)

    def find_pending_duplicate(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args_json: str,
    ) -> ApprovalRecord | None:
        self.expire_due()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM pending_approvals
                WHERE customer_id=? AND thread_id=? AND action_name=? AND action_args_json=? AND status='pending'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (
                    str(customer_id or "").strip(),
                    str(thread_id or "").strip(),
                    str(action_name or "").strip(),
                    str(action_args_json or ""),
                ),
            ).fetchone()
        return self._row_to_record(row) if row else None

    def find_recent_matching(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args_json: str | None = None,
        summary: str | None = None,
        statuses: tuple[str, ...] = ("approved", "executed"),
        lookback_seconds: int = 600,
    ) -> ApprovalRecord | None:
        self.expire_due()
        safe_statuses = tuple(str(s or "").strip() for s in statuses if str(s or "").strip())
        if not safe_statuses:
            return None
        filters = [
            "customer_id=?",
            "thread_id=?",
            "action_name=?",
            f"status IN ({','.join(['?'] * len(safe_statuses))})",
        ]
        values: list[Any] = [
            str(customer_id or "").strip(),
            str(thread_id or "").strip(),
            str(action_name or "").strip(),
            *safe_statuses,
        ]
        if action_args_json is not None:
            filters.append("action_args_json=?")
            values.append(str(action_args_json))
        if summary is not None:
            filters.append("summary=?")
            values.append(str(summary))

        query = f"""
            SELECT *
            FROM pending_approvals
            WHERE {' AND '.join(filters)}
            ORDER BY created_at DESC
            LIMIT 20
        """
        with self._conn() as conn:
            rows = conn.execute(query, tuple(values)).fetchall()

        if not rows:
            return None
        now = self._utc_now()
        cutoff = now - timedelta(seconds=max(0, int(lookback_seconds)))
        for row in rows:
            record = self._row_to_record(row)
            try:
                created = datetime.fromisoformat(record.created_at)
            except Exception:
                continue
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            if created >= cutoff:
                return record
        return None

    def create_pending(
        self,
        *,
        approval_id: str,
        customer_id: str,
        thread_id: str,
        origin_interface: str,
        origin_user_id: str,
        origin_conversation_id: str,
        action_name: str,
        action_args: dict[str, Any],
        recipient_scope: str,
        impact_type: str,
        summary: str,
        reason: str,
        confidence: float,
        ttl_seconds: int,
    ) -> ApprovalRecord:
        created_at = self._utc_now()
        expires_at = created_at + timedelta(seconds=max(30, int(ttl_seconds)))
        args_json = json.dumps(action_args if isinstance(action_args, dict) else {}, sort_keys=True)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO pending_approvals (
                    id, customer_id, thread_id, origin_interface, origin_user_id, origin_conversation_id,
                    action_name, action_args_json, recipient_scope, impact_type, summary, reason, confidence,
                    status, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    str(approval_id),
                    str(customer_id),
                    str(thread_id),
                    str(origin_interface),
                    str(origin_user_id),
                    str(origin_conversation_id),
                    str(action_name),
                    args_json,
                    str(recipient_scope),
                    str(impact_type),
                    str(summary),
                    str(reason),
                    float(confidence),
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            conn.commit()
        record = self.get(approval_id)
        if record is None:
            raise RuntimeError("failed to persist pending approval")
        return record

    def set_decision(
        self,
        *,
        approval_id: str,
        decision: str,
        actor_id: str,
    ) -> ApprovalRecord | None:
        self.expire_due()
        status = "approved" if str(decision).strip().lower() == "approve" else "denied"
        now = self._utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE pending_approvals
                SET status=?, decided_at=?, decision_actor_id=?
                WHERE id=? AND status='pending'
                """,
                (status, now, str(actor_id or "").strip(), str(approval_id or "").strip()),
            )
            conn.commit()
        return self.get(approval_id)

    def mark_executed(self, approval_id: str) -> ApprovalRecord | None:
        now = self._utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE pending_approvals
                SET status='executed', executed_at=?
                WHERE id=? AND status='approved'
                """,
                (now, str(approval_id or "").strip()),
            )
            conn.commit()
        return self.get(approval_id)

    def as_dict(self, record: ApprovalRecord | None) -> dict[str, Any] | None:
        if record is None:
            return None
        args: dict[str, Any] = {}
        with suppress(Exception):
            parsed = json.loads(record.action_args_json)
            args = parsed if isinstance(parsed, dict) else {}
        return {
            "id": record.id,
            "customer_id": record.customer_id,
            "thread_id": record.thread_id,
            "origin_interface": record.origin_interface,
            "origin_user_id": record.origin_user_id,
            "origin_conversation_id": record.origin_conversation_id,
            "action_name": record.action_name,
            "action_args": args,
            "recipient_scope": record.recipient_scope,
            "impact_type": record.impact_type,
            "summary": record.summary,
            "reason": record.reason,
            "confidence": record.confidence,
            "status": record.status,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "decided_at": record.decided_at,
            "executed_at": record.executed_at,
            "decision_actor_id": record.decision_actor_id,
        }

    def list_thread_window(
        self,
        *,
        customer_id: str,
        thread_id: str,
        anchor_created_at: str,
        window_seconds: int,
    ) -> list[ApprovalRecord]:
        cid = str(customer_id or "").strip()
        tid = str(thread_id or "").strip()
        if not cid or not tid:
            return []
        try:
            anchor = datetime.fromisoformat(str(anchor_created_at or "").strip())
        except Exception:
            return []
        if anchor.tzinfo is None:
            anchor = anchor.replace(tzinfo=UTC)
        delta = timedelta(seconds=max(1, int(window_seconds)))
        start = (anchor - delta).isoformat()
        end = (anchor + delta).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM pending_approvals
                WHERE customer_id=? AND thread_id=? AND created_at>=? AND created_at<=?
                ORDER BY created_at ASC
                """,
                (cid, tid, start, end),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]
