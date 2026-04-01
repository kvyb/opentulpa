"""Durable external signal inbox, wake rules, and outbound outbox."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


class SignalInboxService:
    """Store external signals, configurable wake rules, and pending outbound replies."""

    def __init__(self, *, db_path: Path) -> None:
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
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    text TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL,
                    dispatch_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    available_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_signals_ready
                    ON signals(source, customer_id, thread_id, status, available_at, id);

                CREATE TABLE IF NOT EXISTS signal_rules (
                    source TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    wake_mode TEXT NOT NULL,
                    batch_window_seconds INTEGER NOT NULL DEFAULT 0,
                    auto_reply INTEGER NOT NULL DEFAULT 1,
                    guidance_text TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (source, customer_id, thread_id)
                );

                CREATE TABLE IF NOT EXISTS signal_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    dispatch_json TEXT NOT NULL,
                    signal_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    sent_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_signal_outbox_pending
                    ON signal_outbox(source, status, id);
                """
            )

    @staticmethod
    def _json_dict(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip()

    def ingest_signal(
        self,
        *,
        source: str,
        customer_id: str,
        thread_id: str,
        event_type: str = "message",
        text: str = "",
        payload: dict[str, Any] | None = None,
        dispatch: dict[str, Any] | None = None,
        batch_window_seconds: int = 0,
    ) -> dict[str, Any]:
        safe_source = self._normalize_text(source)
        safe_customer = self._normalize_text(customer_id)
        safe_thread = self._normalize_text(thread_id)
        if not safe_source:
            raise ValueError("source is required")
        if not safe_customer:
            raise ValueError("customer_id is required")
        if not safe_thread:
            raise ValueError("thread_id is required")

        now = _utc_now()
        safe_batch_window = max(0, min(int(batch_window_seconds), 3600))
        available_at = (now + timedelta(seconds=safe_batch_window)).isoformat()
        created_at = now.isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO signals (
                    source, customer_id, thread_id, event_type, text,
                    payload_json, dispatch_json, status, available_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """,
                (
                    safe_source,
                    safe_customer,
                    safe_thread,
                    self._normalize_text(event_type) or "message",
                    self._normalize_text(text),
                    json.dumps(self._json_dict(payload), ensure_ascii=False),
                    json.dumps(self._json_dict(dispatch), ensure_ascii=False),
                    available_at,
                    created_at,
                    created_at,
                ),
            )
            conn.commit()
            signal_id = int(cur.lastrowid)
        return {
            "id": signal_id,
            "source": safe_source,
            "customer_id": safe_customer,
            "thread_id": safe_thread,
            "available_at": available_at,
            "batch_window_seconds": safe_batch_window,
        }

    def resolve_rule(self, *, source: str, customer_id: str, thread_id: str) -> dict[str, Any]:
        safe_source = self._normalize_text(source)
        safe_customer = self._normalize_text(customer_id)
        safe_thread = self._normalize_text(thread_id)
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT source, customer_id, thread_id, wake_mode, batch_window_seconds,
                       auto_reply, guidance_text, updated_at
                FROM signal_rules
                WHERE source=?
                  AND customer_id IN ('', ?)
                  AND thread_id IN ('', ?)
                """,
                (safe_source, safe_customer, safe_thread),
            ).fetchall()
        if not rows:
            return {
                "source": safe_source,
                "customer_id": "",
                "thread_id": "",
                "wake_mode": "classifier",
                "batch_window_seconds": 0,
                "auto_reply": True,
                "guidance_text": "",
                "updated_at": "",
            }
        best = max(
            rows,
            key=lambda row: (
                2 if str(row["customer_id"] or "") == safe_customer else 0,
                1 if str(row["thread_id"] or "") == safe_thread else 0,
                str(row["updated_at"] or ""),
            ),
        )
        return {
            "source": str(best["source"]),
            "customer_id": str(best["customer_id"]),
            "thread_id": str(best["thread_id"]),
            "wake_mode": str(best["wake_mode"]),
            "batch_window_seconds": int(best["batch_window_seconds"]),
            "auto_reply": bool(int(best["auto_reply"])),
            "guidance_text": str(best["guidance_text"] or ""),
            "updated_at": str(best["updated_at"] or ""),
        }

    def upsert_rule(
        self,
        *,
        source: str,
        wake_mode: str,
        customer_id: str = "",
        thread_id: str = "",
        batch_window_seconds: int = 0,
        auto_reply: bool = True,
        guidance_text: str = "",
    ) -> dict[str, Any]:
        safe_source = self._normalize_text(source)
        if not safe_source:
            raise ValueError("source is required")
        safe_mode = self._normalize_text(wake_mode).lower()
        if safe_mode not in {"always", "classifier", "never"}:
            raise ValueError("wake_mode must be one of: always, classifier, never")
        safe_customer = self._normalize_text(customer_id)
        safe_thread = self._normalize_text(thread_id)
        updated_at = _utc_now_iso()
        safe_batch_window = max(0, min(int(batch_window_seconds), 3600))
        safe_guidance = str(guidance_text or "").strip()[:4000]
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO signal_rules (
                    source, customer_id, thread_id, wake_mode,
                    batch_window_seconds, auto_reply, guidance_text, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, customer_id, thread_id)
                DO UPDATE SET
                    wake_mode=excluded.wake_mode,
                    batch_window_seconds=excluded.batch_window_seconds,
                    auto_reply=excluded.auto_reply,
                    guidance_text=excluded.guidance_text,
                    updated_at=excluded.updated_at
                """,
                (
                    safe_source,
                    safe_customer,
                    safe_thread,
                    safe_mode,
                    safe_batch_window,
                    1 if auto_reply else 0,
                    safe_guidance,
                    updated_at,
                ),
            )
            conn.commit()
        return self.resolve_rule(source=safe_source, customer_id=safe_customer, thread_id=safe_thread)

    def list_rules(
        self,
        *,
        source: str = "",
        customer_id: str = "",
        thread_id: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = (
            "SELECT source, customer_id, thread_id, wake_mode, batch_window_seconds, "
            "auto_reply, guidance_text, updated_at FROM signal_rules WHERE 1=1"
        )
        params: list[Any] = []
        safe_source = self._normalize_text(source)
        safe_customer = self._normalize_text(customer_id)
        safe_thread = self._normalize_text(thread_id)
        if safe_source:
            query += " AND source=?"
            params.append(safe_source)
        if safe_customer:
            query += " AND customer_id=?"
            params.append(safe_customer)
        if safe_thread:
            query += " AND thread_id=?"
            params.append(safe_thread)
        query += " ORDER BY source ASC, customer_id ASC, thread_id ASC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "source": str(row["source"]),
                "customer_id": str(row["customer_id"]),
                "thread_id": str(row["thread_id"]),
                "wake_mode": str(row["wake_mode"]),
                "batch_window_seconds": int(row["batch_window_seconds"]),
                "auto_reply": bool(int(row["auto_reply"])),
                "guidance_text": str(row["guidance_text"] or ""),
                "updated_at": str(row["updated_at"] or ""),
            }
            for row in rows
        ]

    def claim_ready_batch(
        self,
        *,
        source: str,
        customer_id: str,
        thread_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        safe_source = self._normalize_text(source)
        safe_customer = self._normalize_text(customer_id)
        safe_thread = self._normalize_text(thread_id)
        safe_limit = max(1, min(int(limit), 500))
        now = _utc_now_iso()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, source, customer_id, thread_id, event_type, text,
                       payload_json, dispatch_json, status, available_at, created_at, updated_at
                FROM signals
                WHERE source=? AND customer_id=? AND thread_id=?
                  AND status='pending' AND available_at <= ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (safe_source, safe_customer, safe_thread, now, safe_limit),
            ).fetchall()
            if not rows:
                return []
            ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"""
                UPDATE signals
                SET status='processing', updated_at=?
                WHERE id IN ({placeholders})
                """,
                (now, *ids),
            )
            conn.commit()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                payload = json.loads(row["payload_json"] or "{}")
            except Exception:
                payload = {}
            try:
                dispatch = json.loads(row["dispatch_json"] or "{}")
            except Exception:
                dispatch = {}
            out.append(
                {
                    "id": int(row["id"]),
                    "source": str(row["source"]),
                    "customer_id": str(row["customer_id"]),
                    "thread_id": str(row["thread_id"]),
                    "event_type": str(row["event_type"]),
                    "text": str(row["text"] or ""),
                    "payload": payload if isinstance(payload, dict) else {},
                    "dispatch": dispatch if isinstance(dispatch, dict) else {},
                    "available_at": str(row["available_at"]),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                }
            )
        return out

    def mark_done(self, signal_ids: list[int]) -> int:
        return self._update_signal_status(signal_ids, status="done")

    def release_processing(self, signal_ids: list[int]) -> int:
        return self._update_signal_status(signal_ids, status="pending")

    def _update_signal_status(self, signal_ids: list[int], *, status: str) -> int:
        ids = [int(item) for item in signal_ids if int(item) > 0]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as conn:
            cur = conn.execute(
                f"""
                UPDATE signals
                SET status=?, updated_at=?
                WHERE id IN ({placeholders})
                """,
                (str(status), _utc_now_iso(), *ids),
            )
            conn.commit()
            return int(cur.rowcount or 0)

    def create_outbound_message(
        self,
        *,
        source: str,
        customer_id: str,
        thread_id: str,
        text: str,
        dispatch: dict[str, Any],
        signal_ids: list[int],
    ) -> dict[str, Any]:
        safe_text = self._normalize_text(text)
        if not safe_text:
            raise ValueError("text is required")
        safe_dispatch = self._json_dict(dispatch)
        if not safe_dispatch:
            raise ValueError("dispatch is required")
        now = _utc_now_iso()
        ids = [int(item) for item in signal_ids if int(item) > 0]
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO signal_outbox (
                    source, customer_id, thread_id, text,
                    dispatch_json, signal_ids_json, status, created_at, updated_at, sent_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, NULL)
                """,
                (
                    self._normalize_text(source),
                    self._normalize_text(customer_id),
                    self._normalize_text(thread_id),
                    safe_text,
                    json.dumps(safe_dispatch, ensure_ascii=False),
                    json.dumps(ids, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            conn.commit()
            outbox_id = int(cur.lastrowid)
        return {
            "id": outbox_id,
            "source": self._normalize_text(source),
            "customer_id": self._normalize_text(customer_id),
            "thread_id": self._normalize_text(thread_id),
            "text": safe_text,
            "dispatch": safe_dispatch,
            "signal_ids": ids,
            "status": "pending",
            "created_at": now,
        }

    def list_outbox(self, *, source: str = "", status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        query = (
            "SELECT id, source, customer_id, thread_id, text, dispatch_json, "
            "signal_ids_json, status, created_at, updated_at, sent_at "
            "FROM signal_outbox WHERE 1=1"
        )
        params: list[Any] = []
        safe_source = self._normalize_text(source)
        safe_status = self._normalize_text(status)
        if safe_source:
            query += " AND source=?"
            params.append(safe_source)
        if safe_status:
            query += " AND status=?"
            params.append(safe_status)
        query += " ORDER BY id ASC LIMIT ?"
        params.append(max(1, min(int(limit), 200)))
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            try:
                dispatch = json.loads(row["dispatch_json"] or "{}")
            except Exception:
                dispatch = {}
            try:
                signal_ids = json.loads(row["signal_ids_json"] or "[]")
            except Exception:
                signal_ids = []
            out.append(
                {
                    "id": int(row["id"]),
                    "source": str(row["source"]),
                    "customer_id": str(row["customer_id"]),
                    "thread_id": str(row["thread_id"]),
                    "text": str(row["text"] or ""),
                    "dispatch": dispatch if isinstance(dispatch, dict) else {},
                    "signal_ids": signal_ids if isinstance(signal_ids, list) else [],
                    "status": str(row["status"]),
                    "created_at": str(row["created_at"]),
                    "updated_at": str(row["updated_at"]),
                    "sent_at": str(row["sent_at"] or ""),
                }
            )
        return out

    def mark_outbound_sent(self, outbox_id: int) -> dict[str, Any] | None:
        safe_id = int(outbox_id)
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE signal_outbox
                SET status='sent', updated_at=?, sent_at=?
                WHERE id=?
                """,
                (now, now, safe_id),
            )
            conn.commit()
        matches = self.list_outbox(status="", limit=200)
        for row in matches:
            if int(row["id"]) == safe_id:
                return row
        return None
