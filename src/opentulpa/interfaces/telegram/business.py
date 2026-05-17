"""Telegram Business connection and message storage helpers."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.persistence.sqlite import connect_sqlite


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _epoch_to_iso(value: Any) -> str:
    try:
        timestamp = int(value)
    except Exception:
        return ""
    with suppress(Exception):
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    return ""


def _message_text(message: dict[str, Any]) -> str:
    return str(message.get("text") or message.get("caption") or "").strip()


class TelegramBusinessService:
    """Persist Telegram Business connections and normalized message state."""

    def __init__(
        self,
        *,
        db_path: Path,
        owner_customer_id: str | None = None,
        resolve_customer_id: Callable[[str], str] | None = None,
    ) -> None:
        self._db_path = db_path.resolve()
        self._owner_customer_id = str(owner_customer_id or "").strip()
        self._resolve_customer_id_callback = resolve_customer_id
        self.client: Any | None = None
        self._init_db()
        self._bind_existing_connections_to_owner()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self._db_path, wal=True)

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS telegram_business_connections (
                    business_connection_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    user_chat_id TEXT NOT NULL,
                    is_enabled INTEGER NOT NULL,
                    rights_json TEXT NOT NULL,
                    connection_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tg_business_connections_customer
                    ON telegram_business_connections(customer_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS telegram_business_messages (
                    business_connection_id TEXT NOT NULL,
                    customer_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    date_iso TEXT NOT NULL,
                    from_user_id TEXT,
                    from_username TEXT,
                    sender_role TEXT NOT NULL,
                    text TEXT,
                    deleted INTEGER NOT NULL,
                    raw_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (business_connection_id, chat_id, message_id)
                );
                CREATE INDEX IF NOT EXISTS idx_tg_business_messages_customer
                    ON telegram_business_messages(customer_id, business_connection_id, chat_id, date_iso DESC);
                """
            )

    def _customer_id_from_connection(self, connection: dict[str, Any]) -> str:
        if self._owner_customer_id:
            return self._resolve_customer_id(self._owner_customer_id)
        user = _safe_dict(connection.get("user"))
        user_id = str(user.get("id", "") or "").strip()
        if not user_id:
            return ""
        return self._resolve_customer_id(f"telegram_{user_id}")

    def _resolve_customer_id(self, customer_id: str) -> str:
        safe_customer = str(customer_id or "").strip()
        if not safe_customer or self._resolve_customer_id_callback is None:
            return safe_customer
        resolved = str(self._resolve_customer_id_callback(safe_customer) or "").strip()
        return resolved or safe_customer

    def _rebind_connection_customer_id(
        self,
        *,
        business_connection_id: str,
        customer_id: str,
    ) -> None:
        safe_business_connection_id = str(business_connection_id or "").strip()
        safe_customer = str(customer_id or "").strip()
        if not safe_business_connection_id or not safe_customer:
            return
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE telegram_business_connections
                SET customer_id = ?, updated_at = ?
                WHERE business_connection_id = ?
                """,
                (safe_customer, now, safe_business_connection_id),
            )
            conn.execute(
                """
                UPDATE telegram_business_messages
                SET customer_id = ?, updated_at = ?
                WHERE business_connection_id = ?
                """,
                (safe_customer, now, safe_business_connection_id),
            )
            conn.commit()

    def _bind_existing_connections_to_owner(self) -> None:
        if not self._owner_customer_id:
            return
        with self._conn() as conn:
            conn.execute(
                "UPDATE telegram_business_connections SET customer_id = ?",
                (self._owner_customer_id,),
            )
            conn.execute(
                "UPDATE telegram_business_messages SET customer_id = ?",
                (self._owner_customer_id,),
            )
            conn.commit()

    @staticmethod
    def _connection_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "business_connection_id": str(row["business_connection_id"]),
            "customer_id": str(row["customer_id"]),
            "user_id": str(row["user_id"]),
            "user_chat_id": str(row["user_chat_id"]),
            "is_enabled": bool(row["is_enabled"]),
            "rights": json.loads(row["rights_json"] or "{}"),
            "connection": json.loads(row["connection_json"] or "{}"),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    @staticmethod
    def _message_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "business_connection_id": str(row["business_connection_id"]),
            "customer_id": str(row["customer_id"]),
            "chat_id": str(row["chat_id"]),
            "message_id": str(row["message_id"]),
            "date_iso": str(row["date_iso"]),
            "from_user_id": str(row["from_user_id"]) if row["from_user_id"] else "",
            "from_username": str(row["from_username"]) if row["from_username"] else "",
            "sender_role": str(row["sender_role"]),
            "text": str(row["text"]) if row["text"] else "",
            "deleted": bool(row["deleted"]),
            "raw": json.loads(row["raw_json"] or "{}"),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    def upsert_connection(self, connection: dict[str, Any]) -> dict[str, Any]:
        safe_connection = _safe_dict(connection)
        business_connection_id = str(safe_connection.get("id", "") or "").strip()
        user = _safe_dict(safe_connection.get("user"))
        user_id = str(user.get("id", "") or "").strip()
        user_chat_id = str(safe_connection.get("user_chat_id", "") or "").strip()
        customer_id = self._customer_id_from_connection(safe_connection)
        if not business_connection_id or not user_id or not user_chat_id or not customer_id:
            raise ValueError("invalid Telegram Business connection payload")
        rights = _safe_dict(safe_connection.get("rights"))
        is_enabled = bool(safe_connection.get("is_enabled", False))
        now = _utc_now_iso()
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT created_at
                FROM telegram_business_connections
                WHERE business_connection_id = ?
                """,
                (business_connection_id,),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO telegram_business_connections (
                    business_connection_id,
                    customer_id,
                    user_id,
                    user_chat_id,
                    is_enabled,
                    rights_json,
                    connection_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(business_connection_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    user_id=excluded.user_id,
                    user_chat_id=excluded.user_chat_id,
                    is_enabled=excluded.is_enabled,
                    rights_json=excluded.rights_json,
                    connection_json=excluded.connection_json,
                    updated_at=excluded.updated_at
                """,
                (
                    business_connection_id,
                    customer_id,
                    user_id,
                    user_chat_id,
                    1 if is_enabled else 0,
                    _json_dumps(rights),
                    _json_dumps(safe_connection),
                    created_at,
                    now,
                ),
            )
            conn.commit()
        return self.get_connection(business_connection_id) or {}

    def get_connection(self, business_connection_id: str) -> dict[str, Any] | None:
        safe_id = str(business_connection_id or "").strip()
        if not safe_id:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM telegram_business_connections
                WHERE business_connection_id = ?
                """,
                (safe_id,),
            ).fetchone()
        return self._connection_row_to_dict(row) if row else None

    def status(self, *, customer_id: str) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        if not safe_customer:
            return {"ok": True, "connected": False, "connections": []}
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM telegram_business_connections
                WHERE customer_id = ?
                ORDER BY updated_at DESC
                """,
                (safe_customer,),
            ).fetchall()
        items = [self._connection_row_to_dict(row) for row in rows]
        return {
            "ok": True,
            "connected": any(bool(item.get("is_enabled")) for item in items),
            "connections": items,
        }

    def list_customer_summaries(self) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT customer_id,
                       COUNT(*) AS business_connection_count,
                       MAX(updated_at) AS last_business_at,
                       MAX(is_enabled) AS any_enabled
                FROM telegram_business_connections
                GROUP BY customer_id
                ORDER BY last_business_at DESC
                """
            ).fetchall()
        return [
            {
                "customer_id": str(row["customer_id"]),
                "business_connection_count": int(row["business_connection_count"] or 0),
                "telegram_business_connected": bool(row["any_enabled"]),
                "last_business_at": str(row["last_business_at"] or ""),
            }
            for row in rows
        ]

    @staticmethod
    def _sender_role(message: dict[str, Any]) -> str:
        if isinstance(message.get("sender_business_bot"), dict):
            return "assistant"
        return "customer"

    def upsert_message(
        self,
        *,
        business_connection_id: str,
        customer_id: str,
        message: dict[str, Any],
        deleted: bool = False,
    ) -> dict[str, Any]:
        safe_message = _safe_dict(message)
        chat = _safe_dict(safe_message.get("chat"))
        sender = _safe_dict(safe_message.get("from"))
        chat_id = str(chat.get("id", "") or "").strip()
        message_id = str(safe_message.get("message_id", "") or "").strip()
        if not business_connection_id or not customer_id or not chat_id or not message_id:
            raise ValueError("invalid Telegram Business message payload")
        now = _utc_now_iso()
        date_iso = _epoch_to_iso(safe_message.get("date")) or now
        sender_role = self._sender_role(safe_message)
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT created_at
                FROM telegram_business_messages
                WHERE business_connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (business_connection_id, chat_id, message_id),
            ).fetchone()
            created_at = str(existing["created_at"]) if existing else now
            conn.execute(
                """
                INSERT INTO telegram_business_messages (
                    business_connection_id,
                    customer_id,
                    chat_id,
                    message_id,
                    date_iso,
                    from_user_id,
                    from_username,
                    sender_role,
                    text,
                    deleted,
                    raw_json,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(business_connection_id, chat_id, message_id) DO UPDATE SET
                    customer_id=excluded.customer_id,
                    date_iso=excluded.date_iso,
                    from_user_id=excluded.from_user_id,
                    from_username=excluded.from_username,
                    sender_role=excluded.sender_role,
                    text=excluded.text,
                    deleted=excluded.deleted,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                (
                    business_connection_id,
                    customer_id,
                    chat_id,
                    message_id,
                    date_iso,
                    str(sender.get("id", "") or "").strip() or None,
                    str(sender.get("username", "") or "").strip() or None,
                    sender_role,
                    _message_text(safe_message) or None,
                    1 if deleted else 0,
                    _json_dumps(safe_message),
                    created_at,
                    now,
                ),
            )
            conn.commit()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM telegram_business_messages
                WHERE business_connection_id = ? AND chat_id = ? AND message_id = ?
                """,
                (business_connection_id, chat_id, message_id),
            ).fetchone()
        return self._message_row_to_dict(row) if row else {}

    def mark_deleted_messages(
        self,
        *,
        business_connection_id: str,
        customer_id: str,
        chat_id: str,
        message_ids: list[Any],
    ) -> int:
        safe_ids = [str(item or "").strip() for item in message_ids if str(item or "").strip()]
        if not business_connection_id or not customer_id or not chat_id or not safe_ids:
            return 0
        placeholders = ",".join("?" for _ in safe_ids)
        with self._conn() as conn:
            result = conn.execute(
                f"""
                UPDATE telegram_business_messages
                SET deleted = 1, updated_at = ?
                WHERE business_connection_id = ?
                  AND customer_id = ?
                  AND chat_id = ?
                  AND message_id IN ({placeholders})
                """,
                (_utc_now_iso(), business_connection_id, customer_id, chat_id, *safe_ids),
            )
            conn.commit()
        return int(getattr(result, "rowcount", 0) or 0)

    def list_conversations(
        self,
        *,
        customer_id: str,
        business_connection_id: str,
        limit: int = 10,
        chat_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_business_connection_id = str(business_connection_id or "").strip()
        safe_chat_ids = [str(item or "").strip() for item in (chat_ids or []) if str(item or "").strip()]
        if not safe_customer or not safe_business_connection_id:
            return {"ok": True, "items": []}
        safe_limit = max(1, min(int(limit or 10), 50))
        query = """
            SELECT *
            FROM telegram_business_messages
            WHERE customer_id = ? AND business_connection_id = ?
        """
        params: list[Any] = [safe_customer, safe_business_connection_id]
        if safe_chat_ids:
            placeholders = ",".join("?" for _ in safe_chat_ids)
            query += f" AND chat_id IN ({placeholders})"
            params.extend(safe_chat_ids)
        query += " ORDER BY chat_id ASC, date_iso DESC, message_id DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            item = self._message_row_to_dict(row)
            grouped.setdefault(str(item["chat_id"]), []).append(item)
        items: list[dict[str, Any]] = []
        for chat_id, messages in grouped.items():
            messages.sort(key=lambda item: (str(item.get("date_iso", "")), str(item.get("message_id", ""))), reverse=True)
            latest = next((item for item in messages if not item.get("deleted")), None)
            latest_inbound = next(
                (item for item in messages if not item.get("deleted") and str(item.get("sender_role")) == "customer"),
                None,
            )
            latest_outbound = next(
                (item for item in messages if not item.get("deleted") and str(item.get("sender_role")) == "assistant"),
                None,
            )
            if latest is None or (latest_inbound is None and latest_outbound is None):
                continue
            items.append(
                {
                    "conversation_id": chat_id,
                    "business_connection_id": safe_business_connection_id,
                    "recipient_id": chat_id,
                    "latest_message_id": str(latest.get("message_id", "") or ""),
                    "latest_message_created_time": str(latest.get("date_iso", "") or ""),
                    "latest_inbound_message_id": str((latest_inbound or {}).get("message_id", "") or ""),
                    "latest_inbound_message_created_time": str((latest_inbound or {}).get("date_iso", "") or ""),
                    "latest_inbound_message_text_preview": str((latest_inbound or {}).get("text", "") or "")[:280],
                    "latest_inbound_sender_id": str((latest_inbound or {}).get("from_user_id", "") or ""),
                    "latest_inbound_sender_username": str((latest_inbound or {}).get("from_username", "") or ""),
                    "latest_outbound_message_id": str((latest_outbound or {}).get("message_id", "") or ""),
                    "latest_outbound_message_created_time": str((latest_outbound or {}).get("date_iso", "") or ""),
                    "conversation_updated_time": str(latest.get("date_iso", "") or ""),
                }
            )
        items.sort(
            key=lambda item: (
                str(item.get("conversation_updated_time", "")),
                str(item.get("conversation_id", "")),
            ),
            reverse=True,
        )
        return {"ok": True, "items": items[:safe_limit]}

    def get_conversation(
        self,
        *,
        customer_id: str,
        business_connection_id: str,
        conversation_id: str,
    ) -> dict[str, Any]:
        safe_customer = str(customer_id or "").strip()
        safe_business_connection_id = str(business_connection_id or "").strip()
        safe_conversation_id = str(conversation_id or "").strip()
        if not safe_customer or not safe_business_connection_id or not safe_conversation_id:
            return {"ok": False, "error": "conversation not found"}
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM telegram_business_messages
                WHERE customer_id = ? AND business_connection_id = ? AND chat_id = ?
                ORDER BY date_iso ASC, message_id ASC
                """,
                (safe_customer, safe_business_connection_id, safe_conversation_id),
            ).fetchall()
        if not rows:
            return {"ok": False, "error": "conversation not found"}
        messages = [self._message_row_to_dict(row) for row in rows if not bool(row["deleted"])]
        summary_payload = self.list_conversations(
            customer_id=safe_customer,
            business_connection_id=safe_business_connection_id,
            limit=200,
            chat_ids=[safe_conversation_id],
        )
        items = _safe_list(summary_payload.get("items"))
        summary = _safe_dict(items[0]) if items else {}
        return {
            "ok": True,
            "summary": summary,
            "conversation": {
                "conversation_id": safe_conversation_id,
                "business_connection_id": safe_business_connection_id,
                "messages": messages,
            },
        }

    def ingest_update(self, body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            return {"handled": False}

        connection = body.get("business_connection")
        if isinstance(connection, dict):
            record = self.upsert_connection(connection)
            return {
                "handled": True,
                "kind": "business_connection",
                "customer_id": str(record.get("customer_id", "") or ""),
                "business_connection_id": str(record.get("business_connection_id", "") or ""),
                "user_chat_id": str(record.get("user_chat_id", "") or ""),
                "trigger_workflows": False,
            }

        for key in ("business_message", "edited_business_message"):
            message = body.get(key)
            if not isinstance(message, dict):
                continue
            business_connection_id = str(message.get("business_connection_id", "") or "").strip()
            connection_record = self.get_connection(business_connection_id)
            if connection_record is None:
                return {
                    "handled": True,
                    "kind": key,
                    "business_connection_id": business_connection_id,
                    "customer_id": "",
                    "user_chat_id": "",
                    "trigger_workflows": False,
                }
            customer_id = self._resolve_customer_id(str(connection_record.get("customer_id", "") or ""))
            if customer_id and customer_id != str(connection_record.get("customer_id", "") or ""):
                self._rebind_connection_customer_id(
                    business_connection_id=business_connection_id,
                    customer_id=customer_id,
                )
            record = self.upsert_message(
                business_connection_id=business_connection_id,
                customer_id=customer_id,
                message=message,
            )
            return {
                "handled": True,
                "kind": key,
                "business_connection_id": business_connection_id,
                "customer_id": customer_id,
                "user_chat_id": str(connection_record.get("user_chat_id", "") or ""),
                "chat_id": str(record.get("chat_id", "") or ""),
                "message_id": str(record.get("message_id", "") or ""),
                "trigger_workflows": str(record.get("sender_role", "") or "") == "customer" and not bool(record.get("deleted")),
            }

        deleted = body.get("deleted_business_messages")
        if isinstance(deleted, dict):
            business_connection_id = str(deleted.get("business_connection_id", "") or "").strip()
            connection_record = self.get_connection(business_connection_id)
            customer_id = str((connection_record or {}).get("customer_id", "") or "")
            user_chat_id = str((connection_record or {}).get("user_chat_id", "") or "")
            chat = _safe_dict(deleted.get("chat"))
            chat_id = str(chat.get("id", "") or "").strip()
            deleted_count = self.mark_deleted_messages(
                business_connection_id=business_connection_id,
                customer_id=customer_id,
                chat_id=chat_id,
                message_ids=_safe_list(deleted.get("message_ids")),
            )
            return {
                "handled": True,
                "kind": "deleted_business_messages",
                "business_connection_id": business_connection_id,
                "customer_id": customer_id,
                "user_chat_id": user_chat_id,
                "chat_id": chat_id,
                "deleted_count": deleted_count,
                "trigger_workflows": False,
            }

        return {"handled": False}
