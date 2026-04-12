"""Durable customer-scoped profile storage."""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _normalize_utc_offset(value: str) -> str:
    raw = str(value or "").strip().upper()
    m = re.fullmatch(r"([+-])(\d{2}):(\d{2})", raw)
    if not m:
        raise ValueError("utc_offset must match +HH:MM or -HH:MM")
    sign = -1 if m.group(1) == "-" else 1
    hours = int(m.group(2))
    minutes = int(m.group(3))
    if hours > 14 or minutes > 59:
        raise ValueError("utc_offset out of range")
    total = sign * (hours * 60 + minutes)
    if total < -12 * 60 or total > 14 * 60:
        raise ValueError("utc_offset out of supported range")
    return f"{m.group(1)}{hours:02d}:{minutes:02d}"


class CustomerProfileService:
    """Store stable per-customer metadata (directive, timezone, locale, lessons)."""

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
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    customer_id TEXT PRIMARY KEY,
                    directive_text TEXT,
                    utc_offset TEXT,
                    locale TEXT,
                    lessons_learnt TEXT,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "customer_profiles", "lessons_learnt", "TEXT")

    @staticmethod
    def _ensure_column(
        conn: sqlite3.Connection,
        table: str,
        column: str,
        column_sql: str,
    ) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        existing = {str(row["name"]).strip().lower() for row in rows if isinstance(row, sqlite3.Row)}
        if str(column).strip().lower() in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_sql}")
        conn.commit()

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    def _upsert(
        self,
        customer_id: str,
        *,
        directive_text: str | None = None,
        utc_offset: str | None = None,
        locale: str | None = None,
        lessons_learnt: str | None = None,
        source: str = "agent",
    ) -> None:
        cid = str(customer_id or "").strip()
        if not cid:
            raise ValueError("customer_id is required")
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT directive_text, utc_offset, locale, lessons_learnt
                FROM customer_profiles
                WHERE customer_id=?
                """,
                (cid,),
            ).fetchone()
            cur_directive = str(existing["directive_text"]) if existing and existing["directive_text"] is not None else None
            cur_offset = str(existing["utc_offset"]) if existing and existing["utc_offset"] is not None else None
            cur_locale = str(existing["locale"]) if existing and existing["locale"] is not None else None
            cur_lessons = str(existing["lessons_learnt"]) if existing and existing["lessons_learnt"] is not None else None

            next_directive = cur_directive if directive_text is None else directive_text
            next_offset = cur_offset if utc_offset is None else utc_offset
            next_locale = cur_locale if locale is None else locale
            next_lessons = cur_lessons if lessons_learnt is None else lessons_learnt

            conn.execute(
                """
                INSERT INTO customer_profiles
                    (customer_id, directive_text, utc_offset, locale, lessons_learnt, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id)
                DO UPDATE SET
                    directive_text=excluded.directive_text,
                    utc_offset=excluded.utc_offset,
                    locale=excluded.locale,
                    lessons_learnt=excluded.lessons_learnt,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    cid,
                    next_directive,
                    next_offset,
                    next_locale,
                    next_lessons,
                    str(source or "agent"),
                    self._utc_now_iso(),
                ),
            )
            conn.commit()

    def get_profile(self, customer_id: str) -> dict[str, Any] | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT customer_id, directive_text, utc_offset, locale, source, updated_at, lessons_learnt
                FROM customer_profiles
                WHERE customer_id=?
                """,
                (cid,),
            ).fetchone()
        if not row:
            return None
        return {
            "customer_id": str(row["customer_id"]),
            "directive_text": str(row["directive_text"]) if row["directive_text"] is not None else None,
            "utc_offset": str(row["utc_offset"]) if row["utc_offset"] is not None else None,
            "locale": str(row["locale"]) if row["locale"] is not None else None,
            "lessons_learnt": str(row["lessons_learnt"]) if row["lessons_learnt"] is not None else None,
            "source": str(row["source"]),
            "updated_at": str(row["updated_at"]),
        }

    def get_directive(self, customer_id: str) -> str | None:
        profile = self.get_profile(customer_id)
        if not profile:
            return None
        text = str(profile.get("directive_text") or "").strip()
        return text or None

    def set_directive(self, customer_id: str, directive: str, *, source: str = "agent") -> None:
        text = str(directive or "").strip()
        if not text:
            raise ValueError("directive is required")
        self._upsert(customer_id, directive_text=text, source=source)

    def clear_directive(self, customer_id: str, *, source: str = "agent") -> bool:
        cid = str(customer_id or "").strip()
        if not cid:
            return False
        profile = self.get_profile(cid)
        if profile is None:
            return False
        self._upsert(cid, directive_text="", source=source)
        return True

    def get_utc_offset(self, customer_id: str) -> str | None:
        profile = self.get_profile(customer_id)
        if not profile:
            return None
        offset = str(profile.get("utc_offset") or "").strip()
        return offset or None

    def set_utc_offset(self, customer_id: str, utc_offset: str, *, source: str = "agent") -> str:
        normalized = _normalize_utc_offset(utc_offset)
        self._upsert(customer_id, utc_offset=normalized, source=source)
        return normalized

    def get_lessons_learnt(self, customer_id: str) -> str | None:
        profile = self.get_profile(customer_id)
        if not profile:
            return None
        text = str(profile.get("lessons_learnt") or "").strip()
        return text or None

    def set_lessons_learnt(self, customer_id: str, lessons: str, *, source: str = "agent") -> str:
        text = str(lessons or "").strip()
        if not text:
            raise ValueError("lessons_learnt is required")
        self._upsert(customer_id, lessons_learnt=text, source=source)
        return text

    def append_lesson(
        self,
        customer_id: str,
        lesson: str,
        *,
        source: str = "agent",
        max_chars: int = 20000,
    ) -> str:
        safe_lesson = str(lesson or "").strip()
        if not safe_lesson:
            raise ValueError("lesson is required")
        current = str(self.get_lessons_learnt(customer_id) or "").strip()
        merged = f"{current}\n{safe_lesson}".strip() if current else safe_lesson
        safe_limit = max(500, min(int(max_chars), 200000))
        if len(merged) > safe_limit:
            merged = merged[-safe_limit:]
        self._upsert(customer_id, lessons_learnt=merged, source=source)
        return merged

    def clear_lessons_learnt(self, customer_id: str, *, source: str = "agent") -> bool:
        cid = str(customer_id or "").strip()
        if not cid:
            return False
        profile = self.get_profile(cid)
        if profile is None:
            return False
        self._upsert(cid, lessons_learnt="", source=source)
        return True

    def import_legacy(
        self,
        *,
        directives_db_path: Path | None = None,
        time_profiles_db_path: Path | None = None,
    ) -> dict[str, int]:
        """Best-effort one-way import from legacy stores."""
        imported_directives = 0
        imported_offsets = 0

        if directives_db_path and directives_db_path.exists():
            with suppress(Exception):
                with sqlite3.connect(directives_db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """
                        SELECT customer_id, directive_text, source
                        FROM customer_directives
                        """
                    ).fetchall()
                for row in rows:
                    cid = str(row["customer_id"] or "").strip()
                    directive = str(row["directive_text"] or "").strip()
                    source = str(row["source"] or "legacy_directives")
                    if not cid or not directive:
                        continue
                    self.set_directive(cid, directive, source=source)
                    imported_directives += 1

        if time_profiles_db_path and time_profiles_db_path.exists():
            with suppress(Exception):
                with sqlite3.connect(time_profiles_db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """
                        SELECT customer_id, utc_offset, source
                        FROM customer_time_profiles
                        """
                    ).fetchall()
                for row in rows:
                    cid = str(row["customer_id"] or "").strip()
                    offset = str(row["utc_offset"] or "").strip()
                    source = str(row["source"] or "legacy_time_profiles")
                    if not cid or not offset:
                        continue
                    with suppress(Exception):
                        self.set_utc_offset(cid, offset, source=source)
                        imported_offsets += 1

        return {
            "directives": imported_directives,
            "utc_offsets": imported_offsets,
        }
