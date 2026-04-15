"""Durable customer-scoped profile storage."""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from opentulpa.context.customer_profile_models import (
    CustomerProfileRecord,
    LegacyProfileImportSummary,
)


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
    """Store stable per-customer metadata (directive, timezone, locale)."""

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
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _utc_now_iso() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _optional_text(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _upsert(
        self,
        customer_id: str,
        *,
        directive_text: str | None = None,
        utc_offset: str | None = None,
        locale: str | None = None,
        source: str = "agent",
    ) -> CustomerProfileRecord:
        cid = str(customer_id or "").strip()
        if not cid:
            raise ValueError("customer_id is required")
        updated_at = self._utc_now_iso()
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT directive_text, utc_offset, locale
                FROM customer_profiles
                WHERE customer_id=?
                """,
                (cid,),
            ).fetchone()
            cur_directive = str(existing["directive_text"]) if existing and existing["directive_text"] is not None else None
            cur_offset = str(existing["utc_offset"]) if existing and existing["utc_offset"] is not None else None
            cur_locale = str(existing["locale"]) if existing and existing["locale"] is not None else None

            next_directive = cur_directive if directive_text is None else directive_text
            next_offset = cur_offset if utc_offset is None else utc_offset
            next_locale = cur_locale if locale is None else locale

            conn.execute(
                """
                INSERT INTO customer_profiles
                    (customer_id, directive_text, utc_offset, locale, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id)
                DO UPDATE SET
                    directive_text=excluded.directive_text,
                    utc_offset=excluded.utc_offset,
                    locale=excluded.locale,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (
                    cid,
                    next_directive,
                    next_offset,
                    next_locale,
                    str(source or "agent"),
                    updated_at,
                ),
            )
            conn.commit()
        return CustomerProfileRecord(
            customer_id=cid,
            directive_text=self._optional_text(next_directive),
            utc_offset=self._optional_text(next_offset),
            locale=self._optional_text(next_locale),
            source=str(source or "agent"),
            updated_at=updated_at,
        )

    def get_profile(self, customer_id: str) -> CustomerProfileRecord | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT customer_id, directive_text, utc_offset, locale, source, updated_at
                FROM customer_profiles
                WHERE customer_id=?
                """,
                (cid,),
            ).fetchone()
        if not row:
            return None
        return CustomerProfileRecord(
            customer_id=str(row["customer_id"]),
            directive_text=self._optional_text(row["directive_text"]),
            utc_offset=self._optional_text(row["utc_offset"]),
            locale=self._optional_text(row["locale"]),
            source=str(row["source"]),
            updated_at=str(row["updated_at"]),
        )

    def get_directive(self, customer_id: str) -> str | None:
        profile = self.get_profile(customer_id)
        if not profile:
            return None
        return profile.directive_text

    def set_directive(self, customer_id: str, directive: str, *, source: str = "agent") -> CustomerProfileRecord:
        text = str(directive or "").strip()
        if not text:
            raise ValueError("directive is required")
        return self._upsert(customer_id, directive_text=text, source=source)

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
        return profile.utc_offset

    def set_utc_offset(self, customer_id: str, utc_offset: str, *, source: str = "agent") -> CustomerProfileRecord:
        normalized = _normalize_utc_offset(utc_offset)
        return self._upsert(customer_id, utc_offset=normalized, source=source)

    def import_legacy(
        self,
        *,
        directives_db_path: Path | None = None,
        time_profiles_db_path: Path | None = None,
    ) -> LegacyProfileImportSummary:
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

        return LegacyProfileImportSummary(
            directives=imported_directives,
            utc_offsets=imported_offsets,
        )
