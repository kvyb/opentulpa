"""Durable customer-scoped profile storage."""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from opentulpa.context.customer_profile_models import (
    CustomerProfileRecord,
    IdentityBindingRecord,
    LegacyProfileImportSummary,
    ProfileIdentityRecord,
)
from opentulpa.persistence.sqlite import connect_sqlite


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
        return connect_sqlite(self.db_path, wal=True)

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
                CREATE TABLE IF NOT EXISTS customer_identity_aliases (
                    alias_user_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    storage_user_id TEXT NOT NULL,
                    alias_kind TEXT NOT NULL,
                    provider TEXT,
                    provider_user_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_customer_identity_aliases_provider
                    ON customer_identity_aliases(provider, provider_user_id)
                    WHERE provider IS NOT NULL AND provider_user_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_customer_identity_aliases_storage
                    ON customer_identity_aliases(storage_user_id);
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

    @staticmethod
    def _telegram_alias(telegram_user_id: str | int) -> str:
        tid = str(telegram_user_id or "").strip()
        if not tid:
            raise ValueError("telegram_user_id is required")
        return f"telegram_{tid}"

    def resolve_customer_id(self, customer_id: str) -> str:
        cid = str(customer_id or "").strip()
        if not cid:
            return ""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT storage_user_id
                FROM customer_identity_aliases
                WHERE alias_user_id=?
                """,
                (cid,),
            ).fetchone()
        return str(row["storage_user_id"]) if row else cid

    def resolve_telegram_customer_id(self, telegram_user_id: str | int) -> str:
        return self.resolve_customer_id(self._telegram_alias(telegram_user_id))

    def alias_user_ids(self, customer_id: str) -> list[str]:
        cid = str(customer_id or "").strip()
        if not cid:
            return []
        storage_user_id = self.resolve_customer_id(cid)
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT alias_user_id
                FROM customer_identity_aliases
                WHERE storage_user_id=?
                ORDER BY alias_kind, alias_user_id
                """,
                (storage_user_id,),
            ).fetchall()
        aliases = [str(row["alias_user_id"]) for row in rows]
        if cid not in aliases:
            aliases.insert(0, cid)
        if storage_user_id and storage_user_id not in aliases:
            aliases.append(storage_user_id)
        return aliases

    def _profile_exists(self, conn: sqlite3.Connection, customer_id: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM customer_profiles WHERE customer_id=?",
            (customer_id,),
        ).fetchone()
        return row is not None

    def _select_identity_storage(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        telegram_alias: str,
    ) -> str:
        assert user_id
        assert telegram_alias
        uid_row = conn.execute(
            "SELECT storage_user_id FROM customer_identity_aliases WHERE alias_user_id=?",
            (user_id,),
        ).fetchone()
        telegram_row = conn.execute(
            "SELECT storage_user_id FROM customer_identity_aliases WHERE alias_user_id=?",
            (telegram_alias,),
        ).fetchone()
        uid_storage = str(uid_row["storage_user_id"]) if uid_row else ""
        telegram_storage = str(telegram_row["storage_user_id"]) if telegram_row else ""
        uid_has_profile = self._profile_exists(conn, user_id)
        telegram_has_profile = self._profile_exists(conn, telegram_alias)
        if uid_storage and telegram_storage and uid_storage != telegram_storage:
            raise ValueError("identity aliases already point at different storage_user_id values")
        if uid_storage:
            if telegram_has_profile and telegram_alias != uid_storage:
                raise ValueError("generic and telegram profiles both exist; manual merge is required")
            return uid_storage
        if telegram_storage:
            if uid_has_profile and user_id != telegram_storage:
                raise ValueError("generic and telegram profiles both exist; manual merge is required")
            return telegram_storage
        if uid_has_profile and telegram_has_profile:
            raise ValueError("generic and telegram profiles both exist; manual merge is required")
        if uid_has_profile:
            return user_id
        if telegram_has_profile:
            return telegram_alias
        return user_id

    def _upsert_identity_alias(
        self,
        conn: sqlite3.Connection,
        *,
        alias_user_id: str,
        user_id: str,
        storage_user_id: str,
        alias_kind: str,
        provider: str | None,
        provider_user_id: str | None,
        now: str,
    ) -> None:
        assert alias_user_id
        assert storage_user_id
        conn.execute(
            """
            INSERT INTO customer_identity_aliases (
                alias_user_id, user_id, storage_user_id, alias_kind,
                provider, provider_user_id, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(alias_user_id)
            DO UPDATE SET
                user_id=excluded.user_id,
                storage_user_id=excluded.storage_user_id,
                alias_kind=excluded.alias_kind,
                provider=excluded.provider,
                provider_user_id=excluded.provider_user_id,
                updated_at=excluded.updated_at
            """,
            (
                alias_user_id,
                user_id,
                storage_user_id,
                alias_kind,
                provider,
                provider_user_id,
                now,
                now,
            ),
        )

    def bind_telegram_user_id(self, *, user_id: str, telegram_user_id: str | int) -> IdentityBindingRecord:
        uid = str(user_id or "").strip()
        telegram_id = str(telegram_user_id or "").strip()
        if not uid:
            raise ValueError("user_id is required")
        telegram_alias = self._telegram_alias(telegram_id)
        if uid == telegram_alias:
            raise ValueError("user_id must be generic, not telegram-derived")

        now = self._utc_now_iso()
        with self._conn() as conn:
            storage_user_id = self._select_identity_storage(
                conn,
                user_id=uid,
                telegram_alias=telegram_alias,
            )
            assert uid
            assert telegram_alias
            for alias_user_id, alias_kind, provider, provider_user_id in (
                (uid, "generic", None, None),
                (telegram_alias, "telegram", "telegram", telegram_id),
            ):
                self._upsert_identity_alias(
                    conn,
                    alias_user_id=alias_user_id,
                    user_id=uid,
                    storage_user_id=storage_user_id,
                    alias_kind=alias_kind,
                    provider=provider,
                    provider_user_id=provider_user_id,
                    now=now,
                )
            conn.commit()
        return IdentityBindingRecord(
            user_id=uid,
            alias_user_id=telegram_alias,
            storage_user_id=storage_user_id,
            alias_kind="telegram",
            provider="telegram",
            provider_user_id=telegram_id,
            updated_at=now,
        )

    def list_identity_bindings(self) -> list[IdentityBindingRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT alias_user_id, user_id, storage_user_id, alias_kind,
                       provider, provider_user_id, updated_at
                FROM customer_identity_aliases
                ORDER BY user_id, alias_kind, alias_user_id
                """
            ).fetchall()
        return [
            IdentityBindingRecord(
                user_id=str(row["user_id"]),
                alias_user_id=str(row["alias_user_id"]),
                storage_user_id=str(row["storage_user_id"]),
                alias_kind=str(row["alias_kind"]),
                provider=self._optional_text(row["provider"]),
                provider_user_id=self._optional_text(row["provider_user_id"]),
                updated_at=str(row["updated_at"]),
            )
            for row in rows
        ]

    def list_profiles(self) -> list[ProfileIdentityRecord]:
        bindings = self.list_identity_bindings()
        by_user: dict[str, ProfileIdentityRecord] = {}
        for binding in bindings:
            item = by_user.setdefault(
                binding.user_id,
                ProfileIdentityRecord(
                    user_id=binding.user_id,
                    storage_user_id=binding.storage_user_id,
                    aliases=[],
                ),
            )
            item.aliases.append(binding.alias_user_id)
            if binding.provider == "telegram":
                item.telegram_user_id = binding.provider_user_id

        known = set(by_user)
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT customer_id FROM customer_profiles ORDER BY customer_id"
            ).fetchall()
        for row in rows:
            cid = str(row["customer_id"])
            storage = self.resolve_customer_id(cid)
            if cid in known or any(cid in item.aliases for item in by_user.values()):
                continue
            by_user[cid] = ProfileIdentityRecord(
                user_id=cid,
                storage_user_id=storage,
                telegram_user_id=cid.removeprefix("telegram_") if cid.startswith("telegram_") else None,
                aliases=[cid],
            )

        return sorted(by_user.values(), key=lambda item: item.user_id)

    def _upsert(
        self,
        customer_id: str,
        *,
        directive_text: str | None = None,
        utc_offset: str | None = None,
        locale: str | None = None,
        source: str = "agent",
    ) -> CustomerProfileRecord:
        cid = self.resolve_customer_id(customer_id)
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
        cid = self.resolve_customer_id(customer_id)
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

    def list_customer_summaries(self) -> list[dict[str, str]]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT customer_id, updated_at
                FROM customer_profiles
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            {
                "customer_id": str(row["customer_id"]),
                "last_profile_at": str(row["updated_at"] or ""),
            }
            for row in rows
        ]

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
