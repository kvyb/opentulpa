"""Atomic tenant-scoped SQLite repository for villa inventory."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite
from opentulpa.villas.models import VillaImportResult, VillaRecord

_SOURCE_COLUMNS = (
    "property_name",
    "location",
    "owner_agency",
    "property_type",
    "bedrooms",
    "bathrooms",
    "monthly_idr",
    "yearly_idr",
    "weekly_idr",
    "daily_idr",
    "availability_text",
    "pet_friendly",
    "pool",
    "parking",
    "construction_text",
    "deposit_monthly_idr",
    "deposit_yearly_idr",
    "commission_text",
    "included_text",
    "excluded_text",
    "map_link",
    "source_sheet",
    "raw_notes",
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _db_bool(value: bool | None) -> int | None:
    return None if value is None else int(value)


class VillaRepository:
    """Own the separate villas.db and preserve manual state across source imports."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        connection = connect_sqlite(self.db_path, wal=True)
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _init_db(self) -> None:
        with self._conn() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS import_runs (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    sheet_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
                    parsed_count INTEGER NOT NULL DEFAULT 0,
                    inserted_count INTEGER NOT NULL DEFAULT 0,
                    updated_count INTEGER NOT NULL DEFAULT 0,
                    unchanged_count INTEGER NOT NULL DEFAULT 0,
                    missing_count INTEGER NOT NULL DEFAULT 0,
                    error_code TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_import_runs_success_source
                    ON import_runs (tenant_id, file_id, sheet_name, source_sha256)
                    WHERE status='succeeded';

                CREATE TABLE IF NOT EXISTS villas (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    identity_fingerprint TEXT NOT NULL,
                    source_status TEXT NOT NULL DEFAULT 'active'
                        CHECK (source_status IN ('active', 'missing')),
                    manual_status TEXT,
                    manual_overrides_json TEXT NOT NULL DEFAULT '{}',
                    property_name TEXT NOT NULL,
                    location TEXT NOT NULL,
                    owner_agency TEXT NOT NULL,
                    property_type TEXT NOT NULL,
                    bedrooms REAL,
                    bathrooms REAL,
                    monthly_idr INTEGER,
                    yearly_idr INTEGER,
                    weekly_idr INTEGER,
                    daily_idr INTEGER,
                    availability_text TEXT NOT NULL,
                    pet_friendly INTEGER,
                    pool INTEGER,
                    parking INTEGER,
                    construction_text TEXT NOT NULL,
                    deposit_monthly_idr INTEGER,
                    deposit_yearly_idr INTEGER,
                    commission_text TEXT NOT NULL,
                    included_text TEXT NOT NULL,
                    excluded_text TEXT NOT NULL,
                    map_link TEXT NOT NULL,
                    source_sheet TEXT NOT NULL,
                    raw_notes TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    last_import_run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id),
                    UNIQUE (tenant_id, identity_fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_villas_tenant_status
                    ON villas (tenant_id, source_status, manual_status);
                CREATE INDEX IF NOT EXISTS idx_villas_tenant_location
                    ON villas (tenant_id, location);

                CREATE TABLE IF NOT EXISTS villa_source_records (
                    tenant_id TEXT NOT NULL,
                    villa_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    sheet_name TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    source_row INTEGER NOT NULL,
                    source_hash TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    first_import_run_id TEXT NOT NULL,
                    last_import_run_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, file_id, sheet_name, source_key),
                    FOREIGN KEY (tenant_id, villa_id)
                        REFERENCES villas (tenant_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_villa_source_records_villa
                    ON villa_source_records (tenant_id, villa_id);
                """
            )
            connection.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
                (_utc_now_iso(),),
            )
            # Version 2 makes replay detection independent of the FileVault upload ID.
            # Equivalent workbook content uploaded again remains one successful import.
            if connection.execute(
                "SELECT 1 FROM schema_migrations WHERE version=2"
            ).fetchone() is None:
                connection.execute("DROP INDEX IF EXISTS idx_import_runs_success_source")
                connection.execute(
                    """
                    CREATE UNIQUE INDEX idx_import_runs_success_source
                    ON import_runs (tenant_id, sheet_name, source_sha256)
                    WHERE status='succeeded'
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at) VALUES (2, ?)",
                    (_utc_now_iso(),),
                )
            connection.commit()

    @staticmethod
    def _result_from_row(row: sqlite3.Row, *, replayed: bool) -> VillaImportResult:
        return VillaImportResult(
            import_run_id=str(row["id"]),
            file_id=str(row["file_id"]),
            filename=str(row["filename"]),
            sheet_name=str(row["sheet_name"]),
            source_sha256=str(row["source_sha256"]),
            parsed_count=int(row["parsed_count"]),
            inserted_count=int(row["inserted_count"]),
            updated_count=int(row["updated_count"]),
            unchanged_count=int(row["unchanged_count"]),
            missing_count=int(row["missing_count"]),
            replayed=replayed,
        )

    @staticmethod
    def _source_values(record: VillaRecord) -> tuple[Any, ...]:
        values = record.as_dict()
        values["pet_friendly"] = _db_bool(record.pet_friendly)
        values["pool"] = _db_bool(record.pool)
        values["parking"] = _db_bool(record.parking)
        return tuple(values[column] for column in _SOURCE_COLUMNS)

    def import_records(
        self,
        *,
        tenant_id: str,
        file_id: str,
        filename: str,
        sheet_name: str,
        source_sha256: str,
        records: list[VillaRecord],
    ) -> VillaImportResult:
        tenant = str(tenant_id or "").strip()
        source_file = str(file_id or "").strip()
        if not tenant or not source_file or not records:
            raise ValueError("tenant_id, file_id, and records are required")
        now = _utc_now_iso()
        with self._conn() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute(
                """
                SELECT * FROM import_runs
                WHERE tenant_id=? AND sheet_name=?
                    AND source_sha256=? AND status='succeeded'
                """,
                (tenant, sheet_name, source_sha256),
            ).fetchone()
            existing_scope_rows = connection.execute(
                """
                SELECT v.identity_fingerprint, v.source_hash, v.source_status
                FROM villas AS v
                WHERE v.tenant_id=? AND EXISTS (
                    SELECT 1 FROM villa_source_records AS source
                    WHERE source.tenant_id=v.tenant_id
                        AND source.villa_id=v.id
                        AND source.sheet_name=?
                )
                """,
                (tenant, sheet_name),
            ).fetchall()
            expected_state = {
                record.identity_fingerprint: record.source_hash for record in records
            }
            actual_active_state = {
                str(row["identity_fingerprint"]): str(row["source_hash"])
                for row in existing_scope_rows
                if str(row["source_status"]) == "active"
            }
            if previous is not None and actual_active_state == expected_state:
                connection.commit()
                return self._result_from_row(previous, replayed=True)

            run_id = new_short_id("vimport")
            inserted = 0
            updated = 0
            unchanged = 0
            active_villa_ids: set[str] = set()
            source_insert_sql = f"""
                INSERT INTO villas (
                    tenant_id, id, identity_fingerprint, source_status,
                    manual_status, manual_overrides_json,
                    {', '.join(_SOURCE_COLUMNS)}, source_hash,
                    first_seen_at, last_seen_at, last_import_run_id, created_at, updated_at
                ) VALUES (
                    ?, ?, ?, 'active', NULL, '{{}}',
                    {', '.join('?' for _ in _SOURCE_COLUMNS)}, ?,
                    ?, ?, ?, ?, ?
                )
            """
            source_update_sql = f"""
                UPDATE villas SET
                    source_status='active',
                    {', '.join(f'{column}=?' for column in _SOURCE_COLUMNS)},
                    source_hash=?, last_seen_at=?, last_import_run_id=?, updated_at=?
                WHERE tenant_id=? AND id=?
            """
            for record in records:
                existing = connection.execute(
                    """
                    SELECT id, source_hash, source_status FROM villas
                    WHERE tenant_id=? AND identity_fingerprint=?
                    """,
                    (tenant, record.identity_fingerprint),
                ).fetchone()
                if existing is None:
                    villa_id = new_short_id("villa", suffix_chars=8)
                    connection.execute(
                        source_insert_sql,
                        (
                            tenant,
                            villa_id,
                            record.identity_fingerprint,
                            *self._source_values(record),
                            record.source_hash,
                            now,
                            now,
                            run_id,
                            now,
                            now,
                        ),
                    )
                    inserted += 1
                else:
                    villa_id = str(existing["id"])
                    changed = (
                        str(existing["source_hash"]) != record.source_hash
                        or str(existing["source_status"]) != "active"
                    )
                    connection.execute(
                        source_update_sql,
                        (
                            *self._source_values(record),
                            record.source_hash,
                            now,
                            run_id,
                            now,
                            tenant,
                            villa_id,
                        ),
                    )
                    if changed:
                        updated += 1
                    else:
                        unchanged += 1
                active_villa_ids.add(villa_id)
                source_json = json.dumps(
                    record.source_values,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                connection.execute(
                    """
                    INSERT INTO villa_source_records (
                        tenant_id, villa_id, file_id, sheet_name, source_key, source_row,
                        source_hash, source_json, first_import_run_id, last_import_run_id,
                        first_seen_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(tenant_id, file_id, sheet_name, source_key) DO UPDATE SET
                        villa_id=excluded.villa_id,
                        source_row=excluded.source_row,
                        source_hash=excluded.source_hash,
                        source_json=excluded.source_json,
                        last_import_run_id=excluded.last_import_run_id,
                        last_seen_at=excluded.last_seen_at
                    """,
                    (
                        tenant,
                        villa_id,
                        source_file,
                        sheet_name,
                        record.source_key,
                        record.source_row,
                        record.source_hash,
                        source_json,
                        run_id,
                        run_id,
                        now,
                        now,
                    ),
                )

            current_ids = {
                str(row["id"])
                for row in connection.execute(
                    """
                    SELECT DISTINCT v.id
                    FROM villas AS v
                    JOIN villa_source_records AS source
                        ON source.tenant_id=v.tenant_id AND source.villa_id=v.id
                    WHERE v.tenant_id=? AND source.sheet_name=?
                        AND v.source_status='active'
                    """,
                    (tenant, sheet_name),
                ).fetchall()
            }
            missing_ids = current_ids - active_villa_ids
            if missing_ids:
                placeholders = ",".join("?" for _ in missing_ids)
                connection.execute(
                    f"""
                    UPDATE villas SET source_status='missing', updated_at=?
                    WHERE tenant_id=? AND id IN ({placeholders})
                    """,
                    (now, tenant, *sorted(missing_ids)),
                )
            if previous is None:
                connection.execute(
                    """
                    INSERT INTO import_runs (
                        tenant_id, id, file_id, filename, sheet_name, source_sha256,
                        status, parsed_count, inserted_count, updated_count,
                        unchanged_count, missing_count, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'succeeded', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant,
                        run_id,
                        source_file,
                        filename,
                        sheet_name,
                        source_sha256,
                        len(records),
                        inserted,
                        updated,
                        unchanged,
                        len(missing_ids),
                        now,
                        now,
                    ),
                )
                result_run_id = run_id
            else:
                # Historical content can be replayed after later imports changed state.
                # Reconcile it without inserting a duplicate content-unique audit row.
                result_run_id = str(previous["id"])
            connection.commit()
            row = connection.execute(
                "SELECT * FROM import_runs WHERE tenant_id=? AND id=?",
                (tenant, result_run_id),
            ).fetchone()
            if row is None:
                raise RuntimeError("villa import summary was not persisted")
            return self._result_from_row(row, replayed=previous is not None)

    def counts(self, *, tenant_id: str) -> dict[str, int]:
        tenant = str(tenant_id or "").strip()
        with self._conn() as connection:
            total = connection.execute(
                "SELECT COUNT(*) FROM villas WHERE tenant_id=?",
                (tenant,),
            ).fetchone()[0]
            active = connection.execute(
                "SELECT COUNT(*) FROM villas WHERE tenant_id=? AND source_status='active'",
                (tenant,),
            ).fetchone()[0]
            sources = connection.execute(
                "SELECT COUNT(*) FROM villa_source_records WHERE tenant_id=?",
                (tenant,),
            ).fetchone()[0]
        return {"total": int(total), "active": int(active), "source_records": int(sources)}

    def list_villas(self, *, tenant_id: str, limit: int = 200) -> list[dict[str, Any]]:
        tenant = str(tenant_id or "").strip()
        safe_limit = max(1, min(int(limit), 500))
        with self._conn() as connection:
            rows = connection.execute(
                """
                SELECT * FROM villas
                WHERE tenant_id=?
                ORDER BY property_name COLLATE NOCASE, id
                LIMIT ?
                """,
                (tenant, safe_limit),
            ).fetchall()
        return [dict(row) for row in rows]
