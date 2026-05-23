"""SQLite repository for business knowledge sources and sections."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

from opentulpa.business_knowledge.extraction import metadata_json
from opentulpa.business_knowledge.models import KnowledgeSourceSection
from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite

KNOWLEDGE_PREFLIGHT_CACHE_VERSION = 1
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    customer_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    status TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    warnings_json TEXT NOT NULL,
    section_count INTEGER NOT NULL,
    char_count INTEGER NOT NULL,
    indexed_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, scope_type, scope_id, file_id)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_sources_scope
    ON knowledge_sources(customer_id, scope_type, scope_id);

CREATE TABLE IF NOT EXISTS knowledge_sections (
    section_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    file_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_sections_scope
    ON knowledge_sections(customer_id, scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_sections_file
    ON knowledge_sections(customer_id, scope_type, scope_id, file_id);

CREATE TABLE IF NOT EXISTS knowledge_preflight_cache (
    customer_id TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    cache_version INTEGER NOT NULL,
    source_signature TEXT NOT NULL,
    workflow_goal_hash TEXT NOT NULL,
    oracle_model TEXT NOT NULL,
    file_ids_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (customer_id, cache_key)
);
CREATE INDEX IF NOT EXISTS idx_knowledge_preflight_cache_lookup
    ON knowledge_preflight_cache(
        customer_id, source_signature, workflow_goal_hash, oracle_model
    );
"""


class BusinessKnowledgeRepository:
    def __init__(self, *, root_dir: Path, db_path: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.db_path = db_path.resolve()
        self.init_db()

    def conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def init_db(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_SCHEMA_SQL)

    def get_source_row(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        file_id: str,
    ) -> sqlite3.Row | None:
        with self.conn() as conn:
            return cast(
                "sqlite3.Row | None",
                conn.execute(
                    """
                    SELECT *
                    FROM knowledge_sources
                    WHERE customer_id=? AND scope_type=? AND scope_id=? AND file_id=?
                    """,
                    (customer_id, scope_type, scope_id, file_id),
                ).fetchone(),
            )

    def replace_source(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        file_id: str,
        source_hash: str,
        filename: str,
        mime_type: str,
        status: str,
        source_kind: str,
        warnings: list[str],
        sections: list[KnowledgeSourceSection],
        indexed_at: str,
    ) -> None:
        char_count = sum(len(section.content) for section in sections)
        with self.conn() as conn:
            _delete_sections(
                conn,
                customer_id=customer_id,
                scope_type=scope_type,
                scope_id=scope_id,
                file_id=file_id,
            )
            _insert_sections(
                conn,
                customer_id=customer_id,
                scope_type=scope_type,
                scope_id=scope_id,
                file_id=file_id,
                filename=filename,
                sections=sections,
                created_at=indexed_at,
            )
            _upsert_source(
                conn,
                customer_id=customer_id,
                scope_type=scope_type,
                scope_id=scope_id,
                file_id=file_id,
                source_hash=source_hash,
                filename=filename,
                mime_type=mime_type,
                status=status,
                source_kind=source_kind,
                warnings=warnings,
                section_count=len(sections),
                char_count=char_count,
                indexed_at=indexed_at,
            )
            conn.commit()

    def load_sections(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
    ) -> list[KnowledgeSourceSection]:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM knowledge_sections
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                ORDER BY file_id, sort_order, source_ref, section_id
                """,
                (customer_id, scope_type, scope_id),
            ).fetchall()
        return [
            KnowledgeSourceSection(
                content=str(row["content"]),
                source_ref=str(row["source_ref"]),
                source_kind=str(row["source_kind"]),
                metadata={
                    "file_id": str(row["file_id"]),
                    "filename": str(row["filename"]),
                    **_json_loads_dict(row["metadata_json"]),
                },
                sort_order=int(row["sort_order"] or 0),
            )
            for row in rows
        ]

    def scope_warnings(self, *, customer_id: str, scope_type: str, scope_id: str) -> list[str]:
        with self.conn() as conn:
            rows = conn.execute(
                """
                SELECT warnings_json, status, filename
                FROM knowledge_sources
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                ORDER BY filename
                """,
                (customer_id, scope_type, scope_id),
            ).fetchall()
        warnings: list[str] = []
        for row in rows:
            warnings.extend(_safe_list_json(row["warnings_json"]))
            if str(row["status"]) != "indexed":
                warnings.append(f"{row['filename']}: {row['status']}")
        return _unique_strings(warnings)

    def scope_source_count(self, *, customer_id: str, scope_type: str, scope_id: str) -> int:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM knowledge_sources
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                """,
                (customer_id, scope_type, scope_id),
            ).fetchone()
        return int((row or {})["count"] or 0) if row is not None else 0

    def source_rows(self, *, customer_id: str, scope_type: str, scope_id: str) -> list[sqlite3.Row]:
        if not scope_id:
            return []
        with self.conn() as conn:
            return list(
                conn.execute(
                    """
                    SELECT *
                    FROM knowledge_sources
                    WHERE customer_id=? AND scope_type=? AND scope_id=?
                    ORDER BY filename
                    """,
                    (customer_id, scope_type, scope_id),
                ).fetchall()
            )

    def get_preflight_cache(self, *, customer_id: str, cache_key: str) -> dict[str, Any]:
        with self.conn() as conn:
            row = conn.execute(
                """
                SELECT result_json
                FROM knowledge_preflight_cache
                WHERE customer_id=? AND cache_key=? AND cache_version=?
                """,
                (customer_id, cache_key, KNOWLEDGE_PREFLIGHT_CACHE_VERSION),
            ).fetchone()
        if row is None:
            return {}
        return _json_loads_dict(row["result_json"])

    def store_preflight_cache(
        self,
        *,
        customer_id: str,
        cache_meta: dict[str, Any],
        result: dict[str, Any],
        now: str,
    ) -> None:
        with self.conn() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_preflight_cache (
                    customer_id, cache_key, cache_version, source_signature,
                    workflow_goal_hash, oracle_model, file_ids_json, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(customer_id, cache_key) DO UPDATE SET
                    source_signature=excluded.source_signature,
                    workflow_goal_hash=excluded.workflow_goal_hash,
                    oracle_model=excluded.oracle_model,
                    file_ids_json=excluded.file_ids_json,
                    result_json=excluded.result_json,
                    updated_at=excluded.updated_at
                """,
                (
                    customer_id,
                    str(cache_meta.get("cache_key", "") or ""),
                    int(cache_meta.get("cache_version") or KNOWLEDGE_PREFLIGHT_CACHE_VERSION),
                    str(cache_meta.get("source_signature", "") or ""),
                    str(cache_meta.get("workflow_goal_hash", "") or ""),
                    str(cache_meta.get("oracle_model", "") or ""),
                    _json_dumps(_safe_text_list(cache_meta.get("file_ids"))),
                    _json_dumps(result),
                    now,
                    now,
                ),
            )
            conn.commit()

    def promote_scope(
        self,
        *,
        customer_id: str,
        source_scope_type: str,
        source_scope_id: str,
        target_scope_type: str,
        target_scope_id: str,
        indexed_at: str,
    ) -> tuple[int, int]:
        with self.conn() as conn:
            sources = conn.execute(
                """
                SELECT *
                FROM knowledge_sources
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                """,
                (customer_id, source_scope_type, source_scope_id),
            ).fetchall()
            sections = conn.execute(
                """
                SELECT *
                FROM knowledge_sections
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                ORDER BY file_id, sort_order, source_ref
                """,
                (customer_id, source_scope_type, source_scope_id),
            ).fetchall()
            _delete_scope(conn, customer_id=customer_id, scope_type=target_scope_type, scope_id=target_scope_id)
            _copy_source_rows(
                conn,
                customer_id=customer_id,
                scope_type=target_scope_type,
                scope_id=target_scope_id,
                rows=sources,
                indexed_at=indexed_at,
            )
            _copy_section_rows(
                conn,
                customer_id=customer_id,
                scope_type=target_scope_type,
                scope_id=target_scope_id,
                rows=sections,
                created_at=indexed_at,
            )
            conn.commit()
        return len(sources), len(sections)


def _delete_sections(
    conn: sqlite3.Connection,
    *,
    customer_id: str,
    scope_type: str,
    scope_id: str,
    file_id: str,
) -> None:
    conn.execute(
        """
        DELETE FROM knowledge_sections
        WHERE customer_id=? AND scope_type=? AND scope_id=? AND file_id=?
        """,
        (customer_id, scope_type, scope_id, file_id),
    )


def _insert_sections(
    conn: sqlite3.Connection,
    *,
    customer_id: str,
    scope_type: str,
    scope_id: str,
    file_id: str,
    filename: str,
    sections: list[KnowledgeSourceSection],
    created_at: str,
) -> None:
    for section in sections:
        conn.execute(
            """
            INSERT INTO knowledge_sections (
                section_id, customer_id, scope_type, scope_id, file_id, filename,
                source_ref, source_kind, content, metadata_json, sort_order, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_short_id("knsec"),
                customer_id,
                scope_type,
                scope_id,
                file_id,
                filename,
                section.source_ref,
                section.source_kind,
                section.content,
                metadata_json(section.metadata),
                int(section.sort_order),
                created_at,
            ),
        )


def _upsert_source(
    conn: sqlite3.Connection,
    *,
    customer_id: str,
    scope_type: str,
    scope_id: str,
    file_id: str,
    source_hash: str,
    filename: str,
    mime_type: str,
    status: str,
    source_kind: str,
    warnings: list[str],
    section_count: int,
    char_count: int,
    indexed_at: str,
) -> None:
    conn.execute(
        """
        INSERT INTO knowledge_sources (
            customer_id, scope_type, scope_id, file_id, source_hash, filename,
            mime_type, status, source_kind, warnings_json, section_count, char_count, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(customer_id, scope_type, scope_id, file_id) DO UPDATE SET
            source_hash=excluded.source_hash,
            filename=excluded.filename,
            mime_type=excluded.mime_type,
            status=excluded.status,
            source_kind=excluded.source_kind,
            warnings_json=excluded.warnings_json,
            section_count=excluded.section_count,
            char_count=excluded.char_count,
            indexed_at=excluded.indexed_at
        """,
        (
            customer_id,
            scope_type,
            scope_id,
            file_id,
            source_hash,
            filename,
            mime_type,
            status,
            source_kind,
            _json_dumps(warnings),
            section_count,
            char_count,
            indexed_at,
        ),
    )


def _delete_scope(
    conn: sqlite3.Connection,
    *,
    customer_id: str,
    scope_type: str,
    scope_id: str,
) -> None:
    conn.execute(
        "DELETE FROM knowledge_sources WHERE customer_id=? AND scope_type=? AND scope_id=?",
        (customer_id, scope_type, scope_id),
    )
    conn.execute(
        "DELETE FROM knowledge_sections WHERE customer_id=? AND scope_type=? AND scope_id=?",
        (customer_id, scope_type, scope_id),
    )


def _copy_source_rows(
    conn: sqlite3.Connection,
    *,
    customer_id: str,
    scope_type: str,
    scope_id: str,
    rows: list[sqlite3.Row],
    indexed_at: str,
) -> None:
    for row in rows:
        conn.execute(
            """
            INSERT INTO knowledge_sources (
                customer_id, scope_type, scope_id, file_id, source_hash, filename,
                mime_type, status, source_kind, warnings_json, section_count, char_count, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                customer_id,
                scope_type,
                scope_id,
                str(row["file_id"]),
                str(row["source_hash"]),
                str(row["filename"]),
                str(row["mime_type"]),
                str(row["status"]),
                str(row["source_kind"]),
                str(row["warnings_json"]),
                int(row["section_count"] or 0),
                int(row["char_count"] or 0),
                indexed_at,
            ),
        )


def _copy_section_rows(
    conn: sqlite3.Connection,
    *,
    customer_id: str,
    scope_type: str,
    scope_id: str,
    rows: list[sqlite3.Row],
    created_at: str,
) -> None:
    for row in rows:
        conn.execute(
            """
            INSERT INTO knowledge_sections (
                section_id, customer_id, scope_type, scope_id, file_id, filename,
                source_ref, source_kind, content, metadata_json, sort_order, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_short_id("knsec"),
                customer_id,
                scope_type,
                scope_id,
                str(row["file_id"]),
                str(row["filename"]),
                str(row["source_ref"]),
                str(row["source_kind"]),
                str(row["content"]),
                str(row["metadata_json"]),
                int(row["sort_order"] or 0),
                created_at,
            ),
        )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_list_json(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _safe_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_strings([str(item).strip() for item in value if str(item).strip()])
    text = str(value or "").strip()
    return [text] if text else []


def _unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out
