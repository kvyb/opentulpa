"""Interactive user-context source management over the knowledge engine.

User context is the durable, customer-scoped knowledge shelf for interactive chat.
It can manage any uploaded content type that OpenTulpa can preserve in the file
vault: text, documents, spreadsheets, PDFs, images, audio, and video. The service
does not make multimodal files directly queryable; upstream preparation turns
each file into normalized text evidence first. Local parsers handle text-like
documents and structured sheets, while multimodal processors attach derived
summaries/transcripts/visual notes to media records before indexing.

The resulting evidence is indexed under ``user_context:<customer_id>`` and queried
through the same business-knowledge engine used by intake. Intake scopes remain
separate unless selected sources are explicitly promoted into a workflow.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from opentulpa.persistence.sqlite import connect_sqlite

USER_CONTEXT_SCOPE_TYPE = "user_context"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _unique_strings(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


def _tokenize(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\w-]+", str(value or ""), flags=re.UNICODE)
        if len(token.strip()) >= 2
    }


def _compact_source_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "file_id": str(record.get("file_id", "") or "").strip(),
            "filename": str(record.get("filename", "") or "").strip(),
            "mime_type": str(record.get("mime_type", "") or "").strip(),
            "status": str(record.get("status", "") or "").strip(),
            "source_kind": str(record.get("source_kind", "") or "").strip(),
            "section_count": int(record.get("section_count") or 0),
            "char_count": int(record.get("char_count") or 0),
            "warnings": record.get("warnings") or [],
            "archived": bool(record.get("archived", False)),
            "created_at": str(record.get("created_at", "") or "").strip(),
            "updated_at": str(record.get("updated_at", "") or "").strip(),
        }.items()
        if value not in ("", [], None)
    }


def _compact_source_ref(section: Any) -> dict[str, Any]:
    metadata = getattr(section, "metadata", {}) if section is not None else {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        key: value
        for key, value in {
            "file_id": str(metadata.get("file_id", "") or "").strip(),
            "filename": str(metadata.get("filename", "") or "").strip(),
            "source_ref": str(getattr(section, "source_ref", "") or "").strip(),
            "source_kind": str(getattr(section, "source_kind", "") or "").strip(),
            "locator": str(metadata.get("locator", "") or "").strip(),
            "sheet": str(metadata.get("sheet", "") or "").strip(),
            "row_start": int(metadata["row_start"]) if metadata.get("row_start") is not None else None,
            "row_end": int(metadata["row_end"]) if metadata.get("row_end") is not None else None,
            "section_title": str(metadata.get("section_title", "") or "").strip(),
        }.items()
        if value not in ("", [], None)
    }


class UserContextService:
    """Manage durable interactive sources for one customer.

    This layer tracks which uploaded file records belong to the customer's
    reusable chat context, archives or reindexes them, and delegates evidence
    extraction/querying to ``BusinessKnowledgeService``. It intentionally stores
    source membership, not user intent; agents decide from the current
    conversation whether to add, query, archive, or promote files.
    """

    def __init__(self, *, db_path: Any, knowledge_service: Any, file_vault: Any) -> None:
        self.db_path = db_path.resolve()
        self.knowledge_service = knowledge_service
        self.file_vault = file_vault
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS user_context_sources (
                    customer_id TEXT NOT NULL,
                    file_id TEXT NOT NULL,
                    archived INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (customer_id, file_id)
                );
                CREATE INDEX IF NOT EXISTS idx_user_context_sources_customer
                    ON user_context_sources(customer_id, archived, updated_at DESC);
                """
            )

    @staticmethod
    def scope_id(customer_id: str) -> str:
        return str(customer_id or "").strip()

    def _active_file_ids(self, customer_id: str) -> list[str]:
        cid = self.scope_id(customer_id)
        if not cid:
            return []
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT file_id
                FROM user_context_sources
                WHERE customer_id=? AND archived=0
                ORDER BY updated_at DESC
                """,
                (cid,),
            ).fetchall()
        return [str(row["file_id"]) for row in rows]

    def _source_rows(self, customer_id: str, *, include_archived: bool = False) -> list[dict[str, Any]]:
        cid = self.scope_id(customer_id)
        if not cid:
            return []
        with self._conn() as conn:
            meta_rows = conn.execute(
                """
                SELECT file_id, archived, created_at, updated_at
                FROM user_context_sources
                WHERE customer_id=? AND (? OR archived=0)
                ORDER BY updated_at DESC
                """,
                (cid, 1 if include_archived else 0),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for meta in meta_rows:
            file_id = str(meta["file_id"])
            record = self.file_vault.get_file(cid, file_id) or {}
            knowledge = self.knowledge_service._get_source_row(
                customer_id=cid,
                scope_type=USER_CONTEXT_SCOPE_TYPE,
                scope_id=cid,
                file_id=file_id,
            )
            out.append(
                {
                    "file_id": file_id,
                    "filename": str(
                        (record or {}).get("original_filename")
                        or (knowledge["filename"] if knowledge is not None else "")
                        or ""
                    ),
                    "mime_type": str(
                        (record or {}).get("mime_type")
                        or (knowledge["mime_type"] if knowledge is not None else "")
                        or ""
                    ),
                    "status": str(knowledge["status"]) if knowledge is not None else "not_indexed",
                    "source_kind": str(knowledge["source_kind"]) if knowledge is not None else "",
                    "section_count": int(knowledge["section_count"] or 0)
                    if knowledge is not None
                    else 0,
                    "char_count": int(knowledge["char_count"] or 0)
                    if knowledge is not None
                    else 0,
                    "warnings": self.knowledge_service._scope_warnings(
                        customer_id=cid,
                        scope_type=USER_CONTEXT_SCOPE_TYPE,
                        scope_id=cid,
                    ),
                    "summary": str((record or {}).get("summary", "") or "")[:1200],
                    "text_excerpt": str((record or {}).get("text_excerpt", "") or "")[:1200],
                    "archived": bool(int(meta["archived"] or 0)),
                    "created_at": str(meta["created_at"]),
                    "updated_at": str(meta["updated_at"]),
                }
            )
        return out

    def _source_refs(self, customer_id: str, *, file_ids: list[str] | None = None) -> list[dict[str, Any]]:
        cid = self.scope_id(customer_id)
        if not cid:
            return []
        wanted = set(_unique_strings(file_ids or []))
        if not wanted:
            wanted = set(self._active_file_ids(cid))
        if not wanted:
            return []
        sections = self.knowledge_service._load_sections(
            customer_id=cid,
            scope_type=USER_CONTEXT_SCOPE_TYPE,
            scope_id=cid,
        )
        refs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for section in sections:
            metadata = getattr(section, "metadata", {}) if section is not None else {}
            if not isinstance(metadata, dict):
                metadata = {}
            file_id = str(metadata.get("file_id", "") or "").strip()
            if file_id not in wanted:
                continue
            source_ref = str(getattr(section, "source_ref", "") or "").strip()
            key = (file_id, source_ref)
            if key in seen:
                continue
            seen.add(key)
            refs.append(_compact_source_ref(section))
            if len(refs) >= 24:
                break
        return refs

    def add_files(self, *, customer_id: str, file_ids: list[Any]) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        ids = _unique_strings(file_ids)
        if not cid:
            raise ValueError("customer_id is required")
        if not ids:
            raise ValueError("file_ids is required")
        now = _utc_now_iso()
        with self._conn() as conn:
            for file_id in ids:
                conn.execute(
                    """
                    INSERT INTO user_context_sources (customer_id, file_id, archived, created_at, updated_at)
                    VALUES (?, ?, 0, ?, ?)
                    ON CONFLICT(customer_id, file_id) DO UPDATE SET
                        archived=0,
                        updated_at=excluded.updated_at
                    """,
                    (cid, file_id, now, now),
                )
            conn.commit()
        indexed = self.knowledge_service.index_sources(
            customer_id=cid,
            scope_type=USER_CONTEXT_SCOPE_TYPE,
            scope_id=cid,
            file_ids=ids,
        )
        return {
            "ok": True,
            "scope_type": USER_CONTEXT_SCOPE_TYPE,
            "scope_id": cid,
            "indexed": indexed,
            "sources": self.list_sources(customer_id=cid, include_archived=False)["sources"],
            "source_refs": self._source_refs(cid, file_ids=ids),
        }

    def list_sources(self, *, customer_id: str, include_archived: bool = False) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        rows = self._source_rows(cid, include_archived=include_archived)
        return {
            "ok": True,
            "scope_type": USER_CONTEXT_SCOPE_TYPE,
            "scope_id": cid,
            "sources": [_compact_source_record(row) for row in rows],
            "source_refs": self._source_refs(cid),
        }

    def find_sources(self, *, customer_id: str, query: str, limit: int = 10) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        q_tokens = _tokenize(query)
        rows = self._source_rows(cid, include_archived=False)
        scored: list[tuple[int, dict[str, Any]]] = []
        for row in rows:
            haystack = " ".join(
                [
                    str(row.get("filename", "")),
                    str(row.get("mime_type", "")),
                    str(row.get("summary", "")),
                    str(row.get("text_excerpt", "")),
                ]
            )
            score = len(q_tokens & _tokenize(haystack)) if q_tokens else 1
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("updated_at", ""))))
        return {
            "ok": True,
            "query": str(query or "").strip(),
            "sources": [_compact_source_record(row) for _, row in scored[: max(1, min(int(limit), 50))]],
            "source_refs": self._source_refs(
                cid,
                file_ids=[str(row.get("file_id", "") or "") for _, row in scored[: max(1, min(int(limit), 50))]],
            )
            if scored
            else [],
        }

    def reindex(self, *, customer_id: str, file_ids: list[Any] | None = None) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        ids = _unique_strings(file_ids or []) or self._active_file_ids(cid)
        if not ids:
            return {
                "ok": True,
                "scope_type": USER_CONTEXT_SCOPE_TYPE,
                "scope_id": cid,
                "sources": [],
                "source_refs": [],
            }
        indexed = self.knowledge_service.index_sources(
            customer_id=cid,
            scope_type=USER_CONTEXT_SCOPE_TYPE,
            scope_id=cid,
            file_ids=ids,
        )
        now = _utc_now_iso()
        with self._conn() as conn:
            for file_id in ids:
                conn.execute(
                    "UPDATE user_context_sources SET updated_at=? WHERE customer_id=? AND file_id=?",
                    (now, cid, file_id),
                )
            conn.commit()
        return {"ok": True, "indexed": indexed, "source_refs": self._source_refs(cid, file_ids=ids)}

    def archive_sources(self, *, customer_id: str, file_ids: list[Any]) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        ids = _unique_strings(file_ids)
        if not ids:
            raise ValueError("file_ids is required")
        now = _utc_now_iso()
        with self._conn() as conn:
            for file_id in ids:
                conn.execute(
                    """
                    INSERT INTO user_context_sources (customer_id, file_id, archived, created_at, updated_at)
                    VALUES (?, ?, 1, ?, ?)
                    ON CONFLICT(customer_id, file_id) DO UPDATE SET
                        archived=1,
                        updated_at=excluded.updated_at
                    """,
                    (cid, file_id, now, now),
                )
            conn.commit()
        return {"ok": True, "archived_file_ids": ids}

    def query(self, *, customer_id: str, query: str, max_extract_chars: int = 3000) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        safe_query = str(query or "").strip()
        if not safe_query:
            raise ValueError("query is required")
        active_ids = self._active_file_ids(cid)
        if not active_ids:
            return {
                "ok": False,
                "query": safe_query,
                "answer_extract": "NO_SOURCE",
                "sources": [],
                "warnings": ["no active user-context sources"],
            }
        result = self.knowledge_service.query(
            customer_id=cid,
            scope_type=USER_CONTEXT_SCOPE_TYPE,
            scope_id=cid,
            query=safe_query,
            max_extract_chars=max_extract_chars,
            workflow_context={"context_type": USER_CONTEXT_SCOPE_TYPE},
            file_ids=active_ids,
        )
        answer = getattr(result, "answer", None)
        answer_text = str(getattr(answer, "answer_extract", "") or "").strip()
        relevant_sources = self.find_sources(customer_id=cid, query=safe_query, limit=8)
        return {
            "ok": bool(getattr(result, "ok", False)),
            "query": safe_query,
            "answer_extract": answer_text,
            "warnings": list(getattr(result, "warnings", []) or []),
            "source_count": int(getattr(result, "source_count", 0) or 0),
            "section_count": int(getattr(result, "section_count", 0) or 0),
            "sources": relevant_sources["sources"],
            "source_refs": self._source_refs(cid, file_ids=active_ids),
            "diagnostics": getattr(result, "diagnostics", {}) or {},
        }

    def promote_to_intake(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        file_ids: list[Any],
    ) -> dict[str, Any]:
        cid = self.scope_id(customer_id)
        wid = str(workflow_id or "").strip()
        ids = _unique_strings(file_ids)
        if not wid:
            raise ValueError("workflow_id is required")
        if not ids:
            raise ValueError("file_ids is required")
        with suppress(Exception):
            self.knowledge_service.index_sources(
                customer_id=cid,
                scope_type=USER_CONTEXT_SCOPE_TYPE,
                scope_id=cid,
                file_ids=ids,
            )
        indexed = self.knowledge_service.index_sources(
            customer_id=cid,
            scope_type="intake_workflow",
            scope_id=wid,
            file_ids=ids,
        )
        return {
            "ok": True,
            "workflow_id": wid,
            "indexed": indexed,
            "source_refs": self._source_refs(cid, file_ids=ids),
        }
