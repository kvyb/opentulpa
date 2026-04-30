"""Workflow-scoped business knowledge oracle over normalized source packs."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import suppress
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import httpx

from opentulpa.business_knowledge.extraction import (
    content_hash,
    extract_source_sections,
    metadata_json,
)
from opentulpa.business_knowledge.models import (
    KnowledgeIndexedSource,
    KnowledgeQueryAnswer,
    KnowledgeQueryResult,
    KnowledgeSourceSection,
)
from opentulpa.business_knowledge.table_normalizer import (
    select_table_evidence,
    table_evidence_selection_stats,
    table_evidence_to_toon,
    table_facts_from_sections,
    table_overview_to_toon,
)
from opentulpa.context.file_vault import FileVaultService
from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite

_VALID_SCOPE_TYPES = {"workflow_setup", "intake_workflow", "customer_business"}
_DEFAULT_SOURCE_PACK_CHAR_LIMIT = 800_000
_DEFAULT_ORACLE_MODEL = "google/gemini-3.1-flash-lite-preview"
_DEFAULT_ORACLE_MAX_OUTPUT_TOKENS = 1000
_NO_SOURCE_MARKERS = {"no_source", "no source", "not found", "unsupported"}
_DEFAULT_OPENROUTER_APP_REFERER = "https://github.com/kvyb/opentulpa"
_DEFAULT_OPENROUTER_APP_TITLE = "OpenTulpa"
logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _safe_scope_type(value: Any) -> str:
    scope_type = str(value or "").strip().lower()
    if scope_type not in _VALID_SCOPE_TYPES:
        raise ValueError("scope_type must be workflow_setup|intake_workflow|customer_business")
    return scope_type


def _safe_id(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads_dict(value: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value or "{}"))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_int_dict(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw_value in value.items():
        try:
            out[str(key)] = int(raw_value)
        except (TypeError, ValueError):
            continue
    return out


def _safe_intent_diagnostics(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(intent.get("mode", "") or ""),
        "target_term_count": len(_safe_text_list(intent.get("target_terms"))),
        "qualifier_term_count": len(_safe_text_list(intent.get("qualifier_terms"))),
    }


class OpenAICompatibleKnowledgeOracleClient:
    """Small OpenAI-compatible chat client for the business knowledge oracle."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str = _DEFAULT_ORACLE_MODEL,
        timeout_seconds: float = 45.0,
        trace_path: Path | None = None,
    ) -> None:
        self.api_key = str(api_key or "").strip()
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.model = str(model or "").strip() or _DEFAULT_ORACLE_MODEL
        self.timeout_seconds = float(timeout_seconds)
        if not self.api_key:
            raise ValueError("api_key is required")
        if not self.base_url:
            raise ValueError("base_url is required")
        self.trace_path = trace_path.resolve() if isinstance(trace_path, Path) else None
        self._trace_lock = threading.Lock()

    def answer(
        self,
        *,
        source_pack: str,
        query: str,
        workflow_context: dict[str, Any] | None = None,
        max_output_tokens: int = _DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    ) -> str:
        started = time.monotonic()
        request_body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max(1, int(max_output_tokens)),
            "messages": [
                {"role": "system", "content": _oracle_system_prompt()},
                {
                    "role": "user",
                    "content": _oracle_user_prompt(
                        source_pack=source_pack,
                        query=query,
                        workflow_context=workflow_context,
                    ),
                },
            ],
        }
        request_body.update(_oracle_reasoning_control(self.base_url))
        response_payload: dict[str, Any] | None = None
        answer_text = ""
        error_text: str | None = None
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **_openrouter_app_headers(self.base_url),
                },
                json=request_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw_payload = response.json()
            response_payload = raw_payload if isinstance(raw_payload, dict) else {}
            answer_text = _extract_chat_text(response_payload)
            return answer_text
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_oracle_trace(
                source_pack=source_pack,
                query=query,
                workflow_context=workflow_context,
                max_output_tokens=int(request_body["max_tokens"]),
                response_payload=response_payload,
                response_text=answer_text,
                error=error_text,
                elapsed_ms=int((time.monotonic() - started) * 1000),
            )

    def extract_intent(self, *, query: str) -> dict[str, Any]:
        started = time.monotonic()
        request_body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 180,
            "messages": [
                {"role": "system", "content": _oracle_intent_system_prompt()},
                {"role": "user", "content": str(query or "").strip()},
            ],
        }
        request_body.update(_oracle_reasoning_control(self.base_url))
        response_payload: dict[str, Any] | None = None
        response_text = ""
        error_text: str | None = None
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    **_openrouter_app_headers(self.base_url),
                },
                json=request_body,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            raw_payload = response.json()
            response_payload = raw_payload if isinstance(raw_payload, dict) else {}
            response_text = _extract_chat_text(response_payload)
            return _clean_query_intent(_parse_json_object(response_text), str(query or ""))
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._record_oracle_trace(
                source_pack="",
                query=query,
                workflow_context={},
                max_output_tokens=int(request_body["max_tokens"]),
                response_payload=response_payload,
                response_text=response_text,
                error=error_text,
                elapsed_ms=_elapsed_ms(started),
                call_site="knowledge_oracle_intent",
            )

    def _record_oracle_trace(
        self,
        *,
        source_pack: str,
        query: str,
        workflow_context: dict[str, Any] | None,
        max_output_tokens: int,
        response_payload: dict[str, Any] | None,
        response_text: str,
        error: str | None,
        elapsed_ms: int,
        call_site: str = "knowledge_oracle",
    ) -> None:
        path = self.trace_path
        if path is None:
            return
        source_text = str(source_pack or "")
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "model_name": self.model,
            "stable_prefix_count": 0,
            "prompt_messages": [
                {"role": "system", "type": "SystemMessage", "text": _oracle_system_prompt()},
                {
                    "role": "user",
                    "type": "HumanMessage",
                    "text": (
                        f"QUERY:\n{str(query or '').strip()}\n\n"
                        f"WORKFLOW_CONTEXT_JSON:\n{_json_dumps(workflow_context or {})}\n\n"
                        f"SOURCE_PACK_CHARS: {len(source_text)}\n"
                        f"SOURCE_PACK_SHA256: {content_hash(source_text.encode('utf-8', errors='replace'))}"
                    ),
                },
            ],
            "prompt_message_count": 2,
            "response_type": "KnowledgeOracleAnswer",
            "response_message": None,
            "response_text": str(response_text or "").strip(),
            "response_content": str(response_text or "").strip(),
            "response_tool_calls": None,
            "error": str(error or "").strip() or None,
            "call_site": str(call_site or "knowledge_oracle"),
            "source_pack_chars": len(source_text),
            "source_pack_sha256": content_hash(source_text.encode("utf-8", errors="replace")),
            "max_output_tokens": int(max_output_tokens),
            "elapsed_ms": int(elapsed_ms),
            "usage": (response_payload or {}).get("usage") if isinstance(response_payload, dict) else None,
        }
        serialized = json.dumps(payload, ensure_ascii=False, default=str)

        def _commit() -> None:
            existing: list[str] = []
            with suppress(Exception):
                existing = [
                    line.rstrip("\n")
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            kept = existing[-499:]
            kept.append(serialized)
            with path.open("w", encoding="utf-8") as f:
                f.write("\n".join(kept) + "\n")

        with suppress(Exception):
            path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(Exception), self._trace_lock:
            _commit()


class BusinessKnowledgeService:
    """Prepares source packs and answers scoped business questions through an oracle."""

    def __init__(
        self,
        *,
        root_dir: Path,
        db_path: Path,
        file_vault: FileVaultService,
        oracle_client: Any | None = None,
        oracle_model: str = _DEFAULT_ORACLE_MODEL,
        max_source_pack_chars: int = _DEFAULT_SOURCE_PACK_CHAR_LIMIT,
        max_output_tokens: int = _DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.db_path = db_path.resolve()
        self.file_vault = file_vault
        self.oracle_client = oracle_client
        self.oracle_model = str(oracle_model or "").strip() or _DEFAULT_ORACLE_MODEL
        self.max_source_pack_chars = max(1, int(max_source_pack_chars))
        self.max_output_tokens = max(1, int(max_output_tokens))
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(
                """
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
                """
            )

    def index_sources(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        file_ids: list[str],
    ) -> dict[str, Any]:
        started = time.monotonic()
        cid = _safe_id(customer_id, field="customer_id")
        safe_scope_type = _safe_scope_type(scope_type)
        safe_scope_id = _safe_id(scope_id, field="scope_id")
        safe_file_ids = _unique_strings(file_ids)
        if not safe_file_ids:
            raise ValueError("file_ids is required")

        sources = [
            self._index_one_source(
                customer_id=cid,
                scope_type=safe_scope_type,
                scope_id=safe_scope_id,
                file_id=file_id,
            )
            for file_id in safe_file_ids
        ]
        section_count = len(
            self._load_sections(customer_id=cid, scope_type=safe_scope_type, scope_id=safe_scope_id)
        )
        timing_ms = {"total": _elapsed_ms(started)}
        return {
            "ok": True,
            "customer_id": cid,
            "scope_type": safe_scope_type,
            "scope_id": safe_scope_id,
            "sources": [_indexed_source_payload(source) for source in sources],
            "index": {
                "engine": "knowledge_oracle",
                "model": self.oracle_model,
                "source_count": len(sources),
                "section_count": section_count,
            },
            "diagnostics": {"timing_ms": timing_ms},
        }

    def _index_one_source(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        file_id: str,
    ) -> KnowledgeIndexedSource:
        record = self.file_vault.get_file(customer_id, file_id)
        raw_bytes = self.file_vault.read_file_bytes(customer_id, file_id)
        if not record or raw_bytes is None:
            raise ValueError(f"file not found: {file_id}")

        filename = str(record.get("original_filename", "") or "file.bin").strip() or "file.bin"
        mime_type = str(record.get("mime_type", "") or "").strip()
        source_hash = content_hash(
            raw_bytes
            + b"\0"
            + str(record.get("summary", "") or "").encode("utf-8", errors="replace")
            + b"\0"
            + str(record.get("text_excerpt", "") or "").encode("utf-8", errors="replace")
        )
        existing = self._get_source_row(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
            file_id=file_id,
        )
        if existing is not None and str(existing["source_hash"]) == source_hash:
            return KnowledgeIndexedSource(
                file_id=file_id,
                filename=str(existing["filename"]),
                mime_type=str(existing["mime_type"]),
                status=str(existing["status"]),
                source_kind=str(existing["source_kind"]),
                section_count=int(existing["section_count"] or 0),
                char_count=int(existing["char_count"] or 0),
                warnings=_safe_list_json(existing["warnings_json"]),
            )

        sections, warnings, source_kind = extract_source_sections(record=record, raw_bytes=raw_bytes)
        status = "indexed" if sections else "unsupported"
        char_count = sum(len(section.content) for section in sections)
        now = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                DELETE FROM knowledge_sections
                WHERE customer_id=? AND scope_type=? AND scope_id=? AND file_id=?
                """,
                (customer_id, scope_type, scope_id, file_id),
            )
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
                        now,
                    ),
                )
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
                    len(sections),
                    char_count,
                    now,
                ),
            )
            conn.commit()
        return KnowledgeIndexedSource(
            file_id=file_id,
            filename=filename,
            mime_type=mime_type,
            status=status,
            source_kind=source_kind,
            section_count=len(sections),
            char_count=char_count,
            warnings=warnings,
        )

    def query(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        query: str,
        max_extract_chars: int = 3000,
        workflow_context: dict[str, Any] | None = None,
    ) -> KnowledgeQueryResult:
        query_started = time.monotonic()
        timing_ms: dict[str, int] = {}
        cid = _safe_id(customer_id, field="customer_id")
        safe_scope_type = _safe_scope_type(scope_type)
        safe_scope_id = _safe_id(scope_id, field="scope_id")
        safe_query = str(query or "").strip()
        if not safe_query:
            raise ValueError("query is required")
        max_chars = max(200, min(int(max_extract_chars), 5000))
        load_started = time.monotonic()
        sections = self._load_sections(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
        )
        warnings = self._scope_warnings(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
        )
        source_count = self._scope_source_count(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
        )
        timing_ms["load_scope"] = _elapsed_ms(load_started)
        if not sections:
            timing_ms["total"] = _elapsed_ms(query_started)
            return self._query_result(
                ok=False,
                query=safe_query,
                scope_type=safe_scope_type,
                scope_id=safe_scope_id,
                answer=self._no_source_answer(),
                warnings=[*warnings, "no prepared source sections found"],
                source_count=source_count,
                section_count=0,
                diagnostics={"timing_ms": timing_ms},
            )

        source_pack, source_pack_diagnostics = self._source_pack_for_query_with_diagnostics(
            sections=sections,
            query=safe_query,
        )
        timing_ms.update(_safe_int_dict(source_pack_diagnostics.get("timing_ms")))
        if len(source_pack) > self.max_source_pack_chars:
            timing_ms["total"] = _elapsed_ms(query_started)
            return self._query_result(
                ok=False,
                query=safe_query,
                scope_type=safe_scope_type,
                scope_id=safe_scope_id,
                answer=self._no_source_answer(),
                warnings=[
                    *warnings,
                    (
                        "business knowledge source pack exceeds "
                        f"{self.max_source_pack_chars} characters; split or narrow source files"
                    ),
                ],
                source_count=source_count,
                section_count=len(sections),
                diagnostics={
                    "timing_ms": timing_ms,
                    "source_pack": source_pack_diagnostics,
                },
            )
        if self.oracle_client is None:
            timing_ms["total"] = _elapsed_ms(query_started)
            return self._query_result(
                ok=False,
                query=safe_query,
                scope_type=safe_scope_type,
                scope_id=safe_scope_id,
                answer=self._no_source_answer(),
                warnings=[*warnings, "business knowledge oracle client is not configured"],
                source_count=source_count,
                section_count=len(sections),
                diagnostics={
                    "timing_ms": timing_ms,
                    "source_pack": source_pack_diagnostics,
                },
            )

        answer_started = time.monotonic()
        raw_answer = str(
            self.oracle_client.answer(
                source_pack=source_pack,
                query=safe_query,
                workflow_context=workflow_context,
                max_output_tokens=self.max_output_tokens,
            )
            or ""
        )
        timing_ms["oracle_answer"] = _elapsed_ms(answer_started)
        timing_ms["total"] = _elapsed_ms(query_started)
        answer_extract = _clean_oracle_answer(raw_answer)
        result = self._query_result(
            ok=bool(answer_extract),
            query=safe_query,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
            answer=KnowledgeQueryAnswer(
                answer_extract=_trim_text(answer_extract, max_chars=max_chars),
            ),
            warnings=warnings,
            source_count=source_count,
            section_count=len(sections),
            diagnostics={
                "timing_ms": timing_ms,
                "source_pack": source_pack_diagnostics,
            },
        )
        logger.info(
            "business_knowledge.query timing customer_id=%s scope_type=%s scope_id=%s total_ms=%s source_pack_ms=%s oracle_answer_ms=%s section_count=%s source_pack_chars=%s",
            cid,
            safe_scope_type,
            safe_scope_id,
            timing_ms.get("total"),
            timing_ms.get("source_pack_total"),
            timing_ms.get("oracle_answer"),
            len(sections),
            source_pack_diagnostics.get("chars"),
        )
        return result

    def _source_pack_for_query(
        self,
        *,
        sections: list[KnowledgeSourceSection],
        query: str,
    ) -> str:
        source_pack, _ = self._source_pack_for_query_with_diagnostics(
            sections=sections,
            query=query,
        )
        return source_pack

    def _source_pack_for_query_with_diagnostics(
        self,
        *,
        sections: list[KnowledgeSourceSection],
        query: str,
    ) -> tuple[str, dict[str, Any]]:
        started = time.monotonic()
        timing_ms: dict[str, int] = {}
        facts_started = time.monotonic()
        facts = table_facts_from_sections(sections)
        timing_ms["table_facts"] = _elapsed_ms(facts_started)
        if not facts:
            source_pack = _source_pack_for_sections(sections)
            timing_ms["source_pack_total"] = _elapsed_ms(started)
            return source_pack, {
                "mode": "section_pack",
                "chars": len(source_pack),
                "section_count": len(sections),
                "fact_count": 0,
                "timing_ms": timing_ms,
            }
        intent_started = time.monotonic()
        intent = self._query_intent(query)
        timing_ms["intent"] = _elapsed_ms(intent_started)
        if intent["mode"] in {"category_overview", "corpus_overview"}:
            overview_started = time.monotonic()
            source_pack = table_overview_to_toon(
                facts,
                query=query,
                category_terms=[*intent["target_terms"], *intent["qualifier_terms"]],
            )
            timing_ms["table_overview"] = _elapsed_ms(overview_started)
            timing_ms["source_pack_total"] = _elapsed_ms(started)
            return source_pack, {
                "mode": intent["mode"],
                "chars": len(source_pack),
                "section_count": len(sections),
                "fact_count": len(facts),
                "intent": _safe_intent_diagnostics(intent),
                "timing_ms": timing_ms,
            }
        selection_started = time.monotonic()
        rows = select_table_evidence(
            facts,
            query=query,
            target_terms=intent["target_terms"],
            qualifier_terms=intent["qualifier_terms"],
            limit=20,
        )
        timing_ms["table_select"] = _elapsed_ms(selection_started)
        if not rows:
            overview_started = time.monotonic()
            source_pack = table_overview_to_toon(
                facts,
                query=query,
                category_terms=[*intent["target_terms"], *intent["qualifier_terms"]],
            )
            timing_ms["table_overview"] = _elapsed_ms(overview_started)
            timing_ms["source_pack_total"] = _elapsed_ms(started)
            return source_pack, {
                "mode": "fallback_overview",
                "chars": len(source_pack),
                "section_count": len(sections),
                "fact_count": len(facts),
                "selected_row_count": 0,
                "intent": _safe_intent_diagnostics(intent),
                "timing_ms": timing_ms,
            }
        evidence_started = time.monotonic()
        source_pack = table_evidence_to_toon(
            rows,
            query=query,
            target_terms=intent["target_terms"],
            qualifier_terms=intent["qualifier_terms"],
        )
        timing_ms["table_evidence_pack"] = _elapsed_ms(evidence_started)
        timing_ms["source_pack_total"] = _elapsed_ms(started)
        selection_stats = table_evidence_selection_stats(
            facts,
            query=query,
            target_terms=intent["target_terms"],
            qualifier_terms=intent["qualifier_terms"],
            limit=20,
        )
        return source_pack, {
            "mode": intent["mode"],
            "chars": len(source_pack),
            "section_count": len(sections),
            "fact_count": len(facts),
            "selected_row_count": len(rows),
            "selection": selection_stats,
            "intent": _safe_intent_diagnostics(intent),
            "timing_ms": timing_ms,
        }

    def _query_intent(self, query: str) -> dict[str, Any]:
        extractor = getattr(self.oracle_client, "extract_intent", None)
        if callable(extractor):
            with suppress(Exception):
                return _clean_query_intent(extractor(query=query), query)
        return _clean_query_intent({}, query)

    def preflight_scope(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        workflow_goal: str,
    ) -> dict[str, Any]:
        started = time.monotonic()
        goal = str(workflow_goal or "").strip() or "business services pricing policies required fields"
        result = self.query(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
            query=(
                "Can these source files support this intake workflow? "
                "Mention only source-backed useful facts and missing gaps. "
                f"Workflow goal: {goal}"
            ),
        )
        source_rows = self._source_rows(
            customer_id=customer_id,
            scope_type=_safe_scope_type(scope_type),
            scope_id=str(scope_id or "").strip(),
        )
        derived_only = bool(source_rows) and all(
            str(row["source_kind"]) == "derived_from_media" for row in source_rows
        )
        has_answer = bool(result.answer.answer_extract.strip())
        ready = bool(result.section_count) and has_answer and not derived_only and result.ok
        diagnostics = _safe_dict(result.diagnostics)
        timing_ms = dict(_safe_int_dict(diagnostics.get("timing_ms")))
        timing_ms["preflight_total"] = _elapsed_ms(started)
        diagnostics["timing_ms"] = timing_ms
        logger.info(
            "business_knowledge.preflight timing customer_id=%s scope_type=%s scope_id=%s status=%s preflight_total_ms=%s query_total_ms=%s source_pack_ms=%s",
            customer_id,
            scope_type,
            scope_id,
            "ready" if ready else "needs_better_source",
            timing_ms.get("preflight_total"),
            timing_ms.get("total"),
            timing_ms.get("source_pack_total"),
        )
        return {
            "ok": ready,
            "status": "ready" if ready else "needs_better_source",
            "source_count": result.source_count,
            "section_count": result.section_count,
            "answer_extract": result.answer.answer_extract,
            "diagnostics": diagnostics,
            "warnings": [
                *result.warnings,
                *(
                    [
                        "only media-derived evidence was prepared; exact prices, policies, and service menus need a text/table/document source"
                    ]
                    if derived_only
                    else []
                ),
            ],
        }

    def promote_scope(
        self,
        *,
        customer_id: str,
        source_scope_type: str,
        source_scope_id: str,
        target_scope_type: str,
        target_scope_id: str,
    ) -> dict[str, Any]:
        cid = _safe_id(customer_id, field="customer_id")
        source_scope_type = _safe_scope_type(source_scope_type)
        target_scope_type = _safe_scope_type(target_scope_type)
        source_scope_id = _safe_id(source_scope_id, field="source_scope_id")
        target_scope_id = _safe_id(target_scope_id, field="target_scope_id")
        now = _utc_now_iso()
        with self._conn() as conn:
            sources = conn.execute(
                """
                SELECT *
                FROM knowledge_sources
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                """,
                (cid, source_scope_type, source_scope_id),
            ).fetchall()
            sections = conn.execute(
                """
                SELECT *
                FROM knowledge_sections
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                ORDER BY file_id, sort_order, source_ref
                """,
                (cid, source_scope_type, source_scope_id),
            ).fetchall()
            conn.execute(
                "DELETE FROM knowledge_sources WHERE customer_id=? AND scope_type=? AND scope_id=?",
                (cid, target_scope_type, target_scope_id),
            )
            conn.execute(
                "DELETE FROM knowledge_sections WHERE customer_id=? AND scope_type=? AND scope_id=?",
                (cid, target_scope_type, target_scope_id),
            )
            for row in sources:
                conn.execute(
                    """
                    INSERT INTO knowledge_sources (
                        customer_id, scope_type, scope_id, file_id, source_hash, filename,
                        mime_type, status, source_kind, warnings_json, section_count, char_count, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid,
                        target_scope_type,
                        target_scope_id,
                        str(row["file_id"]),
                        str(row["source_hash"]),
                        str(row["filename"]),
                        str(row["mime_type"]),
                        str(row["status"]),
                        str(row["source_kind"]),
                        str(row["warnings_json"]),
                        int(row["section_count"] or 0),
                        int(row["char_count"] or 0),
                        now,
                    ),
                )
            for row in sections:
                conn.execute(
                    """
                    INSERT INTO knowledge_sections (
                        section_id, customer_id, scope_type, scope_id, file_id, filename,
                        source_ref, source_kind, content, metadata_json, sort_order, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        new_short_id("knsec"),
                        cid,
                        target_scope_type,
                        target_scope_id,
                        str(row["file_id"]),
                        str(row["filename"]),
                        str(row["source_ref"]),
                        str(row["source_kind"]),
                        str(row["content"]),
                        str(row["metadata_json"]),
                        int(row["sort_order"] or 0),
                        now,
                    ),
                )
            conn.commit()
        return {
            "ok": True,
            "source_count": len(sources),
            "section_count": len(sections),
            "target_scope_type": target_scope_type,
            "target_scope_id": target_scope_id,
            "index": {
                "engine": "knowledge_oracle",
                "model": self.oracle_model,
                "source_count": len(sources),
                "section_count": len(sections),
            },
        }

    def _query_result(
        self,
        *,
        ok: bool,
        query: str,
        scope_type: str,
        scope_id: str,
        answer: KnowledgeQueryAnswer,
        warnings: list[str],
        source_count: int,
        section_count: int,
        cached: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> KnowledgeQueryResult:
        return KnowledgeQueryResult(
            ok=ok,
            query=query,
            scope_type=scope_type,
            scope_id=scope_id,
            answer=answer,
            warnings=_unique_strings(warnings),
            source_count=source_count,
            section_count=section_count,
            cached=cached,
            diagnostics=_safe_dict(diagnostics),
        )

    def _no_source_answer(self) -> KnowledgeQueryAnswer:
        return KnowledgeQueryAnswer(answer_extract="")

    def _get_source_row(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        file_id: str,
    ) -> sqlite3.Row | None:
        with self._conn() as conn:
            return conn.execute(
                """
                SELECT *
                FROM knowledge_sources
                WHERE customer_id=? AND scope_type=? AND scope_id=? AND file_id=?
                """,
                (customer_id, scope_type, scope_id, file_id),
            ).fetchone()

    def _load_sections(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
    ) -> list[KnowledgeSourceSection]:
        with self._conn() as conn:
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

    def _scope_warnings(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
    ) -> list[str]:
        with self._conn() as conn:
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

    def _scope_source_count(self, *, customer_id: str, scope_type: str, scope_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM knowledge_sources
                WHERE customer_id=? AND scope_type=? AND scope_id=?
                """,
                (customer_id, scope_type, scope_id),
            ).fetchone()
        return int((row or {})["count"] or 0) if row is not None else 0

    def _source_rows(self, *, customer_id: str, scope_type: str, scope_id: str) -> list[sqlite3.Row]:
        if not scope_id:
            return []
        with self._conn() as conn:
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


def _oracle_system_prompt() -> str:
    return (
        "You are a workflow business knowledge oracle. Answer only from the SOURCE PACK. "
        "The SOURCE PACK is normalized evidence from user-uploaded files. It may be compact TOON table evidence. "
        "For TOON evidence, row_label is the original row text and cells contain column/header/value pairs. "
        "When cells include header_group N, the group numbers are distinct column/header groups from left to right; "
        "if the query names class, tier, category, option, or group N, use that matching header_group N. "
        "Duplicate header suffixes like [1] and [2] mean repeated adjacent source columns in spreadsheet order; "
        "when row_label clearly lists ordered variants, map earlier/later variants to earlier/later duplicate columns. "
        "If SOURCE PACK mode is overview, summarize available matching services/categories and ask a concise clarifying "
        "question when an exact price requires a service, variant, class, size, or other qualifier. "
        "For capability questions, answer yes only if a matching source row supports that capability; otherwise say you need to confirm. "
        "If the query or workflow context says the latest customer request is outside the configured workflow scope, "
        "do not answer source facts for that out-of-scope category; return exactly NO_SOURCE. "
        "If the query is only about cancelling, rescheduling, or correcting an existing booking and no new source-backed business fact is needed, return exactly NO_SOURCE. "
        "Do not guess, infer unsupported facts, or use outside knowledge. "
        "If multiple rows support the same requested fact with different values, state the concise distinction instead of choosing silently. "
        "Return a plain string only: concise but informative facts the intake agent needs, with source refs when useful. "
        "For broad pricing questions, include the relevant ranges/options and the missing qualifiers needed for an exact answer. "
        "If the source does not answer the query, return exactly NO_SOURCE. "
        "If the source is ambiguous, return AMBIGUOUS: followed by one concise clarifying question. "
        "Stay under 1000 tokens, and prefer much shorter when the answer is narrow."
    )


def _oracle_intent_system_prompt() -> str:
    return (
        "Extract structured search intent for local matching over arbitrary business files. "
        "Return raw JSON only, no markdown. Keys: mode, target_terms, qualifier_terms, ignore_terms. "
        "mode is one of specific_fact, category_overview, corpus_overview, capability_check. "
        "target_terms are service/product/item/action names likely found in row labels, headings, or column groups. "
        "qualifier_terms are variants/classes/sizes/dimensions/locations likely found in headers, row labels, or nearby context. "
        "Use category_overview or corpus_overview for broad questions asking what exists, what services are offered, "
        "or price lists for a whole category. Use capability_check for yes/no service availability questions. "
        "ignore_terms are generic request words that should not drive retrieval. Preserve user wording. Do not answer."
    )


def _openrouter_app_headers(base_url: str) -> dict[str, str]:
    if "openrouter.ai" not in str(base_url or "").casefold():
        return {}
    title = str(os.environ.get("OPENROUTER_APP_TITLE", "")).strip() or _DEFAULT_OPENROUTER_APP_TITLE
    headers = {"HTTP-Referer": _DEFAULT_OPENROUTER_APP_REFERER}
    if title:
        headers["X-OpenRouter-Title"] = title
    return headers


def _oracle_reasoning_control(base_url: str) -> dict[str, Any]:
    if "openrouter.ai" not in str(base_url or "").casefold():
        return {}
    return {"reasoning": {"effort": "none", "exclude": True}}


def _oracle_user_prompt(
    *,
    source_pack: str,
    query: str,
    workflow_context: dict[str, Any] | None,
) -> str:
    context = _json_dumps(workflow_context or {})
    return (
        f"QUERY:\n{query.strip()}\n\n"
        f"WORKFLOW_CONTEXT_JSON:\n{context}\n\n"
        "SOURCE PACK:\n"
        f"{source_pack}"
    )


def _source_pack_for_sections(sections: list[KnowledgeSourceSection]) -> str:
    parts: list[str] = []
    for index, section in enumerate(sections, start=1):
        metadata = section.metadata if isinstance(section.metadata, dict) else {}
        filename = str(metadata.get("filename", "") or "source").strip()
        parts.append(
            "\n".join(
                [
                    f"## Source {index}",
                    f"source_ref: {section.source_ref}",
                    f"filename: {filename}",
                    f"source_kind: {section.source_kind}",
                    _section_content_for_oracle(section),
                ]
            ).strip()
        )
    return "\n\n".join(parts).strip()


def _section_content_for_oracle(section: KnowledgeSourceSection) -> str:
    content = str(section.content or "").strip()
    if section.source_kind != "structured_table":
        return content
    lines = content.splitlines()
    table_lines: list[str] = []
    for line in lines:
        if line.startswith(("Workbook:", "Sheet:", "Table:", "Rows:")):
            table_lines.append(line)
    cell_rows = _normalized_cell_rows(content)
    if not cell_rows:
        return content
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["source_ref", "row", "column", "value"])
    for row in cell_rows:
        writer.writerow([section.source_ref, *row])
    return "\n".join([*table_lines, "normalized_csv:", out.getvalue().strip()]).strip()


def _normalized_cell_rows(content: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in content.splitlines():
        match = re.match(r"Row\s+(\d+):\s*(.*)$", line.strip())
        if not match:
            continue
        row_number = match.group(1)
        rest = match.group(2)
        for cell in rest.split(" | "):
            if "=" not in cell:
                continue
            column, value = cell.split("=", 1)
            column = column.strip()
            value = value.strip()
            if column and value:
                rows.append((row_number, column, value))
    return rows


def _extract_chat_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            return "\n".join(parts)
    text = first.get("text")
    return str(text or "")


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"\s*```$", "", raw).strip()
    with suppress(Exception):
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        with suppress(Exception):
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
    return {}


def _clean_query_intent(value: Any, query: str) -> dict[str, Any]:
    parsed = value if isinstance(value, dict) else {}
    mode = str(parsed.get("mode", "") or "").strip().lower()
    if mode not in {"specific_fact", "category_overview", "corpus_overview", "capability_check"}:
        mode = "specific_fact"
    intent = {
        "mode": mode,
        "target_terms": _safe_text_list(parsed.get("target_terms")),
        "qualifier_terms": _safe_text_list(parsed.get("qualifier_terms")),
        "ignore_terms": _safe_text_list(parsed.get("ignore_terms")),
    }
    if not intent["target_terms"]:
        intent["target_terms"] = [str(query or "").strip()]
    return intent


def _safe_text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return _unique_strings([str(item).strip() for item in value if str(item).strip()])
    text = str(value or "").strip()
    return [text] if text else []


def _clean_oracle_answer(text: str) -> str:
    answer = re.sub(r"\s+", " ", str(text or "")).strip()
    if not answer:
        return ""
    folded = answer.casefold().strip(" .:")
    if folded in _NO_SOURCE_MARKERS or folded.startswith("no_source"):
        return ""
    return answer


def _trim_text(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _indexed_source_payload(source: KnowledgeIndexedSource) -> dict[str, Any]:
    return {
        "file_id": source.file_id,
        "filename": source.filename,
        "mime_type": source.mime_type,
        "status": source.status,
        "source_kind": source.source_kind,
        "section_count": source.section_count,
        "char_count": source.char_count,
        "warnings": source.warnings,
    }


def query_result_payload(result: KnowledgeQueryResult) -> dict[str, Any]:
    answer = result.answer
    return {
        "ok": bool(result.ok),
        "query": result.query,
        "scope_type": result.scope_type,
        "scope_id": result.scope_id,
        "answer_extract": answer.answer_extract,
        "warnings": result.warnings,
        "source_count": result.source_count,
        "section_count": result.section_count,
        "cached": result.cached,
        "diagnostics": result.diagnostics,
    }


def _safe_list_json(value: Any) -> list[str]:
    with suppress(Exception):
        parsed = json.loads(str(value or "[]"))
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _unique_strings(values: Any) -> list[str]:
    if isinstance(values, str):
        values = re.split(r"[\n,;]+", values)
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        folded = text.casefold()
        if not text or folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out
