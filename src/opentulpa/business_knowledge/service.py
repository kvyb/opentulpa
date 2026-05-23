"""Workflow-scoped business knowledge oracle over normalized source packs."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.business_knowledge.extraction import (
    content_hash,
)
from opentulpa.business_knowledge.indexer import BusinessKnowledgeIndexer
from opentulpa.business_knowledge.models import (
    KnowledgeIndexedSource,
    KnowledgeQueryAnswer,
    KnowledgeQueryResult,
    KnowledgeSourceSection,
)
from opentulpa.business_knowledge.oracle_client import (
    DEFAULT_ORACLE_MAX_OUTPUT_TOKENS as _DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
)
from opentulpa.business_knowledge.oracle_client import (
    DEFAULT_ORACLE_MODEL as _DEFAULT_ORACLE_MODEL,
)
from opentulpa.business_knowledge.oracle_client import (
    OpenAICompatibleKnowledgeOracleClient as OpenAICompatibleKnowledgeOracleClient,
)
from opentulpa.business_knowledge.oracle_client import clean_query_intent
from opentulpa.business_knowledge.repository import (
    KNOWLEDGE_PREFLIGHT_CACHE_VERSION,
    BusinessKnowledgeRepository,
)
from opentulpa.business_knowledge.source_pack_planner import BusinessKnowledgeSourcePackPlanner
from opentulpa.context.file_vault import FileVaultService

_VALID_SCOPE_TYPES = {"workflow_setup", "intake_workflow", "customer_business", "user_context"}
_DEFAULT_SOURCE_PACK_CHAR_LIMIT = 800_000
_NO_SOURCE_MARKERS = {"no_source", "no source", "not found", "unsupported"}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedQueryScope:
    customer_id: str
    scope_type: str
    scope_id: str
    query: str
    max_chars: int
    sections: list[KnowledgeSourceSection]
    warnings: list[str]
    source_count: int
    timing_ms: dict[str, int]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _safe_scope_type(value: Any) -> str:
    scope_type = str(value or "").strip().lower()
    if scope_type not in _VALID_SCOPE_TYPES:
        raise ValueError(
            "scope_type must be workflow_setup|intake_workflow|customer_business|user_context"
        )
    return scope_type


def _safe_id(value: Any, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} is required")
    return text


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _hash_json(value: Any) -> str:
    return content_hash(_json_dumps(value).encode("utf-8", errors="replace"))


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


class BusinessKnowledgeService:
    """Prepares source packs and answers scoped business questions through an oracle."""

    def __init__(
        self,
        *,
        root_dir: Path,
        db_path: Path,
        file_vault: FileVaultService,
        oracle_client: Any | None = None,
        langfuse_tracer: Any | None = None,
        oracle_model: str = _DEFAULT_ORACLE_MODEL,
        max_source_pack_chars: int = _DEFAULT_SOURCE_PACK_CHAR_LIMIT,
        max_output_tokens: int = _DEFAULT_ORACLE_MAX_OUTPUT_TOKENS,
    ) -> None:
        self.root_dir = root_dir.resolve()
        self.db_path = db_path.resolve()
        self.repository = BusinessKnowledgeRepository(root_dir=self.root_dir, db_path=self.db_path)
        self.indexer = BusinessKnowledgeIndexer(
            file_vault=file_vault,
            repository=self.repository,
            now_iso=_utc_now_iso,
        )
        self.oracle_client = oracle_client
        self.langfuse_tracer = langfuse_tracer
        self.oracle_model = str(oracle_model or "").strip() or _DEFAULT_ORACLE_MODEL
        self.max_source_pack_chars = max(1, int(max_source_pack_chars))
        self.max_output_tokens = max(1, int(max_output_tokens))
        self._source_pack_planner = BusinessKnowledgeSourcePackPlanner(
            extract_intent=self._query_intent,
            record_span=self._record_observability_span,
        )

    def _record_observability_span(
        self,
        *,
        name: str,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
        status: str = "ok",
    ) -> None:
        tracer = getattr(self, "langfuse_tracer", None)
        record_span = getattr(tracer, "record_span", None)
        if callable(record_span):
            with suppress(Exception):
                record_span(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    trace_id=trace_id,
                    status=status,
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
            self.indexer.index_one_source(
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
        result = {
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
        self._record_observability_span(
            name="knowledge.index_sources",
            input={"customer_id": cid, "scope_type": safe_scope_type, "scope_id": safe_scope_id},
            output={"source_count": len(sources), "section_count": section_count},
            metadata={"timing_ms": timing_ms, "file_ids": safe_file_ids},
        )
        return result

    def query(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        query: str,
        max_extract_chars: int = 3000,
        workflow_context: dict[str, Any] | None = None,
        file_ids: list[str] | None = None,
    ) -> KnowledgeQueryResult:
        query_started = time.monotonic()
        prepared = self._prepare_query_scope(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
            query=query,
            max_extract_chars=max_extract_chars,
            file_ids=file_ids,
        )
        timing_ms = prepared.timing_ms
        if not prepared.sections:
            return self._empty_source_query_result(prepared=prepared, started=query_started)

        source_pack, source_pack_diagnostics = self._source_pack_planner.plan(
            sections=prepared.sections,
            query=prepared.query,
        )
        timing_ms.update(_safe_int_dict(source_pack_diagnostics.get("timing_ms")))
        if len(source_pack) > self.max_source_pack_chars:
            return self._oversized_source_pack_result(
                prepared=prepared,
                source_pack_diagnostics=source_pack_diagnostics,
                started=query_started,
            )
        if self.oracle_client is None:
            return self._missing_oracle_query_result(
                prepared=prepared,
                source_pack_diagnostics=source_pack_diagnostics,
                started=query_started,
            )

        raw_answer = self._answer_with_oracle(
            prepared=prepared,
            source_pack=source_pack,
            workflow_context=workflow_context,
        )
        timing_ms["total"] = _elapsed_ms(query_started)
        self._log_query_timing(prepared=prepared, source_pack_diagnostics=source_pack_diagnostics)
        return self._successful_query_result(
            prepared=prepared,
            raw_answer=raw_answer,
            source_pack_diagnostics=source_pack_diagnostics,
        )

    def _prepare_query_scope(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        query: str,
        max_extract_chars: int,
        file_ids: list[str] | None,
    ) -> _PreparedQueryScope:
        timing_ms: dict[str, int] = {}
        cid = _safe_id(customer_id, field="customer_id")
        safe_scope_type = _safe_scope_type(scope_type)
        safe_scope_id = _safe_id(scope_id, field="scope_id")
        safe_query = str(query or "").strip()
        if not safe_query:
            raise ValueError("query is required")

        load_started = time.monotonic()
        sections = self._load_sections(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
        )
        safe_file_ids = set(_unique_strings(file_ids or []))
        if safe_file_ids:
            sections = [section for section in sections if str(section.metadata.get("file_id", "")) in safe_file_ids]
        source_count = self._scope_source_count(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
        )
        timing_ms["load_scope"] = _elapsed_ms(load_started)
        return _PreparedQueryScope(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
            query=safe_query,
            max_chars=max(200, min(int(max_extract_chars), 5000)),
            sections=sections,
            warnings=self._scope_warnings(
                customer_id=cid,
                scope_type=safe_scope_type,
                scope_id=safe_scope_id,
            ),
            source_count=len(safe_file_ids) if safe_file_ids else source_count,
            timing_ms=timing_ms,
        )

    def _empty_source_query_result(
        self,
        *,
        prepared: _PreparedQueryScope,
        started: float,
    ) -> KnowledgeQueryResult:
        prepared.timing_ms["total"] = _elapsed_ms(started)
        return self._query_result(
            ok=False,
            query=prepared.query,
            scope_type=prepared.scope_type,
            scope_id=prepared.scope_id,
            answer=self._no_source_answer(),
            warnings=[*prepared.warnings, "no prepared source sections found"],
            source_count=prepared.source_count,
            section_count=0,
            diagnostics={"timing_ms": prepared.timing_ms},
        )

    def _oversized_source_pack_result(
        self,
        *,
        prepared: _PreparedQueryScope,
        source_pack_diagnostics: dict[str, Any],
        started: float,
    ) -> KnowledgeQueryResult:
        prepared.timing_ms["total"] = _elapsed_ms(started)
        warning = (
            "business knowledge source pack exceeds "
            f"{self.max_source_pack_chars} characters; split or narrow source files"
        )
        return self._query_result(
            ok=False,
            query=prepared.query,
            scope_type=prepared.scope_type,
            scope_id=prepared.scope_id,
            answer=self._no_source_answer(),
            warnings=[*prepared.warnings, warning],
            source_count=prepared.source_count,
            section_count=len(prepared.sections),
            diagnostics={"timing_ms": prepared.timing_ms, "source_pack": source_pack_diagnostics},
        )

    def _missing_oracle_query_result(
        self,
        *,
        prepared: _PreparedQueryScope,
        source_pack_diagnostics: dict[str, Any],
        started: float,
    ) -> KnowledgeQueryResult:
        prepared.timing_ms["total"] = _elapsed_ms(started)
        return self._query_result(
            ok=False,
            query=prepared.query,
            scope_type=prepared.scope_type,
            scope_id=prepared.scope_id,
            answer=self._no_source_answer(),
            warnings=[*prepared.warnings, "business knowledge oracle client is not configured"],
            source_count=prepared.source_count,
            section_count=len(prepared.sections),
            diagnostics={"timing_ms": prepared.timing_ms, "source_pack": source_pack_diagnostics},
        )

    def _answer_with_oracle(
        self,
        *,
        prepared: _PreparedQueryScope,
        source_pack: str,
        workflow_context: dict[str, Any] | None,
    ) -> str:
        assert self.oracle_client is not None
        answer_started = time.monotonic()
        raw_answer = str(
            self.oracle_client.answer(
                source_pack=source_pack,
                query=prepared.query,
                workflow_context=workflow_context,
                max_output_tokens=self.max_output_tokens,
            )
            or ""
        )
        prepared.timing_ms["oracle_answer"] = _elapsed_ms(answer_started)
        self._record_observability_span(
            name="knowledge.oracle_answer",
            input={
                "query": prepared.query,
                "scope_type": prepared.scope_type,
                "scope_id": prepared.scope_id,
            },
            output={"answer_chars": len(raw_answer)},
            metadata={"timing_ms": {"oracle_answer": prepared.timing_ms["oracle_answer"]}},
        )
        return raw_answer

    def _successful_query_result(
        self,
        *,
        prepared: _PreparedQueryScope,
        raw_answer: str,
        source_pack_diagnostics: dict[str, Any],
    ) -> KnowledgeQueryResult:
        answer_extract = _clean_oracle_answer(raw_answer)
        return self._query_result(
            ok=bool(answer_extract),
            query=prepared.query,
            scope_type=prepared.scope_type,
            scope_id=prepared.scope_id,
            answer=KnowledgeQueryAnswer(
                answer_extract=_trim_text(answer_extract, max_chars=prepared.max_chars),
            ),
            warnings=prepared.warnings,
            source_count=prepared.source_count,
            section_count=len(prepared.sections),
            diagnostics={
                "timing_ms": prepared.timing_ms,
                "source_pack": source_pack_diagnostics,
            },
        )

    def _log_query_timing(
        self,
        *,
        prepared: _PreparedQueryScope,
        source_pack_diagnostics: dict[str, Any],
    ) -> None:
        logger.info(
            "business_knowledge.query timing customer_id=%s scope_type=%s scope_id=%s total_ms=%s source_pack_ms=%s oracle_answer_ms=%s section_count=%s source_pack_chars=%s",
            prepared.customer_id,
            prepared.scope_type,
            prepared.scope_id,
            prepared.timing_ms.get("total"),
            prepared.timing_ms.get("source_pack_total"),
            prepared.timing_ms.get("oracle_answer"),
            len(prepared.sections),
            source_pack_diagnostics.get("chars"),
        )

    def _query_intent(self, query: str) -> dict[str, Any]:
        extractor = getattr(self.oracle_client, "extract_intent", None)
        if callable(extractor):
            with suppress(Exception):
                return clean_query_intent(extractor(query=query), query)
        return clean_query_intent({}, query)

    def preflight_scope(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        workflow_goal: str,
    ) -> dict[str, Any]:
        started = time.monotonic()
        cid = _safe_id(customer_id, field="customer_id")
        safe_scope_type = _safe_scope_type(scope_type)
        safe_scope_id = _safe_id(scope_id, field="scope_id")
        goal = str(workflow_goal or "").strip() or "business services pricing policies required fields"
        source_rows = self._source_rows(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
        )
        cache_meta = self._preflight_cache_meta(
            customer_id=cid,
            source_rows=source_rows,
            workflow_goal=goal,
        )
        cached = self._cached_preflight_result(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
            cache_meta=cache_meta,
            started=started,
        )
        if cached:
            return cached

        return self._uncached_preflight_result(
            customer_id=cid,
            scope_type=safe_scope_type,
            scope_id=safe_scope_id,
            workflow_goal=goal,
            source_rows=source_rows,
            cache_meta=cache_meta,
            started=started,
        )

    def _uncached_preflight_result(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        workflow_goal: str,
        source_rows: list[sqlite3.Row],
        cache_meta: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        result = self._query_preflight_scope(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
            workflow_goal=workflow_goal,
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
        diagnostics["cache"] = {**cache_meta, "hit": False}
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
        payload = self._preflight_payload(
            result=result,
            ready=ready,
            derived_only=derived_only,
            diagnostics=diagnostics,
        )
        self._store_preflight_cache(
            customer_id=customer_id,
            cache_meta=cache_meta,
            result=payload,
        )
        return payload

    def _cached_preflight_result(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        cache_meta: dict[str, Any],
        started: float,
    ) -> dict[str, Any]:
        cached = self._get_preflight_cache(customer_id=customer_id, cache_key=cache_meta["cache_key"])
        if not cached:
            return {}

        diagnostics = _safe_dict(cached.get("diagnostics"))
        timing_ms = dict(_safe_int_dict(diagnostics.get("timing_ms")))
        timing_ms["preflight_total"] = _elapsed_ms(started)
        diagnostics["timing_ms"] = timing_ms
        diagnostics["cache"] = {**cache_meta, "hit": True}
        cached["diagnostics"] = diagnostics
        cached["cache_hit"] = True
        logger.info(
            "business_knowledge.preflight cache_hit customer_id=%s scope_type=%s scope_id=%s cache_key=%s preflight_total_ms=%s",
            customer_id,
            scope_type,
            scope_id,
            cache_meta["cache_key"],
            timing_ms.get("preflight_total"),
        )
        return cached

    def _query_preflight_scope(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        workflow_goal: str,
    ) -> KnowledgeQueryResult:
        return self.query(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
            query=(
                "Can these source files support this intake workflow? "
                "Mention only source-backed useful facts and missing gaps. "
                f"Workflow goal: {workflow_goal}"
            ),
        )

    def _preflight_payload(
        self,
        *,
        result: KnowledgeQueryResult,
        ready: bool,
        derived_only: bool,
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        warnings = list(result.warnings)
        if derived_only:
            warnings.append(
                "only media-derived evidence was prepared; exact prices, policies, and service menus need a text/table/document source"
            )
        return {
            "ok": ready,
            "status": "ready" if ready else "needs_better_source",
            "source_count": result.source_count,
            "section_count": result.section_count,
            "answer_extract": result.answer.answer_extract,
            "diagnostics": diagnostics,
            "cache_hit": False,
            "warnings": warnings,
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
        source_count, section_count = self.repository.promote_scope(
            customer_id=cid,
            source_scope_type=source_scope_type,
            source_scope_id=source_scope_id,
            target_scope_type=target_scope_type,
            target_scope_id=target_scope_id,
            indexed_at=now,
        )
        return {
            "ok": True,
            "source_count": source_count,
            "section_count": section_count,
            "target_scope_type": target_scope_type,
            "target_scope_id": target_scope_id,
            "index": {
                "engine": "knowledge_oracle",
                "model": self.oracle_model,
                "source_count": source_count,
                "section_count": section_count,
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

    def _load_sections(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
    ) -> list[KnowledgeSourceSection]:
        return self.repository.load_sections(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    def _scope_warnings(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
    ) -> list[str]:
        return self.repository.scope_warnings(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    def _scope_source_count(self, *, customer_id: str, scope_type: str, scope_id: str) -> int:
        return self.repository.scope_source_count(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    def _preflight_cache_meta(
        self,
        *,
        customer_id: str,
        source_rows: list[sqlite3.Row],
        workflow_goal: str,
    ) -> dict[str, Any]:
        source_versions = [
            {
                "file_id": str(row["file_id"]),
                "source_hash": str(row["source_hash"]),
                "status": str(row["status"]),
                "source_kind": str(row["source_kind"]),
                "section_count": int(row["section_count"] or 0),
                "char_count": int(row["char_count"] or 0),
                "indexed_at": str(row["indexed_at"]),
            }
            for row in sorted(source_rows, key=lambda item: str(item["file_id"]))
        ]
        source_signature = _hash_json(source_versions)
        workflow_goal_hash = _hash_json(str(workflow_goal or "").strip())
        cache_key = _hash_json(
            {
                "cache_version": KNOWLEDGE_PREFLIGHT_CACHE_VERSION,
                "customer_id": customer_id,
                "source_signature": source_signature,
                "workflow_goal_hash": workflow_goal_hash,
                "oracle_model": self.oracle_model,
                "max_output_tokens": self.max_output_tokens,
            }
        )
        return {
            "cache_key": cache_key,
            "cache_version": KNOWLEDGE_PREFLIGHT_CACHE_VERSION,
            "source_signature": source_signature,
            "workflow_goal_hash": workflow_goal_hash,
            "oracle_model": self.oracle_model,
            "file_count": len(source_versions),
            "file_ids": [item["file_id"] for item in source_versions],
        }

    def _get_preflight_cache(self, *, customer_id: str, cache_key: str) -> dict[str, Any]:
        return self.repository.get_preflight_cache(customer_id=customer_id, cache_key=cache_key)

    def _store_preflight_cache(
        self,
        *,
        customer_id: str,
        cache_meta: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        self.repository.store_preflight_cache(
            customer_id=customer_id,
            cache_meta=cache_meta,
            result=result,
            now=_utc_now_iso(),
        )

    def _source_rows(self, *, customer_id: str, scope_type: str, scope_id: str) -> list[sqlite3.Row]:
        return self.repository.source_rows(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
        )

    def _get_source_row(
        self,
        *,
        customer_id: str,
        scope_type: str,
        scope_id: str,
        file_id: str,
    ) -> sqlite3.Row | None:
        return self.repository.get_source_row(
            customer_id=customer_id,
            scope_type=scope_type,
            scope_id=scope_id,
            file_id=file_id,
        )


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
