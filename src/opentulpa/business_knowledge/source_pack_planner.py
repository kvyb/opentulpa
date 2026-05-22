"""Business knowledge source-pack planning."""

from __future__ import annotations

import csv
import re
import time
from collections.abc import Callable
from io import StringIO
from typing import Any

from opentulpa.business_knowledge.models import KnowledgeSourceSection
from opentulpa.business_knowledge.table_normalizer import (
    select_table_evidence,
    table_evidence_selection_stats,
    table_evidence_to_toon,
    table_facts_from_sections,
    table_overview_to_toon,
)

IntentExtractor = Callable[[str], dict[str, Any]]
SpanRecorder = Callable[..., None]


class BusinessKnowledgeSourcePackPlanner:
    """Selects compact source evidence for one business-knowledge query."""

    def __init__(
        self,
        *,
        extract_intent: IntentExtractor,
        record_span: SpanRecorder,
    ) -> None:
        self._extract_intent = extract_intent
        self._record_span = record_span

    def plan(
        self,
        *,
        sections: list[KnowledgeSourceSection],
        query: str,
    ) -> tuple[str, dict[str, Any]]:
        started = time.monotonic()
        timing_ms: dict[str, int] = {}
        facts = self._table_facts(sections=sections, query=query, timing_ms=timing_ms)
        if not facts:
            return _section_pack(sections=sections, started=started, timing_ms=timing_ms)

        intent = self._query_intent(query=query, timing_ms=timing_ms)
        if intent["mode"] in {"category_overview", "corpus_overview"}:
            return _overview_pack(
                sections=sections,
                facts=facts,
                query=query,
                intent=intent,
                started=started,
                timing_ms=timing_ms,
                mode=str(intent["mode"]),
            )

        rows = self._select_table_rows(
            facts=facts,
            query=query,
            intent=intent,
            timing_ms=timing_ms,
        )
        if not rows:
            return _overview_pack(
                sections=sections,
                facts=facts,
                query=query,
                intent=intent,
                started=started,
                timing_ms=timing_ms,
                mode="fallback_overview",
                selected_row_count=0,
            )

        return _evidence_pack(
            sections=sections,
            facts=facts,
            query=query,
            intent=intent,
            rows=rows,
            started=started,
            timing_ms=timing_ms,
        )

    def _table_facts(
        self,
        *,
        sections: list[KnowledgeSourceSection],
        query: str,
        timing_ms: dict[str, int],
    ) -> list[Any]:
        facts_started = time.monotonic()
        facts = table_facts_from_sections(sections)
        timing_ms["table_facts"] = _elapsed_ms(facts_started)
        self._record_span(
            name="knowledge.table_facts",
            input={"section_count": len(sections), "query": query},
            output={"fact_count": len(facts)},
            metadata={"timing_ms": {"table_facts": timing_ms["table_facts"]}},
        )
        return facts

    def _query_intent(self, *, query: str, timing_ms: dict[str, int]) -> dict[str, Any]:
        intent_started = time.monotonic()
        intent = self._extract_intent(query)
        timing_ms["intent"] = _elapsed_ms(intent_started)
        self._record_span(
            name="knowledge.extract_intent",
            input={"query": query},
            output=safe_intent_diagnostics(intent),
            metadata={"timing_ms": {"intent": timing_ms["intent"]}},
        )
        return intent

    def _select_table_rows(
        self,
        *,
        facts: list[Any],
        query: str,
        intent: dict[str, Any],
        timing_ms: dict[str, int],
    ) -> list[Any]:
        selection_started = time.monotonic()
        rows = select_table_evidence(
            facts,
            query=query,
            target_terms=intent["target_terms"],
            qualifier_terms=intent["qualifier_terms"],
            limit=20,
        )
        timing_ms["table_select"] = _elapsed_ms(selection_started)
        self._record_span(
            name="knowledge.select_table_evidence",
            input={"query": query, "fact_count": len(facts)},
            output={"selected_row_count": len(rows)},
            metadata={"timing_ms": {"table_select": timing_ms["table_select"]}},
        )
        return rows


def _section_pack(
    *,
    sections: list[KnowledgeSourceSection],
    started: float,
    timing_ms: dict[str, int],
) -> tuple[str, dict[str, Any]]:
    source_pack = source_pack_for_sections(sections)
    timing_ms["source_pack_total"] = _elapsed_ms(started)
    return source_pack, {
        "mode": "section_pack",
        "chars": len(source_pack),
        "section_count": len(sections),
        "fact_count": 0,
        "timing_ms": timing_ms,
    }


def _overview_pack(
    *,
    sections: list[KnowledgeSourceSection],
    facts: list[Any],
    query: str,
    intent: dict[str, Any],
    started: float,
    timing_ms: dict[str, int],
    mode: str,
    selected_row_count: int | None = None,
) -> tuple[str, dict[str, Any]]:
    overview_started = time.monotonic()
    source_pack = table_overview_to_toon(
        facts,
        query=query,
        category_terms=[*intent["target_terms"], *intent["qualifier_terms"]],
    )
    source_pack, supplemental_section_count = with_relevant_non_table_sections(
        source_pack=source_pack,
        sections=sections,
        query=query,
    )
    timing_ms["table_overview"] = _elapsed_ms(overview_started)
    timing_ms["source_pack_total"] = _elapsed_ms(started)
    diagnostics: dict[str, Any] = {
        "mode": mode,
        "chars": len(source_pack),
        "section_count": len(sections),
        "supplemental_section_count": supplemental_section_count,
        "fact_count": len(facts),
        "intent": safe_intent_diagnostics(intent),
        "timing_ms": timing_ms,
    }
    if selected_row_count is not None:
        diagnostics["selected_row_count"] = selected_row_count
    return source_pack, diagnostics


def _evidence_pack(
    *,
    sections: list[KnowledgeSourceSection],
    facts: list[Any],
    query: str,
    intent: dict[str, Any],
    rows: list[Any],
    started: float,
    timing_ms: dict[str, int],
) -> tuple[str, dict[str, Any]]:
    evidence_started = time.monotonic()
    source_pack = table_evidence_to_toon(
        rows,
        query=query,
        target_terms=intent["target_terms"],
        qualifier_terms=intent["qualifier_terms"],
    )
    source_pack, supplemental_section_count = with_relevant_non_table_sections(
        source_pack=source_pack,
        sections=sections,
        query=query,
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
        "supplemental_section_count": supplemental_section_count,
        "fact_count": len(facts),
        "selected_row_count": len(rows),
        "selection": selection_stats,
        "intent": safe_intent_diagnostics(intent),
        "timing_ms": timing_ms,
    }


def source_pack_for_sections(sections: list[KnowledgeSourceSection]) -> str:
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
                    section_content_for_oracle(section),
                ]
            ).strip()
        )
    return "\n\n".join(parts).strip()


def with_relevant_non_table_sections(
    *,
    source_pack: str,
    sections: list[KnowledgeSourceSection],
    query: str,
) -> tuple[str, int]:
    selected = select_relevant_non_table_sections(sections=sections, query=query)
    if not selected:
        return source_pack, 0
    supplement = source_pack_for_sections(selected)
    if not supplement:
        return source_pack, 0
    return "\n\n## Supplemental non-table sources\n\n".join([source_pack, supplement]).strip(), len(selected)


def select_relevant_non_table_sections(
    *,
    sections: list[KnowledgeSourceSection],
    query: str,
    limit: int = 8,
) -> list[KnowledgeSourceSection]:
    candidates = [section for section in sections if section.source_kind != "structured_table"]
    if not candidates:
        return []
    tokens = query_tokens(query)
    if not tokens:
        return candidates[:limit]
    scored: list[tuple[int, int, str, KnowledgeSourceSection]] = []
    for section in candidates:
        metadata = section.metadata if isinstance(section.metadata, dict) else {}
        haystack = " ".join(
            [
                str(section.source_ref or ""),
                str(section.source_kind or ""),
                str(metadata.get("filename", "") or ""),
                str(metadata.get("section_title", "") or ""),
                str(section.content or ""),
            ]
        )
        score = len(tokens & query_tokens(haystack))
        if score:
            scored.append((score, int(section.sort_order or 0), str(section.source_ref), section))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [item[3] for item in scored[:limit]]


def query_tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\w-]+", str(value or ""), flags=re.UNICODE)
        if len(token.strip()) >= 2
    }


def section_content_for_oracle(section: KnowledgeSourceSection) -> str:
    content = str(section.content or "").strip()
    if section.source_kind != "structured_table":
        return content
    lines = content.splitlines()
    table_lines: list[str] = []
    for line in lines:
        if line.startswith(("Workbook:", "Sheet:", "Table:", "Rows:")):
            table_lines.append(line)
    cell_rows = normalized_cell_rows(content)
    if not cell_rows:
        return content
    out = StringIO()
    writer = csv.writer(out)
    writer.writerow(["source_ref", "row", "column", "value"])
    for row in cell_rows:
        writer.writerow([section.source_ref, *row])
    return "\n".join([*table_lines, "normalized_csv:", out.getvalue().strip()]).strip()


def normalized_cell_rows(content: str) -> list[tuple[str, str, str]]:
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


def safe_intent_diagnostics(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": str(intent.get("mode", "") or ""),
        "target_term_count": len(_safe_text_list(intent.get("target_terms"))),
        "qualifier_term_count": len(_safe_text_list(intent.get("qualifier_terms"))),
    }


def _safe_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
