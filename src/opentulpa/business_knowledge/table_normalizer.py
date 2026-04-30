"""Generic table normalization for business knowledge sources."""

from __future__ import annotations

import difflib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from opentulpa.business_knowledge.models import KnowledgeSourceSection

_ROW_RE = re.compile(r"^Row\s+(\d+):\s*(.*)$")
_TOKEN_RE = re.compile(r"[\w`-]+", re.UNICODE)
_MAX_FUZZY_CANDIDATES = 80


@dataclass(frozen=True, slots=True)
class TableCell:
    column: str
    column_index: int
    value: str


@dataclass(frozen=True, slots=True)
class TableFact:
    filename: str
    table: str
    source_order: int
    source_ref: str
    row: int
    item: str
    row_label: str
    column: str
    header: str
    value: str
    value_kind: str
    row_confidence: float
    header_confidence: float


@dataclass(frozen=True, slots=True)
class TableRowEvidence:
    filename: str
    table: str
    source_order: int
    source_ref: str
    row: int
    item: str
    row_label: str
    cells: tuple[TableFact, ...]
    score: float = 0.0


def table_facts_from_sections(sections: list[KnowledgeSourceSection]) -> list[TableFact]:
    """Normalize structured table sections into header-bound cell facts.

    The implementation is deliberately structural: it uses row geometry and cell value types,
    not domain aliases, language stopwords, or workbook-specific labels.
    """

    facts: list[TableFact] = []
    for section in sections:
        if section.source_kind != "structured_table":
            continue
        facts.extend(_table_facts_from_section(section))
    return facts


def select_table_evidence(
    facts: list[TableFact],
    *,
    query: str,
    target_terms: list[str],
    qualifier_terms: list[str],
    limit: int = 20,
) -> list[TableRowEvidence]:
    """Select compact table rows for an oracle using structured query intent."""

    rows = _rows_from_facts(facts)
    safe_targets = [term for term in _clean_terms(target_terms) if term]
    safe_qualifiers = [term for term in _clean_terms(qualifier_terms) if term]
    if not safe_targets:
        safe_targets = [str(query or "").strip()]

    candidates = _candidate_rows(
        rows,
        target_terms=safe_targets,
        qualifier_terms=safe_qualifiers,
        limit=max(_MAX_FUZZY_CANDIDATES, max(1, int(limit)) * 8),
    )
    scored: list[TableRowEvidence] = []
    for row, identity_text, context_text in candidates:
        target_score = _target_coverage_score(
            target_terms=safe_targets,
            identity_text=identity_text,
            context_text=context_text,
        )
        qualifier_score = _coverage_score(safe_qualifiers, context_text)
        direct_label_score = max((_direct_label_score(term, row.item) for term in safe_targets), default=0.0)
        # Item/table match is required. Qualifiers only sort plausible rows.
        if target_score <= 0:
            continue
        score = (target_score * 100.0) + (qualifier_score * 18.0) + (direct_label_score * 30.0)
        scored.append(
            TableRowEvidence(
                filename=row.filename,
                table=row.table,
                source_order=row.source_order,
                source_ref=row.source_ref,
                row=row.row,
                item=row.item,
                row_label=row.row_label,
                cells=row.cells,
                score=score,
            )
        )
    scored.sort(key=lambda row: (-row.score, row.source_order, row.row, row.table, row.item.casefold()))
    return scored[: max(1, int(limit))]


def table_evidence_selection_stats(
    facts: list[TableFact],
    *,
    query: str,
    target_terms: list[str],
    qualifier_terms: list[str],
    limit: int = 20,
) -> dict[str, Any]:
    rows = _rows_from_facts(facts)
    safe_targets = [term for term in _clean_terms(target_terms) if term]
    safe_qualifiers = [term for term in _clean_terms(qualifier_terms) if term]
    if not safe_targets:
        safe_targets = [str(query or "").strip()]
    candidates = _candidate_rows(
        rows,
        target_terms=safe_targets,
        qualifier_terms=safe_qualifiers,
        limit=max(_MAX_FUZZY_CANDIDATES, max(1, int(limit)) * 8),
    )
    return {
        "row_count": len(rows),
        "candidate_count": len(candidates),
        "candidate_limit": max(_MAX_FUZZY_CANDIDATES, max(1, int(limit)) * 8),
        "target_term_count": len(safe_targets),
        "qualifier_term_count": len(safe_qualifiers),
    }


def table_evidence_to_toon(
    rows: list[TableRowEvidence],
    *,
    query: str,
    target_terms: list[str],
    qualifier_terms: list[str],
) -> str:
    """Serialize selected table rows in a TOON-like compact form."""

    lines = [
        "intent:",
        f"  query: {_toon_value(query)}",
        f"  target_terms: {_toon_value(' | '.join(_clean_terms(target_terms)))}",
        f"  qualifier_terms: {_toon_value(' | '.join(_clean_terms(qualifier_terms)))}",
        "",
        "evidence_rows[rank,score,file,table,row,item,row_label,cells]:",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            ",".join(
                [
                    str(rank),
                    f"{row.score:.2f}",
                    _toon_value(row.filename),
                    _toon_value(row.table),
                    str(row.row),
                    _toon_value(row.item),
                    _toon_value(row.row_label),
                    _toon_value(_row_cells_text(row)),
                ]
            )
        )
    return "\n".join(lines).strip()


def table_overview_to_toon(
    facts: list[TableFact],
    *,
    query: str,
    category_terms: list[str],
    max_tables: int = 12,
    rows_per_table: int = 10,
) -> str:
    """Serialize a compact table/service inventory for broad business questions."""

    rows = _rows_from_facts(facts)
    safe_terms = _clean_terms(category_terms)
    rows_by_table: dict[str, list[TableRowEvidence]] = defaultdict(list)
    for row in rows:
        if safe_terms:
            table_score = _coverage_score(safe_terms, f"{row.table} {row.item} {row.row_label}")
            if table_score <= 0:
                continue
        rows_by_table[row.table].append(row)
    if safe_terms and not rows_by_table:
        for row in rows:
            if _coverage_score(safe_terms, f"{row.table} {row.item} {row.row_label}") > 0:
                rows_by_table[row.table].append(row)
    if not rows_by_table:
        for row in rows:
            rows_by_table[row.table].append(row)

    table_items = sorted(
        rows_by_table.items(),
        key=lambda item: (
            min(row.source_order for row in item[1]),
            item[0].casefold(),
        ),
    )[: max(1, int(max_tables))]

    lines = [
        "intent:",
        f"  query: {_toon_value(query)}",
        f"  category_terms: {_toon_value(' | '.join(safe_terms))}",
        "  mode: overview",
        "",
        "tables[rank,table,row_count,sample_items]:",
    ]
    for rank, (table, table_rows) in enumerate(table_items, start=1):
        sample_items = _unique([row.item for row in table_rows])[: max(1, int(rows_per_table))]
        lines.append(
            ",".join(
                [
                    str(rank),
                    _toon_value(table),
                    str(len(table_rows)),
                    _toon_value(" | ".join(sample_items)),
                ]
            )
        )
    lines.append("")
    lines.append("sample_rows[table,row,item,row_label,cells]:")
    for table, table_rows in table_items:
        for row in table_rows[: max(1, int(rows_per_table))]:
            lines.append(
                ",".join(
                    [
                        _toon_value(table),
                        str(row.row),
                        _toon_value(row.item),
                        _toon_value(row.row_label),
                        _toon_value(_row_cells_text(row)),
                    ]
                )
            )
    return "\n".join(lines).strip()


def _table_facts_from_section(section: KnowledgeSourceSection) -> list[TableFact]:
    metadata = section.metadata if isinstance(section.metadata, dict) else {}
    filename = str(metadata.get("filename", "") or "source").strip()
    table = str(
        metadata.get("section_title")
        or metadata.get("sheet")
        or metadata.get("source_label")
        or filename
    ).strip()
    parsed_rows = _parsed_rows(section.content)
    header_row_numbers = _detect_header_row_numbers(parsed_rows)
    header_rows = [cells for row, cells in parsed_rows if row in header_row_numbers]
    headers_by_column = _headers_by_column(header_rows)

    facts: list[TableFact] = []
    for row_number, cells in parsed_rows:
        if not cells or row_number in header_row_numbers:
            continue
        label_cell = _label_cell(cells)
        if label_cell is None:
            continue
        row_label = _clean_text(label_cell.value)
        item = _primary_label(row_label)
        if not item:
            continue
        row_confidence = 0.95 if label_cell.column_index <= 2 else 0.75
        for cell in cells:
            if cell.column_index == label_cell.column_index and cell.value == label_cell.value:
                continue
            value = cell.value
            if not value:
                continue
            header = headers_by_column.get(cell.column_index, cell.column)
            facts.append(
                TableFact(
                    filename=filename,
                    table=table,
                    source_order=int(section.sort_order),
                    source_ref=section.source_ref,
                    row=row_number,
                    item=item,
                    row_label=row_label,
                    column=cell.column,
                    header=header,
                    value=value,
                    value_kind=_value_kind(value),
                    row_confidence=row_confidence,
                    header_confidence=0.95 if header != cell.column else 0.5,
                )
            )
    return facts


def _parsed_rows(content: str) -> list[tuple[int, list[TableCell]]]:
    rows: list[tuple[int, list[TableCell]]] = []
    seen: set[int] = set()
    for line in str(content or "").splitlines():
        parsed = _parse_row_line(line)
        if parsed is None:
            continue
        row_number, cells = parsed
        if row_number in seen:
            continue
        seen.add(row_number)
        rows.append((row_number, cells))
    return rows


def _parse_row_line(line: str) -> tuple[int, list[TableCell]] | None:
    match = _ROW_RE.match(str(line or "").strip())
    if not match:
        return None
    row_number = int(match.group(1))
    cells: list[TableCell] = []
    for part in match.group(2).split(" | "):
        if "=" not in part:
            continue
        raw_column, raw_value = part.split("=", 1)
        column_index = _column_index(raw_column.strip())
        value = _clean_text(raw_value)
        if column_index <= 0 or not value:
            continue
        cells.append(
            TableCell(
                column=_column_label(column_index),
                column_index=column_index,
                value=value,
            )
        )
    return row_number, cells


def _headers_by_column(header_rows: list[list[TableCell]]) -> dict[int, str]:
    values_by_column: dict[int, list[str]] = defaultdict(list)
    for cells in header_rows[:3]:
        if _is_repeated_context_row(cells):
            continue
        for cell in cells:
            if _value_kind(cell.value) in {"blank", "empty_marker"}:
                continue
            values_by_column[cell.column_index].append(cell.value)
    return {
        column_index: " / ".join(_unique(values))
        for column_index, values in values_by_column.items()
        if values
    }


def _detect_header_row_numbers(parsed_rows: list[tuple[int, list[TableCell]]]) -> set[int]:
    header_rows: set[int] = set()
    for row_number, cells in parsed_rows:
        if _is_data_like_row(cells):
            break
        if _is_header_candidate(cells):
            header_rows.add(row_number)
    if header_rows:
        return header_rows
    # Fallback for section fragments where the first true data row is outside the slice.
    return {row_number for row_number, cells in parsed_rows[:3] if _is_header_candidate(cells)}


def _is_header_candidate(cells: list[TableCell]) -> bool:
    if len(cells) < 2:
        return False
    text_count = sum(1 for cell in cells if _value_kind(cell.value) == "text")
    return text_count >= max(2, len(cells) // 2)


def _is_repeated_context_row(cells: list[TableCell]) -> bool:
    if len(cells) < 3:
        return False
    values = [_normalized_text(cell.value) for cell in cells if _normalized_text(cell.value)]
    if len(values) < 3:
        return False
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[value] += 1
    return max(counts.values(), default=0) / len(values) >= 0.6


def _is_data_like_row(cells: list[TableCell]) -> bool:
    if len(cells) < 2:
        return False
    label = _label_cell(cells)
    if label is None or _value_kind(label.value) != "text":
        return False
    value_count = sum(
        1
        for cell in cells
        if cell.column_index != label.column_index and _value_kind(cell.value) != "text"
    )
    return value_count >= 1


def _label_cell(cells: list[TableCell]) -> TableCell | None:
    for cell in cells:
        if _value_kind(cell.value) == "text":
            return cell
    return cells[0] if cells else None


def _value_kind(value: str) -> str:
    text = _clean_text(value)
    if not text:
        return "blank"
    if re.fullmatch(r"[-—–]+", text):
        return "empty_marker"
    if re.fullmatch(r"[\d\s.,]+", text):
        return "number"
    if re.fullmatch(r"\d{1,4}[-./]\d{1,2}[-./]\d{1,4}", text):
        return "date"
    chars = [char for char in text if not char.isspace()]
    digits = sum(char.isdigit() for char in chars)
    letters = sum(char.isalpha() for char in chars)
    if digits and letters == 0:
        return "number"
    if digits and letters > 0 and digits >= letters * 2:
        return "mixed_value"
    return "text"


def _rows_from_facts(facts: list[TableFact]) -> list[TableRowEvidence]:
    grouped: dict[tuple[str, int, str], list[TableFact]] = defaultdict(list)
    for fact in facts:
        grouped[(fact.source_ref, fact.row, fact.item)].append(fact)
    rows = [
        TableRowEvidence(
            filename=items[0].filename,
            table=items[0].table,
            source_order=items[0].source_order,
            source_ref=items[0].source_ref,
            row=items[0].row,
            item=items[0].item,
            row_label=items[0].row_label,
            cells=tuple(items),
        )
        for items in grouped.values()
        if items
    ]
    rows.sort(key=lambda row: (row.source_order, row.row, row.table, row.item.casefold()))
    return rows


def _candidate_rows(
    rows: list[TableRowEvidence],
    *,
    target_terms: list[str],
    qualifier_terms: list[str],
    limit: int,
) -> list[tuple[TableRowEvidence, str, str]]:
    prepared: list[tuple[float, TableRowEvidence, str, str]] = []
    for row in rows:
        identity_text = f"{row.table} {row.item} {row.row_label}"
        context_text = " ".join(
            f"{fact.item} {fact.row_label} {fact.header} {fact.column} {fact.value}"
            for fact in row.cells
        )
        target_score = _cheap_coverage_score(target_terms, identity_text, context_text)
        if target_score <= 0:
            continue
        qualifier_score = _cheap_coverage_score(qualifier_terms, context_text, "")
        direct_score = max(
            (_cheap_phrase_score(term, row.item) for term in target_terms),
            default=0.0,
        )
        prepared.append(
            (
                (target_score * 100.0) + (qualifier_score * 18.0) + (direct_score * 30.0),
                row,
                identity_text,
                context_text,
            )
        )
    if not prepared:
        prepared = _fallback_identity_candidates(rows, target_terms=target_terms)
    prepared.sort(
        key=lambda item: (
            -item[0],
            item[1].source_order,
            item[1].row,
            item[1].table,
            item[1].item.casefold(),
        )
    )
    return [(row, identity, context) for _, row, identity, context in prepared[: max(1, int(limit))]]


def _fallback_identity_candidates(
    rows: list[TableRowEvidence],
    *,
    target_terms: list[str],
) -> list[tuple[float, TableRowEvidence, str, str]]:
    candidates: list[tuple[float, TableRowEvidence, str, str]] = []
    for row in rows:
        identity_text = f"{row.table} {row.item} {row.row_label}"
        context_text = " ".join(
            f"{fact.item} {fact.row_label} {fact.header} {fact.column} {fact.value}"
            for fact in row.cells
        )
        score = max((_phrase_match_score(term, identity_text) for term in target_terms), default=0.0)
        if score > 0:
            candidates.append((score * 100.0, row, identity_text, context_text))
    return candidates


def _cheap_coverage_score(needles: list[str], identity_text: str, context_text: str) -> float:
    terms = [term for term in needles if _normalized_text(term)]
    if not terms:
        return 0.0
    identity = _normalized_text(identity_text)
    context = _normalized_text(context_text)
    return sum(
        max(
            _cheap_phrase_score(term, identity),
            _cheap_phrase_score(term, context) * 0.75,
        )
        for term in terms
    ) / len(terms)


def _cheap_phrase_score(needle: str, haystack: str) -> float:
    needle_norm = _normalized_text(needle)
    haystack_norm = _normalized_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0
    if needle_norm in haystack_norm or haystack_norm in needle_norm:
        return 1.0
    needle_tokens = _tokens(needle_norm)
    haystack_tokens = _tokens(haystack_norm)
    if not needle_tokens or not haystack_tokens:
        return 0.0
    haystack_token_set = set(haystack_tokens)
    haystack_prefixes = {
        token[:4]
        for token in haystack_token_set
        if len(token) >= 4 and not any(char.isdigit() for char in token)
    }
    matched = sum(
        1
        for token in needle_tokens
        if token in haystack_token_set
        or (
            len(token) >= 4
            and not any(char.isdigit() for char in token)
            and token[:4] in haystack_prefixes
        )
    )
    return matched / len(needle_tokens)


def _phrase_match_score(needle: str, haystack: str) -> float:
    needle_norm = _normalized_text(needle)
    haystack_norm = _normalized_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0
    if needle_norm in haystack_norm or haystack_norm in needle_norm:
        return 1.0
    if len(needle_norm) > 80 or len(haystack_norm) > 300:
        return _cheap_phrase_score(needle_norm, haystack_norm)
    needle_tokens = _tokens(needle_norm)
    haystack_tokens = _tokens(haystack_norm)
    if not needle_tokens or not haystack_tokens:
        return 0.0
    token_score = sum(
        1 for token in needle_tokens if any(_token_matches(token, candidate) for candidate in haystack_tokens)
    ) / len(needle_tokens)
    sequence_score = difflib.SequenceMatcher(None, needle_norm, haystack_norm).ratio() * 0.8
    return max(token_score, sequence_score)


def _direct_label_score(term: str, item: str) -> float:
    term_norm = _normalized_text(term)
    item_norm = _normalized_text(item)
    if not term_norm or not item_norm:
        return 0.0
    if term_norm == item_norm:
        return 1.0
    if item_norm in term_norm or term_norm in item_norm:
        shorter = min(len(term_norm), len(item_norm))
        longer = max(len(term_norm), len(item_norm))
        return shorter / longer if longer else 0.0
    return _phrase_match_score(term_norm, item_norm) * 0.5


def _target_coverage_score(*, target_terms: list[str], identity_text: str, context_text: str) -> float:
    if not target_terms:
        return 0.0
    identity_score = _coverage_score(target_terms, identity_text)
    context_score = _coverage_score(target_terms, context_text)
    return (identity_score * 0.75) + (context_score * 0.25)


def _coverage_score(needles: list[str], haystack: str) -> float:
    terms = [term for term in needles if _normalized_text(term)]
    if not terms:
        return 0.0
    return sum(_phrase_match_score(term, haystack) for term in terms) / len(terms)


def _token_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    if any(char.isdigit() for char in left + right):
        return False
    shared_prefix = _shared_prefix_len(left, right)
    shorter = min(len(left), len(right))
    if shared_prefix >= 4 and shorter >= 4 and shared_prefix / shorter >= 0.66:
        return True
    if abs(len(left) - len(right)) > max(2, shorter // 2):
        return False
    return len(left) >= 4 and len(right) >= 4 and difflib.SequenceMatcher(None, left, right).ratio() >= 0.9


def _shared_prefix_len(left: str, right: str) -> int:
    count = 0
    for left_char, right_char in zip(left, right, strict=False):
        if left_char != right_char:
            break
        count += 1
    return count


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or ""))


def _normalized_text(text: str) -> str:
    value = str(text or "").casefold().replace("ё", "е").replace("х", "x")
    return " ".join(_TOKEN_RE.findall(value))


def _primary_label(value: str) -> str:
    text = _clean_text(value)
    parts = re.split(r"\s[*•●]\s*|\s{2,}", text, maxsplit=1)
    return _clean_text(parts[0]) or text


def _row_cells_text(row: TableRowEvidence) -> str:
    cells = [fact for fact in row.cells if fact.value_kind != "blank"]
    header_counts: dict[str, int] = defaultdict(int)
    header_totals: dict[str, int] = defaultdict(int)
    for fact in cells:
        header_totals[_normalized_text(fact.header)] += 1

    group_by_column = _header_group_by_column(cells)
    show_groups = bool(group_by_column) and any(total > 1 for total in header_totals.values())

    parts: list[str] = []
    for fact in cells:
        key = _normalized_text(fact.header)
        header_counts[key] += 1
        occurrence = f" [{header_counts[key]}]" if header_totals[key] > 1 else ""
        group = f"header_group {group_by_column[fact.column]} " if show_groups else ""
        parts.append(f"{fact.column} {group}{fact.header}{occurrence} = {fact.value}")
    return "; ".join(parts)


def _header_group_by_column(cells: list[TableFact]) -> dict[str, int]:
    groups: dict[str, int] = {}
    last_header = ""
    current_group = 0
    for fact in sorted(cells, key=lambda item: _column_index(item.column)):
        header = _normalized_text(fact.header)
        if not header:
            continue
        if header != last_header:
            current_group += 1
            last_header = header
        groups[fact.column] = current_group
    return groups


def _toon_value(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return '""'
    if any(char in text for char in [",", ":", "\n", '"']) or len(text.split()) > 1:
        return json.dumps(text, ensure_ascii=False)
    return text


def _clean_terms(values: list[str]) -> list[str]:
    return _unique([_clean_text(value) for value in values])


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _column_index(column: str) -> int:
    text = str(column or "").strip()
    if text.casefold().startswith("col_"):
        try:
            return int(text.split("_", 1)[1])
        except ValueError:
            return -1
    total = 0
    for char in text.upper():
        if not ("A" <= char <= "Z"):
            return -1
        total = total * 26 + (ord(char) - ord("A") + 1)
    return total


def _column_label(index: int) -> str:
    if index <= 0:
        return ""
    value = index
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
