"""Models for workflow-scoped business knowledge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class KnowledgeSourceSection:
    """One LLM-readable source section with stable provenance."""

    content: str
    source_ref: str
    source_kind: str = "local_source"
    metadata: dict[str, Any] = field(default_factory=dict)
    sort_order: int = 0


@dataclass(frozen=True, slots=True)
class KnowledgeIndexedSource:
    file_id: str
    filename: str
    mime_type: str
    status: str
    source_kind: str
    section_count: int
    char_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KnowledgeQueryAnswer:
    answer_extract: str


@dataclass(frozen=True, slots=True)
class KnowledgeQueryResult:
    ok: bool
    query: str
    scope_type: str
    scope_id: str
    answer: KnowledgeQueryAnswer
    warnings: list[str] = field(default_factory=list)
    source_count: int = 0
    section_count: int = 0
    cached: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
