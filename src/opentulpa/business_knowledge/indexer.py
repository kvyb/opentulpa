"""Business knowledge source indexing."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from opentulpa.business_knowledge.extraction import content_hash, extract_source_sections
from opentulpa.business_knowledge.models import KnowledgeIndexedSource
from opentulpa.business_knowledge.repository import BusinessKnowledgeRepository


class BusinessKnowledgeIndexer:
    def __init__(
        self,
        *,
        file_vault: Any,
        repository: BusinessKnowledgeRepository,
        now_iso: Callable[[], str],
    ) -> None:
        self.file_vault = file_vault
        self.repository = repository
        self.now_iso = now_iso

    def index_one_source(
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
        existing = self.repository.get_source_row(
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
        self.repository.replace_source(
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
            sections=sections,
            indexed_at=self.now_iso(),
        )
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


def _safe_list_json(value: Any) -> list[str]:
    try:
        parsed = json.loads(str(value or "[]"))
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []
