"""Bounded parsing and inspection for uploaded files."""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field

from opentulpa.business_knowledge.extraction import extract_source_sections
from opentulpa.context.file_vault import FileVaultService
from opentulpa.jobs import (
    JobArguments,
    JobExecutionContext,
    JobHandlerRegistry,
    JobHandlerResult,
)


class FileAnalysisJobArguments(JobArguments):
    file_id: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=20_000)


class FileAnalysisService:
    def __init__(self, files: FileVaultService) -> None:
        self._files = files

    def register_handlers(self, registry: JobHandlerRegistry) -> None:
        registry.register(
            name="file_analyze",
            arguments_model=FileAnalysisJobArguments,
            handler=self._analyze_job,
            timeout_seconds=300,
        )

    def inspect(
        self,
        *,
        tenant_id: str,
        file_id: str,
        question: str | None = None,
    ) -> dict[str, Any]:
        record = self._files.get_file(tenant_id, file_id)
        raw = self._files.read_file_bytes(tenant_id, file_id)
        if record is None or raw is None:
            raise KeyError(file_id)
        sections, warnings, source_kind = extract_source_sections(record=record, raw_bytes=raw)
        query_terms = {
            term
            for term in re.findall(r"[\w-]{3,}", str(question or "").casefold())
            if term
        }
        ranked = sorted(
            sections,
            key=lambda section: sum(
                term in section.content.casefold() for term in query_terms
            ),
            reverse=True,
        )
        return {
            "tenant_id": tenant_id,
            "file_id": file_id,
            "filename": str(record.get("original_filename") or "file.bin"),
            "mime_type": str(record.get("mime_type") or ""),
            "source_kind": source_kind,
            "warnings": warnings[:20],
            "section_count": len(sections),
            "sections": [
                {
                    "source_ref": section.source_ref,
                    "source_kind": section.source_kind,
                    "metadata": section.metadata,
                    "preview": section.content[:4_000],
                }
                for section in ranked[:20]
            ],
        }

    async def _analyze_job(
        self,
        arguments: FileAnalysisJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        await context.progress({"stage": "parsing"})
        inspection = self.inspect(
            tenant_id=context.tenant_id,
            file_id=arguments.file_id,
            question=arguments.instruction,
        )
        sections = inspection.get("sections")
        previews = [
            str(item.get("preview") or "")
            for item in sections if isinstance(item, dict)
        ] if isinstance(sections, list) else []
        extracted = "\n\n".join(previews).strip()[:8_000]
        if not extracted:
            extracted = "No extractable text was available for this file type."
        summary = (
            f"instruction={arguments.instruction[:1_000]}\n"
            f"source_kind={inspection['source_kind']}\n"
            f"section_count={inspection['section_count']}\n"
            f"extract={extracted}"
        )[:12_000]
        updated = self._files.set_ai_summary(context.tenant_id, arguments.file_id, summary)
        if updated is None:
            raise KeyError(arguments.file_id)
        await context.progress({"stage": "completed"})
        return JobHandlerResult(
            summary="File analysis completed",
            data={
                "file_id": arguments.file_id,
                "source_kind": str(inspection["source_kind"]),
                "section_count": int(inspection["section_count"]),
                "warnings": list(inspection["warnings"]),
                "summary": summary[:4_000],
            },
        )


__all__ = ["FileAnalysisJobArguments", "FileAnalysisService"]
