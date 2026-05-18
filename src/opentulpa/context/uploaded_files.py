"""Shared uploaded-file prompt and summarization policy."""

from __future__ import annotations

import re
from typing import Any

XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCUMENT_EXTENSIONS = {
    ".csv",
    ".doc",
    ".docx",
    ".md",
    ".ods",
    ".pdf",
    ".rtf",
    ".tsv",
    ".txt",
    ".xls",
    ".xlsx",
}


def build_uploaded_files_context(
    records: list[dict[str, Any]], *, include_unclear_intent_guidance: bool = True
) -> str:
    if not records:
        return ""
    lines = [
        "Internal uploaded-file context. Do not quote this metadata verbatim to the user.",
        "Use file_id values with uploaded_file_* tools when deeper inspection is needed.",
        "If a spreadsheet, price list, FAQ, or policy is intended for workflow setup, start/open the setup session first, then prepare it with business_knowledge_index and query it with business_knowledge_query before activation.",
    ]
    if include_unclear_intent_guidance:
        lines.insert(
            2,
            "Infer upload intent from the recent message and conversation when it is clear. If the files should become reusable user/chat context, use user_context_add_files. If intent is unclear, ask what the user wants done with the files; do not infer intent from filenames or content alone.",
        )
        lines.append(
            "User-facing reply guidance: briefly acknowledge the upload and ask one focused follow-up question when the intended action is unclear."
        )
    for rec in records:
        mime_type = str(rec.get("mime_type", "")).strip()
        summary = str(rec.get("summary", "")).strip()
        if "ai_summary=" in summary:
            summary = summary.split("ai_summary=", 1)[1].strip()
        if (
            (mime_type.lower() == XLSX_MIME_TYPE or str(rec.get("original_filename", "")).lower().endswith(".xlsx"))
            and "no extractable text was available" in summary.lower()
        ):
            summary = (
                "Spreadsheet file stored. Use uploaded_file_inspect_structure with this file_id "
                "or business_knowledge_index to prepare workflow knowledge without loading the workbook into chat context. "
                "Do not use uploaded_file_analyze to create broad workflow source packs."
            )
        summary = re.sub(r"\s+", " ", summary)[:1200]
        lines.append(
            "- file_id={id} name={name} kind={kind} mime_type={mime_type} created_at={created_at} summary={summary}".format(
                id=str(rec.get("id", "")).strip(),
                name=str(rec.get("original_filename", "")).strip(),
                kind=str(rec.get("kind", "")).strip(),
                mime_type=mime_type or "unknown",
                created_at=str(rec.get("created_at", "")).strip(),
                summary=summary or "stored; no summary available yet",
            )
        )
    return "\n".join(lines)


def should_skip_auto_summary_for_upload(
    *, kind: str | None, filename: str | None, mime_type: str | None
) -> bool:
    """Avoid hauling knowledge-source documents into the LLM at upload time."""
    safe_kind = str(kind or "").strip().lower()
    if safe_kind in {"photo", "video", "video_note", "audio", "voice"}:
        return False
    safe_mime = str(mime_type or "").strip().lower()
    safe_name = str(filename or "").strip().lower()
    if safe_kind == "document":
        return True
    if safe_mime.startswith("text/") or safe_mime in {
        "application/pdf",
        "application/msword",
        "application/rtf",
        "application/vnd.ms-excel",
        XLSX_MIME_TYPE,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/csv",
        "text/tab-separated-values",
    }:
        return True
    return any(safe_name.endswith(ext) for ext in DOCUMENT_EXTENSIONS)
