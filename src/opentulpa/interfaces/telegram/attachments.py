"""Telegram attachment extraction and ingest pipeline."""

from __future__ import annotations

import re
from contextlib import suppress
from pathlib import Path
from typing import Any

from opentulpa.context.file_vault import FileVaultService
from opentulpa.interfaces.telegram.client import TelegramClient
from opentulpa.interfaces.telegram.models import TelegramAttachment

PROJECT_ROOT = Path(__file__).resolve().parents[4]
XLSX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_DOCUMENT_EXTENSIONS = {
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


def _safe_segment(value: str, *, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return clean[:180] or fallback


def _mirror_uploaded_file(
    *,
    customer_id: str,
    file_id: str,
    filename: str,
    raw_bytes: bytes,
) -> str | None:
    customer_seg = _safe_segment(customer_id, fallback="customer")
    safe_name = _safe_segment(filename, fallback="file.bin")
    rel_path = Path("tulpa_stuff") / "uploads" / customer_seg / f"{file_id}_{safe_name}"
    abs_path = (PROJECT_ROOT / rel_path).resolve()
    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(raw_bytes)
        return str(rel_path)
    except Exception:
        return None


def extract_attachments(message: dict[str, Any]) -> list[TelegramAttachment]:
    attachments: list[TelegramAttachment] = []

    document = message.get("document")
    if isinstance(document, dict):
        fid = str(document.get("file_id", "")).strip()
        if fid:
            attachments.append(
                TelegramAttachment(
                    kind="document",
                    file_id=fid,
                    filename=str(document.get("file_name", "")).strip() or None,
                    mime_type=str(document.get("mime_type", "")).strip() or None,
                    file_size=int(document.get("file_size") or 0) or None,
                )
            )

    photos = message.get("photo")
    if isinstance(photos, list) and photos:
        chosen: dict[str, Any] | None = None
        for item in photos:
            if not isinstance(item, dict):
                continue
            if chosen is None or int(item.get("file_size") or 0) >= int(chosen.get("file_size") or 0):
                chosen = item
        if chosen:
            fid = str(chosen.get("file_id", "")).strip()
            if fid:
                unique = str(chosen.get("file_unique_id", "")).strip() or "photo"
                attachments.append(
                    TelegramAttachment(
                        kind="photo",
                        file_id=fid,
                        filename=f"{unique}.jpg",
                        mime_type="image/jpeg",
                        file_size=int(chosen.get("file_size") or 0) or None,
                    )
                )

    for key in ("video", "video_note", "audio", "voice"):
        item = message.get(key)
        if not isinstance(item, dict):
            continue
        fid = str(item.get("file_id", "")).strip()
        if not fid:
            continue
        unique = str(item.get("file_unique_id", "")).strip() or key
        ext = {
            "video": ".mp4",
            "video_note": ".mp4",
            "audio": ".mp3",
            "voice": ".ogg",
        }.get(key, "")
        filename = str(item.get("file_name", "")).strip() or f"{unique}{ext}"
        attachments.append(
            TelegramAttachment(
                kind=key,
                file_id=fid,
                filename=filename,
                mime_type=str(item.get("mime_type", "")).strip() or None,
                file_size=int(item.get("file_size") or 0) or None,
            )
        )
    return attachments


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


def _skip_auto_summary_for_upload(*, kind: str | None, filename: str | None, mime_type: str | None) -> bool:
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
    return any(safe_name.endswith(ext) for ext in _DOCUMENT_EXTENSIONS)


async def ingest_attachments(
    *,
    attachments: list[TelegramAttachment],
    bot_token: str,
    file_vault: FileVaultService,
    memory: Any | None,
    agent_runtime: Any | None,
    customer_id: str,
    chat_id: int,
    caption: str | None,
) -> list[dict[str, Any]]:
    ingested: list[dict[str, Any]] = []
    client = TelegramClient(bot_token)
    try:
        for attachment in attachments:
            downloaded = await client.download_file(file_id=attachment.file_id)
            if not downloaded:
                original_name = attachment.filename or f"{attachment.kind}.bin"
                unavailable_note = (
                    "Telegram file was received but could not be downloaded by the bot. "
                    f"original_name={original_name} "
                    f"original_kind={attachment.kind} "
                    f"original_mime_type={attachment.mime_type or 'unknown'} "
                    f"telegram_file_id={attachment.file_id} "
                    f"telegram_file_size_bytes={attachment.file_size or 'unknown'}. "
                    "This can happen when Telegram refuses Bot API download for the file size "
                    "or the file is temporarily unavailable. Ask the user for a shorter/compressed "
                    "clip or a direct video URL before claiming to analyze the video."
                )
                record = file_vault.ingest_file(
                    customer_id=customer_id,
                    chat_id=chat_id,
                    kind=f"unavailable_{attachment.kind}",
                    telegram_file_id=attachment.file_id,
                    original_filename=f"{_safe_segment(original_name, fallback=attachment.kind)}.download-unavailable.txt",
                    mime_type="text/plain",
                    caption=caption,
                    raw_bytes=unavailable_note.encode("utf-8"),
                )
                updated = file_vault.set_ai_summary(
                    customer_id, str(record.get("id", "")), unavailable_note
                )
                if isinstance(updated, dict):
                    record = updated
                ingested.append(record)
                if memory is not None:
                    with suppress(Exception):
                        memory.add_text(
                            unavailable_note,
                            user_id=customer_id,
                            metadata={
                                "kind": "file_fact",
                                "file_id": record.get("id"),
                                "file_kind": record.get("kind"),
                            },
                            infer=False,
                        )
                continue
            raw_bytes = downloaded.get("raw_bytes")
            if not isinstance(raw_bytes, (bytes, bytearray)) or not raw_bytes:
                continue
            file_path_name = str(downloaded.get("file_path", "")).split("/")[-1].strip()
            record = file_vault.ingest_file(
                customer_id=customer_id,
                chat_id=chat_id,
                kind=attachment.kind,
                telegram_file_id=attachment.file_id,
                original_filename=attachment.filename or file_path_name or f"{attachment.kind}.bin",
                mime_type=attachment.mime_type or str(downloaded.get("mime_type", "")).strip() or None,
                caption=caption,
                raw_bytes=bytes(raw_bytes),
            )
            local_path = _mirror_uploaded_file(
                customer_id=customer_id,
                file_id=str(record.get("id", "")).strip(),
                filename=str(record.get("original_filename", "")).strip() or "file.bin",
                raw_bytes=bytes(raw_bytes),
            )
            if local_path:
                record = {**record, "local_path": local_path}
            if (
                attachment.kind == "voice"
                and agent_runtime is not None
                and hasattr(agent_runtime, "transcribe_audio_blob")
            ):
                with suppress(Exception):
                    transcript = await agent_runtime.transcribe_audio_blob(
                        filename=attachment.filename or file_path_name or f"{attachment.kind}.ogg",
                        mime_type=attachment.mime_type
                        or str(downloaded.get("mime_type", "")).strip()
                        or None,
                        kind=attachment.kind,
                        raw_bytes=bytes(raw_bytes),
                    )
                    if transcript:
                        record = {**record, "voice_transcript": str(transcript).strip()[:4000]}
            if (
                agent_runtime is not None
                and hasattr(agent_runtime, "summarize_uploaded_blob")
                and not _skip_auto_summary_for_upload(
                    kind=attachment.kind,
                    filename=attachment.filename or file_path_name,
                    mime_type=attachment.mime_type or str(downloaded.get("mime_type", "")).strip() or None,
                )
            ):
                if attachment.kind == "voice":
                    ingested.append(record)
                    if memory is not None:
                        with suppress(Exception):
                            memory.add_text(
                                (
                                    "Voice note stored for this user. "
                                    f"name={record.get('original_filename')} "
                                    f"vault_path={record.get('stored_path')} "
                                    f"local_path={record.get('local_path', '')} "
                                    f"transcript={str(record.get('voice_transcript', ''))[:1200]}"
                                ),
                                user_id=customer_id,
                                metadata={
                                    "kind": "media_fact",
                                    "file_id": record.get("id"),
                                    "file_kind": record.get("kind"),
                                },
                                infer=False,
                            )
                    continue
                with suppress(Exception):
                    ai_summary = await agent_runtime.summarize_uploaded_blob(
                        filename=str(record.get("original_filename", "")).strip() or None,
                        mime_type=str(record.get("mime_type", "")).strip() or None,
                        kind=str(record.get("kind", "")).strip() or None,
                        raw_bytes=bytes(raw_bytes),
                        caption=caption,
                    )
                    if ai_summary:
                        updated = file_vault.set_ai_summary(
                            customer_id, str(record.get("id", "")), ai_summary
                        )
                        if isinstance(updated, dict):
                            record = updated
            ingested.append(record)
            if memory is not None:
                with suppress(Exception):
                    record_kind = str(record.get("kind", "")).strip().lower()
                    memory_kind = (
                        "media_fact"
                        if record_kind in {"photo", "video", "video_note", "audio", "voice"}
                        else "file_fact"
                    )
                    memory.add_text(
                        (
                            "User file stored in vault. "
                            f"name={record.get('original_filename')} "
                            f"kind={record.get('kind')} "
                            f"vault_path={record.get('stored_path')} "
                            f"local_path={record.get('local_path', '')} "
                            f"summary={record.get('summary', '')[:1200]}"
                        ),
                        user_id=customer_id,
                        metadata={
                            "kind": memory_kind,
                            "file_id": record.get("id"),
                            "file_kind": record.get("kind"),
                        },
                        infer=False,
                    )
    finally:
        if hasattr(client, "aclose"):
            with suppress(Exception):
                await client.aclose()
    return ingested
