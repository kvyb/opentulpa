"""Durable customer-scoped uploaded file storage + metadata index."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

from opentulpa.core.ids import new_short_id


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    raw = str(name or "").strip()
    if not raw:
        return "file.bin"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    safe = safe.strip("._") or "file.bin"
    return safe[:180]


def _extract_docx_text(raw_bytes: bytes) -> str:
    try:
        with ZipFile(BytesIO(raw_bytes)) as zf:
            xml_bytes = zf.read("word/document.xml")
    except (BadZipFile, KeyError):
        return ""
    try:
        root = ElementTree.fromstring(xml_bytes)
    except Exception:
        return ""
    out: list[str] = []
    for node in root.iter():
        if node.tag.endswith("}t") and node.text:
            out.append(node.text)
    return " ".join(out).strip()


def _extract_pdf_text(raw_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(BytesIO(raw_bytes))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return ""


def _extract_text_preview(
    *,
    raw_bytes: bytes,
    filename: str | None,
    mime_type: str | None,
    max_chars: int = 16000,
) -> str:
    name = str(filename or "").lower()
    mime = str(mime_type or "").lower()

    text = ""
    if mime.startswith("text/") or any(
        name.endswith(ext)
        for ext in (".txt", ".md", ".csv", ".tsv", ".json", ".yaml", ".yml", ".log")
    ):
        text = raw_bytes.decode("utf-8", errors="replace")
    elif mime == "application/pdf" or name.endswith(".pdf"):
        text = _extract_pdf_text(raw_bytes)
    elif (
        mime
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        or name.endswith(".docx")
    ):
        text = _extract_docx_text(raw_bytes)

    normalized = re.sub(r"\n{3,}", "\n\n", str(text or "").strip())
    return normalized[:max_chars]


def _build_summary(
    *,
    kind: str,
    filename: str,
    mime_type: str | None,
    caption: str | None,
    text_excerpt: str,
) -> str:
    parts: list[str] = [f"{kind} file '{filename}'"]
    if mime_type:
        parts.append(f"type={mime_type}")
    if caption:
        parts.append(f"caption={caption.strip()[:280]}")
    if text_excerpt:
        head = text_excerpt[:1200].replace("\n", " ")
        parts.append(f"content_preview={head}")
    return " | ".join(parts)[:2200]


class FileVaultService:
    """Store uploaded files + searchable metadata."""

    def __init__(self, *, root_dir: Path, db_path: Path) -> None:
        self.root_dir = root_dir.resolve()
        self.db_path = db_path.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    chat_id INTEGER,
                    telegram_file_id TEXT,
                    kind TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    mime_type TEXT,
                    size_bytes INTEGER NOT NULL,
                    caption TEXT,
                    summary TEXT,
                    text_excerpt TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_uploaded_files_customer_created
                    ON uploaded_files(customer_id, created_at DESC);
                """
            )

    def ingest_file(
        self,
        *,
        customer_id: str,
        chat_id: int | None,
        kind: str,
        telegram_file_id: str | None,
        original_filename: str | None,
        mime_type: str | None,
        caption: str | None,
        raw_bytes: bytes,
    ) -> dict[str, Any]:
        cid = str(customer_id or "").strip()
        if not cid:
            raise ValueError("customer_id is required")
        fid = new_short_id("file")
        safe_name = _safe_filename(original_filename or f"{kind}.bin")
        cid_dir = self.root_dir / re.sub(r"[^A-Za-z0-9._-]+", "_", cid)
        cid_dir.mkdir(parents=True, exist_ok=True)
        stored_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{fid}_{safe_name}"
        stored_path = (cid_dir / stored_name).resolve()
        stored_path.write_bytes(raw_bytes)

        text_excerpt = _extract_text_preview(
            raw_bytes=raw_bytes,
            filename=safe_name,
            mime_type=mime_type,
        )
        summary = _build_summary(
            kind=kind,
            filename=safe_name,
            mime_type=mime_type,
            caption=caption,
            text_excerpt=text_excerpt,
        )
        created_at = _utc_now_iso()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO uploaded_files (
                    id, customer_id, chat_id, telegram_file_id, kind, original_filename,
                    stored_path, mime_type, size_bytes, caption, summary, text_excerpt, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fid,
                    cid,
                    int(chat_id) if chat_id is not None else None,
                    str(telegram_file_id or "").strip() or None,
                    str(kind or "file"),
                    safe_name,
                    str(stored_path),
                    str(mime_type).strip() if mime_type else None,
                    int(len(raw_bytes)),
                    str(caption).strip() if caption else None,
                    summary,
                    text_excerpt or None,
                    created_at,
                ),
            )
            conn.commit()
        return self.get_file(cid, fid) or {}

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "customer_id": str(row["customer_id"]),
            "chat_id": int(row["chat_id"]) if row["chat_id"] is not None else None,
            "telegram_file_id": str(row["telegram_file_id"]) if row["telegram_file_id"] else None,
            "kind": str(row["kind"]),
            "original_filename": str(row["original_filename"]),
            "stored_path": str(row["stored_path"]),
            "mime_type": str(row["mime_type"]) if row["mime_type"] else None,
            "size_bytes": int(row["size_bytes"]),
            "caption": str(row["caption"]) if row["caption"] else None,
            "summary": str(row["summary"]) if row["summary"] else "",
            "text_excerpt": str(row["text_excerpt"]) if row["text_excerpt"] else "",
            "created_at": str(row["created_at"]),
        }

    def get_file(self, customer_id: str, file_id: str) -> dict[str, Any] | None:
        cid = str(customer_id or "").strip()
        fid = str(file_id or "").strip()
        if not cid or not fid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM uploaded_files
                WHERE customer_id=? AND id=?
                """,
                (cid, fid),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_many(self, customer_id: str, file_ids: list[str]) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        ids = [str(fid).strip() for fid in file_ids if str(fid).strip()]
        if not cid or not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM uploaded_files
                WHERE customer_id=? AND id IN ({placeholders})
                ORDER BY created_at DESC
                """,
                (cid, *ids),
            ).fetchall()
        by_id = {str(r["id"]): self._row_to_dict(r) for r in rows}
        return [by_id[fid] for fid in ids if fid in by_id]

    def read_file_bytes(self, customer_id: str, file_id: str) -> bytes | None:
        record = self.get_file(customer_id, file_id)
        if not record:
            return None
        try:
            stored_path = Path(str(record.get("stored_path", ""))).resolve()
            stored_path.relative_to(self.root_dir)
        except Exception:
            return None
        try:
            return stored_path.read_bytes()
        except Exception:
            return None

    def set_ai_summary(self, customer_id: str, file_id: str, ai_summary: str) -> dict[str, Any] | None:
        cid = str(customer_id or "").strip()
        fid = str(file_id or "").strip()
        insight = str(ai_summary or "").strip()
        if not cid or not fid or not insight:
            return self.get_file(cid, fid)
        current = self.get_file(cid, fid)
        if not current:
            return None

        base = str(current.get("summary", "")).strip()
        marker = f"ai_summary={insight[:4000]}"
        if "ai_summary=" in base:
            merged = re.sub(r"ai_summary=.*$", marker, base).strip(" |")
        else:
            merged = f"{base} | {marker}".strip(" |")
        merged = merged[:8000]

        with self._conn() as conn:
            conn.execute(
                """
                UPDATE uploaded_files
                SET summary=?
                WHERE customer_id=? AND id=?
                """,
                (merged, cid, fid),
            )
            conn.commit()
        return self.get_file(cid, fid)

    def search(self, customer_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        if not cid:
            return []
        safe_limit = max(1, min(int(limit), 20))
        q = str(query or "").strip().lower()
        with self._conn() as conn:
            if not q:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM uploaded_files
                    WHERE customer_id=?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (cid, safe_limit),
                ).fetchall()
            else:
                like_q = f"%{q}%"
                rows = conn.execute(
                    """
                    SELECT *
                    FROM uploaded_files
                    WHERE customer_id=?
                      AND (
                        lower(original_filename) LIKE ?
                        OR lower(COALESCE(caption, '')) LIKE ?
                        OR lower(COALESCE(summary, '')) LIKE ?
                        OR lower(COALESCE(text_excerpt, '')) LIKE ?
                      )
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (cid, like_q, like_q, like_q, like_q, safe_limit),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]
