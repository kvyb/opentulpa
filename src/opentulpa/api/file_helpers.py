"""Helpers for uploaded files, web-image fetch, and routine cleanup metadata."""

from __future__ import annotations

import mimetypes
import re
from typing import Any
from urllib.parse import unquote, urlparse

import httpx


def sanitize_uploaded_file_record(
    record: dict[str, Any],
    *,
    include_excerpt: bool = False,
    max_excerpt_chars: int = 16000,
) -> dict[str, Any]:
    """Return safe uploaded-file metadata for API responses."""
    local_path = str(record.get("local_path", "")).strip() or None
    clean = {
        "id": record.get("id"),
        "customer_id": record.get("customer_id"),
        "chat_id": record.get("chat_id"),
        "telegram_file_id": record.get("telegram_file_id"),
        "kind": record.get("kind"),
        "original_filename": record.get("original_filename"),
        "mime_type": record.get("mime_type"),
        "size_bytes": record.get("size_bytes"),
        "caption": record.get("caption"),
        "summary": record.get("summary"),
        "created_at": record.get("created_at"),
        "vault_path": record.get("stored_path"),
        "local_path": local_path,
    }
    if include_excerpt:
        excerpt = str(record.get("text_excerpt", "") or "")
        clean["text_excerpt"] = excerpt[:max_excerpt_chars]
    return clean


def normalize_cleanup_paths(value: Any) -> list[str]:
    """Normalize an untrusted list of cleanup paths into de-duplicated strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def collect_routine_cleanup_paths(payload: dict[str, Any]) -> list[str]:
    """Collect cleanup file paths from supported routine payload keys."""
    if not isinstance(payload, dict):
        return []
    candidates: list[str] = []
    list_keys = ("cleanup_paths", "script_paths", "file_paths")
    scalar_keys = ("cleanup_path", "script_path", "file_path")
    for key in list_keys:
        candidates.extend(normalize_cleanup_paths(payload.get(key)))
    for key in scalar_keys:
        raw = str(payload.get(key, "")).strip()
        if raw:
            candidates.append(raw)
    seen: set[str] = set()
    out: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def safe_telegram_filename(name: str, *, fallback: str = "image.jpg") -> str:
    """Build a filesystem-safe Telegram filename."""
    raw = str(name or "").strip()
    if not raw:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return (safe or fallback)[:180]


def infer_image_filename(image_url: str, content_type: str) -> str:
    """Infer a stable filename for downloaded web images."""
    parsed = urlparse(image_url)
    candidate = unquote(str(parsed.path or "").split("/")[-1]).strip()
    safe = safe_telegram_filename(candidate, fallback="")
    if safe and "." in safe:
        return safe
    ext = mimetypes.guess_extension(str(content_type or "").strip().lower()) or ".jpg"
    return safe_telegram_filename(f"image{ext}")


async def download_image_from_web_url(
    image_url: str,
    *,
    max_bytes: int = 10_000_000,
) -> dict[str, Any]:
    """Download a web image after validating URL scheme, type, and size."""
    raw_url = str(image_url or "").strip()
    parsed = urlparse(raw_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must start with http:// or https://")

    safe_limit = max(250_000, min(int(max_bytes), 25_000_000))
    timeout = httpx.Timeout(45.0, connect=10.0, read=45.0)
    headers = {"User-Agent": "OpenTulpa/0.1 (+send-web-image)"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        try:
            head = await client.head(raw_url)
            if head.status_code < 400:
                head_type = str(head.headers.get("content-type", "")).split(";")[0].strip().lower()
                if head_type and not head_type.startswith("image/"):
                    raise ValueError(f"url does not point to an image (content-type={head_type})")
                head_len = str(head.headers.get("content-length", "")).strip()
                if head_len.isdigit() and int(head_len) > safe_limit:
                    raise ValueError(f"image too large ({head_len} bytes > {safe_limit} bytes)")
        except ValueError:
            raise
        except Exception:
            # Some origins reject HEAD; proceed with GET validation.
            pass

        async with client.stream("GET", raw_url) as resp:
            if resp.status_code >= 400:
                raise ValueError(f"image fetch failed: HTTP {resp.status_code}")
            ctype = str(resp.headers.get("content-type", "")).split(";")[0].strip().lower()
            if not ctype.startswith("image/"):
                raise ValueError(f"url does not point to an image (content-type={ctype or 'unknown'})")
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                total += len(chunk)
                if total > safe_limit:
                    raise ValueError(f"image too large (>{safe_limit} bytes)")
                chunks.append(chunk)
            raw_bytes = b"".join(chunks)
            if not raw_bytes:
                raise ValueError("image fetch returned empty body")
            final_url = str(resp.url)

    filename = infer_image_filename(final_url, ctype)
    return {
        "raw_bytes": raw_bytes,
        "content_type": ctype,
        "filename": filename,
        "final_url": final_url,
        "size_bytes": len(raw_bytes),
    }
