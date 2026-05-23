"""Durable customer-scoped short aliases for long URLs."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from opentulpa.core.ids import new_short_id
from opentulpa.persistence.sqlite import connect_sqlite

_HTTP_URL_RE = re.compile(r"https?://[^\s<>'\"`]+", re.IGNORECASE)
_LINK_ID_RE = re.compile(r"\blink_[A-Za-z0-9]{4,12}\b")
_TRAILING_PUNCT = ".,;:!?\"'`]>}"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _trim_url_candidate(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    while text and text[-1] in _TRAILING_PUNCT:
        text = text[:-1]
    while text.endswith(")") and text.count(")") > text.count("("):
        text = text[:-1]
    return text


def _normalize_http_url(value: str) -> str | None:
    raw = _trim_url_candidate(value)
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    scheme = str(parsed.scheme or "").lower()
    host = str(parsed.hostname or "").strip().lower()
    if scheme not in {"http", "https"} or not host:
        return None
    if "." not in host and host != "localhost":
        return None
    if not re.fullmatch(r"[a-z0-9.-]{1,253}", host):
        return None
    port = parsed.port
    userinfo = ""
    if parsed.username:
        userinfo += parsed.username
    if parsed.password:
        userinfo += f":{parsed.password}"
    if userinfo:
        userinfo += "@"
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{userinfo}{host}:{port}"
    else:
        netloc = f"{userinfo}{host}"
    normalized = urlunparse(
        (
            scheme,
            netloc,
            parsed.path or "",
            parsed.params or "",
            parsed.query or "",
            parsed.fragment or "",
        )
    )
    return normalized


class LinkAliasService:
    """Persist and resolve compact `link_*` aliases for URLs."""

    def __init__(self, *, db_path: Path) -> None:
        self.db_path = db_path.resolve()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        return connect_sqlite(self.db_path, wal=True)

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS link_aliases (
                    id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    seen_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_link_aliases_customer_url
                    ON link_aliases(customer_id, url);
                CREATE INDEX IF NOT EXISTS idx_link_aliases_customer_last_seen
                    ON link_aliases(customer_id, last_seen_at DESC);
                """
            )

    @staticmethod
    def extract_urls(text: str, *, limit: int = 40) -> list[str]:
        raw = str(text or "")
        if not raw:
            return []
        out: list[str] = []
        seen: set[str] = set()
        max_items = max(1, min(int(limit), 200))
        for match in _HTTP_URL_RE.finditer(raw):
            candidate = _normalize_http_url(match.group(0))
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            out.append(candidate)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def extract_link_ids(text: str, *, limit: int = 40) -> list[str]:
        raw = str(text or "")
        if not raw:
            return []
        out: list[str] = []
        seen: set[str] = set()
        max_items = max(1, min(int(limit), 200))
        for match in _LINK_ID_RE.finditer(raw):
            link_id = match.group(0).lower()
            if link_id in seen:
                continue
            seen.add(link_id)
            out.append(link_id)
            if len(out) >= max_items:
                break
        return out

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": str(row["id"]),
            "customer_id": str(row["customer_id"]),
            "url": str(row["url"]),
            "source": str(row["source"]),
            "seen_count": int(row["seen_count"] or 1),
            "created_at": str(row["created_at"]),
            "last_seen_at": str(row["last_seen_at"]),
        }

    def register_link(self, customer_id: str, url: str, *, source: str = "unknown") -> dict[str, Any] | None:
        cid = str(customer_id or "").strip()
        if not cid:
            return None
        normalized = _normalize_http_url(url)
        if not normalized:
            return None
        now = _utc_now_iso()
        with self._conn() as conn:
            existing = conn.execute(
                """
                SELECT *
                FROM link_aliases
                WHERE customer_id=? AND url=?
                """,
                (cid, normalized),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE link_aliases
                    SET source=?, seen_count=seen_count+1, last_seen_at=?
                    WHERE id=?
                    """,
                    (str(source or "unknown")[:64], now, str(existing["id"])),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM link_aliases WHERE id=?", (str(existing["id"]),)).fetchone()
                return self._row_to_dict(row) if row else None

            for _ in range(8):
                link_id = new_short_id("link")
                clash = conn.execute(
                    "SELECT 1 FROM link_aliases WHERE id=?",
                    (link_id,),
                ).fetchone()
                if clash:
                    continue
                conn.execute(
                    """
                    INSERT INTO link_aliases (
                        id, customer_id, url, source, seen_count, created_at, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        link_id,
                        cid,
                        normalized,
                        str(source or "unknown")[:64],
                        1,
                        now,
                        now,
                    ),
                )
                conn.commit()
                row = conn.execute("SELECT * FROM link_aliases WHERE id=?", (link_id,)).fetchone()
                return self._row_to_dict(row) if row else None
        return None

    def register_links_from_text(
        self,
        customer_id: str,
        text: str,
        *,
        source: str = "text",
        limit: int = 40,
    ) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        if not cid:
            return []
        raw = str(text or "")
        if not raw:
            return []
        urls = self.extract_urls(raw[:200000], limit=limit)
        out: list[dict[str, Any]] = []
        for url in urls:
            row = self.register_link(cid, url, source=source)
            if row:
                out.append(row)
        return out

    def get_by_id(self, customer_id: str, link_id: str) -> dict[str, Any] | None:
        cid = str(customer_id or "").strip()
        lid = str(link_id or "").strip().lower()
        if not cid or not lid:
            return None
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM link_aliases
                WHERE customer_id=? AND id=?
                """,
                (cid, lid),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def resolve_links(self, customer_id: str, link_ids: list[str]) -> dict[str, str]:
        cid = str(customer_id or "").strip()
        if not cid:
            return {}
        ids = [str(item or "").strip().lower() for item in link_ids]
        ids = [item for item in ids if item]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._conn() as conn:
            rows = conn.execute(
                f"""
                SELECT id, url
                FROM link_aliases
                WHERE customer_id=? AND id IN ({placeholders})
                """,
                (cid, *ids),
            ).fetchall()
        return {str(row["id"]).lower(): str(row["url"]) for row in rows}

    def expand_link_ids_in_text(
        self,
        customer_id: str,
        text: str,
        *,
        max_replacements: int = 40,
    ) -> str:
        raw = str(text or "")
        if not raw:
            return raw
        ids = self.extract_link_ids(raw, limit=max_replacements)
        if not ids:
            return raw
        mapping = self.resolve_links(customer_id, ids)
        if not mapping:
            return raw

        def _replace(match: re.Match[str]) -> str:
            link_id = match.group(0).lower()
            return mapping.get(link_id, match.group(0))

        return _LINK_ID_RE.sub(_replace, raw)

    def list_recent(self, customer_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
        cid = str(customer_id or "").strip()
        if not cid:
            return []
        safe_limit = max(1, min(int(limit), 200))
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM link_aliases
                WHERE customer_id=?
                ORDER BY last_seen_at DESC
                LIMIT ?
                """,
                (cid, safe_limit),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]
