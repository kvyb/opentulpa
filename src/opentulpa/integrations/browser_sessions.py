"""Tenant-owned Browser Use Cloud sessions behind registered background jobs."""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field

from opentulpa.core.ids import new_short_id
from opentulpa.jobs import (
    JobArguments,
    JobExecutionContext,
    JobHandlerRegistry,
    JobHandlerResult,
)
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.persistence.sqlite import connect_sqlite


class BrowserStartJobArguments(JobArguments):
    start_url: str | None = Field(default=None, max_length=8_192)
    allowed_domains: list[str] = Field(default_factory=list, max_length=50)


class BrowserActJobArguments(JobArguments):
    session_id: str = Field(min_length=1, max_length=300)
    action: dict[str, Any] = Field(min_length=1)


class BrowserSession(Protocol):
    async def start(self) -> None: ...

    async def navigate_to(self, url: str) -> None: ...

    async def get_current_page(self) -> Any: ...

    async def get_current_page_url(self) -> str: ...

    async def get_current_page_title(self) -> str: ...

    async def get_state_as_text(self) -> str: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BrowserSessionHandle:
    session: BrowserSession
    backend: str = "browser-use-cloud"


class BrowserSessionProvider(Protocol):
    async def create(
        self,
        *,
        tenant_id: str,
        allowed_domains: list[str],
    ) -> BrowserSessionHandle: ...


class TenantBrowserService:
    def __init__(
        self,
        *,
        db_path: Path,
        idempotency: IdempotencyStore,
        session_provider: BrowserSessionProvider | None = None,
    ) -> None:
        self._db_path = db_path.expanduser().resolve()
        self._idempotency = idempotency
        self._session_provider = session_provider
        self._sessions: dict[tuple[str, str], BrowserSession] = {}
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with connect_sqlite(self._db_path, wal=True) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS browser_sessions (
                    tenant_id TEXT NOT NULL,
                    id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    allowed_domains_json TEXT NOT NULL,
                    current_url TEXT NOT NULL,
                    backend TEXT NOT NULL DEFAULT 'browser-use-cloud',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, id)
                );
                CREATE INDEX IF NOT EXISTS idx_browser_sessions_tenant
                ON browser_sessions (tenant_id, updated_at DESC);
                """
            )
            conn.execute(
                """
                UPDATE browser_sessions SET status = 'unavailable', updated_at = ?
                WHERE status IN ('starting', 'active')
                """,
                (datetime.now(UTC).isoformat(),),
            )
            columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(browser_sessions)")
            }
            if "backend" not in columns:
                conn.execute(
                    "ALTER TABLE browser_sessions "
                    "ADD COLUMN backend TEXT NOT NULL DEFAULT 'browser-use-cloud'"
                )

    def register_handlers(self, registry: JobHandlerRegistry) -> None:
        registry.register(
            name="browser_start",
            arguments_model=BrowserStartJobArguments,
            handler=self._start_job,
            timeout_seconds=120,
        )
        registry.register(
            name="browser_act",
            arguments_model=BrowserActJobArguments,
            handler=self._act_job,
            timeout_seconds=120,
        )

    def get(self, *, tenant_id: str, session_id: str) -> dict[str, Any]:
        tenant = self._required(tenant_id, "tenant_id")
        safe_id = self._required(session_id, "session_id")
        with connect_sqlite(self._db_path, wal=True) as conn:
            row = conn.execute(
                "SELECT * FROM browser_sessions WHERE tenant_id = ? AND id = ?",
                (tenant, safe_id),
            ).fetchone()
        if row is None:
            raise KeyError(safe_id)
        return {
            "tenant_id": tenant,
            "session_id": safe_id,
            "status": str(row["status"]),
            "current_url": str(row["current_url"] or ""),
            "allowed_domains": json.loads(str(row["allowed_domains_json"] or "[]")),
            "backend": str(row["backend"] or "browser-use-cloud"),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        }

    async def stop(
        self,
        *,
        tenant_id: str,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        del idempotency_key
        existing = self.get(tenant_id=tenant_id, session_id=session_id)
        session = self._sessions.pop((tenant_id, session_id), None)
        if session is not None:
            await session.stop()
        self._update(tenant_id, session_id, status="stopped")
        return {**existing, "status": "stopped"}

    async def shutdown(self) -> None:
        sessions = list(self._sessions.items())
        self._sessions.clear()
        for (tenant_id, session_id), session in sessions:
            with suppress(Exception):
                await session.stop()
            self._update(tenant_id, session_id, status="stopped")

    async def _start_job(
        self,
        arguments: BrowserStartJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        async def start() -> dict[str, Any]:
            domains = self._domains(arguments.allowed_domains, arguments.start_url)
            session_id = new_short_id("browser", suffix_chars=12)
            now = datetime.now(UTC).isoformat()
            with connect_sqlite(self._db_path, wal=True) as conn:
                conn.execute(
                    """
                    INSERT INTO browser_sessions (
                        tenant_id, id, status, allowed_domains_json, current_url,
                        backend, created_at, updated_at
                    ) VALUES (?, ?, 'starting', ?, '', 'pending', ?, ?)
                    """,
                    (context.tenant_id, session_id, json.dumps(domains), now, now),
                )
            handle = await self._create_session(
                tenant_id=context.tenant_id,
                allowed_domains=domains,
            )
            session = handle.session
            try:
                await session.start()
                if arguments.start_url:
                    await session.navigate_to(arguments.start_url)
                current_url = await session.get_current_page_url()
            except Exception:
                with suppress(Exception):
                    await session.stop()
                self._update(context.tenant_id, session_id, status="failed")
                raise
            self._sessions[(context.tenant_id, session_id)] = session
            self._update(
                context.tenant_id,
                session_id,
                status="active",
                current_url=current_url,
                backend=handle.backend,
            )
            return {
                "tenant_id": context.tenant_id,
                "session_id": session_id,
                "status": "active",
                "current_url": current_url,
                "allowed_domains": domains,
                "backend": handle.backend,
            }

        result = await self._idempotency.execute(
            tenant_id=context.tenant_id,
            operation="browser_start",
            idempotency_key=context.idempotency_key,
            request_hash=self._hash(arguments.model_dump(mode="json")),
            invoke=start,
        )
        return JobHandlerResult(summary="Browser session started", data=dict(result or {}))

    async def _act_job(
        self,
        arguments: BrowserActJobArguments,
        context: JobExecutionContext,
    ) -> JobHandlerResult:
        existing = self.get(tenant_id=context.tenant_id, session_id=arguments.session_id)
        if existing["status"] != "active":
            raise RuntimeError("browser session is not active")
        session = self._sessions.get((context.tenant_id, arguments.session_id))
        if session is None:
            raise RuntimeError("browser session is unavailable after restart; start a new session")

        async def act() -> dict[str, Any]:
            page = await session.get_current_page()
            action = dict(arguments.action)
            kind = str(action.pop("kind", "") or action.pop("type", "")).strip().lower()
            if kind == "navigate":
                await session.navigate_to(self._required(str(action.get("url") or ""), "url"))
            elif kind == "click":
                await page.locator(self._selector(action)).click(timeout=15_000)
            elif kind == "fill":
                value = str(action.get("value") or "")[:20_000]
                await page.locator(self._selector(action)).fill(value, timeout=15_000)
            elif kind == "press":
                key = self._required(str(action.get("key") or ""), "key")[:100]
                await page.locator(self._selector(action)).press(key, timeout=15_000)
            elif kind == "select":
                value = str(action.get("value") or "")[:1_000]
                await page.locator(self._selector(action)).select_option(value, timeout=15_000)
            elif kind == "wait":
                milliseconds = max(0, min(int(action.get("milliseconds") or 1000), 10_000))
                await page.wait_for_timeout(milliseconds)
            else:
                raise ValueError("unsupported browser action")
            current_url = await session.get_current_page_url()
            title = await session.get_current_page_title()
            text = await session.get_state_as_text()
            self._update(
                context.tenant_id,
                arguments.session_id,
                status="active",
                current_url=current_url,
            )
            return {
                "tenant_id": context.tenant_id,
                "session_id": arguments.session_id,
                "status": "active",
                "current_url": current_url,
                "title": title[:1_000],
                "text": text[:20_000],
            }

        result = await self._idempotency.execute(
            tenant_id=context.tenant_id,
            operation="browser_act",
            idempotency_key=context.idempotency_key,
            request_hash=self._hash(arguments.model_dump(mode="json")),
            invoke=act,
        )
        return JobHandlerResult(summary="Browser action completed", data=dict(result or {}))

    def _update(
        self,
        tenant_id: str,
        session_id: str,
        *,
        status: Literal["active", "failed", "stopped", "unavailable"],
        current_url: str | None = None,
        backend: str | None = None,
    ) -> None:
        fields = ["status = ?", "updated_at = ?"]
        values: list[Any] = [status, datetime.now(UTC).isoformat()]
        if current_url is not None:
            fields.append("current_url = ?")
            values.append(str(current_url)[:8_192])
        if backend is not None:
            fields.append("backend = ?")
            values.append(str(backend or "browser-use-cloud")[:100])
        values.extend((tenant_id, session_id))
        with connect_sqlite(self._db_path, wal=True) as conn:
            conn.execute(
                f"UPDATE browser_sessions SET {', '.join(fields)} WHERE tenant_id = ? AND id = ?",
                values,
            )

    async def _create_session(
        self,
        *,
        tenant_id: str,
        allowed_domains: list[str],
    ) -> BrowserSessionHandle:
        if self._session_provider is None:
            raise RuntimeError(
                "no isolated browser provider is configured; set BROWSER_USE_API_KEY"
            )
        return await self._session_provider.create(
            tenant_id=tenant_id,
            allowed_domains=allowed_domains,
        )

    @staticmethod
    def _domains(values: list[str], start_url: str | None) -> list[str]:
        domains: list[str] = []
        for raw in values:
            parsed = urlsplit(raw if "://" in raw else f"//{raw}")
            domain = str(parsed.hostname or "").lower().rstrip(".")
            if domain and domain not in domains:
                domains.append(domain)
        if start_url:
            hostname = str(urlsplit(start_url).hostname or "").lower().rstrip(".")
            if hostname and hostname not in domains:
                domains.append(hostname)
        if not domains:
            raise ValueError("start_url or allowed_domains is required")
        return domains

    @staticmethod
    def _selector(action: dict[str, Any]) -> str:
        selector = str(action.get("selector") or "").strip()
        if not selector or len(selector) > 2_000:
            raise ValueError("selector is required")
        return selector

    @staticmethod
    def _required(value: str, field: str) -> str:
        safe = str(value or "").strip()
        if not safe:
            raise ValueError(f"{field} is required")
        return safe

    @staticmethod
    def _hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest()


__all__ = [
    "BrowserActJobArguments",
    "BrowserSession",
    "BrowserSessionHandle",
    "BrowserSessionProvider",
    "BrowserStartJobArguments",
    "TenantBrowserService",
]
