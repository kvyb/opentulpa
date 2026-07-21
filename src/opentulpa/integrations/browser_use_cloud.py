"""Browser Use Cloud helpers for hosted browser sessions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from opentulpa.integrations.browser_sessions import BrowserSessionHandle
from opentulpa.integrations.content_fetch import (
    HostResolver,
    SocketHostResolver,
    is_forbidden_address,
    is_forbidden_hostname,
)
from opentulpa.integrations.playwright_browser_session import PlaywrightBrowserSession

logger = logging.getLogger(__name__)
_PROFILE_METADATA_NAME = ".browser-use-cloud.json"


class BrowserUseCloudError(RuntimeError):
    """Raised when Browser Use Cloud cannot create a usable browser session."""


@dataclass(frozen=True, slots=True)
class BrowserUseCloudBrowserSession:
    id: str
    cdp_url: str
    profile_id: str
    live_url: str | None = None


class BrowserUseCloudClient:
    """Small async wrapper around Browser Use Cloud browser-session APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        proxy_country_code: str | None = "us",
        browser_timeout_minutes: int = 15,
        sdk_client: Any | None = None,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._proxy_country_code = str(proxy_country_code or "").strip().lower()
        self._browser_timeout_minutes = max(1, min(int(browser_timeout_minutes), 240))
        self._sdk_client = sdk_client
        assert self._browser_timeout_minutes > 0

    async def create_profile(self, *, name: str) -> str:
        safe_name = str(name or "").strip()
        if not safe_name:
            raise BrowserUseCloudError("Browser Use Cloud profile requires a name")
        profile = await self._get_sdk_client().profiles.create_profile(name=safe_name)
        profile_id = self._response_str(profile, "id")
        if not profile_id:
            raise BrowserUseCloudError("Browser Use Cloud profile response missed id")
        return profile_id

    async def create_browser_session(self, *, profile_id: str) -> BrowserUseCloudBrowserSession:
        safe_profile_id = str(profile_id or "").strip()
        if not safe_profile_id:
            raise BrowserUseCloudError("Browser Use Cloud browser session requires a profile id")
        kwargs: dict[str, Any] = {
            "profile_id": safe_profile_id,
            "timeout": self._browser_timeout_minutes,
        }
        if self._proxy_country_code:
            kwargs["proxy_country_code"] = self._proxy_country_code
        session = await self._get_sdk_client().browsers.create_browser_session(**kwargs)
        session_id = self._response_str(session, "id")
        cdp_url = self._response_str(session, "cdp_url") or self._response_str(session, "cdpUrl")
        live_url = self._response_str(session, "live_url") or self._response_str(session, "liveUrl")
        if not session_id or not cdp_url:
            raise BrowserUseCloudError("Browser Use Cloud browser session response missed id or cdp_url")
        return BrowserUseCloudBrowserSession(
            id=session_id,
            cdp_url=cdp_url,
            profile_id=safe_profile_id,
            live_url=live_url or None,
        )

    async def stop_browser_session(self, session_id: str) -> None:
        safe_session_id = str(session_id or "").strip()
        if not safe_session_id:
            return
        await self._get_sdk_client().browsers.update_browser_session(
            safe_session_id,
            action="stop",
        )

    def _get_sdk_client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        if not self._api_key:
            raise BrowserUseCloudError("BROWSER_USE_API_KEY is required")
        try:
            from browser_use_sdk import AsyncBrowserUse  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BrowserUseCloudError(
                "browser_use_sdk is required for Browser Use Cloud sessions"
            ) from exc
        self._sdk_client = AsyncBrowserUse(api_key=self._api_key)
        return self._sdk_client

    @staticmethod
    def _response_str(payload: Any, key: str) -> str:
        if isinstance(payload, dict):
            return str(payload.get(key) or "").strip()
        return str(getattr(payload, key, "") or "").strip()


class _ManagedBrowserUseSession:
    """Close the Browser Use remote session with its local CDP client."""

    def __init__(self, *, session: Any, client: BrowserUseCloudClient, remote_id: str) -> None:
        self._session = session
        self._client = client
        self._remote_id = remote_id
        self._stopped = False

    async def start(self) -> None:
        await self._session.start()

    async def navigate_to(self, url: str) -> None:
        await self._session.navigate_to(url)

    async def get_current_page(self) -> Any:
        return await self._session.get_current_page()

    async def get_current_page_url(self) -> str:
        return str(await self._session.get_current_page_url())

    async def get_current_page_title(self) -> str:
        return str(await self._session.get_current_page_title())

    async def get_state_as_text(self) -> str:
        return str(await self._session.get_state_as_text())

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            await self._session.stop()
        finally:
            with suppress(Exception):
                await self._client.stop_browser_session(self._remote_id)


class BrowserUseCloudSessionProvider:
    """Open tenant-profiled Browser Use Cloud sessions without a host-network fallback."""

    def __init__(
        self,
        *,
        client: BrowserUseCloudClient,
        profile_metadata_root: Path,
        session_factory: Callable[..., Any] = PlaywrightBrowserSession,
        host_resolver: HostResolver | None = None,
    ) -> None:
        self._client = client
        self._profile_metadata_root = profile_metadata_root.expanduser().resolve()
        self._session_factory = session_factory
        self._host_resolver = host_resolver or SocketHostResolver()
        self._profile_locks: dict[str, asyncio.Lock] = {}
        self._profile_metadata_root.mkdir(parents=True, exist_ok=True)

    async def create(
        self,
        *,
        tenant_id: str,
        allowed_domains: list[str],
    ) -> BrowserSessionHandle:
        if not allowed_domains or any(
            str(domain or "").strip() in {"", "*"} for domain in allowed_domains
        ):
            raise BrowserUseCloudError(
                "Browser Use Cloud requires explicit allowed_domains"
            )
        remote: BrowserUseCloudBrowserSession | None = None
        try:
            profile_id = await self._get_or_create_profile_id(tenant_id)
            remote = await self._client.create_browser_session(profile_id=profile_id)
            await self._validate_public_url(remote.cdp_url, schemes={"https", "wss"})
            local = self._session_factory(
                cdp_url=remote.cdp_url,
                allowed_domains=allowed_domains,
            )
            return BrowserSessionHandle(
                session=_ManagedBrowserUseSession(
                    session=local,
                    client=self._client,
                    remote_id=remote.id,
                ),
                backend="browser-use-cloud",
            )
        except Exception as exc:
            if remote is not None:
                with suppress(Exception):
                    await self._client.stop_browser_session(remote.id)
            logger.warning(
                "Browser Use Cloud session unavailable: exception=%s",
                type(exc).__name__,
            )
            if isinstance(exc, BrowserUseCloudError):
                raise
            raise BrowserUseCloudError(
                "Browser Use Cloud session could not be created safely"
            ) from exc

    async def _get_or_create_profile_id(self, tenant_id: str) -> str:
        tenant_key = self._tenant_key(tenant_id)
        lock = self._profile_locks.setdefault(tenant_key, asyncio.Lock())
        async with lock:
            metadata_path = self._profile_metadata_path(tenant_key)
            profile_id = self._load_profile_id(metadata_path)
            if profile_id:
                return profile_id
            profile_id = await self._client.create_profile(name=f"opentulpa-{tenant_key}")
            self._write_profile_id(metadata_path, profile_id)
            return profile_id

    async def _validate_public_url(self, value: str, *, schemes: set[str]) -> None:
        candidate = str(value or "").strip()
        try:
            parsed = urlsplit(candidate)
            port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
        except ValueError as exc:
            raise BrowserUseCloudError("Browser Use Cloud returned an invalid endpoint") from exc
        hostname = str(parsed.hostname or "").lower().rstrip(".")
        if (
            parsed.scheme not in schemes
            or not hostname
            or parsed.username is not None
            or parsed.password is not None
            or is_forbidden_hostname(hostname)
        ):
            raise BrowserUseCloudError("Browser Use Cloud returned an unsafe endpoint")
        try:
            addresses: Sequence[str] = await self._host_resolver.resolve(hostname, port)
        except Exception as exc:
            raise BrowserUseCloudError("Browser Use Cloud endpoint could not be resolved") from exc
        normalized = [str(address or "").strip() for address in addresses]
        if not normalized or any(is_forbidden_address(address) for address in normalized):
            raise BrowserUseCloudError("Browser Use Cloud returned an unsafe endpoint")

    @staticmethod
    def _tenant_key(tenant_id: str) -> str:
        safe_tenant_id = str(tenant_id or "").strip()
        if not safe_tenant_id:
            raise BrowserUseCloudError("Browser Use Cloud profile requires a tenant")
        return hashlib.sha256(safe_tenant_id.encode("utf-8")).hexdigest()[:24]

    def _profile_metadata_path(self, tenant_key: str) -> Path:
        return self._profile_metadata_root / tenant_key / _PROFILE_METADATA_NAME

    @staticmethod
    def _load_profile_id(metadata_path: Path) -> str:
        if not metadata_path.is_file():
            return ""
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return ""
        if not isinstance(payload, dict):
            return ""
        return str(payload.get("profile_id") or "").strip()[:500]

    @staticmethod
    def _write_profile_id(metadata_path: Path, profile_id: str) -> None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"profile_id": str(profile_id)}, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(metadata_path)


__all__ = [
    "BrowserUseCloudBrowserSession",
    "BrowserUseCloudClient",
    "BrowserUseCloudError",
    "BrowserUseCloudSessionProvider",
]
