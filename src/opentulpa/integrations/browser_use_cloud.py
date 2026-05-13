"""Browser Use Cloud helpers for hosted browser sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
            from browser_use_sdk import AsyncBrowserUse
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
