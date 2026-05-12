"""Browser Use Cloud helpers for native agent sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class BrowserUseCloudError(RuntimeError):
    """Raised when Browser Use Cloud cannot create a usable browser session."""


@dataclass(frozen=True, slots=True)
class BrowserUseCloudBrowserSession:
    id: str
    cdp_url: str
    profile_id: str
    live_url: str | None = None
    recording_url: str | None = None


@dataclass(frozen=True, slots=True)
class BrowserUseCloudAgentSession:
    id: str
    model: str | None = None
    profile_id: str | None = None
    live_url: str | None = None
    recording_urls: list[str] | None = None
    status: str | None = None
    output: Any = None
    is_task_successful: bool | None = None
    screenshot_url: str | None = None
    step_count: int | None = None
    last_step_summary: str | None = None


@dataclass(frozen=True, slots=True)
class BrowserUseCloudMessage:
    id: str
    role: str | None = None
    data: str | None = None
    summary: str | None = None
    message_type: str | None = None
    screenshot_url: str | None = None


class BrowserUseCloudClient:
    """Small async wrapper around Browser Use Cloud v3 agent APIs."""

    def __init__(
        self,
        *,
        api_key: str,
        proxy_country_code: str | None = "us",
        browser_timeout_minutes: int = 15,
        sdk_client: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.browser-use.com/api/v3",
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._proxy_country_code = str(proxy_country_code or "").strip().lower()
        self._browser_timeout_minutes = max(1, min(int(browser_timeout_minutes), 240))
        self._sdk_client = sdk_client
        self._http_client = http_client
        self._base_url = str(base_url or "").strip().rstrip("/")
        assert self._browser_timeout_minutes > 0
        assert self._base_url

    async def create_profile(self, *, name: str) -> str:
        if not self._api_key:
            raise BrowserUseCloudError("BROWSER_USE_API_KEY is required")
        profile = await self._request(
            "POST",
            "/profiles",
            json={"name": str(name or "").strip() or None},
        )
        profile_id = self._response_str(profile, "id")
        if not profile_id:
            raise BrowserUseCloudError("Browser Use profile response did not include id")
        return profile_id

    async def create_agent_session(
        self,
        *,
        task: str,
        model: str,
        profile_id: str | None = None,
        session_id: str | None = None,
        keep_alive: bool = True,
    ) -> BrowserUseCloudAgentSession:
        safe_task = str(task or "").strip()
        safe_model = str(model or "").strip()
        if not safe_task:
            raise BrowserUseCloudError("Browser Use Cloud agent session requires a task")
        if not safe_model:
            raise BrowserUseCloudError("Browser Use Cloud agent session requires a model")
        body: dict[str, Any] = {
            "task": safe_task,
            "model": safe_model,
            "keepAlive": bool(keep_alive),
            "skills": True,
            "agentmail": True,
            "cacheScript": False,
        }
        if profile_id:
            body["profileId"] = str(profile_id).strip()
        if session_id:
            body["sessionId"] = str(session_id).strip()
        if self._proxy_country_code:
            body["proxyCountryCode"] = self._proxy_country_code
        data = await self._request("POST", "/sessions", json=body)
        return self._agent_session_from_payload(data)

    async def get_agent_session(self, session_id: str) -> BrowserUseCloudAgentSession:
        safe_session_id = str(session_id or "").strip()
        if not safe_session_id:
            raise BrowserUseCloudError("Browser Use Cloud agent session id is required")
        data = await self._request("GET", f"/sessions/{safe_session_id}")
        return self._agent_session_from_payload(data)

    async def list_agent_messages(self, session_id: str, *, limit: int = 20) -> list[BrowserUseCloudMessage]:
        safe_session_id = str(session_id or "").strip()
        if not safe_session_id:
            return []
        data = await self._request(
            "GET",
            f"/sessions/{safe_session_id}/messages",
            params={"limit": max(1, min(int(limit), 100))},
        )
        raw_messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(raw_messages, list):
            return []
        out: list[BrowserUseCloudMessage] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                continue
            message_id = self._response_str(item, "id")
            if not message_id:
                continue
            out.append(
                BrowserUseCloudMessage(
                    id=message_id,
                    role=self._optional_response_str(item, "role"),
                    data=self._optional_response_str(item, "data"),
                    summary=self._optional_response_str(item, "summary"),
                    message_type=self._optional_response_str(item, "type"),
                    screenshot_url=self._optional_response_str(item, "screenshotUrl"),
                )
            )
        return out

    async def stop_agent_session(self, session_id: str, *, strategy: str = "session") -> None:
        safe_session_id = str(session_id or "").strip()
        if not safe_session_id:
            return
        safe_strategy = str(strategy or "session").strip().lower()
        if safe_strategy not in {"task", "session"}:
            safe_strategy = "session"
        await self._request(
            "POST",
            f"/sessions/{safe_session_id}/stop",
            json={"strategy": safe_strategy},
        )

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
            raise BrowserUseCloudError("Browser Use browser session response missed id or cdp_url")
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
        try:
            from browser_use_sdk import AsyncBrowserUse
        except ImportError as exc:
            raise BrowserUseCloudError(
                "browser_use_sdk is required for Browser Use Cloud sessions"
            ) from exc
        self._sdk_client = AsyncBrowserUse(api_key=self._api_key)
        return self._sdk_client

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        if not self._api_key:
            raise BrowserUseCloudError("BROWSER_USE_API_KEY is required")
        client = self._http_client
        owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(timeout=60.0)
        assert client is not None
        try:
            headers = dict(kwargs.pop("headers", {}) or {})
            headers["X-Browser-Use-API-Key"] = self._api_key
            response = await client.request(
                method,
                f"{self._base_url}{path}",
                headers=headers,
                **kwargs,
            )
            if response.status_code >= 400:
                raise BrowserUseCloudError(
                    f"Browser Use Cloud API {response.status_code}: {response.text[:1000]}"
                )
            if not response.content:
                return {}
            data = response.json()
            return data if isinstance(data, dict) else {}
        except httpx.HTTPError as exc:
            raise BrowserUseCloudError(f"Browser Use Cloud API request failed: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    @classmethod
    def _agent_session_from_payload(cls, payload: Any) -> BrowserUseCloudAgentSession:
        if not isinstance(payload, dict):
            raise BrowserUseCloudError("Browser Use Cloud session response was not an object")
        session_id = cls._response_str(payload, "id")
        if not session_id:
            raise BrowserUseCloudError("Browser Use Cloud session response did not include id")
        recording_urls = payload.get("recordingUrls")
        if not isinstance(recording_urls, list):
            recording_urls = None
        success = payload.get("isTaskSuccessful")
        return BrowserUseCloudAgentSession(
            id=session_id,
            model=cls._optional_response_str(payload, "model"),
            profile_id=cls._optional_response_str(payload, "profileId"),
            live_url=cls._optional_response_str(payload, "liveUrl"),
            recording_urls=[str(item) for item in recording_urls] if recording_urls else None,
            status=cls._optional_response_str(payload, "status"),
            output=payload.get("output"),
            is_task_successful=success if isinstance(success, bool) else None,
            screenshot_url=cls._optional_response_str(payload, "screenshotUrl"),
            step_count=payload.get("stepCount") if isinstance(payload.get("stepCount"), int) else None,
            last_step_summary=cls._optional_response_str(payload, "lastStepSummary"),
        )

    @staticmethod
    def _response_str(payload: Any, key: str) -> str:
        if isinstance(payload, dict):
            return str(payload.get(key) or "").strip()
        return str(getattr(payload, key, "") or "").strip()

    @classmethod
    def _optional_response_str(cls, payload: Any, key: str) -> str | None:
        value = cls._response_str(payload, key)
        return value or None
