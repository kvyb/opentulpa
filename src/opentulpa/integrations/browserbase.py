"""Browserbase REST helpers for cloud Browser Use sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class BrowserbaseError(RuntimeError):
    """Raised when Browserbase rejects a request or returns malformed data."""


@dataclass(frozen=True, slots=True)
class BrowserbaseSession:
    id: str
    connect_url: str
    context_id: str
    recording_url: str
    live_url: str | None = None
    debugger_url: str | None = None


class BrowserbaseClient:
    """Small async Browserbase API client.

    OpenTulpa prefers Browserbase's SDK when available and keeps a REST path
    for tests and environments where the SDK cannot be imported.
    """

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str | None = None,
        base_url: str = "https://api.browserbase.com",
        timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
        sdk_client: Any | None = None,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._project_id = str(project_id or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/") or "https://api.browserbase.com"
        self._timeout_seconds = max(5.0, float(timeout_seconds))
        self._http_client = http_client
        self._sdk_client = sdk_client
        assert self._base_url.startswith(("https://", "http://"))
        assert self._timeout_seconds > 0

    async def create_context(self) -> str:
        if not self._api_key:
            raise BrowserbaseError("BROWSERBASE_API_KEY is required")
        if not self._project_id:
            raise BrowserbaseError("BROWSERBASE_PROJECT_ID is required to create a persistent context")
        sdk_client = self._get_sdk_client()
        if sdk_client is not None:
            payload = await sdk_client.contexts.create(project_id=self._project_id)
            context_id = self._response_str(payload, "id")
            if not context_id:
                raise BrowserbaseError("Browserbase create context response did not include id")
            return context_id
        payload = await self._request("POST", "/v1/contexts", json={"projectId": self._project_id})
        context_id = str(payload.get("id") or "").strip()
        if not context_id:
            raise BrowserbaseError("Browserbase create context response did not include id")
        return context_id

    async def create_session(
        self,
        *,
        context_id: str,
        customer_id: str,
        profile_id: str,
        persist: bool = True,
        keep_alive: bool = True,
        proxy_country_code: str | None = None,
        solve_captchas: bool | None = None,
        advanced_stealth: bool | None = None,
    ) -> BrowserbaseSession:
        safe_context_id = str(context_id or "").strip()
        if not safe_context_id:
            raise BrowserbaseError("Browserbase session requires a context id")
        safe_proxy_country_code = str(proxy_country_code or "").strip().lower()
        sdk_client = self._get_sdk_client()
        if sdk_client is not None:
            return await self._create_session_with_sdk(
                sdk_client=sdk_client,
                context_id=safe_context_id,
                customer_id=customer_id,
                profile_id=profile_id,
                persist=persist,
                keep_alive=keep_alive,
                proxy_country_code=safe_proxy_country_code or None,
                solve_captchas=solve_captchas,
                advanced_stealth=advanced_stealth,
            )
        body: dict[str, Any] = {
            "browserSettings": {
                "context": {"id": safe_context_id, "persist": bool(persist)},
                "keepAlive": bool(keep_alive),
            },
            "userMetadata": {
                "opentulpaCustomerId": str(customer_id or "").strip(),
                "opentulpaProfileId": str(profile_id or "").strip(),
            },
        }
        if self._project_id:
            body["projectId"] = self._project_id
        self._apply_session_options(
            body,
            sdk_style=False,
            proxy_country_code=safe_proxy_country_code or None,
            solve_captchas=solve_captchas,
            advanced_stealth=advanced_stealth,
        )
        payload = await self._request("POST", "/v1/sessions", json=body)
        session_id = str(payload.get("id") or "").strip()
        connect_url = str(payload.get("connectUrl") or payload.get("connect_url") or "").strip()
        response_context_id = str(payload.get("contextId") or "").strip() or safe_context_id
        if not session_id or not connect_url:
            raise BrowserbaseError("Browserbase create session response missed id or connectUrl")
        return BrowserbaseSession(
            id=session_id,
            connect_url=connect_url,
            context_id=response_context_id,
            recording_url=f"https://browserbase.com/sessions/{session_id}",
        )

    async def get_live_urls(self, session_id: str) -> dict[str, Any]:
        safe_session_id = str(session_id or "").strip()
        if not safe_session_id:
            raise BrowserbaseError("Browserbase live view requires a session id")
        sdk_client = self._get_sdk_client()
        if sdk_client is not None:
            payload = await sdk_client.sessions.debug(safe_session_id)
            return self._response_dict(payload)
        return await self._request("GET", f"/v1/sessions/{safe_session_id}/debug")

    async def _create_session_with_sdk(
        self,
        *,
        sdk_client: Any,
        context_id: str,
        customer_id: str,
        profile_id: str,
        persist: bool,
        keep_alive: bool,
        proxy_country_code: str | None,
        solve_captchas: bool | None,
        advanced_stealth: bool | None,
    ) -> BrowserbaseSession:
        kwargs: dict[str, Any] = {
            "browser_settings": {"context": {"id": context_id, "persist": bool(persist)}},
            "keep_alive": bool(keep_alive),
            "user_metadata": {
                "opentulpaCustomerId": str(customer_id or "").strip(),
                "opentulpaProfileId": str(profile_id or "").strip(),
            },
        }
        if self._project_id:
            kwargs["project_id"] = self._project_id
        self._apply_session_options(
            kwargs,
            sdk_style=True,
            proxy_country_code=proxy_country_code,
            solve_captchas=solve_captchas,
            advanced_stealth=advanced_stealth,
        )
        payload = await sdk_client.sessions.create(**kwargs)
        session_id = self._response_str(payload, "id")
        connect_url = self._response_str(payload, "connect_url") or self._response_str(
            payload, "connectUrl"
        )
        response_context_id = (
            self._response_str(payload, "context_id")
            or self._response_str(payload, "contextId")
            or context_id
        )
        if not session_id or not connect_url:
            raise BrowserbaseError("Browserbase create session response missed id or connectUrl")
        return BrowserbaseSession(
            id=session_id,
            connect_url=connect_url,
            context_id=response_context_id,
            recording_url=f"https://browserbase.com/sessions/{session_id}",
        )

    @staticmethod
    def _apply_session_options(
        body: dict[str, Any],
        *,
        sdk_style: bool,
        proxy_country_code: str | None,
        solve_captchas: bool | None,
        advanced_stealth: bool | None,
    ) -> None:
        settings_key = "browser_settings" if sdk_style else "browserSettings"
        browser_settings = body.get(settings_key)
        if not isinstance(browser_settings, dict):
            raise BrowserbaseError("Browserbase session browser settings must be an object")
        if solve_captchas is not None:
            browser_settings["solve_captchas" if sdk_style else "solveCaptchas"] = bool(solve_captchas)
        if advanced_stealth is not None:
            browser_settings["advanced_stealth" if sdk_style else "advancedStealth"] = bool(
                advanced_stealth
            )
        if proxy_country_code:
            body["proxies"] = [
                {
                    "type": "browserbase",
                    "geolocation": {"country": proxy_country_code},
                }
            ]

    def _get_sdk_client(self) -> Any | None:
        if self._http_client is not None:
            return None
        if self._sdk_client is not None:
            return self._sdk_client
        try:
            from browserbase import AsyncBrowserbase
        except ImportError:
            return None
        self._sdk_client = AsyncBrowserbase(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout_seconds,
        )
        return self._sdk_client

    @staticmethod
    def _response_str(payload: Any, key: str) -> str:
        if isinstance(payload, dict):
            return str(payload.get(key) or "").strip()
        return str(getattr(payload, key, "") or "").strip()

    @staticmethod
    def _response_dict(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if hasattr(payload, "to_dict"):
            data = payload.to_dict()
            if isinstance(data, dict):
                return data
        if hasattr(payload, "model_dump"):
            data = payload.model_dump(by_alias=True)
            if isinstance(data, dict):
                return data
        out = {
            key: value
            for key in dir(payload)
            if not key.startswith("_") and not callable(value := getattr(payload, key, None))
        }
        return out

    async def _request(self, method: str, path: str, *, json: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self._api_key:
            raise BrowserbaseError("BROWSERBASE_API_KEY is required")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-BB-API-Key": self._api_key,
        }
        url = f"{self._base_url}{path}"
        try:
            if self._http_client is not None:
                response = await self._http_client.request(method, url, headers=headers, json=json)
            else:
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await client.request(method, url, headers=headers, json=json)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500] if exc.response is not None else ""
            raise BrowserbaseError(
                f"Browserbase {method} {path} failed with {exc.response.status_code}: {body}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise BrowserbaseError(f"Browserbase {method} {path} failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise BrowserbaseError(f"Browserbase {method} {path} returned non-object JSON")
        return payload
