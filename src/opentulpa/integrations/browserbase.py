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

    The SDK is intentionally not required here; OpenTulpa only needs contexts,
    sessions, and live-view URLs.
    """

    def __init__(
        self,
        *,
        api_key: str,
        project_id: str | None = None,
        base_url: str = "https://api.browserbase.com",
        timeout_seconds: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._project_id = str(project_id or "").strip()
        self._base_url = str(base_url or "").strip().rstrip("/") or "https://api.browserbase.com"
        self._timeout_seconds = max(5.0, float(timeout_seconds))
        self._http_client = http_client
        assert self._base_url.startswith(("https://", "http://"))
        assert self._timeout_seconds > 0

    async def create_context(self) -> str:
        if not self._api_key:
            raise BrowserbaseError("BROWSERBASE_API_KEY is required")
        if not self._project_id:
            raise BrowserbaseError("BROWSERBASE_PROJECT_ID is required to create a persistent context")
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
    ) -> BrowserbaseSession:
        safe_context_id = str(context_id or "").strip()
        if not safe_context_id:
            raise BrowserbaseError("Browserbase session requires a context id")
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
        return await self._request("GET", f"/v1/sessions/{safe_session_id}/debug")

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
