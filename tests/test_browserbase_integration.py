from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from opentulpa.integrations.browserbase import BrowserbaseClient


class _FakeSdkContexts:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": "ctx_sdk"}


class _FakeSdkSessions:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.debugged: list[str] = []

    async def create(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {
            "id": "ses_sdk",
            "connect_url": "wss://connect.browserbase.sdk/session",
            "context_id": "ctx_sdk",
        }

    async def debug(self, session_id: str) -> dict[str, Any]:
        self.debugged.append(session_id)
        return {"debuggerFullscreenUrl": "https://browserbase.com/live/ses_sdk"}


class _FakeSdkClient:
    def __init__(self) -> None:
        self.contexts = _FakeSdkContexts()
        self.sessions = _FakeSdkSessions()


@pytest.mark.asyncio
async def test_browserbase_client_creates_context_session_and_live_url() -> None:
    requests: list[tuple[str, str, dict[str, Any] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        json_body = None
        if request.content:
            json_body = json.loads(request.content)
        requests.append((request.method, request.url.path, json_body))
        if request.url.path == "/v1/contexts":
            return httpx.Response(200, json={"id": "ctx_123"})
        if request.url.path == "/v1/sessions":
            return httpx.Response(
                200,
                json={
                    "id": "ses_123",
                    "connectUrl": "wss://connect.browserbase.com/session",
                    "contextId": "ctx_123",
                },
            )
        if request.url.path == "/v1/sessions/ses_123/debug":
            return httpx.Response(
                200,
                json={
                    "debuggerFullscreenUrl": "https://browserbase.com/live/ses_123",
                    "debuggerUrl": "https://browserbase.com/debug/ses_123",
                },
            )
        return httpx.Response(404, json={"error": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = BrowserbaseClient(
            api_key="bb_test",
            project_id="proj_123",
            base_url="https://api.browserbase.test",
            http_client=http_client,
        )
        context_id = await client.create_context()
        session = await client.create_session(
            context_id=context_id,
            customer_id="cust_1",
            profile_id="github",
            proxy_country_code="US",
            solve_captchas=True,
            advanced_stealth=True,
        )
        live = await client.get_live_urls(session.id)

    assert context_id == "ctx_123"
    assert session.id == "ses_123"
    assert session.connect_url == "wss://connect.browserbase.com/session"
    assert live["debuggerFullscreenUrl"] == "https://browserbase.com/live/ses_123"
    assert requests[0] == ("POST", "/v1/contexts", {"projectId": "proj_123"})
    assert requests[1][2]["browserSettings"]["context"] == {
        "id": "ctx_123",
        "persist": True,
    }
    assert requests[1][2]["browserSettings"]["solveCaptchas"] is True
    assert requests[1][2]["browserSettings"]["advancedStealth"] is True
    assert requests[1][2]["proxies"] == [
        {"type": "browserbase", "geolocation": {"country": "us"}}
    ]
    assert requests[1][2]["userMetadata"]["opentulpaProfileId"] == "github"


@pytest.mark.asyncio
async def test_browserbase_client_uses_sdk_client_when_available() -> None:
    sdk_client = _FakeSdkClient()
    client = BrowserbaseClient(
        api_key="bb_test",
        project_id="proj_123",
        base_url="https://api.browserbase.test",
        sdk_client=sdk_client,
    )

    context_id = await client.create_context()
    session = await client.create_session(
        context_id=context_id,
        customer_id="cust_1",
        profile_id="github",
        proxy_country_code="US",
        solve_captchas=True,
        advanced_stealth=True,
    )
    live = await client.get_live_urls(session.id)

    assert context_id == "ctx_sdk"
    assert session.id == "ses_sdk"
    assert session.connect_url == "wss://connect.browserbase.sdk/session"
    assert session.context_id == "ctx_sdk"
    assert live["debuggerFullscreenUrl"] == "https://browserbase.com/live/ses_sdk"
    assert sdk_client.contexts.created == [{"project_id": "proj_123"}]
    assert sdk_client.sessions.created[0]["project_id"] == "proj_123"
    assert sdk_client.sessions.created[0]["browser_settings"]["context"] == {
        "id": "ctx_sdk",
        "persist": True,
    }
    assert sdk_client.sessions.created[0]["browser_settings"]["solve_captchas"] is True
    assert sdk_client.sessions.created[0]["browser_settings"]["advanced_stealth"] is True
    assert sdk_client.sessions.created[0]["proxies"] == [
        {"type": "browserbase", "geolocation": {"country": "us"}}
    ]
    assert sdk_client.sessions.created[0]["keep_alive"] is True
    assert sdk_client.sessions.created[0]["user_metadata"]["opentulpaProfileId"] == "github"
    assert sdk_client.sessions.debugged == ["ses_sdk"]
