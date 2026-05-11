from __future__ import annotations

from typing import Any

import httpx
import pytest

from opentulpa.integrations.browser_use_cloud import BrowserUseCloudClient


class _FakeBrowsers:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    async def create_browser_session(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {
            "id": "browser_123",
            "cdp_url": "wss://connect.browser-use.test/session",
            "live_url": "https://browser-use.test/live/browser_123",
        }

    async def update_browser_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        self.updated.append({"session_id": session_id, **kwargs})
        return {"id": session_id, "status": "stopped"}


class _FakeSdkClient:
    def __init__(self) -> None:
        self.browsers = _FakeBrowsers()


@pytest.mark.asyncio
async def test_browser_use_cloud_client_creates_profile_and_agent_session() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read().decode("utf-8")
        requests.append({"method": request.method, "path": request.url.path, "body": body})
        if request.url.path == "/api/v3/profiles":
            return httpx.Response(201, json={"id": "prof_123"})
        if request.url.path == "/api/v3/sessions":
            return httpx.Response(
                200,
                json={
                    "id": "sess_123",
                    "status": "running",
                    "profileId": "prof_123",
                    "liveUrl": "https://live.browser-use.test/sess_123",
                    "recordingUrls": [],
                    "isTaskSuccessful": None,
                },
            )
        if request.url.path == "/api/v3/sessions/sess_123":
            return httpx.Response(
                200,
                json={
                    "id": "sess_123",
                    "status": "idle",
                    "profileId": "prof_123",
                    "output": "done",
                    "isTaskSuccessful": True,
                },
            )
        if request.url.path == "/api/v3/sessions/sess_123/messages":
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "id": "msg_1",
                            "role": "assistant",
                            "summary": "Opened the page",
                            "type": "browser",
                        }
                    ],
                    "hasMore": False,
                },
            )
        if request.url.path == "/api/v3/sessions/sess_123/stop":
            return httpx.Response(200, json={"id": "sess_123", "status": "stopped"})
        return httpx.Response(404, json={"error": "not found"})

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="https://api.browser-use.com")
    client = BrowserUseCloudClient(
        api_key="bu-key",
        proxy_country_code="US",
        browser_timeout_minutes=120,
        http_client=http_client,
    )

    profile_id = await client.create_profile(name="opentulpa-customer-default")
    session = await client.create_agent_session(
        task="open reddit",
        model="gemini-3-flash",
        profile_id=profile_id,
    )
    refreshed = await client.get_agent_session(session.id)
    messages = await client.list_agent_messages(session.id)
    await client.stop_agent_session(session.id)
    await http_client.aclose()

    assert profile_id == "prof_123"
    assert session.id == "sess_123"
    assert session.profile_id == "prof_123"
    assert session.live_url == "https://live.browser-use.test/sess_123"
    assert refreshed.status == "idle"
    assert refreshed.output == "done"
    assert refreshed.is_task_successful is True
    assert messages[0].summary == "Opened the page"
    assert requests[0] == {
        "method": "POST",
        "path": "/api/v3/profiles",
        "body": '{"name":"opentulpa-customer-default"}',
    }
    assert '"model":"gemini-3-flash"' in requests[1]["body"]
    assert '"keepAlive":true' in requests[1]["body"]
    assert '"proxyCountryCode":"us"' in requests[1]["body"]


@pytest.mark.asyncio
async def test_browser_use_cloud_client_keeps_legacy_browser_session_wrapper() -> None:
    sdk_client = _FakeSdkClient()
    client = BrowserUseCloudClient(
        api_key="bu-key",
        proxy_country_code="US",
        browser_timeout_minutes=120,
        sdk_client=sdk_client,
    )

    session = await client.create_browser_session(profile_id="prof_123")

    assert session.id == "browser_123"
    assert session.profile_id == "prof_123"
    assert session.cdp_url == "wss://connect.browser-use.test/session"
    assert session.live_url == "https://browser-use.test/live/browser_123"
    assert sdk_client.browsers.created == [
        {"profile_id": "prof_123", "timeout": 120, "proxy_country_code": "us"}
    ]

    await client.stop_browser_session(session.id)

    assert sdk_client.browsers.updated == [{"session_id": "browser_123", "action": "stop"}]
