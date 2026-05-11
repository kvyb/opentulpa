from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from opentulpa.integrations.browserbase import BrowserbaseClient


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
    assert requests[1][2]["userMetadata"]["opentulpaProfileId"] == "github"
