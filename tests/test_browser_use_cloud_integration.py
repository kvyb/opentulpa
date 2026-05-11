from __future__ import annotations

from typing import Any

import pytest

from opentulpa.integrations.browser_use_cloud import BrowserUseCloudClient


class _FakeProfiles:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    async def create_profile(self, **kwargs: Any) -> dict[str, Any]:
        self.created.append(kwargs)
        return {"id": "prof_123"}


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
        self.profiles = _FakeProfiles()
        self.browsers = _FakeBrowsers()


@pytest.mark.asyncio
async def test_browser_use_cloud_client_creates_profile_and_browser_session() -> None:
    sdk_client = _FakeSdkClient()
    client = BrowserUseCloudClient(
        api_key="bu-key",
        proxy_country_code="US",
        browser_timeout_minutes=120,
        sdk_client=sdk_client,
    )

    profile_id = await client.create_profile(name="opentulpa-customer-default")
    session = await client.create_browser_session(profile_id=profile_id)

    assert profile_id == "prof_123"
    assert session.id == "browser_123"
    assert session.profile_id == "prof_123"
    assert session.cdp_url == "wss://connect.browser-use.test/session"
    assert session.live_url == "https://browser-use.test/live/browser_123"
    assert sdk_client.profiles.created == [{"name": "opentulpa-customer-default"}]
    assert sdk_client.browsers.created == [
        {"profile_id": "prof_123", "timeout": 120, "proxy_country_code": "us"}
    ]

    await client.stop_browser_session(session.id)

    assert sdk_client.browsers.updated == [{"session_id": "browser_123", "action": "stop"}]
