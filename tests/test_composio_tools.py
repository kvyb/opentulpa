from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


@pytest.mark.asyncio
async def test_composio_status_hits_expected_endpoint() -> None:
    runtime = DummyRuntime([Response(200, {"ok": True})])
    tools = register_runtime_tools(runtime)

    result = await tools["composio_status"].ainvoke({})

    assert result == {"ok": True}
    assert runtime.calls[0][0] == "GET"
    assert runtime.calls[0][1] == "/internal/composio/status"


@pytest.mark.asyncio
async def test_composio_authorize_toolkit_passes_customer_scope() -> None:
    runtime = DummyRuntime([Response(200, {"redirect_url": "https://example.com/oauth"})])
    tools = register_runtime_tools(runtime)

    result = await tools["composio_authorize_toolkit"].ainvoke({"toolkit": "gmail"})

    assert result["redirect_url"] == "https://example.com/oauth"
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/composio/authorize"
    assert kwargs["json_body"]["customer_id"] == "telegram_123"
    assert kwargs["json_body"]["toolkit"] == "gmail"
