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


@pytest.mark.asyncio
async def test_composio_execute_preflights_tool_slug_and_returns_candidates() -> None:
    runtime = DummyRuntime(
        [
            Response(404, {"error": {"message": "Tool GOOGLESHEETS_GET_ROWS not found"}}),
            Response(200, {"items": [{"slug": "GOOGLESHEETS_VALUES_GET"}]}),
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["composio_tool_execute"].ainvoke(
        {"tool_slug": "GOOGLESHEETS_GET_ROWS", "arguments": {}}
    )

    assert result["ok"] is False
    assert result["retryable"] is False
    assert result["candidate_tools"] == [{"slug": "GOOGLESHEETS_VALUES_GET"}]
    assert [call[1] for call in runtime.calls] == [
        "/internal/composio/tools/GOOGLESHEETS_GET_ROWS/schema",
        "/internal/composio/tools/search",
    ]


@pytest.mark.asyncio
async def test_composio_execute_preflight_transient_error_is_retryable() -> None:
    runtime = DummyRuntime([Response(503, {"error": {"message": "Bad Gateway"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["composio_tool_execute"].ainvoke(
        {"tool_slug": "GOOGLESHEETS_GET_ROWS", "arguments": {}}
    )

    assert result["ok"] is False
    assert result["retryable"] is True
    assert result["candidate_tools"] == []
    assert result["suggested_next_action"] == "Retry the same call later."
    assert [call[1] for call in runtime.calls] == [
        "/internal/composio/tools/GOOGLESHEETS_GET_ROWS/schema"
    ]


@pytest.mark.asyncio
async def test_composio_execute_returns_structured_nonretryable_account_error() -> None:
    runtime = DummyRuntime(
        [
            Response(200, {"tool": {"slug": "GOOGLESHEETS_VALUES_GET"}}),
            Response(400, {"error": {"slug": "ActionExecute_ConnectedAccountNotFound"}}),
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["composio_tool_execute"].ainvoke(
        {"tool_slug": "GOOGLESHEETS_VALUES_GET", "arguments": {"range": "A1:B2"}}
    )

    assert result["ok"] is False
    assert result["retryable"] is False
    assert "authorize" in result["suggested_next_action"]
    assert runtime.calls[-1][1] == "/internal/composio/tools/execute"
    assert runtime.calls[-1][2]["retries"] == 0
