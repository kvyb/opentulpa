from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


@pytest.mark.asyncio
async def test_memory_search_passes_customer_scope() -> None:
    runtime = DummyRuntime([Response(200, {"results": [{"id": "mem_1"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["memory_search"].ainvoke({"query": "car wash"})

    assert result == [{"id": "mem_1"}]
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/memory/search"
    assert kwargs["json_body"]["user_id"] == "telegram_123"


@pytest.mark.asyncio
async def test_server_time_returns_expected_keys() -> None:
    runtime = DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["server_time"].ainvoke({})

    assert "server_time_local_iso" in result
    assert "server_time_utc_iso" in result
    assert "unix_timestamp" in result
