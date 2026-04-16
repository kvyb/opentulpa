from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


@pytest.mark.asyncio
async def test_routine_list_passes_customer_scope() -> None:
    runtime = DummyRuntime([Response(200, {"routines": [{"id": "rtn_abc"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["routine_list"].ainvoke({})
    assert result == [{"id": "rtn_abc"}]
    assert runtime.calls[0][0] == "GET"
    assert runtime.calls[0][1] == "/internal/scheduler/routines"
    assert runtime.calls[0][2].get("params") == {"customer_id": "telegram_123"}


@pytest.mark.asyncio
async def test_routine_delete_verifies_removed() -> None:
    runtime = DummyRuntime(
        [
            Response(200, {"ok": True}),
            Response(200, {"routines": []}),
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["routine_delete"].ainvoke({"routine_id": "rtn_deadbeef"})
    assert result["ok"] is True
    assert result["verified_removed"] is True


@pytest.mark.asyncio
async def test_customer_scoped_tool_fails_closed_without_customer_context() -> None:
    runtime = DummyRuntime([Response(200, {"routines": []})], customer_id="")
    tools = register_runtime_tools(runtime)
    with pytest.raises(RuntimeError, match="customer_id is missing"):
        await tools["routine_list"].ainvoke({})
