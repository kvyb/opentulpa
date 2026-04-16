from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


@pytest.mark.asyncio
async def test_skill_list_passes_customer_scope() -> None:
    runtime = DummyRuntime([Response(200, {"skills": [{"name": "test-skill"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["skill_list"].ainvoke({})

    assert result == [{"name": "test-skill"}]
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/skills/list"
    assert kwargs["json_body"]["customer_id"] == "telegram_123"


@pytest.mark.asyncio
async def test_skill_get_uses_name_lookup() -> None:
    runtime = DummyRuntime([Response(200, {"skill": {"name": "sales"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["skill_get"].ainvoke({"name": "sales"})

    assert result == {"name": "sales"}
    assert runtime.calls[0][2]["json_body"]["name"] == "sales"
