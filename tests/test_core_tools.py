from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


class _UpdateRuntime(DummyRuntime):
    def __init__(self) -> None:
        super().__init__([])
        self.updates: list[dict[str, str]] = []

    async def emit_interactive_update(self, *, text: str, dedupe_key: str = "") -> dict[str, bool]:
        self.updates.append({"text": text, "dedupe_key": dedupe_key})
        return {"ok": True, "sent": True}


@pytest.mark.asyncio
async def test_send_owner_update_uses_runtime_interactive_emitter() -> None:
    runtime = _UpdateRuntime()
    tools = register_runtime_tools(runtime)

    result = await tools["send_owner_update"].ainvoke(
        {"message": "  Checking the price list now.  ", "dedupe_key": "price-check"}
    )

    assert result == {"ok": True, "sent": True}
    assert runtime.updates == [
        {"text": "Checking the price list now.", "dedupe_key": "price-check"}
    ]


@pytest.mark.asyncio
async def test_send_owner_update_noops_without_interactive_emitter() -> None:
    runtime = DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["send_owner_update"].ainvoke({"message": "Still working."})

    assert result == {
        "ok": False,
        "sent": False,
        "reason": "interactive_update_unavailable",
    }


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
