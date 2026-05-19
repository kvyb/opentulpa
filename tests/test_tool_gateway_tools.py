from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain.tools import tool

from opentulpa.agent.tools.tool_gateway_tools import register_tool_gateway_tools


@pytest.mark.asyncio
async def test_tool_group_exec_keeps_single_call_shape() -> None:
    @tool
    async def server_time() -> dict[str, Any]:
        """Return server time."""
        return {"now": "2026-05-15T00:00:00Z"}

    tools = register_tool_gateway_tools(None, {"server_time": server_time})

    result = await tools["tool_group_exec"].ainvoke({"group": "memory", "command": "server_time"})

    assert result == {
        "group": "memory",
        "command": "server_time",
        "ok": True,
        "result": {"now": "2026-05-15T00:00:00Z"},
    }


@pytest.mark.asyncio
async def test_tool_group_exec_batches_allowed_commands_in_parallel() -> None:
    state = {"active": 0, "max_active": 0}

    async def _track(label: str) -> dict[str, Any]:
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.01)
        state["active"] -= 1
        return {"label": label}

    @tool
    async def memory_search(query: str) -> dict[str, Any]:
        """Search memory."""
        return await _track(query)

    @tool
    async def server_time() -> dict[str, Any]:
        """Return server time."""
        return await _track("time")

    tools = register_tool_gateway_tools(
        None,
        {
            "memory_search": memory_search,
            "server_time": server_time,
        },
    )

    result = await tools["tool_group_exec"].ainvoke(
        {
            "calls": [
                {
                    "group": "memory",
                    "command": "memory_search",
                    "args_json": {"query": "pricing"},
                },
                {"group": "memory", "command": "server_time", "args_json": {}},
            ]
        }
    )

    assert state["max_active"] == 2
    assert result["ok"] is True
    assert result["batched"] is True
    assert result["parallel"] is True
    assert [item["command"] for item in result["results"]] == ["memory_search", "server_time"]


@pytest.mark.asyncio
async def test_tool_group_exec_rejects_unsupported_multi_call_batch() -> None:
    @tool
    async def browser_use_run(task: str) -> dict[str, Any]:
        """Run browser task."""
        return {"task": task}

    @tool
    async def server_time() -> dict[str, Any]:
        """Return server time."""
        return {"ok": True}

    tools = register_tool_gateway_tools(
        None,
        {
            "browser_use_run": browser_use_run,
            "server_time": server_time,
        },
    )

    result = await tools["tool_group_exec"].ainvoke(
        {
            "calls": [
                {"group": "browser", "command": "browser_use_run", "args_json": {"task": "open"}},
                {"group": "memory", "command": "server_time", "args_json": {}},
            ]
        }
    )

    assert result["error"] == "unsupported batch commands"
    assert result["unsupported_commands"] == ["browser_use_run"]


@pytest.mark.asyncio
async def test_tool_group_exec_allows_one_call_inside_calls_list() -> None:
    @tool
    async def browser_use_run(task: str) -> dict[str, Any]:
        """Run browser task."""
        return {"task": task}

    tools = register_tool_gateway_tools(None, {"browser_use_run": browser_use_run})

    result = await tools["tool_group_exec"].ainvoke(
        {
            "calls": [
                {"group": "browser", "command": "browser_use_run", "args_json": {"task": "open"}}
            ]
        }
    )

    assert result == {
        "group": "browser",
        "command": "browser_use_run",
        "ok": True,
        "result": {"task": "open"},
    }


@pytest.mark.asyncio
async def test_tool_group_exec_batch_returns_per_call_errors() -> None:
    @tool
    async def memory_search(query: str) -> dict[str, Any]:
        """Search memory."""
        return {"query": query}

    @tool
    async def server_time() -> dict[str, Any]:
        """Return server time."""
        return {"error": "clock unavailable"}

    tools = register_tool_gateway_tools(
        None,
        {
            "memory_search": memory_search,
            "server_time": server_time,
        },
    )

    result = await tools["tool_group_exec"].ainvoke(
        {
            "calls": [
                {
                    "group": "memory",
                    "command": "memory_search",
                    "args_json": {"query": "pricing"},
                },
                {"group": "memory", "command": "server_time", "args_json": {}},
            ]
        }
    )

    assert result["ok"] is False
    assert [item["ok"] for item in result["results"]] == [True, False]
    assert result["results"][1]["error"] == "clock unavailable"


@pytest.mark.asyncio
async def test_tool_group_exec_repairs_web_image_send_image_url_alias() -> None:
    @tool
    async def web_image_send(url: str) -> dict[str, Any]:
        """Send a web image."""
        return {"sent_url": url}

    tools = register_tool_gateway_tools(None, {"web_image_send": web_image_send})

    result = await tools["tool_group_exec"].ainvoke(
        {
            "group": "web",
            "command": "web_image_send",
            "args_json": {"image_url": "https://example.com/chipmunk.jpg"},
        }
    )

    assert result == {
        "group": "web",
        "command": "web_image_send",
        "ok": True,
        "result": {"sent_url": "https://example.com/chipmunk.jpg"},
    }
