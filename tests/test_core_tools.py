from __future__ import annotations

import pytest

from opentulpa.agent.graph_control_tools import execute_graph_control_tool
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.agent.turn_plan import TurnPlanValidationError, update_turn_plan
from tests.tool_test_helpers import DummyRuntime, Response

CORE_TOOL_NAMES = {
    "send_owner_update",
    "turn_plan",
    "memory_search",
    "memory_add",
    "uploaded_file_search",
    "uploaded_file_get",
    "uploaded_file_send",
    "tulpa_file_send",
    "web_image_send",
    "uploaded_file_analyze",
    "uploaded_file_inspect_structure",
    "business_knowledge_index",
    "business_knowledge_query",
    "user_context_add_files",
    "user_context_query",
    "user_context_list_sources",
    "user_context_find_sources",
    "user_context_reindex",
    "user_context_archive_sources",
    "user_context_promote_to_intake",
    "directive_get",
    "directive_set",
    "directive_clear",
    "time_profile_get",
    "time_profile_set",
    "web_search",
    "fetch_url_content",
    "fetch_file_content",
    "tulpa_write_file",
    "tulpa_validate_file",
    "tulpa_reload",
    "tulpa_run_terminal",
    "tulpa_read_file",
    "tulpa_catalog",
    "task_status",
    "task_events",
    "task_artifacts",
    "task_relaunch",
    "task_cancel",
    "server_time",
}


class _UpdateRuntime(DummyRuntime):
    def __init__(self) -> None:
        super().__init__([])
        self.updates: list[dict[str, str]] = []

    async def emit_interactive_update(self, *, text: str, dedupe_key: str = "") -> dict[str, bool]:
        self.updates.append({"text": text, "dedupe_key": dedupe_key})
        return {"ok": True, "sent": True}


def test_register_runtime_tools_keeps_core_tool_names() -> None:
    tools = register_runtime_tools(DummyRuntime([]))

    assert set(tools) >= CORE_TOOL_NAMES


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
async def test_turn_plan_direct_tool_is_graph_control_only() -> None:
    runtime = DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["turn_plan"].ainvoke(
        {
            "items": [
                {"id": "scope", "content": "Define the deliverable", "status": "completed"},
                {"id": "search", "content": "Gather evidence", "status": "in_progress"},
            ]
        }
    )

    assert result["ok"] is False
    assert "GRAPH_CONTROL_TOOL_ONLY" in result["error"]


def test_turn_plan_graph_control_tracks_current_turn_items() -> None:
    result = execute_graph_control_tool(
        tool_name="turn_plan",
        args={
            "items": [
                {"id": "scope", "content": "Define the deliverable", "status": "completed"},
                {"id": "search", "content": "Gather evidence", "status": "in_progress"},
            ]
        },
        state={"turn_plan": []},
    ).result

    assert result["ok"] is True
    assert result["summary"] == {
        "total": 2,
        "pending": 0,
        "in_progress": 1,
        "completed": 1,
        "cancelled": 0,
    }
    assert result["items"][1]["content"] == "Gather evidence"
    assert result["next_item"] == {
        "id": "search",
        "content": "Gather evidence",
        "status": "in_progress",
    }


def test_turn_plan_graph_control_accepts_json_encoded_items() -> None:
    result = execute_graph_control_tool(
        tool_name="turn_plan",
        args={
            "items": (
                '[{"id":"scope","content":"Define the deliverable","status":"completed"},'
                '{"id":"search","content":"Gather evidence","status":"in_progress"}]'
            )
        },
        state={"turn_plan": []},
    ).result

    assert result["ok"] is True
    assert result["summary"]["total"] == 2
    assert result["items"][1] == {
        "id": "search",
        "content": "Gather evidence",
        "status": "in_progress",
    }


def test_turn_plan_merge_updates_existing_items() -> None:
    result = update_turn_plan(
        [
            {"id": "search", "content": "Gather evidence", "status": "in_progress"},
            {"id": "answer", "content": "Report answer", "status": "pending"},
        ],
        items=[{"id": "search", "content": "Gather evidence", "status": "completed"}],
        merge=True,
    )

    assert result == [
        {"id": "search", "content": "Gather evidence", "status": "completed"},
        {"id": "answer", "content": "Report answer", "status": "pending"},
    ]


def test_turn_plan_rejects_invalid_status() -> None:
    with pytest.raises(TurnPlanValidationError, match="status must be one of"):
        update_turn_plan(
            [],
            items=[{"id": "search", "content": "Gather evidence", "status": "working"}],
        )


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


def test_memory_add_documents_interactive_style_preferences() -> None:
    tools = register_runtime_tools(DummyRuntime([]))
    description = str(tools["memory_add"].description)

    assert "stable preferences" in description
    assert "style instructions" in description
    assert "normal interactive chat" in description


@pytest.mark.asyncio
async def test_server_time_returns_expected_keys() -> None:
    runtime = DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["server_time"].ainvoke({})

    assert "server_time_local_iso" in result
    assert "server_time_utc_iso" in result
    assert "unix_timestamp" in result
