from __future__ import annotations

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


@pytest.mark.asyncio
async def test_intake_workflow_list_passes_customer_scope() -> None:
    runtime = DummyRuntime([Response(200, {"workflows": [{"workflow_id": "iwf_abc"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_list"].ainvoke({})
    assert result == [{"workflow_id": "iwf_abc"}]
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/intake/workflows/list"
    assert runtime.calls[0][2]["json_body"] == {
        "customer_id": "telegram_123",
        "include_disabled": False,
    }


@pytest.mark.asyncio
async def test_intake_workflow_upsert_posts_expected_payload() -> None:
    runtime = DummyRuntime([Response(200, {"workflow": {"workflow_id": "iwf_abc"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_upsert"].ainvoke(
        {
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        }
    )
    assert result["workflow_id"] == "iwf_abc"
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/intake/workflows/upsert"
    payload = kwargs["json_body"]
    assert payload["customer_id"] == "telegram_123"
    assert payload["name"] == "Car Wash Intake"
    assert payload["schedule"] == "*/5 * * * *"
    assert payload["channel"] == "instagram_dm"
    assert payload["provider"] == "composio"
    assert payload["assistant_instructions"] == ""
    assert payload["knowledge_file_ids"] == []


@pytest.mark.asyncio
async def test_intake_workflow_upsert_accepts_telegram_business_fields() -> None:
    runtime = DummyRuntime([Response(200, {"workflow": {"workflow_id": "iwf_tg"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_upsert"].ainvoke(
        {
            "name": "Salon Telegram Intake",
            "intent_description": "Handle Telegram Business booking requests.",
            "required_fields": ["name", "time"],
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": "bc_123"},
            "assistant_instructions": "Be concise and friendly.",
            "knowledge_file_ids": ["file_1", "file_2"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        }
    )

    assert result["workflow_id"] == "iwf_tg"
    payload = runtime.calls[0][2]["json_body"]
    assert payload["channel"] == "telegram_business_dm"
    assert payload["provider"] == "telegram_bot_api"
    assert payload["schedule"] == ""
    assert payload["assistant_instructions"] == "Be concise and friendly."
    assert payload["knowledge_file_ids"] == ["file_1", "file_2"]


@pytest.mark.asyncio
async def test_intake_workflow_setup_begin_posts_expected_payload() -> None:
    runtime = DummyRuntime([Response(200, {"session": {"session_id": "iwsetup_abc"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_setup_begin"].ainvoke({"mode": "create"})

    assert result["session_id"] == "iwsetup_abc"
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/intake/setup/begin"
    assert runtime.calls[0][2]["json_body"] == {
        "customer_id": "telegram_123",
        "thread_id": "thread_123",
        "mode": "create",
        "workflow_id": None,
    }


@pytest.mark.asyncio
async def test_intake_workflow_setup_update_requires_patch() -> None:
    runtime = DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_setup_update"].ainvoke({})

    assert "draft_patch or scratchpad_patch is required" in str(result.get("error", ""))
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_uploaded_file_inspect_structure_posts_expected_payload() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "ok": True,
                    "file": {
                        "id": "file_raw",
                        "original_filename": "price.xlsx",
                        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "size_bytes": 123,
                        "summary": "uploaded price | content_preview=" + ("x" * 2000),
                    },
                    "inspection": {
                        "filename": "price.xlsx",
                        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "format": "xlsx",
                        "structure": {
                            "sheets": [
                                {
                                    "index": 1,
                                    "name": "Мойка",
                                    "matched_terms": ["мойка"],
                                    "max_row": 120,
                                    "max_column": 20,
                                    "nonempty_rows": 80,
                                    "sample_rows": [
                                        {
                                            "source_ref": "Мойка!1",
                                            "row": 1,
                                            "values": ["service", "price", "x" * 500],
                                        }
                                    ],
                                    "matches": [
                                        {
                                            "source_ref": "Мойка!5",
                                            "row": 5,
                                            "values": ["2х-фазная мойка", "1200", "x" * 500],
                                        }
                                    ],
                                    "table_candidates": [],
                                }
                            ],
                            "selection_format": {"sheet_name": "exact sheet name"},
                        },
                    },
                },
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["uploaded_file_inspect_structure"].ainvoke(
        {
            "file_id": "file_raw",
            "search_terms": ["мойка", "шиномонтаж"],
        }
    )

    assert result["inspection"]["format"] == "xlsx"
    assert "sheet_inventory" in result["inspection"]["structure"]
    assert "relevant_sheets" in result["inspection"]["structure"]
    assert "content_preview" not in str(result)
    assert len(str(result)) < 3000
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/files/inspect_structure"
    assert kwargs["json_body"] == {
        "customer_id": "telegram_123",
        "file_id": "file_raw",
        "search_terms": ["мойка", "шиномонтаж"],
    }


@pytest.mark.asyncio
async def test_uploaded_file_prepare_intake_knowledge_posts_expected_payload() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "ok": True,
                    "knowledge_file_id": "file_prepared",
                    "knowledge_file": {
                        "id": "file_prepared",
                        "original_filename": "knowledge.md",
                        "mime_type": "text/markdown",
                        "size_bytes": 20000,
                        "summary": "workflow knowledge | content_preview=" + ("markdown " * 2000),
                    },
                    "source_file_ids": ["file_raw"],
                    "matched_sections": ["price.xlsx:Мойка:1-20"],
                },
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["uploaded_file_prepare_intake_knowledge"].ainvoke(
        {
            "file_ids": ["file_raw"],
            "include_hints": ["Мойка", "Шиномонтаж"],
            "selected_sections": [
                {
                    "file_id": "file_raw",
                    "sheet_name": "Мойка",
                    "row_start": 1,
                    "row_end": 20,
                }
            ],
            "workflow_goal": "Handle car wash and tire fitting bookings.",
            "output_name": "autospa_intake_knowledge.md",
        }
    )

    assert result["knowledge_file_id"] == "file_prepared"
    assert result["knowledge_file"]["id"] == "file_prepared"
    assert "content_preview" not in str(result)
    assert len(str(result)) < 1500
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/files/prepare_intake_knowledge"
    assert kwargs["json_body"] == {
        "customer_id": "telegram_123",
        "file_ids": ["file_raw"],
        "include_hints": ["Мойка", "Шиномонтаж"],
        "selected_sections": [
            {
                "file_id": "file_raw",
                "sheet_name": "Мойка",
                "row_start": 1,
                "row_end": 20,
            }
        ],
        "workflow_goal": "Handle car wash and tire fitting bookings.",
        "output_name": "autospa_intake_knowledge.md",
    }


@pytest.mark.asyncio
async def test_telegram_business_status_posts_expected_payload() -> None:
    runtime = DummyRuntime([Response(200, {"ok": True, "connected": True, "connections": []})])
    tools = register_runtime_tools(runtime)

    result = await tools["telegram_business_status"].ainvoke({})

    assert result["connected"] is True
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/telegram/business/status"
    assert runtime.calls[0][2]["json_body"] == {"customer_id": "telegram_123"}


@pytest.mark.asyncio
async def test_intake_workflow_upsert_accepts_string_guidance_and_null_workflow_id() -> None:
    runtime = DummyRuntime([Response(200, {"workflow": {"workflow_id": "iwf_new"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_upsert"].ainvoke(
        {
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            "field_guidance": "Collect the date, time, car type, and wash type.",
            "workflow_id": None,
        }
    )
    assert result["workflow_id"] == "iwf_new"
    payload = runtime.calls[0][2]["json_body"]
    assert payload["field_guidance"] == {"notes": "Collect the date, time, car type, and wash type."}
    assert payload["workflow_id"] is None


@pytest.mark.asyncio
async def test_intake_workflow_upsert_normalizes_string_none_workflow_id_to_null() -> None:
    runtime = DummyRuntime([Response(200, {"workflow": {"workflow_id": "iwf_new"}})])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_upsert"].ainvoke(
        {
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            "workflow_id": "None",
        }
    )

    assert result["workflow_id"] == "iwf_new"
    payload = runtime.calls[0][2]["json_body"]
    assert payload["workflow_id"] is None


@pytest.mark.asyncio
async def test_intake_workflow_upsert_rejects_google_sheets_shorthand_before_api_call() -> None:
    runtime = DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_upsert"].ainvoke(
        {
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["date", "time", "car_type", "wash_type"],
            "sink_type": "google_sheets",
            "sink_config": {
                "spreadsheet_id": "sheet_123",
                "worksheet_name": "Bookings",
            },
        }
    )

    assert "sink_type=google_sheets is not supported here" in str(result.get("error", ""))
    assert runtime.calls == []
