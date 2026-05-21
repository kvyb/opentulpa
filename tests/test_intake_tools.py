from __future__ import annotations

import pytest

from opentulpa.agent.tools.intake_setup_tools import register_intake_setup_tools
from opentulpa.agent.tools_registry import register_runtime_tools
from tests.tool_test_helpers import DummyRuntime, Response


def test_intake_setup_tools_can_be_registered_directly() -> None:
    runtime = DummyRuntime([])
    tools = register_intake_setup_tools(runtime)

    assert "intake_workflow_setup_begin" in tools
    assert "intake_workflow_setup_update" in tools
    assert "intake_workflow_upsert" not in tools


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
    assert payload["schedule"] == "*/2 * * * *"
    assert payload["channel"] == "instagram_dm"
    assert payload["provider"] == "composio"
    assert payload["assistant_instructions"] == ""
    assert payload["knowledge_file_ids"] == []
    assert payload["reply_mode"] == "auto"


@pytest.mark.asyncio
async def test_intake_workflow_upsert_defaults_web_origin_to_auto() -> None:
    runtime = DummyRuntime(
        [Response(200, {"workflow": {"workflow_id": "iwf_web"}})],
        thread_id="dashboard-owner-dep_123",
    )
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_upsert"].ainvoke(
        {
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        }
    )

    assert result["workflow_id"] == "iwf_web"
    assert runtime.calls[0][2]["json_body"]["reply_mode"] == "auto"


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
    assert payload["reply_mode"] == "auto"


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
async def test_intake_workflow_setup_preflight_posts_expected_payload() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "preflight": {
                        "ok": True,
                        "status": "ready",
                        "sink_preflight": {"dry_run": {"will_execute": False}},
                    }
                },
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_setup_preflight"].ainvoke({})

    assert result["status"] == "ready"
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/intake/setup/preflight"
    assert runtime.calls[0][2]["json_body"] == {
        "customer_id": "telegram_123",
        "thread_id": "thread_123",
    }


@pytest.mark.asyncio
async def test_intake_workflow_setup_finalize_confirmation_posts_expected_payload() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {"session": {"status": "completed", "created_or_updated_workflow_id": "iwf_abc"}},
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_setup_finalize_confirmation"].ainvoke(
        {"draft_patch": {"assistant_instructions": "Use '-' when optional pricing is unknown."}}
    )

    assert result["status"] == "completed"
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/intake/setup/finalize_confirmation"
    assert runtime.calls[0][2]["json_body"] == {
        "customer_id": "telegram_123",
        "thread_id": "thread_123",
        "draft_patch": {"assistant_instructions": "Use '-' when optional pricing is unknown."},
        "scratchpad_patch": None,
    }


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
async def test_tulpa_file_send_marks_delivered_file_for_agent() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "ok": True,
                    "path": "tulpa_stuff/sample_delivery_report.txt",
                    "chat_id": 12345,
                    "delivered_to_chat": True,
                },
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["tulpa_file_send"].ainvoke(
        {
            "path": "tulpa_stuff/sample_delivery_report.txt",
            "caption": "Sample delivery report",
        }
    )

    assert result["delivered_to_chat"] is True
    assert result["delivery_status"] == "delivered_to_telegram_chat"
    assert "DELIVERED_TO_CHAT" in result["model_instruction"]
    assert "Do not call the file-send tool again" in result["model_instruction"]
    assert runtime.calls[0][1] == "/internal/files/send_local"
    assert runtime.calls[0][2]["json_body"] == {
        "path": "tulpa_stuff/sample_delivery_report.txt",
        "customer_id": "telegram_123",
        "caption": "Sample delivery report",
    }


@pytest.mark.asyncio
async def test_business_knowledge_index_posts_expected_payload() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "ok": True,
                    "scope_type": "workflow_setup",
                    "scope_id": "iwsetup_123",
                    "sources": [
                        {
                            "file_id": "file_raw",
                            "filename": "price.xlsx",
                            "status": "indexed",
                            "source_kind": "structured_table",
                            "section_count": 12,
                            "char_count": 1234,
                            "warnings": [],
                        }
                    ],
                },
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["business_knowledge_index"].ainvoke(
        {
            "file_ids": ["file_raw"],
            "scope_type": "workflow_setup",
            "scope_id": "iwsetup_123",
        }
    )

    assert result["sources"][0]["file_id"] == "file_raw"
    assert result["sources"][0]["section_count"] == 12
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/knowledge/index_sources"
    assert kwargs["json_body"] == {
        "customer_id": "telegram_123",
        "scope_type": "workflow_setup",
        "scope_id": "iwsetup_123",
        "file_ids": ["file_raw"],
    }


@pytest.mark.asyncio
async def test_business_knowledge_query_posts_expected_payload() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "ok": True,
                    "query": "2 phase wash price",
                    "scope_type": "intake_workflow",
                    "scope_id": "iwf_abc",
                    "answer_extract": "2х-фазная мойка кузова = 1200",
                    "source_count": 1,
                    "section_count": 3,
                    "warnings": [],
                },
            )
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["business_knowledge_query"].ainvoke(
        {
            "query": "2 phase wash price",
            "scope_type": "intake_workflow",
            "scope_id": "iwf_abc",
        }
    )

    assert result == {
        "query": "2 phase wash price",
        "answer_extract": "2х-фазная мойка кузова = 1200",
    }
    method, path, kwargs = runtime.calls[0]
    assert method == "POST"
    assert path == "/internal/knowledge/query"
    assert kwargs["json_body"] == {
        "customer_id": "telegram_123",
        "scope_type": "intake_workflow",
        "scope_id": "iwf_abc",
        "query": "2 phase wash price",
        "max_extract_chars": 3000,
    }


@pytest.mark.asyncio
async def test_user_context_tools_post_expected_payloads() -> None:
    runtime = DummyRuntime(
        [
            Response(
                200,
                {
                    "ok": True,
                    "scope_type": "user_context",
                    "scope_id": "telegram_123",
                    "sources": [{"file_id": "file_1", "status": "indexed"}],
                },
            ),
            Response(
                200,
                {
                    "ok": True,
                    "query": "brand voice",
                    "answer_extract": "Short hooks and direct CTAs.",
                    "sources": [{"file_id": "file_1"}],
                },
            ),
        ]
    )
    tools = register_runtime_tools(runtime)

    add_result = await tools["user_context_add_files"].ainvoke({"file_ids": ["file_1"]})
    query_result = await tools["user_context_query"].ainvoke({"query": "brand voice"})

    assert add_result["scope_type"] == "user_context"
    assert query_result["answer_extract"] == "Short hooks and direct CTAs."
    assert runtime.calls[0][1] == "/internal/user_context/add_files"
    assert runtime.calls[0][2]["json_body"] == {
        "customer_id": "telegram_123",
        "file_ids": ["file_1"],
    }
    assert runtime.calls[1][1] == "/internal/user_context/query"
    assert runtime.calls[1][2]["json_body"] == {
        "customer_id": "telegram_123",
        "query": "brand voice",
        "max_extract_chars": 3000,
    }


@pytest.mark.asyncio
async def test_business_knowledge_query_current_workflow_resolves_active_setup_scope() -> None:
    runtime = DummyRuntime(
        [
            Response(200, {"session": {"session_id": "iwsetup_123"}}),
            Response(
                200,
                {
                    "ok": True,
                    "query": "wash prices",
                    "scope_type": "workflow_setup",
                    "scope_id": "iwsetup_123",
                    "answer_extract": "Мойка starts at 200",
                },
            ),
        ],
        thread_id="chat_123",
    )
    tools = register_runtime_tools(runtime)

    result = await tools["business_knowledge_query"].ainvoke(
        {
            "query": "wash prices",
            "scope_type": "current_workflow",
            "scope_id": "iwsetup_123",
        }
    )

    assert result == {
        "query": "wash prices",
        "answer_extract": "Мойка starts at 200",
    }
    assert runtime.calls[0][1] == "/internal/intake/setup/get"
    assert runtime.calls[1][1] == "/internal/knowledge/query"
    assert runtime.calls[1][2]["json_body"]["scope_type"] == "workflow_setup"
    assert runtime.calls[1][2]["json_body"]["scope_id"] == "iwsetup_123"


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
