from __future__ import annotations

from typing import Any

import pytest

from opentulpa.agent import tools_registry as tools_registry_module
from opentulpa.agent.tools_registry import register_runtime_tools


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | list[Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else str(payload)
        self.content = b"" if payload is None else b"x"

    def json(self) -> dict[str, Any] | list[Any]:
        return self._payload if self._payload is not None else {}


class _DummyRuntime:
    def __init__(
        self,
        responses: list[_Response],
        *,
        customer_id: str = "telegram_123",
        thread_id: str = "thread_123",
    ) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._active_customer_id = customer_id
        self._active_thread_id = thread_id

    async def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> _Response:
        self.calls.append((method, path, kwargs))
        if not self._responses:
            raise RuntimeError("unexpected internal API call")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_routine_list_passes_customer_scope() -> None:
    runtime = _DummyRuntime([_Response(200, {"routines": [{"id": "rtn_abc"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["routine_list"].ainvoke({})
    assert result == [{"id": "rtn_abc"}]
    assert runtime.calls[0][0] == "GET"
    assert runtime.calls[0][1] == "/internal/scheduler/routines"
    assert runtime.calls[0][2].get("params") == {"customer_id": "telegram_123"}


@pytest.mark.asyncio
async def test_routine_delete_verifies_removed() -> None:
    runtime = _DummyRuntime(
        [
            _Response(200, {"ok": True}),
            _Response(200, {"routines": []}),
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["routine_delete"].ainvoke(
        {"routine_id": "rtn_deadbeef"}
    )
    assert result["ok"] is True
    assert result["verified_removed"] is True


@pytest.mark.asyncio
async def test_intake_workflow_list_passes_customer_scope() -> None:
    runtime = _DummyRuntime([_Response(200, {"workflows": [{"workflow_id": "iwf_abc"}]})])
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
    runtime = _DummyRuntime([_Response(200, {"workflow": {"workflow_id": "iwf_abc"}})])
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
async def test_intake_workflow_upsert_does_not_use_approval_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _unexpected_evaluate(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("approval guard should not run for intake_workflow_upsert")

    monkeypatch.setattr(
        tools_registry_module.ExecutionBoundaryGuard,
        "evaluate",
        _unexpected_evaluate,
    )
    runtime = _DummyRuntime([_Response(200, {"workflow": {"workflow_id": "iwf_abc"}})])
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
    assert runtime.calls[0][1] == "/internal/intake/workflows/upsert"


@pytest.mark.asyncio
async def test_intake_workflow_upsert_accepts_telegram_business_fields() -> None:
    runtime = _DummyRuntime([_Response(200, {"workflow": {"workflow_id": "iwf_tg"}})])
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
    assert payload["assistant_instructions"] == "Be concise and friendly."
    assert payload["knowledge_file_ids"] == ["file_1", "file_2"]


@pytest.mark.asyncio
async def test_intake_workflow_setup_begin_posts_expected_payload() -> None:
    runtime = _DummyRuntime([_Response(200, {"session": {"session_id": "iwsetup_abc"}})])
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
    runtime = _DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["intake_workflow_setup_update"].ainvoke({})

    assert "draft_patch or scratchpad_patch is required" in str(result.get("error", ""))
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_telegram_business_status_posts_expected_payload() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "connected": True, "connections": []})])
    tools = register_runtime_tools(runtime)

    result = await tools["telegram_business_status"].ainvoke({})

    assert result["connected"] is True
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/telegram/business/status"
    assert runtime.calls[0][2]["json_body"] == {"customer_id": "telegram_123"}


@pytest.mark.asyncio
async def test_intake_workflow_upsert_accepts_string_guidance_and_null_workflow_id() -> None:
    runtime = _DummyRuntime([_Response(200, {"workflow": {"workflow_id": "iwf_new"}})])
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
    runtime = _DummyRuntime([_Response(200, {"workflow": {"workflow_id": "iwf_new"}})])
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
    runtime = _DummyRuntime([])
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

@pytest.mark.asyncio
async def test_customer_scoped_tool_fails_closed_without_customer_context() -> None:
    runtime = _DummyRuntime([_Response(200, {"routines": []})], customer_id="")
    tools = register_runtime_tools(runtime)
    with pytest.raises(RuntimeError, match="customer_id is missing"):
        await tools["routine_list"].ainvoke({})
