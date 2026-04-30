from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.integrations.composio import ComposioService
from opentulpa.scheduler.service import SchedulerService

LIVE_FLAG = "OPENTULPA_ENABLE_LIVE_COMPOSIO_INTAKE_E2E"
LIVE_WRITE_FLAG = "OPENTULPA_ENABLE_LIVE_COMPOSIO_INTAKE_WRITE_E2E"
LIVE_CUSTOMER_ID_ENV = "OPENTULPA_LIVE_INTAKE_CUSTOMER_ID"
LIVE_CONVERSATION_ID_ENV = "OPENTULPA_LIVE_INTAKE_TEST_CONVERSATION_ID"
LIVE_CONNECTED_ACCOUNT_ENV = "OPENTULPA_LIVE_INTAKE_INSTAGRAM_CONNECTED_ACCOUNT_ID"
LIVE_EXPECTED_FIELDS_ENV = "OPENTULPA_LIVE_INTAKE_EXPECTED_FIELDS_JSON"
LIVE_INTENT_ENV = "OPENTULPA_LIVE_INTAKE_INTENT_DESCRIPTION"
LIVE_WRITE_TOOL_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_TOOL_SLUG"
LIVE_WRITE_CONNECTED_ACCOUNT_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_CONNECTED_ACCOUNT_ID"
LIVE_WRITE_FIELD_MAPPING_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_FIELD_MAPPING_JSON"
LIVE_WRITE_STATIC_ARGUMENTS_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_STATIC_ARGUMENTS_JSON"
LIVE_WRITE_VERIFY_TOOL_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_VERIFY_TOOL_SLUG"
LIVE_WRITE_VERIFY_ARGUMENTS_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_VERIFY_ARGUMENTS_JSON"
LIVE_WRITE_VERIFY_EXPECT_ENV = "OPENTULPA_LIVE_INTAKE_WRITE_VERIFY_EXPECT_JSON"
LIVE_WORKFLOW_NAME = "Live Instagram Intake E2E"
pytestmark = [pytest.mark.e2e]


def _truthy_env(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = str(os.getenv(name, "")).strip()
    if not value:
        pytest.skip(f"{name} is required for live intake e2e", allow_module_level=False)
    return value


def _optional_env(name: str) -> str | None:
    value = str(os.getenv(name, "")).strip()
    return value or None


def _json_env(name: str, *, required: bool = False, default: Any = None) -> Any:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        if required:
            pytest.skip(f"{name} is required for this live intake test", allow_module_level=False)
        return default
    try:
        return json.loads(raw)
    except Exception as exc:  # pragma: no cover - env misconfiguration
        pytest.fail(f"{name} must be valid JSON: {exc}")


def _assert_subset(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            _assert_subset(actual[key], value)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) >= len(expected)
        for index, value in enumerate(expected):
            _assert_subset(actual[index], value)
        return
    assert actual == expected


class _StaticDecisionRuntime:
    def __init__(self, *, fields: dict[str, Any]) -> None:
        self.fields = dict(fields)
        self._link_alias_service = None

    async def decide_intake_workflow(self, **_: Any) -> dict[str, Any]:
        return {
            "ok": True,
            "matches_workflow": True,
            "confidence": 1.0,
            "conversation_summary": "Live sink smoke decision.",
            "extracted_fields": dict(self.fields),
            "missing_fields": [],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": True,
            "booking_action": "create_new_booking",
            "save_payload": dict(self.fields),
            "reason": "static live sink smoke decision",
        }


if not _truthy_env(LIVE_FLAG):
    pytest.skip(
        f"set {LIVE_FLAG}=1 to run live Composio intake e2e tests",
        allow_module_level=True,
    )

_settings_probe = get_settings()
if not str(_settings_probe.composio_api_key or "").strip():
    pytest.skip("COMPOSIO_API_KEY is required for live intake e2e tests", allow_module_level=True)


@pytest.fixture()
def live_composio_service() -> ComposioService:
    settings = get_settings()
    return ComposioService(
        api_key=str(settings.composio_api_key or "").strip(),
        default_callback_url=str(settings.composio_default_callback_url or "").strip() or None,
    )


@pytest.fixture()
def live_runtime(tmp_path: Path) -> OpenTulpaLangGraphRuntime:
    settings = get_settings()
    if not str(settings.openrouter_api_key or "").strip():
        pytest.skip("OPENAI_COMPATIBLE_API_KEY or OPENROUTER_API_KEY is required for live model intake e2e")
    return OpenTulpaLangGraphRuntime(
        app_url="http://testserver",
        openrouter_api_key=str(settings.openrouter_api_key or "").strip(),
        model_name=settings.llm_model,
        wake_classifier_model_name=settings.wake_classifier_model,
        checkpoint_db_path=str(tmp_path / "live_intake_checkpoints.sqlite"),
        behavior_log_path=str(tmp_path / "live_intake_behavior.jsonl"),
        behavior_log_enabled=True,
    )


def test_live_composio_instagram_read_smoke(live_composio_service: ComposioService) -> None:
    customer_id = _required_env(LIVE_CUSTOMER_ID_ENV)
    conversation_id = _required_env(LIVE_CONVERSATION_ID_ENV)
    connected_account_id = _optional_env(LIVE_CONNECTED_ACCOUNT_ENV)

    result = live_composio_service.get_instagram_conversation(
        customer_id=customer_id,
        conversation_id=conversation_id,
        connected_account_id=connected_account_id,
    )

    assert result["ok"] is True
    summary = result["summary"]
    assert summary["conversation_id"] == conversation_id
    assert summary["latest_message_id"]
    assert summary["latest_message_created_time"]
    assert "conversation" in result


def test_live_intake_workflow_local_sink_end_to_end(
    tmp_path: Path,
    live_runtime: OpenTulpaLangGraphRuntime,
    live_composio_service: ComposioService,
) -> None:
    customer_id = _required_env(LIVE_CUSTOMER_ID_ENV)
    conversation_id = _required_env(LIVE_CONVERSATION_ID_ENV)
    connected_account_id = _optional_env(LIVE_CONNECTED_ACCOUNT_ENV)
    expected_fields = _json_env(LIVE_EXPECTED_FIELDS_ENV, required=True, default={})
    if not isinstance(expected_fields, dict) or not expected_fields:
        pytest.fail(f"{LIVE_EXPECTED_FIELDS_ENV} must be a non-empty JSON object")
    intent_description = _optional_env(LIVE_INTENT_ENV) or (
        "Handle booking requests that arrive in Instagram DMs and extract the required booking fields."
    )

    scheduler = SchedulerService(db_path=tmp_path / "live_intake_scheduler.db")
    app = create_app(
        agent_runtime=live_runtime,
        scheduler=scheduler,
        composio_service=live_composio_service,
    )
    csv_relative_path = "tulpa_stuff/live_intake_e2e.csv"

    with TestClient(app) as client:
        upsert = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": customer_id,
                "name": LIVE_WORKFLOW_NAME,
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {
                    "connected_account_id": connected_account_id,
                    "conversation_id": conversation_id,
                },
                "intent_description": intent_description,
                "required_fields": list(expected_fields.keys()),
                "sink_type": "local_csv",
                "sink_config": {"file_path": csv_relative_path},
                "notify_user": False,
                "enabled": True,
            },
        )
        assert upsert.status_code == 200, upsert.text
        workflow = upsert.json()["workflow"]

        run = client.post(
            "/internal/intake/workflows/run",
            json={
                "customer_id": customer_id,
                "workflow_id": workflow["workflow_id"],
                "force": True,
                "event_type": "manual_live_e2e",
            },
        )
        assert run.status_code == 200, run.text
        payload = run.json()
        assert payload["ok"] is True
        assert payload["processed_conversations"] >= 1
        assert payload["matched_conversations"] >= 1

        csv_path = Path.cwd() / csv_relative_path
        assert csv_path.exists()
        rows = csv_path.read_text(encoding="utf-8")
        for value in expected_fields.values():
            assert str(value) in rows


def test_live_intake_workflow_external_sink_smoke(
    tmp_path: Path,
    live_composio_service: ComposioService,
) -> None:
    if not _truthy_env(LIVE_WRITE_FLAG):
        pytest.skip(f"set {LIVE_WRITE_FLAG}=1 to run live external sink intake test")

    customer_id = _required_env(LIVE_CUSTOMER_ID_ENV)
    conversation_id = _required_env(LIVE_CONVERSATION_ID_ENV)
    expected_fields = _json_env(LIVE_EXPECTED_FIELDS_ENV, required=True, default={})
    if not isinstance(expected_fields, dict) or not expected_fields:
        pytest.fail(f"{LIVE_EXPECTED_FIELDS_ENV} must be a non-empty JSON object")

    sink_tool_slug = _required_env(LIVE_WRITE_TOOL_ENV)
    sink_connected_account_id = _optional_env(LIVE_WRITE_CONNECTED_ACCOUNT_ENV)
    field_mapping = _json_env(LIVE_WRITE_FIELD_MAPPING_ENV, required=True, default={})
    static_arguments = _json_env(LIVE_WRITE_STATIC_ARGUMENTS_ENV, default={})
    if not isinstance(field_mapping, dict) or not field_mapping:
        pytest.fail(f"{LIVE_WRITE_FIELD_MAPPING_ENV} must be a non-empty JSON object")
    if not isinstance(static_arguments, dict):
        pytest.fail(f"{LIVE_WRITE_STATIC_ARGUMENTS_ENV} must be a JSON object when set")

    scheduler = SchedulerService(db_path=tmp_path / "live_intake_write_scheduler.db")
    runtime = _StaticDecisionRuntime(fields=expected_fields)
    app = create_app(
        agent_runtime=runtime,
        scheduler=scheduler,
        composio_service=live_composio_service,
    )

    with TestClient(app) as client:
        upsert = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": customer_id,
                "name": f"{LIVE_WORKFLOW_NAME} Write",
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {
                    "connected_account_id": _optional_env(LIVE_CONNECTED_ACCOUNT_ENV),
                    "conversation_id": conversation_id,
                },
                "intent_description": "Write one confirmed intake booking to the configured external sink.",
                "required_fields": list(expected_fields.keys()),
                "sink_type": "google_sheets_composio",
                "sink_config": {
                    "tool_slug": sink_tool_slug,
                    "connected_account_id": sink_connected_account_id,
                    "field_mapping": field_mapping,
                    "static_arguments": static_arguments,
                },
                "notify_user": False,
                "enabled": True,
            },
        )
        assert upsert.status_code == 200, upsert.text
        workflow = upsert.json()["workflow"]

        run = client.post(
            "/internal/intake/workflows/run",
            json={
                "customer_id": customer_id,
                "workflow_id": workflow["workflow_id"],
                "force": True,
                "event_type": "manual_live_write_e2e",
            },
        )
        assert run.status_code == 200, run.text
        payload = run.json()
        assert payload["ok"] is True
        service = client.app.state.intake_workflows
        bookings = service.list_bookings(
            customer_id=customer_id,
            workflow_id=workflow["workflow_id"],
            conversation_id=conversation_id,
        )
        assert bookings
        sink_ref = bookings[0]["sink_record_ref"]
        assert sink_ref.get("toolkit") == "googlesheets"

        verify_tool = _optional_env(LIVE_WRITE_VERIFY_TOOL_ENV)
        if verify_tool:
            verify_args = _json_env(LIVE_WRITE_VERIFY_ARGUMENTS_ENV, required=True, default={})
            verify_expect = _json_env(LIVE_WRITE_VERIFY_EXPECT_ENV, required=True, default={})
            verify_connected_account = sink_connected_account_id
            verify_result = live_composio_service.execute_tool(
                customer_id=customer_id,
                tool_slug=verify_tool,
                arguments=verify_args,
                connected_account_id=verify_connected_account,
            )
            assert verify_result["successful"] is True
            _assert_subset(verify_result.get("data"), verify_expect)
