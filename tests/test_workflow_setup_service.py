from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from opentulpa.application.workflow_setup_orchestrator import WorkflowSetupOrchestrator
from opentulpa.context.file_vault import FileVaultService
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.intake.workflow_setup_service import WorkflowSetupService
from opentulpa.intake.workflow_setup_store import WorkflowSetupSessionStore
from opentulpa.interfaces.telegram.business import TelegramBusinessService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        raise AssertionError("wizard tests should not call decide_intake_workflow")


class _FakeComposio:
    enabled = False

    def status(self) -> dict[str, object]:
        return {"ok": True, "enabled": False}


class _FakeTelegramClient:
    async def send_message(self, **kwargs: Any) -> bool:
        _ = kwargs
        return True


def _mk_setup_service(
    tmp_path: Path,
    *,
    composio: Any | None = None,
) -> tuple[WorkflowSetupService, Any, TelegramBusinessService]:
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.db")
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    telegram_business = TelegramBusinessService(db_path=tmp_path / "telegram_business.db")
    telegram_business.client = _FakeTelegramClient()
    file_vault = FileVaultService(
        root_dir=tmp_path / "file_vault",
        db_path=tmp_path / "file_vault.db",
    )
    intake_service = IntakeWorkflowService(
        db_path=tmp_path / "intake.db",
        project_root=tmp_path,
        scheduler=scheduler,
        skill_store=skills,
        composio=composio or _FakeComposio(),
        telegram_business=telegram_business,
        file_vault=file_vault,
        get_agent_runtime=lambda: _FakeRuntime(),
    )
    store = WorkflowSetupSessionStore(db_path=tmp_path / "workflow_setup.db")
    setup = WorkflowSetupService(store=store, intake_workflows=intake_service)
    return setup, intake_service, telegram_business


def test_workflow_setup_begin_create_persists_session(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)

    session = setup.begin_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        mode="create",
    )

    assert session["status"] == "active"
    assert session["mode"] == "create"
    assert session["draft_upsert"]["channel"] == "instagram_dm"
    assert session["scratchpad"]["mode"] == "create"


def test_workflow_setup_begin_edit_loads_existing_workflow(tmp_path: Path) -> None:
    setup, intake_service, _ = _mk_setup_service(tmp_path)
    workflow = intake_service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle booking requests from Instagram DMs.",
        required_fields=["day", "time", "car_type", "wash_type"],
        assistant_instructions="Be direct.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    session = setup.begin_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        mode="edit",
        workflow_id=workflow["workflow_id"],
    )

    assert session["mode"] == "edit"
    assert session["target_workflow_id"] == workflow["workflow_id"]
    assert session["draft_upsert"]["name"] == "Car Wash Intake"
    assert session["target_workflow_snapshot"]["workflow_id"] == workflow["workflow_id"]


def test_workflow_setup_confirm_requires_fresh_proposal(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        },
    )
    setup.mark_proposed(customer_id="telegram_123", thread_id="thread_123")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={"assistant_instructions": "Be direct."},
    )

    with pytest.raises(ValueError, match="changed after proposal"):
        setup.confirm_current(customer_id="telegram_123", thread_id="thread_123")


def test_workflow_setup_commit_create_persists_active_workflow(tmp_path: Path) -> None:
    setup, intake_service, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "assistant_instructions": "Be direct.",
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        },
    )
    setup.mark_proposed(customer_id="telegram_123", thread_id="thread_123")
    setup.confirm_current(customer_id="telegram_123", thread_id="thread_123")

    session = setup.commit(customer_id="telegram_123", thread_id="thread_123")

    assert session["status"] == "completed"
    assert session["created_or_updated_workflow_id"]
    workflows = intake_service.list_workflows(customer_id="telegram_123", include_disabled=True)
    assert len(workflows) == 1
    assert workflows[0]["name"] == "Car Wash Intake"


def test_workflow_setup_commit_edit_recreates_telegram_workflow(tmp_path: Path) -> None:
    setup, intake_service, telegram_business = _mk_setup_service(tmp_path)
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    original = intake_service.upsert_workflow(
        customer_id="telegram_123",
        name="Original Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        intent_description="Handle booking requests.",
        required_fields=["name", "time"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    setup.begin_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        mode="edit",
        workflow_id=original["workflow_id"],
    )
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "Updated Telegram Intake",
            "intent_description": "Handle booking and reschedule requests.",
            "required_fields": ["name", "time", "service"],
        },
    )
    setup.mark_proposed(customer_id="telegram_123", thread_id="thread_123")
    setup.confirm_current(customer_id="telegram_123", thread_id="thread_123")

    session = setup.commit(customer_id="telegram_123", thread_id="thread_123")

    assert session["status"] == "completed"
    assert session["workflow"]["workflow_id"] != original["workflow_id"]
    workflows = intake_service.list_workflows(customer_id="telegram_123", include_disabled=True)
    assert len(workflows) == 1
    assert workflows[0]["name"] == "Updated Telegram Intake"


def test_workflow_setup_update_clears_schedule_for_telegram_channel(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")

    session = setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "schedule": "*/5 * * * *",
        },
    )

    assert session["draft_upsert"]["channel"] == "telegram_business_dm"
    assert session["draft_upsert"]["schedule"] == ""


def test_workflow_setup_update_replaces_field_guidance_and_sink_field_mapping(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "field_guidance": {
                "customer_name": "Collect the lead's name.",
                "vehicle_type": "Collect the vehicle type.",
            },
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "toolkit": "googlesheets",
                "field_mapping": {
                    "customer_name": "Customer Name",
                    "vehicle_type": "Vehicle Type",
                },
                "static_arguments": {"spreadsheet_id": "sheet_123"},
            },
        },
    )

    session = setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "field_guidance": {
                "car_model": "Collect the car model.",
                "wash_type": "Collect the wash package.",
            },
            "sink_config": {
                "field_mapping": {
                    "car_model": "Car Model",
                    "wash_type": "Wash Type",
                }
            },
        },
    )

    assert session["draft_upsert"]["field_guidance"] == {
        "car_model": "Collect the car model.",
        "wash_type": "Collect the wash package.",
    }
    assert session["draft_upsert"]["sink_config"]["field_mapping"] == {
        "car_model": "Car Model",
        "wash_type": "Wash Type",
    }
    assert session["draft_upsert"]["sink_config"]["static_arguments"] == {
        "spreadsheet_id": "sheet_123"
    }


def test_workflow_setup_update_normalizes_local_csv_filename_alias(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")

    session = setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "sink_type": "local_csv",
            "sink_config": {"filename": "tulpa_stuff/bookings.csv"},
        },
    )

    assert session["draft_upsert"]["sink_config"] == {
        "file_path": "tulpa_stuff/bookings.csv"
    }


def test_workflow_setup_orchestrator_reports_active_and_paused_states(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    orchestrator = WorkflowSetupOrchestrator(setup_service=setup)

    assert orchestrator.thread_status(customer_id="telegram_123", thread_id="thread_123")["status"] == "none"

    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    assert orchestrator.thread_status(customer_id="telegram_123", thread_id="thread_123")["status"] == "active"

    setup.pause(customer_id="telegram_123", thread_id="thread_123")
    assert orchestrator.thread_status(customer_id="telegram_123", thread_id="thread_123")["status"] == "paused"
