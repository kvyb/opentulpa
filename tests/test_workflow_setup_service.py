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


class _SheetsComposio:
    enabled = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def status(self) -> dict[str, object]:
        return {"ok": True, "enabled": True}

    def list_google_sheets_tab_names(
        self,
        *,
        customer_id: str,
        spreadsheet_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "list_google_sheets_tab_names",
                "customer_id": customer_id,
                "spreadsheet_id": spreadsheet_id,
                "connected_account_id": connected_account_id,
            }
        )
        return {"ok": True, "sheet_names": ["Записи клиентов"]}

    def search_tools(
        self,
        *,
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "search_tools",
                "query": query,
                "toolkits": list(toolkits or []),
                "limit": limit,
            }
        )
        return {
            "ok": True,
            "items": [
                {
                    "slug": "GOOGLESHEETS_UPSERT_ROWS",
                    "toolkit_slug": "googlesheets",
                    "name": "Google Sheets Upsert Rows",
                    "description": "Upsert rows in a Google Sheet.",
                    "input_schema": {"type": "object", "properties": {"rows": {"type": "array"}}},
                }
            ],
        }


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
    assert session["draft_upsert"]["channel"] == ""
    assert session["draft_upsert"]["provider"] == ""
    assert session["draft_upsert"]["reply_mode"] == "auto"
    assert session["scratchpad"]["mode"] == "create"
    assert session["scratchpad"]["source_file_ids"] == []
    assert session["scratchpad"]["knowledge_source_file_ids"] == []
    assert session["scratchpad"]["current_requested_channel"] == ""
    assert session["scratchpad"]["current_requested_provider"] == ""


def test_workflow_setup_begin_web_create_defaults_to_auto_reply_mode(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)

    session = setup.begin_session(
        customer_id="usr_default",
        thread_id="dashboard-owner-dep_123",
        mode="create",
    )

    assert session["draft_upsert"]["channel"] == ""
    assert session["draft_upsert"]["provider"] == ""
    assert session["draft_upsert"]["reply_mode"] == "auto"


def test_workflow_setup_web_update_keeps_explicit_instagram_reply_mode(
    tmp_path: Path,
) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(
        customer_id="usr_default",
        thread_id="dashboard-owner-dep_123",
        mode="create",
    )

    session = setup.update_session(
        customer_id="usr_default",
        thread_id="dashboard-owner-dep_123",
        draft_patch={"channel": "instagram_dm", "reply_mode": "auto"},
    )

    assert session["draft_upsert"]["reply_mode"] == "auto"


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
            "channel": "instagram_dm",
            "provider": "composio",
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
            "channel": "instagram_dm",
            "provider": "composio",
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


def test_workflow_setup_finalize_confirmation_applies_final_patch_and_commits(
    tmp_path: Path,
) -> None:
    setup, intake_service, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "Car Wash Intake",
            "channel": "instagram_dm",
            "provider": "composio",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "assistant_instructions": "Be direct.",
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        },
    )
    setup.mark_proposed(customer_id="telegram_123", thread_id="thread_123")

    session = setup.finalize_confirmation(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "assistant_instructions": (
                "Be direct. If a price is missing, ask the owner; use '-' after one hour."
            )
        },
    )

    assert session["status"] == "completed"
    assert session["created_or_updated_workflow_id"]
    assert session["preflight"]["status"] == "ready"
    workflows = intake_service.list_workflows(customer_id="telegram_123", include_disabled=True)
    assert len(workflows) == 1
    assert workflows[0]["assistant_instructions"].endswith("use '-' after one hour.")


def test_workflow_setup_web_finalize_keeps_explicit_instagram_reply_mode(
    tmp_path: Path,
) -> None:
    setup, intake_service, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="usr_default", thread_id="dashboard-owner-dep_123", mode="create")
    setup.update_session(
        customer_id="usr_default",
        thread_id="dashboard-owner-dep_123",
        draft_patch={
            "name": "Car Wash Intake",
            "channel": "instagram_dm",
            "provider": "composio",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            "reply_mode": "auto",
        },
    )
    setup.mark_proposed(customer_id="usr_default", thread_id="dashboard-owner-dep_123")

    session = setup.finalize_confirmation(
        customer_id="usr_default",
        thread_id="dashboard-owner-dep_123",
        draft_patch={"reply_mode": "auto"},
    )

    assert session["status"] == "completed"
    workflows = intake_service.list_workflows(customer_id="usr_default", include_disabled=True)
    assert len(workflows) == 1
    assert workflows[0]["reply_mode"] == "auto"


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
            "schedule": "*/2 * * * *",
        },
    )

    assert session["draft_upsert"]["channel"] == "telegram_business_dm"
    assert session["draft_upsert"]["provider"] == "telegram_bot_api"
    assert session["draft_upsert"]["schedule"] == ""


def test_workflow_setup_rejects_nested_draft_patch_shape(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")

    with pytest.raises(ValueError, match="workflow fields directly"):
        setup.update_session(
            customer_id="telegram_123",
            thread_id="thread_123",
            draft_patch={
                "draft": {
                    "name": "Car Wash Intake",
                    "channel": "telegram_business_dm",
                }
            },
        )


def test_workflow_setup_latest_source_request_wins(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    session = setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={"channel": "telegram_business_dm"},
    )
    assert session["draft_upsert"]["provider"] == "telegram_bot_api"
    assert session["scratchpad"]["current_requested_channel"] == "telegram_business_dm"

    session = setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={"channel": "instagram_dm"},
    )

    assert session["draft_upsert"]["channel"] == "instagram_dm"
    assert session["draft_upsert"]["provider"] == "composio"
    assert session["scratchpad"]["current_requested_channel"] == "instagram_dm"


def test_workflow_setup_preflight_blocks_stale_source_mismatch(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "channel": "telegram_business_dm",
            "name": "Telegram booking",
            "intent_description": "Book inbound Telegram leads.",
            "required_fields": ["name", "time"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        },
    )
    setup._store.update_session(  # type: ignore[attr-defined]
        session_id=setup.get_thread_session(
            customer_id="telegram_123",
            thread_id="thread_123",
        )["session_id"],
        draft_upsert={
            "name": "Telegram booking",
            "channel": "instagram_dm",
            "provider": "composio",
            "source_config": {},
            "intent_description": "Book inbound Telegram leads.",
            "required_fields": ["name", "time"],
            "field_guidance": {},
            "assistant_instructions": "",
            "business_facts": {},
            "knowledge_file_ids": [],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            "schedule": "*/2 * * * *",
            "notify_user": True,
            "enabled": True,
            "reply_mode": "auto",
        },
    )

    session = setup.preflight_current(customer_id="telegram_123", thread_id="thread_123")

    preflight = session["preflight"]
    assert preflight["status"] == "needs_clarification"
    assert "currently requested channel=telegram_business_dm" in preflight["errors"][0]


def test_workflow_setup_update_replaces_field_guidance_and_sink_field_mapping(
    tmp_path: Path,
) -> None:
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


def test_workflow_setup_update_tracks_source_knowledge_files(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")

    session = setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={"knowledge_file_ids": ["file_source"]},
        scratchpad_patch={
            "source_file_ids": ["file_source"],
            "knowledge_source_file_ids": ["file_source"],
            "candidate_files": [
                {
                    "file_id": "file_source",
                    "filename": "price-list.xlsx",
                    "reason": "Owner uploaded it for the workflow.",
                }
            ],
        },
    )

    assert session["draft_upsert"]["knowledge_file_ids"] == ["file_source"]
    assert session["scratchpad"]["source_file_ids"] == ["file_source"]
    assert session["scratchpad"]["knowledge_source_file_ids"] == ["file_source"]
    assert session["scratchpad"]["candidate_files"][0]["filename"] == "price-list.xlsx"


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

    assert session["draft_upsert"]["sink_config"] == {"file_path": "tulpa_stuff/bookings.csv"}


def test_workflow_setup_preflight_normalizes_single_google_sheet_tab_and_dry_runs(
    tmp_path: Path,
) -> None:
    composio = _SheetsComposio()
    setup, _, _ = _mk_setup_service(tmp_path, composio=composio)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "AutoSpa Telegram Intake",
            "channel": "instagram_dm",
            "provider": "composio",
            "intent_description": "Записывать клиентов на мойку.",
            "required_fields": ["тип услуги", "телефон клиента"],
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "toolkit": "googlesheets",
                "field_mapping": {
                    "тип услуги": "тип услуги",
                    "телефон клиента": "телефон клиента",
                },
                "static_arguments": {"spreadsheet_id": "sheet_123"},
            },
        },
    )

    session = setup.preflight_current(customer_id="telegram_123", thread_id="thread_123")

    preflight = session["preflight"]
    assert preflight["ok"] is True
    assert preflight["status"] == "ready"
    assert "scheduled Composio polling" in " ".join(preflight["warnings"])
    assert preflight["sink_preflight"]["dry_run"]["will_execute"] is False
    assert preflight["sink_preflight"]["dry_run"]["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"
    static_arguments = session["draft_upsert"]["sink_config"]["static_arguments"]
    assert static_arguments == {
        "spreadsheetId": "sheet_123",
        "sheetName": "Записи клиентов",
    }
    preview_args = preflight["sink_preflight"]["dry_run"]["arguments_preview"]
    assert preview_args["spreadsheetId"] == "sheet_123"
    assert preview_args["sheetName"] == "Записи клиентов"
    assert preview_args["headers"][0] == "Booking ID"
    assert set(preview_args["headers"][1:]) == {"тип услуги", "телефон клиента"}
    assert "arguments_preview" not in session["scratchpad"]["last_preflight"]["dry_run"]
    assert all(call["method"] != "execute_tool" for call in composio.calls)


def test_workflow_setup_preflight_reuses_ready_result_for_unchanged_draft(
    tmp_path: Path,
) -> None:
    composio = _SheetsComposio()
    setup, _, _ = _mk_setup_service(tmp_path, composio=composio)
    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "AutoSpa Telegram Intake",
            "channel": "instagram_dm",
            "provider": "composio",
            "intent_description": "Записывать клиентов на мойку.",
            "required_fields": ["тип услуги", "телефон клиента"],
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "toolkit": "googlesheets",
                "field_mapping": {
                    "тип услуги": "тип услуги",
                    "телефон клиента": "телефон клиента",
                },
                "static_arguments": {"spreadsheet_id": "sheet_123"},
            },
        },
    )

    first = setup.preflight_current(customer_id="telegram_123", thread_id="thread_123")
    first_call_count = len(composio.calls)
    second = setup.preflight_current(customer_id="telegram_123", thread_id="thread_123")

    assert first["preflight"]["status"] == "ready"
    assert second["preflight"]["status"] == "ready"
    assert second["preflight"]["cache_hit"] is True
    assert len(composio.calls) == first_call_count

    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={"assistant_instructions": "Offer ceramic coating if the user asks."},
    )
    third = setup.preflight_current(customer_id="telegram_123", thread_id="thread_123")

    assert third["preflight"]["status"] == "ready"
    assert third["preflight"]["cache_hit"] is False
    assert len(composio.calls) > first_call_count


def test_workflow_setup_orchestrator_reports_active_and_paused_states(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    orchestrator = WorkflowSetupOrchestrator(setup_service=setup)

    assert (
        orchestrator.thread_status(customer_id="telegram_123", thread_id="thread_123")["status"]
        == "none"
    )

    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    assert (
        orchestrator.thread_status(customer_id="telegram_123", thread_id="thread_123")["status"]
        == "active"
    )

    setup.pause(customer_id="telegram_123", thread_id="thread_123")
    assert (
        orchestrator.thread_status(customer_id="telegram_123", thread_id="thread_123")["status"]
        == "paused"
    )


def test_workflow_setup_orchestrator_marks_confirmable_proposal_reply(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    orchestrator = WorkflowSetupOrchestrator(setup_service=setup)

    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    setup.update_session(
        customer_id="telegram_123",
        thread_id="thread_123",
        draft_patch={
            "name": "Telegram booking",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "intent_description": "Book inbound Telegram leads.",
            "required_fields": ["name", "time"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        },
    )

    result = orchestrator.after_reply(
        customer_id="telegram_123",
        thread_id="thread_123",
        reply_text="Here's the proposed workflow. Does this look right? Confirm and I'll activate it.",
    )

    assert result["marked"] is True
    session = setup.get_thread_session(customer_id="telegram_123", thread_id="thread_123")
    assert session is not None
    assert str(session["last_proposed_draft_hash"])


def test_workflow_setup_orchestrator_does_not_mark_clarifying_question(tmp_path: Path) -> None:
    setup, _, _ = _mk_setup_service(tmp_path)
    orchestrator = WorkflowSetupOrchestrator(setup_service=setup)

    setup.begin_session(customer_id="telegram_123", thread_id="thread_123", mode="create")
    result = orchestrator.after_reply(
        customer_id="telegram_123",
        thread_id="thread_123",
        reply_text="Draft updated. Before I propose the final workflow, one clarifying question.",
    )

    assert result["marked"] is False
    session = setup.get_thread_session(customer_id="telegram_123", thread_id="thread_123")
    assert session is not None
    assert session["last_proposed_draft_hash"] == ""
