from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService


class _DisabledComposio:
    enabled = False

    def status(self) -> dict[str, object]:
        return {"ok": True, "enabled": False}


def _mk_client(
    tmp_path: Path,
    *,
    customer_profiles: CustomerProfileService | None = None,
) -> TestClient:
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.db")
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(
        scheduler=scheduler,
        skill_store_service=skills,
        intake_workflow_service=IntakeWorkflowService(
            db_path=tmp_path / "intake.db",
            project_root=tmp_path,
            scheduler=scheduler,
            skill_store=skills,
            composio=_DisabledComposio(),
        ),
        composio_service=_DisabledComposio(),
        customer_profile_service=customer_profiles,
    )
    return TestClient(app)

def test_intake_workflow_routes_crud(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        upsert = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": "telegram_123",
                "name": "Car Wash Intake",
                "intent_description": "Handle booking requests that arrive in Instagram DMs.",
                "required_fields": ["day", "time", "car_type", "wash_type"],
                "assistant_instructions": "Be concise and helpful.",
                "knowledge_file_ids": ["file_1"],
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            },
        )
        assert upsert.status_code == 200
        workflow = upsert.json()["workflow"]
        workflow_id = workflow["workflow_id"]

        listed = client.post(
            "/internal/intake/workflows/list",
            json={"customer_id": "telegram_123", "include_disabled": True},
        )
        assert listed.status_code == 200
        assert {item["workflow_id"] for item in listed.json()["workflows"]} == {workflow_id}

        fetched = client.post(
            "/internal/intake/workflows/get",
            json={"customer_id": "telegram_123", "workflow_id": workflow_id},
        )
        assert fetched.status_code == 200
        assert fetched.json()["workflow"]["name"] == "Car Wash Intake"
        assert fetched.json()["workflow"]["assistant_instructions"] == "Be concise and helpful."
        assert fetched.json()["workflow"]["knowledge_file_ids"] == ["file_1"]

        deleted = client.post(
            "/internal/intake/workflows/delete",
            json={"customer_id": "telegram_123", "workflow_id": workflow_id},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True


def test_intake_workflow_routes_resolve_customer_alias(tmp_path: Path) -> None:
    profiles = CustomerProfileService(tmp_path / "profiles.db")
    profiles.bind_telegram_user_id(user_id="usr_default", telegram_user_id="123")

    with _mk_client(tmp_path, customer_profiles=profiles) as client:
        upsert = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": "telegram_123",
                "name": "Alias Intake",
                "intent_description": "Handle alias-routed requests.",
                "required_fields": ["name"],
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            },
        )
        assert upsert.status_code == 200
        assert upsert.json()["workflow"]["customer_id"] == "usr_default"

        listed = client.post(
            "/internal/intake/workflows/list",
            json={"customer_id": "usr_default", "include_disabled": True},
        )
        assert listed.status_code == 200
        assert [item["name"] for item in listed.json()["workflows"]] == ["Alias Intake"]


def test_telegram_business_workflow_route_requires_delete_then_recreate(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        first = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": "telegram_123",
                "name": "Salon Telegram Intake",
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
                "source_config": {"business_connection_id": "bc_123"},
                "intent_description": "Handle Telegram Business booking requests.",
                "required_fields": ["name", "time"],
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            },
        )
        assert first.status_code == 200
        first_workflow = first.json()["workflow"]
        assert first_workflow["schedule"] == ""
        assert first_workflow["routine_id"] == ""

        second = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": "telegram_123",
                "name": "Salon Telegram Intake Updated",
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
                "source_config": {"business_connection_id": "bc_123"},
                "intent_description": "Handle Telegram Business booking and reschedule requests.",
                "required_fields": ["name", "time", "service"],
                "assistant_instructions": "Be concise and confirm the service.",
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            },
        )
        assert second.status_code == 400
        assert "cannot be updated in place" in second.json()["detail"]

        explicit_update = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": "telegram_123",
                "workflow_id": first_workflow["workflow_id"],
                "name": "Salon Telegram Intake Updated",
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
                "source_config": {"business_connection_id": "bc_123"},
                "intent_description": "Handle Telegram Business booking and reschedule requests.",
                "required_fields": ["name", "time", "service"],
                "assistant_instructions": "Be concise and confirm the service.",
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
            },
        )
        assert explicit_update.status_code == 400
        assert "cannot be edited in place" in explicit_update.json()["detail"]

        listed = client.post(
            "/internal/intake/workflows/list",
            json={"customer_id": "telegram_123", "include_disabled": True},
        )
        assert listed.status_code == 200
        telegram_workflows = [
            item
            for item in listed.json()["workflows"]
            if item["channel"] == "telegram_business_dm"
        ]
        assert len(telegram_workflows) == 1
        assert telegram_workflows[0]["schedule"] == ""
        assert telegram_workflows[0]["routine_id"] == ""


def test_intake_workflow_setup_routes_create_confirm_commit(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        begin = client.post(
            "/internal/intake/setup/begin",
            json={
                "customer_id": "telegram_123",
                "thread_id": "thread_123",
                "mode": "create",
            },
        )
        assert begin.status_code == 200

        updated = client.post(
            "/internal/intake/setup/update",
            json={
                "customer_id": "telegram_123",
                "thread_id": "thread_123",
                "draft_patch": {
                    "name": "Car Wash Intake",
                    "channel": "instagram_dm",
                    "provider": "composio",
                    "intent_description": "Handle booking requests that arrive in Instagram DMs.",
                    "required_fields": ["day", "time", "car_type", "wash_type"],
                    "sink_type": "local_csv",
                    "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
                },
                "scratchpad_patch": {"open_questions": ["Need tone guidance."]},
            },
        )
        assert updated.status_code == 200
        assert updated.json()["session"]["draft_upsert"]["name"] == "Car Wash Intake"

        preflight = client.post(
            "/internal/intake/setup/preflight",
            json={"customer_id": "telegram_123", "thread_id": "thread_123"},
        )
        assert preflight.status_code == 200
        preflight_payload = preflight.json()["preflight"]
        assert preflight_payload["status"] == "ready"
        assert preflight_payload["sink_preflight"]["dry_run"]["will_execute"] is False
        assert preflight_payload["sink_preflight"]["dry_run"]["target"] == {
            "file_path": "tulpa_stuff/bookings.csv"
        }

        proposed = client.post(
            "/internal/intake/setup/mark_proposed",
            json={"customer_id": "telegram_123", "thread_id": "thread_123"},
        )
        assert proposed.status_code == 200
        assert proposed.json()["session"]["last_proposed_draft_hash"]

        confirmed = client.post(
            "/internal/intake/setup/confirm_current",
            json={"customer_id": "telegram_123", "thread_id": "thread_123"},
        )
        assert confirmed.status_code == 200
        assert confirmed.json()["session"]["confirmed_draft_hash"]

        committed = client.post(
            "/internal/intake/setup/commit",
            json={"customer_id": "telegram_123", "thread_id": "thread_123"},
        )
        assert committed.status_code == 200
        session = committed.json()["session"]
        assert session["status"] == "completed"
        assert session["workflow"]["name"] == "Car Wash Intake"


def test_intake_workflow_setup_propose_current_route_preflights_and_marks(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        assert client.post(
            "/internal/intake/setup/begin",
            json={"customer_id": "telegram_123", "thread_id": "thread_123", "mode": "create"},
        ).status_code == 200
        assert client.post(
            "/internal/intake/setup/update",
            json={
                "customer_id": "telegram_123",
                "thread_id": "thread_123",
                "draft_patch": {
                    "name": "Car Wash Intake",
                    "channel": "instagram_dm",
                    "provider": "composio",
                    "intent_description": "Handle booking requests that arrive in Instagram DMs.",
                    "required_fields": ["day", "time"],
                    "sink_type": "local_csv",
                    "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
                },
            },
        ).status_code == 200

        proposed = client.post(
            "/internal/intake/setup/propose_current",
            json={"customer_id": "telegram_123", "thread_id": "thread_123"},
        )

        assert proposed.status_code == 200
        payload = proposed.json()
        assert payload["preflight"]["status"] == "ready"
        assert payload["session"]["last_proposed_draft_hash"]


def test_intake_workflow_setup_finalize_confirmation_route_commits(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        begin = client.post(
            "/internal/intake/setup/begin",
            json={
                "customer_id": "telegram_123",
                "thread_id": "thread_123",
                "mode": "create",
            },
        )
        assert begin.status_code == 200

        updated = client.post(
            "/internal/intake/setup/update",
            json={
                "customer_id": "telegram_123",
                "thread_id": "thread_123",
                "draft_patch": {
                    "name": "Car Wash Intake",
                    "channel": "instagram_dm",
                    "provider": "composio",
                    "intent_description": "Handle booking requests that arrive in Instagram DMs.",
                    "required_fields": ["day", "time", "car_type", "wash_type"],
                    "sink_type": "local_csv",
                    "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
                },
            },
        )
        assert updated.status_code == 200

        finalized = client.post(
            "/internal/intake/setup/finalize_confirmation",
            json={
                "customer_id": "telegram_123",
                "thread_id": "thread_123",
                "draft_patch": {"assistant_instructions": "Use '-' when optional pricing is unknown."},
            },
        )

        assert finalized.status_code == 200
        session = finalized.json()["session"]
        assert session["status"] == "completed"
        assert session["workflow"]["name"] == "Car Wash Intake"
        assert session["workflow"]["assistant_instructions"] == (
            "Use '-' when optional pricing is unknown."
        )
