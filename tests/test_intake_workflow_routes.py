from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService


class _DisabledComposio:
    enabled = False

    def status(self) -> dict[str, object]:
        return {"ok": True, "enabled": False}


def _mk_client(tmp_path: Path) -> TestClient:
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
