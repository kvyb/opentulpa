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

        deleted = client.post(
            "/internal/intake/workflows/delete",
            json={"customer_id": "telegram_123", "workflow_id": workflow_id},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
