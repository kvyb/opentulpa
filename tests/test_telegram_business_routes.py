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


def test_telegram_business_status_route_reports_customer_connections(tmp_path: Path) -> None:
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
    service = app.state.telegram_business
    service.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/telegram/business/status",
            json={"customer_id": "telegram_123"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["connected"] is True
    assert payload["connections"][0]["business_connection_id"] == "bc_123"
