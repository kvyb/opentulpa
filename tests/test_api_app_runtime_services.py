from __future__ import annotations

from pathlib import Path
from typing import Any

from opentulpa.api.app import create_app
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService


class _DisabledComposio:
    enabled = False

    def status(self) -> dict[str, object]:
        return {"ok": True, "enabled": False}


class _RuntimeWithPublicServiceConfigurer:
    def __init__(self) -> None:
        self.configured: list[dict[str, Any]] = []

    def configure_api_services(
        self,
        *,
        link_alias_service: Any,
        composio_service: Any,
        workflow_setup_service: Any,
    ) -> None:
        self.configured.append(
            {
                "link_alias_service": link_alias_service,
                "composio_service": composio_service,
                "workflow_setup_service": workflow_setup_service,
            }
        )


def test_create_app_wires_runtime_services_through_public_configurer(tmp_path: Path) -> None:
    runtime = _RuntimeWithPublicServiceConfigurer()
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.db")
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    composio: Any = _DisabledComposio()

    create_app(
        agent_runtime=runtime,
        scheduler=scheduler,
        skill_store_service=skills,
        intake_workflow_service=IntakeWorkflowService(
            db_path=tmp_path / "intake.db",
            project_root=tmp_path,
            scheduler=scheduler,
            skill_store=skills,
            composio=composio,
        ),
        composio_service=composio,
    )

    assert len(runtime.configured) == 1
    configured = runtime.configured[0]
    assert configured["composio_service"] is composio
    assert hasattr(configured["link_alias_service"], "register_links_from_text")
    assert hasattr(configured["workflow_setup_service"], "get_thread_session")
    assert not hasattr(runtime, "_link_alias_service")
    assert not hasattr(runtime, "_composio_service")
    assert not hasattr(runtime, "_workflow_setup_service")
