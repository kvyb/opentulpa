from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.generic_chat import register_generic_chat_routes
from opentulpa.api.routes.intake import register_intake_workflow_routes
from opentulpa.context.file_vault import FileVaultService
from opentulpa.intake.service import IntakeWorkflowService


def _service(tmp_path: Path) -> IntakeWorkflowService:
    return IntakeWorkflowService(
        db_path=tmp_path / "intake.db",
        project_root=tmp_path,
    )


def _file_vault(tmp_path: Path) -> FileVaultService:
    return FileVaultService(
        root_dir=tmp_path / "files",
        db_path=tmp_path / "files.db",
    )


def _app(
    service: IntakeWorkflowService,
    vault: FileVaultService,
    *,
    web_token: str | None = "secret",
) -> FastAPI:
    app = FastAPI()
    register_intake_workflow_routes(
        app,
        get_intake_workflows=lambda: service,
        get_workflow_setup_service=lambda: None,
        get_file_vault=lambda: vault,
        web_token=web_token,
    )
    register_generic_chat_routes(
        app,
        web_token=web_token,
        get_agent_runtime=lambda: None,
        get_file_vault=lambda: vault,
        get_workflow_setup_service=lambda: None,
    )
    return app


def _workflow_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "customer_id": "dashboard",
        "name": "Car wash booking",
        "channel": "instagram_dm",
        "provider": "composio",
        "source_config": {"conversation_filter": "booking intent"},
        "intent_description": "Book qualified car wash leads.",
        "required_fields": ["name", "phone", "service"],
        "field_guidance": {"service": "Ask for package type."},
        "assistant_instructions": "Keep replies short.",
        "business_facts": {"deposit_required": False},
        "knowledge_file_ids": [],
        "sink_type": "local_csv",
        "sink_config": {"file_path": "tulpa_stuff/bookings.csv"},
        "schedule": "*/5 * * * *",
        "notify_user": True,
        "enabled": True,
        "reply_mode": "auto",
    }
    payload.update(overrides)
    return payload


def test_web_workflow_routes_require_bearer_token(tmp_path: Path) -> None:
    app = _app(_service(tmp_path), _file_vault(tmp_path))

    with TestClient(app) as client:
        rejected = client.get("/web/intake/workflows", params={"customer_id": "dashboard"})
        accepted = client.get(
            "/web/intake/workflows",
            params={"customer_id": "dashboard"},
            headers={"authorization": "Bearer secret"},
        )

    assert rejected.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json() == {"ok": True, "workflows": []}


def test_web_workflow_routes_keep_unauthorized_when_web_token_missing(tmp_path: Path) -> None:
    app = _app(_service(tmp_path), _file_vault(tmp_path), web_token=None)

    with TestClient(app) as client:
        response = client.get(
            "/web/intake/workflows",
            params={"customer_id": "dashboard"},
            headers={"authorization": "Bearer secret"},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "unauthorized"}


def test_web_workflow_list_and_get_include_knowledge_files(tmp_path: Path) -> None:
    service = _service(tmp_path)
    vault = _file_vault(tmp_path)
    file_record = vault.ingest_file(
        customer_id="dashboard",
        chat_id=None,
        kind="document",
        telegram_file_id=None,
        original_filename="services.txt",
        mime_type="text/plain",
        caption="Service menu",
        raw_bytes=b"Basic wash\nFull detail",
    )
    workflow = service.upsert_workflow(
        **_workflow_payload(knowledge_file_ids=[file_record["id"]])  # type: ignore[arg-type]
    )
    app = _app(service, vault)

    with TestClient(app) as client:
        list_response = client.get(
            "/web/intake/workflows",
            params={"customer_id": "dashboard"},
            headers={"authorization": "Bearer secret"},
        )
        get_response = client.get(
            f"/web/intake/workflows/{workflow['workflow_id']}",
            params={"customer_id": "dashboard"},
            headers={"authorization": "Bearer secret"},
        )
        content_response = client.get(
            get_response.json()["workflow"]["knowledge_files"][0]["content_path"],
            headers={"authorization": "Bearer secret"},
        )
        metadata_response = client.get(
            get_response.json()["workflow"]["knowledge_files"][0]["metadata_path"],
            headers={"authorization": "Bearer secret"},
        )

    assert list_response.status_code == 200
    assert get_response.status_code == 200
    assert content_response.status_code == 200
    assert metadata_response.status_code == 200
    listed = list_response.json()["workflows"][0]
    fetched = get_response.json()["workflow"]
    assert listed["workflow_id"] == workflow["workflow_id"]
    assert fetched["knowledge_files"][0]["original_filename"] == "services.txt"
    assert fetched["knowledge_files"][0]["content_path"].endswith("/content?customer_id=dashboard")
    assert fetched["knowledge_files"][0]["metadata_path"].endswith("/metadata?customer_id=dashboard")
    assert content_response.text == "Basic wash\nFull detail"
    assert metadata_response.json()["file"]["content_path"].endswith("/content?customer_id=dashboard")


def test_web_workflow_put_validates_and_persists(tmp_path: Path) -> None:
    service = _service(tmp_path)
    vault = _file_vault(tmp_path)
    workflow = service.upsert_workflow(**_workflow_payload())  # type: ignore[arg-type]
    app = _app(service, vault)
    payload = _workflow_payload(
        workflow_id=workflow["workflow_id"],
        name="Updated booking",
        enabled=False,
        required_fields=["name", "vehicle"],
    )

    with TestClient(app) as client:
        updated = client.put(
            f"/web/intake/workflows/{workflow['workflow_id']}",
            json=payload,
            headers={"authorization": "Bearer secret"},
        )
        invalid = client.put(
            f"/web/intake/workflows/{workflow['workflow_id']}",
            json={**payload, "name": ""},
            headers={"authorization": "Bearer secret"},
        )

    assert updated.status_code == 200
    assert updated.json()["workflow"]["name"] == "Updated booking"
    assert updated.json()["workflow"]["enabled"] is False
    persisted = service.get_workflow(
        customer_id="dashboard",
        workflow_id=str(workflow["workflow_id"]),
    )
    assert persisted is not None
    assert persisted["required_fields"] == ["name", "vehicle"]
    assert invalid.status_code == 400
    assert "name is required" in invalid.json()["detail"]


def test_web_workflow_delete_removes_workflow(tmp_path: Path) -> None:
    service = _service(tmp_path)
    vault = _file_vault(tmp_path)
    workflow = service.upsert_workflow(**_workflow_payload())  # type: ignore[arg-type]
    app = _app(service, vault)

    with TestClient(app) as client:
        deleted = client.delete(
            f"/web/intake/workflows/{workflow['workflow_id']}",
            params={"customer_id": "dashboard"},
            headers={"authorization": "Bearer secret"},
        )
        missing = client.get(
            f"/web/intake/workflows/{workflow['workflow_id']}",
            params={"customer_id": "dashboard"},
            headers={"authorization": "Bearer secret"},
        )

    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert missing.status_code == 404
