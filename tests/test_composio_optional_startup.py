from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.skills.service import SkillStoreService


def test_create_app_tolerates_configured_composio_without_sdk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    settings = get_settings().model_copy(
        update={
            "composio_api_key": "configured-but-sdk-missing",
            "composio_default_callback_url": None,
        }
    )
    monkeypatch.setattr("opentulpa.api.app.get_settings", lambda: settings)

    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=skills)

    with TestClient(app) as client:
        response = client.get("/internal/composio/status")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "enabled": True,
        "callback_url_configured": False,
        "default_callback_url": None,
        "resolved_callback_url": None,
    }
