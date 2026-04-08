from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.skills.service import SkillStoreService, _rmtree_ignore_missing


def _mk_client(tmp_path: Path) -> TestClient:
    store = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=store)
    return TestClient(app)


def test_skills_endpoints_crud(tmp_path: Path) -> None:
    with _mk_client(tmp_path) as client:
        upsert = client.post(
            "/internal/skills/upsert",
            json={
                "customer_id": "telegram_123",
                "scope": "user",
                "name": "sales-pitch",
                "description": "Write concise B2B sales pitches.",
                "instructions": "Use problem-solution-benefit structure and CTA.",
            },
        )
        assert upsert.status_code == 200
        assert upsert.json()["ok"] is True

        listed = client.post(
            "/internal/skills/list",
            json={"customer_id": "telegram_123", "include_global": True, "limit": 50},
        )
        assert listed.status_code == 200
        names = {s["name"] for s in listed.json()["skills"]}
        assert "sales-pitch" in names
        assert "skill-creator" in names
        assert "routine-schedule-composer" in names

        fetched = client.post(
            "/internal/skills/get",
            json={"customer_id": "telegram_123", "name": "sales-pitch", "include_files": True},
        )
        assert fetched.status_code == 200
        skill = fetched.json()["skill"]
        assert skill["name"] == "sales-pitch"
        assert "SKILL.md" in skill["skill_path"]

        deleted = client.post(
            "/internal/skills/delete",
            json={"customer_id": "telegram_123", "scope": "user", "name": "sales-pitch"},
        )
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True


def test_rmtree_ignore_missing_accepts_python312_onexc_exception_object() -> None:
    _rmtree_ignore_missing(lambda *_: None, "ignored", FileNotFoundError("gone"))


def test_rmtree_ignore_missing_raises_non_missing_error() -> None:
    with pytest.raises(PermissionError):
        _rmtree_ignore_missing(lambda *_: None, "ignored", PermissionError("blocked"))
