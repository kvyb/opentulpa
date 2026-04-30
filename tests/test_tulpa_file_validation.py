from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from opentulpa.api.app import create_app
from opentulpa.tasks import sandbox


def _tmp_rel(name: str) -> str:
    return f"tulpa_stuff/{name}_{uuid.uuid4().hex}.py"


def _cleanup(rel_path: str) -> None:
    target = (Path(sandbox.PROJECT_ROOT) / rel_path).resolve()
    if target.exists():
        target.unlink()


def test_write_file_rejects_python_syntax_errors() -> None:
    rel = _tmp_rel("invalid_syntax")
    with pytest.raises(ValueError, match="Python syntax validation failed"):
        sandbox.write_file(rel, "def broken(:\n    return 1\n")
    _cleanup(rel)


def test_write_file_rejects_tulpa_modules_without_router() -> None:
    rel = _tmp_rel("missing_router")
    with pytest.raises(ValueError, match="must either define a top-level 'router'"):
        sandbox.write_file(rel, "def helper() -> str:\n    return 'ok'\n")
    _cleanup(rel)


def test_validate_file_endpoint_accepts_valid_router_module() -> None:
    rel = _tmp_rel("valid_router")
    content = (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n\n"
        "@router.get('/ping')\n"
        "def ping():\n"
        "    return {'ok': True}\n"
    )
    try:
        sandbox.write_file(rel, content)
        app = create_app()
        with TestClient(app) as client:
            resp = client.post("/internal/tulpa/validate_file", json={"path": rel})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["ok"] is True
        assert payload["python_syntax_ok"] is True
        assert payload["router_contract_ok"] is True
    finally:
        _cleanup(rel)


def test_write_file_allows_tulpa_script_with_main_guard() -> None:
    rel = _tmp_rel("script_main_guard")
    content = (
        "import json\n\n"
        "def run() -> None:\n"
        "    print(json.dumps({'ok': True}))\n\n"
        "if __name__ == '__main__':\n"
        "    run()\n"
    )
    try:
        sandbox.write_file(rel, content)
        validation = sandbox.validate_generated_file(rel)
        assert validation["ok"] is True
        assert validation["python_syntax_ok"] is True
        assert validation["router_contract_ok"] is None
    finally:
        _cleanup(rel)
