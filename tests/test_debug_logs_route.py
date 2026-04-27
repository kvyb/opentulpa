from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes import debug_logs as debug_logs_module
from opentulpa.api.routes.debug_logs import register_debug_log_routes


def test_debug_logs_route_returns_app_log(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "app.log"
    log_path.write_text("alpha\nbeta\n", encoding="utf-8")
    monkeypatch.setattr(debug_logs_module, "get_debug_log_path", lambda: log_path)

    app = FastAPI()
    register_debug_log_routes(app)
    with TestClient(app) as client:
        response = client.get("/debug_logs")

    assert response.status_code == 200
    assert response.text == "alpha\nbeta\n"
    assert "app.log" in response.headers.get("content-disposition", "")


def test_debug_logs_route_returns_404_when_missing(monkeypatch) -> None:
    missing_path = Path("/tmp/does-not-exist-app.log")
    monkeypatch.setattr(debug_logs_module, "get_debug_log_path", lambda: missing_path)

    app = FastAPI()
    register_debug_log_routes(app)
    with TestClient(app) as client:
        response = client.get("/debug_logs")

    assert response.status_code == 404
    assert response.json() == {"detail": "app.log not found"}
