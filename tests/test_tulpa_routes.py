from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes import tulpa


def test_internal_tulpa_run_terminal_uses_threadpool(monkeypatch) -> None:
    app = FastAPI()
    tulpa.register_tulpa_routes(app, get_tulpa_loader=lambda: object())
    calls: list[dict[str, object]] = []

    async def _fake_run_in_threadpool(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append({"func": func, "args": args, "kwargs": dict(kwargs)})
        return {"ok": True, "stdout": "done"}

    monkeypatch.setattr(tulpa, "run_in_threadpool", _fake_run_in_threadpool)

    with TestClient(app) as client:
        response = client.post(
            "/internal/tulpa/run_terminal",
            json={
                "command": "agent-context query hello --json",
                "working_dir": "tulpa_stuff",
                "timeout_seconds": 45,
            },
        )

    assert response.status_code == 200
    assert response.json() == {"ok": True, "stdout": "done"}
    assert calls == [
        {
            "func": tulpa.sandbox_run_terminal,
            "args": (),
            "kwargs": {
                "command": "agent-context query hello --json",
                "working_dir": "tulpa_stuff",
                "timeout_seconds": 45,
            },
        }
    ]


def test_internal_tulpa_run_terminal_returns_400_for_missing_binary(monkeypatch) -> None:
    app = FastAPI()
    tulpa.register_tulpa_routes(app, get_tulpa_loader=lambda: object())

    async def _fake_run_in_threadpool(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        del func, args, kwargs
        raise FileNotFoundError("[Errno 2] No such file or directory: 'phantom-cli'")

    monkeypatch.setattr(tulpa, "run_in_threadpool", _fake_run_in_threadpool)

    with TestClient(app) as client:
        response = client.post(
            "/internal/tulpa/run_terminal",
            json={
                "command": "phantom-cli status",
                "working_dir": "tulpa_stuff",
                "timeout_seconds": 45,
            },
        )

    assert response.status_code == 400
    assert response.json()["command"] == "phantom-cli status"
    assert response.json()["working_dir"] == "tulpa_stuff"
    assert "No such file or directory" in response.json()["detail"]


def test_internal_tulpa_read_file_returns_404_for_missing_file(monkeypatch) -> None:
    app = FastAPI()
    tulpa.register_tulpa_routes(app, get_tulpa_loader=lambda: object())

    def _fake_read_file(path: str, max_chars: int = 12000) -> str:
        del max_chars
        raise FileNotFoundError(f"file not found under allowed read roots: {path}")

    monkeypatch.setattr(tulpa, "sandbox_read_file", _fake_read_file)

    with TestClient(app) as client:
        response = client.get("/internal/tulpa/read_file", params={"path": "tulpa_stuff/missing.txt"})

    assert response.status_code == 404
    assert response.json()["requested_path"] == "tulpa_stuff/missing.txt"
    assert "file not found under allowed read roots" in response.json()["detail"]
    assert "allowed_read_roots" in response.json()


def test_internal_tulpa_read_file_returns_403_for_disallowed_path(monkeypatch) -> None:
    app = FastAPI()
    tulpa.register_tulpa_routes(app, get_tulpa_loader=lambda: object())

    def _fake_read_file(path: str, max_chars: int = 12000) -> str:
        del max_chars
        raise PermissionError(
            "path outside allowed read roots; allowed roots: "
            "tulpa_stuff/, src/opentulpa/integrations/"
        )

    monkeypatch.setattr(tulpa, "sandbox_read_file", _fake_read_file)

    with TestClient(app) as client:
        response = client.get("/internal/tulpa/read_file", params={"path": "README.md"})

    assert response.status_code == 403
    assert response.json()["requested_path"] == "README.md"
    assert "path outside allowed read roots" in response.json()["detail"]
    assert "allowed_read_roots" in response.json()
