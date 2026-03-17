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
