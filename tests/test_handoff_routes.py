from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.handoffs import register_handoff_routes


class _FakeHandoffs:
    def __init__(self) -> None:
        self.respond_calls: list[dict[str, str]] = []

    def list_handoffs(self, *, customer_id: str, status: str = "", limit: int = 50) -> list[dict[str, Any]]:
        assert customer_id == "owner_1"
        assert status == "awaiting_owner"
        assert limit == 25
        return [{"handoff_id": "hnd_1", "status": "awaiting_owner"}]

    def get_handoff(self, *, customer_id: str, handoff_id: str) -> dict[str, Any] | None:
        if customer_id == "owner_1" and handoff_id == "hnd_1":
            return {"handoff_id": "hnd_1", "status": "awaiting_owner"}
        return None

    async def respond(self, *, customer_id: str, handoff_id: str, owner_feedback: str) -> dict[str, Any]:
        self.respond_calls.append(
            {
                "customer_id": customer_id,
                "handoff_id": handoff_id,
                "owner_feedback": owner_feedback,
            }
        )
        return {"ok": True, "handoff": {"handoff_id": handoff_id, "status": "resolved"}}


def _client(fake: _FakeHandoffs) -> TestClient:
    app = FastAPI()
    register_handoff_routes(
        app,
        get_handoffs=lambda: fake,
        web_token="secret",
        resolve_customer_id=lambda value: value,
    )
    return TestClient(app)


def test_handoff_routes_require_bearer_auth() -> None:
    client = _client(_FakeHandoffs())

    response = client.get("/web/intake/handoffs?customer_id=owner_1")

    assert response.status_code == 401


def test_handoff_routes_list_and_respond() -> None:
    fake = _FakeHandoffs()
    client = _client(fake)
    headers = {"Authorization": "Bearer secret"}

    listed = client.get(
        "/web/intake/handoffs?customer_id=owner_1&status=awaiting_owner&limit=25",
        headers=headers,
    )
    detail = client.get("/web/intake/handoffs/hnd_1?customer_id=owner_1", headers=headers)
    responded = client.post(
        "/web/intake/handoffs/hnd_1/respond?customer_id=owner_1",
        headers=headers,
        json={"owner_feedback": "Approve 10%, not 20%."},
    )

    assert listed.status_code == 200
    assert listed.json()["handoffs"] == [{"handoff_id": "hnd_1", "status": "awaiting_owner"}]
    assert detail.status_code == 200
    assert responded.status_code == 200
    assert fake.respond_calls == [
        {
            "customer_id": "owner_1",
            "handoff_id": "hnd_1",
            "owner_feedback": "Approve 10%, not 20%.",
        }
    ]
