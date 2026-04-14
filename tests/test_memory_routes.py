from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.memory import register_memory_routes


class _FakeMemory:
    user_id = "default-user"

    def __init__(self) -> None:
        self.last_search: dict[str, Any] | None = None

    def add(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {"ok": True}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_search = {
            "query": query,
            "user_id": user_id,
            "limit": limit,
            "metadata": metadata,
        }
        return []


def test_internal_memory_search_passes_metadata_and_clamps_limit() -> None:
    memory = _FakeMemory()
    app = FastAPI()
    register_memory_routes(app, get_memory=lambda: memory)
    client = TestClient(app)

    response = client.post(
        "/internal/memory/search",
        json={
            "query": "durable context",
            "user_id": "customer-1",
            "limit": 999,
            "metadata": {"kind": ["directive_fact", "life_fact"]},
        },
    )

    assert response.status_code == 200
    assert memory.last_search == {
        "query": "durable context",
        "user_id": "customer-1",
        "limit": 25,
        "metadata": {"kind": ["directive_fact", "life_fact"]},
    }
