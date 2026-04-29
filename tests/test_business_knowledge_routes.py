from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.knowledge import register_knowledge_routes
from opentulpa.business_knowledge.models import KnowledgeQueryAnswer, KnowledgeQueryResult


class _Knowledge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def index_sources(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("index", kwargs))
        return {
            "ok": True,
            "scope_type": kwargs["scope_type"],
            "scope_id": kwargs["scope_id"],
            "sources": [],
        }

    def query(self, **kwargs: object) -> KnowledgeQueryResult:
        self.calls.append(("query", kwargs))
        return KnowledgeQueryResult(
            ok=True,
            query="ceramic wash",
            scope_type="intake_workflow",
            scope_id="iwf_1",
            answer=KnowledgeQueryAnswer(answer_extract="Ceramic wash SUV price 129"),
            warnings=[],
            source_count=1,
            section_count=2,
        )

    def preflight_scope(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("preflight", kwargs))
        return {
            "ok": True,
            "status": "ready",
            "source_count": 1,
            "section_count": 2,
            "warnings": [],
        }

    def promote_scope(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("promote", kwargs))
        return {"ok": True, "source_count": 1, "section_count": 2}


def test_knowledge_routes_delegate_to_service() -> None:
    knowledge = _Knowledge()
    app = FastAPI()
    register_knowledge_routes(app, get_knowledge_service=lambda: knowledge)
    client = TestClient(app)

    index = client.post(
        "/internal/knowledge/index_sources",
        json={
            "customer_id": "telegram_123",
            "scope_type": "workflow_setup",
            "scope_id": "iwsetup_1",
            "file_ids": ["file_1"],
        },
    )
    query = client.post(
        "/internal/knowledge/query",
        json={
            "customer_id": "telegram_123",
            "scope_type": "intake_workflow",
            "scope_id": "iwf_1",
            "query": "ceramic wash",
        },
    )
    preflight = client.post(
        "/internal/knowledge/preflight",
        json={
            "customer_id": "telegram_123",
            "scope_type": "workflow_setup",
            "scope_id": "iwsetup_1",
            "workflow_goal": "wash bookings",
        },
    )
    promote = client.post(
        "/internal/knowledge/promote_scope",
        json={
            "customer_id": "telegram_123",
            "source_scope_type": "workflow_setup",
            "source_scope_id": "iwsetup_1",
            "target_scope_type": "intake_workflow",
            "target_scope_id": "iwf_1",
        },
    )

    assert index.status_code == 200
    assert query.json()["answer_extract"] == "Ceramic wash SUV price 129"
    assert query.json()["section_count"] == 2
    assert preflight.json()["status"] == "ready"
    assert promote.json()["section_count"] == 2
    assert [call[0] for call in knowledge.calls] == ["index", "query", "preflight", "promote"]
