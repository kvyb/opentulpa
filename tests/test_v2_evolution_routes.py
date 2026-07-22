from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from opentulpa.api.routes.v2_evolution import register_v2_evolution_routes
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvaluationCheck,
    EvaluationReport,
    PromotionAttempt,
    Release,
)
from opentulpa.persistence.idempotency import IdempotencyConflictError


@dataclass(frozen=True)
class _Principal:
    tenant_id: str
    actor_id: str


@dataclass
class _EvolutionService:
    candidates: dict[str, Candidate] = field(default_factory=dict)
    attempts: dict[str, PromotionAttempt] = field(default_factory=dict)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    review_patch_path: Path | None = None

    async def get_candidate(self, candidate_id: str) -> Candidate | None:
        return self.candidates.get(candidate_id)

    async def list_candidates(self, **kwargs: Any) -> list[Candidate]:
        self.calls.append(("list", kwargs))
        return list(self.candidates.values())

    async def get_promotion_attempt(self, attempt_id: str) -> PromotionAttempt | None:
        return self.attempts.get(attempt_id)

    async def prepare_contribution(
        self,
        candidate_id: str,
        *,
        expected_revision: int | None = None,
        audit_context: dict[str, str] | None = None,
    ) -> Candidate:
        self.calls.append(
            (
                "contribution",
                {
                    "candidate_id": candidate_id,
                    "expected_revision": expected_revision,
                    "audit_context": audit_context,
                },
            )
        )
        return self.candidates[candidate_id]

    async def review_patch(self, candidate_id: str) -> Path:
        self.calls.append(("review_patch", {"candidate_id": candidate_id}))
        if self.review_patch_path is None:
            raise RuntimeError("patch unavailable")
        return self.review_patch_path


@dataclass
class _Idempotency:
    records: dict[tuple[str, str], tuple[str, str, dict[str, Any]]] = field(default_factory=dict)

    async def execute(
        self,
        *,
        tenant_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        invoke: Callable[[], Any],
    ) -> dict[str, Any]:
        record_key = (tenant_id, idempotency_key)
        existing = self.records.get(record_key)
        if existing is not None:
            if existing[:2] != (operation, request_hash):
                raise IdempotencyConflictError("conflict")
            return existing[2]
        pending = invoke()
        result = await pending if inspect.isawaitable(pending) else pending
        assert isinstance(result, dict)
        self.records[record_key] = (operation, request_hash, result)
        return result


def _candidate(
    *,
    status: CandidateStatus = CandidateStatus.READY,
    evaluated: bool = True,
) -> Candidate:
    now = datetime.now(UTC)
    report = (
        EvaluationReport(
            candidate_id="candidate-1",
            source_commit="a" * 40,
            artifact_digest="sha256:" + "b" * 64,
            evaluator_fingerprint="sha256:" + "e" * 64,
            evaluator_version="v1",
            passed=True,
            checks=(EvaluationCheck(name="website", passed=True),),
            evaluated_at=now,
        )
        if evaluated
        else None
    )
    return Candidate(
        id="candidate-1",
        base_commit="1" * 40,
        requested_improvement="Add a website",
        status=status,
        revision=3 if evaluated else 1,
        source_commit="a" * 40 if evaluated else None,
        artifact_digest="sha256:" + "b" * 64 if evaluated else None,
        evaluator_fingerprint="sha256:" + "e" * 64 if evaluated else None,
        evaluation_report=report,
        worktree_path="/private/worktree",
        created_at=now,
        updated_at=now,
    )


def _attempt(candidate: Candidate) -> PromotionAttempt:
    return PromotionAttempt(
        candidate_id=candidate.id,
        candidate_revision=candidate.revision,
        release=Release(
            candidate_id=candidate.id,
            source_commit=str(candidate.source_commit),
            artifact_digest=str(candidate.artifact_digest),
            reason="Source release requested by owner agent",
        ),
    )


def _client(
    service: _EvolutionService | None,
) -> tuple[TestClient, _EvolutionService | None]:
    app = FastAPI()
    idempotency = _Idempotency()

    async def principal(request: Request) -> _Principal:
        return _Principal(
            tenant_id=request.headers.get("x-tenant-id", ""),
            actor_id=request.headers.get("x-actor-id", ""),
        )

    register_v2_evolution_routes(
        app,
        get_evolution_service=lambda: service,
        resolve_principal=principal,
        get_idempotency_store=lambda: idempotency,
    )
    return TestClient(app), service


def _headers(*, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {"x-tenant-id": "owner", "x-actor-id": "owner-1"}
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def test_candidate_audit_routes_are_authenticated_and_hide_worktree_path() -> None:
    candidate = _candidate()
    service = _EvolutionService(candidates={candidate.id: candidate})
    client, _ = _client(service)

    unauthorized = client.get("/v2/evolution/candidates")
    listed = client.get("/v2/evolution/candidates", headers=_headers())
    fetched = client.get(
        "/v2/evolution/candidates/candidate-1",
        headers=_headers(),
    )

    assert unauthorized.status_code == 401
    assert listed.status_code == 200
    assert fetched.status_code == 200
    assert listed.json()["candidates"][0]["candidate_id"] == candidate.id
    assert "worktree_path" not in listed.text
    assert "worktree_path" not in fetched.text
    assert service.calls == [("list", {"status": None, "limit": 50})]


def test_promotion_status_remains_available_for_source_release_audit() -> None:
    candidate = _candidate()
    attempt = _attempt(candidate)
    service = _EvolutionService(
        candidates={candidate.id: candidate},
        attempts={attempt.id: attempt},
    )
    client, _ = _client(service)

    response = client.get(
        f"/v2/evolution/promotions/{attempt.id}",
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["attempt_id"] == attempt.id
    assert response.json()["candidate_id"] == candidate.id


def test_contribution_preparation_requires_revision_and_replays_idempotently() -> None:
    candidate = _candidate()
    service = _EvolutionService(candidates={candidate.id: candidate})
    client, _ = _client(service)
    path = "/v2/evolution/candidates/candidate-1/contribution"

    missing_key = client.post(path, headers=_headers(), json={"expected_revision": 3})
    stale = client.post(
        path,
        headers=_headers(idempotency_key="contribution-stale"),
        json={"expected_revision": 2},
    )
    first = client.post(
        path,
        headers=_headers(idempotency_key="contribution-1"),
        json={"expected_revision": 3},
    )
    replay = client.post(
        path,
        headers=_headers(idempotency_key="contribution-1"),
        json={"expected_revision": 3},
    )

    assert missing_key.status_code == 400
    assert stale.status_code == 409
    assert first.status_code == 200
    assert replay.json() == first.json()
    assert [call[0] for call in service.calls] == ["contribution"]
    assert service.calls[0][1]["expected_revision"] == 3


def test_direct_evolution_mutation_routes_are_removed() -> None:
    service = _EvolutionService(candidates={"candidate-1": _candidate()})
    client, _ = _client(service)

    create = client.post(
        "/v2/evolution/candidates",
        headers=_headers(idempotency_key="removed-create"),
        json={"instruction": "Add a website"},
    )
    decision = client.post(
        "/v2/evolution/candidates/candidate-1/decision",
        headers=_headers(idempotency_key="removed-decision"),
        json={"decision": "approve", "expected_revision": 3},
    )
    rollback = client.post(
        "/v2/evolution/rollback",
        headers=_headers(idempotency_key="removed-rollback"),
        json={"decision": "approve"},
    )

    assert create.status_code == 405
    assert decision.status_code == 404
    assert rollback.status_code == 404
    assert service.calls == []


def test_candidate_patch_is_owner_authenticated_and_downloadable(tmp_path: Path) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_text("diff --git a/app.py b/app.py\n", encoding="utf-8")
    service = _EvolutionService(
        candidates={"candidate-1": _candidate()},
        review_patch_path=patch,
    )
    client, _ = _client(service)

    unauthorized = client.get("/v2/evolution/candidates/candidate-1/patch")
    response = client.get(
        "/v2/evolution/candidates/candidate-1/patch",
        headers=_headers(),
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.content == patch.read_bytes()
    assert "attachment" in response.headers["content-disposition"]


def test_evolution_routes_fail_closed_when_capability_is_unavailable() -> None:
    client, _ = _client(None)

    response = client.get("/v2/evolution/candidates", headers=_headers())

    assert response.status_code == 503
