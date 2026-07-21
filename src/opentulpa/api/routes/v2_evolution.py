"""Owner-authenticated API for archived OpenTulpa source candidates."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Protocol, cast

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from opentulpa.evolution.models import Candidate, CandidateStatus, PromotionAttempt
from opentulpa.persistence.idempotency import (
    IdempotencyConflictError,
    IdempotencyPendingError,
)


class EvolutionPrincipal(Protocol):
    tenant_id: str
    actor_id: str


class EvolutionService(Protocol):
    async def start(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def get_candidate(self, candidate_id: str) -> Candidate | None: ...

    async def list_candidates(
        self,
        *,
        status: CandidateStatus | str | None = None,
        limit: int = 100,
    ) -> list[Candidate]: ...

    async def get_promotion_attempt(self, attempt_id: str) -> PromotionAttempt | None: ...

    async def prepare_contribution(
        self,
        candidate_id: str,
        *,
        expected_revision: int | None = None,
        audit_context: dict[str, str] | None = None,
    ) -> Candidate: ...

    async def review_patch(self, candidate_id: str) -> Any: ...


class EvolutionIdempotency(Protocol):
    async def execute(
        self,
        *,
        tenant_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        invoke: Callable[[], Any],
    ) -> Any: ...


class ContributionPrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expected_revision: int = Field(ge=1)


async def _principal(
    request: Request,
    resolver: Callable[
        [Request],
        EvolutionPrincipal | Awaitable[EvolutionPrincipal],
    ],
) -> EvolutionPrincipal:
    value = resolver(request)
    resolved = await value if inspect.isawaitable(value) else value
    if not str(getattr(resolved, "tenant_id", "") or "").strip():
        raise HTTPException(status_code=401, detail="authenticated owner is required")
    if not str(getattr(resolved, "actor_id", "") or "").strip():
        raise HTTPException(status_code=401, detail="authenticated owner is required")
    return resolved


def _public_evaluation(candidate: Candidate) -> dict[str, Any] | None:
    report = candidate.evaluation_report
    if report is None:
        return None
    return {
        "id": report.id,
        "passed": report.passed,
        "source_commit": report.source_commit,
        "artifact_digest": report.artifact_digest,
        "evaluator_fingerprint": report.evaluator_fingerprint,
        "evaluator_version": report.evaluator_version,
        "summary": report.summary,
        "evaluated_at": report.evaluated_at.isoformat(),
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "required": check.required,
                "summary": check.summary,
                "duration_seconds": check.duration_seconds,
            }
            for check in report.checks
        ],
    }


def _public_candidate(candidate: Candidate) -> dict[str, Any]:
    contribution = candidate.contribution
    return {
        "candidate_id": candidate.id,
        "status": candidate.status.value,
        "revision": candidate.revision,
        "requested_improvement": candidate.requested_improvement,
        "parent_candidate_id": candidate.parent_candidate_id,
        "base_commit": candidate.base_commit,
        "source_commit": candidate.source_commit,
        "artifact_digest": candidate.artifact_digest,
        "evaluation": _public_evaluation(candidate),
        "contribution": (
            {
                "upstream_repository": contribution.upstream_repository,
                "base_commit": contribution.base_commit,
                "head_commit": contribution.head_commit,
                "branch_name": contribution.branch_name,
                "pull_request_url": contribution.pull_request_url,
                "sanitized": contribution.sanitized,
                "prepared_at": contribution.prepared_at.isoformat(),
            }
            if contribution is not None
            else None
        ),
        "created_at": candidate.created_at.isoformat(),
        "updated_at": candidate.updated_at.isoformat(),
        "review_patch_url": f"/v2/evolution/candidates/{candidate.id}/patch",
        "audit": {
            key: candidate.metadata[key]
            for key in ("requested_by", "rejected_by")
            if key in candidate.metadata
        },
    }


def _public_promotion_attempt(attempt: PromotionAttempt) -> dict[str, Any]:
    return {
        "attempt_id": attempt.id,
        "candidate_id": attempt.candidate_id,
        "status": attempt.status.value,
        "revision": attempt.revision,
        "release_id": attempt.release.id,
        "operation": (
            "rollback" if attempt.release.metadata.get("rollback_target") else "promotion"
        ),
        "failure_code": attempt.failure_code,
        "failure_message": attempt.failure_message,
        "created_at": attempt.created_at.isoformat(),
        "updated_at": attempt.updated_at.isoformat(),
    }


def register_v2_evolution_routes(
    app: FastAPI,
    *,
    get_evolution_service: Callable[[], EvolutionService | None],
    resolve_principal: Callable[
        [Request],
        EvolutionPrincipal | Awaitable[EvolutionPrincipal],
    ],
    get_idempotency_store: Callable[[], EvolutionIdempotency | None],
) -> None:
    """Register deployment-wide source evolution behind owner authentication."""

    def service_or_503() -> EvolutionService:
        service = get_evolution_service()
        if service is None:
            raise HTTPException(status_code=503, detail="source evolution is unavailable")
        return service

    def required_idempotency_key(value: str | None) -> str:
        key = str(value or "").strip()
        if not key:
            raise HTTPException(status_code=400, detail="Idempotency-Key is required")
        return key

    def audit_context(principal: EvolutionPrincipal, request: Request) -> dict[str, str]:
        correlation_id = str(request.headers.get("x-correlation-id", "") or "").strip()
        thread_id = str(request.headers.get("x-opentulpa-thread-id", "") or "").strip()
        if not thread_id:
            thread_id = correlation_id or f"web:evolution:{principal.actor_id}"
        return {
            "tenant_id": principal.tenant_id,
            "actor_id": principal.actor_id,
            "thread_id": thread_id,
            "correlation_id": correlation_id or thread_id,
            "channel": "web",
            "run_kind": "owner",
            "origin": json.dumps(
                {
                    "interface": "web",
                    "source_id": "v2-evolution",
                    "conversation_id": thread_id,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        }

    async def idempotent_mutation(
        *,
        principal: EvolutionPrincipal,
        key: str,
        operation: str,
        arguments: dict[str, Any],
        invoke: Callable[[], Any],
    ) -> dict[str, Any]:
        store = get_idempotency_store()
        if store is None:
            raise HTTPException(status_code=503, detail="idempotency store is unavailable")
        payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"), default=str)
        request_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        try:
            result = await store.execute(
                tenant_id=principal.tenant_id,
                operation=operation,
                idempotency_key=key,
                request_hash=request_hash,
                invoke=invoke,
            )
        except (IdempotencyConflictError, IdempotencyPendingError) as exc:
            raise HTTPException(status_code=409, detail="idempotency key conflict") from exc
        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="invalid idempotent response")
        return cast("dict[str, Any]", result)

    @app.get("/v2/evolution/candidates")
    async def list_candidates(
        request: Request,
        candidate_status: Annotated[CandidateStatus | None, Query(alias="status")] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 50,
    ) -> dict[str, Any]:
        await _principal(request, resolve_principal)
        candidates = await service_or_503().list_candidates(
            status=candidate_status,
            limit=limit,
        )
        return {"candidates": [_public_candidate(candidate) for candidate in candidates]}

    @app.get("/v2/evolution/candidates/{candidate_id}")
    async def get_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
        await _principal(request, resolve_principal)
        candidate = await service_or_503().get_candidate(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        return _public_candidate(candidate)

    @app.get("/v2/evolution/candidates/{candidate_id}/patch", response_model=None)
    async def review_candidate_patch(candidate_id: str, request: Request) -> FileResponse:
        await _principal(request, resolve_principal)
        service = service_or_503()
        candidate = await service.get_candidate(candidate_id)
        if candidate is None:
            raise HTTPException(status_code=404, detail="candidate not found")
        try:
            path = await service.review_patch(candidate_id)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail="candidate patch is unavailable") from exc
        return FileResponse(
            path=path,
            media_type="text/x-patch",
            filename=f"opentulpa-{candidate_id}.patch",
        )

    @app.get("/v2/evolution/promotions/{attempt_id}")
    async def get_promotion_attempt(attempt_id: str, request: Request) -> dict[str, Any]:
        await _principal(request, resolve_principal)
        attempt = await service_or_503().get_promotion_attempt(attempt_id)
        if attempt is None:
            raise HTTPException(status_code=404, detail="promotion attempt not found")
        return _public_promotion_attempt(attempt)

    @app.post("/v2/evolution/candidates/{candidate_id}/contribution")
    async def prepare_contribution(
        candidate_id: str,
        body: ContributionPrepareRequest,
        request: Request,
        idempotency_key: Annotated[
            str | None,
            Header(alias="Idempotency-Key", max_length=300),
        ] = None,
    ) -> dict[str, Any]:
        principal = await _principal(request, resolve_principal)
        service = service_or_503()

        async def invoke() -> dict[str, Any]:
            candidate = await service.get_candidate(candidate_id)
            if candidate is None:
                raise HTTPException(status_code=404, detail="candidate not found")
            if candidate.revision != body.expected_revision:
                raise HTTPException(status_code=409, detail="candidate revision changed")
            return _public_candidate(
                await service.prepare_contribution(
                    candidate_id,
                    expected_revision=body.expected_revision,
                    audit_context=audit_context(principal, request),
                )
            )

        return await idempotent_mutation(
            principal=principal,
            key=required_idempotency_key(idempotency_key),
            operation="evolution.candidate.contribution.prepare",
            arguments={"candidate_id": candidate_id, **body.model_dump(mode="json")},
            invoke=invoke,
        )

__all__ = [
    "ContributionPrepareRequest",
    "EvolutionPrincipal",
    "EvolutionIdempotency",
    "EvolutionService",
    "register_v2_evolution_routes",
]
