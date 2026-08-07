from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from opentulpa.bootstrap.evolution_api import EvolutionClient, register_evolution_control_api
from opentulpa.evolution.models import Candidate, PromotionAttempt, Release


class _EvolutionService:
    def __init__(self, patch: Path) -> None:
        self.candidate = Candidate(
            id="candidate_test",
            base_commit="a" * 40,
            requested_improvement="Improve one thing",
        )
        self.patch = patch
        self.attempts: dict[str, PromotionAttempt] = {}
        self.source_calls: list[tuple[str, dict[str, Any]]] = []

    async def source_status(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls.append(("status", kwargs))
        return {
            "active": True,
            "candidate_id": self.candidate.id,
            "diff_sha256": "d" * 64,
            "current_release_id": "release-current",
            "rollback_target_release_id": "release-prior",
        }

    async def source_shell(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls.append(("shell", kwargs))
        return {
            "active": True,
            "candidate": {"id": self.candidate.id, "status": "building"},
            "exit_code": 0,
            "output": "tests passed\n",
        }

    async def source_sync_upstream(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls.append(("sync-upstream", kwargs))
        return {
            "synced": True,
            "candidate_id": self.candidate.id,
            "upstream_commit": "b" * 40,
        }

    async def source_resolve_dependencies(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls.append(("resolve-dependencies", kwargs))
        return {
            "candidate_id": self.candidate.id,
            "dependency_base_id": "e" * 64,
            "dependency_lock_hash": "f" * 64,
        }

    async def source_release(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls.append(("release", kwargs))
        return {
            "active": False,
            "candidate": {"id": self.candidate.id, "status": "ready"},
            "promotion": None,
        }

    async def source_rollback(self, **kwargs: Any) -> PromotionAttempt:
        self.source_calls.append(("rollback", kwargs))
        attempt = PromotionAttempt(
            candidate_id=self.candidate.id,
            candidate_revision=self.candidate.revision,
            release=Release(
                candidate_id=self.candidate.id,
                source_commit="a" * 40,
                artifact_digest=f"sha256:{'b' * 64}",
                metadata={"rollback_target": "release-prior"},
            ),
        )
        self.attempts[attempt.id] = attempt
        return attempt

    async def source_set_runtime_env(self, **kwargs: Any) -> dict[str, Any]:
        self.source_calls.append(("runtime-env", kwargs))
        return {
            "status": "updated",
            "name": kwargs["name"],
            "changed": True,
            "restarted": True,
            "value": "[set]",
        }

    async def list_candidates(self, **_: Any) -> list[Candidate]:
        return [self.candidate]

    async def get_candidate(self, candidate_id: str) -> Candidate | None:
        return self.candidate if candidate_id == self.candidate.id else None

    async def get_promotion_attempt(self, attempt_id: str) -> PromotionAttempt | None:
        return self.attempts.get(attempt_id)

    async def prepare_contribution(self, candidate_id: str, **_: Any) -> Candidate:
        assert candidate_id == self.candidate.id
        return self.candidate

    async def review_patch(self, candidate_id: str) -> Path:
        assert candidate_id == self.candidate.id
        return self.patch


@pytest.mark.asyncio
async def test_evolution_control_client_is_authenticated_and_source_only(
    tmp_path: Path,
) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_text("diff --git a/a b/a\n", encoding="utf-8")
    service = _EvolutionService(patch)
    app = FastAPI()
    token = "t" * 48
    register_evolution_control_api(app, service=service, token=token)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = EvolutionClient(
            base_url="http://bootstrap/bootstrap/internal/v1/evolution",
            token=token,
            review_cache_root=tmp_path / "reviews",
            client=http_client,
        )
        await client.start()
        audit = {"tenant_id": "owner", "thread_id": "thread-1"}

        candidate = await client.get_candidate("candidate_test")
        source_status = await client.source_status(audit_context=audit)
        source_env = await client.source_set_runtime_env(
            name="TELEGRAM_BOT_TOKEN",
            value="raw-token-value",
            idempotency_key="source-env-1",
            audit_context=audit,
        )
        unauthorized = await http_client.get(
            "http://bootstrap/bootstrap/internal/v1/evolution/candidates"
        )

    assert candidate == service.candidate
    assert source_status["candidate_id"] == "candidate_test"
    assert source_env == {
        "status": "updated",
        "name": "TELEGRAM_BOT_TOKEN",
        "changed": True,
        "restarted": True,
        "value": "[set]",
    }
    assert unauthorized.status_code == 401
    assert service.source_calls == [
        ("status", {"audit_context": audit}),
        (
            "runtime-env",
            {
                "name": "TELEGRAM_BOT_TOKEN",
                "value": "raw-token-value",
                "idempotency_key": "source-env-1",
                "audit_context": audit,
            },
        ),
    ]
