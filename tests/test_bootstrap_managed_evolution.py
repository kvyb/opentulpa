from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from opentulpa.bootstrap.evolution_api import EvolutionClient, register_evolution_control_api
from opentulpa.bootstrap.evolution_runtime import ManagedEvolutionRuntime
from opentulpa.bootstrap.gateway import (
    ActiveReleaseTransport,
    BootstrapGateway,
    create_gateway_app,
)
from opentulpa.bootstrap.host import InMemoryReleaseHost
from opentulpa.bootstrap.models import ReleaseRecord
from opentulpa.bootstrap.store import BootstrapStore
from opentulpa.bootstrap.supervisor import BootstrapSupervisor, SupervisorPolicy
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.models import (
    Candidate,
    PromotionAttempt,
    Release,
)


def _release() -> ReleaseRecord:
    return ReleaseRecord(
        id="release_initial_test",
        candidate_id="bootstrap_initial_test",
        source_commit="a" * 40,
        artifact_digest=f"sha256:{'b' * 64}",
        manifest_digest=f"sha256:{'c' * 64}",
        entrypoint=("./start.sh", "run", "server"),
        metadata={
            "dependency_lock_hash": "d" * 64,
            "evaluator_fingerprint": f"sha256:{'e' * 64}",
            "evaluator_version": "test-evaluator-v2",
        },
    )


class _InitialProvider:
    async def build(self) -> ReleaseRecord:
        return _release()


class _EvolutionLifecycle:
    def __init__(self) -> None:
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.started = False


@pytest.mark.asyncio
async def test_managed_evolution_installs_and_seeds_first_release(tmp_path: Path) -> None:
    host = InMemoryReleaseHost()
    bootstrap = BootstrapSupervisor(
        store=BootstrapStore(tmp_path / "bootstrap.db"),
        host=host,
        policy=SupervisorPolicy(
            production_probe_attempts=1,
            probe_interval_seconds=0,
            probation_seconds=0,
            probation_probe_interval_seconds=1,
        ),
    )
    await bootstrap.start()
    archive = EvolutionArchive(tmp_path / "evolution.db")
    evolution = _EvolutionLifecycle()
    runtime = ManagedEvolutionRuntime(
        bootstrap=bootstrap,
        archive=archive,
        evolution=evolution,  # type: ignore[arg-type]
        initial_release=_InitialProvider(),
    )

    await runtime.start()

    assert bootstrap.store.get_state().serving_release_id == _release().id
    assert evolution.started is True
    current = await archive.get_current_release()
    assert current is not None
    assert current.id == _release().id
    assert current.metadata["base_commit"] == "a" * 40
    assert current.metadata["changed_paths"] == []
    assert current.metadata["diff_sha256"] == hashlib.sha256(b"").hexdigest()
    assert current.metadata["evaluation_report_id"]
    assert current.metadata["evaluator_fingerprint"] == f"sha256:{'e' * 64}"
    candidate = await archive.get_candidate(_release().candidate_id)
    assert candidate is not None
    assert candidate.evaluation_report is not None
    assert candidate.evaluation_report.evaluator_fingerprint == f"sha256:{'e' * 64}"
    assert candidate.evaluation_report.evaluator_version == "test-evaluator-v2"
    await runtime.shutdown()
    await archive.shutdown()


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


class _ManagedFacade:
    def __init__(self, service: _EvolutionService) -> None:
        self.service = service

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_evolution_control_client_is_typed_authenticated_and_digest_checked(
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

        candidate = await client.get_candidate("candidate_test")
        assert candidate is not None
        candidates = await client.list_candidates()
        downloaded = await client.review_patch(candidate.id)
        audit = {"tenant_id": "owner", "thread_id": "thread-1"}
        source_status = await client.source_status(audit_context=audit)
        source_shell = await client.source_shell(
            command="pytest -q",
            timeout_seconds=120,
            audit_context=audit,
        )
        source_release = await client.source_release(
            idempotency_key="source-release-1",
            expected_candidate_id="candidate_test",
            expected_diff_sha256="d" * 64,
            message="Improve source",
            audit_context=audit,
        )
        source_rollback = await client.source_rollback(
            idempotency_key="source-rollback-1",
            expected_current_release_id="release-current",
            expected_target_release_id="release-prior",
            reason="Undo source release",
            audit_context=audit,
        )
        promotion = await client.get_promotion_attempt(source_rollback.id)
        contribution = await client.prepare_contribution(
            candidate.id,
            expected_revision=candidate.revision,
            audit_context=audit,
        )

        assert candidate == service.candidate
        assert candidates == [service.candidate]
        assert downloaded.read_bytes() == patch.read_bytes()
        assert source_status["candidate_id"] == "candidate_test"
        assert source_shell["output"] == "tests passed\n"
        assert source_release["candidate"]["status"] == "ready"
        assert source_rollback.candidate_id == service.candidate.id
        assert promotion == source_rollback
        assert contribution == candidate
        assert service.source_calls == [
            ("status", {"audit_context": audit}),
            (
                "shell",
                {
                    "command": "pytest -q",
                    "timeout_seconds": 120,
                    "audit_context": audit,
                },
            ),
            (
                "release",
                {
                    "idempotency_key": "source-release-1",
                    "expected_candidate_id": "candidate_test",
                    "expected_diff_sha256": "d" * 64,
                    "message": "Improve source",
                    "audit_context": audit,
                },
            ),
            (
                "rollback",
                {
                    "idempotency_key": "source-rollback-1",
                    "expected_current_release_id": "release-current",
                    "expected_target_release_id": "release-prior",
                    "reason": "Undo source release",
                    "audit_context": audit,
                },
            ),
        ]
        unauthorized = await http_client.get(
            "http://bootstrap/bootstrap/internal/v1/evolution/candidates"
        )
        assert unauthorized.status_code == 401


@pytest.mark.asyncio
async def test_internal_control_exposes_only_source_mutations_and_audit(tmp_path: Path) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_text("diff --git a/a b/a\n", encoding="utf-8")
    evolution = _EvolutionService(patch)
    store = BootstrapStore(tmp_path / "bootstrap.db")
    host = InMemoryReleaseHost()
    transport = ActiveReleaseTransport(store=store, host=host)
    bootstrap = BootstrapSupervisor(store=store, host=host)
    gateway = BootstrapGateway(store=store, transport=transport)
    runtime_token = "t" * 48
    app = create_gateway_app(
        supervisor=bootstrap,
        gateway=gateway,
        recovery_token="r" * 48,
        ingress_token="i" * 48,
        managed_evolution=_ManagedFacade(evolution),  # type: ignore[arg-type]
        evolution_token=runtime_token,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://bootstrap",
    ) as client:
        source_status = await client.post(
            "/bootstrap/internal/v1/evolution/source/status",
            headers={"X-OpenTulpa-Evolution-Token": runtime_token},
            json={"audit_context": {"tenant_id": "owner"}},
        )
        removed_create = await client.post(
            "/bootstrap/internal/v1/evolution/candidates",
            headers={"X-OpenTulpa-Evolution-Token": runtime_token},
            json={"instruction": "bypass the owner agent"},
        )
        removed_promotion = await client.post(
            "/bootstrap/internal/v1/evolution/candidates/candidate_test/promotion",
            headers={"X-OpenTulpa-Evolution-Token": runtime_token},
            json={"expected_revision": 1, "reason": "runtime request"},
        )
        removed_reject = await client.post(
            "/bootstrap/internal/v1/evolution/candidates/candidate_test/reject",
            headers={"X-OpenTulpa-Evolution-Token": runtime_token},
            json={"expected_revision": 1},
        )
        removed_rollback = await client.post(
            "/bootstrap/internal/v1/evolution/rollback",
            headers={"X-OpenTulpa-Evolution-Token": runtime_token},
            json={"reason": "runtime rollback request"},
        )

    assert source_status.status_code == 200
    assert removed_create.status_code == 404
    assert removed_promotion.status_code == 404
    assert removed_reject.status_code == 404
    assert removed_rollback.status_code == 404
    assert evolution.source_calls == [
        ("status", {"audit_context": {"tenant_id": "owner"}})
    ]
    await transport.aclose()
