from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from opentulpa.bootstrap.evolution_api import EvolutionClient, register_evolution_control_api
from opentulpa.bootstrap.evolution_runtime import (
    ManagedEvolutionRuntime,
    TrustedSourceReleaseProvider,
)
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
    EvaluationCheck,
    EvaluationReport,
    PromotionAttempt,
    Release,
)
from opentulpa.evolution.release_builder import OciReleaseArtifact, ReleaseBuildRequest


def _release() -> ReleaseRecord:
    return ReleaseRecord(
        id="release_initial_test",
        candidate_id="bootstrap_initial_test",
        source_commit="a" * 40,
        artifact_digest=f"sha256:{'b' * 64}",
        manifest_digest=f"sha256:{'c' * 64}",
        entrypoint=("./start.sh", "run", "server"),
        metadata={
            "artifact_kind": "oci_image",
            "image_reference": f"registry.example/opentulpa@sha256:{'b' * 64}",
            "dependency_lock_hash": "d" * 64,
            "evaluator_fingerprint": f"sha256:{'e' * 64}",
            "evaluator_version": "test-evaluator-v2",
        },
    )


class _InitialProvider:
    async def build(self) -> ReleaseRecord:
        return _release()


class _RecordingInitialBuilder:
    def __init__(self) -> None:
        self.requests: list[ReleaseBuildRequest] = []

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        self.requests.append(request)
        generation_id = "9" * 64
        return OciReleaseArtifact(
            artifact_kind="python_generation",
            artifact_digest=f"sha256:{'8' * 64}",
            manifest_digest=f"sha256:{'8' * 64}",
            image_reference=f"python-generation:{generation_id}",
            entrypoint=("venv/bin/python", "-I", "-m", "opentulpa"),
        )


@pytest.mark.asyncio
async def test_initial_trusted_release_uses_pre_artifact_evaluation_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    builder = _RecordingInitialBuilder()
    provider = TrustedSourceReleaseProvider(
        source_repository=source,
        builder=builder,
        evaluator_version="initial-v2",
        evaluator_fingerprint=f"sha256:{'e' * 64}",
    )
    source_commit = "a" * 40
    lock_hash = "d" * 64
    monkeypatch.setattr(provider, "_source_commit", lambda: source_commit)
    monkeypatch.setattr(provider, "_lock_hash", lambda: lock_hash)

    release = await provider.build()

    expected = hashlib.sha256(
        f"{source_commit}:{lock_hash}:initial-v2:sha256:{'e' * 64}".encode()
    ).hexdigest()
    assert builder.requests[0].evaluation_input_sha256 == expected
    assert release.metadata["evaluation_input_digest"] == f"sha256:{expected}"
    assert release.metadata["artifact_kind"] == "python_generation"
    assert release.metadata["generation_id"] == "9" * 64


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


@pytest.mark.asyncio
async def test_initial_seed_rejects_existing_candidate_dependency_provenance(
    tmp_path: Path,
) -> None:
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
    await bootstrap.install_initial(_release())
    archive = EvolutionArchive(tmp_path / "evolution.db")
    await archive.start()
    await archive.create_candidate(
        Candidate(
            id=_release().candidate_id,
            base_commit=_release().source_commit,
            requested_improvement="stale initial candidate",
            source_commit=_release().source_commit,
            dependency_lock_hash="f" * 64,
            artifact_digest=_release().artifact_digest,
            evaluator_fingerprint=f"sha256:{'e' * 64}",
        )
    )
    runtime = ManagedEvolutionRuntime(
        bootstrap=bootstrap,
        archive=archive,
        evolution=_EvolutionLifecycle(),  # type: ignore[arg-type]
        initial_release=None,
    )

    with pytest.raises(RuntimeError, match="lineage conflicts"):
        await runtime.start()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_initial_seed_rejects_stale_existing_evaluation_report(tmp_path: Path) -> None:
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
    await bootstrap.install_initial(_release())
    archive = EvolutionArchive(tmp_path / "evolution.db")
    await archive.start()
    candidate = await archive.create_candidate(
        Candidate(
            id=_release().candidate_id,
            base_commit=_release().source_commit,
            requested_improvement="stale initial evaluation",
            source_commit=_release().source_commit,
            dependency_lock_hash="d" * 64,
            artifact_digest=_release().artifact_digest,
            evaluator_fingerprint=f"sha256:{'e' * 64}",
        )
    )
    await archive.append_evaluation(
        EvaluationReport(
            candidate_id=candidate.id,
            source_commit=_release().source_commit,
            artifact_digest=_release().artifact_digest,
            evaluator_fingerprint=f"sha256:{'e' * 64}",
            evaluator_version="stale-evaluator-v1",
            passed=True,
            checks=(
                EvaluationCheck(
                    name="bootstrap.initial_release",
                    passed=True,
                    summary="stale evidence",
                ),
            ),
            summary="stale evidence",
        ),
        expected_revision=candidate.revision,
    )
    runtime = ManagedEvolutionRuntime(
        bootstrap=bootstrap,
        archive=archive,
        evolution=_EvolutionLifecycle(),  # type: ignore[arg-type]
        initial_release=None,
    )

    with pytest.raises(RuntimeError, match="evaluation conflicts"):
        await runtime.start()
    await runtime.shutdown()


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
        source_sync = await client.source_sync_upstream(
            expected_active_release_id="release-current",
            audit_context=audit,
        )
        source_dependencies = await client.source_resolve_dependencies(
            expected_candidate_id="candidate_test",
            expected_diff_sha256="d" * 64,
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
        assert source_sync["upstream_commit"] == "b" * 40
        assert source_dependencies["dependency_base_id"] == "e" * 64
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
                "sync-upstream",
                {
                    "expected_active_release_id": "release-current",
                    "audit_context": audit,
                },
            ),
            (
                "resolve-dependencies",
                {
                    "expected_candidate_id": "candidate_test",
                    "expected_diff_sha256": "d" * 64,
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
