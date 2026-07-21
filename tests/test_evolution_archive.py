from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from opentulpa.evolution.archive import (
    CandidateConflictError,
    EvaluationAlreadyExistsError,
    EvolutionArchive,
    EvolutionArchiveNotStartedError,
    InvalidCandidateTransitionError,
    ReleaseConflictError,
    SourceReleaseOperationConflictError,
)
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    ContributionMetadata,
    EvaluationCheck,
    EvaluationReport,
    PromotionAttempt,
    PromotionAttemptStatus,
    Release,
    SourceReleaseOperation,
    SourceReleaseOperationStatus,
)

NOW = datetime(2026, 7, 20, 8, tzinfo=UTC)
EVALUATOR_FINGERPRINT = "sha256:" + "e" * 64


@pytest.mark.asyncio
async def test_source_release_operation_is_durable_exclusive_and_replayable(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "evolution.sqlite"
    archive = EvolutionArchive(db_path)
    await archive.start()
    candidate = await archive.create_candidate(_candidate("candidate-source-op"))
    operation = SourceReleaseOperation(
        id="source_release_test",
        tenant_id="owner",
        idempotency_key="release-1",
        candidate_id=candidate.id,
        expected_diff_sha256="d" * 64,
        message="Improve source",
    )
    await archive.create_source_release_operation(operation)

    assert (
        await archive.get_source_release_operation(
            tenant_id="owner",
            idempotency_key="release-1",
        )
        == operation
    )
    assert await archive.get_pending_source_release_operation(candidate.id) == operation
    with pytest.raises(SourceReleaseOperationConflictError, match="candidate is busy"):
        await archive.create_source_release_operation(
            operation.model_copy(
                update={
                    "id": "source_release_other",
                    "idempotency_key": "release-2",
                }
            )
        )

    result = {"active": False, "candidate": {"id": candidate.id}, "promotion": None}
    completed = await archive.complete_source_release_operation(
        operation.id,
        expected_revision=operation.revision,
        result=result,
    )
    assert completed.status is SourceReleaseOperationStatus.COMPLETED
    assert completed.result == result
    assert await archive.complete_source_release_operation(
        operation.id,
        expected_revision=operation.revision,
        result=result,
    ) == completed
    await archive.shutdown()

    restarted = EvolutionArchive(db_path)
    await restarted.start()
    assert await restarted.get_source_release_operation(
        tenant_id="owner",
        idempotency_key="release-1",
    ) == completed
    assert await restarted.list_pending_source_release_operations() == []
    await restarted.shutdown()


@pytest.mark.asyncio
async def test_archive_cancels_legacy_pending_approval_during_upgrade(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "evolution.sqlite"
    archive = EvolutionArchive(db_path)
    await archive.start()
    candidate = await archive.create_candidate(_candidate("candidate-legacy-approval"))
    attempt = PromotionAttempt(
        id="promotion-legacy-approval",
        candidate_id=candidate.id,
        candidate_revision=candidate.revision,
        release=Release(
            id="release-legacy-approval",
            candidate_id=candidate.id,
            source_commit="b" * 40,
            artifact_digest="sha256:" + "d" * 64,
        ),
    )
    await archive.create_promotion_attempt(attempt)
    await archive.shutdown()

    legacy_payload = attempt.model_dump(mode="json")
    legacy_payload.update({"approval_status": "pending", "approval_audit": {}})
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE evolution_promotion_attempts SET payload_json = ? WHERE id = ?",
            (json.dumps(legacy_payload), attempt.id),
        )

    restarted = EvolutionArchive(db_path)
    await restarted.start()
    migrated = await restarted.get_promotion_attempt(attempt.id)
    assert migrated is not None
    assert migrated.status is PromotionAttemptStatus.FAILED
    assert migrated.failure_code == "obsolete_approval_state"
    assert "approval_status" not in migrated.model_dump()
    await restarted.shutdown()


def _candidate(candidate_id: str, **changes: object) -> Candidate:
    values: dict[str, object] = {
        "id": candidate_id,
        "base_commit": "a" * 40,
        "requested_improvement": "Add a useful status website",
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(changes)
    return Candidate.model_validate(values)


def _report(
    candidate_id: str,
    *,
    report_id: str = "eval-1",
    source_commit: str = "b" * 40,
    artifact_digest: str = "sha256:" + "d" * 64,
) -> EvaluationReport:
    return EvaluationReport(
        id=report_id,
        candidate_id=candidate_id,
        source_commit=source_commit,
        artifact_digest=artifact_digest,
        evaluator_fingerprint=EVALUATOR_FINGERPRINT,
        evaluator_version="canonical-v1",
        passed=True,
        checks=(
            EvaluationCheck(
                name="regression suite",
                passed=True,
                details={"tests": 42},
                completed_at=NOW + timedelta(minutes=1),
            ),
        ),
        summary="All required checks passed",
        metadata={"runner": "isolated"},
        evaluated_at=NOW + timedelta(minutes=1),
    )


def test_evolution_models_are_frozen_strict_and_normalize_timestamps_to_utc() -> None:
    offset = timezone(timedelta(hours=3))
    candidate = _candidate(
        "candidate-1",
        created_at=NOW.astimezone(offset),
        updated_at=NOW.astimezone(offset),
        contribution=ContributionMetadata(
            upstream_repository="https://github.com/example/opentulpa",
            base_commit="a" * 40,
            branch_name="candidate/status-site",
            prepared_at=NOW.astimezone(offset),
        ),
    )

    assert candidate.created_at.tzinfo is UTC
    assert candidate.contribution is not None
    assert candidate.contribution.prepared_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="frozen"):
        candidate.status = CandidateStatus.READY
    with pytest.raises(ValidationError, match="UTC offset"):
        _candidate("candidate-naive", created_at=datetime(2026, 7, 20, 8))
    with pytest.raises(ValidationError, match="failed required check"):
        EvaluationReport(
            candidate_id="candidate-1",
            source_commit="b" * 40,
            artifact_digest="sha256:" + "d" * 64,
            evaluator_fingerprint=EVALUATOR_FINGERPRINT,
            evaluator_version="canonical-v1",
            passed=True,
            checks=(EvaluationCheck(name="security", passed=False),),
        )
    with pytest.raises(ValidationError, match="different candidate"):
        _candidate("candidate-1", evaluation_report=_report("candidate-2"))
    with pytest.raises(ValidationError, match="recorded together"):
        ContributionMetadata(
            upstream_repository="https://github.com/example/opentulpa",
            base_commit="a" * 40,
            branch_name="candidate/status-site",
            pull_request_url="https://github.com/example/opentulpa/pull/1",
        )


@pytest.mark.asyncio
async def test_archive_persists_json_evidence_and_uses_optimistic_transitions(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "evolution.sqlite"
    clock_values = iter(
        [
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=2),
            NOW + timedelta(minutes=3),
        ]
    )
    archive = EvolutionArchive(db_path, clock=lambda: next(clock_values))
    with pytest.raises(EvolutionArchiveNotStartedError):
        await archive.get_candidate("candidate-1")
    await archive.start()

    created = await archive.create_candidate(
        _candidate("candidate-1", metadata={"label": "local candidate"})
    )
    assert await archive.get_candidate(created.id) == created
    assert await archive.list_candidates(status="building") == [created]
    assert await archive.list_candidates(status="ready") == []
    with pytest.raises(InvalidCandidateTransitionError, match="passing evaluation"):
        await archive.transition_status(
            created.id,
            expected_status=CandidateStatus.BUILDING,
            new_status=CandidateStatus.READY,
            expected_revision=1,
        )

    changed = created.model_copy(
        update={
            "source_commit": "b" * 40,
            "dependency_lock_hash": "sha256:" + "c" * 64,
            "artifact_digest": "sha256:" + "d" * 64,
            "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        }
    )
    updated = await archive.update_candidate(changed, expected_revision=1)
    assert updated.revision == 2
    assert updated.source_commit == "b" * 40
    with pytest.raises(InvalidCandidateTransitionError, match="passing evaluation"):
        await archive.transition_status(
            created.id,
            expected_status=CandidateStatus.BUILDING,
            new_status=CandidateStatus.READY,
            expected_revision=2,
        )
    with pytest.raises(CandidateConflictError, match="revision"):
        await archive.update_candidate(
            created.model_copy(update={"source_commit": "e" * 40}),
            expected_revision=1,
        )

    report = _report(created.id)
    evaluated = await archive.append_evaluation(report, expected_revision=2)
    assert evaluated.revision == 3
    assert evaluated.evaluation_report == report
    assert await archive.list_evaluations(created.id) == [report]
    with pytest.raises(EvaluationAlreadyExistsError):
        await archive.append_evaluation(report)

    ready = await archive.transition_status(
        created.id,
        expected_status=CandidateStatus.BUILDING,
        new_status=CandidateStatus.READY,
        expected_revision=3,
    )
    assert ready.status is CandidateStatus.READY
    assert ready.revision == 4
    with pytest.raises(CandidateConflictError, match="status"):
        await archive.transition_status(
            created.id,
            expected_status=CandidateStatus.BUILDING,
            new_status=CandidateStatus.FAILED,
        )
    with pytest.raises(InvalidCandidateTransitionError):
        await archive.transition_status(
            created.id,
            expected_status=CandidateStatus.READY,
            new_status=CandidateStatus.ROLLED_BACK,
        )

    unsafe = _candidate("candidate-unsafe", metadata={"score": float("nan")})
    with pytest.raises(ValueError, match="JSON compliant"):
        await archive.create_candidate(unsafe)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        stored = connection.execute(
            "SELECT payload_json FROM evolution_candidates WHERE id = ?",
            (created.id,),
        ).fetchone()
        assert stored is not None
        assert isinstance(stored[0], str)
        assert stored[0].startswith("{")

    await archive.shutdown()
    with pytest.raises(EvolutionArchiveNotStartedError):
        await archive.get_candidate(created.id)

    restarted = EvolutionArchive(db_path)
    await restarted.start()
    durable = await restarted.get_candidate(created.id)
    assert durable is not None
    assert durable.status is CandidateStatus.READY
    assert durable.evaluation_report == report
    await restarted.shutdown()


@pytest.mark.asyncio
async def test_release_history_tracks_current_release_and_rollback_target(tmp_path: Path) -> None:
    archive = EvolutionArchive(tmp_path / "evolution.sqlite")
    await archive.start()
    first_candidate = await archive.create_candidate(_candidate("candidate-1"))
    first_candidate = await archive.update_candidate(
        first_candidate.model_copy(
            update={
                "source_commit": "b" * 40,
                "artifact_digest": "sha256:" + "1" * 64,
                "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
            }
        )
    )
    first_candidate = await archive.append_evaluation(
        _report(
            first_candidate.id,
            report_id="eval-first",
            source_commit=first_candidate.source_commit or "",
            artifact_digest=first_candidate.artifact_digest or "",
        )
    )
    first_candidate = await archive.transition_status(
        first_candidate.id,
        expected_status=CandidateStatus.BUILDING,
        new_status=CandidateStatus.READY,
        expected_revision=first_candidate.revision,
    )
    first_candidate, first = await archive.promote_candidate(
        Release(
            id="release-1",
            candidate_id=first_candidate.id,
            source_commit=first_candidate.source_commit or "",
            artifact_digest=first_candidate.artifact_digest or "",
            promoted_at=NOW,
        ),
        expected_revision=first_candidate.revision,
    )
    assert first_candidate.status is CandidateStatus.PROMOTED
    assert first.previous_release_id is None
    assert await archive.get_current_release() == first
    assert await archive.get_rollback_target() is None

    second_candidate = await archive.create_candidate(_candidate("candidate-2"))
    second_candidate = await archive.update_candidate(
        second_candidate.model_copy(
            update={
                "source_commit": "c" * 40,
                "artifact_digest": "sha256:" + "2" * 64,
                "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
            }
        )
    )
    second_candidate = await archive.append_evaluation(
        _report(
            second_candidate.id,
            report_id="eval-second",
            source_commit=second_candidate.source_commit or "",
            artifact_digest=second_candidate.artifact_digest or "",
        )
    )
    second_candidate = await archive.transition_status(
        second_candidate.id,
        expected_status=CandidateStatus.BUILDING,
        new_status=CandidateStatus.READY,
        expected_revision=second_candidate.revision,
    )
    second_candidate, second = await archive.promote_candidate(
        Release(
            id="release-2",
            candidate_id=second_candidate.id,
            source_commit=second_candidate.source_commit or "",
            artifact_digest=second_candidate.artifact_digest or "",
            promoted_at=NOW + timedelta(minutes=1),
        ),
        expected_revision=second_candidate.revision,
    )
    assert second_candidate.status is CandidateStatus.PROMOTED
    assert second.previous_release_id == first.id
    assert await archive.current_release() == second
    assert await archive.rollback_target() == first
    assert await archive.release_history() == [second, first]

    with pytest.raises(ReleaseConflictError, match="predecessor"):
        await archive.record_promotion(
            Release(
                id="release-3",
                candidate_id=first_candidate.id,
                source_commit=first_candidate.source_commit or "",
                artifact_digest=first_candidate.artifact_digest or "",
                previous_release_id="release-1",
                promoted_at=NOW + timedelta(minutes=2),
            )
        )
    assert await archive.get_current_release() == second
    assert await archive.list_release_history() == [second, first]

    with pytest.raises(ReleaseConflictError, match="target release artifact"):
        await archive.activate_rollback(
            first.id,
            activation=Release(
                id="release-bad-rollback",
                candidate_id=first_candidate.id,
                source_commit=first.source_commit,
                artifact_digest="sha256:" + "9" * 64,
                promoted_at=NOW + timedelta(minutes=2),
            ),
            expected_current_release_id=second.id,
        )
    still_promoted = await archive.get_candidate(second_candidate.id)
    assert still_promoted is not None
    assert still_promoted.status is CandidateStatus.PROMOTED
    assert await archive.get_current_release() == second

    rolled_back, rollback_activation = await archive.activate_rollback(
        first.id,
        activation=Release(
            id="release-rollback-1",
            candidate_id=first_candidate.id,
            source_commit=first.source_commit,
            artifact_digest=first.artifact_digest,
            promoted_at=NOW + timedelta(minutes=2),
            reason="health check failed",
        ),
        expected_current_release_id=second.id,
        expected_current_candidate_revision=second_candidate.revision,
    )
    assert rolled_back.id == second_candidate.id
    assert rolled_back.status is CandidateStatus.ROLLED_BACK
    assert rollback_activation.candidate_id == first_candidate.id
    assert rollback_activation.previous_release_id == second.id
    assert await archive.get_current_release() == rollback_activation
    assert await archive.get_rollback_target() == second
    assert await archive.list_release_history() == [rollback_activation, second, first]

    await archive.shutdown()
    restarted = EvolutionArchive(tmp_path / "evolution.sqlite")
    await restarted.start()
    assert await restarted.get_current_release() == rollback_activation
    assert await restarted.get_rollback_target() == second
    durable_rolled_back = await restarted.get_candidate(second_candidate.id)
    assert durable_rolled_back is not None
    assert durable_rolled_back.status is CandidateStatus.ROLLED_BACK
    await restarted.shutdown()


@pytest.mark.asyncio
async def test_ready_transition_rejects_evaluation_for_different_source(tmp_path: Path) -> None:
    archive = EvolutionArchive(tmp_path / "evolution.sqlite")
    await archive.start()
    candidate = await archive.create_candidate(_candidate("candidate-bound"))
    candidate = await archive.update_candidate(
        candidate.model_copy(
            update={
                "source_commit": "b" * 40,
                "artifact_digest": "sha256:" + "d" * 64,
                "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
            }
        )
    )
    candidate = await archive.append_evaluation(
        _report(
            candidate.id,
            source_commit="c" * 40,
            artifact_digest=candidate.artifact_digest or "",
        )
    )

    with pytest.raises(InvalidCandidateTransitionError, match="source commit"):
        await archive.transition_status(
            candidate.id,
            expected_status=CandidateStatus.BUILDING,
            new_status=CandidateStatus.READY,
            expected_revision=candidate.revision,
        )

    await archive.shutdown()
