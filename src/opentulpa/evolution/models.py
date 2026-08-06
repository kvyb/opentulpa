"""Immutable records for candidate evolution and release provenance."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return value.astimezone(UTC)


class _EvolutionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CandidateStatus(StrEnum):
    """Durable candidate lifecycle states."""

    BUILDING = "building"
    FAILED = "failed"
    READY = "ready"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


class PromotionAttemptStatus(StrEnum):
    """Durable coordination state before release history becomes active."""

    QUEUED = "queued"
    ACTIVATING = "activating"
    ACTIVE = "active"
    FAILED = "failed"


class SourceReleaseOperationStatus(StrEnum):
    """Durable state for one idempotent interactive source release."""

    PENDING = "pending"
    COMPLETED = "completed"


class EvaluationCheck(_EvolutionModel):
    """One reproducible check in a candidate evaluation."""

    name: str = Field(min_length=1, max_length=200)
    passed: bool
    required: bool = True
    summary: str = Field(default="", max_length=4_000)
    details: dict[str, JsonValue] = Field(default_factory=dict)
    duration_seconds: float | None = Field(default=None, ge=0)
    completed_at: datetime = Field(default_factory=_utc_now)

    @field_validator("completed_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value, label="evaluation check timestamp")


class EvaluationReport(_EvolutionModel):
    """Append-only evidence produced by one evaluator run."""

    id: str = Field(
        default_factory=lambda: f"eval_{uuid4().hex}",
        min_length=1,
        max_length=100,
    )
    candidate_id: str = Field(min_length=1, max_length=100)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluator_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    evaluator_version: str = Field(min_length=1, max_length=200)
    passed: bool
    checks: tuple[EvaluationCheck, ...] = Field(min_length=1, max_length=1_000)
    summary: str = Field(default="", max_length=10_000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    evaluated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("evaluated_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value, label="evaluation report timestamp")

    @model_validator(mode="after")
    def _passing_report_has_no_required_failures(self) -> Self:
        if self.passed and any(check.required and not check.passed for check in self.checks):
            raise ValueError("a passing report cannot contain a failed required check")
        return self


class ContributionMetadata(_EvolutionModel):
    """Sanitized upstream-contribution lineage for a local candidate."""

    upstream_repository: str = Field(min_length=1, max_length=2_000)
    base_commit: str = Field(min_length=1, max_length=200)
    branch_name: str = Field(min_length=1, max_length=500)
    head_commit: str | None = Field(default=None, min_length=1, max_length=200)
    pull_request_url: str | None = Field(default=None, min_length=1, max_length=2_000)
    pull_request_number: int | None = Field(default=None, ge=1)
    sanitized: bool = False
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    prepared_at: datetime = Field(default_factory=_utc_now)

    @field_validator("prepared_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value, label="contribution timestamp")

    @model_validator(mode="after")
    def _pull_request_fields_move_together(self) -> Self:
        if (self.pull_request_url is None) != (self.pull_request_number is None):
            raise ValueError("pull request URL and number must be recorded together")
        return self


class Candidate(_EvolutionModel):
    """Versioned snapshot of an isolated self-improvement candidate."""

    id: str = Field(min_length=1, max_length=100)
    base_commit: str = Field(min_length=1, max_length=200)
    requested_improvement: str = Field(min_length=1, max_length=20_000)
    status: CandidateStatus = CandidateStatus.BUILDING
    revision: int = Field(default=1, ge=1)
    parent_candidate_id: str | None = Field(default=None, min_length=1, max_length=100)
    source_commit: str | None = Field(default=None, min_length=1, max_length=200)
    worktree_path: str | None = Field(default=None, min_length=1, max_length=4_096)
    # None is the legacy discriminator and stays absent from canonical legacy payloads.
    workspace_kind: Literal["full_repository", "linked_worktree"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    dependency_lock_hash: str | None = Field(default=None, min_length=1, max_length=200)
    artifact_digest: str | None = Field(default=None, min_length=1, max_length=300)
    evaluator_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    evaluation_report: EvaluationReport | None = None
    contribution: ContributionMetadata | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime) -> datetime:
        return _as_utc(value, label="candidate timestamp")

    @model_validator(mode="after")
    def _valid_lineage_and_time(self) -> Self:
        if self.parent_candidate_id == self.id:
            raise ValueError("a candidate cannot be its own parent")
        if self.updated_at < self.created_at:
            raise ValueError("candidate updated_at cannot precede created_at")
        if self.evaluation_report is not None and self.evaluation_report.candidate_id != self.id:
            raise ValueError("evaluation report belongs to a different candidate")
        return self


class Release(_EvolutionModel):
    """One immutable activation in release history."""

    id: str = Field(
        default_factory=lambda: f"release_{uuid4().hex}",
        min_length=1,
        max_length=100,
    )
    candidate_id: str = Field(min_length=1, max_length=100)
    source_commit: str = Field(min_length=1, max_length=200)
    artifact_digest: str = Field(min_length=1, max_length=300)
    previous_release_id: str | None = Field(default=None, min_length=1, max_length=100)
    reason: str = Field(default="", max_length=4_000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    promoted_at: datetime = Field(default_factory=_utc_now)

    @field_validator("promoted_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime) -> datetime:
        return _as_utc(value, label="release timestamp")

    @model_validator(mode="after")
    def _not_its_own_predecessor(self) -> Self:
        if self.previous_release_id == self.id:
            raise ValueError("a release cannot be its own predecessor")
        return self


class PromotionAttempt(_EvolutionModel):
    """One resumable attempt to activate a built release through bootstrap."""

    id: str = Field(
        default_factory=lambda: f"promotion_{uuid4().hex}",
        min_length=1,
        max_length=100,
    )
    candidate_id: str = Field(min_length=1, max_length=100)
    candidate_revision: int = Field(ge=1)
    release: Release
    status: PromotionAttemptStatus = PromotionAttemptStatus.QUEUED
    revision: int = Field(default=1, ge=1)
    bootstrap_activation_id: str | None = Field(default=None, min_length=1, max_length=100)
    origin: dict[str, JsonValue] = Field(default_factory=dict)
    failure_code: str | None = Field(default=None, min_length=1, max_length=100)
    failure_message: str | None = Field(default=None, min_length=1, max_length=2_000)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime) -> datetime:
        return _as_utc(value, label="promotion attempt timestamp")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.release.candidate_id != self.candidate_id:
            raise ValueError("promotion release belongs to a different candidate")
        if self.updated_at < self.created_at:
            raise ValueError("promotion attempt updated_at cannot precede created_at")
        if (self.failure_code is None) != (self.failure_message is None):
            raise ValueError("promotion failure code and message must be recorded together")
        if self.status is PromotionAttemptStatus.FAILED and self.failure_code is None:
            raise ValueError("failed promotion attempts require a sanitized failure")
        if self.status is not PromotionAttemptStatus.FAILED and self.failure_code is not None:
            raise ValueError("only failed promotion attempts can carry a failure")
        return self


class SourceReleaseOperation(_EvolutionModel):
    """One source-release request owned and replayed by the stable supervisor."""

    id: str = Field(min_length=1, max_length=100)
    tenant_id: str = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=200)
    candidate_id: str = Field(min_length=1, max_length=100)
    expected_diff_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_release_id: str | None = Field(default=None, min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=500)
    audit_context: dict[str, JsonValue] = Field(default_factory=dict)
    status: SourceReleaseOperationStatus = SourceReleaseOperationStatus.PENDING
    promotion_attempt_id: str | None = Field(default=None, min_length=1, max_length=100)
    result: dict[str, JsonValue] | None = None
    revision: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _utc_timestamps(cls, value: datetime) -> datetime:
        return _as_utc(value, label="source release operation timestamp")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.updated_at < self.created_at:
            raise ValueError("source release operation updated_at cannot precede created_at")
        if self.status is SourceReleaseOperationStatus.PENDING and self.result is not None:
            raise ValueError("a pending source release operation cannot have a result")
        if self.status is SourceReleaseOperationStatus.COMPLETED and self.result is None:
            raise ValueError("a completed source release operation requires a result")
        if self.promotion_attempt_id is not None and self.result is None:
            raise ValueError("a promotion attempt can only be bound with a completed result")
        return self


class EvolutionEvent(_EvolutionModel):
    """Durable owner notification emitted after asynchronous evolution work."""

    id: str = Field(
        default_factory=lambda: f"evolution_event_{uuid4().hex}",
        min_length=1,
        max_length=100,
    )
    event_key: str = Field(min_length=1, max_length=300)
    event_type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,99}$")
    candidate_id: str = Field(min_length=1, max_length=100)
    origin: dict[str, JsonValue] = Field(default_factory=dict)
    payload: dict[str, JsonValue]
    status: str = Field(default="pending", pattern=r"^(?:pending|delivered)$")
    attempt_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_utc_now)
    delivered_at: datetime | None = None

    @field_validator("created_at", "delivered_at")
    @classmethod
    def _event_timestamps(cls, value: datetime | None) -> datetime | None:
        return _as_utc(value, label="evolution event timestamp") if value is not None else None

    @model_validator(mode="after")
    def _delivery_is_consistent(self) -> Self:
        if (self.status == "delivered") != (self.delivered_at is not None):
            raise ValueError("delivered evolution events require delivered_at")
        return self


__all__ = [
    "Candidate",
    "CandidateStatus",
    "ContributionMetadata",
    "EvaluationCheck",
    "EvaluationReport",
    "EvolutionEvent",
    "PromotionAttempt",
    "PromotionAttemptStatus",
    "Release",
    "SourceReleaseOperation",
    "SourceReleaseOperationStatus",
]
