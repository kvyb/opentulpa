"""Typed contracts for deterministic background jobs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
)


class JobArguments(BaseModel):
    """Strict base class for every registered handler's persisted input."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class _JobModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


JobStatus = Literal[
    "queued",
    "running",
    "cancel_requested",
    "succeeded",
    "failed",
    "cancelled",
]

JobEventType = Literal[
    "queued",
    "recovered",
    "running",
    "progress",
    "cancel_requested",
    "cancelled",
    "artifact.ready",
    "completed",
    "failed",
]


class JobError(_JobModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False


class JobArtifactWrite(_JobModel):
    """Artifact metadata returned by a trusted product handler."""

    name: str = Field(min_length=1, max_length=300)
    media_type: str = Field(min_length=1, max_length=200)
    uri: str = Field(min_length=1, max_length=2_000)
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("sha256", mode="before")
    @classmethod
    def _lower_hash(cls, value: Any) -> Any:
        return str(value).strip().lower() if value is not None else None


class JobArtifact(JobArtifactWrite):
    id: str = Field(min_length=1, max_length=100)
    tenant_id: str = Field(min_length=1, max_length=200)
    job_id: str = Field(min_length=1, max_length=100)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("artifact timestamp must include a UTC offset")
        return value


class JobHandlerResult(_JobModel):
    summary: str = Field(default="Completed", min_length=1, max_length=4_000)
    data: dict[str, JsonValue] = Field(default_factory=dict)
    artifacts: list[JobArtifactWrite] = Field(default_factory=list, max_length=100)


class Job(_JobModel):
    id: str = Field(min_length=1, max_length=100)
    tenant_id: str = Field(min_length=1, max_length=200)
    handler_name: str = Field(min_length=1, max_length=100)
    handler_version: int = Field(ge=1)
    status: JobStatus
    arguments: dict[str, JsonValue]
    result: JobHandlerResult | None = None
    error: JobError | None = None
    idempotency_key: str = Field(min_length=1, max_length=200)
    attempt_count: int = Field(ge=0)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @field_validator("created_at", "updated_at", "started_at", "finished_at")
    @classmethod
    def _aware_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("job timestamps must include a UTC offset")
        return value


class JobEvent(_JobModel):
    tenant_id: str = Field(min_length=1, max_length=200)
    job_id: str = Field(min_length=1, max_length=100)
    sequence: int = Field(ge=1)
    event_type: JobEventType
    payload: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamp must include a UTC offset")
        return value


__all__ = [
    "Job",
    "JobArguments",
    "JobArtifact",
    "JobArtifactWrite",
    "JobError",
    "JobEvent",
    "JobEventType",
    "JobHandlerResult",
    "JobStatus",
]
