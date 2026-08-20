"""Persisted contracts for the immutable OpenTulpa bootstrap."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

_CONTROL_PATH_PATTERN = r"^/[A-Za-z0-9._~!$&'()*+,;=:@%/-]{1,200}$"
_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$"


def utc_now() -> datetime:
    return datetime.now(UTC)


def _utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must include a UTC offset")
    return value.astimezone(UTC)


class BootstrapModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ReleaseOrigin(BootstrapModel):
    """Conversation to receive durable release progress."""

    tenant_id: str = Field(min_length=1, max_length=200)
    actor_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=300)
    run_id: str | None = Field(default=None, min_length=1, max_length=200)
    channel: str = Field(min_length=1, max_length=50)
    correlation_id: str = Field(min_length=1, max_length=200)


class ReleaseRecord(BootstrapModel):
    """Immutable, content-addressed release accepted by the bootstrap."""

    id: str = Field(
        default_factory=lambda: f"release_{uuid4().hex}",
        min_length=1,
        max_length=100,
        pattern=_IDENTIFIER_PATTERN,
    )
    candidate_id: str = Field(min_length=1, max_length=100, pattern=_IDENTIFIER_PATTERN)
    source_commit: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    protocol_version: int = Field(default=1, ge=1, le=1)
    agent_api_version: str = Field(default="2", pattern=r"^[0-9]+(?:\.[0-9]+)*$")
    control_api_version: int = Field(default=1, ge=1, le=1)
    control_port: int = Field(default=8000, ge=1024, le=65535)
    health_path: str = Field(default="/_control/v1/health", pattern=_CONTROL_PATH_PATTERN)
    drain_path: str = Field(default="/_control/v1/drain", pattern=_CONTROL_PATH_PATTERN)
    ingress_path: str = Field(default="/_control/v1/ingress", pattern=_CONTROL_PATH_PATTERN)
    event_path: str = Field(default="/_control/v1/events", pattern=_CONTROL_PATH_PATTERN)
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at")
    @classmethod
    def _created_at_utc(cls, value: datetime) -> datetime:
        return _utc(value, label="release created_at")

    @field_validator("entrypoint")
    @classmethod
    def _entrypoint_is_exec_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or "\x00" in item or len(item) > 4_096 for item in value):
            raise ValueError("entrypoint must contain non-empty exec arguments")
        return value


class IngressEnvelope(BootstrapModel):
    id: str = Field(
        default_factory=lambda: f"ingress_{uuid4().hex}", min_length=1, max_length=100
    )
    tenant_id: str = Field(min_length=1, max_length=200)
    thread_id: str = Field(min_length=1, max_length=300)
    channel: str = Field(min_length=1, max_length=50)
    idempotency_key: str = Field(min_length=1, max_length=300)
    payload: dict[str, JsonValue]
    status: Literal["pending", "claimed", "processed"] = "pending"
    claimed_epoch: int | None = Field(default=None, ge=1)
    attempt_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _ingress_timestamps_utc(cls, value: datetime) -> datetime:
        return _utc(value, label="ingress timestamp")


class OutboxEvent(BootstrapModel):
    id: str = Field(
        default_factory=lambda: f"event_{uuid4().hex}", min_length=1, max_length=100
    )
    event_key: str = Field(min_length=1, max_length=300)
    event_type: str = Field(min_length=1, max_length=100)
    origin: ReleaseOrigin | None = None
    payload: dict[str, JsonValue]
    status: Literal["pending", "delivered"] = "pending"
    attempt_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    delivered_at: datetime | None = None

    @field_validator("created_at", "delivered_at")
    @classmethod
    def _outbox_timestamps_utc(cls, value: datetime | None) -> datetime | None:
        return _utc(value, label="outbox timestamp") if value is not None else None


class ReleaseHealth(BootstrapModel):
    healthy: bool
    release_id: str = Field(min_length=1, max_length=100)
    protocol_version: int = Field(default=1, ge=1)
    summary: str = Field(default="", max_length=2_000)
    components: dict[str, bool] = Field(default_factory=dict)


class DrainResult(BootstrapModel):
    drained: bool
    in_flight: int = Field(default=0, ge=0)


__all__ = [
    "DrainResult",
    "IngressEnvelope",
    "OutboxEvent",
    "ReleaseHealth",
    "ReleaseOrigin",
    "ReleaseRecord",
    "utc_now",
]
