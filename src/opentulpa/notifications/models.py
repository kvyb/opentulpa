"""Durable, interface-neutral owner notification contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

NotificationName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_.-]{0,99}$",
    ),
]
ApprovalDecision = Literal["approve", "edit", "reject"]


class _NotificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class NotificationOrigin(_NotificationModel):
    """Opaque delivery lineage retained without exposing trusted run context."""

    interface: str | None = Field(default=None, min_length=1, max_length=64)
    source_id: str | None = Field(default=None, min_length=1, max_length=200)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=200)
    message_id: str | None = Field(default=None, min_length=1, max_length=200)
    channel: str | None = Field(default=None, min_length=1, max_length=64)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=300)

    @field_validator(
        "interface",
        "source_id",
        "conversation_id",
        "message_id",
        "channel",
        "correlation_id",
    )
    @classmethod
    def safe_lineage(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ValueError("notification origin contains control characters")
        return value

    @model_validator(mode="after")
    def require_lineage(self) -> NotificationOrigin:
        if not any(
            (
                self.interface,
                self.source_id,
                self.conversation_id,
                self.message_id,
                self.channel,
                self.correlation_id,
            )
        ):
            raise ValueError("notification origin must contain delivery lineage")
        return self


class NotificationApproval(_NotificationModel):
    """Sanitized approval summary that an owner interface can resume."""

    approval_id: str = Field(min_length=1, max_length=300)
    tool_name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="Approval required.", max_length=2_000)
    allowed_decisions: tuple[ApprovalDecision, ...] = Field(min_length=1, max_length=3)

    @field_validator("approval_id", "tool_name")
    @classmethod
    def safe_identifiers(cls, value: str) -> str:
        if any(ord(character) < 32 for character in value):
            raise ValueError("approval identifier contains control characters")
        return value

    @model_validator(mode="after")
    def unique_decisions(self) -> NotificationApproval:
        if len(self.allowed_decisions) != len(set(self.allowed_decisions)):
            raise ValueError("approval decisions must be unique")
        return self


class NotificationWrite(_NotificationModel):
    """Sanitized payload written once under a tenant-scoped dedupe key."""

    kind: NotificationName
    text: str = Field(min_length=1, max_length=50_000)
    status: NotificationName = "info"
    thread_id: str | None = Field(default=None, min_length=1, max_length=8_192)
    run_id: str | None = Field(default=None, min_length=1, max_length=300)
    origin: NotificationOrigin | None = None
    approvals: tuple[NotificationApproval, ...] = Field(default=(), max_length=100)

    @field_validator("run_id")
    @classmethod
    def safe_identifiers(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 for character in value):
            raise ValueError("notification identifier contains control characters")
        return value

    @model_validator(mode="after")
    def approvals_require_run(self) -> NotificationWrite:
        if self.approvals and not self.run_id:
            raise ValueError("approval notifications require a run_id")
        return self


class OwnerNotification(NotificationWrite):
    """One immutable notification ordered by its SQLite monotonic identifier."""

    id: int = Field(ge=1)
    tenant_id: str = Field(min_length=1, max_length=200)
    dedupe_key: str = Field(min_length=1, max_length=300)
    created_at: datetime

    @field_validator("created_at")
    @classmethod
    def created_at_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        return value.astimezone(UTC)


__all__ = [
    "ApprovalDecision",
    "NotificationApproval",
    "NotificationName",
    "NotificationOrigin",
    "NotificationWrite",
    "OwnerNotification",
]
