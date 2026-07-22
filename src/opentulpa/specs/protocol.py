"""Versioned protocol shared by interfaces, triggers, and the agent runtime."""

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

PROTOCOL_VERSION: Literal["1.0"] = "1.0"

ProtocolSlug = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_-]{0,63}$",
    ),
]
ProtocolId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
ProtocolThreadId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=8_192),
]


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class OriginRef(_ProtocolModel):
    """Opaque reply routing supplied by an authenticated interface or trigger."""

    interface: ProtocolSlug
    source_id: ProtocolId
    conversation_id: ProtocolId | None = None
    message_id: ProtocolId | None = None


class AgentSpecRef(_ProtocolModel):
    """Reference one immutable tenant-owned AgentSpec revision."""

    tenant_id: ProtocolId
    spec_id: ProtocolSlug
    revision: int = Field(ge=1)


class AgentRunBinding(_ProtocolModel):
    """Reviewed, immutable agent identity assigned by an authenticated ingress."""

    agent_spec: AgentSpecRef
    run_kind: ProtocolSlug
    trust_class: Literal["owner", "background", "external"]

    @model_validator(mode="after")
    def validate_authority_shape(self) -> AgentRunBinding:
        if (self.trust_class == "owner") != (self.run_kind == "owner"):
            raise ValueError("owner run kind and owner trust must be granted together")
        if self.trust_class != "owner" and self.agent_spec.spec_id == "owner":
            raise ValueError("restricted bindings cannot reference the owner AgentSpec")
        return self


class AgentRunContext(_ProtocolModel):
    """Trusted identity and routing context; interfaces cannot override tenancy."""

    tenant_id: ProtocolId
    actor_id: ProtocolId
    thread_id: ProtocolThreadId
    channel: ProtocolSlug
    run_kind: ProtocolSlug
    correlation_id: ProtocolThreadId
    origin: OriginRef
    agent_spec: AgentSpecRef
    trust_class: Literal["owner", "background", "external"]

    @model_validator(mode="after")
    def validate_agent_spec_tenant(self) -> AgentRunContext:
        if self.agent_spec.tenant_id != self.tenant_id:
            raise ValueError("agent_spec and context must belong to the same tenant")
        return self


class RunSubmission(_ProtocolModel):
    """Universal input accepted from web, interfaces, and deterministic triggers."""

    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    submission_id: ProtocolId
    agent_spec: AgentSpecRef
    context: AgentRunContext
    text: str = Field(min_length=1, max_length=200_000)
    file_ids: tuple[ProtocolId, ...] = Field(default=(), max_length=100)
    idempotency_key: ProtocolId
    submitted_at: datetime

    @field_validator("submitted_at")
    @classmethod
    def validate_submitted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("submitted_at must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_tenant_boundary(self) -> RunSubmission:
        if self.agent_spec != self.context.agent_spec:
            raise ValueError("submission and context must use the same agent_spec revision")
        if len(self.file_ids) != len(set(self.file_ids)):
            raise ValueError("file_ids must be unique")
        return self


__all__ = [
    "PROTOCOL_VERSION",
    "AgentRunContext",
    "AgentRunBinding",
    "AgentSpecRef",
    "OriginRef",
    "RunSubmission",
]
