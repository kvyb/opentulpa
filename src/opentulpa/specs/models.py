"""Immutable AgentSpec and TriggerSpec contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from opentulpa.specs.protocol import AgentSpecRef, ProtocolId, ProtocolSlug

SpecIsolation = Literal["private", "external"]
MemoryScope = Literal["owner", "spec", "none"]
WorkspaceScope = Literal["read_write", "read_only", "none"]
ToolPolicy = Literal["profile_default", "allowlist"]
TriggerExposure = Literal["private", "external"]

ToolName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[a-z][a-z0-9_]{0,99}$"),
]

_EXTERNAL_AGENT_TOOLS = frozenset(
    {
        "knowledge_find",
        "knowledge_list",
        "knowledge_query",
    }
)


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AgentSpecWrite(_SpecModel):
    """Behavioral configuration used to create an immutable AgentSpec revision."""

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=4_000)
    runtime_profile: ProtocolSlug = "custom"
    model_alias: str = Field(default="default", min_length=1, max_length=200)
    instructions: str = Field(min_length=1, max_length=200_000)
    isolation: SpecIsolation = "private"
    tool_policy: ToolPolicy = "allowlist"
    tools: tuple[ToolName, ...] = Field(default=(), max_length=200)
    memory_scope: MemoryScope = "spec"
    workspace_scope: WorkspaceScope = "none"
    allow_delegation: bool = False
    max_runtime_seconds: int = Field(default=900, ge=1, le=86_400)
    max_model_calls: int = Field(default=100, ge=1, le=10_000)
    output_schema: dict[str, JsonValue] | None = None
    labels: dict[ProtocolSlug, str] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def validate_isolation(self) -> AgentSpecWrite:
        if len(self.tools) != len(set(self.tools)):
            raise ValueError("tools must be unique")
        if self.isolation == "external":
            unsafe = []
            if self.memory_scope != "none":
                unsafe.append("memory")
            if self.workspace_scope != "none":
                unsafe.append("workspace access")
            if self.allow_delegation:
                unsafe.append("delegation")
            if self.tool_policy != "allowlist":
                unsafe.append("profile-default tools")
            unsafe_tools = sorted(set(self.tools) - _EXTERNAL_AGENT_TOOLS)
            if unsafe_tools:
                unsafe.append("non-knowledge tools: " + ", ".join(unsafe_tools))
            if unsafe:
                raise ValueError("external AgentSpec cannot use " + ", ".join(unsafe))
        if self.tool_policy == "profile_default" and self.tools:
            raise ValueError("profile_default tool policy cannot include an allowlist")
        return self


class AgentSpec(AgentSpecWrite):
    """One immutable tenant-owned AgentSpec revision."""

    tenant_id: ProtocolId
    id: ProtocolSlug
    revision: int = Field(ge=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    created_by: ProtocolId

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        return value.astimezone(UTC)

    @property
    def ref(self) -> AgentSpecRef:
        return AgentSpecRef(
            tenant_id=self.tenant_id,
            spec_id=self.id,
            revision=self.revision,
        )


class AtTrigger(_SpecModel):
    kind: Literal["at"] = "at"
    run_at: datetime
    timezone: str = Field(min_length=1, max_length=100)

    @field_validator("run_at")
    @classmethod
    def validate_run_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run_at must include a UTC offset")
        return value

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        return _iana_timezone(value)


class CronTriggerSpec(_SpecModel):
    kind: Literal["cron"] = "cron"
    expression: str = Field(min_length=1, max_length=200)
    timezone: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_cron(self) -> CronTriggerSpec:
        timezone = _iana_timezone(self.timezone)
        try:
            CronTrigger.from_crontab(self.expression, timezone=ZoneInfo(timezone))
        except (TypeError, ValueError) as exc:
            raise ValueError("expression must be a valid five-field cron expression") from exc
        return self


class IntervalTrigger(_SpecModel):
    kind: Literal["interval"] = "interval"
    every_seconds: int = Field(ge=1, le=31_536_000)


class EventTrigger(_SpecModel):
    kind: Literal["event"] = "event"
    event_type: ProtocolSlug
    source: ProtocolSlug
    authentication: Literal["required", "trusted_internal"] = "required"


TriggerSource = Annotated[
    AtTrigger | CronTriggerSpec | IntervalTrigger | EventTrigger,
    Field(discriminator="kind"),
]


class DeliverySpec(_SpecModel):
    mode: Literal["none", "origin", "owner"] = "origin"
    interface: ProtocolSlug | None = None

    @model_validator(mode="after")
    def validate_interface(self) -> DeliverySpec:
        if self.interface is not None and self.mode == "none":
            raise ValueError("delivery interface is invalid when delivery is disabled")
        return self


class TriggerSpecWrite(_SpecModel):
    """Configuration used to create one immutable trigger revision."""

    name: str = Field(min_length=1, max_length=200)
    source: TriggerSource
    exposure: TriggerExposure
    agent_spec: AgentSpecRef
    instruction: str = Field(min_length=1, max_length=200_000)
    delivery: DeliverySpec = Field(default_factory=DeliverySpec)
    enabled: bool = True
    source_key: str | None = Field(default=None, min_length=1, max_length=300)
    source_revision: int | None = Field(default=None, ge=1)
    labels: dict[ProtocolSlug, str] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def validate_exposure(self) -> TriggerSpecWrite:
        if self.exposure == "external":
            if not isinstance(self.source, EventTrigger):
                raise ValueError("external triggers must be authenticated event triggers")
            if self.source.authentication != "required":
                raise ValueError("external event triggers require authentication")
        return self


class TriggerSpec(TriggerSpecWrite):
    """One immutable tenant-owned TriggerSpec revision."""

    tenant_id: ProtocolId
    id: ProtocolSlug
    revision: int = Field(ge=1)
    content_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
    created_by: ProtocolId

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        return value.astimezone(UTC)


def _iana_timezone(value: str) -> str:
    timezone = str(value or "").strip()
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("timezone must be a valid IANA timezone") from exc
    return timezone


__all__ = [
    "AgentSpec",
    "AgentSpecWrite",
    "AtTrigger",
    "CronTriggerSpec",
    "DeliverySpec",
    "EventTrigger",
    "IntervalTrigger",
    "TriggerSource",
    "TriggerSpec",
    "TriggerSpecWrite",
]
