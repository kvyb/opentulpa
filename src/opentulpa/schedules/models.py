"""Strict, tenant-scoped schedule models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _iana_timezone(value: str) -> str:
    timezone = str(value or "").strip()
    if not timezone:
        raise ValueError("timezone is required")
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("timezone must be a valid IANA timezone") from exc
    return timezone


class _ScheduleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class _Trigger(_ScheduleModel):
    timezone: str = Field(min_length=1)
    misfire_policy: Literal["skip"] = "skip"
    misfire_grace_seconds: Literal[1] = 1

    _validate_timezone = field_validator("timezone")(_iana_timezone)


class At(_Trigger):
    """A one-off trigger represented by an absolute instant and display timezone."""

    kind: Literal["at"] = "at"
    run_at: datetime

    @field_validator("run_at")
    @classmethod
    def _require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run_at must include a UTC offset")
        return value


class Cron(_Trigger):
    """A recurring five-field cron trigger evaluated in an IANA timezone."""

    kind: Literal["cron"] = "cron"
    expression: str = Field(min_length=1)
    coalesce: Literal[True] = True

    @model_validator(mode="after")
    def _validate_expression(self) -> Cron:
        try:
            CronTrigger.from_crontab(self.expression, timezone=ZoneInfo(self.timezone))
        except (TypeError, ValueError) as exc:
            raise ValueError("expression must be a valid five-field cron expression") from exc
        return self


Trigger = Annotated[At | Cron, Field(discriminator="kind")]


class Reminder(_ScheduleModel):
    kind: Literal["reminder"] = "reminder"
    message: str = Field(min_length=1)


class AgentJob(_ScheduleModel):
    kind: Literal["agent_job"] = "agent_job"
    instruction: str = Field(min_length=1)


ScheduleAction = Annotated[Reminder | AgentJob, Field(discriminator="kind")]


class ScheduleWrite(_ScheduleModel):
    """Mutable schedule fields accepted by the application service."""

    name: str = Field(min_length=1, max_length=200)
    trigger: Trigger
    action: ScheduleAction
    notify_owner: bool = True
    enabled: bool = True


class Schedule(ScheduleWrite):
    """Persisted schedule with tenant ownership and optimistic revision."""

    id: str = Field(min_length=1, max_length=100)
    tenant_id: str = Field(min_length=1, max_length=200)
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @field_validator("created_at", "updated_at")
    @classmethod
    def _require_aware_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("schedule timestamps must include a UTC offset")
        return value


class LegacyMigrationIssue(_ScheduleModel):
    routine_id: str
    disposition: Literal["invalid", "skipped", "conflict"] = "invalid"
    error: str


class LegacyMigrationReport(_ScheduleModel):
    dry_run: bool
    scanned: int = Field(ge=0)
    eligible: int = Field(ge=0)
    migrated: int = Field(ge=0)
    skipped: int = Field(ge=0)
    invalid: int = Field(ge=0)
    source_checksum: str
    issues: list[LegacyMigrationIssue] = Field(default_factory=list)
