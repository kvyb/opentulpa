"""Typed repository-workspace state."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RepositoryProvider(StrEnum):
    LOCAL = "local"
    DAYTONA = "daytona"


class RepositoryWorkspaceStatus(StrEnum):
    CREATING = "creating"
    READY = "ready"
    STOPPED = "stopped"
    FAILED = "failed"
    PUBLISHED = "published"


class RepositoryWorkspace(BaseModel):
    """One isolated checkout bound to an owner conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(min_length=1, max_length=200)
    tenant_id: str = Field(min_length=1, max_length=200)
    repository_url: str = Field(min_length=1, max_length=2_000)
    provider: RepositoryProvider
    provider_workspace_id: str | None = Field(default=None, max_length=500)
    base_ref: str = Field(min_length=1, max_length=300)
    base_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    branch: str = Field(min_length=1, max_length=250)
    head_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    status: RepositoryWorkspaceStatus
    last_error: str | None = Field(default=None, max_length=500)
    pull_request_url: str | None = Field(default=None, max_length=2_000)
    source_candidate_id: str | None = Field(default=None, min_length=1, max_length=100)
    source_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    verified_tree_oid: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    verified_patch_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    export_verified_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime

    @field_validator("created_at", "updated_at", "last_used_at", "export_verified_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("repository workspace timestamps must be timezone-aware")
        return value.astimezone(UTC)


def utc_now() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "RepositoryProvider",
    "RepositoryWorkspace",
    "RepositoryWorkspaceStatus",
    "utc_now",
]
