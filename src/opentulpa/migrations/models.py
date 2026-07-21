"""Typed reports and source records for the Deep Agents cutover migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

type PreservedDatasetName = Literal[
    "profiles",
    "files",
    "knowledge",
    "active_workflows",
    "bookings",
    "integration_connections",
]


class _MigrationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MigrationIssue(_MigrationModel):
    component: Literal["routines", "drafts", "memories", "skills"]
    legacy_id: str
    disposition: Literal["invalid", "skipped", "conflict"]
    message: str
    disabled: bool = False


class ComponentMigrationReport(_MigrationModel):
    scanned: int = Field(ge=0)
    eligible: int = Field(ge=0)
    migrated: int = Field(ge=0)
    skipped: int = Field(ge=0)
    invalid: int = Field(ge=0)
    disabled: int = Field(ge=0)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    issues: list[MigrationIssue] = Field(default_factory=list)


class PreservedDatasetReport(_MigrationModel):
    dataset: PreservedDatasetName
    database_path: str
    status: Literal["ok", "missing", "invalid", "unreadable"]
    record_count: int = Field(ge=0)
    table_counts: dict[str, int] = Field(default_factory=dict)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    message: str | None = None


class PreservedProductDataReport(_MigrationModel):
    verified: bool
    datasets: list[PreservedDatasetReport]
    combined_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class DeepAgentsMigrationReport(_MigrationModel):
    dry_run: bool
    status: Literal["completed", "blocked"] = "completed"
    preserved_data: PreservedProductDataReport
    routines: ComponentMigrationReport
    drafts: ComponentMigrationReport
    memories: ComponentMigrationReport
    skills: ComponentMigrationReport
    combined_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_paths_pending_rebase: int = Field(default=0, ge=0)
    file_paths_rebased: int = Field(default=0, ge=0)
    checkpoints_migrated: Literal[0] = 0
    checkpoint_policy: Literal["fresh"] = "fresh"


class LegacyMemoryRecord(_MigrationModel):
    legacy_id: str
    tenant_id: str
    content: str
    created_at: str | None = None
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LegacyMemorySource(Protocol):
    """Minimal source contract so tests and offline exports do not require Mem0."""

    def records(self) -> list[LegacyMemoryRecord]: ...

    def disable(self, record: LegacyMemoryRecord, *, reason: str) -> bool: ...


class DeepAgentsMigrationConfig(_MigrationModel):
    customer_profiles_db_path: Path
    file_vault_db_path: Path
    file_vault_root_path: Path
    knowledge_db_path: Path
    intake_workflows_db_path: Path
    integration_connections_db_path: Path
    legacy_routines_db_path: Path | None = None
    agent_specs_db_path: Path
    trigger_specs_db_path: Path
    legacy_setup_db_path: Path | None = None
    intake_drafts_db_path: Path
    legacy_skills_db_path: Path | None = None
    legacy_skills_root_path: Path | None = None
    store_db_path: Path
    default_timezone: str = "UTC"
    allow_missing_preserved_data: bool = False


__all__ = [
    "ComponentMigrationReport",
    "DeepAgentsMigrationConfig",
    "DeepAgentsMigrationReport",
    "LegacyMemoryRecord",
    "LegacyMemorySource",
    "MigrationIssue",
    "PreservedDatasetReport",
    "PreservedDatasetName",
    "PreservedProductDataReport",
]
