"""Model-visible argument schemas for the OpenTulpa product tools."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

from opentulpa.specs.models import AgentSpecWrite, DeliverySpec, TriggerSource
from opentulpa.specs.protocol import ProtocolSlug


class ToolArguments(BaseModel):
    """Strict base model that intentionally contains no host identity fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )


class RequiredIdempotencyArguments(ToolArguments):
    idempotency_key: str = Field(
        min_length=1,
        max_length=200,
        description="Unique key for safely retrying this external side effect.",
    )


class ProfileGetArguments(ToolArguments):
    pass


class ProfileUpdateArguments(ToolArguments):
    updates: dict[str, Any] = Field(
        min_length=1,
        description="Profile fields to update for the current authenticated owner.",
    )


class FileSearchArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class FileGetArguments(ToolArguments):
    file_id: str = Field(min_length=1, max_length=300)


class FileAnalyzeArguments(ToolArguments):
    file_id: str = Field(min_length=1, max_length=300)
    instruction: str = Field(min_length=1, max_length=20_000)


class FileInspectArguments(ToolArguments):
    file_id: str = Field(min_length=1, max_length=300)
    question: str | None = Field(default=None, max_length=20_000)


class ArtifactDeliverArguments(RequiredIdempotencyArguments):
    artifact_id: str = Field(min_length=1, max_length=300)
    caption: str | None = Field(default=None, max_length=4_000)


class KnowledgeListArguments(ToolArguments):
    include_archived: bool = False
    limit: int = Field(default=50, ge=1, le=200)


class KnowledgeFindArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class KnowledgeAttachArguments(ToolArguments):
    file_id: str = Field(min_length=1, max_length=300)
    title: str | None = Field(default=None, max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=30)


class KnowledgeArchiveArguments(ToolArguments):
    source_id: str = Field(min_length=1, max_length=300)


class KnowledgeReindexArguments(ToolArguments):
    source_id: str | None = Field(default=None, max_length=300)


class KnowledgeQueryArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=20_000)
    source_ids: list[str] = Field(default_factory=list, max_length=50)
    limit: int = Field(default=10, ge=1, le=50)


class WebSearchArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=8, ge=1, le=20)


class ContentFetchArguments(ToolArguments):
    url: AnyHttpUrl


class BrowserStartArguments(RequiredIdempotencyArguments):
    start_url: AnyHttpUrl | None = None
    allowed_domains: list[str] = Field(default_factory=list, max_length=50)


class BrowserGetArguments(ToolArguments):
    session_id: str = Field(min_length=1, max_length=300)


class BrowserActArguments(RequiredIdempotencyArguments):
    session_id: str = Field(min_length=1, max_length=300)
    action: dict[str, Any] = Field(
        min_length=1,
        description="One concrete browser action. Unknown actions are rejected.",
    )


class BrowserStopArguments(RequiredIdempotencyArguments):
    session_id: str = Field(min_length=1, max_length=300)


class IntegrationListArguments(ToolArguments):
    query: str | None = Field(default=None, max_length=500)


class IntegrationConnectArguments(RequiredIdempotencyArguments):
    integration_id: str = Field(min_length=1, max_length=300)
    redirect_url: AnyHttpUrl | None = None


class ConnectionListArguments(ToolArguments):
    integration_id: str | None = Field(default=None, max_length=300)


class ConnectionDisconnectArguments(RequiredIdempotencyArguments):
    connection_id: str = Field(min_length=1, max_length=300)


class IntegrationActionSearchArguments(ToolArguments):
    query: str = Field(min_length=1, max_length=1_000)
    integration_id: str | None = Field(default=None, max_length=300)
    limit: int = Field(default=20, ge=1, le=100)


class IntegrationInvokeArguments(RequiredIdempotencyArguments):
    connection_id: str = Field(min_length=1, max_length=300)
    action_name: str = Field(min_length=1, max_length=300)
    parameters: dict[str, Any] = Field(default_factory=dict)


class IntakeWorkflowListArguments(ToolArguments):
    include_inactive: bool = False


class IntakeWorkflowGetArguments(ToolArguments):
    workflow_id: str | None = Field(
        default=None,
        max_length=300,
        description="Workflow ID, or omit to read the active workflow.",
    )


class IntakeDraftSaveArguments(ToolArguments):
    draft_id: str | None = Field(default=None, max_length=300)
    expected_revision: int | None = Field(default=None, ge=1)
    patch: dict[str, Any] = Field(min_length=1)


class IntakeDraftPrepareArguments(ToolArguments):
    draft_id: str = Field(min_length=1, max_length=300)
    expected_revision: int = Field(ge=1)


class IntakeDraftActivateArguments(RequiredIdempotencyArguments):
    draft_id: str = Field(min_length=1, max_length=300)
    expected_revision: int = Field(ge=1)
    confirmation_handle: str = Field(
        min_length=1,
        max_length=2_000,
        description="Hash-bound one-time handle returned by intake_draft_prepare.",
    )


class IntakeWorkflowDeleteArguments(RequiredIdempotencyArguments):
    workflow_id: str = Field(min_length=1, max_length=300)
    expected_revision: int = Field(ge=1)


class IntakeWorkflowTestArguments(ToolArguments):
    workflow_id: str | None = Field(default=None, max_length=300)
    draft_id: str | None = Field(default=None, max_length=300)
    sample: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_one_configuration(self) -> IntakeWorkflowTestArguments:
        if bool(self.workflow_id) == bool(self.draft_id):
            raise ValueError("provide exactly one of workflow_id or draft_id")
        return self


class ScheduleListArguments(ToolArguments):
    include_disabled: bool = False


class ScheduleAtArguments(ToolArguments):
    kind: Literal["at"] = "at"
    run_at: datetime
    timezone: str = Field(min_length=1, max_length=100)

    @field_validator("run_at")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run_at must include a UTC offset")
        return value

    @field_validator("timezone")
    @classmethod
    def require_iana_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value


class ScheduleCronArguments(ToolArguments):
    kind: Literal["cron"] = "cron"
    expression: str = Field(min_length=1, max_length=200)
    timezone: str = Field(min_length=1, max_length=100)

    @field_validator("timezone")
    @classmethod
    def require_iana_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value


class ScheduleReminderArguments(ToolArguments):
    kind: Literal["reminder"] = "reminder"
    message: str = Field(min_length=1, max_length=20_000)


class ScheduleAgentJobArguments(ToolArguments):
    kind: Literal["agent_job"] = "agent_job"
    instruction: str = Field(min_length=1, max_length=20_000)


class SchedulePayload(ToolArguments):
    name: str = Field(min_length=1, max_length=200)
    trigger: Annotated[
        ScheduleAtArguments | ScheduleCronArguments,
        Field(discriminator="kind"),
    ]
    action: Annotated[
        ScheduleReminderArguments | ScheduleAgentJobArguments,
        Field(discriminator="kind"),
    ]
    notify_owner: bool = True
    enabled: bool = True


class ScheduleSaveArguments(ToolArguments):
    schedule_id: str | None = Field(default=None, max_length=100)
    expected_revision: int | None = Field(default=None, ge=1)
    schedule: SchedulePayload


class ScheduleDeleteArguments(RequiredIdempotencyArguments):
    schedule_id: str = Field(min_length=1, max_length=100)
    expected_revision: int = Field(ge=1)


class AgentSpecListArguments(ToolArguments):
    pass


class AgentSpecSaveArguments(ToolArguments):
    spec_id: ProtocolSlug | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    spec: AgentSpecWrite


class AgentSpecActivateArguments(RequiredIdempotencyArguments):
    spec_id: ProtocolSlug
    revision: int = Field(ge=1)
    expected_active_revision: int | None = Field(default=None, ge=1)


class AgentSpecRollbackArguments(RequiredIdempotencyArguments):
    spec_id: ProtocolSlug
    expected_active_revision: int = Field(ge=1)


class LocalAgentSpecRefArguments(ToolArguments):
    spec_id: ProtocolSlug
    revision: int = Field(ge=1)


class TriggerSpecPayload(ToolArguments):
    name: str = Field(min_length=1, max_length=200)
    source: TriggerSource
    exposure: Literal["private", "external"]
    agent_spec: LocalAgentSpecRefArguments
    instruction: str = Field(min_length=1, max_length=200_000)
    delivery: DeliverySpec = Field(default_factory=DeliverySpec)
    enabled: bool = True
    source_key: str | None = Field(default=None, min_length=1, max_length=300)
    source_revision: int | None = Field(default=None, ge=1)
    labels: dict[ProtocolSlug, str] = Field(default_factory=dict, max_length=100)


class TriggerSpecListArguments(ToolArguments):
    pass


class TriggerSpecSaveArguments(ToolArguments):
    trigger_id: ProtocolSlug | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    trigger: TriggerSpecPayload


class TriggerSpecActivateArguments(RequiredIdempotencyArguments):
    trigger_id: ProtocolSlug
    revision: int = Field(ge=1)
    expected_active_revision: int | None = Field(default=None, ge=1)


class TriggerSpecRollbackArguments(RequiredIdempotencyArguments):
    trigger_id: ProtocolSlug
    expected_active_revision: int = Field(ge=1)


class SecretHandleListArguments(ToolArguments):
    pass


class SecretHandleRevokeArguments(RequiredIdempotencyArguments):
    secret_id: ProtocolSlug
    expected_revision: int = Field(ge=1)


class SandboxSshDiagnosticArguments(ToolArguments):
    secret_id: ProtocolSlug = Field(
        description="Tenant-owned secret handle containing an SSH private key or password.",
    )
    host: str = Field(
        min_length=1,
        max_length=253,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}$",
    )
    user: str = Field(
        default="root",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$",
    )
    port: int = Field(default=22, ge=1, le=65_535)
    command: str = Field(min_length=1, max_length=20_000)
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    secret_type: Literal["private_key", "password"] = "private_key"


CapabilityName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    ),
]
CapabilitySecretName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[A-Z][A-Z0-9_]{0,127}$",
    ),
]
CapabilitySecretHandle = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_-]{0,127}$",
    ),
]


class CapabilityListArguments(ToolArguments):
    pass


class CapabilitySeedBundledArguments(ToolArguments):
    pass


class CapabilityTestArguments(ToolArguments):
    capability_name: CapabilityName
    revision: int = Field(ge=1)


class CapabilityActivateArguments(RequiredIdempotencyArguments):
    capability_name: CapabilityName
    revision: int = Field(ge=1)
    expected_generation: int | None = Field(default=None, ge=1)
    config: dict[str, JsonValue] = Field(default_factory=dict, max_length=200)
    secret_handles: dict[CapabilitySecretName, CapabilitySecretHandle] = Field(
        default_factory=dict,
        max_length=100,
        description=(
            "Map declared secret environment names to tenant-owned secret handle IDs. "
            "Never provide plaintext credentials."
        ),
    )
    refresh_agent_binding: bool = Field(
        default=False,
        description=(
            "Create a new generation bound to the currently active AgentSpec revision. "
            "Leave false for idempotent activation."
        ),
    )


class CapabilityRollbackArguments(RequiredIdempotencyArguments):
    capability_name: CapabilityName
    expected_generation: int = Field(ge=1)
    config: dict[str, JsonValue] | None = Field(default=None, max_length=200)
    secret_handles: dict[CapabilitySecretName, CapabilitySecretHandle] | None = Field(
        default=None,
        max_length=100,
        description=(
            "Optional replacement mapping from secret environment names to tenant-owned "
            "secret handle IDs. Omit it to retain the active bindings."
        ),
    )


class CapabilityDeactivateArguments(RequiredIdempotencyArguments):
    capability_name: CapabilityName
    expected_generation: int = Field(ge=1)


class JobGetArguments(ToolArguments):
    job_id: str = Field(min_length=1, max_length=300)


class JobEventsArguments(ToolArguments):
    job_id: str = Field(min_length=1, max_length=300)
    after_sequence: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=500)


class JobArtifactsArguments(ToolArguments):
    job_id: str = Field(min_length=1, max_length=300)


class JobCancelArguments(RequiredIdempotencyArguments):
    job_id: str = Field(min_length=1, max_length=300)


class RepositoryOpenArguments(ToolArguments):
    repository_url: AnyHttpUrl
    base_ref: str = Field(default="main", min_length=1, max_length=300)
    branch: str | None = Field(default=None, min_length=1, max_length=250)
    provider: Literal["auto", "local", "daytona"] = "auto"


class RepositoryListArguments(ToolArguments):
    include_closed: bool = False


class RepositoryStatusArguments(ToolArguments):
    workspace_id: str | None = Field(default=None, min_length=1, max_length=200)


class RepositoryCloseArguments(ToolArguments):
    workspace_id: str | None = Field(default=None, min_length=1, max_length=200)


class RepositoryPublishPullRequestArguments(RequiredIdempotencyArguments):
    workspace_id: str | None = Field(default=None, min_length=1, max_length=200)
    expected_head_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(default="", max_length=20_000)
    draft: bool = True


class SourceStatusArguments(ToolArguments):
    pass


class SourceReadArguments(ToolArguments):
    path: str = Field(min_length=1, max_length=4_096)
    offset: int = Field(default=1, ge=1)
    limit: int = Field(default=2_000, ge=1, le=2_000)


class SourceWriteArguments(ToolArguments):
    path: str = Field(min_length=1, max_length=4_096)
    content: str = Field(max_length=2_000_000)


class SourceEditArguments(ToolArguments):
    path: str = Field(min_length=1, max_length=4_096)
    old_text: str = Field(min_length=1, max_length=1_000_000)
    new_text: str = Field(max_length=1_000_000)
    replace_all: bool = False


class SourceBashArguments(ToolArguments):
    command: str = Field(min_length=1, max_length=100_000)
    timeout_seconds: int = Field(default=300, ge=1, le=600)


class SourceActivateArguments(RequiredIdempotencyArguments):
    message: str = Field(default="OpenTulpa self-update", min_length=1, max_length=500)
    reason: str = Field(default="Trusted source activation", max_length=4_000)


class SourceRollbackArguments(RequiredIdempotencyArguments):
    expected_active_release_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(default="Owner requested rollback", max_length=4_000)


class SourceSetRuntimeEnvArguments(RequiredIdempotencyArguments):
    name: str = Field(
        pattern=r"^[A-Z_][A-Z0-9_]{0,127}$",
        description="Runtime .env variable name to set in the live checkout.",
    )
    value: str | None = Field(
        default=None,
        max_length=65_536,
        description="Non-secret variable value. Use secret_id for credentials.",
    )
    secret_id: ProtocolSlug | None = Field(
        default=None,
        description=(
            "Tenant-owned secret handle whose name matches the runtime variable. "
            "Never provide plaintext credentials."
        ),
    )

    @model_validator(mode="after")
    def require_one_value_source(self) -> SourceSetRuntimeEnvArguments:
        if (self.value is None) == (self.secret_id is None):
            raise ValueError("exactly one of value or secret_id is required")
        return self


class TraceListArguments(ToolArguments):
    status: Literal[
        "running",
        "interrupted",
        "resume_pending",
        "completed",
        "failed",
        "cancelled",
    ] | None = None
    limit: int = Field(default=20, ge=1, le=100)
    before_run_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Return runs older than this previously returned run ID.",
    )


class TraceGetArguments(ToolArguments):
    run_id: str = Field(min_length=1, max_length=100)
    after_sequence: int = Field(default=0, ge=0)
    limit: int = Field(default=200, ge=1, le=500)
    include_messages: bool = False


OPERATION_ARGUMENT_SCHEMAS: Mapping[str, type[ToolArguments]] = MappingProxyType(
    {
        "profile_get": ProfileGetArguments,
        "profile_update": ProfileUpdateArguments,
        "file_search": FileSearchArguments,
        "file_get": FileGetArguments,
        "file_analyze": FileAnalyzeArguments,
        "file_inspect": FileInspectArguments,
        "artifact_deliver": ArtifactDeliverArguments,
        "knowledge_list": KnowledgeListArguments,
        "knowledge_find": KnowledgeFindArguments,
        "knowledge_attach": KnowledgeAttachArguments,
        "knowledge_archive": KnowledgeArchiveArguments,
        "knowledge_reindex": KnowledgeReindexArguments,
        "knowledge_query": KnowledgeQueryArguments,
        "web_search": WebSearchArguments,
        "content_fetch": ContentFetchArguments,
        "browser_start": BrowserStartArguments,
        "browser_get": BrowserGetArguments,
        "browser_act": BrowserActArguments,
        "browser_stop": BrowserStopArguments,
        "integration_list": IntegrationListArguments,
        "integration_connect": IntegrationConnectArguments,
        "connection_list": ConnectionListArguments,
        "connection_disconnect": ConnectionDisconnectArguments,
        "integration_action_search": IntegrationActionSearchArguments,
        "integration_invoke": IntegrationInvokeArguments,
        "intake_workflow_list": IntakeWorkflowListArguments,
        "intake_workflow_get": IntakeWorkflowGetArguments,
        "intake_draft_save": IntakeDraftSaveArguments,
        "intake_draft_prepare": IntakeDraftPrepareArguments,
        "intake_draft_activate": IntakeDraftActivateArguments,
        "intake_workflow_delete": IntakeWorkflowDeleteArguments,
        "intake_workflow_test": IntakeWorkflowTestArguments,
        "schedule_list": ScheduleListArguments,
        "schedule_save": ScheduleSaveArguments,
        "schedule_delete": ScheduleDeleteArguments,
        "agent_spec_list": AgentSpecListArguments,
        "agent_spec_save": AgentSpecSaveArguments,
        "agent_spec_activate": AgentSpecActivateArguments,
        "agent_spec_rollback": AgentSpecRollbackArguments,
        "trigger_spec_list": TriggerSpecListArguments,
        "trigger_spec_save": TriggerSpecSaveArguments,
        "trigger_spec_activate": TriggerSpecActivateArguments,
        "trigger_spec_rollback": TriggerSpecRollbackArguments,
        "secret_handle_list": SecretHandleListArguments,
        "secret_handle_revoke": SecretHandleRevokeArguments,
        "sandbox_ssh_diagnostic": SandboxSshDiagnosticArguments,
        "capability_list": CapabilityListArguments,
        "capability_seed_bundled": CapabilitySeedBundledArguments,
        "capability_test": CapabilityTestArguments,
        "capability_activate": CapabilityActivateArguments,
        "capability_rollback": CapabilityRollbackArguments,
        "capability_deactivate": CapabilityDeactivateArguments,
        "job_get": JobGetArguments,
        "job_events": JobEventsArguments,
        "job_artifacts": JobArtifactsArguments,
        "job_cancel": JobCancelArguments,
        "repository_open": RepositoryOpenArguments,
        "repository_list": RepositoryListArguments,
        "repository_status": RepositoryStatusArguments,
        "repository_close": RepositoryCloseArguments,
        "repository_publish_pr": RepositoryPublishPullRequestArguments,
        "source_status": SourceStatusArguments,
        "source_read": SourceReadArguments,
        "source_write": SourceWriteArguments,
        "source_edit": SourceEditArguments,
        "source_bash": SourceBashArguments,
        "source_activate": SourceActivateArguments,
        "source_rollback": SourceRollbackArguments,
        "source_runtime_env_get": ToolArguments,
        "source_set_runtime_env": SourceSetRuntimeEnvArguments,
        "trace_list": TraceListArguments,
        "trace_get": TraceGetArguments,
    }
)


__all__ = ["OPERATION_ARGUMENT_SCHEMAS", "RequiredIdempotencyArguments", "ToolArguments"]
