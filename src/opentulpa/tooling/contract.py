"""Canonical product-tool contract shared by agent profiles and API surfaces."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from opentulpa.specs.protocol import AgentRunContext

CONTRACT_VERSION: Literal["1.1"] = "1.1"


class AgentChannel(StrEnum):
    WEB = "web"
    TELEGRAM = "telegram"
    ROUTINE = "routine"
    INTAKE = "intake"


class AgentRunKind(StrEnum):
    OWNER = "owner"
    ROUTINE = "routine"
    INTAKE = "intake"


class ToolEffect(StrEnum):
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    SEND = "send"
    EXECUTE = "execute"
    AUTHORIZE = "authorize"


class ApprovalMode(StrEnum):
    AUTO = "auto"
    ALWAYS = "always"
    POLICY = "policy"


class IdempotencyMode(StrEnum):
    NONE = "none"
    DERIVED = "derived"
    REQUIRED = "required"


class ExecutionMode(StrEnum):
    SYNC = "sync"
    JOB = "job"


class ToolStatus(StrEnum):
    OK = "ok"
    ACCEPTED = "accepted"
    ERROR = "error"


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolSpec(_ContractModel):
    """Static policy metadata for one model-visible product operation."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(ge=1)
    provider: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    effect: ToolEffect
    approval: ApprovalMode
    idempotency: IdempotencyMode
    execution: ExecutionMode
    timeout_seconds: float = Field(gt=0)


class ToolError(_ContractModel):
    """Sanitized error safe to return to a model or public client."""

    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1)
    retryable: bool = False


class ToolResult[T](_ContractModel):
    """Uniform result envelope for synchronous and accepted background work."""

    status: ToolStatus
    data: T | None = None
    error: ToolError | None = None
    job_id: str | None = None
    idempotency_key: str | None = None
    audit_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status_payload(self) -> ToolResult[T]:
        if self.status is ToolStatus.ERROR and self.error is None:
            raise ValueError("error results require an error")
        if self.status is not ToolStatus.ERROR and self.error is not None:
            raise ValueError("non-error results cannot include an error")
        if self.status is ToolStatus.ACCEPTED and not self.job_id:
            raise ValueError("accepted results require a job_id")
        return self


class _ToolContractEnvelope[T](_ContractModel):
    context: AgentRunContext
    operation: ToolSpec
    result: ToolResult[T] | None = None


class _ToolContractDocument(_ContractModel):
    contract_version: Literal["1.1"] = CONTRACT_VERSION
    operations: tuple[ToolSpec, ...]


def _tool(
    name: str,
    provider: str,
    effect: ToolEffect,
    *,
    approval: ApprovalMode = ApprovalMode.AUTO,
    idempotency: IdempotencyMode = IdempotencyMode.NONE,
    execution: ExecutionMode = ExecutionMode.SYNC,
    timeout_seconds: float = 30,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        version=1,
        provider=provider,
        effect=effect,
        approval=approval,
        idempotency=idempotency,
        execution=execution,
        timeout_seconds=timeout_seconds,
    )


TOOL_SPECS: tuple[ToolSpec, ...] = (
    _tool("profile_get", "profile", ToolEffect.READ, timeout_seconds=10),
    _tool(
        "profile_update",
        "profile",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
        timeout_seconds=10,
    ),
    _tool("file_search", "files", ToolEffect.READ),
    _tool("file_get", "files", ToolEffect.READ),
    _tool(
        "file_analyze",
        "files",
        ToolEffect.EXECUTE,
        idempotency=IdempotencyMode.DERIVED,
        execution=ExecutionMode.JOB,
        timeout_seconds=300,
    ),
    _tool("file_inspect", "files", ToolEffect.READ, timeout_seconds=60),
    _tool(
        "artifact_deliver",
        "files",
        ToolEffect.SEND,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("knowledge_list", "knowledge", ToolEffect.READ),
    _tool("knowledge_find", "knowledge", ToolEffect.READ),
    _tool(
        "knowledge_attach",
        "knowledge",
        ToolEffect.CREATE,
        idempotency=IdempotencyMode.DERIVED,
        execution=ExecutionMode.JOB,
        timeout_seconds=300,
    ),
    _tool(
        "knowledge_archive",
        "knowledge",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
    ),
    _tool(
        "knowledge_reindex",
        "knowledge",
        ToolEffect.EXECUTE,
        idempotency=IdempotencyMode.DERIVED,
        execution=ExecutionMode.JOB,
        timeout_seconds=300,
    ),
    _tool("knowledge_query", "knowledge", ToolEffect.READ, timeout_seconds=60),
    _tool("web_search", "research", ToolEffect.READ),
    _tool("content_fetch", "research", ToolEffect.READ, timeout_seconds=60),
    _tool(
        "browser_start",
        "browser",
        ToolEffect.CREATE,
        idempotency=IdempotencyMode.REQUIRED,
        execution=ExecutionMode.JOB,
        timeout_seconds=120,
    ),
    _tool("browser_get", "browser", ToolEffect.READ),
    _tool(
        "browser_act",
        "browser",
        ToolEffect.EXECUTE,
        idempotency=IdempotencyMode.REQUIRED,
        execution=ExecutionMode.JOB,
        timeout_seconds=120,
    ),
    _tool(
        "browser_stop",
        "browser",
        ToolEffect.DELETE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("integration_list", "composio", ToolEffect.READ),
    _tool(
        "integration_connect",
        "composio",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("connection_list", "composio", ToolEffect.READ),
    _tool(
        "connection_disconnect",
        "composio",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("integration_action_search", "composio", ToolEffect.READ),
    _tool(
        "integration_invoke",
        "composio",
        ToolEffect.EXECUTE,
        idempotency=IdempotencyMode.REQUIRED,
        execution=ExecutionMode.JOB,
        timeout_seconds=120,
    ),
    _tool("intake_workflow_list", "intake", ToolEffect.READ),
    _tool("intake_workflow_get", "intake", ToolEffect.READ),
    _tool(
        "intake_draft_save",
        "intake",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
    ),
    _tool(
        "intake_draft_prepare",
        "intake",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
        timeout_seconds=60,
    ),
    _tool(
        "intake_draft_activate",
        "intake",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool(
        "intake_workflow_delete",
        "intake",
        ToolEffect.DELETE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool(
        "intake_workflow_test",
        "intake",
        ToolEffect.EXECUTE,
        idempotency=IdempotencyMode.DERIVED,
        execution=ExecutionMode.JOB,
        timeout_seconds=120,
    ),
    _tool("schedule_list", "scheduler", ToolEffect.READ),
    _tool(
        "schedule_save",
        "scheduler",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
    ),
    _tool(
        "schedule_delete",
        "scheduler",
        ToolEffect.DELETE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("agent_spec_list", "agent_specs", ToolEffect.READ),
    _tool(
        "agent_spec_save",
        "agent_specs",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
    ),
    _tool(
        "agent_spec_activate",
        "agent_specs",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool(
        "agent_spec_rollback",
        "agent_specs",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("trigger_spec_list", "trigger_specs", ToolEffect.READ),
    _tool(
        "trigger_spec_save",
        "trigger_specs",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
    ),
    _tool(
        "trigger_spec_activate",
        "trigger_specs",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool(
        "trigger_spec_rollback",
        "trigger_specs",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool("secret_handle_list", "secrets", ToolEffect.READ),
    _tool(
        "secret_handle_revoke",
        "secrets",
        ToolEffect.DELETE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool(
        "sandbox_ssh_diagnostic",
        "sandbox",
        ToolEffect.EXECUTE,
        approval=ApprovalMode.POLICY,
        timeout_seconds=660,
    ),
    _tool("capability_list", "capabilities", ToolEffect.READ),
    _tool(
        "capability_seed_bundled",
        "capabilities",
        ToolEffect.CREATE,
        idempotency=IdempotencyMode.DERIVED,
    ),
    _tool(
        "capability_test",
        "capabilities",
        ToolEffect.EXECUTE,
        timeout_seconds=600,
    ),
    _tool(
        "capability_activate",
        "capabilities",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=600,
    ),
    _tool(
        "capability_rollback",
        "capabilities",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=600,
    ),
    _tool(
        "capability_deactivate",
        "capabilities",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=600,
    ),
    _tool("job_get", "jobs", ToolEffect.READ),
    _tool("job_events", "jobs", ToolEffect.READ),
    _tool("job_artifacts", "jobs", ToolEffect.READ),
    _tool(
        "job_cancel",
        "jobs",
        ToolEffect.DELETE,
        idempotency=IdempotencyMode.REQUIRED,
    ),
    _tool(
        "repository_open",
        "repositories",
        ToolEffect.CREATE,
        idempotency=IdempotencyMode.DERIVED,
        timeout_seconds=300,
    ),
    _tool("repository_list", "repositories", ToolEffect.READ),
    _tool("repository_status", "repositories", ToolEffect.READ, timeout_seconds=120),
    _tool(
        "repository_close",
        "repositories",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.DERIVED,
        timeout_seconds=120,
    ),
    _tool(
        "repository_publish_pr",
        "repositories",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=600,
    ),
    _tool("source_status", "evolution", ToolEffect.READ),
    _tool("source_runtime_env_get", "evolution", ToolEffect.READ),
    _tool(
        "source_sync_upstream",
        "evolution",
        ToolEffect.UPDATE,
        timeout_seconds=300,
    ),
    _tool(
        "source_prepare_pr",
        "evolution",
        ToolEffect.CREATE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=900,
    ),
    _tool(
        "source_resolve_dependencies",
        "evolution",
        ToolEffect.UPDATE,
        timeout_seconds=1_800,
    ),
    _tool(
        "source_shell",
        "evolution",
        ToolEffect.EXECUTE,
        timeout_seconds=660,
    ),
    _tool(
        "source_release",
        "evolution",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=1_800,
    ),
    _tool(
        "source_rollback",
        "evolution",
        ToolEffect.AUTHORIZE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=60,
    ),
    _tool(
        "source_set_runtime_env",
        "evolution",
        ToolEffect.UPDATE,
        idempotency=IdempotencyMode.REQUIRED,
        timeout_seconds=300,
    ),
    _tool("trace_list", "observability", ToolEffect.READ),
    _tool("trace_get", "observability", ToolEffect.READ),
)

TOOL_SPEC_BY_NAME: Mapping[str, ToolSpec] = MappingProxyType(
    {spec.name: spec for spec in TOOL_SPECS}
)


def get_tool_spec(name: str) -> ToolSpec:
    """Return policy metadata for a registered product operation."""

    return TOOL_SPEC_BY_NAME[name]


def tool_contract_document() -> dict[str, Any]:
    """Return the serializable registry document consumed by tooling and audits."""

    document = _ToolContractDocument(operations=TOOL_SPECS)
    return document.model_dump(mode="json")


def tool_contract_json_schema() -> dict[str, Any]:
    """Return JSON Schema for a typed call envelope with the exact registry attached."""

    schema = _ToolContractEnvelope[Any].model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://opentulpa.local/schemas/tool-contract-v1.json"
    schema["x-opentulpa-contract-version"] = CONTRACT_VERSION
    schema["x-opentulpa-operations"] = tool_contract_document()["operations"]
    return schema


def render_tool_contract_markdown() -> str:
    """Render the human contract from the same registry used at runtime."""

    lines = [
        "<!-- Generated by opentulpa.tooling.render_tool_contract_markdown; do not edit. -->",
        "# OpenTulpa Product Tool Contract",
        "",
        f"Contract version: `{CONTRACT_VERSION}`",
        "",
        "The registry below is the complete model-visible product surface. Deep Agents built-ins provide planning, delegation, scratch files, memory, and skills and are not duplicated here.",
        "",
        "## Host Context",
        "",
        "`AgentRunContext` is injected by the host through `ToolRuntime`. Its fields must never be accepted from model-generated arguments.",
        "",
        "| Field | Values |",
        "| --- | --- |",
        "| `tenant_id` | Authenticated tenant identifier |",
        "| `actor_id` | Authenticated owner or system actor |",
        "| `thread_id` | Deep Agents checkpoint thread |",
        "| `channel` | `web`, `telegram`, `routine`, `intake` |",
        "| `run_kind` | `owner`, `routine`, `intake` |",
        "| `correlation_id` | End-to-end audit correlation identifier |",
        "",
        "## Policy Semantics",
        "",
        "- `approval=auto` runs without an interrupt after normal authorization checks.",
        "- `approval=always` persists an interrupt for explicit owner approval.",
        "- `approval=policy` delegates decisions to a tool-specific runtime policy.",
        "- Owner source shell and release tools run without per-call approval; source_release remains restart-safe through health checks and rollback.",
        "- `idempotency=required` rejects calls without a caller-supplied key.",
        "- `idempotency=derived` derives a stable key from canonical tenant-scoped input.",
        "- `execution=job` returns `status=accepted` and a durable `job_id`.",
        "- Every service validates tenant ownership; errors are sanitized before entering `ToolResult`.",
        "- `intake_draft_prepare` returns a hash-bound one-time `confirmation_handle`; only `intake_draft_activate` accepts it.",
        "- Secret handle tools expose metadata and revocation only. Owner-only `source_set_runtime_env` may write raw deployment secrets/config into the host-owned `.env`; results redact values and `.env` is excluded from source releases. Trusted adapters and declared capability bindings redeem only the scope they require.",
        "- Capability activation accepts config plus opaque secret-handle bindings only and requires an exact passing test attestation.",
        "- `trace_list` is newest-first; pass the last returned `run_id` as `before_run_id` to read the next page.",
        "",
        "## Result Envelope",
        "",
        "`ToolResult[T]` returns `status`, typed `data`, a sanitized `error`, optional `job_id`, optional `idempotency_key`, and mandatory `audit_id`. An `error` status requires `ToolError`; an `accepted` status requires `job_id`.",
        "",
        "## Operations",
        "",
        "| Name | Provider | Effect | Approval | Idempotency | Execution | Timeout |",
        "| --- | --- | --- | --- | --- | --- | ---: |",
    ]
    lines.extend(
        f"| `{spec.name}` | `{spec.provider}` | `{spec.effect.value}` | "
        f"`{spec.approval.value}` | `{spec.idempotency.value}` | "
        f"`{spec.execution.value}` | {spec.timeout_seconds:g}s |"
        for spec in TOOL_SPECS
    )
    return "\n".join(lines) + "\n"
