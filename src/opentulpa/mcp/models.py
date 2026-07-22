"""Typed boundary between the capability broker and MCP transports."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from opentulpa.capabilities.models import SchemaDigest, ToolExport


class _MCPModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class MCPRemoteTool(_MCPModel):
    """Tool metadata discovered from an MCP server."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]{0,99}$")
    description: str = Field(default="", max_length=2_000)
    input_schema: dict[str, Any]
    schema_digest: SchemaDigest


class MCPCallMetadata(_MCPModel):
    """Trusted call metadata delivered out-of-band, never in model arguments."""

    tenant_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    audit_id: str = Field(min_length=1)
    idempotency_key: str | None = None


class MCPToolDescriptor(_MCPModel):
    """Exact model-visible schema paired with host-owned policy."""

    instance_id: str = Field(min_length=1)
    capability_name: str = Field(min_length=1)
    capability_revision: int = Field(ge=1)
    worker_name: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(default="", max_length=2_000)
    input_schema: dict[str, Any]
    policy: ToolExport


class MCPBrokerError(_MCPModel):
    code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False


class MCPBrokerResult(_MCPModel):
    status: Literal["ok", "approval_required", "error"]
    data: Any = None
    error: MCPBrokerError | None = None
    audit_id: str = Field(min_length=1)
    idempotency_key: str | None = None
    replayed: bool = False

    @model_validator(mode="after")
    def validate_error(self) -> MCPBrokerResult:
        if self.status == "error" and self.error is None:
            raise ValueError("error broker results require an error")
        if self.status != "error" and self.error is not None:
            raise ValueError("non-error broker results cannot include an error")
        return self


class MCPAuditEvent(_MCPModel):
    audit_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    capability_name: str = Field(min_length=1)
    capability_revision: int = Field(ge=1)
    worker_name: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    tenant_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    correlation_id: str = Field(min_length=1)
    tool_call_id: str = Field(min_length=1)
    idempotency_key: str | None = None
    approval_granted: bool
    outcome: Literal["started", "ok", "approval_required", "error", "replayed"]
    arguments: Any = None
    result: Any = None
    error_code: str | None = None


__all__ = [
    "MCPAuditEvent",
    "MCPBrokerError",
    "MCPBrokerResult",
    "MCPCallMetadata",
    "MCPRemoteTool",
    "MCPToolDescriptor",
]
