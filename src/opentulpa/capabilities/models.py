"""Strict contracts for versioned, out-of-process OpenTulpa capabilities."""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from opentulpa.specs.protocol import AgentRunBinding, ProtocolSlug
from opentulpa.tooling.contract import (
    ApprovalMode,
    ExecutionMode,
    IdempotencyMode,
    ToolEffect,
)

CapabilityName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]{0,63}$",
    ),
]
ExportName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]{0,99}$",
    ),
]
PermissionName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$",
    ),
]
DependencyRequirement = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=200),
]
NetworkHost = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^(?:\*\.)?[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?(?::[0-9]{1,5})?$",
    ),
]
SchemaDigest = Annotated[
    str,
    StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$"),
]
SecretScope = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$",
    ),
]

_BROKERED_OPENTULPA_SECRET_NAMES = frozenset(
    {
        "OPENTULPA_AGENT_API_TOKEN",
        "OPENTULPA_TELEGRAM_PAIRING_CODE",
    }
)
_RUNTIME_ENVIRONMENT_NAMES = frozenset(
    {
        "ALL_PROXY",
        "BASH_ENV",
        "CLASSPATH",
        "DOCKER_HOST",
        "ENV",
        "GCONV_PATH",
        "GEM_HOME",
        "GEM_PATH",
        "HOME",
        "HOSTALIASES",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "IFS",
        "KUBECONFIG",
        "LANG",
        "LANGUAGE",
        "LIBPATH",
        "LOCPATH",
        "LOGNAME",
        "NLSPATH",
        "NO_PROXY",
        "OLDPWD",
        "PATH",
        "PERL5LIB",
        "PERL5OPT",
        "PERLLIB",
        "PWD",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "RES_OPTIONS",
        "RUBYLIB",
        "RUBYOPT",
        "SHELLOPTS",
        "SHELL",
        "SHLIB_PATH",
        "SSH_AUTH_SOCK",
        "TEMP",
        "TERM",
        "TMP",
        "TMPDIR",
        "TZ",
        "TZDIR",
        "USER",
        "ZDOTDIR",
    }
)
_RUNTIME_ENVIRONMENT_SUFFIXES = (
    "_DIR",
    "_FILE",
    "_HOME",
    "_HOOKS",
    "_LIB",
    "_LIBS",
    "_MODULES",
    "_OPTIONS",
    "_PATH",
    "_PLUGIN",
    "_PLUGINS",
    "_PRELOAD",
    "_STARTUP",
)
_RUNTIME_ENVIRONMENT_PREFIXES = (
    "BUNDLE_",
    "CARGO_",
    "DOTNET_",
    "DYLD_",
    "GEM_",
    "JAVA_",
    "JDK_JAVA_",
    "LD_",
    "LUA_",
    "MONO_",
    "NODE_",
    "NPM_",
    "NVM_",
    "OPENTULPA_",
    "PERL",
    "PHP_",
    "PYENV_",
    "PYTHON",
    "RBENV_",
    "RUBY",
    "RUST",
    "YARN_",
)


def is_reserved_worker_environment_name(name: str) -> bool:
    """Return whether a secret name could alter the trusted worker launcher."""

    normalized = str(name or "").strip().upper()
    if normalized in _BROKERED_OPENTULPA_SECRET_NAMES:
        return False
    return (
        normalized in _RUNTIME_ENVIRONMENT_NAMES
        or normalized.startswith(_RUNTIME_ENVIRONMENT_PREFIXES)
        or normalized.endswith(_RUNTIME_ENVIRONMENT_SUFFIXES)
    )


class WorkerKind(StrEnum):
    INTERFACE = "interface"
    MCP = "mcp"
    TRIGGER = "trigger"
    UI = "ui"


class WorkerRuntime(StrEnum):
    SUBPROCESS = "subprocess"
    OCI = "oci"


class WorkerTransport(StrEnum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"


class CapabilityTestStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class CapabilityActivationState(StrEnum):
    """Durable lifecycle state for one generation pointer."""

    ACTIVE = "active"
    DEACTIVATING = "deactivating"
    INACTIVE = "inactive"


class SecretSource(StrEnum):
    """Trusted source allowed to provide one worker environment value."""

    TENANT_HANDLE = "tenant_handle"
    HOST = "host"
    ISSUED = "issued"


class _CapabilityModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class AgentInterfaceBinding(_CapabilityModel):
    """Reviewed agent authority granted to an interface worker credential."""

    agent_spec_id: ProtocolSlug
    run_kind: Literal["owner", "routine", "intake"]
    trust_class: Literal["owner", "background", "external"]

    @model_validator(mode="after")
    def reject_owner_spec_for_restricted_ingress(self) -> Self:
        if (self.trust_class == "owner") != (self.run_kind == "owner"):
            raise ValueError("owner run kind and owner trust must be granted together")
        if self.trust_class != "owner" and self.agent_spec_id == "owner":
            raise ValueError("restricted interfaces cannot bind the owner AgentSpec")
        return self


def canonical_json_digest(value: Any) -> str:
    """Return a stable digest for manifests and discovered JSON schemas."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


class SecretRequirement(_CapabilityModel):
    """Secret grant made available only by the trusted capability host."""

    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,127}$")
    scopes: tuple[SecretScope, ...] = Field(min_length=1, max_length=100)
    source: SecretSource = SecretSource.TENANT_HANDLE
    required: bool = True

    @field_validator("name")
    @classmethod
    def reject_runtime_control_environment(cls, value: str) -> str:
        if is_reserved_worker_environment_name(value):
            raise ValueError("secret requirement name is reserved for worker runtime control")
        return value

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, value: tuple[SecretScope, ...]) -> tuple[SecretScope, ...]:
        if len(value) != len(set(value)):
            raise ValueError("secret requirement scopes must be unique")
        return value


class CapabilitySecretBinding(_CapabilityModel):
    """Revision-bound tenant secret reference persisted with an activation."""

    handle_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,127}$")
    revision: int = Field(ge=1)
    scopes: tuple[SecretScope, ...] = Field(min_length=1, max_length=100)

    @field_validator("scopes")
    @classmethod
    def validate_scopes(cls, value: tuple[SecretScope, ...]) -> tuple[SecretScope, ...]:
        if len(value) != len(set(value)):
            raise ValueError("secret binding scopes must be unique")
        return value


class NetworkPolicy(_CapabilityModel):
    """Declared network access; outbound access denies by default."""

    inbound: bool = False
    outbound: Literal["deny", "allowlist", "tenant_allowlist"] = "deny"
    allowed_hosts: tuple[NetworkHost, ...] = Field(default=(), max_length=100)

    @model_validator(mode="after")
    def validate_allowed_hosts(self) -> Self:
        if len(self.allowed_hosts) != len(set(self.allowed_hosts)):
            raise ValueError("network allowed_hosts must be unique")
        if self.outbound == "allowlist" and not self.allowed_hosts:
            raise ValueError("allowlist network policy requires allowed_hosts")
        if self.outbound != "allowlist" and self.allowed_hosts:
            raise ValueError("allowed_hosts are only valid for allowlist network policy")
        return self


class ResourceLimits(_CapabilityModel):
    """Requested limits enforced by isolated worker hosts."""

    cpu: float = Field(default=1.0, gt=0, le=32)
    memory_mb: int = Field(default=512, ge=64, le=65_536)
    pids: int = Field(default=128, ge=8, le=4_096)
    startup_timeout_seconds: float = Field(default=30, gt=0, le=600)
    stop_timeout_seconds: float = Field(default=10, gt=0, le=120)
    max_output_bytes: int = Field(default=1_000_000, ge=1_024, le=100_000_000)


class HealthCheck(_CapabilityModel):
    """Protocol-level worker health check interpreted by its host."""

    kind: Literal["process", "http", "mcp", "ready_file"] = "process"
    target: str | None = Field(default=None, max_length=500)
    interval_seconds: float = Field(default=10, gt=0, le=300)
    timeout_seconds: float = Field(default=5, gt=0, le=60)

    @model_validator(mode="after")
    def validate_target(self) -> Self:
        if self.kind == "http":
            if not self.target or not self.target.startswith("/"):
                raise ValueError("http health checks require an absolute path target")
        elif self.target is not None:
            raise ValueError("only http health checks accept a target")
        return self


class ToolExport(_CapabilityModel):
    """Policy bound to one exact MCP input schema."""

    name: ExportName
    description: str = Field(min_length=1, max_length=2_000)
    schema_digest: SchemaDigest
    effect: ToolEffect
    approval: ApprovalMode = ApprovalMode.ALWAYS
    idempotency: IdempotencyMode = IdempotencyMode.REQUIRED
    execution: ExecutionMode = ExecutionMode.SYNC
    timeout_seconds: float = Field(default=30, gt=0, le=3_600)

    @model_validator(mode="after")
    def validate_side_effect_idempotency(self) -> Self:
        if self.effect is not ToolEffect.READ and self.idempotency is IdempotencyMode.NONE:
            raise ValueError("side-effecting MCP tools require idempotency")
        return self


class WorkerSpec(_CapabilityModel):
    """Executable worker definition hosted outside the trusted kernel process."""

    name: ExportName
    kind: WorkerKind
    protocol: str = Field(pattern=r"^[a-z][a-z0-9_-]*-v[1-9][0-9]*$", max_length=100)
    runtime: WorkerRuntime = WorkerRuntime.SUBPROCESS
    transport: WorkerTransport = WorkerTransport.STDIO
    command: tuple[str, ...] = Field(min_length=1, max_length=64)
    endpoint: str | None = Field(default=None, max_length=2_000)
    image: str | None = Field(default=None, max_length=500)
    tools: tuple[ToolExport, ...] = Field(default=(), max_length=200)
    agent_binding: AgentInterfaceBinding | None = None
    permissions: tuple[PermissionName, ...] = Field(default=(), max_length=100)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    secrets: tuple[SecretRequirement, ...] = Field(default=(), max_length=100)
    resources: ResourceLimits = Field(default_factory=ResourceLimits)
    healthcheck: HealthCheck = Field(default_factory=HealthCheck)
    required: bool = True

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for argument in value:
            if not argument or argument != argument.strip():
                raise ValueError("worker command arguments must be non-empty and trimmed")
            if len(argument) > 2_000 or any(char in argument for char in ("\0", "\n", "\r")):
                raise ValueError("worker command arguments contain unsupported characters")
        return value

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> Self:
        expected_protocol = {
            WorkerKind.INTERFACE: "agent-interface-v1",
            WorkerKind.MCP: "mcp-v1",
            WorkerKind.TRIGGER: "agent-trigger-v1",
            WorkerKind.UI: "web-ui-v1",
        }[self.kind]
        if self.protocol != expected_protocol:
            raise ValueError(f"{self.kind.value} workers require protocol {expected_protocol!r}")
        if self.transport is WorkerTransport.STDIO and self.endpoint is not None:
            raise ValueError("stdio workers cannot declare an endpoint")
        if self.transport is WorkerTransport.STREAMABLE_HTTP:
            endpoint = str(self.endpoint or "")
            if not endpoint.startswith(("http://", "https://")):
                raise ValueError("streamable HTTP workers require an http(s) endpoint")
        if self.runtime is WorkerRuntime.OCI:
            image = str(self.image or "")
            if not image or (
                "@sha256:" not in image and re.fullmatch(r"sha256:[0-9a-f]{64}", image) is None
            ):
                raise ValueError("OCI workers require a digest-pinned image")
        elif self.image is not None:
            raise ValueError("subprocess workers cannot declare an OCI image")
        if self.healthcheck.kind == "ready_file" and self.runtime is not WorkerRuntime.SUBPROCESS:
            raise ValueError("ready-file health checks require a subprocess worker")
        if self.kind is not WorkerKind.MCP and self.tools:
            raise ValueError("only MCP workers can export tools")
        agent_api_secrets = tuple(
            secret for secret in self.secrets if secret.name == "OPENTULPA_AGENT_API_TOKEN"
        )
        uses_agent_api = bool(agent_api_secrets)
        if any(secret.source is not SecretSource.ISSUED for secret in agent_api_secrets):
            raise ValueError("OPENTULPA_AGENT_API_TOKEN must use the issued secret source")
        if uses_agent_api and self.kind is not WorkerKind.INTERFACE:
            raise ValueError("only interface workers can receive Agent API credentials")
        if uses_agent_api and self.agent_binding is None:
            raise ValueError("Agent API interface workers require an agent_binding")
        if self.agent_binding is not None and not uses_agent_api:
            raise ValueError("agent_binding requires an issued Agent API credential")
        self._validate_unique("tools", tuple(tool.name for tool in self.tools))
        self._validate_unique("permissions", self.permissions)
        self._validate_unique("secrets", tuple(secret.name for secret in self.secrets))
        return self

    @staticmethod
    def _validate_unique(label: str, values: tuple[object, ...]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"worker {label} must be unique")


class EvalCommand(_CapabilityModel):
    """One shell-free verification command executed as an argument vector."""

    argv: tuple[str, ...] = Field(min_length=1, max_length=64)
    timeout_seconds: int = Field(default=300, ge=1, le=3_600)

    @field_validator("argv")
    @classmethod
    def validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for argument in value:
            if not argument or argument != argument.strip():
                raise ValueError("eval command arguments must be non-empty and trimmed")
            if len(argument) > 1_000 or any(
                character in argument for character in ("\0", "\n", "\r")
            ):
                raise ValueError("eval command arguments contain unsupported characters")
        return value


class CapabilityManifest(_CapabilityModel):
    """Immutable capability revision inspected before any implementation is loaded."""

    schema_version: Literal[1] = 1
    name: CapabilityName
    version: str = Field(
        pattern=(
            r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
            r"(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
        ),
        max_length=100,
    )
    revision: int = Field(default=1, ge=1)
    artifact_digest: SchemaDigest | None = None
    config_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "additionalProperties": False}
    )
    workers: tuple[WorkerSpec, ...] = Field(default=(), max_length=100)

    # Import-based metadata remains readable for the seed distribution while worker
    # hosts use ``workers``. Generated capabilities need not define these fields.
    module: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$",
        max_length=300,
    )
    entrypoint: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        max_length=100,
    )
    dependencies: tuple[DependencyRequirement, ...] = Field(default=(), max_length=100)
    tools: tuple[ExportName, ...] = Field(default=(), max_length=200)
    services: tuple[ExportName, ...] = Field(default=(), max_length=100)
    permissions: tuple[PermissionName, ...] = Field(default=(), max_length=100)
    network: NetworkPolicy = Field(default_factory=NetworkPolicy)
    secrets: tuple[SecretRequirement, ...] = Field(default=(), max_length=100)
    eval_commands: tuple[EvalCommand, ...] = Field(min_length=1, max_length=100)
    seed: bool = False

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if (self.module is None) != (self.entrypoint is None):
            raise ValueError("module and entrypoint must be declared together")
        if self.module is None and not self.workers:
            raise ValueError("a capability requires a worker or import entrypoint")
        if self.config_schema.get("type") != "object":
            raise ValueError("capability config_schema must describe an object")
        try:
            canonical_json_digest(self.config_schema)
        except (TypeError, ValueError) as exc:
            raise ValueError("capability config_schema must contain only JSON values") from exc
        named_collections = {
            "dependencies": self.dependencies,
            "tools": self.tools,
            "services": self.services,
            "permissions": self.permissions,
            "secrets": tuple(secret.name for secret in self.secrets),
            "workers": tuple(worker.name for worker in self.workers),
            "eval_commands": tuple(command.argv for command in self.eval_commands),
        }
        for field_name, values in named_collections.items():
            if len(values) != len(set(values)):
                raise ValueError(f"capability {field_name} must be unique")
        worker_tools = tuple(tool.name for worker in self.workers for tool in worker.tools)
        if len(worker_tools) != len(set(worker_tools)):
            raise ValueError("capability worker tool exports must be unique")
        secret_contracts: dict[str, tuple[SecretSource, tuple[SecretScope, ...]]] = {}
        for secret in (
            *self.secrets,
            *(item for worker in self.workers for item in worker.secrets),
        ):
            if (
                secret.name == "OPENTULPA_AGENT_API_TOKEN"
                and secret.source is not SecretSource.ISSUED
            ):
                raise ValueError("OPENTULPA_AGENT_API_TOKEN must use the issued secret source")
            contract = (secret.source, secret.scopes)
            existing = secret_contracts.setdefault(secret.name, contract)
            if existing != contract:
                raise ValueError(f"secret {secret.name!r} has inconsistent source or scopes")
        return self

    @property
    def module_entrypoint(self) -> str:
        if self.module is None or self.entrypoint is None:
            return ""
        return f"{self.module}:{self.entrypoint}"

    @property
    def exported_tools(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.tools, *(t.name for w in self.workers for t in w.tools))))

    @property
    def content_digest(self) -> str:
        return canonical_json_digest(self.model_dump(mode="json"))


class CapabilityActivation(_CapabilityModel):
    """Compare-and-swap pointer to an immutable capability revision."""

    namespace: str = Field(min_length=1, max_length=200)
    capability_name: CapabilityName
    revision: int = Field(ge=1)
    manifest_digest: SchemaDigest
    generation: int = Field(ge=1)
    activated_at: str = Field(min_length=1, max_length=100)
    config: dict[str, Any] = Field(default_factory=dict)
    secret_handles: dict[str, str] = Field(default_factory=dict)
    secret_bindings: dict[str, CapabilitySecretBinding] = Field(default_factory=dict)
    agent_binding: AgentRunBinding | None = None

    @model_validator(mode="after")
    def validate_agent_binding_tenant(self) -> Self:
        if (
            self.agent_binding is not None
            and self.agent_binding.agent_spec.tenant_id != self.namespace
        ):
            raise ValueError("capability agent binding belongs to a different tenant")
        return self


class CapabilityTestCheck(_CapabilityModel):
    """One sanitized evaluator assertion for an immutable revision."""

    name: str = Field(min_length=1, max_length=200)
    status: CapabilityTestStatus
    message: str = Field(default="", max_length=1_000)


class CapabilityTestResult(_CapabilityModel):
    """Persisted test attestation bound to one manifest digest."""

    namespace: str = Field(min_length=1, max_length=200)
    capability_name: CapabilityName
    revision: int = Field(ge=1)
    manifest_digest: SchemaDigest
    status: CapabilityTestStatus
    checks: tuple[CapabilityTestCheck, ...] = Field(min_length=1, max_length=200)
    tested_at: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        has_failure = any(check.status is CapabilityTestStatus.FAILED for check in self.checks)
        if has_failure != (self.status is CapabilityTestStatus.FAILED):
            raise ValueError("capability test status must match its checks")
        return self


__all__ = [
    "AgentInterfaceBinding",
    "CapabilityActivation",
    "CapabilityActivationState",
    "CapabilityManifest",
    "CapabilityTestCheck",
    "CapabilityTestResult",
    "CapabilityTestStatus",
    "EvalCommand",
    "HealthCheck",
    "NetworkPolicy",
    "ResourceLimits",
    "SecretRequirement",
    "ToolExport",
    "WorkerKind",
    "WorkerRuntime",
    "WorkerSpec",
    "WorkerTransport",
    "canonical_json_digest",
    "is_reserved_worker_environment_name",
]
