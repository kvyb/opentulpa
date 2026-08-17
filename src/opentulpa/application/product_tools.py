"""Concrete application boundary for the model-visible product tools."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import shlex
import threading
from builtins import list as list_type
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Protocol, cast

from pydantic import BaseModel, SecretStr

from opentulpa.inference.models import ResolvedInferencePlan
from opentulpa.integrations.tenant_composio import ComposioProviderError
from opentulpa.integrations.web_search import WebSearchProviderError
from opentulpa.repositories.providers import RepositorySandboxError
from opentulpa.repositories.service import RepositoryWorkspaceError
from opentulpa.sandbox.client import SandboxSecretFileMount
from opentulpa.schedules.models import ScheduleWrite
from opentulpa.secrets.models import SecretHandle, SecretState
from opentulpa.secrets.vault import SecretGrantError
from opentulpa.specs.models import AgentSpecWrite, TriggerSpec, TriggerSpecWrite
from opentulpa.specs.protocol import AgentSpecRef
from opentulpa.tooling.adapters import (
    ProductToolApplicationError,
    ProductToolInvocation,
    ProductToolOutput,
)

_OWNER_KEYS = frozenset({"customer_id", "namespace", "tenant_id"})
_INTERNAL_KEYS = frozenset(
    {
        "absolute_path",
        "actor_id",
        "arguments",
        "correlation_id",
        "created_by_actor_id",
        "idempotency_key",
        "internal_path",
        "local_path",
        "profile_dir",
        "provider_workspace_id",
        "request_headers",
        "response_headers",
        "raw_response",
        "stored_path",
        "thread_id",
        "updated_by_actor_id",
        "user_id",
        "user_data_dir",
        "worktree_path",
        "workspace_root",
    }
)
_SECRET_KEY_RE = re.compile(
    r"(authorization|api[_-]?key|cookie|password|passwd|secret|token)",
    re.IGNORECASE,
)
_PUBLIC_SECRET_METADATA_KEYS = frozenset({"secret_scopes"})
_PROFILE_FIELDS = frozenset({"directive_text", "locale", "utc_offset"})
_SANDBOX_SSH_SECRET_SCOPES = ("ssh.connect", "credential.use")
_SSH_HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}\Z")
_SSH_USER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}\Z")


class ProfilePort(Protocol):
    def get(self, *, tenant_id: str) -> Any: ...

    def update(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        updates: Mapping[str, Any],
        idempotency_key: str,
    ) -> Any: ...


class FilePort(Protocol):
    def search(self, *, tenant_id: str, query: str, limit: int) -> Any: ...

    def get(self, *, tenant_id: str, file_id: str) -> Any: ...

    def inspect(self, *, tenant_id: str, file_id: str, question: str | None) -> Any: ...


class ArtifactPort(Protocol):
    def get(self, *, tenant_id: str, artifact_id: str) -> Any: ...

    def deliver(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        thread_id: str,
        channel: str,
        artifact_id: str,
        caption: str | None,
        idempotency_key: str,
    ) -> Any: ...


class KnowledgePort(Protocol):
    def list(self, *, tenant_id: str, include_archived: bool, limit: int) -> Any: ...

    def find(self, *, tenant_id: str, query: str, limit: int) -> Any: ...

    def get(self, *, tenant_id: str, source_id: str) -> Any: ...

    def archive(
        self,
        *,
        tenant_id: str,
        source_id: str,
        idempotency_key: str,
    ) -> Any: ...

    def query(
        self,
        *,
        tenant_id: str,
        query: str,
        source_ids: list_type[str],
        limit: int,
    ) -> Any: ...


class ResearchPort(Protocol):
    def search(self, *, tenant_id: str, query: str, limit: int) -> Any: ...

    def fetch(self, *, tenant_id: str, url: str) -> Any: ...


class BrowserPort(Protocol):
    def get(self, *, tenant_id: str, session_id: str) -> Any: ...

    def stop(
        self,
        *,
        tenant_id: str,
        session_id: str,
        idempotency_key: str,
    ) -> Any: ...


class IntegrationPort(Protocol):
    def list_integrations(self, *, tenant_id: str, query: str | None) -> Any: ...

    def connect(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        integration_id: str,
        redirect_url: str | None,
        idempotency_key: str,
    ) -> Any: ...

    def list_connections(self, *, tenant_id: str, integration_id: str | None) -> Any: ...

    def get_connection(self, *, tenant_id: str, connection_id: str) -> Any: ...

    def disconnect(
        self,
        *,
        tenant_id: str,
        connection_id: str,
        idempotency_key: str,
    ) -> Any: ...

    def search_actions(
        self,
        *,
        tenant_id: str,
        query: str,
        integration_id: str | None,
        limit: int,
    ) -> Any: ...


class IntakePort(Protocol):
    def list_workflows(self, *, tenant_id: str, include_inactive: bool) -> Any: ...

    def get_workflow(self, *, tenant_id: str, workflow_id: str | None) -> Any: ...

    def get_draft(self, *, tenant_id: str, draft_id: str) -> Any: ...

    def save_draft(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str | None,
        expected_revision: int | None,
        patch: Mapping[str, Any],
        idempotency_key: str,
    ) -> Any: ...

    def prepare_draft(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> Any: ...

    def activate_draft(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token: str,
        idempotency_key: str,
    ) -> Any: ...

    def delete_workflow(
        self,
        *,
        tenant_id: str,
        workflow_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> Any: ...


class SchedulePort(Protocol):
    def list(self, *, tenant_id: str) -> Any: ...

    def get(self, *, tenant_id: str, schedule_id: str) -> Any: ...

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        write: ScheduleWrite,
        schedule_id: str | None,
        expected_revision: int | None,
        idempotency_key: str,
    ) -> Any: ...

    def delete(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        schedule_id: str,
        expected_revision: int,
        idempotency_key: str,
    ) -> Any: ...


class AgentSpecPort(Protocol):
    def list_latest(self, *, tenant_id: str) -> Any: ...

    def get_active(self, *, tenant_id: str, spec_id: str) -> Any: ...

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        write: AgentSpecWrite,
        spec_id: str | None,
        expected_revision: int | None,
    ) -> Any: ...

    def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        spec_id: str,
        revision: int,
        expected_active_revision: int | None,
    ) -> Any: ...

    def rollback(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        spec_id: str,
        expected_active_revision: int,
    ) -> Any: ...


class TriggerSpecPort(Protocol):
    def list_latest(self, *, tenant_id: str) -> Any: ...

    def get_active(self, *, tenant_id: str, trigger_id: str) -> Any: ...

    def save(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        write: TriggerSpecWrite,
        trigger_id: str | None,
        expected_revision: int | None,
    ) -> Any: ...

    def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        trigger_id: str,
        revision: int,
        expected_active_revision: int | None,
    ) -> Any: ...

    def rollback(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        trigger_id: str,
        expected_active_revision: int,
    ) -> Any: ...


class SecretHandlePort(Protocol):
    def list(self, *, tenant_id: str) -> Any: ...

    def get(self, *, tenant_id: str, secret_id: str) -> Any: ...

    def resolve_for_sandbox(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
        scope: str,
        mount_type: str,
    ) -> SecretStr: ...

    def resolve_for_runtime_environment(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
        environment_name: str,
    ) -> SecretStr: ...

    def revoke(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
        expected_revision: int,
    ) -> Any: ...


class SandboxExecutionPort(Protocol):
    def execute(
        self,
        *,
        tenant_id: str,
        command: str,
        timeout: int,
        workspace: Path | None = None,
        cancel_event: threading.Event | None = None,
        secret_files: tuple[SandboxSecretFileMount, ...] = (),
    ) -> Any: ...


class CapabilityPort(Protocol):
    def list(self, *, tenant_id: str) -> Any: ...

    def seed_bundled(self, *, tenant_id: str, actor_id: str) -> Any: ...

    async def test(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        revision: int,
    ) -> Any: ...

    async def activate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        revision: int,
        expected_generation: int | None,
        config: Mapping[str, Any] | None = None,
        secret_handles: Mapping[str, str] | None = None,
        refresh_agent_binding: bool = False,
    ) -> Any: ...

    async def rollback(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        expected_generation: int,
        config: Mapping[str, Any] | None = None,
        secret_handles: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def deactivate(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        capability_name: str,
        expected_generation: int,
    ) -> Any: ...


class JobPort(Protocol):
    def create(
        self,
        *,
        tenant_id: str,
        handler_name: str,
        arguments: Mapping[str, Any],
        idempotency_key: str,
    ) -> Any: ...

    def get(self, *, tenant_id: str, job_id: str) -> Any: ...

    def events(
        self,
        *,
        tenant_id: str,
        job_id: str,
        after_sequence: int,
        limit: int,
    ) -> Any: ...

    def artifacts(self, *, tenant_id: str, job_id: str) -> Any: ...

    def cancel(self, *, tenant_id: str, job_id: str, idempotency_key: str) -> Any: ...


class RepositoryPort(Protocol):
    async def open(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        repository_url: str,
        base_ref: str,
        branch: str | None,
        provider: str | None,
    ) -> Any: ...

    async def list(self, *, tenant_id: str, include_closed: bool) -> Any: ...

    async def status(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str | None,
    ) -> Any: ...

    async def close(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str | None,
    ) -> Any: ...

    async def publish(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str | None,
        expected_head_sha: str,
        title: str,
        body: str,
        draft: bool,
    ) -> Any: ...

    async def import_verified_patch(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        workspace_id: str,
        patch: bytes,
        expected_sha256: str,
        message: str,
        source_candidate_id: str | None = None,
        source_commit: str | None = None,
        expected_tree_oid: str | None = None,
    ) -> Any: ...


class EvolutionPort(Protocol):
    async def source_status(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_read(
        self,
        *,
        path: str,
        offset: int = 1,
        limit: int = 2_000,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_write(
        self,
        *,
        path: str,
        content: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_edit(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_bash(
        self,
        *,
        command: str,
        timeout_seconds: int = 300,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_activate(
        self,
        *,
        idempotency_key: str,
        message: str = "OpenTulpa self-update",
        reason: str = "Trusted source activation",
        review_instructions: str,
        inference_plan: ResolvedInferencePlan | None = None,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_rollback(
        self,
        *,
        idempotency_key: str,
        expected_active_release_id: str,
        reason: str = "Owner requested rollback",
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_runtime_env_get(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...

    async def source_set_runtime_env(
        self,
        *,
        name: str,
        value: str,
        idempotency_key: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> Any: ...


class TracePort(Protocol):
    async def trace_list(
        self,
        *,
        tenant_id: str,
        status: str | None = None,
        limit: int = 20,
        before_run_id: str | None = None,
    ) -> Any: ...

    async def trace_get(
        self,
        *,
        tenant_id: str,
        run_id: str,
        after_sequence: int = 0,
        limit: int = 200,
        include_messages: bool = False,
    ) -> Any: ...


class IdempotencyPort(Protocol):
    """Durably claim an effect, replay its result, and reject key/request mismatches."""

    def execute(
        self,
        *,
        tenant_id: str,
        operation: str,
        idempotency_key: str,
        request_hash: str,
        invoke: Callable[[], Any],
    ) -> Any: ...


def _structured(value: Any, *, mode: str) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode=mode)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _assert_owned(value: Any, tenant_id: str) -> None:
    value = _structured(value, mode="python")
    if isinstance(value, Mapping):
        for owner_key in _OWNER_KEYS:
            owner = str(value.get(owner_key, "") or "").strip()
            if owner and owner != tenant_id:
                raise ProductToolApplicationError(
                    "not_found",
                    "The requested resource was not found.",
                )
        for nested in value.values():
            _assert_owned(nested, tenant_id)
    elif isinstance(value, list | tuple | set):
        for nested in value:
            _assert_owned(nested, tenant_id)


def _public_data(value: Any) -> Any:
    value = _structured(value, mode="json")
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for raw_key, nested in value.items():
            key = str(raw_key)
            if key in _OWNER_KEYS or key in _INTERNAL_KEYS:
                continue
            if key not in _PUBLIC_SECRET_METADATA_KEYS and _SECRET_KEY_RE.search(key):
                public[key] = "[redacted]"
                continue
            public[key] = _public_data(nested)
        return public
    if isinstance(value, list | tuple | set):
        return [_public_data(item) for item in value]
    if isinstance(value, bytes | bytearray | memoryview):
        return f"[binary:{len(value)} bytes]"
    if isinstance(value, Path):
        return "[internal path omitted]"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, str) and len(value) > 20_000:
        return f"{value[:20_000]}...[truncated]"
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _public_intake_preparation(value: Any) -> dict[str, Any]:
    structured = _structured(value, mode="json")
    if not isinstance(structured, Mapping):
        raise ProductToolApplicationError(
            "invalid_service_response",
            "The intake draft preparation result is invalid.",
        )
    confirmation_handle = str(structured.get("confirmation_token", "") or "").strip()
    if not confirmation_handle:
        raise ProductToolApplicationError(
            "invalid_service_response",
            "The intake draft confirmation handle is unavailable.",
        )
    without_token = dict(structured)
    without_token.pop("confirmation_token", None)
    public = _public_data(without_token)
    if not isinstance(public, dict):
        raise ProductToolApplicationError(
            "invalid_service_response",
            "The intake draft preparation result is invalid.",
        )
    public["confirmation_handle"] = confirmation_handle
    return public


def _public_secret_handle(value: Any) -> dict[str, Any]:
    """Whitelist public handle metadata so a port can never add secret material."""

    handle = value if isinstance(value, SecretHandle) else SecretHandle.model_validate(value)
    return handle.model_dump(mode="json", exclude={"tenant_id"})


def _public_capability_data(value: Any) -> Any:
    """Expose only secret requirement metadata and opaque handle identifiers."""

    value = _structured(value, mode="json")
    if isinstance(value, Mapping):
        public: dict[str, Any] = {}
        for raw_key, nested in value.items():
            key = str(raw_key)
            if key == "secrets":
                public["credential_requirements"] = _public_capability_data(nested)
                continue
            if key == "secret_handles":
                bindings = _structured(nested, mode="json")
                public["credential_bindings"] = (
                    [
                        {"name": str(name), "handle_id": str(handle_id)}
                        for name, handle_id in sorted(bindings.items())
                    ]
                    if isinstance(bindings, Mapping)
                    else []
                )
                continue
            public[key] = _public_capability_data(nested)
        return _public_data(public)
    if isinstance(value, list | tuple | set):
        return [_public_capability_data(item) for item in value]
    return _public_data(value)


def _resource_id(value: Any, *names: str) -> str:
    structured = _structured(value, mode="python")
    if isinstance(structured, Mapping):
        for name in names:
            candidate = str(structured.get(name, "") or "").strip()
            if candidate:
                return candidate
    for name in names:
        candidate = str(getattr(value, name, "") or "").strip()
        if candidate:
            return candidate
    return ""


def _resource_revision(value: Any) -> int | None:
    structured = _structured(value, mode="python")
    if isinstance(structured, Mapping):
        revision = structured.get("revision")
    else:
        revision = getattr(value, "revision", None)
    return int(revision) if revision is not None else None


def _connection_owner_values(value: Any) -> set[str]:
    value = _structured(value, mode="python")
    owners: set[str] = set()
    if isinstance(value, Mapping):
        for key in ("tenant_id", "customer_id", "user_id"):
            owner = str(value.get(key, "") or "").strip()
            if owner:
                owners.add(owner)
        for nested in value.values():
            owners.update(_connection_owner_values(nested))
    elif isinstance(value, list | tuple | set):
        for nested in value:
            owners.update(_connection_owner_values(nested))
    return owners


def _assert_connection_owned(value: Any, tenant_id: str) -> None:
    owners = _connection_owner_values(value)
    if not owners:
        raise ProductToolApplicationError(
            "invalid_service_response",
            "The connection ownership could not be verified.",
        )
    if owners != {tenant_id}:
        raise ProductToolApplicationError(
            "not_found",
            "The requested resource was not found.",
        )


def _secret_material(value: Any) -> SecretStr:
    if isinstance(value, SecretStr):
        return value
    return SecretStr(str(value or ""))


def _ssh_target(*, host: str, user: str) -> str:
    clean_host = str(host or "").strip()
    clean_user = str(user or "").strip()
    if _SSH_HOST_RE.fullmatch(clean_host) is None:
        raise ProductToolApplicationError(
            "invalid_request",
            "The SSH host is invalid.",
        )
    if _SSH_USER_RE.fullmatch(clean_user) is None:
        raise ProductToolApplicationError(
            "invalid_request",
            "The SSH user is invalid.",
        )
    return f"{clean_user}@{clean_host}"


def _sandbox_ssh_command(
    *,
    target: str,
    port: int,
    remote_command: str,
    secret_type: str,
) -> str:
    known_hosts = ".opentulpa-ssh-known-hosts"
    setup = f"umask 077 && : > {known_hosts} && "
    common_options = (
        f"-o UserKnownHostsFile=\"$PWD/{known_hosts}\" "
        "-o StrictHostKeyChecking=accept-new "
    )
    destination = f"-p {int(port)} {shlex.quote(target)} -- {shlex.quote(remote_command)}"
    if secret_type == "private_key":
        return (
            f"{setup}{{ ssh -i \"$OPENTULPA_SSH_IDENTITY\" "
            f"-o IdentitiesOnly=yes {common_options}{destination}; "
            f"status=$?; rm -f {known_hosts}; exit $status; }}"
        )
    if secret_type == "password":
        helper = ".opentulpa-ssh-askpass"
        helper_body = 'exec /bin/cat -- "$OPENTULPA_SSH_PASSWORD_FILE"'
        return (
            f"{setup}"
            f"printf '%s\\n' '#!/bin/sh' {shlex.quote(helper_body)} > {helper} && "
            f"chmod 700 {helper} && {{ "
            "DISPLAY=opentulpa SSH_ASKPASS_REQUIRE=force "
            f"SSH_ASKPASS=\"$PWD/{helper}\" ssh "
            "-o BatchMode=no "
            "-o PubkeyAuthentication=no "
            "-o PasswordAuthentication=yes "
            "-o KbdInteractiveAuthentication=yes "
            "-o PreferredAuthentications=password,keyboard-interactive "
            "-o NumberOfPasswordPrompts=1 "
            f"{common_options}{destination}; status=$?; "
            f"rm -f {helper} {known_hosts}; exit $status; }}"
        )
    raise ProductToolApplicationError(
        "invalid_request",
        "The SSH secret type is invalid.",
    )


async def _resolve(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await cast(Awaitable[Any], value)
    return value


class ProductToolApplication:
    """Call tenant-scoped application ports without an agent/runtime dependency."""

    def __init__(
        self,
        *,
        profiles: ProfilePort,
        files: FilePort,
        artifacts: ArtifactPort,
        knowledge: KnowledgePort,
        research: ResearchPort,
        browser: BrowserPort,
        integrations: IntegrationPort,
        intake: IntakePort,
        schedules: SchedulePort,
        jobs: JobPort,
        idempotency: IdempotencyPort,
        repositories: RepositoryPort | None = None,
        evolution: EvolutionPort | None = None,
        traces: TracePort | None = None,
        evolution_owner_tenant_id: str | None = None,
        agent_specs: AgentSpecPort | None = None,
        trigger_specs: TriggerSpecPort | None = None,
        secret_handles: SecretHandlePort | None = None,
        sandbox_execution: SandboxExecutionPort | None = None,
        capabilities: CapabilityPort | None = None,
        on_trigger_spec_changed: Callable[[TriggerSpec], Any] | None = None,
    ) -> None:
        self._profiles = profiles
        self._files = files
        self._artifacts = artifacts
        self._knowledge = knowledge
        self._research = research
        self._browser = browser
        self._integrations = integrations
        self._intake = intake
        self._schedules = schedules
        self._jobs = jobs
        self._idempotency = idempotency
        self._repositories = repositories
        self._evolution = evolution
        self._traces = traces
        self._evolution_owner_tenant_id = str(evolution_owner_tenant_id or "").strip()
        self._agent_specs = agent_specs
        self._trigger_specs = trigger_specs
        self._secret_handles = secret_handles
        self._sandbox_execution = sandbox_execution
        self._capabilities = capabilities
        self._on_trigger_spec_changed = on_trigger_spec_changed

    def _require_repositories(self, invocation: ProductToolInvocation) -> RepositoryPort:
        if invocation.context.run_kind != "owner" or self._repositories is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Repository workspaces are unavailable in this deployment.",
            )
        return self._repositories

    def _require_evolution(self, invocation: ProductToolInvocation) -> EvolutionPort:
        if (
            invocation.context.run_kind != "owner"
            or self._evolution is None
            or not self._evolution_owner_tenant_id
            or invocation.context.tenant_id != self._evolution_owner_tenant_id
        ):
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Source evolution is unavailable in this deployment.",
            )
        return self._evolution

    def _require_traces(self, invocation: ProductToolInvocation) -> TracePort:
        if invocation.context.run_kind != "owner" or self._traces is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Trace inspection is unavailable in this deployment.",
            )
        return self._traces

    def _require_agent_specs(self, invocation: ProductToolInvocation) -> AgentSpecPort:
        if invocation.context.run_kind != "owner" or self._agent_specs is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "AgentSpec management is unavailable in this deployment.",
            )
        return self._agent_specs

    def _require_trigger_specs(self, invocation: ProductToolInvocation) -> TriggerSpecPort:
        if invocation.context.run_kind != "owner" or self._trigger_specs is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "TriggerSpec management is unavailable in this deployment.",
            )
        return self._trigger_specs

    def _require_secret_handles(self, invocation: ProductToolInvocation) -> SecretHandlePort:
        if invocation.context.run_kind != "owner" or self._secret_handles is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Secret handle management is unavailable in this deployment.",
            )
        return self._secret_handles

    def _require_sandbox_execution(
        self,
        invocation: ProductToolInvocation,
    ) -> SandboxExecutionPort:
        if invocation.context.run_kind != "owner" or self._sandbox_execution is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Sandbox shell execution is unavailable in this deployment.",
                retryable=True,
            )
        return self._sandbox_execution

    def _require_capabilities(self, invocation: ProductToolInvocation) -> CapabilityPort:
        if invocation.context.run_kind != "owner" or self._capabilities is None:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Capability management is unavailable in this deployment.",
            )
        return self._capabilities

    @staticmethod
    def _required_key(invocation: ProductToolInvocation) -> str:
        key = str(invocation.idempotency_key or "").strip()
        if not key:
            raise ProductToolApplicationError(
                "idempotency_key_required",
                "An idempotency key is required for this operation.",
            )
        return key

    @staticmethod
    async def _call(
        invocation: ProductToolInvocation,
        call: Callable[[], Any],
        *,
        required: bool = False,
    ) -> Any:
        try:
            value = await _resolve(call())
        except ProductToolApplicationError:
            raise
        except ComposioProviderError as exc:
            raise ProductToolApplicationError(
                "integration_provider_failed",
                "The integration provider request failed.",
                retryable=True,
            ) from exc
        except PermissionError as exc:
            raise ProductToolApplicationError(
                "not_found",
                "The requested resource was not found.",
            ) from exc
        except LookupError as exc:
            raise ProductToolApplicationError(
                "not_found",
                "The requested resource was not found.",
            ) from exc
        except ValueError as exc:
            if "conflict" in type(exc).__name__.lower():
                raise ProductToolApplicationError(
                    "conflict",
                    "The resource changed. Refresh it and retry with the current revision.",
                ) from exc
            raise ProductToolApplicationError(
                "invalid_request",
                "The request is invalid or conflicts with the current resource revision.",
            ) from exc
        except (RepositoryWorkspaceError, RepositorySandboxError) as exc:
            raise ProductToolApplicationError(
                "repository_workspace_error",
                str(exc)[:500],
                retryable="unavailable" in type(exc).__name__.casefold(),
            ) from exc
        except Exception as exc:
            name = type(exc).__name__.lower()
            if "notfound" in name or "not_found" in name:
                raise ProductToolApplicationError(
                    "not_found",
                    "The requested resource was not found.",
                ) from exc
            if "conflict" in name:
                raise ProductToolApplicationError(
                    "conflict",
                    "The resource changed. Refresh it and retry with the current revision.",
                ) from exc
            if "capabilitytestrequired" in name:
                raise ProductToolApplicationError(
                    "capability_test_required",
                    "The exact capability revision must pass isolated tests before activation.",
                ) from exc
            if "capabilityevaluationunavailable" in name:
                raise ProductToolApplicationError(
                    "capability_unavailable",
                    "Isolated capability testing is unavailable in this deployment.",
                    retryable=True,
                ) from exc
            if "capabilityruntimeunavailable" in name:
                raise ProductToolApplicationError(
                    "capability_unavailable",
                    "The capability runtime is unavailable in this deployment.",
                    retryable=True,
                ) from exc
            if "workerlifecycle" in name:
                raise ProductToolApplicationError(
                    "capability_runtime_failed",
                    "The capability worker did not complete its lifecycle transition.",
                    retryable=True,
                ) from exc
            raise
        if required and value is None:
            raise ProductToolApplicationError(
                "not_found",
                "The requested resource was not found.",
            )
        _assert_owned(value, invocation.context.tenant_id)
        return value

    @classmethod
    async def _output(
        cls,
        invocation: ProductToolInvocation,
        call: Callable[[], Any],
        *,
        required: bool = False,
    ) -> ProductToolOutput:
        value = await cls._call(invocation, call, required=required)
        return ProductToolOutput(data=_public_data(value))

    async def _submit_job(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        key = self._required_key(invocation)
        job = await self._call(
            invocation,
            lambda: self._jobs.create(
                tenant_id=invocation.context.tenant_id,
                handler_name=invocation.spec.name,
                arguments=dict(invocation.arguments),
                idempotency_key=key,
            ),
            required=True,
        )
        job_id = _resource_id(job, "id", "job_id")
        if not job_id:
            raise ProductToolApplicationError(
                "invalid_service_response",
                "The background job was not accepted.",
            )
        status = _resource_id(job, "status") or "queued"
        return ProductToolOutput(
            data={"job_id": job_id, "status": status},
            job_id=job_id,
        )

    async def _mutate_trigger_spec(
        self,
        invocation: ProductToolInvocation,
        mutate: Callable[[], Any],
    ) -> Any:
        trigger = await self._call(invocation, mutate, required=True)
        if self._on_trigger_spec_changed is not None:
            await _resolve(self._on_trigger_spec_changed(TriggerSpec.model_validate(trigger)))
        return trigger

    async def _idempotent_output(
        self,
        invocation: ProductToolInvocation,
        invoke: Callable[[], Any],
        *,
        project: Callable[[Any], Any] = _public_data,
    ) -> ProductToolOutput:
        key = self._required_key(invocation)
        canonical = json.dumps(
            {
                "operation": invocation.spec.name,
                "version": invocation.spec.version,
                "arguments": invocation.arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        request_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        async def invoke_public_result() -> dict[str, Any]:
            value = await _resolve(invoke())
            _assert_owned(value, invocation.context.tenant_id)
            public = project(value)
            return dict(public) if isinstance(public, Mapping) else {"value": public}

        value = await self._call(
            invocation,
            lambda: self._idempotency.execute(
                tenant_id=invocation.context.tenant_id,
                operation=invocation.spec.name,
                idempotency_key=key,
                request_hash=request_hash,
                invoke=invoke_public_result,
            ),
        )
        return ProductToolOutput(data=_public_data(value))

    async def _idempotent_with_resource(
        self,
        invocation: ProductToolInvocation,
        *,
        read: Callable[[], Any],
        mutate: Callable[[], Any],
        validate: Callable[[Any], None] | None = None,
        project: Callable[[Any], Any] = _public_data,
    ) -> ProductToolOutput:
        async def invoke() -> Any:
            resource = await self._call(invocation, read, required=True)
            if validate is not None:
                validate(resource)
            return await self._call(invocation, mutate)

        return await self._idempotent_output(invocation, invoke, project=project)

    async def profile_get(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._profiles.get(tenant_id=invocation.context.tenant_id),
        )

    async def profile_update(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        updates = dict(cast(Mapping[str, Any], invocation.arguments["updates"]))
        unsupported = sorted(set(updates) - _PROFILE_FIELDS)
        if unsupported:
            raise ProductToolApplicationError(
                "invalid_request",
                "Profile updates support only directive_text, locale, and utc_offset.",
            )
        key = self._required_key(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: self._profiles.update(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                updates=updates,
                idempotency_key=key,
            ),
        )

    async def file_search(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._files.search(
                tenant_id=invocation.context.tenant_id,
                query=str(invocation.arguments["query"]),
                limit=int(invocation.arguments["limit"]),
            ),
        )

    async def file_get(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._files.get(
                tenant_id=invocation.context.tenant_id,
                file_id=str(invocation.arguments["file_id"]),
            ),
            required=True,
        )

    async def file_analyze(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        await self._call(
            invocation,
            lambda: self._files.get(
                tenant_id=invocation.context.tenant_id,
                file_id=str(invocation.arguments["file_id"]),
            ),
            required=True,
        )
        return await self._submit_job(invocation)

    async def file_inspect(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        file_id = str(invocation.arguments["file_id"])
        await self._call(
            invocation,
            lambda: self._files.get(
                tenant_id=invocation.context.tenant_id,
                file_id=file_id,
            ),
            required=True,
        )
        return await self._output(
            invocation,
            lambda: self._files.inspect(
                tenant_id=invocation.context.tenant_id,
                file_id=file_id,
                question=cast("str | None", invocation.arguments.get("question")),
            ),
        )

    async def artifact_deliver(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        artifact_id = str(invocation.arguments["artifact_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._artifacts.get(
                tenant_id=invocation.context.tenant_id,
                artifact_id=artifact_id,
            ),
            mutate=lambda: self._artifacts.deliver(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                thread_id=invocation.context.thread_id,
                channel=invocation.context.channel,
                artifact_id=artifact_id,
                caption=cast("str | None", invocation.arguments.get("caption")),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def knowledge_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._knowledge.list(
                tenant_id=invocation.context.tenant_id,
                include_archived=bool(invocation.arguments["include_archived"]),
                limit=int(invocation.arguments["limit"]),
            ),
        )

    async def knowledge_find(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._knowledge.find(
                tenant_id=invocation.context.tenant_id,
                query=str(invocation.arguments["query"]),
                limit=int(invocation.arguments["limit"]),
            ),
        )

    async def knowledge_attach(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        await self._call(
            invocation,
            lambda: self._files.get(
                tenant_id=invocation.context.tenant_id,
                file_id=str(invocation.arguments["file_id"]),
            ),
            required=True,
        )
        return await self._submit_job(invocation)

    async def knowledge_archive(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        source_id = str(invocation.arguments["source_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._knowledge.get(
                tenant_id=invocation.context.tenant_id,
                source_id=source_id,
            ),
            mutate=lambda: self._knowledge.archive(
                tenant_id=invocation.context.tenant_id,
                source_id=source_id,
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def knowledge_reindex(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        source_id = cast("str | None", invocation.arguments.get("source_id"))
        if source_id:
            await self._call(
                invocation,
                lambda: self._knowledge.get(
                    tenant_id=invocation.context.tenant_id,
                    source_id=source_id,
                ),
                required=True,
            )
        return await self._submit_job(invocation)

    async def knowledge_query(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        source_ids = [str(value) for value in invocation.arguments.get("source_ids", [])]
        for source_id in source_ids:
            await self._call(
                invocation,
                partial(
                    self._knowledge.get,
                    tenant_id=invocation.context.tenant_id,
                    source_id=source_id,
                ),
                required=True,
            )
        return await self._output(
            invocation,
            lambda: self._knowledge.query(
                tenant_id=invocation.context.tenant_id,
                query=str(invocation.arguments["query"]),
                source_ids=source_ids,
                limit=int(invocation.arguments["limit"]),
            ),
        )

    async def web_search(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        try:
            return await self._output(
                invocation,
                lambda: self._research.search(
                    tenant_id=invocation.context.tenant_id,
                    query=str(invocation.arguments["query"]),
                    limit=int(invocation.arguments["limit"]),
                ),
            )
        except WebSearchProviderError as exc:
            raise ProductToolApplicationError(
                "web_search_failed",
                exc.public_message,
                retryable=exc.retryable,
            ) from exc

    async def content_fetch(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._research.fetch(
                tenant_id=invocation.context.tenant_id,
                url=str(invocation.arguments["url"]),
            ),
        )

    async def browser_start(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._submit_job(invocation)

    async def browser_get(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._browser.get(
                tenant_id=invocation.context.tenant_id,
                session_id=str(invocation.arguments["session_id"]),
            ),
            required=True,
        )

    async def browser_act(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        await self._call(
            invocation,
            lambda: self._browser.get(
                tenant_id=invocation.context.tenant_id,
                session_id=str(invocation.arguments["session_id"]),
            ),
            required=True,
        )
        return await self._submit_job(invocation)

    async def browser_stop(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        session_id = str(invocation.arguments["session_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._browser.get(
                tenant_id=invocation.context.tenant_id,
                session_id=session_id,
            ),
            mutate=lambda: self._browser.stop(
                tenant_id=invocation.context.tenant_id,
                session_id=session_id,
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def integration_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._integrations.list_integrations(
                tenant_id=invocation.context.tenant_id,
                query=cast("str | None", invocation.arguments.get("query")),
            ),
        )

    async def integration_connect(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._idempotent_output(
            invocation,
            lambda: self._integrations.connect(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                integration_id=str(invocation.arguments["integration_id"]),
                redirect_url=cast("str | None", invocation.arguments.get("redirect_url")),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def connection_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._integrations.list_connections(
                tenant_id=invocation.context.tenant_id,
                integration_id=cast("str | None", invocation.arguments.get("integration_id")),
            ),
        )

    async def connection_disconnect(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        connection_id = str(invocation.arguments["connection_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._integrations.get_connection(
                tenant_id=invocation.context.tenant_id,
                connection_id=connection_id,
            ),
            mutate=lambda: self._integrations.disconnect(
                tenant_id=invocation.context.tenant_id,
                connection_id=connection_id,
                idempotency_key=self._required_key(invocation),
            ),
            validate=lambda connection: _assert_connection_owned(
                connection,
                invocation.context.tenant_id,
            ),
        )

    async def integration_action_search(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._integrations.search_actions(
                tenant_id=invocation.context.tenant_id,
                query=str(invocation.arguments["query"]),
                integration_id=cast("str | None", invocation.arguments.get("integration_id")),
                limit=int(invocation.arguments["limit"]),
            ),
        )

    async def integration_invoke(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        connection = await self._call(
            invocation,
            lambda: self._integrations.get_connection(
                tenant_id=invocation.context.tenant_id,
                connection_id=str(invocation.arguments["connection_id"]),
            ),
            required=True,
        )
        _assert_connection_owned(connection, invocation.context.tenant_id)
        return await self._submit_job(invocation)

    async def intake_workflow_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._intake.list_workflows(
                tenant_id=invocation.context.tenant_id,
                include_inactive=bool(invocation.arguments["include_inactive"]),
            ),
        )

    async def intake_workflow_get(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._intake.get_workflow(
                tenant_id=invocation.context.tenant_id,
                workflow_id=cast("str | None", invocation.arguments.get("workflow_id")),
            ),
            required=True,
        )

    async def intake_draft_save(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._idempotent_output(
            invocation,
            lambda: self._intake.save_draft(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                draft_id=cast("str | None", invocation.arguments.get("draft_id")),
                expected_revision=cast("int | None", invocation.arguments.get("expected_revision")),
                patch=cast(Mapping[str, Any], invocation.arguments["patch"]),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def intake_draft_prepare(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        draft_id = str(invocation.arguments["draft_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._intake.get_draft(
                tenant_id=invocation.context.tenant_id,
                draft_id=draft_id,
            ),
            mutate=lambda: self._intake.prepare_draft(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                draft_id=draft_id,
                expected_revision=int(invocation.arguments["expected_revision"]),
                idempotency_key=self._required_key(invocation),
            ),
            project=_public_intake_preparation,
        )

    async def intake_draft_activate(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        draft_id = str(invocation.arguments["draft_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._intake.get_draft(
                tenant_id=invocation.context.tenant_id,
                draft_id=draft_id,
            ),
            mutate=lambda: self._intake.activate_draft(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                draft_id=draft_id,
                expected_revision=int(invocation.arguments["expected_revision"]),
                confirmation_token=str(invocation.arguments["confirmation_handle"]),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def intake_workflow_delete(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        workflow_id = str(invocation.arguments["workflow_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._intake.get_workflow(
                tenant_id=invocation.context.tenant_id,
                workflow_id=workflow_id,
            ),
            mutate=lambda: self._intake.delete_workflow(
                tenant_id=invocation.context.tenant_id,
                workflow_id=workflow_id,
                expected_revision=int(invocation.arguments["expected_revision"]),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def intake_workflow_test(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        workflow_id = cast("str | None", invocation.arguments.get("workflow_id"))
        draft_id = cast("str | None", invocation.arguments.get("draft_id"))
        if workflow_id:
            await self._call(
                invocation,
                lambda: self._intake.get_workflow(
                    tenant_id=invocation.context.tenant_id,
                    workflow_id=workflow_id,
                ),
                required=True,
            )
        elif draft_id:
            await self._call(
                invocation,
                lambda: self._intake.get_draft(
                    tenant_id=invocation.context.tenant_id,
                    draft_id=draft_id,
                ),
                required=True,
            )
        return await self._submit_job(invocation)

    async def schedule_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        schedules = await self._call(
            invocation,
            lambda: self._schedules.list(tenant_id=invocation.context.tenant_id),
        )
        if not bool(invocation.arguments["include_disabled"]) and isinstance(schedules, list):
            schedules = [
                item
                for item in schedules
                if bool(_structured(item, mode="python").get("enabled", False))
            ]
        return ProductToolOutput(data=_public_data(schedules))

    async def schedule_save(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        write = ScheduleWrite.model_validate(invocation.arguments["schedule"])
        return await self._idempotent_output(
            invocation,
            lambda: self._schedules.save(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                write=write,
                schedule_id=cast("str | None", invocation.arguments.get("schedule_id")),
                expected_revision=cast("int | None", invocation.arguments.get("expected_revision")),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def schedule_delete(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        schedule_id = str(invocation.arguments["schedule_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._schedules.get(
                tenant_id=invocation.context.tenant_id,
                schedule_id=schedule_id,
            ),
            mutate=lambda: self._schedules.delete(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                schedule_id=schedule_id,
                expected_revision=int(invocation.arguments["expected_revision"]),
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def agent_spec_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_agent_specs(invocation)
        specs = await self._call(
            invocation,
            lambda: service.list_latest(tenant_id=invocation.context.tenant_id),
        )
        if not isinstance(specs, list | tuple):
            raise ProductToolApplicationError(
                "invalid_service_response",
                "The AgentSpec list is invalid.",
            )
        views: list[dict[str, Any]] = []
        for spec in specs:
            spec_id = _resource_id(spec, "id", "spec_id")
            if not spec_id:
                raise ProductToolApplicationError(
                    "invalid_service_response",
                    "An AgentSpec identifier is unavailable.",
                )
            active = await self._call(
                invocation,
                partial(
                    service.get_active,
                    tenant_id=invocation.context.tenant_id,
                    spec_id=spec_id,
                ),
            )
            views.append(
                {
                    "spec": _public_data(spec),
                    "active_revision": _resource_revision(active),
                }
            )
        return ProductToolOutput(data=views)

    async def agent_spec_save(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_agent_specs(invocation)
        write = AgentSpecWrite.model_validate(invocation.arguments["spec"])
        return await self._idempotent_output(
            invocation,
            lambda: service.save(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                write=write,
                spec_id=cast("str | None", invocation.arguments.get("spec_id")),
                expected_revision=cast(
                    "int | None",
                    invocation.arguments.get("expected_revision"),
                ),
            ),
        )

    async def agent_spec_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_agent_specs(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: service.activate(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                spec_id=str(invocation.arguments["spec_id"]),
                revision=int(invocation.arguments["revision"]),
                expected_active_revision=cast(
                    "int | None",
                    invocation.arguments.get("expected_active_revision"),
                ),
            ),
        )

    async def agent_spec_rollback(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_agent_specs(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: service.rollback(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                spec_id=str(invocation.arguments["spec_id"]),
                expected_active_revision=int(invocation.arguments["expected_active_revision"]),
            ),
        )

    async def trigger_spec_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_trigger_specs(invocation)
        triggers = await self._call(
            invocation,
            lambda: service.list_latest(tenant_id=invocation.context.tenant_id),
        )
        if not isinstance(triggers, list | tuple):
            raise ProductToolApplicationError(
                "invalid_service_response",
                "The TriggerSpec list is invalid.",
            )
        views: list[dict[str, Any]] = []
        for trigger in triggers:
            trigger_id = _resource_id(trigger, "id", "trigger_id")
            if not trigger_id:
                raise ProductToolApplicationError(
                    "invalid_service_response",
                    "A TriggerSpec identifier is unavailable.",
                )
            active = await self._call(
                invocation,
                partial(
                    service.get_active,
                    tenant_id=invocation.context.tenant_id,
                    trigger_id=trigger_id,
                ),
            )
            views.append(
                {
                    "trigger": _public_data(trigger),
                    "active_revision": _resource_revision(active),
                }
            )
        return ProductToolOutput(data=views)

    async def trigger_spec_save(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_trigger_specs(invocation)
        payload = dict(cast(Mapping[str, Any], invocation.arguments["trigger"]))
        target = cast(Mapping[str, Any], payload.pop("agent_spec"))
        write = TriggerSpecWrite.model_validate(
            {
                **payload,
                "agent_spec": AgentSpecRef(
                    tenant_id=invocation.context.tenant_id,
                    spec_id=str(target["spec_id"]),
                    revision=int(target["revision"]),
                ),
            }
        )
        return await self._idempotent_output(
            invocation,
            lambda: service.save(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                write=write,
                trigger_id=cast("str | None", invocation.arguments.get("trigger_id")),
                expected_revision=cast(
                    "int | None",
                    invocation.arguments.get("expected_revision"),
                ),
            ),
        )

    async def trigger_spec_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_trigger_specs(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: self._mutate_trigger_spec(
                invocation,
                lambda: service.activate(
                    tenant_id=invocation.context.tenant_id,
                    actor_id=invocation.context.actor_id,
                    trigger_id=str(invocation.arguments["trigger_id"]),
                    revision=int(invocation.arguments["revision"]),
                    expected_active_revision=cast(
                        "int | None",
                        invocation.arguments.get("expected_active_revision"),
                    ),
                ),
            ),
        )

    async def trigger_spec_rollback(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_trigger_specs(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: self._mutate_trigger_spec(
                invocation,
                lambda: service.rollback(
                    tenant_id=invocation.context.tenant_id,
                    actor_id=invocation.context.actor_id,
                    trigger_id=str(invocation.arguments["trigger_id"]),
                    expected_active_revision=int(invocation.arguments["expected_active_revision"]),
                ),
            ),
        )

    async def secret_handle_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_secret_handles(invocation)
        handles = await self._call(
            invocation,
            lambda: service.list(tenant_id=invocation.context.tenant_id),
        )
        if not isinstance(handles, list | tuple):
            raise ProductToolApplicationError(
                "invalid_service_response",
                "The secret handle list is invalid.",
            )
        try:
            public = [_public_secret_handle(handle) for handle in handles]
        except (TypeError, ValueError) as exc:
            raise ProductToolApplicationError(
                "invalid_service_response",
                "The secret handle list is invalid.",
            ) from exc
        return ProductToolOutput(data=public)

    async def secret_handle_revoke(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_secret_handles(invocation)
        secret_id = str(invocation.arguments["secret_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: service.get(
                tenant_id=invocation.context.tenant_id,
                secret_id=secret_id,
            ),
            mutate=lambda: service.revoke(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                secret_id=secret_id,
                expected_revision=int(invocation.arguments["expected_revision"]),
            ),
            project=_public_secret_handle,
        )

    async def sandbox_ssh_diagnostic(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        sandbox = self._require_sandbox_execution(invocation)
        secret_service = self._require_secret_handles(invocation)
        secret_type = str(invocation.arguments.get("secret_type") or "private_key").strip()
        if secret_type not in {"password", "private_key"}:
            raise ProductToolApplicationError(
                "invalid_request",
                "The SSH secret type is invalid.",
            )
        secret_id = str(invocation.arguments["secret_id"])
        handle = await self._call(
            invocation,
            lambda: secret_service.get(
                tenant_id=invocation.context.tenant_id,
                secret_id=secret_id,
            ),
            required=True,
        )
        secret_handle = (
            handle if isinstance(handle, SecretHandle) else SecretHandle.model_validate(handle)
        )
        if secret_handle.state is not SecretState.ACTIVE:
            raise ProductToolApplicationError(
                "not_found",
                "The requested resource was not found.",
            )
        scope = next(
            (
                candidate
                for candidate in _SANDBOX_SSH_SECRET_SCOPES
                if candidate in secret_handle.scopes
            ),
            "",
        )
        if not scope:
            raise ProductToolApplicationError(
                "invalid_request",
                "The SSH secret handle is not scoped for sandbox SSH diagnostics.",
            )
        try:
            material = await _resolve(
                secret_service.resolve_for_sandbox(
                    tenant_id=invocation.context.tenant_id,
                    actor_id=invocation.context.actor_id,
                    secret_id=secret_handle.id,
                    scope=scope,
                    mount_type=f"ssh_{secret_type}",
                )
            )
        except Exception as exc:
            raise ProductToolApplicationError(
                "secret_unavailable",
                "The SSH secret handle could not be mounted.",
            ) from exc
        secret_value = _secret_material(material).get_secret_value()
        host = str(invocation.arguments["host"]).strip()
        user = str(invocation.arguments["user"]).strip()
        target = _ssh_target(host=host, user=user)
        command = _sandbox_ssh_command(
            target=target,
            port=int(invocation.arguments["port"]),
            remote_command=str(invocation.arguments["command"]),
            secret_type=secret_type,
        )
        if secret_type == "private_key":
            secret_file = SandboxSecretFileMount(
                name="id_opentulpa",
                content=secret_value,
                env="OPENTULPA_SSH_IDENTITY",
            )
        else:
            secret_file = SandboxSecretFileMount(
                name="ssh_password",
                content=secret_value,
                env="OPENTULPA_SSH_PASSWORD_FILE",
            )
        try:
            result = await asyncio.to_thread(
                sandbox.execute,
                tenant_id=invocation.context.tenant_id,
                command=command,
                timeout=int(invocation.arguments["timeout_seconds"]),
                secret_files=(secret_file,),
            )
        except TypeError as exc:
            raise ProductToolApplicationError(
                "capability_unavailable",
                "Sandbox secret mounts are unavailable in this deployment.",
                retryable=True,
            ) from exc
        except Exception as exc:
            raise ProductToolApplicationError(
                "sandbox_execution_failed",
                "The sandbox SSH diagnostic could not be executed.",
                retryable=True,
            ) from exc
        return ProductToolOutput(
            data={
                "host": host,
                "user": user,
                "port": int(invocation.arguments["port"]),
                "exit_code": int(getattr(result, "exit_code", 1)),
                "output": str(getattr(result, "output", "")),
                "truncated": bool(getattr(result, "truncated", False)),
            },
        )

    async def capability_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_capabilities(invocation)
        capabilities = await self._call(
            invocation,
            lambda: service.list(tenant_id=invocation.context.tenant_id),
        )
        if not isinstance(capabilities, list | tuple):
            raise ProductToolApplicationError(
                "invalid_service_response",
                "The capability list is invalid.",
            )
        return ProductToolOutput(data=_public_capability_data(capabilities))

    async def capability_seed_bundled(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_capabilities(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: {
                "capabilities": service.seed_bundled(
                    tenant_id=invocation.context.tenant_id,
                    actor_id=invocation.context.actor_id,
                )
            },
            project=_public_capability_data,
        )

    async def capability_test(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        service = self._require_capabilities(invocation)
        result = await self._call(
            invocation,
            lambda: service.test(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                capability_name=str(invocation.arguments["capability_name"]),
                revision=int(invocation.arguments["revision"]),
            ),
            required=True,
        )
        return ProductToolOutput(data=_public_capability_data(result))

    async def capability_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_capabilities(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: service.activate(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                capability_name=str(invocation.arguments["capability_name"]),
                revision=int(invocation.arguments["revision"]),
                expected_generation=cast(
                    "int | None",
                    invocation.arguments.get("expected_generation"),
                ),
                config=cast("Mapping[str, Any]", invocation.arguments["config"]),
                secret_handles=cast(
                    "Mapping[str, str]",
                    invocation.arguments["secret_handles"],
                ),
                **(
                    {"refresh_agent_binding": True}
                    if invocation.arguments.get("refresh_agent_binding", False)
                    else {}
                ),
            ),
            project=_public_capability_data,
        )

    async def capability_rollback(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_capabilities(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: service.rollback(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                capability_name=str(invocation.arguments["capability_name"]),
                expected_generation=int(invocation.arguments["expected_generation"]),
                config=cast(
                    "Mapping[str, Any] | None",
                    invocation.arguments.get("config"),
                ),
                secret_handles=cast(
                    "Mapping[str, str] | None",
                    invocation.arguments.get("secret_handles"),
                ),
            ),
            project=_public_capability_data,
        )

    async def capability_deactivate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        service = self._require_capabilities(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: service.deactivate(
                tenant_id=invocation.context.tenant_id,
                actor_id=invocation.context.actor_id,
                capability_name=str(invocation.arguments["capability_name"]),
                expected_generation=int(invocation.arguments["expected_generation"]),
            ),
            project=_public_capability_data,
        )

    async def job_get(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        return await self._output(
            invocation,
            lambda: self._jobs.get(
                tenant_id=invocation.context.tenant_id,
                job_id=str(invocation.arguments["job_id"]),
            ),
            required=True,
        )

    async def job_events(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        job_id = str(invocation.arguments["job_id"])
        await self._call(
            invocation,
            lambda: self._jobs.get(
                tenant_id=invocation.context.tenant_id,
                job_id=job_id,
            ),
            required=True,
        )
        return await self._output(
            invocation,
            lambda: self._jobs.events(
                tenant_id=invocation.context.tenant_id,
                job_id=job_id,
                after_sequence=int(invocation.arguments["after_sequence"]),
                limit=int(invocation.arguments["limit"]),
            ),
        )

    async def job_artifacts(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        job_id = str(invocation.arguments["job_id"])
        await self._call(
            invocation,
            lambda: self._jobs.get(
                tenant_id=invocation.context.tenant_id,
                job_id=job_id,
            ),
            required=True,
        )
        return await self._output(
            invocation,
            lambda: self._jobs.artifacts(
                tenant_id=invocation.context.tenant_id,
                job_id=job_id,
            ),
        )

    async def job_cancel(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        job_id = str(invocation.arguments["job_id"])
        return await self._idempotent_with_resource(
            invocation,
            read=lambda: self._jobs.get(
                tenant_id=invocation.context.tenant_id,
                job_id=job_id,
            ),
            mutate=lambda: self._jobs.cancel(
                tenant_id=invocation.context.tenant_id,
                job_id=job_id,
                idempotency_key=self._required_key(invocation),
            ),
        )

    async def repository_open(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        repositories = self._require_repositories(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: repositories.open(
                tenant_id=invocation.context.tenant_id,
                thread_id=invocation.context.thread_id,
                repository_url=str(invocation.arguments["repository_url"]),
                base_ref=str(invocation.arguments["base_ref"]),
                branch=cast("str | None", invocation.arguments.get("branch")),
                provider=cast("str | None", invocation.arguments.get("provider")),
            ),
        )

    async def repository_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        repositories = self._require_repositories(invocation)
        return await self._output(
            invocation,
            lambda: repositories.list(
                tenant_id=invocation.context.tenant_id,
                include_closed=bool(invocation.arguments["include_closed"]),
            ),
        )

    async def repository_status(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        repositories = self._require_repositories(invocation)
        return await self._output(
            invocation,
            lambda: repositories.status(
                tenant_id=invocation.context.tenant_id,
                thread_id=invocation.context.thread_id,
                workspace_id=cast("str | None", invocation.arguments.get("workspace_id")),
            ),
        )

    async def repository_close(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        repositories = self._require_repositories(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: repositories.close(
                tenant_id=invocation.context.tenant_id,
                thread_id=invocation.context.thread_id,
                workspace_id=cast("str | None", invocation.arguments.get("workspace_id")),
            ),
        )

    async def repository_publish_pr(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        repositories = self._require_repositories(invocation)
        return await self._idempotent_output(
            invocation,
            lambda: repositories.publish(
                tenant_id=invocation.context.tenant_id,
                thread_id=invocation.context.thread_id,
                workspace_id=cast("str | None", invocation.arguments.get("workspace_id")),
                expected_head_sha=str(invocation.arguments["expected_head_sha"]),
                title=str(invocation.arguments["title"]),
                body=str(invocation.arguments["body"]),
                draft=bool(invocation.arguments["draft"]),
            ),
        )

    async def source_status(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_status(
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_runtime_env_get(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_runtime_env_get(
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_read(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_read(
                path=str(invocation.arguments["path"]),
                offset=int(invocation.arguments["offset"]),
                limit=int(invocation.arguments["limit"]),
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_write(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_write(
                path=str(invocation.arguments["path"]),
                content=str(invocation.arguments["content"]),
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_edit(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_edit(
                path=str(invocation.arguments["path"]),
                old_text=str(invocation.arguments["old_text"]),
                new_text=str(invocation.arguments["new_text"]),
                replace_all=bool(invocation.arguments["replace_all"]),
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_bash(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_bash(
                command=str(invocation.arguments["command"]),
                timeout_seconds=int(invocation.arguments["timeout_seconds"]),
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_activate(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_activate(
                idempotency_key=self._required_key(invocation),
                message=str(invocation.arguments["message"]),
                reason=str(invocation.arguments["reason"]),
                review_instructions=str(invocation.arguments["review_instructions"]),
                inference_plan=invocation.inference_plan,
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_rollback(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)
        return await self._output(
            invocation,
            lambda: evolution.source_rollback(
                idempotency_key=self._required_key(invocation),
                expected_active_release_id=str(
                    invocation.arguments["expected_active_release_id"]
                ),
                reason=str(invocation.arguments["reason"]),
                audit_context=self._evolution_audit_context(invocation),
            ),
        )

    async def source_set_runtime_env(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput:
        evolution = self._require_evolution(invocation)

        async def update() -> Any:
            secret_id = invocation.arguments.get("secret_id")
            if secret_id is not None:
                secret_service = self._require_secret_handles(invocation)
                try:
                    secret = secret_service.resolve_for_runtime_environment(
                        tenant_id=invocation.context.tenant_id,
                        actor_id=invocation.context.actor_id,
                        secret_id=str(secret_id),
                        environment_name=str(invocation.arguments["name"]),
                    )
                except SecretGrantError as exc:
                    raise ProductToolApplicationError(
                        "secret_handle_unavailable",
                        "The runtime environment secret handle is unavailable or mismatched.",
                    ) from exc
                value = secret.get_secret_value()
            else:
                value = str(invocation.arguments["value"])
            result = await _resolve(
                evolution.source_set_runtime_env(
                    name=str(invocation.arguments["name"]),
                    value=value,
                    idempotency_key=self._required_key(invocation),
                    audit_context=self._evolution_audit_context(invocation),
                )
            )
            if isinstance(result, Mapping) and result.get("status") == "failed":
                raise ProductToolApplicationError(
                    "runtime_env_update_failed",
                    "The runtime environment update failed and was not applied.",
                    retryable=result.get("rollback_restored") is True,
                )
            return result

        output = await self._idempotent_output(invocation, update)
        if isinstance(output.data, Mapping) and output.data.get("status") == "failed":
            raise ProductToolApplicationError(
                "runtime_env_update_failed",
                "The previous runtime environment update failed and was not applied. "
                "Retry with a fresh idempotency key.",
                retryable=output.data.get("rollback_restored") is True,
            )
        return output

    async def trace_list(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        traces = self._require_traces(invocation)
        return await self._output(
            invocation,
            lambda: traces.trace_list(
                tenant_id=invocation.context.tenant_id,
                status=cast("str | None", invocation.arguments.get("status")),
                limit=int(invocation.arguments["limit"]),
                before_run_id=cast("str | None", invocation.arguments.get("before_run_id")),
            ),
        )

    async def trace_get(self, invocation: ProductToolInvocation) -> ProductToolOutput:
        traces = self._require_traces(invocation)
        return await self._output(
            invocation,
            lambda: traces.trace_get(
                tenant_id=invocation.context.tenant_id,
                run_id=str(invocation.arguments["run_id"]),
                after_sequence=int(invocation.arguments["after_sequence"]),
                limit=int(invocation.arguments["limit"]),
                include_messages=bool(invocation.arguments["include_messages"]),
            ),
            required=True,
        )

    @staticmethod
    def _evolution_audit_context(invocation: ProductToolInvocation) -> dict[str, str]:
        context = invocation.context
        return {
            "tenant_id": context.tenant_id,
            "actor_id": context.actor_id,
            "thread_id": context.thread_id,
            "correlation_id": context.correlation_id,
            "channel": context.channel,
            "run_kind": context.run_kind,
            "origin": context.origin.model_dump_json(),
        }


__all__ = [
    "AgentSpecPort",
    "ArtifactPort",
    "BrowserPort",
    "CapabilityPort",
    "EvolutionPort",
    "FilePort",
    "IdempotencyPort",
    "IntakePort",
    "IntegrationPort",
    "JobPort",
    "KnowledgePort",
    "ProductToolApplication",
    "ProfilePort",
    "ResearchPort",
    "SchedulePort",
    "SecretHandlePort",
    "TracePort",
    "TriggerSpecPort",
]
