# opentulpa: allow-test-credential
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from deepagents.backends.protocol import ExecuteResponse
from pydantic import SecretStr

from opentulpa.application.product_tools import ProductToolApplication, _sandbox_ssh_command
from opentulpa.integrations.web_search import WebSearchProviderError
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.persistence.idempotency import IdempotencyStore
from opentulpa.repositories.providers import RepositorySandboxUnavailableError
from opentulpa.secrets.models import SecretHandle, SecretState
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling.adapters import (
    ProductToolApplicationError,
    ProductToolInvocation,
    build_product_tools,
)
from opentulpa.tooling.contract import (
    TOOL_SPEC_BY_NAME,
    TOOL_SPECS,
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
)


class _Port:
    def __init__(self, **responses: Any) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Callable[..., Any]:
        if name.startswith("_"):
            raise AttributeError(name)

        def call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            response = self.responses.get(name, {"ok": True})
            if isinstance(response, Exception):
                raise response
            return response(**kwargs) if callable(response) else response

        return call


class _IdempotencyPort(_Port):
    async def execute(self, **kwargs: Any) -> Any:
        invoke = kwargs.pop("invoke")
        self.calls.append(("execute", kwargs))
        value = invoke()
        return await value if inspect.isawaitable(value) else value


class _SandboxExecutionPort:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> ExecuteResponse:
        self.calls.append(kwargs)
        return ExecuteResponse(output="ssh ok", exit_code=0, truncated=False)


def _application(**overrides: Any) -> tuple[ProductToolApplication, dict[str, Any]]:
    ports = {
        "profiles": _Port(),
        "files": _Port(),
        "artifacts": _Port(),
        "knowledge": _Port(),
        "research": _Port(),
        "browser": _Port(),
        "integrations": _Port(),
        "intake": _Port(),
        "schedules": _Port(),
        "jobs": _Port(create={"tenant_id": "tenant-a", "id": "job-1", "status": "queued"}),
        "idempotency": _IdempotencyPort(),
    }
    if overrides.get("evolution") is not None:
        overrides.setdefault("evolution_owner_tenant_id", "tenant-a")
    ports.update(overrides)
    return ProductToolApplication(**ports), ports  # type: ignore[arg-type]


def _invocation(
    name: str,
    arguments: dict[str, Any],
    *,
    tenant_id: str = "tenant-a",
    idempotency_key: str | None = "request-1",
) -> ProductToolInvocation:
    return ProductToolInvocation(
        spec=TOOL_SPEC_BY_NAME[name],
        context=AgentRunContext(
            tenant_id=tenant_id,
            actor_id="owner-1",
            thread_id="thread-1",
            channel=AgentChannel.WEB,
            run_kind=AgentRunKind.OWNER,
            correlation_id="correlation-1",
            origin=OriginRef(interface="web", source_id="test"),
            agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
            trust_class="owner",
        ),
        arguments=arguments,
        idempotency_key=idempotency_key,
        audit_id="audit-1",
    )


def test_concrete_application_covers_registry_without_legacy_runtime_or_http() -> None:
    application, _ = _application()

    assert tuple(tool.name for tool in build_product_tools(application)) == tuple(
        spec.name for spec in TOOL_SPECS
    )
    source = Path(ProductToolApplication.__module__.replace(".", "/") + ".py")
    text = (Path(__file__).resolve().parents[1] / "src" / source).read_text()
    assert "opentulpa" + ".agent" not in text
    assert "httpx" not in text
    assert "/internal/" not in text


@pytest.mark.asyncio
async def test_web_search_provider_failure_is_a_sanitized_retryable_tool_error() -> None:
    research = _Port(
        search=WebSearchProviderError(
            "Web search returned no grounded sources.",
            retryable=True,
        )
    )
    application, _ = _application(research=research)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.web_search(_invocation("web_search", {"query": "OpenTulpa", "limit": 8}))

    assert error.value.code == "web_search_failed"
    assert error.value.public_message == "Web search returned no grounded sources."
    assert error.value.retryable is True


@pytest.mark.asyncio
async def test_repository_provider_failure_preserves_actionable_public_message() -> None:
    repositories = _Port(
        open=RepositorySandboxUnavailableError(
            "no repository sandbox is configured; paste DAYTONA_API_KEY=<key>"
        )
    )
    application, _ = _application(repositories=repositories)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.repository_open(
            _invocation(
                "repository_open",
                {
                    "repository_url": "https://github.com/acme/project",
                    "base_ref": "main",
                    "provider": "auto",
                },
            )
        )

    assert error.value.code == "repository_workspace_error"
    assert error.value.public_message == (
        "no repository sandbox is configured; paste DAYTONA_API_KEY=<key>"
    )
    assert error.value.retryable is True


@pytest.mark.asyncio
async def test_resource_owner_is_injected_and_mismatched_service_data_fails_closed() -> None:
    files = _Port(
        get={
            "file_id": "file-1",
            "customer_id": "tenant-b",
            "original_filename": "private.pdf",
        }
    )
    application, _ = _application(files=files)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.file_get(_invocation("file_get", {"file_id": "file-1"}))

    assert error.value.code == "not_found"
    assert files.calls == [("get", {"tenant_id": "tenant-a", "file_id": "file-1"})]


@pytest.mark.asyncio
async def test_model_output_removes_internal_identity_paths_and_secrets() -> None:
    files = _Port(
        get={
            "file_id": "file-1",
            "customer_id": "tenant-a",
            "original_filename": "report.pdf",
            "stored_path": "/srv/opentulpa/tenant-a/report.pdf",
            "api_key": "must-not-leak",
            "metadata": {
                "tenant_id": "tenant-a",
                "token": "also-secret",
                "pages": 4,
            },
        }
    )
    application, _ = _application(files=files)

    output = await application.file_get(_invocation("file_get", {"file_id": "file-1"}))

    assert output.data == {
        "file_id": "file-1",
        "original_filename": "report.pdf",
        "api_key": "[redacted]",
        "metadata": {"token": "[redacted]", "pages": 4},
    }


@pytest.mark.asyncio
async def test_background_operation_preflights_tenant_resource_then_submits_direct_job() -> None:
    integrations = _Port(
        get_connection={
            "connection_id": "connection-1",
            "tenant_id": "tenant-a",
            "status": "active",
        }
    )
    jobs = _Port(
        create={
            "id": "job-42",
            "tenant_id": "tenant-a",
            "status": "queued",
            "arguments": {"connection_id": "connection-1"},
        }
    )
    application, _ = _application(integrations=integrations, jobs=jobs)
    invocation = _invocation(
        "integration_invoke",
        {
            "connection_id": "connection-1",
            "action_name": "GOOGLEDRIVE_FIND_FILE",
            "parameters": {"query": "proposal"},
        },
        idempotency_key="invoke-1",
    )

    output = await application.integration_invoke(invocation)

    assert integrations.calls == [
        (
            "get_connection",
            {"tenant_id": "tenant-a", "connection_id": "connection-1"},
        )
    ]
    assert jobs.calls == [
        (
            "create",
            {
                "tenant_id": "tenant-a",
                "handler_name": "integration_invoke",
                "arguments": dict(invocation.arguments),
                "idempotency_key": "invoke-1",
            },
        )
    ]
    assert output.job_id == "job-42"
    assert output.data == {"job_id": "job-42", "status": "queued"}


@pytest.mark.asyncio
async def test_background_operation_does_not_submit_when_tenant_preflight_fails() -> None:
    integrations = _Port(get_connection=None)
    jobs = _Port(create={"id": "job-should-not-exist"})
    application, _ = _application(integrations=integrations, jobs=jobs)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.integration_invoke(
            _invocation(
                "integration_invoke",
                {
                    "connection_id": "foreign-connection",
                    "action_name": "ACTION",
                    "parameters": {},
                },
            )
        )

    assert error.value.code == "not_found"
    assert jobs.calls == []


@pytest.mark.asyncio
async def test_connected_account_user_id_must_match_tenant_before_mutation() -> None:
    integrations = _Port(
        get_connection={
            "id": "foreign-connection",
            "user_id": "tenant-b",
            "status": "active",
        }
    )
    application, _ = _application(integrations=integrations)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.connection_disconnect(
            _invocation(
                "connection_disconnect",
                {"connection_id": "foreign-connection"},
                idempotency_key="disconnect-1",
            )
        )

    assert error.value.code == "not_found"
    assert [name for name, _ in integrations.calls] == ["get_connection"]


@pytest.mark.asyncio
async def test_delivery_uses_trusted_actor_channel_and_thread_not_model_arguments() -> None:
    artifacts = _Port(
        get={"artifact_id": "artifact-1", "tenant_id": "tenant-a"},
        deliver={
            "ok": True,
            "tenant_id": "tenant-a",
            "delivery_id": "delivery-1",
            "local_path": "/private/artifact.txt",
        },
    )
    application, _ = _application(artifacts=artifacts)

    output = await application.artifact_deliver(
        _invocation(
            "artifact_deliver",
            {"artifact_id": "artifact-1", "caption": "Result"},
            idempotency_key="deliver-1",
        )
    )

    assert artifacts.calls[-1] == (
        "deliver",
        {
            "tenant_id": "tenant-a",
            "actor_id": "owner-1",
            "thread_id": "thread-1",
            "channel": "web",
            "artifact_id": "artifact-1",
            "caption": "Result",
            "idempotency_key": "deliver-1",
        },
    )
    assert output.data == {"ok": True, "delivery_id": "delivery-1"}


@pytest.mark.asyncio
async def test_source_operations_are_owner_only_iterative_and_hide_worktree_paths() -> None:
    evolution = _Port(
        source_status={
            "active_release_id": "release-current",
            "workspace_head": "a" * 40,
            "worktree_path": "/private/evolution/source",
        },
        source_read={"path": "README.md", "content": "before\n"},
        source_write={"path": "README.md", "bytes_written": 6},
        source_edit={"path": "README.md", "replacements": 1},
        source_bash={
            "exit_code": 0,
            "output": "tests passed",
            "workspace_root": "/private/evolution/source",
        },
        source_activate={"activation_id": "activation-1", "status": "preparing"},
        source_rollback={"activation_id": "activation-2", "status": "preparing"},
    )
    application, ports = _application(evolution=evolution)

    status = await application.source_status(_invocation("source_status", {}, idempotency_key=None))
    read = await application.source_read(
        _invocation(
            "source_read",
            {"path": "README.md", "offset": 1, "limit": 2_000},
            idempotency_key=None,
        )
    )
    written = await application.source_write(
        _invocation(
            "source_write",
            {"path": "README.md", "content": "after\n"},
            idempotency_key=None,
        )
    )
    edited = await application.source_edit(
        _invocation(
            "source_edit",
            {
                "path": "README.md",
                "old_text": "after",
                "new_text": "done",
                "replace_all": False,
            },
            idempotency_key=None,
        )
    )
    bash = await application.source_bash(
        _invocation(
            "source_bash",
            {"command": "pytest -q", "timeout_seconds": 300},
            idempotency_key=None,
        )
    )
    activated = await application.source_activate(
        _invocation(
            "source_activate",
            {
                "message": "Improve website interface",
                "reason": "Owner requested improvement",
                "review_instructions": "Verify the interface in the running deployment.",
            },
            idempotency_key="activate-1",
        )
    )
    assert ports["idempotency"].calls == []
    rolled_back = await application.source_rollback(
        _invocation(
            "source_rollback",
            {
                "expected_active_release_id": "release-current",
                "reason": "Regression found",
            },
            idempotency_key="rollback-1",
        )
    )

    assert status.data == {"active_release_id": "release-current", "workspace_head": "a" * 40}
    assert read.data["content"] == "before\n"
    assert written.data["bytes_written"] == 6
    assert edited.data["replacements"] == 1
    assert bash.data == {"exit_code": 0, "output": "tests passed"}
    assert activated.data == {"activation_id": "activation-1", "status": "preparing"}
    assert rolled_back.data == {"activation_id": "activation-2", "status": "preparing"}
    audit_context = {
        "tenant_id": "tenant-a",
        "actor_id": "owner-1",
        "thread_id": "thread-1",
        "correlation_id": "correlation-1",
        "channel": "web",
        "run_kind": "owner",
        "origin": (
            '{"interface":"web","source_id":"test","conversation_id":null,"message_id":null}'
        ),
    }
    assert evolution.calls == [
        ("source_status", {"audit_context": audit_context}),
        (
            "source_read",
            {
                "path": "README.md",
                "offset": 1,
                "limit": 2_000,
                "audit_context": audit_context,
            },
        ),
        (
            "source_write",
            {
                "path": "README.md",
                "content": "after\n",
                "audit_context": audit_context,
            },
        ),
        (
            "source_edit",
            {
                "path": "README.md",
                "old_text": "after",
                "new_text": "done",
                "replace_all": False,
                "audit_context": audit_context,
            },
        ),
        (
            "source_bash",
            {
                "command": "pytest -q",
                "timeout_seconds": 300,
                "audit_context": audit_context,
            },
        ),
        (
            "source_activate",
            {
                "idempotency_key": "activate-1",
                "message": "Improve website interface",
                "reason": "Owner requested improvement",
                "review_instructions": "Verify the interface in the running deployment.",
                "inference_plan": None,
                "audit_context": audit_context,
            },
        ),
        (
            "source_rollback",
            {
                "idempotency_key": "rollback-1",
                "expected_active_release_id": "release-current",
                "reason": "Regression found",
                "audit_context": audit_context,
            },
        ),
    ]
    assert ports["idempotency"].calls == []

    unavailable, _ = _application()
    with pytest.raises(ProductToolApplicationError, match="unavailable"):
        await unavailable.source_status(_invocation("source_status", {}, idempotency_key=None))

    with pytest.raises(ProductToolApplicationError, match="unavailable"):
        await application.source_status(
            _invocation(
                "source_status",
                {},
                tenant_id="tenant-b",
                idempotency_key=None,
            )
        )


@pytest.mark.asyncio
async def test_trace_reads_are_tenant_scoped_and_direct() -> None:
    traces = _Port(
        trace_list=[{"run_id": "run-1", "status": "failed"}],
        trace_get={"run_id": "run-1", "events": []},
    )
    application, _ = _application(traces=traces)

    listed = await application.trace_list(
        _invocation(
            "trace_list",
            {"status": "failed", "limit": 10, "before_run_id": "run-2"},
            idempotency_key=None,
        )
    )
    fetched = await application.trace_get(
        _invocation(
            "trace_get",
            {
                "run_id": "run-1",
                "after_sequence": 0,
                "limit": 200,
                "include_messages": True,
            },
            idempotency_key=None,
        )
    )

    assert listed.data == [{"run_id": "run-1", "status": "failed"}]
    assert fetched.data == {"run_id": "run-1", "events": []}
    assert traces.calls == [
        (
            "trace_list",
            {
                "tenant_id": "tenant-a",
                "status": "failed",
                "limit": 10,
                "before_run_id": "run-2",
            },
        ),
        (
            "trace_get",
            {
                "tenant_id": "tenant-a",
                "run_id": "run-1",
                "after_sequence": 0,
                "limit": 200,
                "include_messages": True,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_external_delivery_replays_from_durable_idempotency_store(tmp_path: Path) -> None:
    artifacts = _Port(
        get={"artifact_id": "artifact-1", "tenant_id": "tenant-a"},
        deliver={"ok": True, "tenant_id": "tenant-a", "delivery_id": "delivery-1"},
    )
    application, _ = _application(
        artifacts=artifacts,
        idempotency=IdempotencyStore(tmp_path / "effects.sqlite"),
    )
    invocation = _invocation(
        "artifact_deliver",
        {"artifact_id": "artifact-1", "caption": None},
        idempotency_key="delivery-replay-1",
    )

    first = await application.artifact_deliver(invocation)
    replay = await application.artifact_deliver(invocation)

    assert first == replay
    assert [name for name, _ in artifacts.calls] == ["get", "deliver"]


@pytest.mark.asyncio
async def test_source_set_runtime_env_replays_from_durable_idempotency_store(
    tmp_path: Path,
) -> None:
    evolution = _Port(
        source_set_runtime_env={
            "status": "updated",
            "name": "OPENAI_COMPATIBLE_API_KEY",
            "changed": True,
            "restarted": True,
            "value": "[set]",
        }
    )
    application, _ = _application(
        evolution=evolution,
        idempotency=IdempotencyStore(tmp_path / "runtime-env-effects.sqlite"),
    )
    invocation = _invocation(
        "source_set_runtime_env",
        {"name": "OPENAI_COMPATIBLE_API_KEY", "value": "provider-secret"},
        idempotency_key="runtime-env-replay-1",
    )

    first = await application.source_set_runtime_env(invocation)
    replay = await application.source_set_runtime_env(invocation)

    assert first == replay
    assert first.data["value"] == "[set]"
    assert [name for name, _ in evolution.calls] == ["source_set_runtime_env"]
    assert evolution.calls[0][1]["idempotency_key"] == "runtime-env-replay-1"


@pytest.mark.asyncio
async def test_source_set_runtime_env_redeems_secret_handle_inside_trusted_boundary(
    tmp_path: Path,
) -> None:
    secret_handles = _Port(
        resolve_for_runtime_environment=SecretStr("provider-secret"),
    )
    evolution = _Port(
        source_set_runtime_env={
            "status": "updated",
            "name": "COMPOSIO_API_KEY",
            "changed": True,
            "restarted": True,
            "value": "[set]",
        }
    )
    application, _ = _application(
        evolution=evolution,
        secret_handles=secret_handles,
        idempotency=IdempotencyStore(tmp_path / "runtime-env-secret-effects.sqlite"),
    )
    invocation = _invocation(
        "source_set_runtime_env",
        {"name": "COMPOSIO_API_KEY", "secret_id": "composio_api_key"},
        idempotency_key="runtime-env-secret-1",
    )

    result = await application.source_set_runtime_env(invocation)

    assert result.data["status"] == "updated"
    assert result.data["value"] == "[set]"
    assert secret_handles.calls == [
        (
            "resolve_for_runtime_environment",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "secret_id": "composio_api_key",
                "environment_name": "COMPOSIO_API_KEY",
            },
        )
    ]
    assert evolution.calls[0][1]["value"] == "provider-secret"


@pytest.mark.asyncio
async def test_source_set_runtime_env_surfaces_legacy_failure_and_accepts_fresh_key(
    tmp_path: Path,
) -> None:
    idempotency = IdempotencyStore(tmp_path / "runtime-env-legacy-effects.sqlite")
    invocation = _invocation(
        "source_set_runtime_env",
        {"name": "COMPOSIO_API_KEY", "value": "provider-secret"},
        idempotency_key="legacy-runtime-env-key",
    )
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
    await idempotency.execute(
        tenant_id="tenant-a",
        operation="source_set_runtime_env",
        idempotency_key="legacy-runtime-env-key",
        request_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        invoke=lambda: {
            "status": "failed",
            "name": "COMPOSIO_API_KEY",
            "changed": False,
            "rollback_restored": True,
        },
    )
    evolution = _Port(
        source_set_runtime_env={
            "status": "updated",
            "name": "COMPOSIO_API_KEY",
            "changed": True,
            "restarted": True,
            "value": "[set]",
        }
    )
    application, _ = _application(evolution=evolution, idempotency=idempotency)

    with pytest.raises(ProductToolApplicationError) as legacy_error:
        await application.source_set_runtime_env(invocation)

    assert legacy_error.value.code == "runtime_env_update_failed"
    assert legacy_error.value.retryable is True
    assert evolution.calls == []

    retry = await application.source_set_runtime_env(
        _invocation(
            "source_set_runtime_env",
            {"name": "COMPOSIO_API_KEY", "value": "provider-secret"},
            idempotency_key="fresh-runtime-env-key",
        )
    )

    assert retry.data["status"] == "updated"
    assert [name for name, _ in evolution.calls] == ["source_set_runtime_env"]


@pytest.mark.asyncio
async def test_source_set_runtime_env_marks_rolled_back_update_as_failed_effect(
    tmp_path: Path,
) -> None:
    evolution = _Port(
        source_set_runtime_env={
            "status": "failed",
            "name": "COMPOSIO_API_KEY",
            "changed": False,
            "rollback_restored": True,
            "error": {"code": "runtime_env_update_failed"},
        }
    )
    application, _ = _application(
        evolution=evolution,
        idempotency=IdempotencyStore(tmp_path / "runtime-env-failed-effects.sqlite"),
    )
    invocation = _invocation(
        "source_set_runtime_env",
        {"name": "COMPOSIO_API_KEY", "value": "provider-secret"},
        idempotency_key="runtime-env-failure-1",
    )

    with pytest.raises(ProductToolApplicationError) as error:
        await application.source_set_runtime_env(invocation)

    assert error.value.code == "runtime_env_update_failed"
    assert error.value.retryable is True
    with pytest.raises(ProductToolApplicationError) as replay_error:
        await application.source_set_runtime_env(invocation)
    assert replay_error.value.code == "conflict"
    assert [name for name, _ in evolution.calls] == ["source_set_runtime_env"]


@pytest.mark.asyncio
async def test_source_set_runtime_env_exposes_recovery_required() -> None:
    evolution = _Port(
        source_set_runtime_env={
            "status": "failed",
            "name": "META_MESSENGER_VERIFY_TOKEN",
            "changed": False,
            "runtime_restored": False,
            "rollback_restored": False,
        }
    )
    application, _ = _application(evolution=evolution)

    with pytest.raises(ProductToolApplicationError) as error:
        await application.source_set_runtime_env(
            _invocation(
                "source_set_runtime_env",
                {"name": "META_MESSENGER_VERIFY_TOKEN", "value": "verify-token"},
                idempotency_key="runtime-env-recovery-required",
            )
        )

    assert error.value.code == "runtime_recovery_required"
    assert error.value.retryable is False
    assert "do not use SSH as a fallback" in error.value.public_message


@pytest.mark.asyncio
async def test_intake_confirmation_token_is_exposed_only_as_model_handle(
    tmp_path: Path,
) -> None:
    handle = "hash-bound-confirmation-handle-at-least-32-chars"
    intake = _Port(
        get_draft={"id": "draft-1", "tenant_id": "tenant-a", "revision": 2},
        prepare_draft={
            "draft": {"id": "draft-1", "tenant_id": "tenant-a", "revision": 2},
            "proposal": {"workflow_id": "workflow-1", "name": "Lead capture"},
            "confirmation_token": handle,
        },
        activate_draft={
            "draft": {"id": "draft-1", "tenant_id": "tenant-a", "status": "activated"},
            "workflow": {"workflow_id": "workflow-1", "tenant_id": "tenant-a"},
        },
    )
    application, _ = _application(
        intake=intake,
        idempotency=IdempotencyStore(tmp_path / "intake-effects.sqlite"),
    )
    prepare = _invocation(
        "intake_draft_prepare",
        {"draft_id": "draft-1", "expected_revision": 2},
        idempotency_key="prepare-1",
    )

    prepared = await application.intake_draft_prepare(prepare)
    replayed = await application.intake_draft_prepare(prepare)
    activated = await application.intake_draft_activate(
        _invocation(
            "intake_draft_activate",
            {
                "draft_id": "draft-1",
                "expected_revision": 2,
                "confirmation_handle": handle,
            },
            idempotency_key="activate-1",
        )
    )

    assert prepared == replayed
    assert prepared.data["confirmation_handle"] == handle
    assert redact_for_langfuse(prepared.data)["confirmation_handle"] == handle
    assert "confirmation_token" not in prepared.data
    assert [name for name, _ in intake.calls].count("prepare_draft") == 1
    activate_call = next(kwargs for name, kwargs in intake.calls if name == "activate_draft")
    assert activate_call["confirmation_token"] == handle
    assert "confirmation_handle" not in activate_call
    assert activated.data["workflow"]["workflow_id"] == "workflow-1"


@pytest.mark.asyncio
async def test_schedule_list_filters_disabled_and_save_passes_typed_write_and_idempotency() -> None:
    schedules = _Port(
        list=[
            {"id": "enabled", "tenant_id": "tenant-a", "enabled": True},
            {"id": "disabled", "tenant_id": "tenant-a", "enabled": False},
        ],
        save=lambda **kwargs: {
            "id": kwargs["schedule_id"] or "new-schedule",
            "tenant_id": kwargs["tenant_id"],
            "revision": 1,
            **kwargs["write"].model_dump(mode="json"),
        },
    )
    application, _ = _application(schedules=schedules)

    listed = await application.schedule_list(
        _invocation("schedule_list", {"include_disabled": False}, idempotency_key=None)
    )
    saved = await application.schedule_save(
        _invocation(
            "schedule_save",
            {
                "schedule_id": None,
                "expected_revision": None,
                "schedule": {
                    "name": "Morning reminder",
                    "trigger": {
                        "kind": "cron",
                        "expression": "0 9 * * *",
                        "timezone": "Europe/Moscow",
                    },
                    "action": {"kind": "reminder", "message": "Check leads"},
                    "notify_owner": True,
                    "enabled": True,
                },
            },
            idempotency_key="derived-schedule-1",
        )
    )

    assert listed.data == [{"id": "enabled", "enabled": True}]
    save_call = schedules.calls[-1]
    assert save_call[0] == "save"
    assert save_call[1]["tenant_id"] == "tenant-a"
    assert save_call[1]["idempotency_key"] == "derived-schedule-1"
    assert save_call[1]["write"].name == "Morning reminder"
    assert saved.data["id"] == "new-schedule"
    assert "tenant_id" not in saved.data


@pytest.mark.asyncio
async def test_sandbox_ssh_diagnostic_mounts_secret_handle_without_plaintext_command() -> None:
    private_key = "-----BEGIN OPENSSH PRIVATE KEY-----\nsecret-key\n-----END OPENSSH PRIVATE KEY-----"
    handle = SecretHandle(
        tenant_id="tenant-a",
        id="ssh_private_key",
        revision=2,
        name="ssh_private_key",
        state=SecretState.ACTIVE,
        scopes=("ssh.connect",),
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        created_by="owner-1",
    )
    secret_handles = _Port(
        get=handle,
        resolve_for_sandbox=SecretStr(private_key),
    )
    sandbox_execution = _SandboxExecutionPort()
    application, _ = _application(
        secret_handles=secret_handles,
        sandbox_execution=sandbox_execution,
    )

    result = await application.sandbox_ssh_diagnostic(
        _invocation(
            "sandbox_ssh_diagnostic",
            {
                "secret_id": "ssh_private_key",
                "host": "178.214.97.1",
                "user": "root",
                "port": 22,
                "command": "ss -tn | wc -l",
                "timeout_seconds": 30,
                "secret_type": "private_key",
            },
            idempotency_key=None,
        )
    )

    assert result.data == {
        "host": "178.214.97.1",
        "user": "root",
        "port": 22,
        "exit_code": 0,
        "output": "ssh ok",
        "truncated": False,
    }
    assert secret_handles.calls == [
        ("get", {"tenant_id": "tenant-a", "secret_id": "ssh_private_key"}),
        (
            "resolve_for_sandbox",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "secret_id": "ssh_private_key",
                "scope": "ssh.connect",
                "mount_type": "ssh_private_key",
            },
        ),
    ]
    sandbox_call = sandbox_execution.calls[0]
    assert sandbox_call["tenant_id"] == "tenant-a"
    assert sandbox_call["timeout"] == 30
    assert private_key not in sandbox_call["command"]
    assert "ssh -i \"$OPENTULPA_SSH_IDENTITY\"" in sandbox_call["command"]
    assert 'UserKnownHostsFile="$PWD/.opentulpa-ssh-known-hosts"' in sandbox_call["command"]
    assert 'UserKnownHostsFile="$PWD/.ssh/' not in sandbox_call["command"]
    assert "root@178.214.97.1" in sandbox_call["command"]
    assert sandbox_call["secret_files"][0].content == private_key
    assert sandbox_call["secret_files"][0].env == "OPENTULPA_SSH_IDENTITY"


@pytest.mark.asyncio
async def test_sandbox_ssh_diagnostic_mounts_password_for_askpass_only() -> None:
    password = "unique password 'with shell symbols' $HOME"
    handle = SecretHandle(
        tenant_id="tenant-a",
        id="ssh_password",
        revision=2,
        name="ssh_password",
        state=SecretState.ACTIVE,
        scopes=("ssh.connect",),
        created_at=datetime(2026, 7, 31, tzinfo=UTC),
        created_by="owner-1",
    )
    secret_handles = _Port(
        get=handle,
        resolve_for_sandbox=SecretStr(password),
    )
    sandbox_execution = _SandboxExecutionPort()
    application, _ = _application(
        secret_handles=secret_handles,
        sandbox_execution=sandbox_execution,
    )

    result = await application.sandbox_ssh_diagnostic(
        _invocation(
            "sandbox_ssh_diagnostic",
            {
                "secret_id": "ssh_password",
                "host": "13928983",
                "user": "root",
                "port": 22,
                "command": "uptime",
                "timeout_seconds": 30,
                "secret_type": "password",
            },
            idempotency_key=None,
        )
    )

    assert result.data["exit_code"] == 0
    assert secret_handles.calls[-1] == (
        "resolve_for_sandbox",
        {
            "tenant_id": "tenant-a",
            "actor_id": "owner-1",
            "secret_id": "ssh_password",
            "scope": "ssh.connect",
            "mount_type": "ssh_password",
        },
    )
    sandbox_call = sandbox_execution.calls[0]
    assert password not in sandbox_call["command"]
    assert "root@13928983" in sandbox_call["command"]
    assert "SSH_ASKPASS_REQUIRE=force" in sandbox_call["command"]
    assert "BatchMode=no" in sandbox_call["command"]
    assert "PubkeyAuthentication=no" in sandbox_call["command"]
    assert "PasswordAuthentication=yes" in sandbox_call["command"]
    assert "NumberOfPasswordPrompts=1" in sandbox_call["command"]
    assert "ssh -i" not in sandbox_call["command"]
    assert sandbox_call["secret_files"][0].content == password
    assert sandbox_call["secret_files"][0].env == "OPENTULPA_SSH_PASSWORD_FILE"


@pytest.mark.asyncio
async def test_sandbox_ssh_diagnostic_rejects_container_self_recreate() -> None:
    application, _ = _application()

    with pytest.raises(ProductToolApplicationError, match="container lifecycle"):
        await application.sandbox_ssh_diagnostic(
            _invocation(
                "sandbox_ssh_diagnostic",
                {
                    "secret_id": "ssh_password",
                    "host": "84.21.189.71",
                    "user": "root",
                    "port": 22,
                    "command": (
                        "cd /opt/opentulpa; "
                        "docker compose up -d --force-recreate opentulpa"
                    ),
                    "timeout_seconds": 180,
                    "secret_type": "password",
                },
                idempotency_key=None,
            )
        )


def test_password_ssh_command_reads_only_the_mounted_askpass_file(tmp_path: Path) -> None:
    binary_root = tmp_path / "bin"
    binary_root.mkdir()
    fake_ssh = binary_root / "ssh"
    fake_ssh.write_text(
        "#!/bin/sh\n"
        'provided="$("$SSH_ASKPASS" "password:")"\n'
        'expected="$(cat "$OPENTULPA_SSH_PASSWORD_FILE")"\n'
        'test "$provided" = "$expected" || exit 7\n'
        "printf authenticated\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    password_file = tmp_path / "mounted-password"
    password_file.write_text("fake-password-for-test", encoding="utf-8")
    password_file.chmod(0o600)
    command = _sandbox_ssh_command(
        target="root@example.test",
        port=22,
        remote_command="uptime",
        secret_type="password",
    )

    completed = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=workspace,
        env={
            "PATH": f"{binary_root}{os.pathsep}{os.defpath}",
            "OPENTULPA_SSH_PASSWORD_FILE": str(password_file),
        },
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == "authenticated"
    assert "fake-password-for-test" not in command
    assert not (workspace / ".ssh").exists()
    assert not (workspace / ".opentulpa-ssh-askpass").exists()
    assert not (workspace / ".opentulpa-ssh-known-hosts").exists()


@pytest.mark.asyncio
async def test_capability_lifecycle_uses_hidden_owner_context_and_opaque_secret_handles() -> None:
    manifest = {
        "name": "telegram",
        "revision": 1,
        "secrets": [{"name": "TELEGRAM_BOT_TOKEN", "required": True}],
    }
    activation = {
        "namespace": "tenant-a",
        "capability_name": "telegram",
        "revision": 1,
        "generation": 1,
        "secret_handles": {"TELEGRAM_BOT_TOKEN": "telegram_bot"},
    }
    capabilities = _Port(
        list=[{"manifest": manifest, "activation": None, "test": None}],
        seed_bundled=(manifest,),
        test={
            "namespace": "tenant-a",
            "capability_name": "telegram",
            "revision": 1,
            "status": "passed",
        },
        activate=activation,
        rollback={**activation, "revision": 0, "generation": 2},
        deactivate={**activation, "generation": 2},
    )
    application, ports = _application(capabilities=capabilities)

    listed = await application.capability_list(
        _invocation("capability_list", {}, idempotency_key=None)
    )
    seeded = await application.capability_seed_bundled(
        _invocation("capability_seed_bundled", {}, idempotency_key="derived-seed")
    )
    tested = await application.capability_test(
        _invocation(
            "capability_test",
            {"capability_name": "telegram", "revision": 1},
            idempotency_key=None,
        )
    )
    activated = await application.capability_activate(
        _invocation(
            "capability_activate",
            {
                "capability_name": "telegram",
                "revision": 1,
                "expected_generation": None,
                "config": {"pairing_mode": "one_time"},
                "secret_handles": {"TELEGRAM_BOT_TOKEN": "telegram_bot"},
            },
            idempotency_key="activate-telegram-v1",
        )
    )
    await application.capability_rollback(
        _invocation(
            "capability_rollback",
            {"capability_name": "telegram", "expected_generation": 1},
            idempotency_key="rollback-telegram",
        )
    )
    await application.capability_deactivate(
        _invocation(
            "capability_deactivate",
            {"capability_name": "telegram", "expected_generation": 2},
            idempotency_key="deactivate-telegram",
        )
    )

    assert listed.data[0]["manifest"]["credential_requirements"] == [
        {"name": "TELEGRAM_BOT_TOKEN", "required": True}
    ]
    assert seeded.data["capabilities"][0]["name"] == "telegram"
    assert "namespace" not in tested.data
    assert activated.data["credential_bindings"] == [
        {"name": "TELEGRAM_BOT_TOKEN", "handle_id": "telegram_bot"}
    ]
    assert capabilities.calls == [
        ("list", {"tenant_id": "tenant-a"}),
        ("seed_bundled", {"tenant_id": "tenant-a", "actor_id": "owner-1"}),
        (
            "test",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "revision": 1,
            },
        ),
        (
            "activate",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "revision": 1,
                "expected_generation": None,
                "config": {"pairing_mode": "one_time"},
                "secret_handles": {"TELEGRAM_BOT_TOKEN": "telegram_bot"},
            },
        ),
        (
            "rollback",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "expected_generation": 1,
                "config": None,
                "secret_handles": None,
            },
        ),
        (
            "deactivate",
            {
                "tenant_id": "tenant-a",
                "actor_id": "owner-1",
                "capability_name": "telegram",
                "expected_generation": 2,
            },
        ),
    ]
    assert [name for name, _ in ports["idempotency"].calls] == [
        "execute",
        "execute",
        "execute",
        "execute",
    ]
