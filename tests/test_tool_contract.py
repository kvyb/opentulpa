from pathlib import Path

import pytest
from pydantic import ValidationError

from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling import (
    CONTRACT_VERSION,
    TOOL_SPEC_BY_NAME,
    TOOL_SPECS,
    AgentRunContext,
    ApprovalMode,
    ExecutionMode,
    IdempotencyMode,
    ToolEffect,
    ToolError,
    ToolResult,
    ToolStatus,
    render_tool_contract_markdown,
    tool_contract_document,
    tool_contract_json_schema,
)

EXPECTED_OPERATIONS = {
    "profile_get",
    "profile_update",
    "file_search",
    "file_get",
    "file_analyze",
    "file_inspect",
    "artifact_deliver",
    "knowledge_list",
    "knowledge_find",
    "knowledge_attach",
    "knowledge_archive",
    "knowledge_reindex",
    "knowledge_query",
    "web_search",
    "content_fetch",
    "browser_start",
    "browser_get",
    "browser_act",
    "browser_stop",
    "integration_list",
    "integration_connect",
    "connection_list",
    "connection_disconnect",
    "integration_action_search",
    "integration_invoke",
    "intake_workflow_list",
    "intake_workflow_get",
    "intake_draft_save",
    "intake_draft_prepare",
    "intake_draft_activate",
    "intake_workflow_delete",
    "intake_workflow_test",
    "schedule_list",
    "schedule_save",
    "schedule_delete",
    "agent_spec_list",
    "agent_spec_save",
    "agent_spec_activate",
    "agent_spec_rollback",
    "trigger_spec_list",
    "trigger_spec_save",
    "trigger_spec_activate",
    "trigger_spec_rollback",
    "secret_handle_list",
    "secret_handle_revoke",
    "sandbox_ssh_diagnostic",
    "capability_list",
    "capability_seed_bundled",
    "capability_test",
    "capability_activate",
    "capability_rollback",
    "capability_deactivate",
    "job_get",
    "job_events",
    "job_artifacts",
    "job_cancel",
    "repository_open",
    "repository_list",
    "repository_status",
    "repository_close",
    "repository_publish_pr",
    "source_status",
    "source_runtime_env_get",
    "source_sync_upstream",
    "source_prepare_pr",
    "source_resolve_dependencies",
    "source_shell",
    "source_release",
    "source_rollback",
    "source_set_runtime_env",
    "trace_list",
    "trace_get",
}


def test_registry_is_complete_unique_and_versioned() -> None:
    names = [spec.name for spec in TOOL_SPECS]

    assert len(names) == 72
    assert len(names) == len(set(names))
    assert set(names) == EXPECTED_OPERATIONS
    assert set(TOOL_SPEC_BY_NAME) == EXPECTED_OPERATIONS
    assert {spec.version for spec in TOOL_SPECS} == {1}


def test_registry_limits_policy_approval_to_sandbox_diagnostic() -> None:
    for spec in TOOL_SPECS:
        expected = ApprovalMode.POLICY if spec.name == "sandbox_ssh_diagnostic" else ApprovalMode.AUTO
        assert spec.approval is expected
        if spec.effect is ToolEffect.READ:
            assert spec.idempotency is IdempotencyMode.NONE
        if spec.effect in {ToolEffect.DELETE, ToolEffect.SEND, ToolEffect.AUTHORIZE}:
            assert spec.idempotency is IdempotencyMode.REQUIRED

    for name in (
        "agent_spec_activate",
        "agent_spec_rollback",
        "trigger_spec_activate",
        "trigger_spec_rollback",
        "secret_handle_revoke",
        "capability_activate",
        "capability_rollback",
        "capability_deactivate",
    ):
        assert TOOL_SPEC_BY_NAME[name].idempotency is IdempotencyMode.REQUIRED
    assert TOOL_SPEC_BY_NAME["agent_spec_save"].idempotency is IdempotencyMode.DERIVED
    assert TOOL_SPEC_BY_NAME["trigger_spec_save"].idempotency is IdempotencyMode.DERIVED
    assert TOOL_SPEC_BY_NAME["capability_seed_bundled"].idempotency is IdempotencyMode.DERIVED
    assert TOOL_SPEC_BY_NAME["capability_test"].idempotency is IdempotencyMode.NONE
    assert TOOL_SPEC_BY_NAME["source_shell"].approval is ApprovalMode.AUTO
    assert TOOL_SPEC_BY_NAME["source_shell"].idempotency is IdempotencyMode.NONE
    assert TOOL_SPEC_BY_NAME["sandbox_ssh_diagnostic"].approval is ApprovalMode.POLICY
    assert TOOL_SPEC_BY_NAME["sandbox_ssh_diagnostic"].idempotency is IdempotencyMode.NONE
    for name in ("source_release", "source_rollback", "source_set_runtime_env"):
        assert TOOL_SPEC_BY_NAME[name].idempotency is IdempotencyMode.REQUIRED


def test_external_writes_require_idempotency_keys() -> None:
    external_providers = {"browser", "composio"}
    for spec in TOOL_SPECS:
        if spec.provider in external_providers and spec.effect is not ToolEffect.READ:
            assert spec.idempotency is IdempotencyMode.REQUIRED

    assert TOOL_SPEC_BY_NAME["artifact_deliver"].idempotency is IdempotencyMode.REQUIRED


def test_background_operations_are_idempotent() -> None:
    job_specs = [spec for spec in TOOL_SPECS if spec.execution is ExecutionMode.JOB]

    assert job_specs
    assert all(spec.idempotency is not IdempotencyMode.NONE for spec in job_specs)


def test_context_is_strict_and_immutable() -> None:
    context = AgentRunContext(
        tenant_id="tenant-1",
        actor_id="owner-1",
        thread_id="thread-1",
        channel="web",
        run_kind="owner",
        correlation_id="correlation-1",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id="tenant-1", spec_id="owner", revision=1),
        trust_class="owner",
    )

    with pytest.raises(ValidationError):
        AgentRunContext(
            tenant_id="tenant-1",
            actor_id="owner-1",
            thread_id="thread-1",
            channel="web",
            run_kind="owner",
            correlation_id="correlation-1",
            origin=OriginRef(interface="web", source_id="test"),
            agent_spec=AgentSpecRef(tenant_id="tenant-1", spec_id="owner", revision=1),
            trust_class="owner",
            customer_id="untrusted",
        )
    with pytest.raises(ValidationError):
        context.tenant_id = "other-tenant"


def test_tool_result_status_invariants() -> None:
    assert ToolResult[str](status="ok", data="done", audit_id="audit-1").data == "done"
    assert (
        ToolResult[str](
            status="accepted",
            job_id="job-1",
            idempotency_key="key-1",
            audit_id="audit-1",
        ).status
        is ToolStatus.ACCEPTED
    )
    error = ToolError(code="provider_timeout", message="Provider timed out", retryable=True)
    assert ToolResult[str](status="error", error=error, audit_id="audit-1").error == error

    with pytest.raises(ValidationError, match="require an error"):
        ToolResult[str](status="error", audit_id="audit-1")
    with pytest.raises(ValidationError, match="require a job_id"):
        ToolResult[str](status="accepted", audit_id="audit-1")


def test_machine_contract_contains_types_and_exact_registry() -> None:
    document = tool_contract_document()
    schema = tool_contract_json_schema()

    assert document["contract_version"] == CONTRACT_VERSION
    assert len(document["operations"]) == 72
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["x-opentulpa-contract-version"] == CONTRACT_VERSION
    assert schema["x-opentulpa-operations"] == document["operations"]
    assert {"AgentRunContext", "ToolError", "ToolSpec"} <= set(schema["$defs"])


def test_committed_documentation_matches_registry() -> None:
    repository_root = Path(__file__).resolve().parents[1]

    assert (repository_root / "docs" / "tool-contract.md").read_text() == (
        render_tool_contract_markdown()
    )
