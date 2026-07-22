from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from opentulpa.specs import (
    AgentRunContext,
    AgentSpecRef,
    AgentSpecWrite,
    OriginRef,
    RunSubmission,
)


def _context(*, tenant_id: str = "tenant-a") -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="owner",
        thread_id="thread-1",
        channel="telegram",
        run_kind="owner",
        correlation_id="correlation-1",
        origin=OriginRef(
            interface="telegram",
            source_id="bot-main",
            conversation_id="chat-1",
            message_id="message-1",
        ),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=3),
        trust_class="owner",
    )


def test_run_submission_is_versioned_and_tenant_bound() -> None:
    submission = RunSubmission(
        submission_id="submission-1",
        agent_spec=AgentSpecRef(tenant_id="tenant-a", spec_id="owner", revision=3),
        context=_context(),
        text="Hello",
        file_ids=("file-1",),
        idempotency_key="telegram:update-1",
        submitted_at=datetime(2026, 7, 20, tzinfo=UTC),
    )

    assert submission.protocol_version == "1.0"
    assert submission.agent_spec.revision == 3

    invalid = submission.model_dump()
    invalid["agent_spec"] = {
        "tenant_id": "tenant-a",
        "spec_id": "owner",
        "revision": 4,
    }
    with pytest.raises(ValidationError, match="same agent_spec revision"):
        RunSubmission.model_validate(invalid)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"memory_scope": "spec"}, "memory"),
        ({"workspace_scope": "read_only"}, "workspace access"),
        ({"allow_delegation": True}, "delegation"),
        ({"tool_policy": "profile_default"}, "profile-default tools"),
    ],
)
def test_external_agent_specs_fail_closed(changes: dict[str, object], message: str) -> None:
    payload: dict[str, object] = {
        "name": "External intake",
        "instructions": "Classify the inbound request.",
        "isolation": "external",
        "tool_policy": "allowlist",
        "tools": ("knowledge_query",),
        "memory_scope": "none",
        "workspace_scope": "none",
    }
    payload.update(changes)

    with pytest.raises(ValidationError, match=message):
        AgentSpecWrite.model_validate(payload)


def test_external_agent_spec_accepts_explicit_read_only_tool_allowlist() -> None:
    spec = AgentSpecWrite(
        name="External intake",
        instructions="Return a grounded intake decision.",
        isolation="external",
        tools=("knowledge_find", "knowledge_query"),
        memory_scope="none",
        workspace_scope="none",
    )

    assert spec.isolation == "external"
    assert spec.tools == ("knowledge_find", "knowledge_query")


def test_external_agent_spec_rejects_non_knowledge_tools() -> None:
    with pytest.raises(ValidationError, match="non-knowledge tools"):
        AgentSpecWrite(
            name="Unsafe external agent",
            instructions="Do not mutate product state.",
            isolation="external",
            tools=("profile_update",),
            memory_scope="none",
            workspace_scope="none",
        )


def test_agent_spec_rejects_inert_secret_scope_configuration() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        AgentSpecWrite(
            name="No implicit secrets",
            instructions="Use only explicitly registered tools.",
            secret_scopes=("telegram.send",),  # type: ignore[call-arg]
        )
