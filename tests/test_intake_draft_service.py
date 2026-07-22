from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from opentulpa.intake.drafts import (
    ActivatedIntakeDraft,
    IntakeDraftActivationError,
    IntakeDraftConfirmationError,
    IntakeDraftConflictError,
    IntakeDraftService,
    IntakeDraftStore,
    IntakeDraftValidationError,
)


class _WorkflowActivator:
    def __init__(self) -> None:
        self.active: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []
        self.fail = False

    def activate_draft(
        self,
        *,
        draft_store: IntakeDraftStore,
        tenant_id: str,
        actor_id: str,
        draft_id: str,
        expected_revision: int,
        confirmation_token_hash: str,
        proposal: dict[str, Any],
        now: datetime,
    ) -> ActivatedIntakeDraft:
        workflow_id = str(proposal["workflow_id"])
        attempt_id = "test-activation"
        draft_store.claim_activation(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
            confirmation_token_hash=confirmation_token_hash,
            activation_attempt_id=attempt_id,
            now=now,
        )
        self.calls.append(
            {
                "tenant_id": tenant_id,
                "workflow_id": workflow_id,
                "proposal": proposal,
            }
        )
        if self.fail:
            draft_store.release_activation(
                tenant_id=tenant_id,
                draft_id=draft_id,
                activation_attempt_id=attempt_id,
                now=now,
            )
            raise RuntimeError("simulated activation failure")
        workflow = dict(proposal)
        self.active[(tenant_id, workflow_id)] = workflow
        activated = draft_store.finish_activation(
            tenant_id=tenant_id,
            actor_id=actor_id,
            draft_id=draft_id,
            expected_revision=expected_revision,
            activation_attempt_id=attempt_id,
            now=now,
        )
        return ActivatedIntakeDraft(draft=activated, workflow=workflow)


def _service(tmp_path: Path) -> tuple[IntakeDraftService, _WorkflowActivator]:
    activator = _WorkflowActivator()
    service = IntakeDraftService(
        IntakeDraftStore(tmp_path / "intake-drafts.sqlite"),
        workflow_activator=activator,
        clock=lambda: datetime(2026, 7, 19, 12, tzinfo=UTC),
        token_factory=lambda: "confirmation-token-that-is-at-least-32-characters",
    )
    return service, activator


def _valid_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": " Lead capture ",
        "channel": "instagram_dm",
        "provider": "composio",
        "source_config": {"account": "instagram-main"},
        "intent_description": "Book a service",
        "required_fields": ["name", "Email", "email", ""],
        "field_guidance": {"email": "Ask once"},
        "sink_type": "local_csv",
        "sink_config": {"file_path": "leads.csv"},
    }
    payload.update(overrides)
    return payload


def test_draft_save_is_tenant_scoped_and_revisioned(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)
    created = service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id="draft-1",
        workflow_id="workflow-1",
        patch={"name": "Initial"},
    )

    assert created.revision == 1
    assert created.status == "editing"
    assert service.get(tenant_id="tenant-b", draft_id="draft-1") is None
    with pytest.raises(IntakeDraftConflictError, match="expected_revision is required"):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id="draft-1",
            patch={"intent_description": "Changed"},
        )

    updated = service.save(
        tenant_id="tenant-a",
        actor_id="owner-b",
        draft_id="draft-1",
        expected_revision=1,
        patch={"intent_description": "Changed"},
    )
    assert updated.revision == 2
    assert updated.payload == {"name": "Initial", "intent_description": "Changed"}
    assert updated.updated_by_actor_id == "owner-b"
    with pytest.raises(IntakeDraftConflictError, match="expected revision 1, found 2"):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id="draft-1",
            expected_revision=1,
            patch={"name": "Stale"},
        )


@pytest.mark.asyncio
async def test_prepare_validates_normalizes_and_patch_invalidates_token(
    tmp_path: Path,
) -> None:
    service, _ = _service(tmp_path)
    invalid = service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id="draft-1",
        patch={"name": "Incomplete"},
    )
    with pytest.raises(IntakeDraftValidationError, match="workflow draft is invalid"):
        service.prepare(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id=invalid.id,
            expected_revision=invalid.revision,
        )
    assert service.get(tenant_id="tenant-a", draft_id=invalid.id).status == "editing"  # type: ignore[union-attr]

    valid = service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id=invalid.id,
        expected_revision=invalid.revision,
        patch=_valid_payload(),
    )
    prepared = service.prepare(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id=valid.id,
        expected_revision=valid.revision,
    )

    assert prepared.draft.revision == valid.revision
    assert prepared.draft.status == "prepared"
    assert prepared.proposal.name == "Lead capture"
    assert prepared.proposal.required_fields == ["name", "Email"]
    assert prepared.draft.proposal == prepared.proposal
    assert len(prepared.proposal_hash) == 64
    assert "confirmation-token" not in prepared.draft.model_dump_json()

    patched = service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id=valid.id,
        expected_revision=valid.revision,
        patch={"name": "New proposal"},
    )
    assert patched.status == "editing"
    assert patched.revision == valid.revision + 1
    assert patched.proposal is None
    with pytest.raises(IntakeDraftConfirmationError, match="no prepared proposal"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id=valid.id,
            expected_revision=patched.revision,
            confirmation_token=prepared.confirmation_token,
        )


@pytest.mark.asyncio
async def test_activation_is_hash_bound_single_use_and_preserves_previous_on_failure(
    tmp_path: Path,
) -> None:
    service, activator = _service(tmp_path)
    draft = service.save(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id="draft-1",
        workflow_id="workflow-1",
        patch=_valid_payload(),
    )
    previous = {"workflow_id": "workflow-1", "name": "Previous active workflow"}
    activator.active[("tenant-a", "workflow-1")] = dict(previous)
    prepared = service.prepare(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id=draft.id,
        expected_revision=draft.revision,
    )

    with pytest.raises(IntakeDraftConfirmationError, match="confirmation token is invalid"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id=draft.id,
            expected_revision=draft.revision,
            confirmation_token="wrong-token-that-is-still-at-least-32-characters",
        )
    assert activator.calls == []

    activator.fail = True
    with pytest.raises(IntakeDraftActivationError, match="workflow activation failed"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id=draft.id,
            expected_revision=draft.revision,
            confirmation_token=prepared.confirmation_token,
        )
    assert activator.active[("tenant-a", "workflow-1")] == previous
    retryable = service.get(tenant_id="tenant-a", draft_id=draft.id)
    assert retryable is not None
    assert retryable.status == "prepared"

    activator.fail = False
    activated = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id=draft.id,
        expected_revision=draft.revision,
        confirmation_token=prepared.confirmation_token,
    )
    assert activated.draft.status == "activated"
    assert activated.workflow["name"] == "Lead capture"
    assert activator.active[("tenant-a", "workflow-1")]["name"] == "Lead capture"
    with pytest.raises(IntakeDraftConfirmationError, match="not prepared"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id=draft.id,
            expected_revision=draft.revision,
            confirmation_token=prepared.confirmation_token,
        )


def test_draft_payload_rejects_hidden_context_fields(tmp_path: Path) -> None:
    service, _ = _service(tmp_path)

    with pytest.raises(IntakeDraftValidationError, match="reserved fields: tenant_id"):
        service.save(
            tenant_id="tenant-a",
            actor_id="owner-a",
            patch={"tenant_id": "tenant-b"},
        )
