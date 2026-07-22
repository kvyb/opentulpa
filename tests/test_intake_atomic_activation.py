from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from opentulpa.intake.activation import IntakeWorkflowActivator
from opentulpa.intake.drafts import (
    IntakeDraftActivationError,
    IntakeDraftService,
    IntakeDraftStore,
)
from opentulpa.intake.service import IntakeWorkflowService

_NOW = datetime(2026, 7, 19, 12, tzinfo=UTC)
_TOKEN = "confirmation-token-that-is-at-least-32-characters"


def _workflow_payload(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "channel": "instagram_dm",
        "provider": "composio",
        "source_config": {"account": "instagram-main"},
        "intent_description": "Book a service",
        "required_fields": ["name", "email"],
        "field_guidance": {"email": "Ask once"},
        "sink_type": "local_csv",
        "sink_config": {"file_path": "leads.csv"},
    }


def _services(
    tmp_path: Path,
) -> tuple[IntakeWorkflowService, IntakeDraftStore, IntakeDraftService]:
    workflows = IntakeWorkflowService(
        db_path=tmp_path / "workflows.sqlite",
        project_root=tmp_path,
    )
    drafts = IntakeDraftStore(tmp_path / "drafts.sqlite")
    service = IntakeDraftService(
        drafts,
        workflow_activator=IntakeWorkflowActivator(workflows),
        clock=lambda: _NOW,
        token_factory=lambda: _TOKEN,
    )
    return workflows, drafts, service


def _seed_active(
    workflows: IntakeWorkflowService,
    *,
    tenant_id: str = "tenant-a",
    name: str = "Previous active workflow",
) -> dict[str, Any]:
    return workflows.upsert_workflow(
        customer_id=tenant_id,
        workflow_id="workflow-1",
        **_workflow_payload(name),
    )


def _prepare(service: IntakeDraftService, *, tenant_id: str = "tenant-a") -> tuple[str, int]:
    draft = service.save(
        tenant_id=tenant_id,
        actor_id="owner-a",
        draft_id="draft-1",
        workflow_id="workflow-1",
        patch=_workflow_payload("Replacement workflow"),
    )
    prepared = service.prepare(
        tenant_id=tenant_id,
        actor_id="owner-a",
        draft_id=draft.id,
        expected_revision=draft.revision,
    )
    return prepared.confirmation_token, draft.revision


@pytest.mark.asyncio
async def test_activation_commits_workflow_and_draft_together(tmp_path: Path) -> None:
    workflows, drafts, service = _services(tmp_path)
    _seed_active(workflows)
    token, revision = _prepare(service)

    activated = await service.activate(
        tenant_id="tenant-a",
        actor_id="owner-a",
        draft_id="draft-1",
        expected_revision=revision,
        confirmation_token=token,
    )

    active = workflows.get_workflow(customer_id="tenant-a", workflow_id="workflow-1")
    assert active is not None
    assert active["name"] == "Replacement workflow"
    assert active["revision"] == 2
    assert activated.draft.status == "activated"
    assert drafts.get(tenant_id="tenant-a", draft_id="draft-1") == activated.draft
    for path in (tmp_path / "workflows.sqlite", tmp_path / "drafts.sqlite"):
        with sqlite3.connect(path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


@pytest.mark.asyncio
async def test_crash_between_workflow_write_and_draft_finish_rolls_back_both(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflows, drafts, service = _services(tmp_path)
    _seed_active(workflows)
    token, revision = _prepare(service)

    def crash_before_draft_finish(*_: Any, **__: Any) -> Any:
        raise SystemExit("simulated process crash")

    monkeypatch.setattr(drafts, "finish_activation_in_transaction", crash_before_draft_finish)
    with pytest.raises(SystemExit, match="simulated process crash"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id="draft-1",
            expected_revision=revision,
            confirmation_token=token,
        )

    restarted_workflows = IntakeWorkflowService(
        db_path=tmp_path / "workflows.sqlite",
        project_root=tmp_path,
    )
    restarted_drafts = IntakeDraftStore(tmp_path / "drafts.sqlite")
    active = restarted_workflows.get_workflow(
        customer_id="tenant-a",
        workflow_id="workflow-1",
    )
    draft = restarted_drafts.get(tenant_id="tenant-a", draft_id="draft-1")
    assert active is not None
    assert active["name"] == "Previous active workflow"
    assert active["revision"] == 1
    assert draft is not None
    assert draft.status == "prepared"


@pytest.mark.asyncio
async def test_active_revision_conflict_preserves_concurrent_workflow_and_prepared_draft(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflows, drafts, service = _services(tmp_path)
    _seed_active(workflows)
    token, revision = _prepare(service)
    original_normalize = workflows._normalize_workflow_payload
    raced = False

    def normalize_after_concurrent_update(**kwargs: Any) -> dict[str, Any]:
        nonlocal raced
        normalized = original_normalize(**kwargs)
        if not raced:
            raced = True
            current = workflows.get_workflow(
                customer_id="tenant-a",
                workflow_id="workflow-1",
            )
            assert current is not None
            concurrent = dict(current)
            concurrent["name"] = "Concurrent active workflow"
            concurrent["routine_id"] = ""
            workflows._store.upsert_workflow_record(
                workflow=concurrent,
                created_at=str(current["created_at"]),
                updated_at=_NOW.isoformat(),
            )
        return normalized

    monkeypatch.setattr(
        workflows,
        "_normalize_workflow_payload",
        normalize_after_concurrent_update,
    )
    with pytest.raises(IntakeDraftActivationError, match="workflow activation failed"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id="draft-1",
            expected_revision=revision,
            confirmation_token=token,
        )

    active = workflows.get_workflow(customer_id="tenant-a", workflow_id="workflow-1")
    draft = drafts.get(tenant_id="tenant-a", draft_id="draft-1")
    assert active is not None
    assert active["name"] == "Concurrent active workflow"
    assert active["revision"] == 2
    assert draft is not None
    assert draft.status == "prepared"


@pytest.mark.asyncio
async def test_cross_tenant_workflow_conflict_rolls_back_draft_claim(tmp_path: Path) -> None:
    workflows, drafts, service = _services(tmp_path)
    _seed_active(workflows, tenant_id="tenant-b")
    token, revision = _prepare(service, tenant_id="tenant-a")

    with pytest.raises(IntakeDraftActivationError, match="workflow activation failed"):
        await service.activate(
            tenant_id="tenant-a",
            actor_id="owner-a",
            draft_id="draft-1",
            expected_revision=revision,
            confirmation_token=token,
        )

    tenant_b_active = workflows.get_workflow(
        customer_id="tenant-b",
        workflow_id="workflow-1",
    )
    assert tenant_b_active is not None
    assert tenant_b_active["name"] == "Previous active workflow"
    assert workflows.get_workflow(customer_id="tenant-a", workflow_id="workflow-1") is None
    draft = drafts.get(tenant_id="tenant-a", draft_id="draft-1")
    assert draft is not None
    assert draft.status == "prepared"
