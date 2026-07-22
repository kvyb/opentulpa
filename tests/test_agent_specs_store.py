from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from opentulpa.specs import (
    AgentSpec,
    AgentSpecRef,
    AgentSpecService,
    AgentSpecStore,
    AgentSpecWrite,
    DeliverySpec,
    IntervalTrigger,
    SpecConflictError,
    TriggerSpecService,
    TriggerSpecStore,
    TriggerSpecWrite,
)

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


def _store(tmp_path: Path) -> AgentSpecStore:
    return AgentSpecStore(tmp_path / "specs.db", clock=lambda: NOW)


def _write(*, name: str = "Owner") -> AgentSpecWrite:
    return AgentSpecWrite(
        name=name,
        runtime_profile="owner",
        instructions="Help the owner.",
        isolation="private",
        tool_policy="profile_default",
        memory_scope="owner",
        workspace_scope="read_write",
        allow_delegation=True,
    )


def test_agent_spec_revisions_are_immutable_idempotent_and_tenant_scoped(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    first = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(),
        expected_revision=None,
        created_by="owner",
    )
    retry = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(),
        expected_revision=None,
        created_by="owner",
    )
    other = store.create_revision(
        tenant_id="tenant-b",
        spec_id="owner",
        write=_write(),
        expected_revision=None,
        created_by="owner",
    )

    assert retry == first
    assert other.revision == 1
    assert store.get_revision(first.ref) == first

    second = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(name="Owner v2"),
        expected_revision=1,
        created_by="owner",
    )
    assert second.revision == 2
    assert store.get_revision(first.ref) == first

    with pytest.raises(SpecConflictError, match="revision is 2"):
        store.create_revision(
            tenant_id="tenant-a",
            spec_id="owner",
            write=_write(name="Stale edit"),
            expected_revision=1,
            created_by="owner",
        )


def test_agent_spec_activation_is_compare_and_swap_and_can_roll_back(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(),
        expected_revision=None,
        created_by="owner",
    )
    second = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(name="Owner v2"),
        expected_revision=1,
        created_by="owner",
    )

    store.activate(first.ref, expected_active_revision=None, updated_by="owner")
    store.activate(second.ref, expected_active_revision=1, updated_by="owner")
    assert store.get_active(tenant_id="tenant-a", spec_id="owner") == second

    with pytest.raises(SpecConflictError, match="active AgentSpec revision is 2"):
        store.activate(first.ref, expected_active_revision=1, updated_by="owner")

    store.activate(first.ref, expected_active_revision=2, updated_by="owner")
    assert store.get_active(tenant_id="tenant-a", spec_id="owner") == first


def test_agent_spec_activation_preflight_preserves_previous_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(),
        expected_revision=None,
        created_by="owner",
    )
    invalid = store.create_revision(
        tenant_id="tenant-a",
        spec_id="owner",
        write=_write(name="Broken owner"),
        expected_revision=1,
        created_by="owner",
    )
    store.activate(first.ref, expected_active_revision=None, updated_by="owner")
    validated: list[int] = []

    def preflight(spec: AgentSpec) -> None:
        revision = spec.revision
        validated.append(revision)
        if revision == invalid.revision:
            raise ValueError("cannot compile")

    service = AgentSpecService(store, validate_activation=preflight)

    with pytest.raises(ValueError, match="cannot compile"):
        service.activate(
            tenant_id="tenant-a",
            actor_id="owner",
            spec_id="owner",
            revision=invalid.revision,
            expected_active_revision=first.revision,
        )

    assert validated == [invalid.revision]
    assert service.get_active(tenant_id="tenant-a", spec_id="owner") == first


def test_trigger_activation_preflights_its_exact_agent_spec(tmp_path: Path) -> None:
    agent_store = _store(tmp_path)
    routine = agent_store.create_revision(
        tenant_id="tenant-a",
        spec_id="routine",
        write=AgentSpecWrite(
            name="Routine",
            runtime_profile="routine",
            instructions="Run scheduled work.",
            isolation="private",
            tool_policy="allowlist",
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=None,
        created_by="owner",
    )
    trigger_store = TriggerSpecStore(tmp_path / "triggers.db", agent_specs=agent_store)
    trigger = trigger_store.create_revision(
        tenant_id="tenant-a",
        trigger_id="daily",
        write=TriggerSpecWrite(
            name="Daily",
            source=IntervalTrigger(every_seconds=3600),
            exposure="private",
            agent_spec=AgentSpecRef(
                tenant_id="tenant-a",
                spec_id="routine",
                revision=routine.revision,
            ),
            instruction="Prepare an update.",
            delivery=DeliverySpec(mode="owner"),
        ),
        expected_revision=None,
        created_by="owner",
    )
    service = TriggerSpecService(
        trigger_store,
        validate_activation=lambda _trigger: (_ for _ in ()).throw(
            ValueError("cannot compile target AgentSpec")
        ),
    )

    with pytest.raises(ValueError, match="cannot compile target AgentSpec"):
        service.activate(
            tenant_id="tenant-a",
            actor_id="owner",
            trigger_id=trigger.id,
            revision=trigger.revision,
            expected_active_revision=None,
        )

    assert service.get_active(tenant_id="tenant-a", trigger_id=trigger.id) is None
