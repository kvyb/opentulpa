from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from opentulpa.specs import (
    AgentSpecStore,
    AgentSpecWrite,
    CronTriggerSpec,
    EventTrigger,
    SpecConflictError,
    TriggerSpecStore,
    TriggerSpecWrite,
)

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


def _stores(tmp_path: Path) -> tuple[AgentSpecStore, TriggerSpecStore]:
    specs = AgentSpecStore(tmp_path / "specs.db", clock=lambda: NOW)
    triggers = TriggerSpecStore(
        tmp_path / "triggers.db",
        agent_specs=specs,
        clock=lambda: NOW,
    )
    return specs, triggers


def _spec(
    store: AgentSpecStore,
    *,
    tenant_id: str,
    spec_id: str,
    isolation: str,
):
    write = AgentSpecWrite(
        name=spec_id,
        instructions="Handle the run.",
        isolation=isolation,  # type: ignore[arg-type]
        tools=("knowledge_query",),
        memory_scope="spec",
        workspace_scope="none",
    )
    return store.create_revision(
        tenant_id=tenant_id,
        spec_id=spec_id,
        write=write,
        expected_revision=None,
        created_by="owner",
    )


def test_external_trigger_requires_authenticated_event_source() -> None:
    with pytest.raises(ValidationError, match="authenticated event"):
        TriggerSpecWrite(
            name="Unsafe external cron",
            source=CronTriggerSpec(expression="0 9 * * *", timezone="UTC"),
            exposure="external",
            agent_spec={"tenant_id": "tenant-a", "spec_id": "intake", "revision": 1},
            instruction="Handle it.",
        )

    with pytest.raises(ValidationError, match="require authentication"):
        TriggerSpecWrite(
            name="Unauthenticated webhook",
            source=EventTrigger(
                event_type="message_received",
                source="telegram",
                authentication="trusted_internal",
            ),
            exposure="external",
            agent_spec={"tenant_id": "tenant-a", "spec_id": "intake", "revision": 1},
            instruction="Handle it.",
        )


def test_trigger_store_enforces_exact_agent_spec_isolation_and_tenancy(tmp_path: Path) -> None:
    specs, triggers = _stores(tmp_path)
    private = _spec(specs, tenant_id="tenant-a", spec_id="routine", isolation="private")

    with pytest.raises(ValueError, match="isolation"):
        triggers.create_revision(
            tenant_id="tenant-a",
            trigger_id="incoming",
            write=TriggerSpecWrite(
                name="Incoming",
                source=EventTrigger(event_type="message_received", source="telegram"),
                exposure="external",
                agent_spec=private.ref,
                instruction="Handle it.",
            ),
            expected_revision=None,
            created_by="owner",
        )

    with pytest.raises(ValueError, match="same tenant"):
        triggers.create_revision(
            tenant_id="tenant-b",
            trigger_id="daily",
            write=TriggerSpecWrite(
                name="Daily",
                source=CronTriggerSpec(expression="0 9 * * *", timezone="UTC"),
                exposure="private",
                agent_spec=private.ref,
                instruction="Prepare a brief.",
            ),
            expected_revision=None,
            created_by="owner",
        )


def test_trigger_revisions_and_active_reference_use_compare_and_swap(tmp_path: Path) -> None:
    specs, triggers = _stores(tmp_path)
    routine = _spec(specs, tenant_id="tenant-a", spec_id="routine", isolation="private")
    first = triggers.create_revision(
        tenant_id="tenant-a",
        trigger_id="daily",
        write=TriggerSpecWrite(
            name="Daily",
            source=CronTriggerSpec(expression="0 9 * * *", timezone="UTC"),
            exposure="private",
            agent_spec=routine.ref,
            instruction="Prepare a brief.",
        ),
        expected_revision=None,
        created_by="owner",
    )
    second = triggers.create_revision(
        tenant_id="tenant-a",
        trigger_id="daily",
        write=TriggerSpecWrite(
            name="Daily",
            source=CronTriggerSpec(expression="0 10 * * *", timezone="UTC"),
            exposure="private",
            agent_spec=routine.ref,
            instruction="Prepare a brief.",
        ),
        expected_revision=1,
        created_by="owner",
    )

    triggers.activate(
        tenant_id="tenant-a",
        trigger_id="daily",
        revision=first.revision,
        expected_active_revision=None,
        updated_by="owner",
    )
    triggers.activate(
        tenant_id="tenant-a",
        trigger_id="daily",
        revision=second.revision,
        expected_active_revision=1,
        updated_by="owner",
    )
    assert triggers.get_active(tenant_id="tenant-a", trigger_id="daily") == second

    with pytest.raises(SpecConflictError, match="active TriggerSpec revision is 2"):
        triggers.activate(
            tenant_id="tenant-a",
            trigger_id="daily",
            revision=first.revision,
            expected_active_revision=1,
            updated_by="owner",
        )
