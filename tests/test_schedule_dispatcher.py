from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from opentulpa.deep_agent import AgentApproval, AgentRunSnapshot
from opentulpa.schedules import AgentJob, At, Cron, Reminder, ScheduleService, ScheduleWrite
from opentulpa.specs import (
    AgentSpecStore,
    TriggerSpecService,
    TriggerSpecStore,
    seed_default_agent_spec_refs,
)
from opentulpa.specs.dispatcher import TriggerDispatcher, TriggerExecutionStore


class _FakeAgentService:
    def __init__(self, *, status: str = "completed", final_text: str = "finished") -> None:
        self.status = status
        self.final_text = final_text
        self.requests: list[Any] = []

    async def run(self, request):
        self.requests.append(request)
        approvals = ()
        if self.status == "interrupted":
            approvals = (
                AgentApproval(
                    id="approval-1",
                    tool_name="integration_invoke",
                    description="Send the report",
                    arguments={"recipient": "owner@example.com"},
                    allowed_decisions=("approve", "reject"),
                ),
            )
        return AgentRunSnapshot(
            run_id="run-1",
            context=request.context,
            status=self.status,
            final_text=self.final_text if self.status == "completed" else "",
            approvals=approvals,
        )


def _services(tmp_path: Path, now: datetime):
    agent_specs = AgentSpecStore(tmp_path / "agent_specs.db", clock=lambda: now)
    trigger_store = TriggerSpecStore(
        tmp_path / "trigger_specs.db",
        agent_specs=agent_specs,
        clock=lambda: now,
    )

    def resolve(tenant_id: str):
        active = agent_specs.get_active_ref(tenant_id=tenant_id, spec_id="routine")
        if active is None:
            active = seed_default_agent_spec_refs(
                agent_specs,
                tenant_id=tenant_id,
                actor_id="test",
            )["routine"]
        return active

    trigger_specs = TriggerSpecService(trigger_store)
    schedules = ScheduleService(
        trigger_specs,
        resolve_agent_spec=resolve,
        clock=lambda: now,
    )
    return agent_specs, trigger_store, trigger_specs, schedules


def _execution_status(
    db_path: Path,
    *,
    trigger_id: str,
    revision: int,
    fire_key: str,
) -> str:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT status FROM trigger_executions
            WHERE tenant_id = 'tenant-1'
              AND trigger_id = ?
              AND trigger_revision = ?
              AND fire_key = ?
            """,
            (trigger_id, revision, fire_key),
        ).fetchone()
    assert row is not None
    return str(row[0])


@pytest.mark.asyncio
async def test_one_off_schedule_runs_through_trigger_and_persists_disabled(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 19, 12, tzinfo=UTC)
    agent_specs, trigger_store, trigger_specs, schedules = _services(tmp_path, now)
    schedule = schedules.save(
        tenant_id="tenant-1",
        write=ScheduleWrite(
            name="Call back",
            trigger=At(run_at=now + timedelta(minutes=5), timezone="Europe/Moscow"),
            action=Reminder(message="Call the customer"),
            notify_owner=True,
        ),
    )
    trigger = trigger_specs.list_active(tenant_id="tenant-1")[0]
    deliveries = []

    async def deliver(**kwargs):
        deliveries.append(kwargs)

    agent = _FakeAgentService(final_text="Call the customer")
    dispatcher = TriggerDispatcher(
        triggers=trigger_store,
        agent_specs=agent_specs,
        agent_service=agent,
        executions=TriggerExecutionStore(tmp_path / "executions.db", clock=lambda: now),
        deliver=deliver,
        clock=lambda: now,
    )
    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        trigger_revision=trigger.revision,
    )

    updated = schedules.get(tenant_id="tenant-1", schedule_id=schedule.id)
    assert updated is not None
    assert isinstance(updated.action, Reminder)
    assert updated.action.message == "Call the customer"
    assert updated.enabled is False
    assert updated.revision == 2
    assert agent.requests[0].context.agent_spec == trigger.agent_spec
    assert agent.requests[0].context.trust_class == "background"
    assert agent.requests[0].text.endswith("Call the customer")
    assert deliveries[0]["mode"] == "owner"


@pytest.mark.asyncio
async def test_one_off_delivery_retries_after_disable_without_rerunning_agent(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 19, 12, tzinfo=UTC)
    agent_specs, trigger_store, trigger_specs, schedules = _services(tmp_path, now)
    schedule = schedules.save(
        tenant_id="tenant-1",
        write=ScheduleWrite(
            name="One-time reminder",
            trigger=At(run_at=now + timedelta(minutes=5), timezone="UTC"),
            action=Reminder(message="Check the oven"),
            notify_owner=True,
        ),
    )
    trigger = trigger_specs.list_active(tenant_id="tenant-1")[0]
    execution_path = tmp_path / "executions.db"
    agent = _FakeAgentService(final_text="Check the oven")
    deliveries = 0

    async def fail_once(**kwargs):
        nonlocal deliveries
        deliveries += 1
        if deliveries == 1:
            raise RuntimeError("notification transport unavailable")

    dispatcher = TriggerDispatcher(
        triggers=trigger_store,
        agent_specs=agent_specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: now),
        deliver=fail_once,
        clock=lambda: now,
    )
    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        trigger_revision=trigger.revision,
    )

    disabled = schedules.get(tenant_id="tenant-1", schedule_id=schedule.id)
    assert disabled is not None
    assert disabled.enabled is False
    with sqlite3.connect(execution_path) as conn:
        assert conn.execute(
            "SELECT status FROM trigger_delivery_outbox"
        ).fetchone() == ("pending",)

    restarted = TriggerDispatcher(
        triggers=trigger_store,
        agent_specs=agent_specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: now),
        deliver=fail_once,
        clock=lambda: now,
    )
    await restarted._drain_deliveries()

    assert deliveries == 2
    assert len(agent.requests) == 1
    with sqlite3.connect(execution_path) as conn:
        assert conn.execute(
            "SELECT status FROM trigger_delivery_outbox"
        ).fetchone() == ("delivered",)


@pytest.mark.asyncio
async def test_schedule_interrupt_is_never_auto_approved(tmp_path: Path) -> None:
    now = datetime(2026, 7, 19, 12, tzinfo=UTC)
    agent_specs, trigger_store, trigger_specs, schedules = _services(tmp_path, now)
    schedules.save(
        tenant_id="tenant-1",
        write=ScheduleWrite(
            name="Daily report",
            trigger=Cron(expression="0 9 * * *", timezone="Europe/Moscow"),
            action=AgentJob(instruction="Prepare and send the daily report"),
            notify_owner=False,
        ),
    )
    trigger = trigger_specs.list_active(tenant_id="tenant-1")[0]
    deliveries = []

    async def deliver(**kwargs):
        deliveries.append(kwargs)

    agent = _FakeAgentService(status="interrupted")
    dispatcher = TriggerDispatcher(
        triggers=trigger_store,
        agent_specs=agent_specs,
        agent_service=agent,
        executions=TriggerExecutionStore(tmp_path / "executions.db", clock=lambda: now),
        deliver=deliver,
        clock=lambda: now,
    )
    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        trigger_revision=trigger.revision,
    )

    assert len(agent.requests) == 1
    assert deliveries[0]["mode"] == "owner"
    assert deliveries[0]["snapshot"].status == "interrupted"


@pytest.mark.asyncio
async def test_duplicate_cron_workers_claim_one_schedule_occurrence(tmp_path: Path) -> None:
    now = datetime(2026, 7, 19, 12, 34, 45, tzinfo=UTC)
    agent_specs, trigger_store, trigger_specs, schedules = _services(tmp_path, now)
    schedules.save(
        tenant_id="tenant-1",
        write=ScheduleWrite(
            name="Daily report",
            trigger=Cron(expression="34 12 * * *", timezone="UTC"),
            action=AgentJob(instruction="Prepare the daily report"),
            notify_owner=True,
        ),
    )
    trigger = trigger_specs.list_active(tenant_id="tenant-1")[0]
    agent = _FakeAgentService(final_text="Report ready")
    deliveries = []

    async def deliver(**kwargs):
        deliveries.append(kwargs)

    execution_path = tmp_path / "executions.db"
    dispatchers = [
        TriggerDispatcher(
            triggers=trigger_store,
            agent_specs=agent_specs,
            agent_service=agent,
            executions=TriggerExecutionStore(execution_path, clock=lambda: now),
            deliver=deliver,
            clock=lambda: now,
        )
        for _ in range(2)
    ]
    await asyncio.gather(
        *(
            dispatcher._run_scheduled(
                tenant_id="tenant-1",
                trigger_id=trigger.id,
                trigger_revision=trigger.revision,
            )
            for dispatcher in dispatchers
        )
    )

    assert len(agent.requests) == 1
    assert len(deliveries) == 1
    fire_key = "2026-07-19T12:34:00+00:00"
    assert _execution_status(
        execution_path,
        trigger_id=trigger.id,
        revision=trigger.revision,
        fire_key=fire_key,
    ) == "completed"
