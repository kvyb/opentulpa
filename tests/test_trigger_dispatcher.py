from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from opentulpa.deep_agent import AgentApproval, AgentRunSnapshot
from opentulpa.notifications import (
    NotificationService,
    NotificationStore,
    TriggerNotificationSink,
)
from opentulpa.specs import (
    AgentSpecStore,
    AgentSpecWrite,
    CronTriggerSpec,
    DeliverySpec,
    EventTrigger,
    TriggerSpecStore,
    TriggerSpecWrite,
)
from opentulpa.specs.dispatcher import TriggerDispatcher, TriggerExecutionStore

NOW = datetime(2026, 7, 20, 12, 34, 45, tzinfo=UTC)


class _Agent:
    def __init__(self, *, status: str = "completed") -> None:
        self.status = status
        self.requests: list[Any] = []

    async def run(self, request):
        self.requests.append(request)
        approvals = ()
        if self.status == "interrupted":
            approvals = (
                AgentApproval(
                    id="approval-1",
                    tool_name="integration_invoke",
                    description="External write",
                    arguments={},
                    allowed_decisions=("approve", "reject"),
                ),
            )
        return AgentRunSnapshot(
            run_id=f"run-{len(self.requests)}",
            context=request.context,
            status=self.status,
            final_text="finished" if self.status == "completed" else "",
            approvals=approvals,
        )


class _RaisingAgent:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def run(self, request):
        self.requests.append(request)
        raise RuntimeError("secret-provider-token")


def _stores(tmp_path: Path):
    specs = AgentSpecStore(tmp_path / "specs.db", clock=lambda: NOW)
    triggers = TriggerSpecStore(
        tmp_path / "triggers.db",
        agent_specs=specs,
        clock=lambda: NOW,
    )
    return specs, triggers


def _spec(specs: AgentSpecStore, *, isolation: str, spec_id: str):
    return specs.create_revision(
        tenant_id="tenant-1",
        spec_id=spec_id,
        write=AgentSpecWrite(
            name=spec_id,
            runtime_profile="routine" if isolation == "private" else "intake",
            instructions="Handle the event.",
            isolation=isolation,  # type: ignore[arg-type]
            tools=("knowledge_query",),
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=None,
        created_by="owner",
    )


def _event_trigger(
    specs: AgentSpecStore,
    triggers: TriggerSpecStore,
    *,
    trigger_id: str,
):
    intake = _spec(specs, isolation="external", spec_id=f"{trigger_id}-agent")
    trigger = triggers.create_revision(
        tenant_id="tenant-1",
        trigger_id=trigger_id,
        write=TriggerSpecWrite(
            name=trigger_id,
            source=EventTrigger(event_type="message_received", source="webhook"),
            exposure="external",
            agent_spec=intake.ref,
            instruction="Handle the event.",
            delivery=DeliverySpec(mode="owner"),
        ),
        expected_revision=None,
        created_by="owner",
    )
    triggers.activate(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        revision=trigger.revision,
        expected_active_revision=None,
        updated_by="owner",
    )
    return trigger


def _delivery_state(db_path: Path) -> tuple[str, int]:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, attempt_count FROM trigger_delivery_outbox"
        ).fetchone()
    assert row is not None
    return str(row[0]), int(row[1])


@pytest.mark.asyncio
async def test_cron_dispatch_pins_agent_revision_and_fences_duplicates(
    tmp_path: Path,
) -> None:
    specs, triggers = _stores(tmp_path)
    first = _spec(specs, isolation="private", spec_id="routine")
    specs.create_revision(
        tenant_id="tenant-1",
        spec_id="routine",
        write=AgentSpecWrite(
            name="new routine",
            runtime_profile="routine",
            instructions="A newer configuration.",
            tools=("knowledge_query",),
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=1,
        created_by="owner",
    )
    trigger = triggers.create_revision(
        tenant_id="tenant-1",
        trigger_id="daily",
        write=TriggerSpecWrite(
            name="Daily",
            source=CronTriggerSpec(expression="34 12 * * *", timezone="UTC"),
            exposure="private",
            agent_spec=first.ref,
            instruction="Prepare the report.",
        ),
        expected_revision=None,
        created_by="owner",
    )
    triggers.activate(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        revision=trigger.revision,
        expected_active_revision=None,
        updated_by="owner",
    )
    agent = _Agent()
    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(tmp_path / "executions.db"),
        clock=lambda: NOW,
    )

    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id="daily",
        trigger_revision=1,
    )
    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id="daily",
        trigger_revision=1,
    )

    assert len(agent.requests) == 1
    request = agent.requests[0]
    assert request.context.agent_spec == first.ref
    assert request.context.trust_class == "background"
    assert request.context.run_kind == "routine"
    assert request.idempotency_key.startswith("trigger:daily:")


@pytest.mark.asyncio
async def test_external_event_requires_authentication_and_is_idempotent(
    tmp_path: Path,
) -> None:
    specs, triggers = _stores(tmp_path)
    intake = _spec(specs, isolation="external", spec_id="intake")
    trigger = triggers.create_revision(
        tenant_id="tenant-1",
        trigger_id="telegram-message",
        write=TriggerSpecWrite(
            name="Telegram message",
            source=EventTrigger(event_type="message_received", source="telegram"),
            exposure="external",
            agent_spec=intake.ref,
            instruction="Classify the message.",
            delivery=DeliverySpec(mode="none"),
        ),
        expected_revision=None,
        created_by="owner",
    )
    triggers.activate(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        revision=1,
        expected_active_revision=None,
        updated_by="owner",
    )
    agent = _Agent()
    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(tmp_path / "executions.db"),
    )

    with pytest.raises(PermissionError, match="authenticated"):
        await dispatcher.dispatch_event(
            tenant_id="tenant-1",
            trigger_id=trigger.id,
            source_event_id="update-42",
            event_type="message_received",
            source="telegram",
            authenticated=False,
        )
    snapshot = await dispatcher.dispatch_event(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        source_event_id="update-42",
        event_type="message_received",
        source="telegram",
        authenticated=True,
        payload={"text": "hello"},
    )
    duplicate = await dispatcher.dispatch_event(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        source_event_id="update-42",
        event_type="message_received",
        source="telegram",
        authenticated=True,
        payload={"text": "hello"},
    )

    assert snapshot is not None
    assert duplicate is None
    assert len(agent.requests) == 1
    assert agent.requests[0].context.trust_class == "external"
    assert '"text":"hello"' in agent.requests[0].text


@pytest.mark.asyncio
async def test_interrupted_trigger_always_notifies_owner(tmp_path: Path) -> None:
    specs, triggers = _stores(tmp_path)
    routine = _spec(specs, isolation="private", spec_id="routine")
    trigger = triggers.create_revision(
        tenant_id="tenant-1",
        trigger_id="background",
        write=TriggerSpecWrite(
            name="Background",
            source=CronTriggerSpec(expression="34 12 * * *", timezone="UTC"),
            exposure="private",
            agent_spec=routine.ref,
            instruction="Perform an external write.",
            delivery=DeliverySpec(mode="none"),
        ),
        expected_revision=None,
        created_by="owner",
    )
    triggers.activate(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        revision=1,
        expected_active_revision=None,
        updated_by="owner",
    )
    deliveries = []

    async def deliver(**kwargs):
        deliveries.append(kwargs)

    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=_Agent(status="interrupted"),
        executions=TriggerExecutionStore(tmp_path / "executions.db"),
        deliver=deliver,
        clock=lambda: NOW,
    )
    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        trigger_revision=1,
    )

    assert deliveries[0]["mode"] == "owner"
    assert deliveries[0]["snapshot"].status == "interrupted"


@pytest.mark.parametrize("raises_before_snapshot", [False, True])
@pytest.mark.asyncio
async def test_failed_trigger_always_persists_sanitized_owner_notification(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    *,
    raises_before_snapshot: bool,
) -> None:
    specs, triggers = _stores(tmp_path)
    routine = _spec(specs, isolation="private", spec_id="routine")
    trigger = triggers.create_revision(
        tenant_id="tenant-1",
        trigger_id="failing-background",
        write=TriggerSpecWrite(
            name="Failing background task",
            source=CronTriggerSpec(expression="34 12 * * *", timezone="UTC"),
            exposure="private",
            agent_spec=routine.ref,
            instruction="Prepare the report.",
            delivery=DeliverySpec(mode="none"),
        ),
        expected_revision=None,
        created_by="owner",
    )
    triggers.activate(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        revision=trigger.revision,
        expected_active_revision=None,
        updated_by="owner",
    )
    notifications = NotificationService(
        NotificationStore(tmp_path / "notifications.db"),
        clock=lambda: NOW,
    )
    notification_sink = TriggerNotificationSink(notifications)
    delivery_modes = []

    async def deliver(**kwargs):
        delivery_modes.append(kwargs["mode"])
        await notification_sink(**kwargs)

    agent = _RaisingAgent() if raises_before_snapshot else _Agent(status="failed")
    execution_path = tmp_path / "executions.db"
    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: NOW),
        deliver=deliver,
        clock=lambda: NOW,
    )
    caplog.set_level(logging.ERROR)

    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        trigger_revision=trigger.revision,
    )
    await dispatcher._run_scheduled(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        trigger_revision=trigger.revision,
    )

    stored = await notifications.wait(
        tenant_id="tenant-1",
        consumer_id="test-owner",
    )
    assert len(agent.requests) == 1
    assert delivery_modes == ["owner"]
    assert len(stored) == 1
    assert stored[0].kind == "run.failed"
    assert stored[0].status == "failed"
    assert stored[0].run_id is not None
    assert "secret-provider-token" not in stored[0].text
    assert "secret-provider-token" not in caplog.text
    with sqlite3.connect(execution_path) as conn:
        execution = conn.execute(
            """
            SELECT status, run_id, attempt_count
            FROM trigger_executions
            WHERE tenant_id = ? AND trigger_id = ? AND trigger_revision = ?
            """,
            (trigger.tenant_id, trigger.id, trigger.revision),
        ).fetchone()
    assert execution == ("failed", stored[0].run_id, 1)


@pytest.mark.asyncio
async def test_failed_delivery_is_persisted_and_drained_after_restart_without_rerun(
    tmp_path: Path,
) -> None:
    specs, triggers = _stores(tmp_path)
    trigger = _event_trigger(specs, triggers, trigger_id="durable-delivery")
    agent = _Agent()
    execution_path = tmp_path / "executions.db"
    delivered: list[AgentRunSnapshot] = []

    async def fail_delivery(**kwargs):
        delivered.append(kwargs["snapshot"])
        raise RuntimeError("temporary notification outage")

    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: NOW),
        deliver=fail_delivery,
        clock=lambda: NOW,
    )
    snapshot = await dispatcher.dispatch_event(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        source_event_id="event-1",
        event_type="message_received",
        source="webhook",
        authenticated=True,
    )

    assert snapshot is not None
    assert len(agent.requests) == 1
    assert _delivery_state(execution_path) == ("pending", 1)

    async def deliver_after_restart(**kwargs):
        delivered.append(kwargs["snapshot"])

    restarted = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: NOW),
        deliver=deliver_after_restart,
        clock=lambda: NOW,
    )
    restarted.start()
    await asyncio.gather(*tuple(restarted._delivery_tasks))
    restarted.shutdown()

    assert len(agent.requests) == 1
    assert [item.run_id for item in delivered] == [snapshot.run_id, snapshot.run_id]
    assert delivered[-1].final_text == "finished"
    assert _delivery_state(execution_path) == ("delivered", 2)


@pytest.mark.asyncio
async def test_started_dispatcher_retries_transient_delivery_without_new_trigger(
    tmp_path: Path,
) -> None:
    specs, triggers = _stores(tmp_path)
    trigger = _event_trigger(specs, triggers, trigger_id="autonomous-retry")
    agent = _Agent()
    execution_path = tmp_path / "executions.db"
    delivery_calls = 0

    async def fail_once(**kwargs):
        nonlocal delivery_calls
        delivery_calls += 1
        if delivery_calls == 1:
            raise RuntimeError("temporary notification outage")

    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: NOW),
        deliver=fail_once,
        clock=lambda: NOW,
        delivery_retry_base_seconds=0.01,
        delivery_retry_max_seconds=0.02,
    )
    dispatcher.start()
    await dispatcher.dispatch_event(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        source_event_id="event-1",
        event_type="message_received",
        source="webhook",
        authenticated=True,
    )
    await asyncio.gather(*tuple(dispatcher._delivery_tasks))
    dispatcher.shutdown()

    assert delivery_calls == 2
    assert len(agent.requests) == 1
    assert _delivery_state(execution_path) == ("delivered", 2)


@pytest.mark.asyncio
async def test_notification_replay_after_publish_is_deduplicated(tmp_path: Path) -> None:
    specs, triggers = _stores(tmp_path)
    trigger = _event_trigger(specs, triggers, trigger_id="deduped-delivery")
    agent = _Agent()
    execution_path = tmp_path / "executions.db"
    notifications = NotificationService(
        NotificationStore(tmp_path / "notifications.db"),
        clock=lambda: NOW,
    )
    sink = TriggerNotificationSink(notifications)
    calls = 0

    async def publish_then_crash(**kwargs):
        nonlocal calls
        calls += 1
        await sink(**kwargs)
        if calls == 1:
            raise RuntimeError("process stopped after publish")

    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: NOW),
        deliver=publish_then_crash,
        clock=lambda: NOW,
    )
    await dispatcher.dispatch_event(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        source_event_id="event-1",
        event_type="message_received",
        source="webhook",
        authenticated=True,
    )
    assert _delivery_state(execution_path) == ("pending", 1)

    restarted = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=TriggerExecutionStore(execution_path, clock=lambda: NOW),
        deliver=publish_then_crash,
        clock=lambda: NOW,
    )
    await restarted._drain_deliveries()

    stored = await notifications.wait(
        tenant_id="tenant-1",
        consumer_id="owner",
    )
    assert calls == 2
    assert len(agent.requests) == 1
    assert len(stored) == 1
    assert stored[0].run_id == "run-1"
    assert _delivery_state(execution_path) == ("delivered", 2)


@pytest.mark.asyncio
async def test_delivery_lease_reclaim_fences_the_stale_worker(tmp_path: Path) -> None:
    specs, triggers = _stores(tmp_path)
    trigger = _event_trigger(specs, triggers, trigger_id="leased-delivery")
    current = NOW
    execution_path = tmp_path / "executions.db"
    executions = TriggerExecutionStore(execution_path, clock=lambda: current)
    agent = _Agent()
    dispatcher = TriggerDispatcher(
        triggers=triggers,
        agent_specs=specs,
        agent_service=agent,
        executions=executions,
        clock=lambda: current,
    )
    await dispatcher.dispatch_event(
        tenant_id="tenant-1",
        trigger_id=trigger.id,
        source_event_id="event-1",
        event_type="message_received",
        source="webhook",
        authenticated=True,
    )

    first = executions.claim_deliveries(lease_seconds=30)
    assert len(first) == 1
    assert executions.claim_deliveries(lease_seconds=30) == []
    current += timedelta(seconds=31)
    reclaimed = executions.claim_deliveries(lease_seconds=30)

    assert len(reclaimed) == 1
    assert not executions.complete_delivery(first[0])
    assert executions.complete_delivery(reclaimed[0])
    assert len(agent.requests) == 1
    assert _delivery_state(execution_path) == ("delivered", 2)


def test_execution_claim_reclaims_only_expired_running_lease(tmp_path: Path) -> None:
    specs, triggers = _stores(tmp_path)
    routine = _spec(specs, isolation="private", spec_id="routine")
    trigger = triggers.create_revision(
        tenant_id="tenant-1",
        trigger_id="leased",
        write=TriggerSpecWrite(
            name="Leased",
            source=CronTriggerSpec(expression="34 12 * * *", timezone="UTC"),
            exposure="private",
            agent_spec=routine.ref,
            instruction="Run once.",
        ),
        expected_revision=None,
        created_by="owner",
    )
    current = NOW
    executions = TriggerExecutionStore(
        tmp_path / "executions.db",
        clock=lambda: current,
    )

    first_lease = executions.claim(trigger, "fire-1", lease_seconds=30)
    assert first_lease
    assert not executions.claim(trigger, "fire-1", lease_seconds=30)
    current += timedelta(seconds=31)
    second_lease = executions.claim(trigger, "fire-1", lease_seconds=30)
    assert second_lease
    assert not executions.finish(
        trigger,
        "fire-1",
        lease_token=first_lease,
        status="completed",
        run_id="stale-run",
    )
    assert executions.finish(
        trigger,
        "fire-1",
        lease_token=second_lease,
        status="completed",
        run_id="run-1",
    )
    current += timedelta(seconds=31)
    assert not executions.claim(trigger, "fire-1", lease_seconds=30)
