from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from opentulpa.bootstrap.models import OutboxEvent
from opentulpa.deep_agent.contracts import AgentApproval, AgentRunSnapshot
from opentulpa.evolution.models import EvolutionEvent
from opentulpa.notifications import (
    BootstrapNotificationSink,
    EvolutionNotificationSink,
    NotificationService,
    NotificationStore,
    TriggerNotificationSink,
)
from opentulpa.specs import (
    AgentSpecRef,
    AtTrigger,
    DeliverySpec,
    OriginRef,
    TriggerSpec,
)
from opentulpa.tooling.contract import AgentRunContext


@pytest.mark.asyncio
async def test_trigger_sink_preserves_run_context_and_pending_approvals(
    tmp_path: Path,
) -> None:
    store = NotificationStore(tmp_path / "notifications.db")
    service = NotificationService(store)
    sink = TriggerNotificationSink(service)
    spec_ref = AgentSpecRef(tenant_id="tenant-a", spec_id="routine", revision=2)
    trigger = TriggerSpec(
        tenant_id="tenant-a",
        id="daily",
        revision=3,
        content_digest="a" * 64,
        created_at=datetime(2026, 7, 20, tzinfo=UTC),
        created_by="owner",
        name="Daily report",
        source=AtTrigger(
            run_at=datetime(2026, 7, 21, tzinfo=UTC),
            timezone="UTC",
        ),
        exposure="private",
        agent_spec=spec_ref,
        instruction="Prepare the report.",
        delivery=DeliverySpec(mode="owner"),
    )
    context = AgentRunContext(
        tenant_id="tenant-a",
        actor_id="scheduler",
        thread_id="trigger:daily:fire",
        channel="routine",
        run_kind="routine",
        correlation_id="trigger:daily:r3:fire",
        origin=OriginRef(
            interface="web",
            source_id="schedule",
            conversation_id="owner-thread",
        ),
        agent_spec=spec_ref,
        trust_class="background",
    )
    snapshot = AgentRunSnapshot(
        run_id="run-waiting",
        context=context,
        status="interrupted",
        approvals=(
            AgentApproval(
                id="approval-1",
                tool_name="integration_invoke",
                description="Send the report.",
                arguments={"hidden": "not persisted"},
                allowed_decisions=("approve", "reject"),
            ),
        ),
    )

    await sink(trigger=trigger, snapshot=snapshot, mode="owner")

    notification = store.list_unacked(
        tenant_id="tenant-a",
        consumer_id="web:owner",
    )[0]
    assert notification.kind == "approval.required"
    assert notification.run_id == "run-waiting"
    assert notification.thread_id == "trigger:daily:fire"
    assert notification.approvals[0].approval_id == "approval-1"
    assert "hidden" not in notification.model_dump_json()


@pytest.mark.asyncio
async def test_evolution_sink_delivers_failure_to_original_owner_thread(tmp_path: Path) -> None:
    store = NotificationStore(tmp_path / "notifications.db")
    sink = EvolutionNotificationSink(NotificationService(store))
    event = EvolutionEvent(
        event_key="candidate:candidate-1:completed:failed",
        event_type="candidate.failed",
        release_id="release-1",
        origin={
            "tenant_id": "tenant-a",
            "actor_id": "owner",
            "thread_id": "owner-thread",
            "correlation_id": "correlation-1",
            "channel": "web",
            "run_kind": "owner",
            "origin": (
                '{"interface":"web","source_id":"owner-web",'
                '"conversation_id":"owner-thread","message_id":"message-1"}'
            ),
        },
        payload={
            "candidate_id": "candidate-1",
            "status": "failed",
            "summary": "Candidate evaluation failed; the active release was retained.",
        },
    )

    await sink.deliver(event)

    notification = store.list_unacked(
        tenant_id="tenant-a",
        consumer_id="telegram:instance",
    )[0]
    assert notification.kind == "evolution.candidate.failed"
    assert notification.status == "failed"
    assert notification.thread_id == "owner-thread"
    assert notification.origin is not None
    assert notification.origin.interface == "web"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "status", "expected_text"),
    [
        (
            "build.preparing",
            "preparing",
            "Preparing a new OpenTulpa build. I will report again before it restarts.",
        ),
        (
            "build.switching",
            "switching",
            "OpenTulpa is restarting now. I will report again when it is back online.",
        ),
        ("build.active", "active", "OpenTulpa is back online. The new build is active."),
        (
            "promotion.failed",
            "failed",
            "The new build failed, so OpenTulpa kept or restored the previous build.",
        ),
        (
            "runtime_env.restarting",
            "restarting",
            "OpenTulpa is restarting to apply the runtime environment update. I will report again when it is back online.",
        ),
        (
            "runtime_env.updated",
            "updated",
            "OpenTulpa is back online. The runtime environment update is active.",
        ),
    ],
)
async def test_evolution_sink_explains_build_transition(
    tmp_path: Path,
    event_type: str,
    status: str,
    expected_text: str,
) -> None:
    store = NotificationStore(tmp_path / f"{event_type}.db")
    sink = EvolutionNotificationSink(NotificationService(store))

    await sink.deliver(
        EvolutionEvent(
            event_key=f"transition:{event_type}",
            event_type=event_type,
            release_id="release-1",
            origin={
                "tenant_id": "tenant-a",
                "thread_id": "owner-thread",
                "correlation_id": "correlation-1",
                "channel": "web",
            },
            payload={"status": status},
        )
    )

    notification = store.list_unacked(
        tenant_id="tenant-a",
        consumer_id="web:owner",
    )[0]
    assert notification.text == expected_text
    assert notification.thread_id == "owner-thread"


@pytest.mark.asyncio
async def test_system_owned_evolution_and_bootstrap_events_are_consumed_without_guessing_tenant(
    tmp_path: Path,
) -> None:
    store = NotificationStore(tmp_path / "notifications.db")
    service = NotificationService(store)

    await EvolutionNotificationSink(service).deliver(
        EvolutionEvent(
            event_key="startup:recovered",
            event_type="release.recovered",
            release_id="initial",
            payload={"status": "active"},
        )
    )
    await BootstrapNotificationSink(service).deliver(
        OutboxEvent(
            event_key="initial:active",
            event_type="release.active",
            payload={"status": "active"},
        )
    )

    assert store.list_unacked(tenant_id="tenant-a", consumer_id="web:owner") == []
