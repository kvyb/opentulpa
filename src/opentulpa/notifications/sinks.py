"""Adapters from background execution events into the universal notification stream."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

from opentulpa.notifications.models import (
    ApprovalDecision,
    NotificationApproval,
    NotificationOrigin,
    NotificationWrite,
)
from opentulpa.notifications.service import NotificationService

if TYPE_CHECKING:
    from opentulpa.bootstrap.models import OutboxEvent
    from opentulpa.deep_agent.contracts import AgentRunSnapshot
    from opentulpa.evolution.models import EvolutionEvent
    from opentulpa.specs.models import TriggerSpec


_EVOLUTION_EVENT_TEXT = {
    "build.preparing": "The deployment host is preparing a new OpenTulpa build.",
    "build.switching": "The deployment host is switching to the new OpenTulpa build now.",
    "build.active": "The deployment host confirmed that the new OpenTulpa build is active.",
    "promotion.failed": "The deployment host rejected the new build and kept or restored the previous build.",
    "build.rolled_back": "The deployment host confirmed that the previous build has been restored.",
    "rollback.failed": "OpenTulpa could not restore the requested previous build.",
    "runtime_env.restarting": "OpenTulpa is restarting to apply the runtime environment update. I will report again when it is back online.",
    "runtime_env.updated": "OpenTulpa is back online. The runtime environment update is active.",
    "runtime_env.failed": "The runtime environment update failed; OpenTulpa kept or restored the previous environment.",
}


class TriggerNotificationSink:
    """Deliver scheduled and event-triggered runs through the same owner stream."""

    def __init__(self, notifications: NotificationService) -> None:
        self._notifications = notifications

    async def __call__(
        self,
        *,
        trigger: TriggerSpec,
        snapshot: AgentRunSnapshot,
        mode: Literal["origin", "owner"],
    ) -> None:
        del mode
        pending = tuple(
            NotificationApproval(
                approval_id=approval.id,
                tool_name=approval.tool_name or "action",
                description=approval.description or "Approval required.",
                allowed_decisions=tuple(
                    cast(ApprovalDecision, decision)
                    for decision in approval.allowed_decisions
                    if decision in {"approve", "edit", "reject"}
                ),
            )
            for approval in snapshot.approvals
            if approval.status == "pending"
        )
        kind = (
            "approval.required"
            if snapshot.status == "interrupted"
            else "run.completed"
            if snapshot.status == "completed"
            else "run.failed"
        )
        text = (
            snapshot.final_text.strip()
            or snapshot.error.strip()
            or (
                f"{trigger.name} is waiting for approval."
                if pending
                else f"{trigger.name} {snapshot.status}."
            )
        )
        context = snapshot.context
        self._notifications.publish(
            tenant_id=trigger.tenant_id,
            dedupe_key=(
                f"trigger:{trigger.id}:r{trigger.revision}:"
                f"run:{snapshot.run_id}:{snapshot.status}"
            ),
            notification=NotificationWrite(
                kind=kind,
                text=text,
                status=snapshot.status,
                thread_id=context.thread_id,
                run_id=snapshot.run_id,
                origin=NotificationOrigin(
                    interface=context.origin.interface,
                    source_id=context.origin.source_id,
                    conversation_id=context.origin.conversation_id,
                    message_id=context.origin.message_id,
                    channel=context.channel,
                    correlation_id=context.correlation_id,
                ),
                approvals=pending,
            ),
        )


class EvolutionNotificationSink:
    """Translate evolution progress and terminal events into owner notifications."""

    def __init__(self, notifications: NotificationService) -> None:
        self._notifications = notifications

    async def deliver(self, event: EvolutionEvent) -> None:
        origin = dict(event.origin)
        tenant_id = _optional(origin, "tenant_id")
        if tenant_id is None:
            # Bootstrap/system events have no tenant owner and are intentionally consumed.
            return
        payload = dict(event.payload)
        failed = event.event_type.endswith(".failed")
        text = _evolution_text(event.event_type, payload)
        self._notifications.publish(
            tenant_id=tenant_id,
            dedupe_key=f"evolution:{event.event_key}",
            notification=NotificationWrite(
                kind=f"evolution.{event.event_type}",
                text=text[:50_000],
                status="failed" if failed else str(payload.get("status") or "info"),
                thread_id=_optional(origin, "thread_id"),
                run_id=_optional(origin, "run_id"),
                origin=_evolution_origin(origin),
            ),
        )


class EvolutionRepairNotificationSink:
    """Report terminal automatic-repair outcomes independently of the run stream."""

    def __init__(self, notifications: NotificationService) -> None:
        self._notifications = notifications

    async def __call__(self, snapshot: AgentRunSnapshot) -> None:
        context = snapshot.context
        if context.channel != "evolution" or not context.correlation_id.startswith(
            "evolution-repair:"
        ):
            return
        completed = snapshot.status == "completed"
        detail = str(snapshot.final_text if completed else snapshot.error).strip().rstrip(".")
        text = (
            "The repair agent finished: "
            + (detail or "no further activation result was reported")
            if completed
            else "The repair agent ended without a confirmed result: "
            + (detail or "no successful replacement deployment was confirmed")
        )
        text += ". The deployment host will report any activation outcome separately."
        self._notifications.publish(
            tenant_id=context.tenant_id,
            dedupe_key=f"evolution-repair:{snapshot.run_id}:{snapshot.status}",
            notification=NotificationWrite(
                kind=("evolution.repair.completed" if completed else "evolution.repair.failed"),
                text=text[:50_000],
                status="completed" if completed else "failed",
                thread_id=context.thread_id,
                run_id=snapshot.run_id,
                origin=NotificationOrigin(
                    interface=context.origin.interface,
                    source_id=context.origin.source_id,
                    conversation_id=context.origin.conversation_id,
                    message_id=context.origin.message_id,
                    channel=context.channel,
                    correlation_id=context.correlation_id,
                ),
            ),
        )


class BootstrapNotificationSink:
    """Deliver immutable bootstrap activation and rollback outcomes."""

    def __init__(self, notifications: NotificationService) -> None:
        self._notifications = notifications

    async def deliver(self, event: OutboxEvent) -> None:
        if event.origin is None:
            # Initial-install and recovery events are system-owned, not guessed into a tenant.
            return
        payload: dict[str, Any] = dict(event.payload)
        failed = event.event_type.endswith(".failed")
        text = str(
            payload.get("failure_message")
            or payload.get("status")
            or event.event_type
        ).strip()
        origin = event.origin
        self._notifications.publish(
            tenant_id=origin.tenant_id,
            dedupe_key=f"bootstrap:{event.event_key}",
            notification=NotificationWrite(
                kind=f"bootstrap.{event.event_type}",
                text=text[:50_000],
                status="failed" if failed else str(payload.get("status") or "info"),
                thread_id=origin.thread_id,
                run_id=origin.run_id,
                origin=NotificationOrigin(
                    channel=origin.channel,
                    correlation_id=origin.correlation_id,
                ),
            ),
        )


def _evolution_text(event_type: str, payload: dict[str, Any]) -> str:
    activation_id = str(payload.get("activation_id") or "").strip()
    label = f"Activation {activation_id[:12]}" if activation_id else "The deployment"
    if event_type == "build.preparing":
        return f"{label} is queued and the deployment host is preparing the new build."
    if event_type == "build.switching":
        return f"{label} is switching to the new build now. I will report the result."
    if event_type == "build.active":
        return f"{label} succeeded. OpenTulpa is online with the new build."
    if event_type == "build.rolled_back":
        return f"{label} succeeded. OpenTulpa is online with the restored previous build."
    if event_type == "promotion.failed":
        phase = str(payload.get("failure_phase") or "deployment").strip()
        reason = str(
            payload.get("failure_message")
            or payload.get("error")
            or "the new build did not pass host checks"
        ).strip()
        return (
            f"{label} failed during {phase}: {reason}. "
            "OpenTulpa kept or restored the previous build."
        )
    return str(
        payload.get("failure_message")
        or payload.get("summary")
        or _EVOLUTION_EVENT_TEXT.get(event_type)
        or payload.get("status")
        or event_type
    ).strip()


def _evolution_origin(value: dict[str, Any]) -> NotificationOrigin:
    raw = value.get("origin")
    parsed: dict[str, Any] = {}
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except ValueError:
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded
    return NotificationOrigin(
        interface=_optional(parsed, "interface"),
        source_id=_optional(parsed, "source_id"),
        conversation_id=_optional(parsed, "conversation_id"),
        message_id=_optional(parsed, "message_id"),
        channel=_optional(value, "channel"),
        correlation_id=_optional(value, "correlation_id"),
    )


def _optional(value: dict[str, Any], key: str) -> str | None:
    resolved = str(value.get(key) or "").strip()
    return resolved or None


__all__ = [
    "BootstrapNotificationSink",
    "EvolutionNotificationSink",
    "EvolutionRepairNotificationSink",
    "TriggerNotificationSink",
]
