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
    "build.preparing": "Preparing and testing a new OpenTulpa build.",
    "build.switching": "Switching to the new OpenTulpa build now.",
    "promotion.active": "The new OpenTulpa build is active.",
    "promotion.failed": "The new build failed; OpenTulpa kept or restored the previous build.",
    "rollback.active": "The previous OpenTulpa build has been restored.",
    "rollback.failed": "OpenTulpa could not restore the requested previous build.",
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
        text = str(
            payload.get("failure_message")
            or payload.get("summary")
            or _EVOLUTION_EVENT_TEXT.get(event.event_type)
            or payload.get("status")
            or event.event_type
        ).strip()
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
    "TriggerNotificationSink",
]
