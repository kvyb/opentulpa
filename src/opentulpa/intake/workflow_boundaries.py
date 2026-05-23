"""Typed boundaries for intake workflow execution."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

DECISION_BOOKING_ACTIONS = {
    "ignore",
    "update_active",
    "edit_recent_completed",
    "create_new_booking",
}
DECISION_SINK_ACTIONS = {"none", "upsert_partial"}


@dataclass(frozen=True)
class DecisionActions:
    booking_action: str
    ready_to_save: bool
    reply_action: str
    reply_text: str
    sink_action: str
    sink_payload: dict[str, Any]

    @classmethod
    def from_decision(cls, decision: dict[str, Any]) -> DecisionActions:
        raw_sink_payload = decision.get("sink_payload")
        sink_payload: dict[str, Any] = dict(raw_sink_payload) if isinstance(raw_sink_payload, dict) else {}
        actions = cls(
            booking_action=str(decision.get("booking_action", "ignore") or "ignore").strip().lower(),
            ready_to_save=bool(decision.get("ready_to_save")),
            reply_action=str(decision.get("reply_action", "none") or "none").strip().lower(),
            reply_text=str(decision.get("reply_text", "") or "").strip(),
            sink_action=str(decision.get("sink_action", "none") or "none").strip().lower(),
            sink_payload=sink_payload,
        )
        assert actions.booking_action == actions.booking_action.strip().lower()
        assert actions.reply_action == actions.reply_action.strip().lower()
        assert actions.sink_action == actions.sink_action.strip().lower()
        return actions

    def validation_error(self) -> str | None:
        assert self.booking_action == self.booking_action.strip().lower()
        assert self.sink_action == self.sink_action.strip().lower()
        if self.sink_action not in DECISION_SINK_ACTIONS:
            return f"unsupported sink_action={self.sink_action}"
        if self.booking_action not in DECISION_BOOKING_ACTIONS:
            return f"unsupported booking_action={self.booking_action}"
        if self.booking_action == "ignore" and self.sink_action != "none":
            return "sink_action requires an active booking action"
        return None


@dataclass(frozen=True)
class ConversationCursorSignals:
    conversation_id: str
    latest_inbound_message_id: str
    latest_inbound_message_time: str
    conversation_updated_time: str
    latest_outbound_message_id: str

    @classmethod
    def from_summary(cls, conversation_summary: dict[str, Any]) -> ConversationCursorSignals:
        signals = cls(
            conversation_id=str(conversation_summary.get("conversation_id", "") or "").strip(),
            latest_inbound_message_id=str(
                conversation_summary.get("latest_inbound_message_id", "") or ""
            ).strip(),
            latest_inbound_message_time=str(
                conversation_summary.get("latest_inbound_message_created_time", "") or ""
            ).strip(),
            conversation_updated_time=str(
                conversation_summary.get("conversation_updated_time", "") or ""
            ).strip(),
            latest_outbound_message_id=str(
                conversation_summary.get("latest_outbound_message_id", "") or ""
            ).strip(),
        )
        assert signals.conversation_id == signals.conversation_id.strip()
        assert signals.latest_inbound_message_id == signals.latest_inbound_message_id.strip()
        return signals


@dataclass
class WorkflowRunAccumulator:
    processed: int = 0
    matched: int = 0
    saved_notifications: list[str] = dataclass_field(default_factory=list)
    errors: list[str] = dataclass_field(default_factory=list)
    result_items: list[dict[str, Any]] = dataclass_field(default_factory=list)

    def build_response(
        self,
        *,
        workflow: dict[str, Any],
        workflow_id: str,
        event_type: str,
        source_warnings: list[dict[str, Any]],
        empty_summary_token: str,
    ) -> dict[str, Any]:
        assert self.processed >= 0
        assert self.matched >= 0
        if self.errors:
            summary = (
                f"Workflow {workflow['name']} hit errors: " + " | ".join(self.errors[:3])
            )[:2000]
        elif self.saved_notifications:
            summary = "\n".join(self.saved_notifications[:3])[:2000]
        else:
            summary = empty_summary_token
        return {
            "ok": not self.errors,
            "workflow_id": workflow_id,
            "event_type": event_type,
            "processed_conversations": self.processed,
            "matched_conversations": self.matched,
            "results": self.result_items,
            "errors": self.errors,
            "source_warnings": source_warnings,
            "summary": summary,
        }


@dataclass(frozen=True)
class BookingTargetResolution:
    booking_action: str
    booking: dict[str, Any] | None
