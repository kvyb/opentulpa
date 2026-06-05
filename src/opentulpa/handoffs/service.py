"""Owner handoff orchestration for intake workflows."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from opentulpa.handoffs.models import (
    HANDOFF_PLAN_REQUESTED,
    HANDOFF_STATUS_FAILED_REPLY,
    HANDOFF_STATUS_RESOLVED,
    HANDOFF_STATUS_RESOLVED_NO_REPLY,
    HandoffDecisionContext,
    HandoffLead,
    HandoffMessage,
    HandoffMessages,
    HandoffOpenRequest,
    HandoffPlan,
    HandoffRecord,
    HandoffTrigger,
)
from opentulpa.handoffs.store import IntakeHandoffStore
from opentulpa.web.events import append_web_event

QueueCallback = Callable[[HandoffRecord], dict[str, Any]]


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _trim_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def _message_payload(message: dict[str, Any]) -> HandoffMessage:
    return HandoffMessage(
        message_id=str(message.get("id", "") or "").strip(),
        direction="inbound" if str(message.get("sender_role", "") or "") == "customer" else "outbound",
        text=_trim_text(message.get("text"), limit=1000),
        created_at=str(message.get("created_time", "") or "").strip(),
        sender_username=str(message.get("sender_username", "") or "").strip(),
        sender_id=str(message.get("sender_id", "") or "").strip(),
    )


def build_handoff_messages(messages: list[dict[str, Any]]) -> HandoffMessages:
    latest: list[HandoffMessage] = []
    previous: list[HandoffMessage] = []
    unmarked_inbound: list[HandoffMessage] = []
    for item in messages:
        if str(item.get("sender_role", "") or "").strip() != "customer":
            continue
        if not str(item.get("text", "") or "").strip():
            continue
        payload = _message_payload(item)
        group = str(item.get("handoff_context_group", "") or "").strip().lower()
        if group == "latest":
            latest.append(payload)
        elif group == "previous":
            previous.append(payload)
        else:
            unmarked_inbound.append(payload)
    if latest:
        return HandoffMessages(latest=latest, previous=previous[-3:])
    if not unmarked_inbound:
        return HandoffMessages()
    return HandoffMessages(latest=unmarked_inbound[-1:], previous=unmarked_inbound[max(0, len(unmarked_inbound) - 4) : -1])


class IntakeHandoffService:
    """Creates, updates, lists, and resumes owner handoffs."""

    def __init__(self, *, store: IntakeHandoffStore) -> None:
        self._store = store
        self._store.init_db()
        self._queue_callback: QueueCallback | None = None

    def set_queue_callback(self, callback: QueueCallback) -> None:
        self._queue_callback = callback

    def open_or_update(
        self,
        *,
        workflow: dict[str, Any],
        plan: HandoffPlan,
        context: HandoffDecisionContext,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        assert plan.kind == HANDOFF_PLAN_REQUESTED
        assert plan.rule is not None
        message_context = build_handoff_messages(messages)
        lead = self._lead_payload(context=context, messages=messages)
        latest_preview = self._latest_preview(message_context, fallback=context.latest_inbound_text)
        record = self._store.open_or_update(
            HandoffOpenRequest(
                handoff_id=plan.handoff_id_hint,
                customer_id=plan.customer_id,
                workflow_id=plan.workflow_id,
                workflow_name=str(workflow.get("name", "") or ""),
                source_channel=plan.source_channel,
                source_provider=plan.source_provider,
                conversation_id=plan.conversation_id,
                trigger=HandoffTrigger(
                    rule_id=plan.rule.id,
                    rule_label=plan.rule.label,
                    reason=plan.trigger_reason,
                    owner_prompt=plan.owner_prompt,
                    customer_wait_reply=plan.customer_wait_reply,
                ),
                lead=lead,
                messages=message_context,
                latest_inbound_message_id=plan.latest_inbound_message_id,
                latest_customer_message_preview=latest_preview,
            )
        )
        self._emit_event(record)
        return record.to_api_dict()

    @staticmethod
    def _lead_payload(
        *,
        context: HandoffDecisionContext,
        messages: list[dict[str, Any]],
    ) -> HandoffLead:
        latest_customer = next(
            (
                item
                for item in reversed(messages)
                if str(item.get("sender_role", "") or "").strip() == "customer"
            ),
            {},
        )
        username = str(latest_customer.get("sender_username") or context.sender_username or "").strip()
        platform_user_id = str(latest_customer.get("sender_id") or context.sender_id or "").strip()
        return HandoffLead(username=username, display_name=username, platform_user_id=platform_user_id)

    @staticmethod
    def _latest_preview(messages: HandoffMessages, *, fallback: str) -> str:
        if messages.latest:
            return _trim_text(messages.latest[-1].text, limit=500)
        return _trim_text(fallback, limit=500)

    def list_handoffs(self, *, customer_id: str, status: str = "", limit: int = 50) -> list[dict[str, Any]]:
        return [record.to_api_dict() for record in self._store.list_handoffs(customer_id=customer_id, status=status, limit=limit)]

    def get_handoff(self, *, customer_id: str, handoff_id: str) -> dict[str, Any] | None:
        record = self._store.get(customer_id=customer_id, handoff_id=handoff_id)
        return record.to_api_dict() if record is not None else None

    async def respond(
        self,
        *,
        customer_id: str,
        handoff_id: str,
        owner_feedback: str,
    ) -> dict[str, Any]:
        safe_feedback = _trim_text(owner_feedback, limit=4000)
        if not safe_feedback:
            return {"ok": False, "status": "invalid", "detail": "owner_feedback is required"}
        accepted = self._store.accept_owner_response(
            customer_id=customer_id,
            handoff_id=handoff_id,
            owner_feedback=safe_feedback,
        )
        if accepted is None:
            return {"ok": False, "status": "conflict", "detail": "handoff is not awaiting owner"}
        self._emit_event(accepted)
        queued = self._queue_response(accepted)
        handoff = _safe_dict(queued.get("handoff")) or accepted.to_api_dict()
        if not bool(queued.get("ok", False)):
            return {
                "ok": False,
                "status": "queue_failed",
                "detail": str(queued.get("detail") or "handoff resume queue failed"),
                "handoff": handoff,
            }
        return {"ok": True, "queued": bool(queued.get("queued", False)), "handoff": handoff}

    def _queue_response(self, handoff: HandoffRecord) -> dict[str, Any]:
        callback = self._queue_callback
        if callback is None:
            failed = self.mark_failed_reply(handoff, "handoff resume queue callback is unavailable")
            return {
                "ok": False,
                "queued": False,
                "detail": "handoff resume queue callback is unavailable",
                "handoff": failed.to_api_dict() if failed is not None else handoff.to_api_dict(),
            }
        return callback(handoff)

    def mark_resuming(self, handoff: HandoffRecord) -> HandoffRecord | None:
        updated = self._store.mark_resuming(
            customer_id=handoff.customer_id,
            handoff_id=handoff.handoff_id,
        )
        if updated is not None:
            self._emit_event(updated)
        return updated

    def mark_resolved(self, handoff: HandoffRecord, *, no_reply: bool) -> HandoffRecord | None:
        updated = self._store.mark_resolved(
            customer_id=handoff.customer_id,
            handoff_id=handoff.handoff_id,
            no_reply=no_reply,
        )
        if updated is not None:
            assert updated.status in {HANDOFF_STATUS_RESOLVED, HANDOFF_STATUS_RESOLVED_NO_REPLY}
            self._emit_event(updated)
        return updated

    def mark_failed_reply(self, handoff: HandoffRecord, reason: str) -> HandoffRecord | None:
        updated = self._store.mark_failed_reply(
            customer_id=handoff.customer_id,
            handoff_id=handoff.handoff_id,
            failure_reason=_trim_text(reason, limit=1000),
        )
        if updated is not None:
            assert updated.status == HANDOFF_STATUS_FAILED_REPLY
            self._emit_event(updated)
        return updated

    def get_handoff_record(self, *, customer_id: str, handoff_id: str) -> HandoffRecord | None:
        return self._store.get(customer_id=customer_id, handoff_id=handoff_id)

    def _emit_event(self, handoff: HandoffRecord) -> None:
        append_web_event(
            customer_id=handoff.customer_id,
            thread_id=f"handoff:{handoff.handoff_id}",
            source="intake_workflow",
            kind="handoff.updated",
            text=handoff.trigger.owner_prompt,
            metadata_json=json.dumps(
                {
                    "handoff_id": handoff.handoff_id,
                    "status": handoff.status,
                    "workflow_id": handoff.workflow_id,
                    "conversation_id": handoff.conversation_id,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
