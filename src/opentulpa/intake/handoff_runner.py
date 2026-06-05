"""Intake owner handoff orchestration."""

from __future__ import annotations

from typing import Any, Protocol

from opentulpa.handoffs import (
    OWNER_HANDOFF_RESUME_EVENT_TYPE,
    HandoffDecisionContext,
    HandoffRecord,
    IntakeHandoffService,
    plan_handoff_request,
)
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN


class IntakeHandoffWorkflowService(Protocol):
    """Workflow service surface needed by owner handoff runtime."""

    def get_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any] | None: ...

    def _queue_pending_run(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        event_type: str,
        owner_chat_id: str = "",
        delay_seconds: float = 0.0,
        last_inbound_message_id: str = "",
        handoff_id: str = "",
        owner_feedback: str = "",
    ) -> dict[str, Any]: ...

    def _normalize_conversation_messages(
        self,
        *,
        workflow: dict[str, Any],
        conversation: dict[str, Any],
        recipient_id: str | None,
    ) -> list[dict[str, Any]]: ...

    def _get_cursor(self, *, workflow_id: str, conversation_id: str) -> dict[str, Any]: ...

    def _load_source_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]: ...

    def _enrich_conversation_summary(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> dict[str, Any]: ...

    def _get_active_booking(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None: ...

    def _get_recent_completed_booking(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        conversation_id: str,
    ) -> dict[str, Any] | None: ...

    async def _decide_workflow_action(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        owner_handoff_feedback: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str | None]: ...

    def _uses_latest_inbound_stale_guard(self, *, workflow: dict[str, Any], event_type: str, force: bool) -> bool: ...

    async def _apply_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        decision: dict[str, Any],
        stale_guard: bool,
    ) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]: ...


class IntakeHandoffRuntime:
    """Coordinates intake handoff persistence with workflow runtime execution."""

    def __init__(self, *, service: IntakeHandoffWorkflowService, handoff_service: IntakeHandoffService) -> None:
        self._service = service
        self._handoffs = handoff_service

    def queue_owner_handoff_resume(self, handoff: HandoffRecord) -> dict[str, Any]:
        assert handoff.handoff_id
        assert handoff.owner_feedback.strip()
        workflow = self._service.get_workflow(customer_id=handoff.customer_id, workflow_id=handoff.workflow_id)
        if workflow is None:
            failed = self._handoffs.mark_failed_reply(handoff, "handoff workflow not found")
            return {
                "ok": False,
                "queued": False,
                "detail": "handoff workflow not found",
                "handoff": failed.to_api_dict() if failed is not None else handoff.to_api_dict(),
            }
        return self._service._queue_pending_run(
            workflow=workflow,
            conversation_id=handoff.conversation_id,
            event_type=OWNER_HANDOFF_RESUME_EVENT_TYPE,
            delay_seconds=0,
            handoff_id=handoff.handoff_id,
            owner_feedback=handoff.owner_feedback,
        )

    def open_or_update_owner_handoff(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        decision: dict[str, Any],
    ) -> dict[str, Any] | None:
        context = HandoffDecisionContext.from_runtime_inputs(
            workflow=workflow,
            conversation_summary=conversation_summary,
        )
        plan = plan_handoff_request(workflow=workflow, decision=decision, context=context)
        if plan.kind != "handoff_requested":
            return None
        messages = self._handoff_context_messages(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
        )
        return self._handoffs.open_or_update(
            workflow=workflow,
            plan=plan,
            context=context,
            messages=messages,
        )

    def _handoff_context_messages(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
    ) -> list[dict[str, Any]]:
        recent_messages = self._service._normalize_conversation_messages(
            workflow=workflow,
            conversation=conversation,
            recipient_id=str(conversation_summary.get("recipient_id", "") or "").strip() or None,
        )
        latest_index = self._handoff_latest_message_index(
            recent_messages=recent_messages,
            latest_inbound_message_id=str(conversation_summary.get("latest_inbound_message_id", "") or "").strip(),
        )
        if latest_index is None:
            return recent_messages
        start_index = latest_index
        cursor = self._service._get_cursor(
            workflow_id=str(workflow.get("workflow_id", "") or "").strip(),
            conversation_id=str(conversation_summary.get("conversation_id", "") or "").strip(),
        )
        last_seen_id = str(cursor.get("last_seen_inbound_message_id", "") or "").strip()
        if last_seen_id:
            for index, item in enumerate(recent_messages[:latest_index]):
                if str(item.get("id", "") or "").strip() == last_seen_id:
                    start_index = index + 1
                    break
        previous = [
            {**item, "handoff_context_group": "previous"}
            for item in recent_messages[:start_index]
            if str(item.get("sender_role", "") or "").strip() == "customer"
        ][-3:]
        latest = [
            {**item, "handoff_context_group": "latest"}
            for item in recent_messages[start_index : latest_index + 1]
            if str(item.get("sender_role", "") or "").strip() == "customer"
        ]
        if not latest:
            latest = [{**recent_messages[latest_index], "handoff_context_group": "latest"}]
        return previous + latest

    @staticmethod
    def _handoff_latest_message_index(
        *,
        recent_messages: list[dict[str, Any]],
        latest_inbound_message_id: str,
    ) -> int | None:
        for index in range(len(recent_messages) - 1, -1, -1):
            item = recent_messages[index]
            if str(item.get("sender_role", "") or "").strip() != "customer":
                continue
            if latest_inbound_message_id and str(item.get("id", "") or "").strip() != latest_inbound_message_id:
                continue
            return index
        for index in range(len(recent_messages) - 1, -1, -1):
            if str(recent_messages[index].get("sender_role", "") or "").strip() == "customer":
                return index
        return None

    async def run_owner_handoff_resume_row(self, row: dict[str, Any]) -> dict[str, Any]:
        customer_id = str(row.get("customer_id", "") or "").strip()
        handoff_id = str(row.get("handoff_id", "") or "").strip()
        owner_feedback = str(row.get("owner_feedback", "") or "").strip()
        if not customer_id or not handoff_id:
            return {"ok": False, "summary": "owner handoff resume requires customer_id and handoff_id"}
        handoff = self._handoffs.get_handoff_record(customer_id=customer_id, handoff_id=handoff_id)
        if handoff is None:
            return {"ok": False, "summary": "owner handoff not found"}
        owner_feedback = owner_feedback or handoff.owner_feedback
        if not owner_feedback:
            failed = self._handoffs.mark_failed_reply(handoff, "owner feedback missing")
            payload = failed.to_api_dict() if failed is not None else handoff.to_api_dict()
            return {"ok": False, "summary": "owner feedback missing", "handoff": payload}
        running = self._handoffs.mark_resuming(handoff)
        if running is None:
            return {"ok": True, "summary": NO_NOTIFY_TOKEN, "reason": "handoff_no_longer_resumable"}
        result = await self.resume_owner_handoff(running.to_api_dict(), owner_feedback)
        if not bool(result.get("ok", False)):
            reason = str(result.get("detail") or result.get("summary") or "handoff resume failed")
            failed = self._handoffs.mark_failed_reply(running, reason)
            payload = failed.to_api_dict() if failed is not None else running.to_api_dict()
            return {"ok": False, "summary": reason, "handoff": payload}
        resolved = self._handoffs.mark_resolved(
            running,
            no_reply=not bool(result.get("replied", False)),
        )
        payload = resolved.to_api_dict() if resolved is not None else running.to_api_dict()
        return {"ok": True, "summary": NO_NOTIFY_TOKEN, "handoff": payload}

    async def resume_owner_handoff(
        self,
        handoff: dict[str, Any],
        owner_feedback: str,
    ) -> dict[str, Any]:
        workflow = self._service.get_workflow(
            customer_id=str(handoff.get("customer_id", "") or ""),
            workflow_id=str(handoff.get("workflow_id", "") or ""),
        )
        if workflow is None:
            return {"ok": False, "detail": "handoff workflow not found"}
        conversation_id = str(handoff.get("conversation_id", "") or "").strip()
        loaded = self._service._load_source_conversation(
            workflow=workflow,
            conversation_id=conversation_id,
        )
        if len(loaded) != 3:
            return {"ok": False, "detail": "handoff source load failed"}
        summary, conversation, error = loaded
        if error:
            return {"ok": False, "detail": str(error)}
        return await self._resume_loaded_handoff(
            workflow=workflow,
            handoff=handoff,
            owner_feedback=owner_feedback,
            conversation_summary=self._service._enrich_conversation_summary(
                workflow=workflow,
                conversation_summary=summary,
            ),
            conversation=conversation,
        )

    async def _resume_loaded_handoff(
        self,
        *,
        workflow: dict[str, Any],
        handoff: dict[str, Any],
        owner_feedback: str,
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
    ) -> dict[str, Any]:
        conversation_id = str(handoff["conversation_id"])
        active_booking = self._service._get_active_booking(
            customer_id=str(workflow["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        recent_completed = self._service._get_recent_completed_booking(
            customer_id=str(workflow["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        decision, error = await self._service._decide_workflow_action(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed,
            owner_handoff_feedback={
                "handoff_id": str(handoff["handoff_id"]),
                "owner_feedback": owner_feedback,
                "owner_feedback_visible_to_customer": False,
            },
        )
        if error:
            return {"ok": False, "detail": error}
        applied, apply_error, _ = await self._service._apply_decision(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed,
            decision=decision,
            stale_guard=self._service._uses_latest_inbound_stale_guard(
                workflow=workflow,
                event_type=OWNER_HANDOFF_RESUME_EVENT_TYPE,
                force=False,
            ),
        )
        if apply_error:
            return {"ok": False, "detail": apply_error}
        return {
            "ok": True,
            "applied": applied,
            "replied": bool(applied.get("replied", False))
            or str(decision.get("reply_action", "")).strip().lower() == "send_reply",
        }
