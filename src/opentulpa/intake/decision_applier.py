"""Decision application for intake workflows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from opentulpa.intake.reply_policy import build_missing_field_follow_up_reply
from opentulpa.intake.workflow_boundaries import BookingTargetResolution, DecisionActions
from opentulpa.intake.workflow_runtime import (
    DEFAULT_EDIT_WINDOW as _DEFAULT_EDIT_WINDOW,
)
from opentulpa.intake.workflow_runtime import (
    required_field_is_present as _required_field_is_present,
)
from opentulpa.intake.workflow_runtime import (
    safe_dict as _safe_dict,
)
from opentulpa.intake.workflow_runtime import (
    unique_string_list as _unique_string_list,
)

ApplyResult = tuple[dict[str, Any], str | None, dict[str, Any] | None]


class DecisionApplyService(Protocol):
    def _emit_observability(
        self,
        *,
        event: str,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        **extra: Any,
    ) -> None: ...

    def _resolve_booking_target(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        booking_action: str,
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
    ) -> BookingTargetResolution: ...

    def _upsert_booking(self, booking: dict[str, Any]) -> None: ...

    def _build_recovery_feedback(
        self,
        *,
        phase: str,
        error: str,
        decision: dict[str, Any],
    ) -> dict[str, Any]: ...

    def _build_saved_summary(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> str: ...

    def _write_to_sink(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
        payload: dict[str, Any],
        sink_arguments: dict[str, Any] | None = None,
        record_status: str | None = None,
    ) -> tuple[dict[str, Any], str | None]: ...

    async def _send_intake_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None: ...

    def _emit_apply_decision_validation_error(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        error: str,
        booking_action: str,
        sink_action: str,
    ) -> None: ...

    def _build_cancellation_confirmation_reply(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> str: ...

    def _should_enforce_completion_reply(self, *, workflow: dict[str, Any]) -> bool: ...

    def _build_completion_confirmation_reply(
        self,
        *,
        workflow: dict[str, Any],
        booking: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> str: ...

    def _missing_field_for_follow_up(
        self,
        *,
        workflow: dict[str, Any],
        decision: dict[str, Any],
        active_booking: dict[str, Any],
    ) -> str: ...

    def _requeue_if_conversation_stale(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        matched: bool,
    ) -> dict[str, Any] | None: ...


@dataclass
class ApplyRequest:
    workflow: dict[str, Any]
    conversation_summary: dict[str, Any]
    conversation: dict[str, Any]
    active_booking: dict[str, Any] | None
    recent_completed_booking: dict[str, Any] | None
    decision: dict[str, Any]
    stale_guard: bool
    actions: DecisionActions


@dataclass
class ApplyState:
    target_booking: dict[str, Any]
    sink_ref: dict[str, Any]
    sink_status: str
    sink_arguments: dict[str, Any]
    saved_summary: str
    reply_action: str
    reply_text: str


class DecisionApplier:
    """Applies model decisions to bookings, sinks, and replies."""

    def __init__(
        self,
        service: DecisionApplyService,
        *,
        utc_now: Callable[[], datetime],
        utc_now_iso: Callable[[], str],
    ) -> None:
        self._service = service
        self._utc_now = utc_now
        self._utc_now_iso = utc_now_iso

    async def apply_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        decision: dict[str, Any],
        stale_guard: bool = False,
    ) -> ApplyResult:
        actions = DecisionActions.from_decision(decision)
        request = ApplyRequest(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            decision=decision,
            stale_guard=stale_guard,
            actions=actions,
        )
        self._emit_apply_start(request)
        validation_error = actions.validation_error()
        if validation_error is not None:
            return self._decision_validation_error(request, validation_error)
        if actions.booking_action == "ignore":
            return await self._apply_ignore(request)

        state_result = self._build_apply_state(request)
        if isinstance(state_result, tuple):
            return state_result
        state = state_result
        if actions.ready_to_save:
            early_result = self._complete_booking(request, state)
        else:
            early_result = self._update_active_booking(request, state)
        if early_result is not None:
            return early_result
        if state.reply_action == "send_reply":
            reply_result = await self._send_reply(request, state)
            if reply_result is not None:
                return reply_result
        self._emit_apply_ok(request, state)
        return self._applied_result(request, state), None, None

    async def _apply_ignore(self, request: ApplyRequest) -> ApplyResult:
        if request.actions.reply_action == "send_reply":
            stale_result = self._stale_result(request)
            if stale_result is not None:
                return stale_result, None, None
            reply_error = await self._send_reply_text(
                request=request,
                booking_id="",
                reply_text=request.actions.reply_text,
            )
            if reply_error is not None:
                return self._reply_error(request, booking_id="", error=reply_error)
        self._service._emit_observability(
            event="intake.apply.ok",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            status="ignored",
            booking_action=request.actions.booking_action,
            reply_action=request.actions.reply_action,
            ready_to_save=request.actions.ready_to_save,
        )
        result = {
            "conversation_id": str(request.conversation_summary.get("conversation_id", "") or ""),
            "matched": True,
            "status": "ignored",
        }
        if request.actions.reply_action == "send_reply":
            result["replied"] = True
        return result, None, None

    def _build_apply_state(
        self,
        request: ApplyRequest,
    ) -> ApplyState | ApplyResult:
        target = self._service._resolve_booking_target(
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_action=request.actions.booking_action,
            active_booking=request.active_booking,
            recent_completed_booking=request.recent_completed_booking,
        )
        if target.booking is None:
            return self._decision_validation_error(
                request,
                "workflow decision did not resolve a booking target",
                booking_action=request.actions.booking_action,
            )
        target_booking = target.booking
        extracted_fields = dict(_safe_dict(target_booking.get("extracted_fields")))
        extracted_fields.update(_safe_dict(request.decision.get("extracted_fields")))
        if not request.actions.ready_to_save and request.actions.sink_action == "upsert_partial":
            extracted_fields.update(request.actions.sink_payload)
        target_booking["extracted_fields"] = extracted_fields
        target_booking["conversation_summary"] = self._conversation_summary_text(request)
        target_booking["last_customer_message_at"] = str(
            request.conversation_summary.get("latest_inbound_message_created_time", "") or ""
        ).strip()
        return ApplyState(
            target_booking=target_booking,
            sink_ref=dict(_safe_dict(target_booking.get("sink_record_ref"))),
            sink_status=str(target_booking.get("sink_write_status", "pending") or "pending").strip(),
            sink_arguments=dict(_safe_dict(request.decision.get("sink_arguments"))),
            saved_summary="",
            reply_action=request.actions.reply_action,
            reply_text=request.actions.reply_text,
        )

    def _complete_booking(
        self,
        request: ApplyRequest,
        state: ApplyState,
    ) -> ApplyResult | None:
        save_payload = dict(_safe_dict(state.target_booking.get("extracted_fields")))
        save_payload.update(_safe_dict(request.decision.get("save_payload")))
        missing_result = self._missing_required_result(request, state, save_payload)
        if missing_result is not None:
            return missing_result
        stale_result = self._stale_result(request)
        if stale_result is not None:
            return stale_result, None, None
        if self._should_skip_cancel_sink(request, state, save_payload):
            self._persist_cancel_without_sink(request, state, save_payload)
            return None
        return self._write_completed_sink(request, state, save_payload)

    def _update_active_booking(
        self,
        request: ApplyRequest,
        state: ApplyState,
    ) -> ApplyResult | None:
        state.target_booking["status"] = (
            "cancelled" if state.reply_action == "mark_cancelled" else "active"
        )
        stale_result = self._stale_result(request)
        if stale_result is not None:
            return stale_result, None, None
        state.target_booking["sink_write_status"] = state.sink_status
        state.target_booking["sink_record_ref"] = state.sink_ref
        state.target_booking["updated_at"] = self._utc_now_iso()
        self._normalize_active_missing_field_reply(request, state)
        if request.actions.sink_action == "upsert_partial" and request.actions.sink_payload:
            partial_result = self._write_partial_sink(request, state)
            if partial_result is not None:
                return partial_result
        self._service._upsert_booking(state.target_booking)
        return None

    def _missing_required_result(
        self,
        request: ApplyRequest,
        state: ApplyState,
        save_payload: dict[str, Any],
    ) -> ApplyResult | None:
        save_status = str(save_payload.get("status", "") or "").strip().lower()
        is_cancellation = state.reply_action == "mark_cancelled" or save_status == "cancelled"
        missing = [
            field
            for field in _unique_string_list(request.workflow.get("required_fields"))
            if not _required_field_is_present(save_payload, field)
        ]
        if not missing or is_cancellation:
            return None
        error = "decision marked ready_to_save but required fields are missing: " + ", ".join(missing)
        self._service._emit_observability(
            event="intake.apply.error",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            phase="decision_validation",
            error=error,
            booking_id=self._booking_id(state),
            missing_fields=missing,
        )
        feedback = self._service._build_recovery_feedback(
            phase="decision_validation",
            error=f"ready_to_save missing required fields: {', '.join(missing)}",
            decision=request.decision,
        )
        return {}, error, feedback

    def _persist_cancel_without_sink(
        self,
        request: ApplyRequest,
        state: ApplyState,
        save_payload: dict[str, Any],
    ) -> None:
        self._service._emit_observability(
            event="intake.sink_write.skipped",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
            reason="cancellation_not_previously_persisted",
            payload=save_payload,
        )
        now = self._utc_now()
        state.target_booking.update(
            {
                "status": "cancelled",
                "completed_at": now.isoformat(),
                "edit_window_until": (now + _DEFAULT_EDIT_WINDOW).isoformat(),
                "sink_write_status": "not_required",
                "sink_record_ref": state.sink_ref,
                "extracted_fields": save_payload,
                "updated_at": now.isoformat(),
            }
        )
        self._service._upsert_booking(state.target_booking)
        state.saved_summary = self._service._build_saved_summary(
            workflow=request.workflow,
            booking=state.target_booking,
            conversation_summary=request.conversation_summary,
        )
        self._normalize_cancellation_reply(request, state)

    def _write_completed_sink(
        self,
        request: ApplyRequest,
        state: ApplyState,
        save_payload: dict[str, Any],
    ) -> ApplyResult | None:
        self._service._emit_observability(
            event="intake.sink_write.start",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
            payload=save_payload,
        )
        sink_result, sink_error = self._service._write_to_sink(
            workflow=request.workflow,
            booking=state.target_booking,
            conversation_summary=request.conversation_summary,
            payload=save_payload,
            sink_arguments=state.sink_arguments,
            record_status=self._completed_record_status(state),
        )
        if sink_error is not None:
            return self._sink_error(request, state, sink_error)
        self._persist_completed_sink(request, state, save_payload, sink_result)
        self._normalize_completion_reply(request, state)
        return None

    def _persist_completed_sink(
        self,
        request: ApplyRequest,
        state: ApplyState,
        save_payload: dict[str, Any],
        sink_result: Any,
    ) -> None:
        state.sink_status = "succeeded"
        state.sink_ref = _safe_dict(sink_result)
        now = self._utc_now()
        state.target_booking.update(
            {
                "status": "cancelled" if state.reply_action == "mark_cancelled" else "completed",
                "completed_at": now.isoformat(),
                "edit_window_until": (now + _DEFAULT_EDIT_WINDOW).isoformat(),
                "sink_write_status": state.sink_status,
                "sink_record_ref": state.sink_ref,
                "extracted_fields": save_payload,
                "updated_at": now.isoformat(),
            }
        )
        self._service._upsert_booking(state.target_booking)
        self._service._emit_observability(
            event="intake.sink_write.ok",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
            sink_result=state.sink_ref,
        )
        state.saved_summary = self._service._build_saved_summary(
            workflow=request.workflow,
            booking=state.target_booking,
            conversation_summary=request.conversation_summary,
        )

    def _write_partial_sink(
        self,
        request: ApplyRequest,
        state: ApplyState,
    ) -> ApplyResult | None:
        sink_type = str(request.workflow.get("sink_type", "") or "").strip().lower()
        if sink_type not in {"google_sheets_composio", "generic_composio_write"}:
            error = "sink_action=upsert_partial requires a Composio upsert sink"
            return self._decision_validation_error(request, error, booking_id=self._booking_id(state), sink_type=sink_type)
        self._service._emit_observability(
            event="intake.sink_write.partial_start",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
            payload=request.actions.sink_payload,
        )
        sink_result, sink_error = self._service._write_to_sink(
            workflow=request.workflow,
            booking=state.target_booking,
            conversation_summary=request.conversation_summary,
            payload=request.actions.sink_payload,
            sink_arguments=state.sink_arguments,
        )
        if sink_error is not None:
            state.target_booking["sink_write_status"] = "failed"
            self._service._upsert_booking(state.target_booking)
            self._service._emit_observability(
                event="intake.sink_write.partial_error",
                workflow=request.workflow,
                conversation_summary=request.conversation_summary,
                booking_id=self._booking_id(state),
                sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
                error=sink_error,
            )
            return {}, sink_error, self._service._build_recovery_feedback(
                phase="sink_execution",
                error=sink_error,
                decision=request.decision,
            )
        state.sink_status = "partial_succeeded"
        state.sink_ref = _safe_dict(sink_result)
        state.target_booking["sink_write_status"] = state.sink_status
        state.target_booking["sink_record_ref"] = state.sink_ref
        self._service._emit_observability(
            event="intake.sink_write.partial_ok",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
            sink_result=state.sink_ref,
        )
        return None

    async def _send_reply(
        self,
        request: ApplyRequest,
        state: ApplyState,
    ) -> ApplyResult | None:
        stale_result = self._stale_result(request)
        if stale_result is not None:
            return stale_result, None, None
        reply_error = await self._send_reply_text(
            request=request,
            booking_id=self._booking_id(state),
            reply_text=state.reply_text,
        )
        if reply_error is None:
            return None
        return self._reply_error(request, booking_id=self._booking_id(state), error=reply_error)

    async def _send_reply_text(
        self,
        *,
        request: ApplyRequest,
        booking_id: str,
        reply_text: str,
    ) -> str | None:
        self._service._emit_observability(
            event="intake.reply.start",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=booking_id,
            reply_text=reply_text,
        )
        reply_error = await self._service._send_intake_reply(
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            reply_text=reply_text,
        )
        if reply_error is None:
            self._service._emit_observability(
                event="intake.reply.ok",
                workflow=request.workflow,
                conversation_summary=request.conversation_summary,
                booking_id=booking_id,
            )
        return reply_error

    def _sink_error(
        self,
        request: ApplyRequest,
        state: ApplyState,
        sink_error: str,
    ) -> ApplyResult:
        state.target_booking["status"] = "active"
        state.target_booking["sink_write_status"] = "failed"
        state.target_booking["sink_record_ref"] = state.sink_ref
        self._service._upsert_booking(state.target_booking)
        self._service._emit_observability(
            event="intake.sink_write.error",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            sink_type=str(request.workflow.get("sink_type", "") or "").strip(),
            error=sink_error,
        )
        self._service._emit_observability(
            event="intake.apply.error",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            phase="sink_execution",
            error=sink_error,
            booking_id=self._booking_id(state),
        )
        feedback = self._service._build_recovery_feedback(
            phase="sink_execution",
            error=sink_error,
            decision=request.decision,
        )
        return {}, sink_error, feedback

    def _reply_error(
        self,
        request: ApplyRequest,
        *,
        booking_id: str,
        error: str,
    ) -> ApplyResult:
        self._service._emit_observability(
            event="intake.reply.error",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=booking_id,
            error=error,
        )
        self._service._emit_observability(
            event="intake.apply.error",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            phase="reply_execution",
            error=error,
            booking_id=booking_id,
        )
        feedback = self._service._build_recovery_feedback(
            phase="reply_execution",
            error=error,
            decision=request.decision,
        )
        return {}, error, feedback

    def _decision_validation_error(
        self,
        request: ApplyRequest,
        error: str,
        **extra: Any,
    ) -> ApplyResult:
        booking_action = extra.pop("booking_action", request.actions.booking_action)
        if extra:
            self._service._emit_observability(
                event="intake.apply.error",
                workflow=request.workflow,
                conversation_summary=request.conversation_summary,
                phase="decision_validation",
                error=error,
                booking_action=booking_action,
                sink_action=request.actions.sink_action,
                **extra,
            )
        else:
            self._service._emit_apply_decision_validation_error(
                workflow=request.workflow,
                conversation_summary=request.conversation_summary,
                error=error,
                booking_action=booking_action,
                sink_action=request.actions.sink_action,
            )
        feedback = self._service._build_recovery_feedback(
            phase="decision_validation",
            error=error,
            decision=request.decision,
        )
        return {}, error, feedback

    def _normalize_cancellation_reply(self, request: ApplyRequest, state: ApplyState) -> None:
        state.reply_action = "send_reply"
        if state.reply_text:
            return
        state.reply_text = self._service._build_cancellation_confirmation_reply(
            workflow=request.workflow,
            booking=state.target_booking,
            conversation_summary=request.conversation_summary,
        )
        self._service._emit_observability(
            event="intake.reply.normalized_cancellation_confirmation",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            reply_text=state.reply_text,
        )

    def _normalize_completion_reply(self, request: ApplyRequest, state: ApplyState) -> None:
        if state.reply_action == "mark_cancelled":
            self._normalize_cancellation_reply(request, state)
            return
        if not self._service._should_enforce_completion_reply(workflow=request.workflow):
            return
        if state.reply_action == "send_reply" and state.reply_text:
            return
        state.reply_action = "send_reply"
        state.reply_text = self._service._build_completion_confirmation_reply(
            workflow=request.workflow,
            booking=state.target_booking,
            conversation_summary=request.conversation_summary,
        )
        self._service._emit_observability(
            event="intake.reply.normalized_completion_confirmation",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            reply_text=state.reply_text,
        )

    @staticmethod
    def _completed_record_status(state: ApplyState) -> str:
        return "cancelled" if state.reply_action == "mark_cancelled" else "completed"

    def _normalize_active_missing_field_reply(
        self,
        request: ApplyRequest,
        state: ApplyState,
    ) -> None:
        if not self._service._should_enforce_completion_reply(workflow=request.workflow):
            return
        if state.reply_action == "send_reply" and state.reply_text:
            return
        missing_field = self._service._missing_field_for_follow_up(
            workflow=request.workflow,
            decision=request.decision,
            active_booking=state.target_booking,
        )
        follow_up_reply = build_missing_field_follow_up_reply(
            missing_field=missing_field,
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
        )
        if not follow_up_reply:
            return
        state.reply_action = "send_reply"
        state.reply_text = follow_up_reply
        self._service._emit_observability(
            event="intake.reply.normalized_missing_field_follow_up",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_id=self._booking_id(state),
            reply_text=state.reply_text,
        )

    def _emit_apply_start(self, request: ApplyRequest) -> None:
        self._service._emit_observability(
            event="intake.apply.start",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            booking_action=request.actions.booking_action,
            reply_action=request.actions.reply_action,
            ready_to_save=request.actions.ready_to_save,
            sink_action=request.actions.sink_action,
        )

    def _emit_apply_ok(self, request: ApplyRequest, state: ApplyState) -> None:
        self._service._emit_observability(
            event="intake.apply.ok",
            workflow=request.workflow,
            conversation_summary=request.conversation_summary,
            status=str(state.target_booking.get("status", "") or "active"),
            booking_id=self._booking_id(state),
            sink_write_status=str(state.target_booking.get("sink_write_status", "") or "").strip(),
            booking_action=request.actions.booking_action,
            reply_action=state.reply_action,
            ready_to_save=request.actions.ready_to_save,
            saved_summary=state.saved_summary,
        )

    def _applied_result(self, request: ApplyRequest, state: ApplyState) -> dict[str, Any]:
        result = {
            "conversation_id": str(request.conversation_summary.get("conversation_id", "") or ""),
            "matched": True,
            "status": str(state.target_booking.get("status", "") or "active"),
            "booking_id": str(state.target_booking.get("booking_id", "") or ""),
            "saved_summary": state.saved_summary,
        }
        if state.reply_action == "send_reply":
            result["replied"] = True
        return result

    def _conversation_summary_text(self, request: ApplyRequest) -> str:
        text = str(request.decision.get("conversation_summary", "") or "").strip()
        if text:
            return text
        return str(
            request.conversation_summary.get("latest_inbound_message_text_preview", "") or ""
        ).strip()[:300]

    def _should_skip_cancel_sink(
        self,
        request: ApplyRequest,
        state: ApplyState,
        save_payload: dict[str, Any],
    ) -> bool:
        save_status = str(save_payload.get("status", "") or "").strip().lower()
        is_cancellation = state.reply_action == "mark_cancelled" or save_status == "cancelled"
        return is_cancellation and state.sink_status != "succeeded" and not state.sink_ref

    def _stale_result(self, request: ApplyRequest) -> dict[str, Any] | None:
        if not request.stale_guard:
            return None
        return self._service._requeue_if_conversation_stale(
            workflow=request.workflow,
            conversation_id=str(request.conversation_summary.get("conversation_id", "") or ""),
            conversation_summary=request.conversation_summary,
            matched=True,
        )

    @staticmethod
    def _booking_id(state: ApplyState) -> str:
        return str(state.target_booking.get("booking_id", "") or "").strip()
