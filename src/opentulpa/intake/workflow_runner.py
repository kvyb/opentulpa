"""Workflow run orchestration for intake workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

from opentulpa.intake.workflow_boundaries import (
    ConversationCursorSignals,
    WorkflowRunAccumulator,
)
from opentulpa.intake.workflow_runtime import (
    INSTAGRAM_STALE_DECISION_REFRESH_ATTEMPTS as _INSTAGRAM_STALE_DECISION_REFRESH_ATTEMPTS,
)
from opentulpa.intake.workflow_runtime import (
    MAX_DECISION_RECOVERY_ATTEMPTS as _MAX_DECISION_RECOVERY_ATTEMPTS,
)
from opentulpa.intake.workflow_runtime import (
    STALE_TERMINAL_STATUSES as _STALE_TERMINAL_STATUSES,
)
from opentulpa.intake.workflow_runtime import (
    safe_dict as _safe_dict,
)
from opentulpa.intake.workflow_runtime import (
    workflow_requires_intent_match as _workflow_requires_intent_match,
)
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN


@dataclass
class DecisionContext:
    conversation_id: str
    conversation_summary: dict[str, Any]
    signals: ConversationCursorSignals
    conversation: dict[str, Any]
    active_booking: dict[str, Any] | None
    recent_completed_booking: dict[str, Any] | None
    decision: dict[str, Any]


class WorkflowRunService(Protocol):
    def get_workflow(self, *, customer_id: str, workflow_id: str) -> dict[str, Any] | None: ...

    def _load_source_items(
        self,
        *,
        workflow: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str | None, list[dict[str, str]]]: ...

    def _enrich_conversation_summary(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> dict[str, Any]: ...

    def _conversation_lock(self, *, workflow_id: str, conversation_id: str) -> asyncio.Lock: ...

    def _get_cursor(self, *, workflow_id: str, conversation_id: str) -> dict[str, str]: ...

    def _has_new_inbound_signal(
        self,
        *,
        conversation_summary: dict[str, Any],
        cursor: dict[str, str],
        force: bool,
    ) -> bool: ...

    def _conversation_debounce_seconds(self, *, workflow: dict[str, Any], event_type: str) -> float: ...

    def _reload_conversation_summary(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        fallback: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]: ...

    def _latest_inbound_max_age_for_event(
        self,
        *,
        event_type: str,
        workflow: dict[str, Any],
    ) -> timedelta: ...

    def _load_source_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
    ) -> tuple[dict[str, Any], dict[str, Any], str | None]: ...

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
        execution_feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str | None]: ...

    def _uses_latest_inbound_stale_guard(
        self,
        *,
        workflow: dict[str, Any],
        event_type: str,
        force: bool,
    ) -> bool: ...

    def _conversation_became_stale(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        decided_summary: dict[str, Any],
    ) -> tuple[bool, dict[str, Any], str | None]: ...

    def _refreshes_stale_decision_inline(self, *, workflow: dict[str, Any]) -> bool: ...

    def _emit_observability(
        self,
        *,
        event: str,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        **extra: Any,
    ) -> None: ...

    def _requeue_if_conversation_stale(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        matched: bool,
    ) -> dict[str, Any] | None: ...

    def _fallback_out_of_scope_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        decision: dict[str, Any],
    ) -> str: ...

    async def _send_intake_reply(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        reply_text: str,
    ) -> str | None: ...

    async def _apply_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        decision: dict[str, Any],
        stale_guard: bool = False,
    ) -> tuple[dict[str, Any], str | None, dict[str, Any] | None]: ...

    def _set_cursor(
        self,
        *,
        workflow_id: str,
        conversation_id: str,
        latest_inbound_message_id: str,
        latest_inbound_message_time: str,
        conversation_updated_time: str,
        latest_outbound_message_id: str,
        agent_action_at: str = "",
    ) -> None: ...


class WorkflowRunner:
    """Runs intake workflow polling and per-conversation decisions."""

    def __init__(
        self,
        service: WorkflowRunService,
        *,
        is_older_than: Callable[..., bool],
        utc_now_iso: Callable[[], str],
    ) -> None:
        self._service = service
        self._is_older_than = is_older_than
        self._utc_now_iso = utc_now_iso

    async def run_workflow(
        self,
        *,
        customer_id: str,
        workflow_id: str,
        event_type: str = "scheduled",
        force: bool = False,
    ) -> dict[str, Any]:
        workflow = self._service.get_workflow(customer_id=customer_id, workflow_id=workflow_id)
        if workflow is None:
            return {
                "ok": False,
                "workflow_id": workflow_id,
                "summary": f"Intake workflow {workflow_id} was not found.",
            }
        if not bool(workflow.get("enabled")) and not force:
            return {
                "ok": True,
                "workflow_id": workflow_id,
                "summary": NO_NOTIFY_TOKEN,
                "reason": "workflow_disabled",
            }
        items, source_error, source_warnings = self._service._load_source_items(workflow=workflow)
        if source_error is not None:
            return {"ok": False, "workflow_id": workflow_id, "summary": source_error}

        run_state = WorkflowRunAccumulator()
        for item in items:
            await self._run_item(
                workflow=workflow,
                item=item,
                event_type=event_type,
                force=force,
                run_state=run_state,
            )
        return run_state.build_response(
            workflow=workflow,
            workflow_id=str(workflow["workflow_id"]),
            event_type=event_type,
            source_warnings=source_warnings,
            empty_summary_token=NO_NOTIFY_TOKEN,
        )

    async def _run_item(
        self,
        *,
        workflow: dict[str, Any],
        item: dict[str, Any],
        event_type: str,
        force: bool,
        run_state: WorkflowRunAccumulator,
    ) -> None:
        summary = self._service._enrich_conversation_summary(
            workflow=workflow,
            conversation_summary=_safe_dict(item),
        )
        signals = ConversationCursorSignals.from_summary(summary)
        if not signals.conversation_id:
            return
        async with self._service._conversation_lock(
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=signals.conversation_id,
        ):
            await self._run_locked_conversation(
                workflow=workflow,
                conversation_summary=summary,
                conversation_id=signals.conversation_id,
                event_type=event_type,
                force=force,
                run_state=run_state,
            )

    async def _run_locked_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation_id: str,
        event_type: str,
        force: bool,
        run_state: WorkflowRunAccumulator,
    ) -> None:
        prepared = await self._prepare_conversation(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation_id=conversation_id,
            event_type=event_type,
            force=force,
            run_state=run_state,
        )
        if prepared is None:
            return
        summary, signals, cursor = prepared
        run_state.processed += 1
        self._emit_conversation_start(workflow, summary, cursor, force)

        context = await self._load_decision_context(
            workflow=workflow,
            conversation_id=conversation_id,
            conversation_summary=summary,
            run_state=run_state,
        )
        if context is None:
            return
        context = await self._refresh_context_while_stale(
            workflow=workflow,
            context=context,
            event_type=event_type,
            force=force,
            run_state=run_state,
        )
        if context is None:
            return
        if await self._handle_unmatched_decision(
            workflow=workflow,
            context=context,
            event_type=event_type,
            force=force,
            run_state=run_state,
        ):
            return
        await self._apply_matched_decision(
            workflow=workflow,
            context=context,
            event_type=event_type,
            force=force,
            run_state=run_state,
        )

    async def _prepare_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation_id: str,
        event_type: str,
        force: bool,
        run_state: WorkflowRunAccumulator,
    ) -> tuple[dict[str, Any], ConversationCursorSignals, dict[str, str]] | None:
        summary = await self._reload_after_debounce(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation_id=conversation_id,
            event_type=event_type,
            run_state=run_state,
        )
        if summary is None:
            return None
        signals = ConversationCursorSignals.from_summary(summary)
        cursor = self._service._get_cursor(
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        if not self._service._has_new_inbound_signal(
            conversation_summary=summary,
            cursor=cursor,
            force=force,
        ):
            return None
        if self._latest_inbound_too_old(workflow=workflow, summary=summary, event_type=event_type, force=force):
            self._mark_cursor_seen(workflow=workflow, conversation_id=conversation_id, signals=signals)
            return None
        return summary, signals, cursor

    async def _reload_after_debounce(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation_id: str,
        event_type: str,
        run_state: WorkflowRunAccumulator,
    ) -> dict[str, Any] | None:
        debounce_seconds = self._service._conversation_debounce_seconds(
            workflow=workflow,
            event_type=event_type,
        )
        if debounce_seconds <= 0:
            return conversation_summary
        await asyncio.sleep(debounce_seconds)
        summary, refresh_error = self._service._reload_conversation_summary(
            workflow=workflow,
            conversation_id=conversation_id,
            fallback=conversation_summary,
        )
        summary = self._service._enrich_conversation_summary(
            workflow=workflow,
            conversation_summary=summary,
        )
        if refresh_error:
            run_state.errors.append(f"{conversation_id}: {refresh_error}")
            self._emit_error(workflow, summary, "reload_summary", refresh_error)
            return None
        return summary

    def _latest_inbound_too_old(
        self,
        *,
        workflow: dict[str, Any],
        summary: dict[str, Any],
        event_type: str,
        force: bool,
    ) -> bool:
        if force:
            return False
        max_age = self._service._latest_inbound_max_age_for_event(
            event_type=event_type,
            workflow=workflow,
        )
        return bool(
            self._is_older_than(
                summary.get("latest_inbound_message_created_time"),
                max_age=max_age,
            )
        )

    async def _load_decision_context(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        run_state: WorkflowRunAccumulator,
    ) -> DecisionContext | None:
        loaded = self._load_source_conversation(
            workflow=workflow,
            conversation_id=conversation_id,
            conversation_summary=conversation_summary,
            phase="load_conversation",
            run_state=run_state,
        )
        if loaded is None:
            return None
        cursor_summary, conversation = loaded
        return await self._decide_context(
            workflow=workflow,
            conversation_id=conversation_id,
            conversation_summary=cursor_summary,
            conversation=conversation,
            run_state=run_state,
        )

    def _load_source_conversation(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        phase: str,
        run_state: WorkflowRunAccumulator,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        try:
            detailed_summary, conversation, detail_error = self._service._load_source_conversation(
                workflow=workflow,
                conversation_id=conversation_id,
            )
        except Exception as exc:
            detailed_summary = {}
            conversation = {}
            detail_error = str(exc)
        if detail_error:
            error_text = str(detail_error)
            run_state.errors.append(f"{conversation_id}: {error_text}")
            self._emit_error(workflow, conversation_summary, phase, error_text)
            return None
        cursor_summary = self._service._enrich_conversation_summary(
            workflow=workflow,
            conversation_summary=detailed_summary or conversation_summary,
        )
        return cursor_summary, conversation

    async def _decide_context(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        run_state: WorkflowRunAccumulator,
        execution_feedback: list[dict[str, Any]] | None = None,
    ) -> DecisionContext | None:
        active_booking = self._service._get_active_booking(
            customer_id=str(workflow["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        recent_completed_booking = self._service._get_recent_completed_booking(
            customer_id=str(workflow["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        decision, error = await self._service._decide_workflow_action(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
        )
        if error:
            run_state.errors.append(f"{conversation_id}: {error}")
            self._emit_error(workflow, conversation_summary, "decision", error)
            return None
        return DecisionContext(
            conversation_id=conversation_id,
            conversation_summary=conversation_summary,
            signals=ConversationCursorSignals.from_summary(conversation_summary),
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            decision=decision,
        )

    async def _refresh_context_while_stale(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        event_type: str,
        force: bool,
        run_state: WorkflowRunAccumulator,
    ) -> DecisionContext | None:
        if not self._service._uses_latest_inbound_stale_guard(
            workflow=workflow,
            event_type=event_type,
            force=force,
        ):
            return context
        for attempt in range(_INSTAGRAM_STALE_DECISION_REFRESH_ATTEMPTS + 1):
            stale, latest_summary, stale_error = self._service._conversation_became_stale(
                workflow=workflow,
                conversation_id=context.conversation_id,
                decided_summary=context.conversation_summary,
            )
            if stale_error:
                self._emit_error(workflow, context.conversation_summary, "stale_check", stale_error)
            if not stale:
                return context
            if not self._service._refreshes_stale_decision_inline(workflow=workflow) or (
                attempt >= _INSTAGRAM_STALE_DECISION_REFRESH_ATTEMPTS
            ):
                stale_result = self._stale_requeue_result(workflow, context)
                if stale_result is not None:
                    run_state.result_items.append(stale_result)
                return None
            context = await self._refresh_stale_context(
                workflow=workflow,
                context=context,
                latest_summary=latest_summary,
                attempt=attempt,
                run_state=run_state,
            )
            if context is None:
                return None
        return context

    async def _refresh_stale_context(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        latest_summary: dict[str, Any],
        attempt: int,
        run_state: WorkflowRunAccumulator,
    ) -> DecisionContext | None:
        self._service._emit_observability(
            event="intake.conversation.stale_refresh",
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            latest_inbound_message_id=str(latest_summary.get("latest_inbound_message_id", "") or "").strip(),
            attempt=attempt + 1,
        )
        loaded = self._load_source_conversation(
            workflow=workflow,
            conversation_id=context.conversation_id,
            conversation_summary=context.conversation_summary,
            phase="stale_refresh",
            run_state=run_state,
        )
        if loaded is None:
            return None
        summary, conversation = loaded
        return await self._decide_context(
            workflow=workflow,
            conversation_id=context.conversation_id,
            conversation_summary=summary or latest_summary,
            conversation=conversation,
            run_state=run_state,
        )

    async def _handle_unmatched_decision(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        event_type: str,
        force: bool,
        run_state: WorkflowRunAccumulator,
    ) -> bool:
        if self._decision_effectively_matches(workflow, context.decision):
            return False
        reply_action, reply_text = self._unmatched_reply(workflow, context)
        if reply_action == "send_reply" and reply_text:
            if self._stale_guard_enabled(workflow=workflow, event_type=event_type, force=force):
                stale_result = self._service._requeue_if_conversation_stale(
                    workflow=workflow,
                    conversation_id=context.conversation_id,
                    conversation_summary=context.conversation_summary,
                    matched=False,
                )
                if stale_result is not None:
                    run_state.result_items.append(stale_result)
                    return True
            await self._send_unmatched_reply(
                workflow=workflow,
                context=context,
                reply_action=reply_action,
                reply_text=reply_text,
                run_state=run_state,
            )
            return True
        self._record_ignored(workflow=workflow, context=context, run_state=run_state)
        return True

    def _unmatched_reply(
        self,
        workflow: dict[str, Any],
        context: DecisionContext,
    ) -> tuple[str, str]:
        reply_action = str(context.decision.get("reply_action", "none") or "none").strip().lower()
        reply_text = str(context.decision.get("reply_text", "") or "").strip()
        if reply_action == "send_reply" and reply_text:
            return reply_action, reply_text
        fallback_reply = self._service._fallback_out_of_scope_reply(
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            decision=context.decision,
        )
        if fallback_reply:
            self._service._emit_observability(
                event="intake.reply.fallback_out_of_scope",
                workflow=workflow,
                conversation_summary=context.conversation_summary,
                reply_text=fallback_reply,
            )
            return "send_reply", fallback_reply
        return reply_action, reply_text

    async def _send_unmatched_reply(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        reply_action: str,
        reply_text: str,
        run_state: WorkflowRunAccumulator,
    ) -> None:
        self._service._emit_observability(
            event="intake.apply.start",
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            booking_action="ignore",
            reply_action=reply_action,
            ready_to_save=False,
        )
        self._service._emit_observability(
            event="intake.reply.start",
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            booking_id="",
            reply_text=reply_text,
        )
        reply_error = await self._service._send_intake_reply(
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            reply_text=reply_text,
        )
        if reply_error is not None:
            run_state.errors.append(f"{context.conversation_id}: {reply_error}")
            self._service._emit_observability(
                event="intake.reply.error",
                workflow=workflow,
                conversation_summary=context.conversation_summary,
                booking_id="",
                error=reply_error,
            )
            self._emit_error(workflow, context.conversation_summary, "reply_execution", reply_error)
            return
        self._record_ignored(
            workflow=workflow,
            context=context,
            run_state=run_state,
            replied=True,
                agent_action_at=self._utc_now_iso(),
        )

    async def _apply_matched_decision(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        event_type: str,
        force: bool,
        run_state: WorkflowRunAccumulator,
    ) -> None:
        run_state.matched += 1
        applied, apply_error = await self._apply_with_recovery(
            workflow=workflow,
            context=context,
            event_type=event_type,
            force=force,
        )
        if apply_error:
            run_state.errors.append(f"{context.conversation_id}: {apply_error}")
            self._emit_error(workflow, context.conversation_summary, "apply", apply_error)
            return
        if str(applied.get("status", "") or "").strip() in _STALE_TERMINAL_STATUSES:
            run_state.result_items.append(applied)
            return
        self._mark_cursor_seen(
            workflow=workflow,
            conversation_id=context.conversation_id,
            signals=context.signals,
            agent_action_at=self._utc_now_iso(),
        )
        run_state.result_items.append(applied)
        saved_summary = str(applied.get("saved_summary", "") or "").strip()
        if saved_summary:
            run_state.saved_notifications.append(saved_summary)
        self._service._emit_observability(
            event="intake.conversation.complete",
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            matched=True,
            status=str(applied.get("status", "") or "").strip(),
            booking_id=str(applied.get("booking_id", "") or "").strip(),
            saved_summary=saved_summary,
        )

    async def _apply_with_recovery(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        event_type: str,
        force: bool,
    ) -> tuple[dict[str, Any], str | None]:
        recovery_feedback: list[dict[str, Any]] = []
        current = context
        apply_error: str | None = None
        applied: dict[str, Any] = {}
        for attempt in range(_MAX_DECISION_RECOVERY_ATTEMPTS + 1):
            applied, apply_error, feedback = await self._service._apply_decision(
                workflow=workflow,
                conversation_summary=current.conversation_summary,
                conversation=current.conversation,
                active_booking=current.active_booking,
                recent_completed_booking=current.recent_completed_booking,
                decision=current.decision,
                stale_guard=self._stale_guard_enabled(workflow=workflow, event_type=event_type, force=force),
            )
            if apply_error is None:
                return applied, None
            if attempt >= _MAX_DECISION_RECOVERY_ATTEMPTS or feedback is None:
                break
            recovery_feedback.append(feedback)
            current, decision_error = await self._decide_recovery_context(
                workflow=workflow,
                conversation_id=current.conversation_id,
                conversation_summary=current.conversation_summary,
                conversation=current.conversation,
                execution_feedback=recovery_feedback,
            )
            if current is None:
                return applied, decision_error or apply_error
            if _workflow_requires_intent_match(workflow) and not bool(current.decision.get("matches_workflow")):
                return applied, "recovery decision no longer matches workflow"
        return applied, apply_error

    async def _decide_recovery_context(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        execution_feedback: list[dict[str, Any]],
    ) -> tuple[DecisionContext | None, str | None]:
        active_booking = self._service._get_active_booking(
            customer_id=str(workflow["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        recent_completed_booking = self._service._get_recent_completed_booking(
            customer_id=str(workflow["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
        )
        decision, error = await self._service._decide_workflow_action(
            workflow=workflow,
            conversation_summary=conversation_summary,
            conversation=conversation,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
        )
        if error:
            return None, error
        return (
            DecisionContext(
                conversation_id=conversation_id,
                conversation_summary=conversation_summary,
                signals=ConversationCursorSignals.from_summary(conversation_summary),
                conversation=conversation,
                active_booking=active_booking,
                recent_completed_booking=recent_completed_booking,
                decision=decision,
            ),
            None,
        )

    def _record_ignored(
        self,
        *,
        workflow: dict[str, Any],
        context: DecisionContext,
        run_state: WorkflowRunAccumulator,
        replied: bool = False,
        agent_action_at: str = "",
    ) -> None:
        self._mark_cursor_seen(
            workflow=workflow,
            conversation_id=context.conversation_id,
            signals=context.signals,
            agent_action_at=agent_action_at,
        )
        item = {
            "conversation_id": context.conversation_id,
            "matched": False,
            "status": "ignored",
        }
        if replied:
            item["replied"] = True
            self._service._emit_observability(
                event="intake.reply.ok",
                workflow=workflow,
                conversation_summary=context.conversation_summary,
                booking_id="",
            )
            self._service._emit_observability(
                event="intake.apply.ok",
                workflow=workflow,
                conversation_summary=context.conversation_summary,
                status="ignored",
                booking_action="ignore",
                reply_action="send_reply",
                ready_to_save=False,
            )
        run_state.result_items.append(item)
        self._service._emit_observability(
            event="intake.conversation.complete",
            workflow=workflow,
            conversation_summary=context.conversation_summary,
            matched=False,
            status="ignored",
            replied=replied,
        )

    def _stale_requeue_result(
        self,
        workflow: dict[str, Any],
        context: DecisionContext,
    ) -> dict[str, Any] | None:
        effective_matches = self._decision_effectively_matches(workflow, context.decision)
        return self._service._requeue_if_conversation_stale(
            workflow=workflow,
            conversation_id=context.conversation_id,
            conversation_summary=context.conversation_summary,
            matched=effective_matches,
        )

    def _decision_effectively_matches(
        self,
        workflow: dict[str, Any],
        decision: dict[str, Any],
    ) -> bool:
        intent_match_required = _workflow_requires_intent_match(workflow)
        return bool(decision.get("matches_workflow")) or not intent_match_required

    def _stale_guard_enabled(
        self,
        *,
        workflow: dict[str, Any],
        event_type: str,
        force: bool,
    ) -> bool:
        return self._service._uses_latest_inbound_stale_guard(
            workflow=workflow,
            event_type=event_type,
            force=force,
        )

    def _mark_cursor_seen(
        self,
        *,
        workflow: dict[str, Any],
        conversation_id: str,
        signals: ConversationCursorSignals,
        agent_action_at: str = "",
    ) -> None:
        self._service._set_cursor(
            workflow_id=str(workflow["workflow_id"]),
            conversation_id=conversation_id,
            latest_inbound_message_id=signals.latest_inbound_message_id,
            latest_inbound_message_time=signals.latest_inbound_message_time,
            conversation_updated_time=signals.conversation_updated_time,
            latest_outbound_message_id=signals.latest_outbound_message_id,
            agent_action_at=agent_action_at,
        )

    def _emit_conversation_start(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        cursor: dict[str, str],
        force: bool,
    ) -> None:
        self._service._emit_observability(
            event="intake.conversation.start",
            workflow=workflow,
            conversation_summary=conversation_summary,
            force=bool(force),
            cursor_latest_inbound_message_id=str(
                cursor.get("latest_inbound_message_id", "") or ""
            ).strip(),
            cursor_latest_outbound_message_id=str(
                cursor.get("latest_outbound_message_id", "") or ""
            ).strip(),
        )

    def _emit_error(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        phase: str,
        error: str,
    ) -> None:
        self._service._emit_observability(
            event="intake.conversation.error",
            workflow=workflow,
            conversation_summary=conversation_summary,
            phase=phase,
            error=error,
        )
