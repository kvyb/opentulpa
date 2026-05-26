"""Model decision orchestration for intake workflows."""

from __future__ import annotations

from typing import Any

from opentulpa.intake.workflow_runtime import (
    safe_dict as _safe_dict,
)
from opentulpa.intake.workflow_runtime import (
    safe_list as _safe_list,
)
from opentulpa.intake.workflow_runtime import (
    unique_string_list as _unique_string_list,
)
from opentulpa.intake.workflow_runtime import (
    workflow_requires_intent_match as _workflow_requires_intent_match,
)


class DecisionMaker:
    """Builds intake decision context and handles knowledge-assisted retries."""

    def __init__(self, service: Any) -> None:
        self._service = service

    async def decide_workflow_action(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        conversation: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        execution_feedback: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        runtime = self._runtime()
        if runtime is None:
            error = "agent runtime does not support intake workflow decisions"
            self._emit_decision_error(workflow, conversation_summary, error)
            return {}, error
        recent_messages = self._service._normalize_conversation_messages(
            workflow=workflow,
            conversation=conversation,
            recipient_id=str(conversation_summary.get("recipient_id", "") or "").strip() or None,
        )
        unanswered = self._service._unanswered_customer_messages(recent_messages)
        workflow_context = self._workflow_context(workflow)
        self._emit_decision_start(
            workflow=workflow,
            conversation_summary=conversation_summary,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            recent_message_count=len(recent_messages),
            execution_feedback=execution_feedback,
        )
        decision, error = await self._call_runtime(
            runtime=runtime,
            workflow=workflow,
            workflow_context=workflow_context,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            unanswered_customer_messages=unanswered,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
        )
        if error is not None:
            self._emit_decision_error(workflow, conversation_summary, error)
            return {}, error
        decision, error = await self._normalize_or_retry_with_knowledge(
            runtime=runtime,
            workflow=workflow,
            workflow_context=workflow_context,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            unanswered_customer_messages=unanswered,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
            decision=decision,
        )
        if error is not None:
            self._emit_decision_error(workflow, conversation_summary, error)
            return {}, error
        if not isinstance(decision, dict) or not bool(decision.get("ok", False)):
            error = str(_safe_dict(decision).get("error", "invalid intake workflow decision"))
            self._emit_decision_error(workflow, conversation_summary, error, decision=decision)
            return {}, error
        self._emit_decision_ok(workflow, conversation_summary, decision)
        return decision, None

    def _runtime(self) -> Any | None:
        getter = getattr(self._service, "_get_agent_runtime", None)
        runtime = getter() if callable(getter) else None
        if runtime is None or not hasattr(runtime, "decide_intake_workflow"):
            return None
        return runtime

    def _workflow_context(self, workflow: dict[str, Any]) -> dict[str, Any]:
        return {
            "workflow_id": workflow.get("workflow_id"),
            "name": workflow.get("name"),
            "intent_description": workflow.get("intent_description"),
            "intent_match_required": _workflow_requires_intent_match(workflow),
            "required_fields": workflow.get("required_fields"),
            "field_guidance": workflow.get("field_guidance"),
            "assistant_instructions": workflow.get("assistant_instructions", ""),
            "business_facts": _safe_dict(workflow.get("business_facts")),
            "workflow_skill": self._service._workflow_skill_context(
                customer_id=str(workflow.get("customer_id", "") or ""),
                workflow_id=str(workflow.get("workflow_id", "") or ""),
            ),
            "knowledge_file_ids": _unique_string_list(workflow.get("knowledge_file_ids")),
            "knowledge_answer": "",
            "sink_type": workflow.get("sink_type"),
            "sink_config": workflow.get("sink_config"),
            "channel": workflow.get("channel"),
            "provider": workflow.get("provider"),
        }

    async def _call_runtime(
        self,
        *,
        runtime: Any,
        workflow: dict[str, Any],
        workflow_context: dict[str, Any],
        conversation_summary: dict[str, Any],
        recent_messages: list[dict[str, Any]],
        unanswered_customer_messages: list[dict[str, Any]],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        execution_feedback: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], str | None]:
        try:
            decision = await runtime.decide_intake_workflow(
                customer_id=str(workflow["customer_id"]),
                workflow=dict(workflow_context),
                conversation={
                    "summary": conversation_summary,
                    "recent_messages": recent_messages,
                    "unanswered_customer_messages": unanswered_customer_messages,
                },
                active_booking=active_booking,
                recent_completed_booking=recent_completed_booking,
                execution_feedback=execution_feedback,
            )
        except Exception as exc:
            return {}, str(exc)
        return _safe_dict(decision), None

    async def _normalize_or_retry_with_knowledge(
        self,
        *,
        runtime: Any,
        workflow: dict[str, Any],
        workflow_context: dict[str, Any],
        conversation_summary: dict[str, Any],
        recent_messages: list[dict[str, Any]],
        unanswered_customer_messages: list[dict[str, Any]],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        execution_feedback: list[dict[str, Any]] | None,
        decision: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        if not self._needs_business_knowledge(decision):
            return decision, None
        file_ids = _unique_string_list(workflow.get("knowledge_file_ids"))
        if not file_ids:
            return self._normalize_no_file_decision(
                workflow=workflow,
                conversation_summary=conversation_summary,
                active_booking=active_booking,
                decision=decision,
            ), None
        return await self._retry_with_knowledge(
            runtime=runtime,
            workflow=workflow,
            workflow_context=workflow_context,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            unanswered_customer_messages=unanswered_customer_messages,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
            decision=decision,
        )

    async def _retry_with_knowledge(
        self,
        *,
        runtime: Any,
        workflow: dict[str, Any],
        workflow_context: dict[str, Any],
        conversation_summary: dict[str, Any],
        recent_messages: list[dict[str, Any]],
        unanswered_customer_messages: list[dict[str, Any]],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        execution_feedback: list[dict[str, Any]] | None,
        decision: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None]:
        query = str(decision.get("business_knowledge_query", "") or "").strip()
        if not query:
            query = self._service._business_knowledge_query_text(
                workflow=workflow,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
                active_booking=active_booking,
            )
        self._emit_knowledge_start(workflow, conversation_summary, query)
        knowledge_answer = self._service._business_knowledge_answer_for_workflow(
            customer_id=str(workflow["customer_id"]),
            workflow=workflow,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            active_booking=active_booking,
            query_override=query,
            include_no_source=True,
        )
        workflow_context["knowledge_answer"] = knowledge_answer
        self._emit_knowledge_retry(workflow, conversation_summary, query, knowledge_answer)
        return await self._call_runtime(
            runtime=runtime,
            workflow=workflow,
            workflow_context=workflow_context,
            conversation_summary=conversation_summary,
            recent_messages=recent_messages,
            unanswered_customer_messages=unanswered_customer_messages,
            active_booking=active_booking,
            recent_completed_booking=recent_completed_booking,
            execution_feedback=execution_feedback,
        )

    def _normalize_no_file_decision(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        active_booking: dict[str, Any] | None,
        decision: dict[str, Any],
    ) -> dict[str, Any]:
        prior_query = str(decision.get("business_knowledge_query", "") or "").strip()
        normalized = self._service._normalize_no_file_business_knowledge_decision(
            workflow=workflow,
            conversation_summary=conversation_summary,
            active_booking=active_booking,
            decision=decision,
        )
        self._service._emit_observability(
            event="intake.decision.normalized_no_knowledge_files",
            workflow=workflow,
            conversation_summary=conversation_summary,
            business_knowledge_query=prior_query,
            reply_action=str(normalized.get("reply_action", "") or "").strip().lower(),
            missing_fields=_unique_string_list(normalized.get("missing_fields")),
        )
        return normalized

    @staticmethod
    def _needs_business_knowledge(decision: dict[str, Any]) -> bool:
        return bool(decision.get("ok", False)) and bool(
            decision.get("needs_business_knowledge", False)
        )

    def _emit_decision_start(
        self,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
        recent_message_count: int,
        execution_feedback: list[dict[str, Any]] | None,
    ) -> None:
        self._service._emit_observability(
            event="intake.decision.start",
            workflow=workflow,
            conversation_summary=conversation_summary,
            active_booking_id=str((active_booking or {}).get("booking_id", "") or "").strip(),
            recent_completed_booking_id=str(
                (recent_completed_booking or {}).get("booking_id", "") or ""
            ).strip(),
            recent_message_count=recent_message_count,
            knowledge_answer_chars=0,
            execution_feedback_count=len(execution_feedback or []),
            execution_feedback=_safe_list(execution_feedback),
        )

    def _emit_decision_error(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        error: str,
        *,
        decision: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "intake.decision.error",
            "workflow": workflow,
            "conversation_summary": conversation_summary,
            "error": error,
        }
        if decision is not None:
            payload["decision"] = _safe_dict(decision)
        self._service._emit_observability(**payload)

    def _emit_decision_ok(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        decision: dict[str, Any],
    ) -> None:
        self._service._emit_observability(
            event="intake.decision.ok",
            workflow=workflow,
            conversation_summary=conversation_summary,
            matches_workflow=bool(decision.get("matches_workflow")),
            confidence=decision.get("confidence"),
            booking_action=str(decision.get("booking_action", "") or "").strip().lower(),
            reply_action=str(decision.get("reply_action", "") or "").strip().lower(),
            ready_to_save=bool(decision.get("ready_to_save")),
            needs_business_knowledge=bool(decision.get("needs_business_knowledge", False)),
            business_knowledge_query=str(decision.get("business_knowledge_query", "") or "").strip(),
            missing_fields=_unique_string_list(decision.get("missing_fields")),
            extracted_fields=_safe_dict(decision.get("extracted_fields")),
            save_payload=_safe_dict(decision.get("save_payload")),
            sink_action=str(decision.get("sink_action", "") or "").strip().lower(),
            sink_payload=_safe_dict(decision.get("sink_payload")),
            reason=str(decision.get("reason", "") or "").strip(),
        )

    def _emit_knowledge_start(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        query: str,
    ) -> None:
        self._service._emit_observability(
            event="intake.knowledge_query.start",
            workflow=workflow,
            conversation_summary=conversation_summary,
            query=query,
        )

    def _emit_knowledge_retry(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        query: str,
        knowledge_answer: str,
    ) -> None:
        self._service._emit_observability(
            event="intake.knowledge_query.ok",
            workflow=workflow,
            conversation_summary=conversation_summary,
            query=query,
            knowledge_answer_chars=len(knowledge_answer),
        )
        self._service._emit_observability(
            event="intake.decision.retry_with_knowledge",
            workflow=workflow,
            conversation_summary=conversation_summary,
            knowledge_answer_chars=len(knowledge_answer),
        )
