"""Restricted Deep Agent decision orchestration for intake workflows."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, cast

from opentulpa.intake.decision import IntakeDecision
from opentulpa.intake.workflow_runtime import safe_dict, unique_string_list
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling import AgentChannel, AgentRunContext, AgentRunKind

logger = logging.getLogger(__name__)
_PUBLIC_DECISION_ERROR = "intake decision could not be completed"


class DecisionMaker:
    """Ask the intake profile for advice and translate it for deterministic application."""

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
        agent = self._agent()
        if agent is None:
            error = "intake decision agent is unavailable"
            self._emit_error(workflow, conversation_summary, error)
            return {}, error
        recent_messages = self._service._normalize_conversation_messages(
            workflow=workflow,
            conversation=conversation,
            recipient_id=str(conversation_summary.get("recipient_id", "") or "").strip() or None,
        )
        decision_input = {
            "workflow": self._workflow_context(workflow),
            "conversation": {
                "summary": conversation_summary,
                "recent_messages": recent_messages,
                "unanswered_customer_messages": self._service._unanswered_customer_messages(
                    recent_messages
                ),
            },
            "active_booking": active_booking,
            "recent_completed_booking": recent_completed_booking,
            "execution_feedback": execution_feedback or [],
        }
        self._service._emit_observability(
            event="intake.decision.start",
            workflow=workflow,
            conversation_summary=conversation_summary,
            recent_message_count=len(recent_messages),
            execution_feedback_count=len(execution_feedback or []),
        )
        context = AgentRunContext(
            tenant_id=str(workflow.get("customer_id", "") or "").strip(),
            actor_id="intake",
            thread_id=self._service._intake_thread_id(
                workflow=workflow,
                conversation_summary=conversation_summary,
            ),
            channel=AgentChannel.INTAKE,
            run_kind=AgentRunKind.INTAKE,
            correlation_id=self._service._intake_trace_id(
                workflow=workflow,
                conversation_summary=conversation_summary,
            ),
            origin=OriginRef(
                interface="intake",
                source_id=str(workflow.get("workflow_id") or "intake-workflow"),
                conversation_id=str(
                    conversation_summary.get("conversation_id") or "conversation"
                ),
            ),
            agent_spec=self._agent_spec(
                str(workflow.get("customer_id", "") or "").strip()
            ),
            trust_class="external",
        )
        try:
            typed = await agent.decide_intake(context=context, decision_input=decision_input)
            decision = (
                typed
                if isinstance(typed, IntakeDecision)
                else IntakeDecision.model_validate(typed)
            )
            self._validate_grounding(
                decision,
                workflow=workflow,
                conversation_summary=conversation_summary,
                recent_messages=recent_messages,
            )
            normalized = self._to_application_decision(
                decision,
                active_booking=active_booking,
                recent_completed_booking=recent_completed_booking,
            )
        except Exception as exc:
            logger.error(
                "intake decision agent failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={
                    "workflow_id": str(workflow.get("workflow_id") or ""),
                    "conversation_id": str(
                        conversation_summary.get("conversation_id") or ""
                    ),
                },
            )
            self._emit_error(workflow, conversation_summary, _PUBLIC_DECISION_ERROR)
            return {}, _PUBLIC_DECISION_ERROR
        self._service._emit_observability(
            event="intake.decision.ok",
            workflow=workflow,
            conversation_summary=conversation_summary,
            action=decision.action,
            evidence_source_ids=decision.evidence_source_ids,
            booking_action=normalized["booking_action"],
            ready_to_save=normalized["ready_to_save"],
        )
        return normalized, None

    def _agent(self) -> Any | None:
        getter = getattr(self._service, "_get_intake_agent", None)
        agent = getter() if callable(getter) else None
        return agent if agent is not None and hasattr(agent, "decide_intake") else None

    def _agent_spec(self, tenant_id: str) -> AgentSpecRef:
        resolver = getattr(self._service, "_resolve_agent_spec", None)
        if callable(resolver):
            return cast(AgentSpecRef, resolver(tenant_id, AgentRunKind.INTAKE.value))
        return AgentSpecRef(tenant_id=tenant_id, spec_id="intake", revision=1)

    @staticmethod
    def _workflow_context(workflow: dict[str, Any]) -> dict[str, Any]:
        return {
            "workflow_id": workflow.get("workflow_id"),
            "name": workflow.get("name"),
            "intent_description": workflow.get("intent_description"),
            "required_fields": workflow.get("required_fields"),
            "field_guidance": workflow.get("field_guidance"),
            "assistant_instructions": workflow.get("assistant_instructions", ""),
            "business_facts": safe_dict(workflow.get("business_facts")),
            "knowledge_file_ids": unique_string_list(workflow.get("knowledge_file_ids")),
            "reply_mode": workflow.get("reply_mode", "auto"),
        }

    @staticmethod
    def _source_ids(values: Iterable[dict[str, Any]]) -> set[str]:
        source_ids: set[str] = set()
        for value in values:
            for key in ("id", "message_id", "source_id"):
                candidate = str(value.get(key, "") or "").strip()
                if candidate:
                    source_ids.add(candidate)
        return source_ids

    @classmethod
    def _validate_grounding(
        cls,
        decision: IntakeDecision,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        recent_messages: list[dict[str, Any]],
    ) -> None:
        if decision.action == "ignore":
            return
        allowed = cls._source_ids(recent_messages)
        allowed.update(unique_string_list(workflow.get("knowledge_file_ids")))
        allowed.update(
            value
            for value in (
                str(conversation_summary.get("latest_inbound_message_id", "") or "").strip(),
                str(conversation_summary.get("latest_outbound_message_id", "") or "").strip(),
            )
            if value
        )
        if not decision.evidence_source_ids:
            raise ValueError("non-ignore intake decisions require evidence_source_ids")
        unknown = sorted(set(decision.evidence_source_ids) - allowed)
        if unknown:
            raise ValueError("intake decision cited unknown evidence: " + ", ".join(unknown))

    @staticmethod
    def _to_application_decision(
        decision: IntakeDecision,
        *,
        active_booking: dict[str, Any] | None,
        recent_completed_booking: dict[str, Any] | None,
    ) -> dict[str, Any]:
        patch = decision.booking_patch
        fields = dict(patch.fields) if patch is not None else {}
        missing_fields = list(patch.missing_fields) if patch is not None else []
        booking_action = "ignore"
        if patch is not None:
            booking_id = str(patch.booking_id or "").strip()
            if active_booking is not None and (
                not booking_id or booking_id == str(active_booking.get("booking_id", "") or "")
            ):
                booking_action = "update_active"
            elif recent_completed_booking is not None and booking_id == str(
                recent_completed_booking.get("booking_id", "") or ""
            ):
                booking_action = "edit_recent_completed"
            else:
                booking_action = "create_new_booking"
        ready_to_save = bool(patch is not None and patch.status in {"completed", "cancelled"})
        reply_action = "send_reply" if decision.reply_text else "none"
        if patch is not None and patch.status == "cancelled":
            reply_action = "mark_cancelled"
            fields["status"] = "cancelled"
        return {
            "ok": True,
            "matches_workflow": decision.action != "ignore",
            "confidence": 1.0,
            "booking_action": booking_action,
            "reply_action": reply_action,
            "reply_text": decision.reply_text or "",
            "ready_to_save": ready_to_save,
            "missing_fields": missing_fields,
            "extracted_fields": fields,
            "save_payload": fields if ready_to_save else {},
            "sink_action": "none",
            "sink_payload": {},
            "evidence_source_ids": decision.evidence_source_ids,
            "reason": f"intake_action={decision.action}",
        }

    def _emit_error(
        self,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
        error: str,
    ) -> None:
        self._service._emit_observability(
            event="intake.decision.error",
            workflow=workflow,
            conversation_summary=conversation_summary,
            error=error,
        )


__all__ = ["DecisionMaker"]
