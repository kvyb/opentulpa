"""Conversation turn orchestration."""

from __future__ import annotations

from typing import Any

from opentulpa.domain.conversation import ConversationTurnRequest, ConversationTurnResult


class TurnOrchestrator:
    """Executes normalized conversation turns against the agent runtime."""

    def __init__(
        self,
        *,
        agent_runtime: Any | None,
        workflow_setup_orchestrator: Any | None = None,
    ) -> None:
        self._runtime = agent_runtime
        self._workflow_setup_orchestrator = workflow_setup_orchestrator

    async def run_turn(self, request: ConversationTurnRequest) -> ConversationTurnResult:
        customer_id = str(request.customer_id or "").strip()
        thread_id = str(request.thread_id or "").strip()
        text = str(request.text or "").strip()
        if not customer_id or not thread_id:
            return ConversationTurnResult(
                customer_id=customer_id,
                thread_id=thread_id,
                text="customer_id and thread_id are required",
                status="error",
            )
        if not text:
            return ConversationTurnResult(
                customer_id=customer_id,
                thread_id=thread_id,
                text="text is required",
                status="error",
            )
        runtime = self._runtime
        if runtime is None or not hasattr(runtime, "ainvoke_text"):
            return ConversationTurnResult(
                customer_id=customer_id,
                thread_id=thread_id,
                text="agent runtime unavailable",
                status="unavailable",
            )

        turn_mode = "interactive"
        setup_orchestrator = self._workflow_setup_orchestrator
        if setup_orchestrator is not None and hasattr(setup_orchestrator, "thread_status"):
            setup_state = setup_orchestrator.thread_status(
                customer_id=customer_id,
                thread_id=thread_id,
            )
            if str(setup_state.get("status", "") or "").strip().lower() == "active":
                turn_mode = "workflow_setup"

        output = await runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            turn_mode=turn_mode,
            include_pending_context=bool(request.include_pending_context),
            recursion_limit_override=request.recursion_limit_override,
        )
        response_text = str(output or "").strip()
        if (
            turn_mode == "workflow_setup"
            and setup_orchestrator is not None
            and hasattr(setup_orchestrator, "after_reply")
        ):
            setup_orchestrator.after_reply(
                customer_id=customer_id,
                thread_id=thread_id,
                reply_text=response_text,
            )
        return ConversationTurnResult(
            customer_id=customer_id,
            thread_id=thread_id,
            text=response_text,
            status="ok",
        )
