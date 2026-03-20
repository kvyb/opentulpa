"""Conversation turn orchestration."""

from __future__ import annotations

from typing import Any

from opentulpa.domain.conversation import ConversationTurnRequest, ConversationTurnResult


class TurnOrchestrator:
    """Executes normalized conversation turns against the agent runtime."""

    def __init__(self, *, agent_runtime: Any | None) -> None:
        self._runtime = agent_runtime

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

        output = await runtime.ainvoke_text(
            thread_id=thread_id,
            customer_id=customer_id,
            text=text,
            turn_mode="interactive",
            include_pending_context=bool(request.include_pending_context),
            recursion_limit_override=request.recursion_limit_override,
        )
        return ConversationTurnResult(
            customer_id=customer_id,
            thread_id=thread_id,
            text=str(output or "").strip(),
            status="ok",
        )
