from __future__ import annotations

from typing import Any

import pytest

from opentulpa.application.turn_orchestrator import TurnOrchestrator
from opentulpa.domain.conversation import ConversationTurnRequest


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke_text(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "ok"


class _FakeWorkflowSetupOrchestrator:
    def __init__(self, *, status: str) -> None:
        self._status = status

    def thread_status(self, *, customer_id: str, thread_id: str) -> dict[str, Any]:
        return {
            "status": self._status,
            "customer_id": customer_id,
            "thread_id": thread_id,
        }


@pytest.mark.asyncio
async def test_turn_orchestrator_uses_workflow_setup_mode_for_active_session() -> None:
    runtime = _FakeRuntime()
    orchestrator = TurnOrchestrator(
        agent_runtime=runtime,
        workflow_setup_orchestrator=_FakeWorkflowSetupOrchestrator(status="active"),
    )

    result = await orchestrator.run_turn(
        ConversationTurnRequest(
            customer_id="telegram_123",
            thread_id="thread_123",
            text="Let's continue the workflow setup.",
        )
    )

    assert result.status == "ok"
    assert runtime.calls[0]["turn_mode"] == "workflow_setup"


@pytest.mark.asyncio
async def test_turn_orchestrator_keeps_interactive_mode_without_active_session() -> None:
    runtime = _FakeRuntime()
    orchestrator = TurnOrchestrator(
        agent_runtime=runtime,
        workflow_setup_orchestrator=_FakeWorkflowSetupOrchestrator(status="paused"),
    )

    await orchestrator.run_turn(
        ConversationTurnRequest(
            customer_id="telegram_123",
            thread_id="thread_123",
            text="Hello",
        )
    )

    assert runtime.calls[0]["turn_mode"] == "interactive"
