from __future__ import annotations

import pytest

from opentulpa.agent.lc_messages import AIMessage, HumanMessage
from opentulpa.agent.turn_finalizer import (
    finalize_turn_response,
    generate_final_response_from_turn_state,
)


class _FinalizerRuntime:
    recursion_limit = 8
    _model = object()

    def __init__(self) -> None:
        self.prompt = ""

    async def ainvoke_model(self, model, messages, **kwargs):  # noqa: ANN001, ANN202
        assert model is self._model
        assert kwargs["stable_prefix_count"] == 0
        assert kwargs["cacheable_prefix_count"] == 0
        self.prompt = str(messages[0].content)
        return AIMessage(content="final answer")


@pytest.mark.asyncio
async def test_turn_finalizer_prompt_requires_deliverable_not_continue_handoff() -> None:
    runtime = _FinalizerRuntime()

    text = await generate_final_response_from_turn_state(
        runtime=runtime,
        state={
            "turn_mode": "interactive",
            "messages": [HumanMessage(content="Make a research list")],
            "turn_plan": [
                {"id": "1", "content": "Research candidates", "status": "in_progress"}
            ],
        },
        reason="model_call_budget_exhausted_with_open_plan",
    )

    assert text == "final answer"
    assert "Fulfill the requested deliverable directly" in runtime.prompt
    assert "do not end by asking the user to continue" in runtime.prompt
    assert "unless no useful deliverable can be produced" in runtime.prompt


@pytest.mark.asyncio
async def test_turn_finalizer_returns_existing_answer_even_when_plan_open() -> None:
    runtime = _FinalizerRuntime()

    result = await finalize_turn_response(
        runtime=runtime,
        state={
            "turn_mode": "interactive",
            "messages": [
                HumanMessage(content="Make a research list"),
                AIMessage(content="Here is the researched list."),
            ],
            "turn_plan": [
                {"id": "1", "content": "Research candidates", "status": "in_progress"}
            ],
            "turn_budget": {"max_model_calls": 2, "used_model_calls": 2},
        },
    )

    assert result["final_response_text"] == "Here is the researched list."
    assert runtime.prompt == ""
