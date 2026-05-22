from __future__ import annotations

import pytest

from opentulpa.agent.lc_messages import HumanMessage
from opentulpa.agent.tool_execution_policy import ToolExecutionPolicy


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _Runtime:
    def tools_for_turn_mode(self, turn_mode: str) -> list[_Tool]:
        assert turn_mode == "interactive"
        return [_Tool("allowed_tool")]


def test_tool_execution_policy_rejects_unbound_tool() -> None:
    policy = ToolExecutionPolicy.from_runtime_state(
        runtime=_Runtime(),
        state={"turn_mode": "interactive", "thread_id": "thread_1", "customer_id": "telegram_1"},
    )

    with pytest.raises(ValueError, match="not bound in this turn"):
        policy.validate_call(call_name="missing_tool", customer_scoped_tools=set())


def test_tool_execution_policy_requires_customer_scope() -> None:
    policy = ToolExecutionPolicy.from_runtime_state(
        runtime=object(),
        state={"turn_mode": "interactive", "thread_id": "thread_1", "customer_id": ""},
    )

    with pytest.raises(ValueError, match="requires customer scope"):
        policy.validate_call(call_name="customer_tool", customer_scoped_tools={"customer_tool"})


def test_tool_execution_policy_adds_execution_origin_to_routine_create() -> None:
    policy = ToolExecutionPolicy.from_runtime_state(
        runtime=object(),
        state={"turn_mode": "interactive", "thread_id": "thread_1", "customer_id": "telegram_1"},
    )

    args = policy.prepare_args(
        call_name="routine_create",
        args={"schedule": "* * * * *"},
        messages=[HumanMessage(content="remind me in 5 minutes")],
    )

    assert args["thread_id"] == "thread_1"
    assert args["execution_origin"] == "interactive"
    assert "T" in args["schedule"]
