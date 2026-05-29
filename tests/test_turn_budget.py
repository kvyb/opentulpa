from __future__ import annotations

from opentulpa.agent.turn_budget import (
    budget_status_context,
    consume_model_call,
    initial_turn_budget,
    record_search_calls,
    record_tool_round,
)
from opentulpa.agent.turn_control import record_tool_round_for_turn


def test_turn_budget_derives_model_calls_from_graph_recursion_limit() -> None:
    budget = initial_turn_budget(turn_mode="interactive", graph_recursion_limit=30)

    assert budget["max_model_calls"] == 8
    assert budget["used_model_calls"] == 0
    assert budget["used_tool_rounds"] == 0
    assert budget["max_search_calls"] >= 1
    assert budget["used_search_calls"] == 0


def test_turn_budget_consumes_until_finalizer_required() -> None:
    budget = initial_turn_budget(turn_mode="interactive", graph_recursion_limit=8)

    first = consume_model_call(
        budget,
        turn_mode="interactive",
        graph_recursion_limit=8,
    )
    second = consume_model_call(
        first.state,
        turn_mode="interactive",
        graph_recursion_limit=8,
    )
    third = consume_model_call(
        second.state,
        turn_mode="interactive",
        graph_recursion_limit=8,
    )

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False
    assert third.reason == "model_call_budget_exhausted(2/2)"


def test_turn_budget_tracks_tool_rounds_and_near_budget_context() -> None:
    budget = initial_turn_budget(turn_mode="interactive", graph_recursion_limit=8)
    budget = consume_model_call(
        budget,
        turn_mode="interactive",
        graph_recursion_limit=8,
    ).state
    budget = record_tool_round(
        budget,
        turn_mode="interactive",
        graph_recursion_limit=8,
    )

    assert budget["used_tool_rounds"] == 1
    assert "TURN_BUDGET_STATUS" in budget_status_context(budget)


def test_turn_budget_tracks_web_search_calls() -> None:
    budget = initial_turn_budget(turn_mode="interactive", graph_recursion_limit=8)
    budget["max_search_calls"] = 2

    budget = record_search_calls(
        budget,
        search_call_count=3,
        turn_mode="interactive",
        graph_recursion_limit=8,
    )

    assert budget["used_search_calls"] == 2


def test_turn_budget_status_warns_when_web_search_cap_exhausted() -> None:
    budget = initial_turn_budget(turn_mode="interactive", graph_recursion_limit=30)
    budget["max_search_calls"] = 2
    budget["used_search_calls"] = 2

    status = budget_status_context(budget)

    assert "TURN_BUDGET_STATUS" in status
    assert "Web searches used: 2/2" in status
    assert "Do not call web_search again" in status
    assert "near its runtime budget" not in status


def test_turn_control_records_executed_web_search_calls_in_turn_budget() -> None:
    runtime = type("Runtime", (), {"recursion_limit": 8})()
    budget = initial_turn_budget(turn_mode="interactive", graph_recursion_limit=8)
    budget["max_search_calls"] = 5

    updated = record_tool_round_for_turn(
        runtime=runtime,
        state={"turn_budget": budget},
        turn_mode="interactive",
        tool_calls=[{"id": "call_1", "name": "web_search", "args": {"query": "one"}}],
    )

    assert updated["used_tool_rounds"] == 1
    assert updated["used_search_calls"] == 1
