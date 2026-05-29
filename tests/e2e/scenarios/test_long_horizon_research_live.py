from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest
from evaluation.judge import evaluate_e2e_scenario_with_llm_judge
from harness.runner import E2EHarness, load_jsonl

from opentulpa.agent.lc_messages import AIMessage
from opentulpa.agent.tool_budget import web_search_call_count
from opentulpa.integrations.web_search import get_web_search_backend_name

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]


def _tool_call_records(messages: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in getattr(message, "tool_calls", []) or []:
            if isinstance(call, dict):
                records.append(call)
    return records


def _executed_internal_web_search_count(system_events: list[dict[str, Any]]) -> int:
    return sum(
        1
        for event in system_events
        if event.get("kind") == "internal_api_call"
        and event.get("path") == "/internal/web_search"
    )


def test_live_long_horizon_research_uses_plan_and_returns_artifact(
    e2e_harness: E2EHarness,
) -> None:
    customer_id = "cust_e2e_long_horizon_research"
    thread_id = "thread_e2e_instagram_creator_research"
    user_request = (
        "I want you to go and search for instagram influencers/creators that sell stuff "
        "via instagram, then give me a list of those that could benefit greatly from "
        "instagram automation. Send me the list"
    )

    response = e2e_harness.post_chat(
        customer_id=customer_id,
        thread_id=thread_id,
        text=user_request,
    )

    assert response["status_code"] == 200
    assert response["payload"].get("ok") is True
    answer = str(response["payload"].get("text", "") or "").strip()
    assert len(answer) >= 400
    assert "reply timeout" not in answer.lower()
    assert any(term in answer.lower() for term in ("instagram", "creator", "influencer"))
    assert any(term in answer.lower() for term in ("dm", "automation", "inbound"))

    assert e2e_harness.runtime._graph is not None
    snapshot = asyncio.run(
        e2e_harness.runtime._graph.aget_state(
            config={"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
        )
    )
    state = getattr(snapshot, "values", {}) or {}
    messages = state.get("messages", [])
    assert isinstance(messages, list)
    tool_calls = _tool_call_records(messages)
    tool_names = [str(call.get("name", "")).strip() for call in tool_calls]
    attempted_web_search_count = sum(web_search_call_count(call) for call in tool_calls)
    turn_budget = state.get("turn_budget") if isinstance(state.get("turn_budget"), dict) else {}
    executed_web_search_count = int(turn_budget.get("used_search_calls", 0) or 0)
    turn_plan = state.get("turn_plan")
    assert "turn_plan" in tool_names
    assert isinstance(turn_plan, list)
    assert len(turn_plan) >= 2
    system_events = load_jsonl(e2e_harness.system_log_path)
    internal_web_search_count = _executed_internal_web_search_count(system_events)
    if get_web_search_backend_name() == "exa":
        assert executed_web_search_count <= 2
        assert internal_web_search_count <= 2

    traces = load_jsonl(e2e_harness.llm_trace_path)
    behavior = load_jsonl(e2e_harness.behavior_log_path)
    judge_model = (
        os.getenv("OPENTULPA_E2E_JUDGE_MODEL", "").strip()
        or "deepseek/deepseek-v4-pro"
    )
    evaluation = evaluate_e2e_scenario_with_llm_judge(
        scenario="live_long_horizon_research_uses_plan_and_returns_artifact",
        details={
            "judge_instruction": (
                "Pass only if the agent made a current-turn plan, used it reasonably, "
                "did not loop excessively, respected the Exa web_search cap when Exa is enabled, "
                "and returned a concrete list relevant to Instagram creators who could benefit "
                "from DM/inbound automation."
            ),
            "original_user_request": user_request,
            "final_answer_excerpt": answer[:3000],
            "turn_plan": turn_plan,
            "tool_names": tool_names,
            "attempted_web_search_count": attempted_web_search_count,
            "executed_web_search_count": executed_web_search_count,
            "internal_web_search_count": internal_web_search_count,
            "web_search_backend": get_web_search_backend_name(),
            "llm_trace_count": len(traces),
            "behavior_event_count": len(behavior),
            "turn_budget": turn_budget,
        },
        system_log_path=e2e_harness.system_log_path,
        behavior_log_path=e2e_harness.behavior_log_path,
        llm_trace_path=e2e_harness.llm_trace_path,
        model=judge_model,
        timeout_seconds=60.0,
    )
    parsed = evaluation.get("parsed") if isinstance(evaluation.get("parsed"), dict) else {}
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    assert evaluation.get("ok") is True, evaluation
    assert parsed.get("verdict") == "pass", evaluation
    assert int(scores.get("task_completion", 0) or 0) >= 4, evaluation
    assert int(scores.get("correctness", 0) or 0) >= 4, evaluation

    report = e2e_harness.write_status_report(
        scenario="live_long_horizon_research_uses_plan_and_returns_artifact",
        ok=True,
        details={
            "original_user_request": user_request,
            "final_answer_excerpt": answer[:3000],
            "turn_plan": turn_plan,
            "tool_names": tool_names,
            "attempted_web_search_count": attempted_web_search_count,
            "executed_web_search_count": executed_web_search_count,
            "internal_web_search_count": internal_web_search_count,
            "judge": evaluation,
        },
    )
    assert report.exists()
