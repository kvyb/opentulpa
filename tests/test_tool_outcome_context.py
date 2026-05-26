from __future__ import annotations

from opentulpa.agent.tool_outcome_context import (
    add_tool_outcomes,
    build_tool_outcome_context,
    compact_tool_result_for_model,
    next_tool_round_id,
)


def test_tool_outcome_reducer_keeps_latest_ten_rounds() -> None:
    existing = [
        {"round_id": index, "tool_name": "tool", "status": "ok", "result_text": str(index)}
        for index in range(1, 10)
    ]
    incoming = [
        {"round_id": 10, "tool_name": "tool", "status": "ok", "result_text": "10"},
        {"round_id": 11, "tool_name": "tool", "status": "ok", "result_text": "11"},
    ]

    reduced = add_tool_outcomes(existing, incoming)

    assert [item["round_id"] for item in reduced] == list(range(2, 12))
    assert next_tool_round_id(reduced) == 12


def test_tool_result_compaction_removes_noisy_payload_and_preserves_content() -> None:
    compact = compact_tool_result_for_model(
        tool_name="business_knowledge_query",
        result={
            "ok": True,
            "answer": "SUV full wash costs 2500 rubles.",
            "headers": {"authorization": "secret"},
            "raw_response": "noise",
            "sources": [{"title": "prices.xlsx", "row": 4}],
        },
    )

    assert "business_knowledge_query result:" in compact
    assert "SUV full wash costs 2500 rubles" in compact
    assert "prices.xlsx" in compact
    assert "authorization" not in compact
    assert "raw_response" not in compact


def test_tool_outcome_context_groups_previous_rounds() -> None:
    context = build_tool_outcome_context(
        [
            {
                "round_id": 1,
                "tool_name": "business_knowledge_index",
                "status": "ok",
                "result_text": "indexed file_1",
            },
            {
                "round_id": 2,
                "tool_name": "business_knowledge_query",
                "status": "ok",
                "result_text": "Мойка section found",
            },
        ]
    )

    assert "Previous tool results" in context
    assert "Tool round 1" in context
    assert "indexed file_1" in context
    assert "Мойка section found" in context
