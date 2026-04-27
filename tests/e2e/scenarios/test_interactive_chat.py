from __future__ import annotations

import pytest
from harness.runner import E2EHarness, load_jsonl

pytestmark = [pytest.mark.e2e, pytest.mark.live_llm]


@pytest.mark.telegram
def test_live_internal_chat_multi_turn_continuity(e2e_harness: E2EHarness) -> None:
    customer_id = "cust_e2e_chat"
    thread_id = "thread_e2e_chat_1"

    first = e2e_harness.post_chat(
        customer_id=customer_id,
        thread_id=thread_id,
        text="My name is Jamie. Please confirm you got this and ask one concise follow-up question.",
    )
    assert first["status_code"] == 200
    assert first["payload"].get("ok") is True
    assert str(first["payload"].get("text", "")).strip()

    second = e2e_harness.post_chat(
        customer_id=customer_id,
        thread_id=thread_id,
        text="Follow-up: what name did I just give you?",
    )
    assert second["status_code"] == 200
    assert second["payload"].get("ok") is True
    second_text = str(second["payload"].get("text", "")).lower()
    assert "jamie" in second_text

    behavior = load_jsonl(e2e_harness.behavior_log_path)
    traces = load_jsonl(e2e_harness.llm_trace_path)
    assert behavior
    assert traces

    report = e2e_harness.write_status_report(
        scenario="live_internal_chat_multi_turn_continuity",
        ok=True,
        details={
            "thread_id": thread_id,
            "first_elapsed_ms": first["elapsed_ms"],
            "second_elapsed_ms": second["elapsed_ms"],
            "behavior_events": len(behavior),
            "llm_traces": len(traces),
        },
    )
    assert report.exists()
