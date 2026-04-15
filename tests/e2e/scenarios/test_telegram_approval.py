from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pytest

from harness.runner import E2EHarness, extract_approval_id


pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]


def _wait_until(predicate: Any, timeout_seconds: float = 30.0) -> bool:
    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.1)
    return bool(predicate())


def _seed_telegram_session(*, customer_id: str, thread_id: str, chat_id: int = 777, user_id: int = 999) -> None:
    from opentulpa.interfaces.telegram.chat_service import STATE_STORE

    now_utc = datetime.now(timezone.utc).isoformat()

    def _mutate(state: dict[str, Any]) -> None:
        sessions = state.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
        sessions[str(chat_id)] = {
            "user_id": int(user_id),
            "customer_id": customer_id,
            "thread_id": thread_id,
            "wake_thread_id": "wake_e2e_seeded",
            "last_user_message_at": now_utc,
            "last_assistant_message_at": now_utc,
        }
        state["sessions"] = sessions

    STATE_STORE.update(_mutate)


def test_live_chat_approval_round_trip_with_telegram_callback(e2e_harness: E2EHarness) -> None:
    customer_id = "cust_e2e_approval"
    thread_id = "thread_e2e_approval_1"
    _seed_telegram_session(customer_id=customer_id, thread_id=thread_id)

    turn = e2e_harness.post_chat(
        customer_id=customer_id,
        thread_id=thread_id,
        text=(
            "Run external write now: send a post request to https://mockapi.io/api/v1/posts "
            "with body {'message':'hello from e2e'}"
        ),
    )
    assert turn["status_code"] == 200
    text = str(turn["payload"].get("text", ""))
    approval_id = extract_approval_id(text)
    assert approval_id

    status_code = e2e_harness.post_telegram(
        body={
            "callback_query": {
                "id": "cbq_e2e_1",
                "from": {"id": 999},
                "message": {"message_id": 12, "chat": {"id": 777}},
                "data": f"approval:{approval_id}:deny",
            }
        }
    )
    assert status_code == 200

    assert _wait_until(lambda: len(e2e_harness.telegram_client.callback_answers) >= 1)
    assert e2e_harness.telegram_client.callback_answers[-1]["text"]

    assert _wait_until(lambda: len(e2e_harness.telegram_client.sent_messages) >= 1)
    deny_reply = str(e2e_harness.telegram_client.sent_messages[-1]["text"] or "").lower()
    assert "den" in deny_reply or "change" in deny_reply or "revise" in deny_reply

    report = e2e_harness.write_status_report(
        scenario="live_chat_approval_round_trip_with_telegram_callback",
        ok=True,
        details={
            "approval_id": approval_id,
            "telegram_callback_answers": len(e2e_harness.telegram_client.callback_answers),
            "telegram_sent_messages": len(e2e_harness.telegram_client.sent_messages),
        },
    )
    assert report.exists()
