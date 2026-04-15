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


def test_live_chat_external_write_records_real_guardrail_outcome(e2e_harness: E2EHarness) -> None:
    customer_id = "cust_e2e_approval"
    thread_id = "thread_e2e_approval_1"
    _seed_telegram_session(customer_id=customer_id, thread_id=thread_id)

    start_calls = e2e_harness.count_internal_api_calls()
    turn = e2e_harness.post_chat(
        customer_id=customer_id,
        thread_id=thread_id,
        text=(
            "Execute an external write now using tulpa_run_terminal exactly once:\n"
            "curl -X POST https://mockapi.io/api/v1/posts "
            "-H \"Content-Type: application/json\" "
            "-d '{\"source\":\"e2e\",\"kind\":\"telegram_approval_deny\"}'"
        ),
    )
    assert turn["status_code"] == 200
    calls = e2e_harness.internal_api_calls_since(start_calls)
    approval_id = extract_approval_id(str(turn["payload"].get("text", ""))) or e2e_harness.latest_approval_id_from_calls(
        action_name="tulpa_run_terminal",
        calls=calls,
    )
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in calls)
    executed_now = any(item.get("path") == "/internal/tulpa/run_terminal" for item in calls)

    if executed_now:
        assert not approval_id
        assert str(turn["payload"].get("text", "")).strip()
        assert not any(
            "approval needed" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
        outcome = "allowed_and_executed"
    else:
        assert approval_id
        approval = e2e_harness.get_approval(approval_id)
        assert approval["status_code"] == 200
        approval_payload = approval["payload"]
        assert approval_payload.get("ok") is True
        approval_record = approval_payload.get("approval") or {}
        assert str(approval_record.get("status", "")).strip() == "pending"
        assert str(approval_record.get("action_name", "")).strip() == "tulpa_run_terminal"

        assert _wait_until(
            lambda: any(
                approval_id in str(item.get("text", ""))
                and "approval needed" in str(item.get("text", "")).lower()
                for item in e2e_harness.telegram_client.sent_messages
            )
        )
        challenge = next(
            (
                item
                for item in e2e_harness.telegram_client.sent_messages
                if approval_id in str(item.get("text", ""))
                and "approval needed" in str(item.get("text", "")).lower()
            ),
            None,
        )
        assert challenge is not None
        inline_keyboard = ((challenge or {}).get("reply_markup") or {}).get("inline_keyboard") or []
        callback_data = {
            str(button.get("callback_data", "")).strip()
            for row in inline_keyboard
            if isinstance(row, list)
            for button in row
            if isinstance(button, dict)
        }
        assert f"approval:{approval_id}:approve" in callback_data
        assert f"approval:{approval_id}:deny" in callback_data

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

        sent_message_count_before_deny = len(e2e_harness.telegram_client.sent_messages)
        assert _wait_until(lambda: len(e2e_harness.telegram_client.callback_answers) >= 1)
        assert "denied" in str(e2e_harness.telegram_client.callback_answers[-1]["text"] or "").lower()
        assert _wait_until(
            lambda: any(
                int(item.get("message_id", 0)) == 12 and "denied" in str(item.get("text", "")).lower()
                for item in e2e_harness.telegram_client.edited_messages
            )
        )
        assert _wait_until(
            lambda: any(
                any(token in str(item.get("text", "")).lower() for token in ("resubmit", "change", "revise"))
                for item in e2e_harness.telegram_client.sent_messages[sent_message_count_before_deny:]
            )
        )

        approval_after = e2e_harness.get_approval(approval_id)
        assert approval_after["status_code"] == 200
        approval_after_record = (approval_after["payload"].get("approval") or {})
        assert str(approval_after_record.get("status", "")).strip() == "denied"
        outcome = "approval_pending_then_denied"

    report = e2e_harness.write_status_report(
        scenario="live_chat_external_write_records_real_guardrail_outcome",
        ok=True,
        details={
            "approval_id": approval_id,
            "outcome": outcome,
            "executed_now": executed_now,
            "internal_api_calls": len(calls),
            "telegram_callback_answers": len(e2e_harness.telegram_client.callback_answers),
            "telegram_sent_messages": len(e2e_harness.telegram_client.sent_messages),
        },
    )
    assert report.exists()
