from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from harness.runner import E2EHarness


pytestmark = [pytest.mark.e2e, pytest.mark.live_llm, pytest.mark.telegram]


def _wait_until(predicate: Any, timeout_seconds: float = 45.0) -> bool:
    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.2)
    return bool(predicate())


def _seed_telegram_business_connection(
    harness: E2EHarness,
    *,
    owner_user_id: int,
    owner_chat_id: int,
    business_connection_id: str = "bc_e2e_123",
) -> str:
    telegram_business = harness.client.app.state.telegram_business
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": owner_chat_id,
            "is_enabled": True,
            "user": {
                "id": owner_user_id,
                "is_bot": False,
                "first_name": "Kim",
                "username": "kim",
            },
            "rights": {"can_reply": True},
        }
    )
    return business_connection_id


def _telegram_message(*, chat_id: int, user_id: int, text: str, message_id: int = 1) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(timezone.utc).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": f"user_{user_id}"},
            "text": text,
        },
    }


def _telegram_business_message(
    *,
    business_connection_id: str,
    lead_chat_id: int,
    lead_user_id: int,
    text: str,
    message_id: int = 100,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "business_message": {
            "business_connection_id": business_connection_id,
            "message_id": message_id,
            "date": int(datetime.now(timezone.utc).timestamp()),
            "chat": {"id": lead_chat_id, "type": "private", "username": f"lead_{lead_user_id}"},
            "from": {"id": lead_user_id, "is_bot": False, "username": f"lead_{lead_user_id}"},
            "text": text,
        },
    }


def _list_workflows(harness: E2EHarness, *, customer_id: str) -> list[dict[str, Any]]:
    response = harness.client.post(
        "/internal/intake/workflows/list",
        json={"customer_id": customer_id, "include_disabled": True},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    workflows = payload.get("workflows") or []
    return workflows if isinstance(workflows, list) else []


def _latest_message_for_chat(
    harness: E2EHarness,
    *,
    chat_id: int,
    start_index: int = 0,
) -> dict[str, Any] | None:
    for item in reversed(harness.telegram_client.sent_messages[start_index:]):
        if int(item.get("chat_id", 0)) == int(chat_id):
            return item
    return None


def _messages_for_chat(
    harness: E2EHarness,
    *,
    chat_id: int,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    return [
        item
        for item in harness.telegram_client.sent_messages[start_index:]
        if int(item.get("chat_id", 0)) == int(chat_id)
    ]


def test_live_owner_telegram_chat_can_create_telegram_intake_workflow_and_activate_it(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
    )

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=1)
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
    )

    initial_owner_message_count = len(e2e_harness.telegram_client.sent_messages)
    create_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=2,
            text=(
                "Create a Telegram Business DM intake workflow for my car wash. "
                "Use the workflow name 'E2E Telegram Car Wash'. "
                "Collect exactly these fields: car_model, car_type, wash_type, date, time. "
                "Goal: answer direct questions first, then collect only missing booking details. "
                "Save results to local CSV tulpa_stuff/e2e_telegram_carwash.csv. "
                "Start the workflow setup wizard, prepare the exact configuration, and wait for my confirmation before saving."
            ),
        )
    )
    assert create_status == 200
    assert _wait_until(lambda: len(e2e_harness.telegram_client.sent_messages) > initial_owner_message_count)
    assert _list_workflows(e2e_harness, customer_id=customer_id) == []

    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=3,
            text="Yes, save that workflow now exactly as proposed.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1, timeout_seconds=60.0)

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Telegram Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""
    assert workflow["source_config"] == {"business_connection_id": business_connection_id}
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}

    latest_owner_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=initial_owner_message_count,
    )
    assert latest_owner_message is not None
    latest_text = str(latest_owner_message.get("text", "")).lower()
    assert "backend error" not in latest_text
    assert "workflow" in latest_text

    report = e2e_harness.write_status_report(
        scenario="live_owner_telegram_chat_can_create_telegram_intake_workflow_and_activate_it",
        ok=True,
        details={
            "customer_id": customer_id,
            "workflow_id": workflow["workflow_id"],
            "owner_messages": len(e2e_harness.telegram_client.sent_messages),
        },
    )
    assert report.exists()


def test_live_owner_telegram_chat_can_delete_existing_telegram_intake_workflow(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
    )
    create = e2e_harness.client.post(
        "/internal/intake/workflows/upsert",
        json={
            "customer_id": customer_id,
            "name": "Delete Me Telegram Intake",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": business_connection_id},
            "intent_description": "Handle Telegram booking requests.",
            "required_fields": ["name", "time"],
            "assistant_instructions": "Be concise.",
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/e2e_delete_me.csv"},
            "enabled": True,
        },
    )
    assert create.status_code == 200, create.text
    assert len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1

    start_message_count = len(e2e_harness.telegram_client.sent_messages)
    delete_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=10,
            text=(
                "Delete the active Telegram Business intake workflow now. "
                "Do not just explain; perform the deletion and confirm when it is gone."
            ),
        )
    )
    assert delete_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 0, timeout_seconds=60.0)

    latest_owner_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_message_count,
    )
    assert latest_owner_message is not None
    latest_text = str(latest_owner_message.get("text", "")).lower()
    assert "backend error" not in latest_text
    assert "deleted" in latest_text or "removed" in latest_text or "gone" in latest_text

    report = e2e_harness.write_status_report(
        scenario="live_owner_telegram_chat_can_delete_existing_telegram_intake_workflow",
        ok=True,
        details={
            "customer_id": customer_id,
            "owner_messages": len(e2e_harness.telegram_client.sent_messages),
        },
    )
    assert report.exists()


def test_live_telegram_business_lead_message_triggers_active_workflow_reply(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
    )
    create = e2e_harness.client.post(
        "/internal/intake/workflows/upsert",
        json={
            "customer_id": customer_id,
            "name": "Lead Reply Telegram Intake",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": business_connection_id},
            "intent_description": "Reply to Telegram Business leads and collect booking details.",
            "required_fields": ["car_model", "car_type", "wash_type", "date", "time"],
            "assistant_instructions": (
                "Reply directly to the lead, answer what you can, ask only for missing booking fields, "
                "and keep replies concise."
            ),
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/e2e_lead_replies.csv"},
            "enabled": True,
        },
    )
    assert create.status_code == 200, create.text
    workflow = create.json()["workflow"]
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""

    start_message_count = len(e2e_harness.telegram_client.sent_messages)
    lead_chat_id = 555
    webhook_status = e2e_harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=999,
            message_id=100,
            text="Hi, I want to book a wash tomorrow at 10am for my BMW sedan.",
        )
    )
    assert webhook_status == 200

    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == lead_chat_id
            and str(item.get("business_connection_id", "")).strip() == business_connection_id
            and str(item.get("text", "")).strip()
            for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
        ),
        timeout_seconds=60.0,
    )

    lead_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=lead_chat_id,
        start_index=start_message_count,
    )
    assert lead_reply is not None
    assert str(lead_reply.get("business_connection_id", "")).strip() == business_connection_id
    assert str(lead_reply.get("text", "")).strip()

    owner_errors = [
        item
        for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
        if int(item.get("chat_id", 0)) == owner_chat_id
    ]
    assert owner_errors == []

    report = e2e_harness.write_status_report(
        scenario="live_telegram_business_lead_message_triggers_active_workflow_reply",
        ok=True,
        details={
            "customer_id": customer_id,
            "workflow_id": workflow["workflow_id"],
            "lead_chat_id": lead_chat_id,
            "lead_reply_text": str(lead_reply.get("text", ""))[:500],
        },
    )
    assert report.exists()


def test_live_owner_chat_can_create_quality_workflow_over_multiple_turns_and_handle_aligned_lead(
    e2e_harness: E2EHarness,
) -> None:
    owner_user_id = 123
    owner_chat_id = 777
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_quality",
    )

    fresh_status = e2e_harness.post_telegram(
        body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=50)
    )
    assert fresh_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        )
    )

    start_index = len(e2e_harness.telegram_client.sent_messages)
    first_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=51,
            text=(
                "I want to set up a Telegram Business DM intake workflow for my car wash. "
                "Please start the workflow setup wizard and help me shape it step by step."
            ),
        )
    )
    assert first_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=start_index)) >= 1)
    first_wizard_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_index,
    )
    assert first_wizard_reply is not None
    first_wizard_text = str(first_wizard_reply.get("text", "")).lower()
    assert "workflow" in first_wizard_text or "setup" in first_wizard_text

    second_turn_start = len(e2e_harness.telegram_client.sent_messages)
    second_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=52,
            text=(
                "Use the workflow name 'E2E Quality Car Wash'. "
                "Collect exactly: car_model, car_type, wash_type, date, time. "
                "If a lead asks for price, answer directly before asking anything else. "
                "As soon as wash_type and car_type are known, give the exact price immediately. "
                "Use these prices: small car full wash 1000 rubles, SUV full wash 2500 rubles. "
                "Do not repeat already known details. "
                "Only offer exact times like 09:00, 10:00, 11:00, not vague parts of day. "
                "Save bookings to local CSV tulpa_stuff/e2e_quality_carwash.csv."
            ),
        )
    )
    assert second_status == 200
    assert _wait_until(lambda: len(_messages_for_chat(e2e_harness, chat_id=owner_chat_id, start_index=second_turn_start)) >= 1)
    proposal_message = _latest_message_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=second_turn_start,
    )
    assert proposal_message is not None
    proposal_text = str(proposal_message.get("text", "")).lower()
    assert "confirm" in proposal_text or "save" in proposal_text or "workflow" in proposal_text

    confirm_status = e2e_harness.post_telegram(
        body=_telegram_message(
            chat_id=owner_chat_id,
            user_id=owner_user_id,
            message_id=53,
            text="Looks good. Save and activate that workflow now.",
        )
    )
    assert confirm_status == 200
    assert _wait_until(lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1, timeout_seconds=60.0)

    workflows = _list_workflows(e2e_harness, customer_id=customer_id)
    assert len(workflows) == 1
    workflow = workflows[0]
    assert workflow["name"] == "E2E Quality Car Wash"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["source_config"] == {"business_connection_id": business_connection_id}
    assert set(workflow["required_fields"]) == {"car_model", "car_type", "wash_type", "date", "time"}
    instructions = str(workflow.get("assistant_instructions", "")).strip()
    assert len(instructions) >= 120
    lowered_instructions = instructions.lower()
    assert "price" in lowered_instructions
    assert "exact" in lowered_instructions
    assert "time" in lowered_instructions
    assert "repeat" in lowered_instructions or "already known" in lowered_instructions

    lead_start_index = len(e2e_harness.telegram_client.sent_messages)
    lead_chat_id = 556
    lead_text = (
        "How much is a full wash for a small car? "
        "If 10:00 tomorrow works, book it for my BMW 3 Series."
    )
    lead_status = e2e_harness.post_telegram(
        body=_telegram_business_message(
            business_connection_id=business_connection_id,
            lead_chat_id=lead_chat_id,
            lead_user_id=1001,
            message_id=150,
            text=lead_text,
        )
    )
    assert lead_status == 200
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == lead_chat_id
            and str(item.get("business_connection_id", "")).strip() == business_connection_id
            and str(item.get("text", "")).strip()
            for item in e2e_harness.telegram_client.sent_messages[lead_start_index:]
        ),
        timeout_seconds=60.0,
    )

    lead_reply = _latest_message_for_chat(
        e2e_harness,
        chat_id=lead_chat_id,
        start_index=lead_start_index,
    )
    assert lead_reply is not None
    lead_reply_text = str(lead_reply.get("text", "")).strip()
    assert lead_reply_text

    intake_service = e2e_harness.client.app.state.intake_workflows
    bookings = intake_service.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )

    owner_messages = _messages_for_chat(
        e2e_harness,
        chat_id=owner_chat_id,
        start_index=start_index,
    )
    owner_transcript = [
        {"chat_id": int(item.get("chat_id", 0)), "text": str(item.get("text", ""))[:800]}
        for item in owner_messages
    ]
    workflow_snapshot = {
        "workflow_id": workflow["workflow_id"],
        "name": workflow["name"],
        "channel": workflow["channel"],
        "provider": workflow["provider"],
        "required_fields": workflow["required_fields"],
        "assistant_instructions": instructions[:2500],
        "sink_type": workflow["sink_type"],
        "sink_config": workflow["sink_config"],
    }
    booking_snapshot = bookings[0] if bookings else {}

    report = e2e_harness.write_status_report(
        scenario="live_owner_chat_can_create_quality_workflow_over_multiple_turns_and_handle_aligned_lead",
        ok=True,
        details={
            "customer_id": customer_id,
            "owner_transcript": owner_transcript,
            "workflow": workflow_snapshot,
            "lead_message": lead_text,
            "lead_reply_text": lead_reply_text[:1200],
            "bookings_count": len(bookings),
            "first_booking": booking_snapshot,
        },
    )
    assert report.exists()
    report_payload = json.loads(report.read_text(encoding="utf-8"))
    evaluation = report_payload.get("evaluation", {})
    verdict = str(evaluation.get("verdict", "")).strip().lower()
    assert verdict != "fail"
