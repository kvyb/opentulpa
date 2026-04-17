from __future__ import annotations

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
