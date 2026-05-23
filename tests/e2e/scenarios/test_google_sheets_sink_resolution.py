from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from harness.runner import E2EHarness

from opentulpa.intake import service as intake_service_module
from tests.test_intake_workflow_service import (
    _FakeComposio,
    _FakeRuntime,
    _instagram_conversation,
    _mk_service,
    _telegram_business_inbound,
)

pytestmark = [pytest.mark.e2e, pytest.mark.telegram]


def _wait_until(predicate: Any, timeout_seconds: float = 45.0) -> bool:
    import time

    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.2)
    return bool(predicate())


def _business_message(
    *,
    business_connection_id: str,
    lead_chat_id: int,
    lead_user_id: int,
    username: str,
    message_id: int,
    text: str,
) -> dict[str, Any]:
    return {
        "update_id": int(datetime.now(UTC).timestamp() * 1000) + int(message_id),
        "business_message": {
            "business_connection_id": business_connection_id,
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": lead_chat_id, "type": "private", "username": username},
            "from": {"id": lead_user_id, "is_bot": False, "username": username},
            "text": text,
        },
    }


def _telegram_message(*, chat_id: int, user_id: int, text: str, message_id: int) -> dict[str, Any]:
    return {
        "update_id": int(datetime.now(UTC).timestamp() * 1000) + int(message_id),
        "message": {
            "message_id": message_id,
            "date": int(datetime.now(UTC).timestamp()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": f"user_{user_id}"},
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


def _telegram_owner_thread_id(*, chat_id: int) -> str:
    from opentulpa.interfaces.telegram import chat_service as chat_module

    state = chat_module.STATE_STORE.load()
    sessions = state.get("sessions") if isinstance(state, dict) else {}
    slot = sessions.get(str(chat_id)) if isinstance(sessions, dict) else {}
    if not isinstance(slot, dict):
        return ""
    return str(slot.get("thread_id", "") or "").strip()


def _workflow_setup_has_proposal(
    harness: E2EHarness,
    *,
    customer_id: str,
    thread_id: str,
) -> bool:
    response = harness.client.post(
        "/internal/intake/setup/get",
        json={"customer_id": customer_id, "thread_id": thread_id, "include_paused": True},
    )
    if response.status_code != 200:
        return False
    payload = response.json()
    session = payload.get("session")
    if not isinstance(session, dict):
        return False
    return bool(str(session.get("last_proposed_draft_hash", "") or "").strip())


def _first_sheet_row(write: dict[str, Any]) -> dict[str, Any]:
    normalized = write.get("normalized_rows")
    if isinstance(normalized, list) and normalized and isinstance(normalized[0], dict):
        return dict(normalized[0])
    args = write.get("arguments") if isinstance(write.get("arguments"), dict) else {}
    headers = args.get("headers")
    rows = args.get("rows")
    if isinstance(headers, list) and isinstance(rows, list) and rows and isinstance(rows[0], list):
        return dict(zip([str(item) for item in headers], rows[0], strict=False))
    return {}


@pytest.mark.asyncio
async def test_google_sheets_sink_resolves_single_tab_for_telegram_business_intake(
    tmp_path: Path,
) -> None:
    customer_id = "telegram_123"
    business_connection_id = "bc_sheet_resolution"
    summary = {
        "conversation_id": "unused_composio_conv",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="unused_composio_conv",
        latest_message_id="msg_1",
        latest_message_text="unused",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.96,
                "conversation_summary": "Клиент хочет записаться на мойку.",
                "extracted_fields": {
                    "клиент": "Семен",
                    "телефон клиента": "+79990000001",
                    "тип услуги": "Мойка",
                    "модель автомобиля": "Toyota RAV4",
                    "время записи": "завтра 10:00",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Записал вас на мойку завтра в 10:00.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "клиент": "Семен",
                    "телефон клиента": "+79990000001",
                    "модель автомобиля": "Toyota RAV4",
                },
                "reason": "All required fields are available.",
            }
        ]
    )
    composio = _FakeComposio(
        summary,
        conversation,
        sheet_names_by_spreadsheet={"sheet_autospa": ["Записи клиентов"]},
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )

    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="AutoSpa Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description="Записывать клиентов из Telegram Business на мойку.",
        required_fields=["клиент", "телефон клиента", "тип услуги", "модель автомобиля"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "клиент": "клиент",
                "телефон клиента": "телефон клиента",
                "тип услуги": "тип услуги",
                "модель автомобиля": "модель автомобиля",
                "время записи": "время записи",
            },
            "static_arguments": {"spreadsheet_id": "sheet_autospa"},
        },
    )

    sink_config = workflow["sink_config"]
    static_arguments = sink_config["static_arguments"]
    assert static_arguments["spreadsheetId"] == "sheet_autospa"
    assert static_arguments["sheetName"] == "Записи клиентов"

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=1,
            text="Здравствуйте, Семен, Toyota RAV4, хочу мойку завтра в 10, телефон +79990000001.",
            date=int(datetime.now(UTC).timestamp()),
        ),
    )

    result = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    bookings = service.list_bookings(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    assert bookings[0]["status"] == "completed"
    assert bookings[0]["sink_write_status"] == "succeeded"
    sink_calls = [call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"]
    assert len(sink_calls) == 1
    assert sink_calls[0]["arguments"]["sheetName"] == "Записи клиентов"
    written = dict(
        zip(
            sink_calls[0]["arguments"]["headers"],
            sink_calls[0]["arguments"]["rows"][0],
            strict=False,
        )
    )
    assert written["тип услуги"] == "Мойка"
    assert written["время записи"] == "завтра 10:00"


@pytest.mark.live_llm
def test_live_llm_telegram_business_intake_upserts_partial_identity_then_final_row(
    e2e_harness: E2EHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 0.02)

    owner_user_id = 321
    owner_chat_id = 654
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = "bc_e2e_partial_identity"
    lead_chat_id = 7001
    lead_user_id = 9001
    lead_username = "partial_lead_9001"

    e2e_harness.client.app.state.telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": owner_chat_id,
            "is_enabled": True,
            "user": {"id": owner_user_id, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    assert (
        e2e_harness.post_telegram(
            body=_telegram_message(chat_id=owner_chat_id, user_id=owner_user_id, text="/fresh", message_id=100)
        )
        == 200
    )
    assert _wait_until(
        lambda: any(
            int(item.get("chat_id", 0)) == owner_chat_id
            and "fresh chat context" in str(item.get("text", "")).lower()
            for item in e2e_harness.telegram_client.sent_messages
        ),
        timeout_seconds=45.0,
    )

    owner_message_count = len(e2e_harness.telegram_client.sent_messages)
    assert (
        e2e_harness.post_telegram(
            body=_telegram_message(
                chat_id=owner_chat_id,
                user_id=owner_user_id,
                message_id=101,
                text=(
                    "Create a Telegram Business DM intake workflow named 'Live Partial Identity Intake'. "
                    "Use the already connected Telegram Business account. "
                    "The workflow handles car wash booking leads. Required fields are exactly: service, name, phone, time. "
                    "Critical behavior: on the first inbound message from a new intake user, before all required fields are collected, "
                    "record the backend-provided incoming_user_id, username, and conversation_id in Google Sheets by using "
                    "sink_action=upsert_partial with those exact fields in sink_payload. Then ask for missing booking details. "
                    "When the lead later provides service, name, phone, and time, save the final booking normally. "
                    "Do not invent ids; use conversation.summary source identity fields. "
                    "Save to Google Sheets with sink_type google_sheets_composio, toolkit googlesheets, "
                    "spreadsheetId=sheet_partial_identity, sheetName=Bookings, and field_mapping: "
                    "incoming_user_id -> Telegram ID, username -> Username, conversation_id -> Conversation ID, "
                    "service -> Service, name -> Name, phone -> Phone, time -> Time. "
                    "Prepare the exact configuration and wait for my confirmation before saving."
                ),
            )
        )
        == 200
    )
    assert _wait_until(
        lambda: len(e2e_harness.telegram_client.sent_messages) > owner_message_count,
        timeout_seconds=90.0,
    )
    owner_thread_id = _telegram_owner_thread_id(chat_id=owner_chat_id)
    assert owner_thread_id
    if not _wait_until(
        lambda: _workflow_setup_has_proposal(
            e2e_harness,
            customer_id=customer_id,
            thread_id=owner_thread_id,
        ),
        timeout_seconds=90.0,
    ):
        assert (
            e2e_harness.post_telegram(
                body=_telegram_message(
                    chat_id=owner_chat_id,
                    user_id=owner_user_id,
                    message_id=102,
                    text=(
                        "Use the workflow setup tools now: persist that exact draft, run preflight, "
                        "mark it proposed, and wait for my confirmation before saving."
                    ),
                )
            )
            == 200
        )
        assert _wait_until(
            lambda: _workflow_setup_has_proposal(
                e2e_harness,
                customer_id=customer_id,
                thread_id=owner_thread_id,
            ),
            timeout_seconds=90.0,
        )
    assert _list_workflows(e2e_harness, customer_id=customer_id) == []
    assert (
        e2e_harness.post_telegram(
            body=_telegram_message(
                chat_id=owner_chat_id,
                user_id=owner_user_id,
                message_id=103,
                text="Looks correct. Save and activate this workflow now exactly as proposed.",
            )
        )
        == 200
    )
    assert _wait_until(
        lambda: len(_list_workflows(e2e_harness, customer_id=customer_id)) == 1,
        timeout_seconds=90.0,
    )
    workflow = _list_workflows(e2e_harness, customer_id=customer_id)[0]
    assert workflow["name"] == "Live Partial Identity Intake"
    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["enabled"] is True
    assert workflow["source_config"] == {"business_connection_id": business_connection_id}

    assert (
        e2e_harness.post_telegram(
            body=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                username=lead_username,
                message_id=1,
                text="Hi, I want to book a wash.",
            )
        )
        == 200
    )
    assert _wait_until(
        lambda: len(getattr(e2e_harness.composio_service, "sheet_writes", [])) >= 1,
        timeout_seconds=90.0,
    )
    first_write = _first_sheet_row(e2e_harness.composio_service.sheet_writes[0])
    assert first_write["Telegram ID"] == str(lead_user_id)
    assert first_write["Username"] == lead_username
    assert first_write["Conversation ID"] == str(lead_chat_id)
    bookings = e2e_harness.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id=str(lead_chat_id),
    )
    assert bookings and bookings[0]["status"] == "active"

    assert (
        e2e_harness.post_telegram(
            body=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                username=lead_username,
                message_id=2,
                text="Express wash, name Alice, phone +79990000001, tomorrow 10am.",
            )
        )
        == 200
    )
    assert _wait_until(
        lambda: any(
            str(item.get("status", "")).strip() == "completed"
            for item in e2e_harness.list_bookings(
                customer_id=customer_id,
                workflow_id=workflow["workflow_id"],
                conversation_id=str(lead_chat_id),
            )
        )
        and len(getattr(e2e_harness.composio_service, "sheet_writes", [])) >= 2,
        timeout_seconds=90.0,
    )
    second_write = _first_sheet_row(e2e_harness.composio_service.sheet_writes[-1])
    assert second_write["Booking ID"] == first_write["Booking ID"]
    assert second_write["Telegram ID"] == str(lead_user_id)
    assert second_write["Username"] == lead_username
    assert second_write["Service"]
    assert second_write["Name"]
    assert second_write["Phone"]
    assert second_write["Time"]


def test_google_sheets_sink_setup_rejects_ambiguous_multi_tab_target(tmp_path: Path) -> None:
    customer_id = "telegram_123"
    summary: dict[str, Any] = {
        "conversation_id": "unused_composio_conv",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="unused_composio_conv",
        latest_message_id="msg_1",
        latest_message_text="unused",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    composio = _FakeComposio(
        summary,
        conversation,
        sheet_names_by_spreadsheet={"sheet_autospa": ["Заявки", "Архив"]},
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=composio,
    )

    with pytest.raises(ValueError, match="multiple sheets: Заявки, Архив"):
        service.upsert_workflow(
            customer_id=customer_id,
            name="AutoSpa Telegram Intake",
            channel="instagram_dm",
            provider="composio",
            intent_description="Записывать клиентов.",
            required_fields=["тип услуги"],
            sink_type="google_sheets_composio",
            sink_config={
                "toolkit": "googlesheets",
                "field_mapping": {"тип услуги": "тип услуги"},
                "static_arguments": {"spreadsheetId": "sheet_autospa"},
            },
        )
