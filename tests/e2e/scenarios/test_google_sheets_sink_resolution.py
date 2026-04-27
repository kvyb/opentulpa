from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.test_intake_workflow_service import (
    _FakeComposio,
    _FakeRuntime,
    _instagram_conversation,
    _mk_service,
    _telegram_business_inbound,
)

pytestmark = [pytest.mark.e2e, pytest.mark.telegram]


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
