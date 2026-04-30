from __future__ import annotations

from opentulpa.interfaces.telegram.client import parse_telegram_callback_query


def test_parse_telegram_callback_query_extracts_fields() -> None:
    callback_id, user_id, chat_id, data, message_id = parse_telegram_callback_query(
        {
            "callback_query": {
                "id": "cbq_1",
                "from": {"id": 123},
                "message": {"message_id": 77, "chat": {"id": 456}},
                "data": "menu:status",
            }
        }
    )
    assert callback_id == "cbq_1"
    assert user_id == 123
    assert chat_id == 456
    assert data == "menu:status"
    assert message_id == 77


def test_parse_telegram_callback_query_handles_missing() -> None:
    callback_id, user_id, chat_id, data, message_id = parse_telegram_callback_query({})
    assert callback_id is None
    assert user_id is None
    assert chat_id is None
    assert data is None
    assert message_id is None
