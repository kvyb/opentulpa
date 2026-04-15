from __future__ import annotations

from pathlib import Path

from opentulpa.interfaces.telegram.business import TelegramBusinessService


def test_telegram_business_service_persists_connection_and_message_state(tmp_path: Path) -> None:
    service = TelegramBusinessService(db_path=tmp_path / "telegram_business.db")
    connection = service.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )

    assert connection["customer_id"] == "telegram_123"
    status = service.status(customer_id="telegram_123")
    assert status["connected"] is True
    assert status["connections"][0]["business_connection_id"] == "bc_123"

    ingested = service.ingest_update(
        {
            "business_message": {
                "business_connection_id": "bc_123",
                "message_id": 10,
                "date": 1_775_552_400,
                "chat": {"id": 555, "type": "private", "username": "alice"},
                "from": {"id": 999, "is_bot": False, "username": "alice"},
                "text": "Can I book 3pm?",
            }
        }
    )
    assert ingested["handled"] is True
    assert ingested["trigger_workflows"] is True

    conversations = service.list_conversations(
        customer_id="telegram_123",
        business_connection_id="bc_123",
    )
    assert conversations["items"][0]["conversation_id"] == "555"
    assert conversations["items"][0]["latest_inbound_message_id"] == "10"
