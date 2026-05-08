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
    assert conversations["items"][0]["latest_inbound_sender_id"] == "999"


def test_telegram_business_service_can_bind_connection_to_configured_owner(
    tmp_path: Path,
) -> None:
    service = TelegramBusinessService(
        db_path=tmp_path / "telegram_business.db",
        owner_customer_id="telegram_83969136",
    )

    connection = service.upsert_connection(
        {
            "id": "bc_owner",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 6907589464, "is_bot": False, "first_name": "Business"},
            "rights": {"can_reply": True},
        }
    )

    assert connection["customer_id"] == "telegram_83969136"
    assert service.status(customer_id="telegram_83969136")["connected"] is True
    assert service.status(customer_id="telegram_6907589464")["connected"] is False


def test_telegram_business_service_rebinds_existing_connections_to_configured_owner(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "telegram_business.db"
    service = TelegramBusinessService(db_path=db_path)
    service.upsert_connection(
        {
            "id": "bc_existing",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 6907589464, "is_bot": False, "first_name": "Business"},
            "rights": {"can_reply": True},
        }
    )
    service.ingest_update(
        {
            "business_message": {
                "business_connection_id": "bc_existing",
                "message_id": 10,
                "date": 1_775_552_400,
                "chat": {"id": 555, "type": "private"},
                "from": {"id": 999, "is_bot": False},
                "text": "Hello",
            }
        }
    )

    rebound = TelegramBusinessService(
        db_path=db_path,
        owner_customer_id="telegram_83969136",
    )

    assert rebound.status(customer_id="telegram_83969136")["connected"] is True
    assert rebound.status(customer_id="telegram_6907589464")["connected"] is False
    conversations = rebound.list_conversations(
        customer_id="telegram_83969136",
        business_connection_id="bc_existing",
    )
    assert conversations["items"][0]["conversation_id"] == "555"
