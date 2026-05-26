from __future__ import annotations

import pytest

from opentulpa.intake.messaging_adapters import (
    AdapterRegistry,
    ComposioInstagramMessagingAdapter,
    ConversationLoadResult,
    ConversationSummary,
    MessagingAdapterContext,
    SourceItemsResult,
    build_messaging_adapter_registry,
    messaging_adapter_context,
)


class _FakeComposio:
    enabled = True

    def list_instagram_conversations(
        self,
        *,
        customer_id: str,
        connected_account_id: str | None,
        limit: int,
    ) -> dict[str, object]:
        assert customer_id == "cust_123"
        assert connected_account_id == "ca_123"
        assert limit == 5
        return {
            "items": [
                {
                    "conversation_id": 123,
                    "recipient_id": 456,
                    "latest_inbound_message_id": 789,
                    "matched": True,
                    "participant_ids": ["u_1", "u_2"],
                }
            ]
        }


class _FakeMessagingAdapter:
    channel = "whatsapp_dm"
    provider = "hermes"

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def list_source_items(self, *, context: MessagingAdapterContext) -> SourceItemsResult:
        assert context.key == ("whatsapp_dm", "hermes")
        assert context.customer_id == "cust_123"
        return SourceItemsResult(
            items=[
                {
                    "conversation_id": "wa_1",
                    "recipient_id": "recipient_1",
                    "latest_inbound_message_id": "msg_1",
                    "latest_inbound_message_text_preview": "hello",
                }
            ]
        )

    def load_conversation(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_id: str,
    ) -> ConversationLoadResult:
        assert context.provider == self.provider
        assert conversation_id == "wa_1"
        return ConversationLoadResult(
            summary={"conversation_id": conversation_id, "recipient_id": "recipient_1"},
            conversation={"messages": [{"id": "msg_1", "text": "hello"}]},
        )

    async def send_reply(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_summary: ConversationSummary,
        reply_text: str,
    ) -> str | None:
        assert context.channel == self.channel
        self.sent.append(
            {
                "conversation_id": str(conversation_summary.get("conversation_id", "")),
                "reply_text": reply_text,
            }
        )
        return None


def test_default_messaging_adapter_registry_keys_are_explicit() -> None:
    registry = build_messaging_adapter_registry(composio=None, telegram_business=None)

    assert sorted(registry) == [
        ("instagram_dm", "composio"),
        ("telegram_business_dm", "telegram_bot_api"),
    ]


def test_messaging_adapter_context_normalizes_workflow_identity() -> None:
    context = messaging_adapter_context(
        {
            "name": " Inbox ",
            "customer_id": " cust_123 ",
            "channel": " Instagram_DM ",
            "provider": " Composio ",
            "source_config": {"scan_limit": 5},
        }
    )

    assert context.workflow_name == "Inbox"
    assert context.customer_id == "cust_123"
    assert context.key == ("instagram_dm", "composio")
    assert context.source_config == {"scan_limit": 5}


def test_unsupported_provider_has_no_adapter() -> None:
    registry = build_messaging_adapter_registry(composio=None, telegram_business=None)
    context = messaging_adapter_context(
        {
            "name": "WhatsApp",
            "customer_id": "cust_123",
            "channel": "whatsapp_dm",
            "provider": "hermes",
        }
    )

    assert registry.get(context.key) is None


def test_composio_adapter_normalizes_summary_contract() -> None:
    context = messaging_adapter_context(
        {
            "name": "Instagram",
            "customer_id": "cust_123",
            "channel": "instagram_dm",
            "provider": "composio",
            "source_config": {"connected_account_id": "ca_123", "scan_limit": 5},
        }
    )
    adapter = ComposioInstagramMessagingAdapter(composio=_FakeComposio())

    result = adapter.list_source_items(context=context)

    assert result.error is None
    assert result.items == [
        {
            "conversation_id": "123",
            "recipient_id": "456",
            "latest_inbound_message_id": "789",
            "matched": True,
            "participant_ids": ["u_1", "u_2"],
        }
    ]


@pytest.mark.asyncio
async def test_fake_adapter_exercises_shared_contract() -> None:
    fake = _FakeMessagingAdapter()
    registry: AdapterRegistry = {(fake.channel, fake.provider): fake}
    context = messaging_adapter_context(
        {
            "name": "WhatsApp",
            "customer_id": "cust_123",
            "channel": "whatsapp_dm",
            "provider": "hermes",
        }
    )

    adapter = registry[context.key]
    source = adapter.list_source_items(context=context)
    loaded = adapter.load_conversation(context=context, conversation_id="wa_1")
    error = await adapter.send_reply(
        context=context,
        conversation_summary=loaded.summary,
        reply_text="reply",
    )

    assert source.items == [
        {
            "conversation_id": "wa_1",
            "recipient_id": "recipient_1",
            "latest_inbound_message_id": "msg_1",
            "latest_inbound_message_text_preview": "hello",
        }
    ]
    assert loaded.conversation == {"messages": [{"id": "msg_1", "text": "hello"}]}
    assert error is None
    assert fake.sent == [{"conversation_id": "wa_1", "reply_text": "reply"}]
