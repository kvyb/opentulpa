from __future__ import annotations

from typing import Any

import pytest

from opentulpa.interfaces.telegram.business_relay import TelegramBusinessRelay


class _Business:
    def __init__(self) -> None:
        self.ingested: list[dict[str, Any]] = []
        self.completed: list[str] = []

    def ingest_update(self, body: dict[str, Any]) -> dict[str, Any]:
        self.ingested.append(body)
        return {
            "handled": True,
            "trigger_workflows": True,
            "dispatch_pending": True,
            "ingress_key": "ingress-1",
            "customer_id": "tenant-a",
            "business_connection_id": "connection-1",
            "chat_id": "chat-1",
            "user_chat_id": "owner-chat",
        }

    def complete_ingress(self, ingress_key: str) -> None:
        self.completed.append(ingress_key)


class _Workflows:
    def __init__(self) -> None:
        self.enqueued: list[dict[str, Any]] = []
        self.drains = 0

    def list_workflows(
        self,
        *,
        customer_id: str,
        include_disabled: bool,
    ) -> list[dict[str, Any]]:
        assert customer_id == "tenant-a"
        assert include_disabled is False
        return [
            {
                "workflow_id": "workflow-1",
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
            }
        ]

    def _source_matches_workflow(self, **_: Any) -> bool:
        return True

    async def enqueue_telegram_business_workflow_run(
        self,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.enqueued.append(kwargs)
        return {"ok": True, "queued": True}

    async def drain_due_pending_runs(self, *, limit: int = 10) -> int:
        assert limit == 10
        self.drains += 1
        return 0


@pytest.mark.asyncio
async def test_business_relay_only_ingests_business_updates() -> None:
    business = _Business()
    workflows = _Workflows()
    relay = TelegramBusinessRelay(business=business, workflows=workflows)

    ignored = await relay.accept_update(
        {"update_id": 1, "message": {"text": "owner request"}}
    )
    accepted = await relay.accept_update(
        {"update_id": 2, "business_message": {"message_id": 10}}
    )
    await relay.process_update(accepted)

    assert ignored.result == {"handled": False}
    assert business.ingested == [
        {"update_id": 2, "business_message": {"message_id": 10}}
    ]
    assert business.completed == ["ingress-1"]
    assert workflows.enqueued == [
        {
            "customer_id": "tenant-a",
            "workflow_id": "workflow-1",
            "conversation_id": "chat-1",
            "owner_chat_id": "owner-chat",
            "event_type": "telegram_business_webhook",
        }
    ]
    assert workflows.drains == 1
