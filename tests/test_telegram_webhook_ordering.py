from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.telegram_webhook import register_telegram_webhook_routes


class _OrderRecorderTelegramClient:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> bool:
        _ = (chat_id, text, parse_mode)
        self.order.append("send_message")
        return True


class _OrderRecorderApprovals:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def flush_deferred_challenges(
        self,
        *,
        origin_interface: str,
        origin_conversation_id: str,
    ) -> int:
        _ = (origin_interface, origin_conversation_id)
        self.order.append("flush_deferred_challenges")
        return 1


class _FakeTelegramChat:
    async def handle_update(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return "Optional assistant follow-up"

    def touch_assistant_message(self, _chat_id: int) -> None:
        return None


class _FakeTelegramBusiness:
    def ingest_update(self, body: dict[str, object]) -> dict[str, object]:
        _ = body
        return {"handled": False}


class _FakeIntakeWorkflows:
    def list_workflows(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return []


def test_webhook_flushes_approval_challenges_before_optional_follow_up_message() -> None:
    app = FastAPI()
    call_order: list[str] = []
    client = _OrderRecorderTelegramClient(call_order)
    approvals = _OrderRecorderApprovals(call_order)
    chat = _FakeTelegramChat()
    settings = SimpleNamespace(
        telegram_bot_token="bot-token",
        telegram_webhook_secret="secret-token",
        telegram_allowed_user_ids=None,
        telegram_allowed_usernames=None,
    )

    register_telegram_webhook_routes(
        app,
        settings=settings,
        get_telegram_client=lambda: client,
        get_telegram_business=lambda: _FakeTelegramBusiness(),
        get_intake_workflows=lambda: _FakeIntakeWorkflows(),
        get_telegram_chat=lambda: chat,
        get_approvals=lambda: approvals,
        get_agent_runtime=lambda: object(),
        get_approval_execution_orchestrator=lambda: object(),
        decide_approval_and_maybe_wake=lambda **kwargs: {},  # type: ignore[return-value]
    )
    test_client = TestClient(app)
    response = test_client.post(
        "/webhook/telegram",
        json={
            "update_id": 1,
            "message": {
                "message_id": 11,
                "date": 1700000000,
                "chat": {"id": 42, "type": "private"},
                "from": {"id": 42, "is_bot": False},
                "text": "hello",
            },
        },
        headers={"x-telegram-bot-api-secret-token": "secret-token"},
    )
    assert response.status_code == 200
    assert call_order == [
        "flush_deferred_challenges",
        "send_message",
    ]
