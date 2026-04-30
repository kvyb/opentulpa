from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from opentulpa.api.routes.telegram_webhook import register_telegram_webhook_routes
from opentulpa.interfaces.telegram.business import TelegramBusinessService


class _RecordingTelegramClient:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
        **kwargs,
    ) -> bool:
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                **kwargs,
            }
        )
        return True


class _FakeTelegramChat:
    async def handle_update(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return None

    def touch_assistant_message(self, _chat_id: int) -> None:
        return None


class _FakeApprovals:
    async def flush_deferred_challenges(self, **kwargs):  # type: ignore[no-untyped-def]
        _ = kwargs
        return 0


class _FakeIntakeWorkflows:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, str]] = []

    def list_workflows(self, *, customer_id: str, include_disabled: bool = False):  # type: ignore[no-untyped-def]
        _ = include_disabled
        return [
            {
                "workflow_id": "iwf_1",
                "customer_id": customer_id,
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
                "source_config": {"business_connection_id": "bc_123"},
            }
        ]

    def _source_matches_workflow(self, *, workflow, business_connection_id: str, conversation_id: str):  # type: ignore[no-untyped-def]
        _ = conversation_id
        return str(workflow.get("source_config", {}).get("business_connection_id", "")) == business_connection_id

    async def run_workflow(self, *, customer_id: str, workflow_id: str, event_type: str = "manual"):  # type: ignore[no-untyped-def]
        self.run_calls.append(
            {
                "customer_id": customer_id,
                "workflow_id": workflow_id,
                "event_type": event_type,
            }
        )
        return {"ok": False, "summary": "send failed"}


def _wait_for_webhook_tasks(app: FastAPI, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not getattr(app.state, "telegram_webhook_tasks", set()):
            return
        time.sleep(0.01)
    raise AssertionError("telegram webhook background task did not finish")


def test_business_message_webhook_triggers_matching_workflow_and_notifies_owner(tmp_path: Path) -> None:
    app = FastAPI()
    telegram_client = _RecordingTelegramClient()
    telegram_business = TelegramBusinessService(db_path=tmp_path / "telegram_business.db")
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    intake = _FakeIntakeWorkflows()
    settings = SimpleNamespace(
        telegram_bot_token="bot-token",
        telegram_webhook_secret="secret-token",
        telegram_allowed_user_ids=None,
        telegram_allowed_usernames=None,
    )

    register_telegram_webhook_routes(
        app,
        settings=settings,
        get_telegram_client=lambda: telegram_client,
        get_telegram_business=lambda: telegram_business,
        get_intake_workflows=lambda: intake,
        get_telegram_chat=lambda: _FakeTelegramChat(),
        get_agent_runtime=lambda: object(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/webhook/telegram",
            json={
                "update_id": 1,
                "business_message": {
                    "business_connection_id": "bc_123",
                    "message_id": 10,
                    "date": 1_775_552_400,
                    "chat": {"id": 555, "type": "private", "username": "alice"},
                    "from": {"id": 999, "is_bot": False, "username": "alice"},
                    "text": "Can I book 3pm?",
                },
            },
            headers={"x-telegram-bot-api-secret-token": "secret-token"},
        )
        _wait_for_webhook_tasks(app)

    assert response.status_code == 200
    assert intake.run_calls == [
        {
            "customer_id": "telegram_123",
            "workflow_id": "iwf_1",
            "event_type": "telegram_business_webhook",
        }
    ]
    assert telegram_client.messages[0]["chat_id"] == "777"
    assert "Telegram Business workflow issue: send failed" in str(telegram_client.messages[0]["text"])
