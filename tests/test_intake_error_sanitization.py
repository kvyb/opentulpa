from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from opentulpa.intake.decision_maker import DecisionMaker
from opentulpa.intake.messaging_adapters import (
    ComposioInstagramMessagingAdapter,
    messaging_adapter_context,
)
from opentulpa.intake.service import IntakeWorkflowService

_PRIVATE_ERROR = "provider token=private-secret from /srv/private/.env"


class _FailingAgent:
    async def decide_intake(self, **_: Any) -> Any:
        raise RuntimeError(_PRIVATE_ERROR)


class _DecisionService:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.agent = _FailingAgent()

    def _get_intake_agent(self) -> _FailingAgent:
        return self.agent

    def _normalize_conversation_messages(self, **_: Any) -> list[dict[str, Any]]:
        return [{"id": "message-1", "text": "hello"}]

    def _unanswered_customer_messages(self, messages: list[dict[str, Any]]) -> list[Any]:
        return messages

    def _emit_observability(self, **kwargs: Any) -> None:
        self.events.append(kwargs)

    def _intake_thread_id(self, **_: Any) -> str:
        return "intake-thread"

    def _intake_trace_id(self, **_: Any) -> str:
        return "intake-trace"


@pytest.mark.asyncio
async def test_decision_model_exception_is_logged_but_returned_as_generic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = _DecisionService()
    maker = DecisionMaker(service)
    caplog.set_level(logging.ERROR)

    decision, error = await maker.decide_workflow_action(
        workflow={
            "workflow_id": "workflow-1",
            "customer_id": "tenant-a",
            "name": "Lead capture",
            "required_fields": ["email"],
        },
        conversation_summary={"conversation_id": "conversation-1"},
        conversation={},
        active_booking=None,
        recent_completed_booking=None,
    )

    assert decision == {}
    assert error == "intake decision could not be completed"
    assert _PRIVATE_ERROR in caplog.text
    assert all(_PRIVATE_ERROR not in str(event) for event in service.events)


def test_provider_source_exception_is_logged_but_result_is_generic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingComposio:
        enabled = True

        def list_instagram_conversations(self, **_: Any) -> Any:
            raise RuntimeError(_PRIVATE_ERROR)

    adapter = ComposioInstagramMessagingAdapter(composio=FailingComposio())
    context = messaging_adapter_context(
        {
            "name": "Instagram",
            "customer_id": "tenant-a",
            "channel": "instagram_dm",
            "provider": "composio",
        }
    )
    caplog.set_level(logging.ERROR)

    result = adapter.list_source_items(context=context)

    assert result.error == "intake source is temporarily unavailable"
    assert _PRIVATE_ERROR not in str(result)
    assert _PRIVATE_ERROR in caplog.text


@pytest.mark.asyncio
async def test_telegram_owner_notification_never_contains_internal_summary(
    tmp_path: Path,
) -> None:
    class Client:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        async def send_message(self, **kwargs: Any) -> dict[str, Any]:
            self.messages.append(kwargs)
            return {"ok": True}

    class TelegramBusiness:
        def __init__(self) -> None:
            self.client = Client()

    telegram = TelegramBusiness()
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite",
        project_root=tmp_path,
        telegram_business=telegram,
    )

    await service._notify_pending_run_owner(  # noqa: SLF001
        owner_chat_id="777",
        summary=_PRIVATE_ERROR,
    )

    assert telegram.client.messages[0]["text"] == (
        "Telegram Business intake could not process an update. Check server logs."
    )
    assert _PRIVATE_ERROR not in str(telegram.client.messages)
