from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from opentulpa.intake.messaging_adapters import (
    ConversationLoadResult,
    ConversationSummary,
    MessagingAdapterContext,
    SourceItemsResult,
)
from opentulpa.intake.service import IntakeWorkflowService


class _ReplyAdapter:
    channel = "instagram_dm"
    provider = "composio"

    def __init__(self, *, error: str | None = None) -> None:
        self.error = error
        self.calls: list[str] = []

    def list_source_items(self, *, context: MessagingAdapterContext) -> SourceItemsResult:
        return SourceItemsResult(items=[])

    def load_conversation(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_id: str,
    ) -> ConversationLoadResult:
        return ConversationLoadResult(summary={}, conversation={})

    async def send_reply(
        self,
        *,
        context: MessagingAdapterContext,
        conversation_summary: ConversationSummary,
        reply_text: str,
    ) -> str | None:
        self.calls.append(reply_text)
        return self.error


def _service(tmp_path: Path, adapter: _ReplyAdapter) -> IntakeWorkflowService:
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.sqlite3",
        project_root=tmp_path,
    )
    service._messaging_adapters[(adapter.channel, adapter.provider)] = adapter  # noqa: SLF001
    return service


def _workflow() -> dict[str, Any]:
    return {
        "workflow_id": "workflow_1",
        "customer_id": "tenant_1",
        "name": "Leads",
        "channel": "instagram_dm",
        "provider": "composio",
        "source_config": {},
    }


def _summary() -> dict[str, Any]:
    return {
        "conversation_id": "conversation_1",
        "latest_inbound_message_id": "message_1",
    }


@pytest.mark.asyncio
async def test_successful_reply_is_not_sent_again_after_restart(tmp_path: Path) -> None:
    first_adapter = _ReplyAdapter()
    first = _service(tmp_path, first_adapter)

    assert await first._send_intake_reply(  # noqa: SLF001
        workflow=_workflow(),
        conversation_summary=_summary(),
        reply_text="Hello",
    ) is None
    assert first_adapter.calls == ["Hello"]

    restarted_adapter = _ReplyAdapter()
    restarted = _service(tmp_path, restarted_adapter)
    assert await restarted._send_intake_reply(  # noqa: SLF001
        workflow=_workflow(),
        conversation_summary=_summary(),
        reply_text="Hello",
    ) is None
    assert restarted_adapter.calls == []


@pytest.mark.asyncio
async def test_indeterminate_reply_attempt_fails_closed_on_retry(tmp_path: Path) -> None:
    adapter = _ReplyAdapter(error="provider timed out")
    service = _service(tmp_path, adapter)

    assert await service._send_intake_reply(  # noqa: SLF001
        workflow=_workflow(),
        conversation_summary=_summary(),
        reply_text="Hello",
    ) == "provider timed out"
    assert await service._send_intake_reply(  # noqa: SLF001
        workflow=_workflow(),
        conversation_summary=_summary(),
        reply_text="Hello",
    ) is None
    assert await service._send_intake_reply(  # noqa: SLF001
        workflow=_workflow(),
        conversation_summary=_summary(),
        reply_text="Different reply",
    ) is None
    assert adapter.calls == ["Hello"]


@pytest.mark.asyncio
async def test_reply_requires_durable_inbound_identity(tmp_path: Path) -> None:
    adapter = _ReplyAdapter()
    service = _service(tmp_path, adapter)

    error = await service._send_intake_reply(  # noqa: SLF001
        workflow=_workflow(),
        conversation_summary={"conversation_id": "conversation_1"},
        reply_text="Hello",
    )

    assert error is not None
    assert "durable" in error
    assert adapter.calls == []
