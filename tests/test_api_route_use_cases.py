from __future__ import annotations

from typing import Any

import pytest

from opentulpa.api.routes.file_use_cases import FileRouteUseCases
from opentulpa.api.routes.intake_use_cases import workflow_upsert_kwargs
from opentulpa.api.routes.user_context_use_cases import needs_model_processing


class _Vault:
    def __init__(self) -> None:
        self.record = {
            "id": "file_1",
            "original_filename": "report.txt",
            "kind": "document",
            "mime_type": "text/plain",
        }
        self.summary: tuple[str, str, str] | None = None

    def search(self, customer_id: str, *, query: str, limit: int) -> list[dict[str, Any]]:
        assert customer_id == "cust_1"
        assert query == "report"
        assert limit == 2
        return [self.record]

    def get_file(self, customer_id: str, file_id: str) -> dict[str, Any] | None:
        assert customer_id == "cust_1"
        return self.record if file_id == "file_1" else None

    def read_file_bytes(self, customer_id: str, file_id: str) -> bytes | None:
        assert customer_id == "cust_1"
        return b"hello" if file_id == "file_1" else None

    def set_ai_summary(self, customer_id: str, file_id: str, summary: str) -> dict[str, Any]:
        self.summary = (customer_id, file_id, summary)
        return {**self.record, "summary": summary}


class _TelegramChat:
    def find_session_slots(self, customer_id: str) -> list[dict[str, Any]]:
        assert customer_id == "cust_1"
        return [{"chat_id": 123}]


class _TelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def send_file(self, **kwargs: Any) -> bool:
        self.sent.append(kwargs)
        return True


class _Runtime:
    async def analyze_uploaded_file(self, **kwargs: Any) -> dict[str, Any]:
        assert kwargs["raw_bytes"] == b"hello"
        return {"analysis": "short summary"}


@pytest.mark.asyncio
async def test_file_route_use_cases_send_uses_vault_and_telegram() -> None:
    vault = _Vault()
    telegram = _TelegramClient()
    use_cases = FileRouteUseCases(
        get_file_vault=lambda: vault,
        get_telegram_chat=_TelegramChat,
        get_telegram_client=lambda: telegram,
        get_agent_runtime=_Runtime,
        telegram_enabled=True,
    )

    response = await use_cases.send({"customer_id": "cust_1", "file_id": "file_1"})

    assert response["ok"] is True
    assert response["chat_id"] == 123
    assert telegram.sent[0]["filename"] == "report.txt"


@pytest.mark.asyncio
async def test_file_route_use_cases_analyze_updates_summary_without_question() -> None:
    vault = _Vault()
    use_cases = FileRouteUseCases(
        get_file_vault=lambda: vault,
        get_telegram_chat=_TelegramChat,
        get_telegram_client=_TelegramClient,
        get_agent_runtime=_Runtime,
        telegram_enabled=True,
    )

    response = await use_cases.analyze({"customer_id": "cust_1", "file_id": "file_1"})

    assert response["analysis"] == "short summary"
    assert vault.summary == ("cust_1", "file_1", "short summary")


def test_workflow_upsert_kwargs_preserves_internal_schedule_none_quirk() -> None:
    kwargs = workflow_upsert_kwargs(
        {
            "name": "Lead intake",
            "intent_description": "Book visits",
            "required_fields": ["date"],
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/leads.csv"},
            "schedule": None,
        },
        customer_id="cust_1",
    )

    assert kwargs["schedule"] == "None"


def test_needs_model_processing_detects_media_and_pdf_only() -> None:
    assert needs_model_processing({"kind": "photo"})
    assert needs_model_processing({"mime_type": "application/pdf"})
    assert not needs_model_processing({"original_filename": "notes.txt", "mime_type": "text/plain"})
