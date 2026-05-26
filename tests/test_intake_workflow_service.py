from __future__ import annotations

import asyncio
import csv
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from opentulpa.agent.knowledge_prep import inspect_uploaded_file_structure
from opentulpa.business_knowledge.service import BusinessKnowledgeService
from opentulpa.context.file_vault import FileVaultService
from opentulpa.intake import service as intake_service_module
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.intake.workflow_boundaries import (
    ConversationCursorSignals,
    DecisionActions,
    WorkflowRunAccumulator,
)
from opentulpa.interfaces.telegram.business import TelegramBusinessService
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService
from tests.workbook_fixtures import (
    SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
    SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
    sample_vehicle_services_xlsx_bytes,
)


@pytest.fixture(autouse=True)
def _freeze_intake_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 0, 30, tzinfo=UTC),
    )


def _instagram_conversation(
    *,
    conversation_id: str,
    latest_message_id: str,
    latest_message_text: str,
    latest_message_time: str,
    latest_message_sender_id: str = "cust_1",
    latest_message_sender_username: str = "alice",
) -> dict[str, Any]:
    return {
        "data": {
            "id": conversation_id,
            "updated_time": latest_message_time,
            "participants": {
                "data": [
                    {"id": "business_1", "username": "detailer"},
                    {"id": "cust_1", "username": "alice"},
                ]
            },
            "messages": {
                "data": [
                    {
                        "id": latest_message_id,
                        "created_time": latest_message_time,
                        "message": latest_message_text,
                        "from": {
                            "id": latest_message_sender_id,
                            "username": latest_message_sender_username,
                        },
                        "to": {"data": [{"id": "business_1", "username": "detailer"}]},
                    }
                ]
            },
        }
    }


class _FakeRuntime:
    def __init__(
        self,
        decisions: list[dict[str, Any]],
        *,
        status_messages: list[Any] | None = None,
    ) -> None:
        self.decisions = list(decisions)
        self.status_messages = list(status_messages or [])
        self.calls: list[dict[str, Any]] = []
        self.status_calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.decisions:
            raise RuntimeError("unexpected intake decision call")
        return self.decisions.pop(0)

    async def generate_status_message(self, **kwargs: Any) -> Any:
        self.status_calls.append(kwargs)
        if not self.status_messages:
            return None
        result = self.status_messages.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def log_behavior_event(self, *, event: str, **fields: Any) -> None:
        self.behavior_events.append({"event": event, **fields})

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        if customer_id:
            fields.setdefault("customer_id", customer_id)
        self.log_behavior_event(event=event, **fields)


def test_decision_actions_normalize_and_validate_supported_decision() -> None:
    actions = DecisionActions.from_decision(
        {
            "booking_action": " CREATE_NEW_BOOKING ",
            "ready_to_save": True,
            "reply_action": " SEND_REPLY ",
            "reply_text": " Booked. ",
            "sink_action": " UPSERT_PARTIAL ",
            "sink_payload": {"name": "Alice"},
        }
    )

    assert actions.booking_action == "create_new_booking"
    assert actions.ready_to_save is True
    assert actions.reply_action == "send_reply"
    assert actions.reply_text == "Booked."
    assert actions.sink_action == "upsert_partial"
    assert actions.sink_payload == {"name": "Alice"}
    assert actions.validation_error() is None


def test_decision_actions_preserve_existing_validation_errors() -> None:
    unsupported_sink = DecisionActions.from_decision({"sink_action": "delete"})
    unsupported_booking = DecisionActions.from_decision({"booking_action": "reschedule"})
    sink_without_booking = DecisionActions.from_decision(
        {"booking_action": "ignore", "sink_action": "upsert_partial"}
    )

    assert unsupported_sink.validation_error() == "unsupported sink_action=delete"
    assert unsupported_booking.validation_error() == "unsupported booking_action=reschedule"
    assert sink_without_booking.validation_error() == "sink_action requires an active booking action"


def test_conversation_cursor_signals_normalize_cursor_fields() -> None:
    signals = ConversationCursorSignals.from_summary(
        {
            "conversation_id": " conv_1 ",
            "latest_inbound_message_id": " msg_1 ",
            "latest_inbound_message_created_time": " 2026-04-07T08:00:00+00:00 ",
            "conversation_updated_time": " 2026-04-07T08:00:30+00:00 ",
            "latest_outbound_message_id": " out_1 ",
        }
    )

    assert signals.conversation_id == "conv_1"
    assert signals.latest_inbound_message_id == "msg_1"
    assert signals.latest_inbound_message_time == "2026-04-07T08:00:00+00:00"
    assert signals.conversation_updated_time == "2026-04-07T08:00:30+00:00"
    assert signals.latest_outbound_message_id == "out_1"


def test_workflow_run_accumulator_preserves_response_summary_precedence() -> None:
    accumulator = WorkflowRunAccumulator(
        processed=2,
        matched=1,
        saved_notifications=["saved booking"],
        errors=["conv_2: failed"],
        result_items=[{"conversation_id": "conv_1"}],
    )

    response = accumulator.build_response(
        workflow={"name": "Car Wash"},
        workflow_id="wf_1",
        event_type="manual",
        source_warnings=[{"warning": "slow"}],
        empty_summary_token=NO_NOTIFY_TOKEN,
    )

    assert response["ok"] is False
    assert response["workflow_id"] == "wf_1"
    assert response["event_type"] == "manual"
    assert response["processed_conversations"] == 2
    assert response["matched_conversations"] == 1
    assert response["results"] == [{"conversation_id": "conv_1"}]
    assert response["errors"] == ["conv_2: failed"]
    assert response["source_warnings"] == [{"warning": "slow"}]
    assert response["summary"] == "Workflow Car Wash hit errors: conv_2: failed"


class _DelayedRuntime(_FakeRuntime):
    def __init__(self, decisions: list[dict[str, Any]], *, delay_seconds: float) -> None:
        super().__init__(decisions)
        self.delay_seconds = delay_seconds

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(self.delay_seconds)
        return await super().decide_intake_workflow(**kwargs)


class _FakeKnowledgeOracle:
    def answer(self, **kwargs: Any) -> str:
        source_pack = str(kwargs.get("source_pack", ""))
        query = str(kwargs.get("query", ""))
        relevant_query = query
        if "Latest customer message:" in query:
            relevant_query = query.split("Latest customer message:", 1)[1].split("Workflow field contract JSON:", 1)[0]
        folded = relevant_query.casefold()
        if "reference" in source_pack.casefold() or "reference" in folded:
            return "Reference numbers are required for all appointments."
        if "ppf" in folded:
            return "NO_SOURCE"
        if "19r" in folded or "19 r" in folded or "шиномонтаж" in folded:
            return "Комплект 19`R: C=3000, D=3500, E=4 000. Source: Шиномонтаж row 10."
        if "2х-фазная" in folded or "2х фазная" in folded or "мойка" in folded:
            return "2х-фазная мойка кузова: SUV/S-Class price is 1200. Source: Мойка row 7."
        return "NO_SOURCE"


class _FakeComposio:
    enabled = True

    def __init__(
        self,
        summary: dict[str, Any],
        conversation: dict[str, Any],
        *,
        sheet_names_by_spreadsheet: dict[str, list[str]] | None = None,
        list_warnings: list[dict[str, str]] | None = None,
    ) -> None:
        self.summary = summary
        self.conversation = conversation
        self.list_warnings = list(list_warnings or [])
        self.sheet_names_by_spreadsheet = {
            str(key): list(value)
            for key, value in (sheet_names_by_spreadsheet or {}).items()
        }
        self.execute_calls: list[dict[str, Any]] = []
        self.list_calls = 0
        self.list_limits: list[int] = []
        self.get_calls = 0
        self.list_sheet_names_calls: list[dict[str, Any]] = []

    def list_instagram_conversations(
        self,
        *,
        customer_id: str,
        connected_account_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        del customer_id, connected_account_id
        self.list_calls += 1
        self.list_limits.append(limit)
        return {"ok": True, "items": [self.summary], "warnings": self.list_warnings}

    def get_instagram_conversation(
        self,
        *,
        customer_id: str,
        conversation_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        del customer_id, connected_account_id
        self.get_calls += 1
        assert conversation_id == self.summary["conversation_id"]
        return {"ok": True, "conversation": self.conversation, "summary": self.summary}

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        self.execute_calls.append(
            {
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": dict(arguments or {}),
                "connected_account_id": connected_account_id,
                "text": text,
            }
        )
        return {"successful": True, "data": {"ok": True, "tool_slug": tool_slug}}

    def search_tools(
        self,
        *,
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        del limit
        safe_query = str(query or "").strip().lower()
        normalized_toolkits = {str(item or "").strip().lower() for item in (toolkits or [])}
        items: list[dict[str, Any]] = []
        if not normalized_toolkits or "googlesheets" in normalized_toolkits:
            items.append(
                {
                    "slug": "GOOGLESHEETS_UPSERT_ROWS",
                    "toolkit_slug": "googlesheets",
                    "name": "Google Sheets Upsert Rows",
                    "description": "Upsert rows in a Google Sheet.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "headers": {"type": "array"},
                            "rows": {"type": "array"},
                            "keyColumn": {"type": "string"},
                        },
                    },
                }
            )
        if not normalized_toolkits or "crm" in normalized_toolkits:
            items.append(
                {
                    "slug": "CRM_UPSERT_BOOKING",
                    "toolkit_slug": "crm",
                    "name": "CRM Upsert Booking",
                    "description": "Create or update a booking in the CRM.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "booking_id": {"type": "string"},
                            "vehicle": {"type": "string"},
                            "wash": {"type": "string"},
                        },
                    },
                }
            )
        if safe_query:
            filtered = []
            for item in items:
                haystack = " ".join(
                    [
                        str(item.get("slug", "") or ""),
                        str(item.get("name", "") or ""),
                        str(item.get("description", "") or ""),
                    ]
                ).lower()
                if all(token in haystack for token in safe_query.split()):
                    filtered.append(item)
            if filtered:
                items = filtered
        return {"ok": True, "items": items}

    def get_tool_schema(self, *, tool_slug: str) -> dict[str, Any]:
        toolkit = "googlesheets" if tool_slug.upper().startswith("GOOGLESHEETS_") else "crm"
        return {
            "ok": True,
            "tool": {
                "slug": tool_slug,
                "toolkit_slug": toolkit,
                "input_schema": {"type": "object"},
            },
        }

    def list_google_sheets_tab_names(
        self,
        *,
        customer_id: str,
        spreadsheet_id: str,
        connected_account_id: str | None = None,
    ) -> dict[str, Any]:
        self.list_sheet_names_calls.append(
            {
                "customer_id": customer_id,
                "spreadsheet_id": spreadsheet_id,
                "connected_account_id": connected_account_id,
            }
        )
        if spreadsheet_id not in self.sheet_names_by_spreadsheet:
            return {
                "ok": False,
                "spreadsheet_id": spreadsheet_id,
                "sheet_names": [],
                "error": "spreadsheet not found in fake",
            }
        return {
            "ok": True,
            "spreadsheet_id": spreadsheet_id,
            "sheet_names": list(self.sheet_names_by_spreadsheet[spreadsheet_id]),
        }


class _FailingReplyOnceComposio(_FakeComposio):
    def __init__(self, summary: dict[str, Any], conversation: dict[str, Any]) -> None:
        super().__init__(summary, conversation)
        self._failed_reply_once = False

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        result = super().execute_tool(
            customer_id=customer_id,
            tool_slug=tool_slug,
            arguments=arguments,
            connected_account_id=connected_account_id,
            text=text,
        )
        if tool_slug == "INSTAGRAM_SEND_TEXT_MESSAGE" and not self._failed_reply_once:
            self._failed_reply_once = True
            return {
                "successful": False,
                "error": "Invalid request data provided\n• Following fields are missing: {'text'}",
                "data": {"status_code": 400},
            }
        return result


class _AlwaysFailingReplyComposio(_FakeComposio):
    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        super().execute_tool(
            customer_id=customer_id,
            tool_slug=tool_slug,
            arguments=arguments,
            connected_account_id=connected_account_id,
            text=text,
        )
        if tool_slug == "INSTAGRAM_SEND_TEXT_MESSAGE":
            return {"successful": False, "error": "temporary send failure"}
        return {"successful": True, "data": {"ok": True, "tool_slug": tool_slug}}


class _FailingSinkComposio(_FakeComposio):
    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        super().execute_tool(
            customer_id=customer_id,
            tool_slug=tool_slug,
            arguments=arguments,
            connected_account_id=connected_account_id,
            text=text,
        )
        if tool_slug == "GOOGLESHEETS_UPSERT_ROWS":
            return {"successful": False, "error": "sheet write failed"}
        return {"successful": True, "data": {"ok": True, "tool_slug": tool_slug}}


class _FailingSinkOnceComposio(_FakeComposio):
    def __init__(self, summary: dict[str, Any], conversation: dict[str, Any]) -> None:
        super().__init__(summary, conversation)
        self._failed_once = False

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        result = super().execute_tool(
            customer_id=customer_id,
            tool_slug=tool_slug,
            arguments=arguments,
            connected_account_id=connected_account_id,
            text=text,
        )
        if tool_slug == "GOOGLESHEETS_UPSERT_ROWS" and not self._failed_once:
            self._failed_once = True
            return {"successful": False, "error": "sheet write failed"}
        return result


class _SheetNameRequiredSinkComposio(_FakeComposio):
    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        result = super().execute_tool(
            customer_id=customer_id,
            tool_slug=tool_slug,
            arguments=arguments,
            connected_account_id=connected_account_id,
            text=text,
        )
        if tool_slug == "GOOGLESHEETS_UPSERT_ROWS":
            safe_arguments = dict(arguments or {})
            if not str(safe_arguments.get("sheetName", "") or "").strip():
                return {
                    "successful": False,
                    "error": "Invalid request data provided\n- Following fields are missing: {'sheetName'}",
                }
        return result


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self._message_id = 1_000

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        business_connection_id: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        self._message_id += 1
        self.sent_messages.append(
            {
                "chat_id": str(chat_id),
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": dict(reply_markup or {}) if isinstance(reply_markup, dict) else None,
                "business_connection_id": business_connection_id,
                "reply_to_message_id": reply_to_message_id,
                "message_id": self._message_id,
            }
        )
        return {
            "ok": True,
            "result": {
                "message_id": self._message_id,
                "date": int(datetime.now(UTC).timestamp()),
                "chat": {"id": chat_id, "type": "private"},
                "text": text,
                "business_connection_id": business_connection_id,
                "sender_business_bot": {"id": "fake-bot"},
            },
        }


class _SplitResultTelegramClient:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup: dict[str, Any] | None = None,
        business_connection_id: str | None = None,
        reply_to_message_id: int | None = None,
    ) -> dict[str, Any]:
        del parse_mode, reply_markup, reply_to_message_id
        chunks = ["First chunk", "Second chunk"]
        results: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks, start=1):
            self.sent_messages.append(
                {
                    "chat_id": str(chat_id),
                    "text": chunk,
                    "business_connection_id": business_connection_id,
                    "message_id": 2_000 + idx,
                }
            )
            results.append(
                {
                    "ok": True,
                    "result": {
                        "message_id": 2_000 + idx,
                        "date": int(datetime.now(UTC).timestamp()) + idx,
                        "chat": {"id": chat_id, "type": "private"},
                        "text": chunk,
                        "business_connection_id": business_connection_id,
                        "sender_business_bot": {"id": "fake-bot"},
                    },
                }
            )
        return {"ok": True, "result": results[0]["result"], "results": results}


def _mk_service(
    tmp_path: Path,
    *,
    runtime: _FakeRuntime,
    composio: _FakeComposio,
) -> tuple[IntakeWorkflowService, SchedulerService, SkillStoreService, TelegramBusinessService, FileVaultService]:
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.db")
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    telegram_business = TelegramBusinessService(db_path=tmp_path / "telegram_business.db")
    telegram_business.client = _FakeTelegramClient()
    file_vault = FileVaultService(
        root_dir=tmp_path / "file_vault",
        db_path=tmp_path / "file_vault.db",
    )
    knowledge = BusinessKnowledgeService(
        root_dir=tmp_path / "knowledge",
        db_path=tmp_path / "knowledge.db",
        file_vault=file_vault,
        oracle_client=_FakeKnowledgeOracle(),
    )
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.db",
        project_root=tmp_path,
        scheduler=scheduler,
        skill_store=skills,
        composio=composio,
        telegram_business=telegram_business,
        file_vault=file_vault,
        knowledge_service=knowledge,
        get_agent_runtime=lambda: runtime,
    )
    return service, scheduler, skills, telegram_business, file_vault


def test_resolve_booking_target_promotes_new_booking_to_active_update(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )
    active_booking = {
        "booking_id": "bkg_active",
        "workflow_id": "wf_1",
        "customer_id": "telegram_123",
        "conversation_id": "conv_1",
        "extracted_fields": {"service": "wash"},
    }

    target = service._resolve_booking_target(  # noqa: SLF001
        workflow={"workflow_id": "wf_1", "customer_id": "telegram_123"},
        conversation_summary={"conversation_id": "conv_1"},
        booking_action="create_new_booking",
        active_booking=active_booking,
        recent_completed_booking=None,
    )

    assert target.booking_action == "update_active"
    assert target.booking == active_booking
    assert target.booking is not active_booking


def _telegram_business_inbound(
    *,
    business_connection_id: str,
    chat_id: int,
    user_id: int,
    username: str,
    message_id: int,
    text: str,
    date: int,
) -> dict[str, Any]:
    return {
        "business_connection_id": business_connection_id,
        "message_id": message_id,
        "date": date,
        "chat": {"id": chat_id, "type": "private", "username": username},
        "from": {"id": user_id, "is_bot": False, "username": username},
        "text": text,
    }


@pytest.mark.asyncio
async def test_intake_workflow_upsert_creates_routine_and_skill(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    service, scheduler, skills, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["channel"] == "instagram_dm"
    assert workflow["provider"] == "composio"
    assert workflow["schedule"] == "*/2 * * * *"
    assert scheduler.get_routine(workflow["routine_id"]) is not None
    skill = skills.get_skill(
        customer_id="telegram_123",
        name=f"intake-workflow-{workflow['workflow_id']}",
        include_files=True,
        include_global=False,
    )
    assert skill is not None
    assert "workflow.json" in skill["supporting_files"]


@pytest.mark.asyncio
async def test_intake_workflow_upsert_persists_telegram_business_fields(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    service, _, skills, _, file_vault = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )
    record = file_vault.ingest_file(
        customer_id="telegram_123",
        chat_id=123,
        kind="document",
        telegram_file_id="tg_1",
        original_filename="faq.txt",
        mime_type="text/plain",
        caption=None,
        raw_bytes=b"Appointments are 45 minutes and require a $20 deposit.",
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Salon Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["name", "time"],
        field_guidance={"time": "Always confirm the final appointment time explicitly."},
        assistant_instructions="Be concise and never promise unavailable slots.",
        business_facts={"prices": {"basic_wash": "1000 RUB"}},
        knowledge_file_ids=[str(record["id"])],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""
    assert workflow["reply_mode"] == "auto"
    assert workflow["assistant_instructions"] == "Be concise and never promise unavailable slots."
    assert workflow["business_facts"] == {"prices": {"basic_wash": "1000 RUB"}}
    assert workflow["knowledge_file_ids"] == [str(record["id"])]
    skill = skills.get_skill(
        customer_id="telegram_123",
        name=f"intake-workflow-{workflow['workflow_id']}",
        include_files=True,
        include_global=False,
    )
    assert skill is not None
    assert "Telegram Business DMs" in skill["skill_markdown"]
    assert "## Workflow Goal" in skill["skill_markdown"]
    assert "## Operating Context" in skill["skill_markdown"]
    assert "## Save Behavior" in skill["skill_markdown"]
    assert "single durable intake policy" in skill["skill_markdown"]
    assert "cannot be edited in place" in skill["skill_markdown"]
    assert "Always confirm the final appointment time explicitly." in skill["skill_markdown"]
    assert "1000 RUB" in skill["skill_markdown"]
    workflow_file = json.loads(skill["supporting_files"]["workflow.json"])
    assert workflow_file["source_config"] == {"business_connection_id": "bc_123"}
    assert workflow_file["field_guidance"] == {
        "time": "Always confirm the final appointment time explicitly."
    }
    assert workflow_file["business_facts"] == {"prices": {"basic_wash": "1000 RUB"}}


@pytest.mark.asyncio
async def test_intake_workflow_business_facts_do_not_store_large_source_blobs(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time"],
        business_facts={"extracted_spreadsheet_text": "row data\n" * 3000},
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    rendered = json.dumps(workflow["business_facts"], ensure_ascii=False)
    assert len(rendered) < 1000
    assert "too large to store inline" in rendered


@pytest.mark.asyncio
async def test_telegram_business_workflow_upsert_auto_resolves_single_connected_account(
    tmp_path: Path,
) -> None:
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Salon Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["source_config"] == {"business_connection_id": "bc_123"}
    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""


@pytest.mark.asyncio
async def test_telegram_business_workflow_drops_false_intent_match_source_config(
    tmp_path: Path,
) -> None:
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Salon Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={
            "business_connection_id": "bc_123",
            "intent_match_required": False,
            "matching": {"intent_match_required": "false"},
        },
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["source_config"] == {"business_connection_id": "bc_123"}


@pytest.mark.asyncio
async def test_telegram_business_workflow_does_not_create_scheduler_routine(
    tmp_path: Path,
) -> None:
    service, scheduler, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Salon Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["schedule"] == ""
    assert workflow["routine_id"] == ""
    assert scheduler.list_routines() == []


@pytest.mark.asyncio
async def test_telegram_business_workflow_upsert_requires_delete_then_recreate_for_customer(
    tmp_path: Path,
) -> None:
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )

    first = service.upsert_workflow(
        customer_id="telegram_123",
        name="Salon Telegram Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    with pytest.raises(ValueError, match="cannot be updated in place"):
        service.upsert_workflow(
            customer_id="telegram_123",
            name="Salon Telegram Intake Updated",
            channel="telegram_business_dm",
            provider="telegram_bot_api",
            source_config={"business_connection_id": "bc_123"},
            intent_description="Handle Telegram Business appointment and reschedule requests.",
            required_fields=["name", "time", "service"],
            assistant_instructions="Be concise and collect service details.",
            sink_type="local_csv",
            sink_config={"file_path": "tulpa_stuff/bookings.csv"},
        )

    with pytest.raises(ValueError, match="cannot be edited in place"):
        service.upsert_workflow(
            customer_id="telegram_123",
            workflow_id=first["workflow_id"],
            name="Salon Telegram Intake Updated",
            channel="telegram_business_dm",
            provider="telegram_bot_api",
            source_config={"business_connection_id": "bc_123"},
            intent_description="Handle Telegram Business appointment and reschedule requests.",
            required_fields=["name", "time", "service"],
            assistant_instructions="Be concise and collect service details.",
            sink_type="local_csv",
            sink_config={"file_path": "tulpa_stuff/bookings.csv"},
        )

    workflows = service.list_workflows(customer_id="telegram_123", include_disabled=True)
    telegram_workflows = [
        item for item in workflows if item["channel"] == "telegram_business_dm"
    ]
    assert len(telegram_workflows) == 1


@pytest.mark.asyncio
async def test_intake_workflow_upsert_normalizes_none_workflow_id_to_short_generated_id(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    service, scheduler, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        workflow_id="None",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["workflow_id"].startswith("iwf_")
    assert workflow["workflow_id"] != "None"
    routine = scheduler.get_routine(workflow["routine_id"])
    assert routine is not None
    assert str((routine.payload or {}).get("workflow_id", "")) == workflow["workflow_id"]


@pytest.mark.asyncio
async def test_intake_workflow_upsert_accepts_local_csv_filename_alias(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"filename": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["sink_config"] == {"file_path": "tulpa_stuff/bookings.csv"}


@pytest.mark.asyncio
async def test_intake_workflow_run_saves_local_csv_and_skips_reprocessing_same_message(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {
                    "day": "tomorrow",
                    "time": "3pm",
                    "car_type": "SUV",
                    "wash_type": "interior and exterior",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "day": "tomorrow",
                    "time": "3pm",
                    "car_type": "SUV",
                    "wash_type": "interior and exterior",
                },
                "reason": "All required fields are present.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    first_run = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )
    assert first_run["ok"] is True
    assert "Booking saved for Car Wash Intake:" in first_run["summary"]
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "completed"
    csv_path = tmp_path / "tulpa_stuff" / "bookings.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["car_type"] == "SUV"

    second_run = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )
    assert second_run["summary"] == NO_NOTIFY_TOKEN
    assert second_run["processed_conversations"] == 0
    assert len(runtime.calls) == 1


@pytest.mark.asyncio
async def test_local_csv_sink_status_is_runtime_owned(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Book a wash tomorrow at 3pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {"day": "tomorrow", "time": "3pm"},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {"day": "tomorrow", "time": "3pm", "status": "active"},
                "reason": "All required fields are present.",
            }
        ]
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio(summary, conversation),
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert bookings[0]["status"] == "completed"
    with (tmp_path / "tulpa_stuff" / "bookings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_local_csv_sink_uses_runtime_cancelled_status(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Book a wash tomorrow at 3pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {"day": "tomorrow", "time": "3pm"},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {"day": "tomorrow", "time": "3pm"},
                "reason": "All required fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer cancelled the booking.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "mark_cancelled",
                "reply_text": "Cancelled.",
                "ready_to_save": True,
                "booking_action": "edit_recent_completed",
                "save_payload": {"day": "tomorrow", "time": "3pm", "status": "cancelled"},
                "reason": "Clear cancellation.",
            },
        ]
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio(summary, conversation),
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    first_run = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )
    assert first_run["ok"] is True
    second_run = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        force=True,
    )

    assert second_run["ok"] is True
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert bookings[0]["status"] == "cancelled"
    with (tmp_path / "tulpa_stuff" / "bookings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_instagram_scheduled_workflow_uses_poll_interval_for_fresh_inbound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 4, 30, tzinfo=UTC),
    )
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:02:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm.",
        latest_message_time="2026-04-07T08:02:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Within the scheduled polling window.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert workflow["schedule"] == "*/2 * * * *"
    assert service._latest_inbound_max_age_for_event(  # noqa: SLF001
        event_type="scheduled",
        workflow=workflow,
    ) == timedelta(minutes=5)
    assert result["ok"] is True
    assert result["processed_conversations"] == 1
    assert result["matched_conversations"] == 1
    assert len(runtime.calls) == 1
    assert composio.list_limits
    assert set(composio.list_limits) == {20}


@pytest.mark.asyncio
async def test_instagram_decision_uses_unanswered_customer_burst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 4, 45, tzinfo=UTC),
    )
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_5",
        "latest_inbound_message_created_time": "2026-04-07T08:04:30+00:00",
        "latest_inbound_message_text_preview": "Pleeeeease",
        "latest_inbound_sender_username": "alice",
    }
    conversation = {
        "data": {
            "id": "conv_1",
            "updated_time": "2026-04-07T08:04:30+00:00",
            "participants": {
                "data": [
                    {"id": "business_1", "username": "salon"},
                    {"id": "cust_1", "username": "alice"},
                ]
            },
            "messages": {
                "data": [
                    {
                        "id": "msg_0",
                        "created_time": "2026-04-07T07:59:00+00:00",
                        "message": "What time works on June 8?",
                        "from": {"id": "business_1", "username": "salon"},
                    },
                    {
                        "id": "msg_1",
                        "created_time": "2026-04-07T08:00:00+00:00",
                        "message": "Reach ma ballls",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_1b",
                        "created_time": "2026-04-07T08:00:30+00:00",
                        "message": "I mean it",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_1c",
                        "created_time": "2026-04-07T08:01:00+00:00",
                        "message": "Please answer",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_1d",
                        "created_time": "2026-04-07T08:01:30+00:00",
                        "message": "Need help",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_2",
                        "created_time": "2026-04-07T08:02:00+00:00",
                        "message": "Can u do that for me",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_3",
                        "created_time": "2026-04-07T08:03:00+00:00",
                        "message": "Still here",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_3b",
                        "created_time": "2026-04-07T08:03:30+00:00",
                        "message": "Can you reply",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_4",
                        "created_time": "2026-04-07T08:04:00+00:00",
                        "message": "?",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_5",
                        "created_time": "2026-04-07T08:04:30+00:00",
                        "message": "Pleeeeease",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                ]
            },
        }
    }
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer sent a burst of unanswered messages.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Use full unanswered burst.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Salon Intake",
        intent_description="Handle Instagram DMs that ask to book salon appointments.",
        required_fields=["time", "phone"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert len(runtime.calls) == 1
    burst = runtime.calls[0]["conversation"]["unanswered_customer_messages"]
    assert [item["text"] for item in burst] == [
        "Reach ma ballls",
        "I mean it",
        "Please answer",
        "Need help",
        "Can u do that for me",
        "Still here",
        "Can you reply",
        "?",
        "Pleeeeease",
    ]


@pytest.mark.asyncio
async def test_instagram_stale_decision_refreshes_before_replying(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 4, 30, tzinfo=UTC),
    )
    initial_summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:02:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    initial_conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm.",
        latest_message_time="2026-04-07T08:02:00+00:00",
    )
    composio = _FakeComposio(initial_summary, initial_conversation)

    class _Runtime(_FakeRuntime):
        async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
            decision = await super().decide_intake_workflow(**kwargs)
            if len(self.calls) == 1:
                composio.summary = {
                    **initial_summary,
                    "latest_inbound_message_id": "msg_2",
                    "latest_inbound_message_created_time": "2026-04-07T08:04:00+00:00",
                    "latest_inbound_message_text_preview": "Actually make it 4pm.",
                }
                composio.conversation = _instagram_conversation(
                    conversation_id="conv_1",
                    latest_message_id="msg_2",
                    latest_message_text="Actually make it 4pm.",
                    latest_message_time="2026-04-07T08:04:00+00:00",
                )
            return decision

    runtime = _Runtime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {},
                "missing_fields": ["time"],
                "reply_action": "send_reply",
                "reply_text": "What time works?",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Ask for missing time.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer changed the requested time.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great, I have 4pm.",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Reply to the latest inbound message.",
            },
        ]
    )
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
        reply_mode="auto",
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )
    cursor = service._get_cursor(  # noqa: SLF001
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )

    assert workflow["schedule"] == "*/2 * * * *"
    assert result["processed_conversations"] == 1
    assert result["results"][0]["status"] == "ignored"
    assert result["results"][0]["replied"] is True
    assert cursor["last_seen_inbound_message_id"] == "msg_2"
    assert composio.execute_calls == [
        {
            "customer_id": "telegram_123",
            "tool_slug": "INSTAGRAM_SEND_TEXT_MESSAGE",
            "arguments": {
                "recipient_id": "cust_1",
                "conversation_id": "conv_1",
                "text": "Great, I have 4pm.",
                "reply_to_message_id": "msg_2",
            },
            "connected_account_id": None,
            "text": None,
        }
    ]
    assert len(runtime.calls) == 2


@pytest.mark.asyncio
async def test_instagram_apply_stale_wait_does_not_advance_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 4, 30, tzinfo=UTC),
    )
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:02:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm.",
        latest_message_time="2026-04-07T08:02:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {},
                "missing_fields": ["vehicle"],
                "reply_action": "send_reply",
                "reply_text": "What vehicle should we book?",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Ask for missing vehicle.",
            }
        ]
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio(summary, conversation),
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["vehicle"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
        reply_mode="auto",
    )
    stale_checks = 0

    def _fake_stale_check(**kwargs: Any) -> tuple[bool, dict[str, Any], str | None]:
        nonlocal stale_checks
        stale_checks += 1
        raw_summary = kwargs.get("decided_summary")
        decided_summary = dict(raw_summary if isinstance(raw_summary, dict) else {})
        if stale_checks == 1:
            return False, decided_summary, None
        latest_summary = dict(decided_summary)
        latest_summary["latest_inbound_message_id"] = "msg_2"
        latest_summary["latest_inbound_message_created_time"] = "2026-04-07T08:04:00+00:00"
        return True, latest_summary, None

    monkeypatch.setattr(service, "_conversation_became_stale", _fake_stale_check)

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["results"] == [
        {
            "conversation_id": "conv_1",
            "matched": True,
            "status": "stale_waiting_for_next_poll",
            "replied": False,
        }
    ]
    assert stale_checks == 2
    assert service._get_cursor(  # noqa: SLF001
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    ) == {}


@pytest.mark.asyncio
async def test_intake_workflow_ignores_latest_inbound_older_than_poll_interval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 7, 0, tzinfo=UTC),
    )
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a car wash booking.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Should never be reached.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert result["processed_conversations"] == 0
    assert result["matched_conversations"] == 0
    assert result["summary"] == NO_NOTIFY_TOKEN
    assert runtime.calls == []
    cursor = service._get_cursor(workflow_id=workflow["workflow_id"], conversation_id="conv_1")  # noqa: SLF001
    assert cursor["last_seen_inbound_message_id"] == "msg_1"


@pytest.mark.asyncio
async def test_telegram_business_settled_run_handles_delayed_inbound_after_one_minute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 5, 0, tzinfo=UTC),
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer provided booking details.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Могу записать вас сегодня в 17:00.",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Fresh webhook work can be delayed behind slow model calls.",
            }
        ]
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio({}, {}),
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": int(datetime(2026, 4, 7, 8, 0, 0, tzinfo=UTC).timestamp()),
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Можно сегодня в 17:00?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="АвтоSpa — консультация и запись",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Помогать с услугами, ценами и записью в автоцентр.",
        required_fields=["name", "time"],
        assistant_instructions="Reply in Russian.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook_settled",
    )

    assert result["processed_conversations"] == 1
    assert len(runtime.calls) == 1
    assert telegram_business.client.sent_messages[0]["text"] == "Могу записать вас сегодня в 17:00."


@pytest.mark.asyncio
async def test_telegram_business_workflow_uses_bound_files_and_replies_via_business_connection(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants an appointment.",
                "extracted_fields": {"name": "Alice", "time": "3pm"},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "needs_business_knowledge": True,
                "business_knowledge_query": "appointment reference number policy",
                "reason": "Need source-backed policy before replying.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants an appointment.",
                "extracted_fields": {"name": "Alice", "time": "3pm"},
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Booked for 3pm. Please bring your reference number.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {"name": "Alice", "time": "3pm"},
                "reason": "All booking fields are present.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, file_vault = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Hi, can I book 3pm today?",
        },
    )
    knowledge = file_vault.ingest_file(
        customer_id="telegram_123",
        chat_id=123,
        kind="document",
        telegram_file_id="tg_knowledge",
        original_filename="policy.txt",
        mime_type="text/plain",
        caption=None,
        raw_bytes=b"Reference numbers are required for all appointments.",
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Be concise and confirm only explicit booking times.",
        business_facts={"prices": {"express_wash": "1000 RUB"}},
        knowledge_file_ids=[str(knowledge["id"])],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert len(runtime.calls) == 2
    assert runtime.calls[0]["workflow"]["assistant_instructions"] == "Be concise and confirm only explicit booking times."
    assert runtime.calls[0]["workflow"]["business_facts"] == {
        "prices": {"express_wash": "1000 RUB"}
    }
    assert "Owner-Provided Business Facts" in runtime.calls[0]["workflow"]["workflow_skill"]
    assert "1000 RUB" in runtime.calls[0]["workflow"]["workflow_skill"]
    assert runtime.calls[0]["workflow"]["knowledge_file_ids"] == [str(knowledge["id"])]
    assert runtime.calls[0]["workflow"]["knowledge_answer"] == ""
    assert "Reference numbers" in runtime.calls[1]["workflow"]["knowledge_answer"]
    assert runtime.calls[1]["conversation"]["unanswered_customer_messages"] == runtime.calls[0][
        "conversation"
    ]["unanswered_customer_messages"]
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10


@pytest.mark.asyncio
async def test_telegram_business_reply_persists_each_split_message(
    tmp_path: Path,
) -> None:
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )
    telegram_business.client = _SplitResultTelegramClient()
    workflow = {
        "customer_id": "telegram_123",
        "channel": "telegram_business_dm",
        "provider": "telegram_bot_api",
        "source_config": {"business_connection_id": "bc_123"},
    }
    conversation_summary = {
        "conversation_id": "555",
        "latest_inbound_message_id": "10",
    }

    error = await service._send_source_reply(
        workflow=workflow,
        conversation_summary=conversation_summary,
        reply_text="First chunk\n\nSecond chunk",
    )

    assert error is None
    conversation = telegram_business.get_conversation(
        customer_id="telegram_123",
        business_connection_id="bc_123",
        conversation_id="555",
    )
    assert conversation["ok"] is True
    messages = conversation["conversation"]["messages"]
    assert [item["message_id"] for item in messages] == ["2001", "2002"]
    assert [item["text"] for item in messages] == ["First chunk", "Second chunk"]
    assert all(item["sender_role"] == "assistant" for item in messages)


@pytest.mark.asyncio
async def test_telegram_business_default_workflow_does_not_intent_gate_model_reply(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": False,
                "confidence": 0.3,
                "conversation_summary": "Customer opened with a greeting.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Здравствуйте! Чем помочь с услугами или записью?",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Greeting is not an explicit booking intent yet.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Привет",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="АвтоSpa — консультация и запись",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Помогать с услугами, ценами и записью в автоцентр.",
        required_fields=["name", "time"],
        assistant_instructions="Reply in Russian and help customers choose a service.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert result["matched_conversations"] == 1
    assert result["results"] == [
        {
            "conversation_id": "555",
            "matched": True,
            "status": "ignored",
            "replied": True,
        }
    ]
    assert runtime.calls[0]["workflow"]["intent_match_required"] is False
    assert service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    ) == []
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10
    assert "Чем помочь" in sent["text"]


@pytest.mark.asyncio
async def test_telegram_business_no_file_workflow_does_not_silence_business_knowledge_decision(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer asks for an SUV wash price and wants to book.",
                "extracted_fields": {"car_type": "SUV", "wash_type": "exterior", "date": "tomorrow"},
                "missing_fields": ["time"],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "needs_business_knowledge": True,
                "business_knowledge_query": "What is the price for an SUV car wash?",
                "reason": "Need a source-backed price before replying.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "How much for an SUV exterior wash? Tomorrow maybe.",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Car Wash",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business car wash booking requests.",
        required_fields=["car_type", "wash_type", "date", "time"],
        assistant_instructions="If pricing is unavailable, say it needs confirmation and continue booking intake.",
        knowledge_file_ids=[],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert len(runtime.calls) == 1
    assert runtime.calls[0]["workflow"]["knowledge_file_ids"] == []
    assert runtime.calls[0]["workflow"]["knowledge_answer"] == ""
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10
    assert "confirm" in sent["text"].lower()
    assert "time" in sent["text"].lower()
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "active"
    assert bookings[0]["sink_write_status"] == "pending"
    assert bookings[0]["extracted_fields"]["car_type"] == "SUV"
    assert not (tmp_path / "tulpa_stuff" / "bookings.csv").exists()


@pytest.mark.asyncio
async def test_telegram_business_out_of_scope_decision_can_reply_without_booking(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": False,
                "confidence": 0.0,
                "conversation_summary": "Customer asks about a service outside this workflow.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "This workflow only handles wash and tire fitting requests.",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "PPF is outside the scoped workflow.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "How much is PPF?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Wash Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123", "intent_match_required": True},
        intent_description="Handle Telegram Business wash and tire fitting appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Reply politely when the request is outside scope.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert result["matched_conversations"] == 0
    assert result["results"] == [
        {
            "conversation_id": "555",
            "matched": False,
            "status": "ignored",
            "replied": True,
        }
    ]
    assert service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    ) == []
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10
    assert "only handles wash" in sent["text"]


@pytest.mark.asyncio
async def test_telegram_business_out_of_scope_service_question_gets_fallback_reply(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": False,
                "confidence": 0.0,
                "conversation_summary": "Customer asks about a service outside this workflow.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "PPF is outside this workflow.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "How much is PPF?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Wash and tire booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123", "intent_match_required": True},
        intent_description="Handle Telegram Business wash and tire fitting appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Reply politely when the request is outside scope. Phone: +1 555 123 4567.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert result["matched_conversations"] == 0
    assert result["results"] == [
        {
            "conversation_id": "555",
            "matched": False,
            "status": "ignored",
            "replied": True,
        }
    ]
    assert service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    ) == []
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10
    assert "current workflow" in sent["text"]
    assert "+1 555 123 4567" in sent["text"]


@pytest.mark.asyncio
async def test_telegram_business_matched_ignore_reply_does_not_open_booking(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.0,
                "conversation_summary": "Customer asks about wash pricing for an unsupported vehicle.",
                "extracted_fields": {"service_category": "Мойка"},
                "missing_fields": ["service_name", "car_type", "quoted_price"],
                "reply_action": "send_reply",
                "reply_text": "В прайсе нет мойки мотоциклов. Позвоните нам, пожалуйста.",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "The category is in scope, but this vehicle type is not bookable from the price list.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Сколько стоит мойка мотоцикла?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Wash Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business wash and tire fitting appointment requests.",
        required_fields=["name", "time"],
        assistant_instructions="Reply politely when the request cannot be booked from the price list.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert result["matched_conversations"] == 1
    assert result["results"] == [
        {
            "conversation_id": "555",
            "matched": True,
            "status": "ignored",
            "replied": True,
        }
    ]
    assert service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    ) == []
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10
    assert "мотоциклов" in sent["text"]


@pytest.mark.asyncio
async def test_telegram_business_reply_is_persisted_back_into_conversation_history(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a booking.",
                "extracted_fields": {"telegram_username": "alice"},
                "missing_fields": ["time"],
                "reply_action": "send_reply",
                "reply_text": "What time works for you?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need one more field before saving.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Can I book a wash?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "time"],
        assistant_instructions="Ask for time before saving.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    conversation = telegram_business.get_conversation(
        customer_id="telegram_123",
        business_connection_id="bc_123",
        conversation_id="555",
    )
    assert conversation["ok"] is True
    messages = conversation["conversation"]["messages"]
    assert [item["sender_role"] for item in messages] == ["customer", "assistant"]
    assert messages[1]["text"] == "What time works for you?"
    assert conversation["summary"]["latest_outbound_message_id"]


@pytest.mark.asyncio
async def test_telegram_business_reply_with_create_booking_action_opens_pending_booking(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a booking but more fields are needed.",
                "extracted_fields": {"service": "2х-фазная мойка", "price": "1200"},
                "missing_fields": ["client_name", "phone", "desired_time"],
                "reply_action": "send_reply",
                "reply_text": "2х-фазная мойка для вашего авто стоит 1200 ₽. Как вас зовут и на какое время записать?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing fields before saving.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Сколько стоит 2х-фазная мойка?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["service", "client_name", "phone", "desired_time"],
        assistant_instructions="Ask for missing fields before saving.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert telegram_business.client.sent_messages
    assert "1200" in telegram_business.client.sent_messages[0]["text"]
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "active"
    assert bookings[0]["extracted_fields"]["service"] == "2х-фазная мойка"


@pytest.mark.asyncio
async def test_telegram_business_pending_booking_without_model_reply_asks_missing_field(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a booking but omitted contact details.",
                "extracted_fields": {"service": "2х-фазная мойка", "price": "1200"},
                "missing_fields": ["client_name", "phone"],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing fields before saving, but model omitted a reply.",
            }
        ]
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio({}, {}),
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message=_telegram_business_inbound(
            business_connection_id="bc_123",
            chat_id=555,
            user_id=999,
            username="alice",
            message_id=10,
            text="Сколько стоит 2х-фазная мойка?",
            date=1_775_552_400,
        ),
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["service", "client_name", "phone"],
        assistant_instructions="Ask for missing fields before saving.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    sent = telegram_business.client.sent_messages
    assert len(sent) == 1
    assert "client name" in sent[0]["text"].lower()
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "active"


@pytest.mark.asyncio
async def test_telegram_business_workflow_serializes_same_conversation_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_TELEGRAM_BUSINESS_WEBHOOK_DEBOUNCE_SECONDS",
        0.0,
    )
    runtime = _DelayedRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a booking.",
                "extracted_fields": {"telegram_username": "alice"},
                "missing_fields": ["time"],
                "reply_action": "send_reply",
                "reply_text": "What time works for you?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need one more field before saving.",
            }
        ],
        delay_seconds=0.05,
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Can I book a wash?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "time"],
        assistant_instructions="Ask for time before saving.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    first, second = await asyncio.gather(
        service.run_workflow(
            customer_id="telegram_123",
            workflow_id=workflow["workflow_id"],
            event_type="telegram_business_webhook",
        ),
        service.run_workflow(
            customer_id="telegram_123",
            workflow_id=workflow["workflow_id"],
            event_type="telegram_business_webhook",
        ),
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert len(runtime.calls) == 1
    assert len(telegram_business.client.sent_messages) == 1


@pytest.mark.asyncio
async def test_telegram_business_workflow_coalesces_messages_arriving_during_debounce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_TELEGRAM_BUSINESS_WEBHOOK_DEBOUNCE_SECONDS",
        0.05,
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer is asking about car wash services.",
                "extracted_fields": {"telegram_username": "alice"},
                "missing_fields": ["car_model"],
                "reply_action": "send_reply",
                "reply_text": "Да, моем. Какая у вас машина?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need to collect more details.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Привет",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "car_model"],
        assistant_instructions="Answer based on the latest coalesced lead context.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    task = asyncio.create_task(
        service.run_workflow(
            customer_id="telegram_123",
            workflow_id=workflow["workflow_id"],
            event_type="telegram_business_webhook",
        )
    )
    await asyncio.sleep(0.01)
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 11,
            "date": 1_775_552_401,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Привет, вы моете машины?",
        },
    )
    result = await task

    assert result["ok"] is True
    assert runtime.calls[0]["conversation"]["summary"]["latest_inbound_message_id"] == "11"
    assert [item["text"] for item in runtime.calls[0]["conversation"]["recent_messages"]] == [
        "Привет",
        "Привет, вы моете машины?",
    ]
    sent = telegram_business.client.sent_messages[0]
    assert sent["reply_to_message_id"] == 11


@pytest.mark.asyncio
async def test_telegram_business_workflow_suppresses_stale_reply_and_requeues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_TELEGRAM_BUSINESS_WEBHOOK_DEBOUNCE_SECONDS",
        0.0,
    )
    monkeypatch.setattr(
        intake_service_module,
        "_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS",
        0.0,
    )
    runtime = _DelayedRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer asks for a wash.",
                "extracted_fields": {"telegram_username": "alice"},
                "missing_fields": ["car_model"],
                "reply_action": "send_reply",
                "reply_text": "Какая у вас машина?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need the car model.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer asks for a wash and provided car model.",
                "extracted_fields": {"telegram_username": "alice", "car_model": "Rolls-Royce Cullinan"},
                "missing_fields": ["time"],
                "reply_action": "send_reply",
                "reply_text": "Отлично, на какое время записать Rolls-Royce Cullinan?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need appointment time.",
            },
        ],
        delay_seconds=0.05,
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Нужно помыть авто",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "car_model", "time"],
        assistant_instructions="Answer from the newest lead context only.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    first_run_task = asyncio.create_task(
        service.run_workflow(
            customer_id="telegram_123",
            workflow_id=workflow["workflow_id"],
            event_type="telegram_business_webhook",
        )
    )
    await asyncio.sleep(0.01)
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 11,
            "date": 1_775_552_401,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Ролсройс кулинан",
        },
    )
    first_result = await first_run_task
    assert first_result["ok"] is True
    assert first_result["results"] == [
        {
            "conversation_id": "555",
            "matched": True,
            "status": "stale_requeued",
            "replied": False,
        }
    ]
    assert telegram_business.client.sent_messages == []

    drained = await service.drain_due_pending_runs()

    assert drained == 1
    assert len(runtime.calls) == 2
    assert [
        item["text"]
        for item in runtime.calls[1]["conversation"]["recent_messages"]
        if item["sender_role"] == "customer"
    ] == ["Нужно помыть авто", "Ролсройс кулинан"]
    sent = telegram_business.client.sent_messages[0]
    assert sent["reply_to_message_id"] == 11
    assert "Rolls-Royce Cullinan" in sent["text"]


@pytest.mark.asyncio
async def test_telegram_business_pending_run_sends_only_final_reply(
    tmp_path: Path,
) -> None:
    runtime = _DelayedRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer asks for wash pricing.",
                "extracted_fields": {"telegram_username": "alice"},
                "missing_fields": ["car_model"],
                "reply_action": "send_reply",
                "reply_text": "Какая у вас машина?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need car model.",
            }
        ],
        delay_seconds=0.05,
    )
    runtime.status_messages.append({"ok": True, "text": "Уточняю детали и скоро отвечу."})
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio({}, {}),
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Сколько стоит мойка?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "car_model"],
        assistant_instructions="Ask for car model.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )
    service._queue_pending_run(  # noqa: SLF001
        workflow=workflow,
        conversation_id="555",
        event_type="telegram_business_webhook_settled",
        delay_seconds=0.0,
    )

    drained = await service.drain_due_pending_runs()

    assert drained == 1
    assert runtime.status_calls == []
    assert [item["text"] for item in telegram_business.client.sent_messages] == [
        "Какая у вас машина?",
    ]


@pytest.mark.asyncio
async def test_telegram_business_pending_run_ignores_status_generation_failure(tmp_path: Path) -> None:
    runtime = _DelayedRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer asks for wash pricing.",
                "extracted_fields": {"telegram_username": "alice"},
                "missing_fields": ["car_model"],
                "reply_action": "send_reply",
                "reply_text": "Какая у вас машина?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need car model.",
            }
        ],
        delay_seconds=0.05,
    )
    runtime.status_messages.append({"ok": False, "text": ""})
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=_FakeComposio({}, {}),
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "Сколько стоит мойка?",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "car_model"],
        assistant_instructions="Ask for car model.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )
    service._queue_pending_run(  # noqa: SLF001
        workflow=workflow,
        conversation_id="555",
        event_type="telegram_business_webhook_settled",
        delay_seconds=0.0,
    )

    drained = await service.drain_due_pending_runs()

    assert drained == 1
    assert runtime.status_calls == []
    assert [item["text"] for item in telegram_business.client.sent_messages] == [
        "Какая у вас машина?",
    ]


@pytest.mark.asyncio
async def test_telegram_business_final_stale_guard_prevents_sink_and_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer provided all required booking fields.",
                "extracted_fields": {
                    "telegram_username": "alice",
                    "car_model": "BMW sedan",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Booked for tomorrow at 10am.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "telegram_username": "alice",
                    "car_model": "BMW sedan",
                },
                "reason": "All required fields are present.",
            }
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    telegram_business.upsert_message(
        business_connection_id="bc_123",
        customer_id="telegram_123",
        message={
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private", "username": "alice"},
            "from": {"id": 999, "is_bot": False, "username": "alice"},
            "text": "BMW sedan, full wash, tomorrow at 10am.",
        },
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_123"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["telegram_username", "car_model"],
        assistant_instructions="Save only when the latest lead context is stable.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )
    stale_checks = 0

    def _fake_stale_check(**kwargs: Any) -> tuple[bool, dict[str, Any], str | None]:
        nonlocal stale_checks
        stale_checks += 1
        raw_summary = kwargs.get("decided_summary")
        decided_summary = dict(raw_summary if isinstance(raw_summary, dict) else {})
        if stale_checks == 1:
            return False, decided_summary, None
        latest_summary = dict(decided_summary)
        latest_summary["latest_inbound_message_id"] = "11"
        return True, latest_summary, None

    monkeypatch.setattr(service, "_conversation_became_stale", _fake_stale_check)

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert result["ok"] is True
    assert result["results"] == [
        {
            "conversation_id": "555",
            "matched": True,
            "status": "stale_requeued",
            "replied": False,
        }
    ]
    assert stale_checks == 2
    assert telegram_business.client.sent_messages == []
    assert service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    ) == []
    assert not (tmp_path / "tulpa_stuff" / "bookings.csv").exists()


@pytest.mark.asyncio
async def test_intake_pending_runs_drain_concurrently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )
    first = service.upsert_workflow(
        customer_id="telegram_123",
        name="First Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_1"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/first.csv"},
    )
    second = service.upsert_workflow(
        customer_id="telegram_456",
        name="Second Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_2"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/second.csv"},
    )
    service._queue_pending_run(  # noqa: SLF001
        workflow=first,
        conversation_id="111",
        event_type="telegram_business_webhook_settled",
        delay_seconds=0.0,
    )
    service._queue_pending_run(  # noqa: SLF001
        workflow=second,
        conversation_id="222",
        event_type="telegram_business_webhook_settled",
        delay_seconds=0.0,
    )
    active = 0
    max_active = 0

    async def _fake_run_pending_row(row: dict[str, Any]) -> None:
        del row
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.05)
        active -= 1

    monkeypatch.setattr(service, "_run_pending_row", _fake_run_pending_row)

    drained = await service.drain_due_pending_runs(limit=10)

    assert drained == 2
    assert max_active == 2


@pytest.mark.asyncio
async def test_intake_pending_run_drain_logs_row_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio({}, {}),
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Telegram Booking",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": "bc_1"},
        intent_description="Handle Telegram Business appointment requests.",
        required_fields=["time"],
        assistant_instructions="Be concise.",
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )
    service._queue_pending_run(  # noqa: SLF001
        workflow=workflow,
        conversation_id="111",
        event_type="telegram_business_webhook_settled",
        delay_seconds=0.0,
    )

    async def _fake_run_pending_row(row: dict[str, Any]) -> None:
        del row
        raise RuntimeError("boom")

    monkeypatch.setattr(service, "_run_pending_row", _fake_run_pending_row)
    caplog.set_level(logging.ERROR, logger="opentulpa.intake.service")

    drained = await service.drain_due_pending_runs(limit=10)

    assert drained == 1
    assert any("intake pending run failed during drain" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_intake_workflow_run_skips_quiet_inbox_without_model_call(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "conversation_updated_time": "2026-04-07T08:00:00+00:00",
        "latest_message_id": "msg_out_1",
        "latest_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_message_sender_id": "business_1",
        "latest_outbound_message_id": "msg_out_1",
        "latest_outbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_out_1",
        latest_message_text="Thanks, your booking is confirmed.",
        latest_message_time="2026-04-07T08:00:00+00:00",
        latest_message_sender_id="business_1",
        latest_message_sender_username="detailer",
    )
    runtime = _FakeRuntime([])
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert result["processed_conversations"] == 0
    assert result["summary"] == NO_NOTIFY_TOKEN
    assert runtime.calls == []
    assert composio.list_calls == 1
    assert composio.get_calls == 0


@pytest.mark.asyncio
async def test_intake_workflow_run_keeps_source_scan_warnings_nonfatal(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "conversation_updated_time": "2026-04-07T08:00:00+00:00",
        "latest_message_id": "msg_out_1",
        "latest_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_message_sender_id": "business_1",
        "latest_outbound_message_id": "msg_out_1",
        "latest_outbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_out_1",
        latest_message_text="Thanks, your booking is confirmed.",
        latest_message_time="2026-04-07T08:00:00+00:00",
        latest_message_sender_id="business_1",
        latest_message_sender_username="detailer",
    )
    warning = {"conversation_id": "conv_bad", "error": "Unsupported get request"}
    runtime = _FakeRuntime([])
    composio = _FakeComposio(summary, conversation, list_warnings=[warning])
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert result["errors"] == []
    assert result["source_warnings"] == [warning]
    assert result["summary"] == NO_NOTIFY_TOKEN


@pytest.mark.asyncio
async def test_intake_workflow_run_fails_configured_instagram_conversation_fetch(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "conversation_updated_time": "2026-04-07T08:00:00+00:00",
        "latest_message_id": "msg_1",
        "latest_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_message_sender_id": "lead_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Hello",
        latest_message_time="2026-04-07T08:00:00+00:00",
        latest_message_sender_id="lead_1",
        latest_message_sender_username="lead",
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(summary, conversation),
    )
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        source_config={"conversation_id": "conv_missing"},
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is False
    assert result["workflow_id"] == workflow["workflow_id"]
    assert "failed while reading Instagram DMs" in result["summary"]


@pytest.mark.asyncio
async def test_intake_workflow_run_skips_outbound_only_update_without_model_call(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "conversation_updated_time": "2026-04-07T08:05:00+00:00",
        "latest_message_id": "msg_out_2",
        "latest_message_created_time": "2026-04-07T08:05:00+00:00",
        "latest_message_sender_id": "business_1",
        "latest_inbound_message_id": "msg_in_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
        "latest_outbound_message_id": "msg_out_2",
        "latest_outbound_message_created_time": "2026-04-07T08:05:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_out_2",
        latest_message_text="What day is this for?",
        latest_message_time="2026-04-07T08:05:00+00:00",
        latest_message_sender_id="business_1",
        latest_message_sender_username="detailer",
    )
    runtime = _FakeRuntime([])
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )
    service._set_cursor(  # noqa: SLF001
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
        latest_inbound_message_id="msg_in_1",
        latest_inbound_message_time="2026-04-07T08:00:00+00:00",
        conversation_updated_time="2026-04-07T08:00:00+00:00",
        latest_outbound_message_id="msg_out_1",
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert result["processed_conversations"] == 0
    assert result["summary"] == NO_NOTIFY_TOKEN
    assert runtime.calls == []
    assert composio.get_calls == 0


@pytest.mark.asyncio
async def test_intake_workflow_run_recovers_when_model_requests_update_active_without_booking(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Hello I would like to book in a car wash at 4pm, are you available?",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.93,
                "conversation_summary": "Customer wants to book a car wash at 4pm.",
                "extracted_fields": {
                    "day": "today",
                    "time": "4pm",
                    "car_type": "unknown",
                    "wash_type": "unknown",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "day": "today",
                    "time": "4pm",
                    "car_type": "sedan",
                    "wash_type": "full wash",
                },
                "reason": "Treat as ongoing booking.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert "Booking saved for Car Wash Intake:" in result["summary"]
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "completed"
    assert bookings[0]["extracted_fields"]["time"] == "4pm"


@pytest.mark.asyncio
async def test_intake_workflow_reply_uses_instagram_text_argument(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Hello I would like to book a car wash at 4pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.92,
                "conversation_summary": "Customer wants a car wash booking but details are missing.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "What day is this for, what car type do you have, and do you want a full wash, exterior only, or interior only?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing details before saving.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert len(composio.execute_calls) == 1
    sent = composio.execute_calls[0]
    assert sent["tool_slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"
    assert sent["arguments"]["text"].startswith("What day is this for")
    assert "message" not in sent["arguments"]


@pytest.mark.asyncio
async def test_intake_workflow_emits_observability_for_successful_save_and_reply(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Book me tomorrow at 3pm for a full wash on my SUV.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Customer wants a booking tomorrow at 3pm.",
                "extracted_fields": {
                    "day": "tomorrow",
                    "time": "3pm",
                    "car_type": "SUV",
                    "wash_type": "full wash",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Booked for tomorrow at 3pm.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "day": "tomorrow",
                    "time": "3pm",
                    "car_type": "SUV",
                    "wash_type": "full wash",
                },
                "reason": "All required fields are present.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    event_names = [item["event"] for item in runtime.behavior_events]
    assert event_names == [
        "intake.conversation.start",
        "intake.decision.start",
        "intake.decision.ok",
        "intake.apply.start",
        "intake.sink_write.start",
        "intake.sink_write.ok",
        "intake.reply.start",
        "intake.reply.ok",
        "intake.apply.ok",
        "intake.conversation.complete",
    ]
    apply_ok = next(item for item in runtime.behavior_events if item["event"] == "intake.apply.ok")
    assert apply_ok["booking_id"].startswith("bkg_")
    assert apply_ok["status"] == "completed"
    decision_ok = next(item for item in runtime.behavior_events if item["event"] == "intake.decision.ok")
    assert decision_ok["workflow_id"] == workflow["workflow_id"]
    assert decision_ok["conversation_id"] == "conv_1"
    assert decision_ok["save_payload"]["wash_type"] == "full wash"
    assert runtime.behavior_events[0]["event"] == "intake.conversation.start"
    assert runtime.behavior_events[0]["customer_id"] == workflow["customer_id"]


@pytest.mark.asyncio
async def test_intake_workflow_retries_with_execution_feedback_after_reply_failure(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Hello I would like to book a car wash at 4pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.92,
                "conversation_summary": "Customer wants a car wash booking but details are missing.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "What day is this for, what car type is it, and do you want full, exterior, or interior only?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing details before saving.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Retry the follow-up after the prior send failed.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "Sorry, what day is this for, what car type is it, and do you want full, exterior, or interior only?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Retry with corrected outbound action after execution feedback.",
            },
        ]
    )
    composio = _FailingReplyOnceComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is True
    assert len(runtime.calls) == 2
    feedback = runtime.calls[1]["execution_feedback"]
    assert isinstance(feedback, list)
    assert feedback
    assert feedback[0]["phase"] == "reply_execution"
    assert "Following fields are missing" in feedback[0]["error"]
    assert feedback[0]["prior_decision"]["extracted_fields"] == {"time": "4pm"}
    assert feedback[0]["prior_decision"]["save_payload"] == {}
    assert len(composio.execute_calls) == 2
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "active"


@pytest.mark.asyncio
async def test_intake_workflow_emits_observability_for_reply_failure(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Hello I would like to book a car wash at 4pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.92,
                "conversation_summary": "Customer wants a car wash booking but details are missing.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "What day is this for?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing details before saving.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.92,
                "conversation_summary": "Customer still wants a car wash booking but details are missing.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "What day is this for?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing details before saving.",
            },
        ]
    )
    composio = _AlwaysFailingReplyComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    result = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert result["ok"] is False
    event_names = [item["event"] for item in runtime.behavior_events]
    assert "intake.reply.error" in event_names
    assert "intake.apply.error" in event_names
    assert "intake.conversation.error" in event_names
    reply_error = next(item for item in runtime.behavior_events if item["event"] == "intake.reply.error")
    assert "temporary send failure" in reply_error["error"]


@pytest.mark.asyncio
async def test_intake_workflow_failed_apply_does_not_advance_cursor_and_retries_same_inbound(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
        "latest_message_id": "msg_1",
        "latest_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_message_sender_id": "cust_1",
        "conversation_updated_time": "2026-04-07T08:00:00+00:00",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Hello I would like to book a car wash at 4pm.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.92,
                "conversation_summary": "Customer wants a car wash booking but details are missing.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "What day is this for, what car type is it, and do you want full, exterior, or interior only?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing details before saving.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.92,
                "conversation_summary": "Customer wants a car wash booking but details are missing.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": ["day", "car_type", "wash_type"],
                "reply_action": "send_reply",
                "reply_text": "What day is this for, what car type is it, and do you want full, exterior, or interior only?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need missing details before saving.",
            },
        ]
    )
    composio = _AlwaysFailingReplyComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    first = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )
    cursor_after_first = service._get_cursor(  # noqa: SLF001
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    second = await service.run_workflow(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
    )

    assert first["ok"] is False
    assert second["ok"] is False
    assert cursor_after_first == {}
    assert len(runtime.calls) == 4


@pytest.mark.asyncio
async def test_intake_workflow_repeat_request_lifecycle_and_composio_sink(tmp_path: Path) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Need a car wash tomorrow 3pm, SUV, interior and exterior.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Initial booking.",
                "extracted_fields": {
                    "day": "tomorrow",
                    "time": "3pm",
                    "car_type": "SUV",
                    "wash_type": "interior and exterior",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "day": "tomorrow",
                    "time": "3pm",
                    "car_type": "SUV",
                    "wash_type": "interior and exterior",
                },
                "reason": "Initial booking complete.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.98,
                "conversation_summary": "Customer changed the time.",
                "extracted_fields": {"time": "4pm"},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "edit_recent_completed",
                "save_payload": {
                    "day": "tomorrow",
                    "time": "4pm",
                    "car_type": "SUV",
                    "wash_type": "interior and exterior",
                },
                "reason": "Edit within edit window.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.99,
                "conversation_summary": "Customer started a second booking.",
                "extracted_fields": {
                    "day": "tomorrow",
                    "time": "5pm",
                    "car_type": "Sedan",
                    "wash_type": "exterior",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "day": "tomorrow",
                    "time": "5pm",
                    "car_type": "Sedan",
                    "wash_type": "exterior",
                },
                "reason": "New booking after edit window.",
            },
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time", "car_type", "wash_type"],
        sink_type="generic_composio_write",
        sink_config={
            "toolkit": "crm",
            "operation_hint": "upsert booking",
            "field_mapping": {
                "booking_id": "booking_id",
                "day": "day",
                "time": "time",
                "vehicle": "car_type",
                "wash": "wash_type",
            },
            "static_arguments": {"status": "confirmed"},
        },
    )
    assert workflow["sink_config"]["toolkit"] == "crm"
    assert workflow["sink_config"]["operation_hint"] == "upsert booking"
    assert "tool_slug" not in workflow["sink_config"]

    first = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])
    assert first["ok"] is True
    initial_bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(initial_bookings) == 1
    original_booking_id = initial_bookings[0]["booking_id"]
    assert composio.execute_calls[-1]["tool_slug"] == "CRM_UPSERT_BOOKING"
    assert composio.execute_calls[-1]["arguments"]["vehicle"] == "SUV"

    summary["latest_inbound_message_id"] = "msg_2"
    summary["latest_inbound_message_created_time"] = "2026-04-07T08:30:00+00:00"
    conversation["data"]["messages"]["data"][0]["id"] = "msg_2"
    conversation["data"]["messages"]["data"][0]["created_time"] = "2026-04-07T08:30:00+00:00"
    conversation["data"]["messages"]["data"][0]["message"] = "Actually make it 4pm."
    second = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])
    assert second["ok"] is True
    edited_bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(edited_bookings) == 1
    assert edited_bookings[0]["booking_id"] == original_booking_id
    assert edited_bookings[0]["extracted_fields"]["time"] == "4pm"

    with service._conn() as conn:  # noqa: SLF001
        conn.execute(
            "UPDATE intake_bookings SET edit_window_until = ? WHERE booking_id = ?",
            ("2026-04-07T07:00:00+00:00", original_booking_id),
        )
        conn.commit()

    summary["latest_inbound_message_id"] = "msg_3"
    summary["latest_inbound_message_created_time"] = "2026-04-07T12:30:00+00:00"
    conversation["data"]["messages"]["data"][0]["id"] = "msg_3"
    conversation["data"]["messages"]["data"][0]["created_time"] = "2026-04-07T12:30:00+00:00"
    conversation["data"]["messages"]["data"][0]["message"] = "Also book my sedan at 5pm."
    third = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])
    assert third["ok"] is True
    final_bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(final_bookings) == 2
    booking_ids = {item["booking_id"] for item in final_bookings}
    assert original_booking_id in booking_ids
    assert len(booking_ids) == 2


@pytest.mark.asyncio
async def test_google_sheets_sink_normalizes_prefixed_slug_and_builds_headers_rows(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Book my sedan tomorrow at 3pm for a full wash.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a booking.",
                "extracted_fields": {
                    "date": "tomorrow",
                    "time": "3pm",
                    "car_type": "sedan",
                    "wash_type": "full wash",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "date": "tomorrow",
                    "time": "3pm",
                    "car_type": "sedan",
                    "wash_type": "full wash",
                },
                "reason": "All fields present.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["date", "time", "car_type", "wash_type"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "date": "Date",
                "time": "Time",
                "car_type": "Car Type",
                "wash_type": "Wash Type",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_123",
                "sheetName": "Sheet1",
            },
        },
    )
    assert workflow["sink_config"]["toolkit"] == "googlesheets"
    assert "tool_slug" not in workflow["sink_config"]

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    call = composio.execute_calls[-1]
    assert call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"
    assert call["arguments"]["sheetName"] == "Sheet1"
    assert call["arguments"]["spreadsheetId"] == "sheet_123"
    assert call["arguments"]["keyColumn"] == "Booking ID"
    headers = call["arguments"]["headers"]
    row = call["arguments"]["rows"][0]
    assert headers[0] == "Booking ID"
    assert set(headers[1:]) == {"Date", "Time", "Car Type", "Wash Type"}
    mapped = dict(zip(headers[1:], row[1:], strict=False))
    assert mapped == {
        "Date": "tomorrow",
        "Time": "3pm",
        "Car Type": "sedan",
        "Wash Type": "full wash",
    }


@pytest.mark.asyncio
async def test_ready_save_merges_extracted_fields_into_save_payload_before_validation(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Семен, телефон 89516767677, нужна 3х-фазная мойка Rolls-Royce Cullinan.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer provided the final phone number.",
                "extracted_fields": {
                    "клиент": "Семен Артурыч",
                    "телефон клиента": "89516767677",
                    "тип услуги": "3х-фазная детейлинг-мойка",
                    "модель автомобиля": "Rolls-Royce Cullinan",
                    "время записи": "пятница, полдень",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "клиент": "Семен Артурыч",
                    "телефон клиента": "89516767677",
                    "модель автомобиля": "Rolls-Royce Cullinan",
                },
                "reason": "The model forgot to duplicate service type into save_payload.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="AutoSpa Intake",
        intent_description="Записывать клиентов.",
        required_fields=["клиент", "телефон клиента", "тип услуги", "модель автомобиля"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "клиент": "клиент",
                "телефон клиента": "телефон клиента",
                "тип услуги": "тип услуги",
                "модель автомобиля": "модель автомобиля",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_123",
                "sheetName": "Лист1",
            },
        },
    )

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    sink_calls = [call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"]
    assert len(sink_calls) == 1
    written = dict(
        zip(
            sink_calls[0]["arguments"]["headers"],
            sink_calls[0]["arguments"]["rows"][0],
            strict=False,
        )
    )
    assert written["тип услуги"] == "3х-фазная детейлинг-мойка"
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert bookings[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_google_sheets_sink_normalizes_aliases_reversed_booking_mapping_and_blank_cells(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Book a wash.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer wants a booking.",
                "extracted_fields": {"service": "wash"},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {"service": "wash"},
                "reason": "All required fields are present.",
            }
        ]
    )
    composio = _FakeComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle bookings.",
        required_fields=["service"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "Booking ID": "booking_id",
                "service": "Service",
                "optional_note": "Optional Note",
            },
            "static_arguments": {
                "spreadsheet_id": "sheet_123",
                "sheet_name": "Лист1",
            },
        },
    )

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    call = composio.execute_calls[-1]
    assert call["arguments"]["spreadsheetId"] == "sheet_123"
    assert call["arguments"]["sheetName"] == "Лист1"
    assert "spreadsheet_id" not in call["arguments"]
    assert "sheet_name" not in call["arguments"]
    assert call["arguments"]["keyColumn"] == "Booking ID"
    headers = call["arguments"]["headers"]
    row = call["arguments"]["rows"][0]
    assert headers[0] == "Booking ID"
    written = dict(zip(headers, row, strict=False))
    assert written["Booking ID"]
    assert written["Service"] == "wash"
    assert written["Optional Note"] == ""


@pytest.mark.asyncio
async def test_google_sheets_sink_auto_resolves_single_unknown_sheet_name_at_setup(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Нужна мойка завтра в 10, телефон +79990000001.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Клиент хочет записаться на мойку.",
                "extracted_fields": {
                    "service": "Мойка",
                    "time": "завтра 10:00",
                    "phone": "+79990000001",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {"service": "Мойка"},
                "reason": "All required fields are present.",
            }
        ]
    )
    composio = _FakeComposio(
        summary,
        conversation,
        sheet_names_by_spreadsheet={"sheet_123": ["Заявки"]},
    )
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="AutoSpa Intake",
        intent_description="Записывать клиентов на мойку.",
        required_fields=["service"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {"service": "Тип услуги", "phone": "Телефон"},
            "static_arguments": {"spreadsheet_id": "sheet_123"},
        },
    )

    assert workflow["sink_config"]["static_arguments"] == {
        "spreadsheetId": "sheet_123",
        "sheetName": "Заявки",
    }
    assert composio.list_sheet_names_calls == [
        {
            "customer_id": "telegram_123",
            "spreadsheet_id": "sheet_123",
            "connected_account_id": None,
        }
    ]

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    sink_calls = [call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"]
    assert len(sink_calls) == 1
    assert sink_calls[0]["arguments"]["spreadsheetId"] == "sheet_123"
    assert sink_calls[0]["arguments"]["sheetName"] == "Заявки"
    written = dict(
        zip(
            sink_calls[0]["arguments"]["headers"],
            sink_calls[0]["arguments"]["rows"][0],
            strict=False,
        )
    )
    assert written["Тип услуги"] == "Мойка"
    assert written["Телефон"] == "+79990000001"


@pytest.mark.asyncio
async def test_telegram_business_intake_can_upsert_partial_google_sheets_row(
    tmp_path: Path,
) -> None:
    customer_id = "telegram_123"
    business_connection_id = "bc_partial"
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.96,
                "conversation_summary": "Customer opened a booking.",
                "extracted_fields": {"service": "wash"},
                "missing_fields": ["name", "phone", "time"],
                "reply_action": "send_reply",
                "reply_text": "What name, phone, and time should I use?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "sink_action": "upsert_partial",
                "sink_payload": {
                    "incoming_user_id": "999",
                    "service": "wash",
                },
                "reason": "Workflow asks to record source identity on first contact.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.98,
                "conversation_summary": "Customer provided all booking details.",
                "extracted_fields": {
                    "name": "Alice",
                    "phone": "+79990000001",
                    "time": "tomorrow 10am",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Booked for tomorrow at 10am.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "name": "Alice",
                    "phone": "+79990000001",
                    "time": "tomorrow 10am",
                },
                "sink_action": "none",
                "sink_payload": {},
                "reason": "All required fields are present.",
            },
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="Partial Row Intake",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description="Collect booking details and record lead identity immediately.",
        required_fields=["name", "phone", "time"],
        assistant_instructions="Record incoming_user_id and username on first contact before all fields are collected.",
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "incoming_user_id": "Telegram ID",
                "username": "Username",
                "service": "Service",
                "name": "Name",
                "phone": "Phone",
                "time": "Time",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_123",
                "sheetName": "Bookings",
            },
        },
    )

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=555,
            user_id=999,
            username="alice",
            message_id=10,
            text="Hi, I need a wash.",
            date=int(datetime.now(UTC).timestamp()),
        ),
    )

    first = await service.run_workflow(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert first["ok"] is True
    assert len(composio.execute_calls) == 1
    first_write = dict(
        zip(
            composio.execute_calls[0]["arguments"]["headers"],
            composio.execute_calls[0]["arguments"]["rows"][0],
            strict=False,
        )
    )
    assert first_write["Telegram ID"] == "999"
    assert first_write["Username"] == "alice"
    assert first_write["Service"] == "wash"
    bookings = service.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    )
    assert bookings[0]["status"] == "active"
    assert bookings[0]["sink_write_status"] == "partial_succeeded"
    assert bookings[0]["extracted_fields"]["incoming_user_id"] == "999"

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=555,
            user_id=999,
            username="alice",
            message_id=11,
            text="Alice, +79990000001, tomorrow 10am.",
            date=int(datetime.now(UTC).timestamp()),
        ),
    )

    second = await service.run_workflow(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        event_type="telegram_business_webhook",
    )

    assert second["ok"] is True
    assert len(composio.execute_calls) == 2
    second_write = dict(
        zip(
            composio.execute_calls[1]["arguments"]["headers"],
            composio.execute_calls[1]["arguments"]["rows"][0],
            strict=False,
        )
    )
    assert second_write["Booking ID"] == first_write["Booking ID"]
    assert second_write["Telegram ID"] == "999"
    assert second_write["Name"] == "Alice"
    assert second_write["Phone"] == "+79990000001"
    assert second_write["Time"] == "tomorrow 10am"
    bookings = service.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id="555",
    )
    assert bookings[0]["status"] == "completed"
    assert bookings[0]["sink_write_status"] == "succeeded"


def test_google_sheets_sink_requires_explicit_sheet_name_when_target_has_multiple_tabs(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="Нужна мойка.",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    composio = _FakeComposio(
        summary,
        conversation,
        sheet_names_by_spreadsheet={"sheet_123": ["Заявки", "Архив"]},
    )
    service, _, _, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=composio,
    )

    with pytest.raises(ValueError, match="multiple sheets: Заявки, Архив"):
        service.upsert_workflow(
            customer_id="telegram_123",
            name="AutoSpa Intake",
            intent_description="Записывать клиентов на мойку.",
            required_fields=["service"],
            sink_type="google_sheets_composio",
            sink_config={
                "toolkit": "googlesheets",
                "field_mapping": {"service": "Тип услуги"},
                "static_arguments": {"spreadsheetId": "sheet_123"},
            },
        )


@pytest.mark.asyncio
async def test_autospa_xlsx_telegram_inbound_books_wash_and_tire_to_google_sheets(
    tmp_path: Path,
) -> None:
    raw_bytes = sample_vehicle_services_xlsx_bytes()

    inspection = inspect_uploaded_file_structure(
        raw_bytes=raw_bytes,
        filename=SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
        mime_type=SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
        search_terms=["Мойка", "Шиномонтаж"],
    )
    sheets = inspection["structure"]["sheets"]
    assert any(sheet["name"] == "Мойка" for sheet in sheets)
    assert any(sheet["name"] == "Шиномонтаж" for sheet in sheets)

    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.96,
                "conversation_summary": "Lead asks for a source-backed wash price.",
                "extracted_fields": {
                    "service_category": "Мойка",
                    "service_name": "2х-фазная мойка кузова",
                    "vehicle_type": "S-Class / SUV",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "needs_business_knowledge": True,
                "business_knowledge_query": "2х-фазная мойка кузова SUV цена",
                "reason": "Need source-backed wash price.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.96,
                "conversation_summary": "Lead asks for a source-backed wash price.",
                "extracted_fields": {
                    "service_category": "Мойка",
                    "service_name": "2х-фазная мойка кузова",
                    "vehicle_type": "S-Class / SUV",
                    "quoted_price": "1200",
                },
                "missing_fields": ["time", "lead_name", "phone"],
                "reply_action": "send_reply",
                "reply_text": (
                    "2х-фазная мойка кузова для S-Class / SUV стоит 1200. "
                    "На какое время вас записать?"
                ),
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Answered from workflow knowledge but still missing booking details.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Lead finished the wash booking.",
                "extracted_fields": {
                    "service_category": "Мойка",
                    "service_name": "2х-фазная мойка кузова",
                    "vehicle_type": "Toyota RAV4 / SUV",
                    "quoted_price": "1200",
                    "date": "tomorrow",
                    "time": "10:00",
                    "lead_name": "Алексей",
                    "phone": "+79990000001",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Записал на 2х-фазную мойку завтра в 10:00.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "service_category": "Мойка",
                    "service_name": "2х-фазная мойка кузова",
                    "vehicle_type": "Toyota RAV4 / SUV",
                    "quoted_price": "1200",
                    "date": "tomorrow",
                    "time": "10:00",
                    "lead_name": "Алексей",
                    "phone": "+79990000001",
                },
                "reason": "All wash booking fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Lead wants tire fitting.",
                "extracted_fields": {
                    "service_category": "Шиномонтаж",
                    "service_name": "Комплект 19`R",
                    "vehicle_type": "кросовер + низкий профиль",
                    "date": "Friday",
                    "time": "15:00",
                    "lead_name": "Мария",
                    "phone": "+79990000002",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "needs_business_knowledge": True,
                "business_knowledge_query": "Шиномонтаж Комплект 19R кросовер низкий профиль цена",
                "reason": "Need source-backed tire fitting price.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Lead wants tire fitting.",
                "extracted_fields": {
                    "service_category": "Шиномонтаж",
                    "service_name": "Комплект 19`R",
                    "vehicle_type": "кросовер + низкий профиль",
                    "quoted_price": "4000",
                    "date": "Friday",
                    "time": "15:00",
                    "lead_name": "Мария",
                    "phone": "+79990000002",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Записал на шиномонтаж 19R в пятницу в 15:00.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "service_category": "Шиномонтаж",
                    "service_name": "Комплект 19`R",
                    "vehicle_type": "кросовер + низкий профиль",
                    "quoted_price": "4000",
                    "date": "Friday",
                    "time": "15:00",
                    "lead_name": "Мария",
                    "phone": "+79990000002",
                },
                "reason": "All tire fitting booking fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": False,
                "confidence": 0.2,
                "conversation_summary": "Lead asks about an out-of-scope PPF service.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "PPF is outside the workflow scope.",
            },
        ]
    )
    composio = _FakeComposio(
        {
            "conversation_id": "unused",
            "recipient_id": "unused",
            "latest_inbound_message_id": "unused",
            "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        },
        _instagram_conversation(
            conversation_id="unused",
            latest_message_id="unused",
            latest_message_text="unused",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, telegram_business, file_vault = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    customer_id = "telegram_123"
    business_connection_id = "bc_autospa"
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    source_file = file_vault.ingest_file(
        customer_id=customer_id,
        chat_id=777,
        kind="document",
        telegram_file_id=None,
        original_filename=SAMPLE_VEHICLE_SERVICES_XLSX_FILENAME,
        mime_type=SAMPLE_VEHICLE_SERVICES_XLSX_MIME_TYPE,
        caption="Sample service price list source for workflow setup",
        raw_bytes=raw_bytes,
    )
    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="AutoSpa Мойка + Шиномонтаж",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description=(
            "Handle Telegram Business inbound leads only for AutoSpa car wash "
            "and tire fitting requests. Answer source-backed price questions, "
            "collect booking details, and save completed bookings."
        ),
        required_fields=[
            "service_category",
            "service_name",
            "vehicle_type",
            "date",
            "time",
            "lead_name",
            "phone",
        ],
        field_guidance={
            "service_category": "Must be exactly Мойка or Шиномонтаж.",
            "quoted_price": "Use the bound source knowledge; do not invent prices.",
        },
        assistant_instructions=(
            "Scope is only Мойка and Шиномонтаж. If another service is requested, ignore or clarify scope."
        ),
        knowledge_file_ids=[str(source_file["id"])],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "service_category": "Category",
                "service_name": "Service",
                "vehicle_type": "Vehicle",
                "quoted_price": "Quoted Price",
                "date": "Date",
                "time": "Time",
                "lead_name": "Lead Name",
                "phone": "Phone",
                "conversation_id": "Conversation ID",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_autospa_test",
                "sheetName": "Bookings",
            },
        },
    )

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=1,
            text="Сколько стоит 2х-фазная мойка для SUV? Можно завтра?",
            date=1_775_552_400,
        ),
    )
    first = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    assert first["ok"] is True
    assert composio.execute_calls == []
    assert any("1200" in str(item.get("text", "")) for item in telegram_business.client.sent_messages)

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=2,
            text="Алексей, +79990000001. Toyota RAV4, завтра в 10:00.",
            date=1_775_552_460,
        ),
    )
    wash = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    assert wash["ok"] is True

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5102,
            user_id=9102,
            username="tire_lead",
            message_id=1,
            text="Нужен шиномонтаж 19R для кроссовера с низким профилем, Мария +79990000002, пятница 15:00.",
            date=1_775_552_520,
        ),
    )
    tire = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    assert tire["ok"] is True

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5103,
            user_id=9103,
            username="ppf_lead",
            message_id=1,
            text="Сколько стоит PPF пакет?",
            date=1_775_552_580,
        ),
    )
    out_of_scope = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    assert out_of_scope["ok"] is True

    assert len(runtime.calls) == 6
    assert runtime.calls[0]["workflow"]["knowledge_answer"] == ""
    assert runtime.calls[2]["workflow"]["knowledge_answer"] == ""
    assert runtime.calls[3]["workflow"]["knowledge_answer"] == ""
    assert runtime.calls[5]["workflow"]["knowledge_answer"] == ""
    all_knowledge_text = (
        str(runtime.calls[1]["workflow"]["knowledge_answer"])
        + str(runtime.calls[4]["workflow"]["knowledge_answer"])
    )
    assert str(runtime.calls[1]["workflow"]["knowledge_answer"]).strip()
    assert str(runtime.calls[4]["workflow"]["knowledge_answer"]).strip()
    assert "2х-фазная мойка кузова" in all_knowledge_text
    assert "Комплект 19`R" in all_knowledge_text

    sink_calls = [
        call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"
    ]
    assert len(sink_calls) == 2
    assert all(call["arguments"]["spreadsheetId"] == "sheet_autospa_test" for call in sink_calls)
    assert all(call["arguments"]["sheetName"] == "Bookings" for call in sink_calls)
    written_rows = [
        dict(zip(call["arguments"]["headers"], call["arguments"]["rows"][0], strict=False))
        for call in sink_calls
    ]
    assert written_rows[0]["Category"] == "Мойка"
    assert written_rows[0]["Service"] == "2х-фазная мойка кузова"
    assert written_rows[0]["Quoted Price"] == "1200"
    assert written_rows[1]["Category"] == "Шиномонтаж"
    assert written_rows[1]["Service"] == "Комплект 19`R"
    assert written_rows[1]["Quoted Price"] == "4000"

    bookings = service.list_bookings(customer_id=customer_id, workflow_id=workflow["workflow_id"])
    completed = [item for item in bookings if item["status"] == "completed"]
    assert len(completed) == 2
    assert {item["conversation_id"] for item in completed} == {"5101", "5102"}
    assert all(item["sink_write_status"] == "succeeded" for item in completed)


@pytest.mark.asyncio
async def test_sink_failure_retries_until_recovery_limit_then_stops_without_customer_confirmation(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="7pm is ok",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer confirmed 7pm.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "reason": "All required fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Retry the same sheet write after first sink failure.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "reason": "Retry after sink failure.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Retry the same sheet write after second sink failure.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "reason": "Retry after second sink failure.",
            },
        ]
    )
    composio = _FailingSinkComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["date", "time", "car_type", "wash_type"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "date": "Date",
                "time": "Time",
                "car_type": "Car Type",
                "wash_type": "Wash Type",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_123",
                "sheetName": "Sheet1",
            },
        },
    )

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is False
    assert len(runtime.calls) == 3
    assert all(call["tool_slug"] != "INSTAGRAM_SEND_TEXT_MESSAGE" for call in composio.execute_calls)
    sink_calls = [call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"]
    assert len(sink_calls) == 3
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["sink_write_status"] == "failed"


@pytest.mark.asyncio
async def test_sink_failure_retries_with_execution_feedback_and_redoes_sheet_write(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="7pm is ok",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer confirmed 7pm.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "reason": "All required fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Retry the sheet write after sink failure.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "reason": "Retry after sink failure.",
            },
        ]
    )
    composio = _FailingSinkOnceComposio(summary, conversation)
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["date", "time", "car_type", "wash_type"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "date": "Date",
                "time": "Time",
                "car_type": "Car Type",
                "wash_type": "Wash Type",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_123",
                "sheetName": "Sheet1",
            },
        },
    )

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    assert len(runtime.calls) == 2
    feedback = runtime.calls[1]["execution_feedback"]
    assert isinstance(feedback, list)
    assert feedback
    assert feedback[0]["phase"] == "sink_execution"
    sink_calls = [call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"]
    assert len(sink_calls) == 2
    reply_calls = [call for call in composio.execute_calls if call["tool_slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"]
    assert len(reply_calls) == 1
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "completed"
    assert bookings[0]["sink_write_status"] == "succeeded"


@pytest.mark.asyncio
async def test_telegram_business_completed_booking_without_model_reply_sends_confirmation(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Клиент дал все данные для записи.",
                "extracted_fields": {
                    "service_category": "Мойка",
                    "service_name": "2х-фазная мойка кузова",
                    "vehicle": "Toyota RAV4",
                    "desired_date": "завтра",
                    "desired_time": "10:00",
                    "client_name": "Алексей",
                    "phone": "+79990000001",
                    "quoted_price": "1200",
                },
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "service_category": "Мойка",
                    "service_name": "2х-фазная мойка кузова",
                    "vehicle": "Toyota RAV4",
                    "desired_date": "завтра",
                    "desired_time": "10:00",
                    "client_name": "Алексей",
                    "phone": "+79990000001",
                    "quoted_price": "1200",
                },
                "reason": "All fields are present, but the model omitted a reply.",
            }
        ]
    )
    composio = _FakeComposio({}, {})
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    customer_id = "telegram_123"
    business_connection_id = "bc_autospa"
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="AutoSpa Мойка",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description="Записывать клиентов на мойку.",
        required_fields=[
            "service_category",
            "service_name",
            "vehicle",
            "desired_date",
            "desired_time",
            "client_name",
            "phone",
        ],
        assistant_instructions="Отвечай клиенту на русском языке.",
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "service_category": "Category",
                "service_name": "Service",
                "vehicle": "Vehicle",
                "desired_date": "Date",
                "desired_time": "Time",
                "client_name": "Lead Name",
                "phone": "Phone",
                "quoted_price": "Quoted Price",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_autospa_test",
                "sheetName": "Bookings",
            },
        },
    )
    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=1,
            text="Алексей, Toyota RAV4, завтра в 10:00, телефон +79990000001.",
            date=1_775_552_400,
        ),
    )

    result = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    assert len(composio.execute_calls) == 1
    sent = telegram_business.client.sent_messages
    assert len(sent) == 1
    assert sent[0]["chat_id"] == "5101"
    assert "запись сохранена" in sent[0]["text"].lower()
    assert "2х-фазная мойка кузова" in sent[0]["text"]
    assert "10:00" in sent[0]["text"]
    assert "1200" in sent[0]["text"]


@pytest.mark.asyncio
async def test_telegram_business_ready_save_allows_empty_note_field(
    tmp_path: Path,
) -> None:
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Клиент выбрал свободного мастера.",
                "extracted_fields": {
                    "lead_name": "Кристиан",
                    "phone": "+35799889577",
                    "car_model": "Mercedes SLS кабриолет",
                    "car_class": "C-Class",
                    "service_name": "3х-фазная детейлинг-мойка",
                    "service_price": "3100",
                    "add_ons": "нет",
                    "appointment_date": "2026-05-07",
                    "appointment_time": "17:00",
                    "master": "свободный мастер",
                    "note": "",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Записал к свободному мастеру на сегодня в 17:00.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "lead_name": "Кристиан",
                    "phone": "+35799889577",
                    "car_model": "Mercedes SLS кабриолет",
                    "car_class": "C-Class",
                    "service_name": "3х-фазная детейлинг-мойка",
                    "service_price": "3100",
                    "add_ons": "нет",
                    "appointment_date": "2026-05-07",
                    "appointment_time": "17:00",
                    "master": "свободный мастер",
                    "note": "",
                },
                "reason": "No extra note was provided.",
            }
        ]
    )
    composio = _FakeComposio({}, {})
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    customer_id = "telegram_123"
    business_connection_id = "bc_autospa"
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="AutoSpa Мойка",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description="Записывать клиентов на мойку.",
        required_fields=[
            "lead_name",
            "phone",
            "car_model",
            "car_class",
            "service_name",
            "service_price",
            "add_ons",
            "appointment_date",
            "appointment_time",
            "master",
            "note",
        ],
        assistant_instructions="Отвечай клиенту на русском языке.",
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "lead_name": "Lead Name",
                "phone": "Phone",
                "car_model": "Car",
                "car_class": "Class",
                "service_name": "Service",
                "service_price": "Price",
                "add_ons": "Add-ons",
                "appointment_date": "Date",
                "appointment_time": "Time",
                "master": "Master",
                "note": "Note",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_autospa_test",
                "sheetName": "Bookings",
            },
        },
    )
    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=1,
            text="Запишите к свободному",
            date=1_775_552_400,
        ),
    )

    result = await service.run_workflow(customer_id=customer_id, workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    assert len(runtime.calls) == 1
    assert len(composio.execute_calls) == 1
    assert telegram_business.client.sent_messages[0]["text"] == "Записал к свободному мастеру на сегодня в 17:00."


@pytest.mark.asyncio
async def test_telegram_business_cancel_ready_save_preserves_cancellation_reply(
    tmp_path: Path,
) -> None:
    initial_fields = {
        "service_category": "Мойка",
        "service_name": "2х-фазная мойка кузова",
        "vehicle": "Toyota RAV4",
        "desired_date": "завтра",
        "desired_time": "10:00",
        "client_name": "Алексей",
        "phone": "+79990000001",
        "quoted_price": "1200",
    }
    cancel_payload = {**initial_fields, "desired_time": "11:00", "status": "cancelled"}
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Клиент дал все данные для записи.",
                "extracted_fields": initial_fields,
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": initial_fields,
                "reason": "All fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.99,
                "conversation_summary": "Клиент отменяет существующую запись.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "mark_cancelled",
                "reply_text": "Алексей, вашу запись на 2х-фазную мойку кузова завтра в 11:00 отменена.",
                "ready_to_save": True,
                "booking_action": "edit_recent_completed",
                "save_payload": cancel_payload,
                "reason": "Clear cancellation inside the edit window.",
            },
        ]
    )
    composio = _FakeComposio({}, {})
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    customer_id = "telegram_123"
    business_connection_id = "bc_autospa"
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="AutoSpa Мойка",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description="Записывать клиентов на мойку.",
        required_fields=[
            "service_category",
            "service_name",
            "vehicle",
            "desired_date",
            "desired_time",
            "client_name",
            "phone",
        ],
        assistant_instructions="Отвечай клиенту на русском языке.",
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "service_category": "Category",
                "service_name": "Service",
                "vehicle": "Vehicle",
                "desired_date": "Date",
                "desired_time": "Time",
                "client_name": "Lead Name",
                "phone": "Phone",
                "quoted_price": "Quoted Price",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_autospa_test",
                "sheetName": "Bookings",
            },
        },
    )
    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=1,
            text="Алексей, Toyota RAV4, завтра в 10:00, телефон +79990000001.",
            date=1_775_552_400,
        ),
    )
    first_result = await service.run_workflow(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )
    assert first_result["ok"] is True

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=2,
            text="Тогда отмените запись, пожалуйста.",
            date=1_775_552_460,
        ),
    )
    second_result = await service.run_workflow(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )

    assert second_result["ok"] is True
    assert len(composio.execute_calls) == 2
    sent = telegram_business.client.sent_messages
    assert len(sent) == 2
    assert "отменена" in sent[1]["text"].lower()
    assert "запись сохранена" not in sent[1]["text"].lower()
    bookings = service.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id="5101",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "cancelled"


@pytest.mark.asyncio
async def test_telegram_business_cancel_partial_booking_without_required_fields(
    tmp_path: Path,
) -> None:
    partial_fields = {
        "category": "Мойка",
        "conversation_id": "5101",
        "date": "30.04.2026",
        "lead_name": "Алексей",
        "phone": "+79990000001",
        "time": "11:00",
        "vehicle": "SUV",
    }
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Клиент выбрал время, но еще не выбрал конкретную услугу.",
                "extracted_fields": partial_fields,
                "missing_fields": ["service_name", "quoted_price"],
                "reply_action": "send_reply",
                "reply_text": "Какой вариант мойки вас интересует?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Booking is partial.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.99,
                "conversation_summary": "Клиент отменяет частично заполненную запись.",
                "extracted_fields": {"status": "cancelled"},
                "missing_fields": [],
                "reply_action": "mark_cancelled",
                "reply_text": "Алексей, вашу запись на мойку 30.04.2026 в 11:00 отменена.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {"status": "cancelled"},
                "reason": "Clear cancellation of a partial active booking.",
            },
        ]
    )
    composio = _FakeComposio({}, {})
    service, _, _, telegram_business, _ = _mk_service(
        tmp_path,
        runtime=runtime,
        composio=composio,
    )
    customer_id = "telegram_123"
    business_connection_id = "bc_autospa"
    telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False, "first_name": "Kim"},
            "rights": {"can_reply": True},
        }
    )
    workflow = service.upsert_workflow(
        customer_id=customer_id,
        name="AutoSpa Мойка",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": business_connection_id},
        intent_description="Записывать клиентов на мойку.",
        required_fields=[
            "category",
            "service_name",
            "vehicle",
            "date",
            "time",
            "lead_name",
            "phone",
            "quoted_price",
            "conversation_id",
        ],
        assistant_instructions="Отвечай клиенту на русском языке.",
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "category": "Category",
                "service_name": "Service",
                "vehicle": "Vehicle",
                "date": "Date",
                "time": "Time",
                "lead_name": "Lead Name",
                "phone": "Phone",
                "quoted_price": "Quoted Price",
                "conversation_id": "Conversation ID",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_autospa_test",
                "sheetName": "Bookings",
            },
        },
    )
    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=1,
            text="Алексей, SUV, завтра в 11:00, телефон +79990000001.",
            date=1_775_552_400,
        ),
    )
    first_result = await service.run_workflow(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )
    assert first_result["ok"] is True

    telegram_business.upsert_message(
        business_connection_id=business_connection_id,
        customer_id=customer_id,
        message=_telegram_business_inbound(
            business_connection_id=business_connection_id,
            chat_id=5101,
            user_id=9101,
            username="wash_lead",
            message_id=2,
            text="Тогда отмените запись, пожалуйста.",
            date=1_775_552_460,
        ),
    )
    second_result = await service.run_workflow(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
    )

    assert second_result["ok"] is True
    assert composio.execute_calls == []
    sent = telegram_business.client.sent_messages
    assert len(sent) == 2
    assert "отменена" in sent[1]["text"].lower()
    bookings = service.list_bookings(
        customer_id=customer_id,
        workflow_id=workflow["workflow_id"],
        conversation_id="5101",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "cancelled"
    assert bookings[0]["sink_write_status"] == "not_required"


@pytest.mark.asyncio
async def test_sink_failure_can_recover_with_sink_argument_overrides(
    tmp_path: Path,
) -> None:
    summary = {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": "msg_1",
        "latest_inbound_message_created_time": "2026-04-07T08:00:00+00:00",
        "latest_inbound_sender_username": "alice",
    }
    conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_1",
        latest_message_text="7pm is ok",
        latest_message_time="2026-04-07T08:00:00+00:00",
    )
    runtime = _FakeRuntime(
        [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Customer confirmed 7pm.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "create_new_booking",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "sink_arguments": {},
                "reason": "All required fields are present.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.95,
                "conversation_summary": "Recovered by inspecting the sheet and adding the sheet name.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "Great! Your booking is confirmed.",
                "ready_to_save": True,
                "booking_action": "update_active",
                "save_payload": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "sink_arguments": {"sheetName": "Лист1"},
                "reason": "Retry with the discovered sheet name.",
            },
        ]
    )
    composio = _SheetNameRequiredSinkComposio(summary, conversation)
    composio.list_google_sheets_tab_names = None  # type: ignore[method-assign]
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["date", "time", "car_type", "wash_type"],
        sink_type="google_sheets_composio",
        sink_config={
            "toolkit": "googlesheets",
            "field_mapping": {
                "date": "Date",
                "time": "Time",
                "car_type": "Car Type",
                "wash_type": "Wash Type",
            },
            "static_arguments": {
                "spreadsheetId": "sheet_123",
            },
        },
    )

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["ok"] is True
    assert len(runtime.calls) == 2
    feedback = runtime.calls[1]["execution_feedback"]
    assert isinstance(feedback, list)
    assert feedback
    assert feedback[0]["phase"] == "sink_execution"
    sink_calls = [call for call in composio.execute_calls if call["tool_slug"] == "GOOGLESHEETS_UPSERT_ROWS"]
    assert len(sink_calls) == 2
    assert "sheetName" not in sink_calls[0]["arguments"]
    assert sink_calls[1]["arguments"]["sheetName"] == "Лист1"
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["status"] == "completed"
    assert bookings[0]["sink_write_status"] == "succeeded"
