from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from opentulpa.context.file_vault import FileVaultService
from opentulpa.intake.service import IntakeWorkflowService
from opentulpa.interfaces.telegram.business import TelegramBusinessService
from opentulpa.interfaces.telegram.relay import NO_NOTIFY_TOKEN
from opentulpa.scheduler.service import SchedulerService
from opentulpa.skills.service import SkillStoreService


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
    def __init__(self, decisions: list[dict[str, Any]]) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []
        self.posthog_events: list[dict[str, Any]] = []

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.decisions:
            raise RuntimeError("unexpected intake decision call")
        return self.decisions.pop(0)

    def log_behavior_event(self, *, event: str, **fields: Any) -> None:
        self.behavior_events.append({"event": event, **fields})

    def capture_posthog_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.posthog_events.append(
            {
                "event": event,
                "customer_id": customer_id,
                "properties": dict(properties or {}),
            }
        )

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        posthog_event: str | None = None,
        **fields: Any,
    ) -> None:
        self.log_behavior_event(event=event, **fields)
        self.capture_posthog_event(
            event=str(posthog_event or event or "").strip(),
            customer_id=customer_id,
            properties={"behavior_event": event, **fields},
        )


class _FakeComposio:
    enabled = True

    def __init__(self, summary: dict[str, Any], conversation: dict[str, Any]) -> None:
        self.summary = summary
        self.conversation = conversation
        self.execute_calls: list[dict[str, Any]] = []
        self.list_calls = 0
        self.get_calls = 0

    def list_instagram_conversations(
        self,
        *,
        customer_id: str,
        connected_account_id: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        del customer_id, connected_account_id, limit
        self.list_calls += 1
        return {"ok": True, "items": [self.summary]}

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


class _FakeTelegramClient:
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
    ) -> bool:
        self.sent_messages.append(
            {
                "chat_id": str(chat_id),
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": dict(reply_markup or {}) if isinstance(reply_markup, dict) else None,
                "business_connection_id": business_connection_id,
                "reply_to_message_id": reply_to_message_id,
            }
        )
        return True


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
    service = IntakeWorkflowService(
        db_path=tmp_path / "intake.db",
        project_root=tmp_path,
        scheduler=scheduler,
        skill_store=skills,
        composio=composio,
        telegram_business=telegram_business,
        file_vault=file_vault,
        get_agent_runtime=lambda: runtime,
    )
    return service, scheduler, skills, telegram_business, file_vault


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
    assert workflow["schedule"] == "*/5 * * * *"
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
        knowledge_file_ids=[str(record["id"])],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    assert workflow["channel"] == "telegram_business_dm"
    assert workflow["provider"] == "telegram_bot_api"
    assert workflow["assistant_instructions"] == "Be concise and never promise unavailable slots."
    assert workflow["knowledge_file_ids"] == [str(record["id"])]
    skill = skills.get_skill(
        customer_id="telegram_123",
        name=f"intake-workflow-{workflow['workflow_id']}",
        include_files=True,
        include_global=False,
    )
    assert skill is not None
    assert "Telegram Business DMs" in skill["skill_markdown"]
    assert "Always confirm the final appointment time explicitly." in skill["skill_markdown"]
    workflow_file = json.loads(skill["supporting_files"]["workflow.json"])
    assert workflow_file["source_config"] == {"business_connection_id": "bc_123"}
    assert workflow_file["field_guidance"] == {
        "time": "Always confirm the final appointment time explicitly."
    }


@pytest.mark.asyncio
async def test_telegram_business_workflow_upsert_reuses_existing_workflow_for_customer(
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
    second = service.upsert_workflow(
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

    assert second["workflow_id"] == first["workflow_id"]
    assert second["name"] == "Salon Telegram Intake Updated"
    assert second["required_fields"] == ["name", "time", "service"]
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
    assert runtime.calls[0]["workflow"]["assistant_instructions"] == "Be concise and confirm only explicit booking times."
    assert runtime.calls[0]["workflow"]["knowledge_file_ids"] == [str(knowledge["id"])]
    assert runtime.calls[0]["workflow"]["knowledge_files"][0]["id"] == str(knowledge["id"])
    sent = telegram_business.client.sent_messages[0]
    assert sent["chat_id"] == "555"
    assert sent["business_connection_id"] == "bc_123"
    assert sent["reply_to_message_id"] == 10


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
    assert len(runtime.posthog_events) == len(runtime.behavior_events)
    assert runtime.posthog_events[0]["event"] == "intake.conversation.start"


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
async def test_sink_failure_does_not_send_customer_confirmation_or_retry_reply(
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
                "conversation_summary": "Internal sink retry.",
                "extracted_fields": {
                    "date": "April 20",
                    "time": "7pm",
                    "car_type": "sedan",
                    "wash_type": "full wash with rims",
                },
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": "We have a system delay.",
                "ready_to_save": False,
                "booking_action": "update_active",
                "save_payload": {},
                "reason": "Retry after sink failure.",
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
    assert len(runtime.calls) == 1
    assert all(call["tool_slug"] != "INSTAGRAM_SEND_TEXT_MESSAGE" for call in composio.execute_calls)
    bookings = service.list_bookings(
        customer_id="telegram_123",
        workflow_id=workflow["workflow_id"],
        conversation_id="conv_1",
    )
    assert len(bookings) == 1
    assert bookings[0]["sink_write_status"] == "failed"
