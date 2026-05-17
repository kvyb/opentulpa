from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from fastapi.testclient import TestClient
from harness.runner import E2EHarness
from mocks.telegram import FakeTelegramClient

from opentulpa.api.app import create_app
from opentulpa.context.customer_profiles import CustomerProfileService
from opentulpa.core.config import get_settings
from opentulpa.intake import service as intake_service_module
from opentulpa.scheduler.service import SchedulerService

pytestmark = [pytest.mark.e2e, pytest.mark.telegram]


class _DebounceRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []
        self.first_call_started = Event()

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        call_index = len(self.calls)
        if call_index == 1:
            self.first_call_started.set()
            await asyncio.sleep(0.08)
            return {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer asks for a wash.",
                "extracted_fields": {},
                "missing_fields": ["car_model"],
                "reply_action": "send_reply",
                "reply_text": "Какая у вас машина?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need the car model.",
            }
        return {
            "ok": True,
            "matches_workflow": True,
            "confidence": 0.95,
            "conversation_summary": "Customer asks for a wash and provided car model.",
            "extracted_fields": {"car_model": "Rolls-Royce Cullinan"},
            "missing_fields": ["time"],
            "reply_action": "send_reply",
            "reply_text": "Отлично, на какое время записать Rolls-Royce Cullinan?",
            "ready_to_save": False,
            "booking_action": "create_new_booking",
            "save_payload": {},
            "reason": "Need appointment time.",
        }

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        self.behavior_events.append(
            {
                "event": event,
                "customer_id": customer_id,
                **fields,
            }
        )


class _ThreeMessageRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []
        self.first_call_started = Event()

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            self.first_call_started.set()
            await asyncio.sleep(0.08)
            return {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.9,
                "conversation_summary": "Customer asks for a wash.",
                "extracted_fields": {},
                "missing_fields": ["car_model", "time"],
                "reply_action": "send_reply",
                "reply_text": "Какая у вас машина?",
                "ready_to_save": False,
                "booking_action": "create_new_booking",
                "save_payload": {},
                "reason": "Need the car model and time.",
            }
        return {
            "ok": True,
            "matches_workflow": True,
            "confidence": 0.96,
            "conversation_summary": "Customer asks for a wash and split car/time details across messages.",
            "extracted_fields": {"car_model": "BMW sedan", "time": "tomorrow at 10am"},
            "missing_fields": [],
            "reply_action": "send_reply",
            "reply_text": "Ок, BMW sedan на завтра в 10:00.",
            "ready_to_save": False,
            "booking_action": "create_new_booking",
            "save_payload": {},
            "reason": "Use every unanswered customer message.",
        }

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        self.behavior_events.append(
            {
                "event": event,
                "customer_id": customer_id,
                **fields,
            }
        )


class _ParallelRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []
        self.first_call_started = Event()
        self.active = 0
        self.max_active = 0

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            self.first_call_started.set()
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.15)
        finally:
            self.active -= 1
        conversation_id = str(
            kwargs.get("conversation", {}).get("summary", {}).get("conversation_id", "")
        ).strip()
        return {
            "ok": True,
            "matches_workflow": True,
            "confidence": 0.95,
            "conversation_summary": f"Customer {conversation_id} needs intake follow-up.",
            "extracted_fields": {},
            "missing_fields": ["time"],
            "reply_action": "send_reply",
            "reply_text": f"Reply for {conversation_id}",
            "ready_to_save": False,
            "booking_action": "ignore",
            "save_payload": {},
            "reason": "Ask for one missing detail.",
        }

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        self.behavior_events.append(
            {
                "event": event,
                "customer_id": customer_id,
                **fields,
            }
        )


def _business_message(
    *,
    business_connection_id: str,
    lead_chat_id: int,
    lead_user_id: int,
    message_id: int,
    text: str,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000),
        "business_message": {
            "business_connection_id": business_connection_id,
            "message_id": message_id,
            "date": int(time.time()),
            "chat": {"id": lead_chat_id, "type": "private", "username": f"lead_{lead_user_id}"},
            "from": {"id": lead_user_id, "is_bot": False, "username": f"lead_{lead_user_id}"},
            "text": text,
        },
    }


def _wait_until(predicate: Any, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.02)
    return bool(predicate())


def _seed_telegram_business_connection(
    harness: E2EHarness,
    *,
    owner_user_id: int,
    owner_chat_id: int,
    business_connection_id: str,
) -> str:
    harness.client.app.state.telegram_business.upsert_connection(
        {
            "id": business_connection_id,
            "user_chat_id": owner_chat_id,
            "is_enabled": True,
            "user": {
                "id": owner_user_id,
                "is_bot": False,
                "first_name": "Kim",
                "username": "kim",
            },
            "rights": {"can_reply": True},
        }
    )
    return business_connection_id


def _behavior_events(harness: E2EHarness) -> list[dict[str, Any]]:
    if not harness.behavior_log_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in harness.behavior_log_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _create_fake_telegram_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
    customer_profiles: CustomerProfileService | None = None,
) -> tuple[Any, FakeTelegramClient]:
    from opentulpa.api import app as app_module
    from opentulpa.tasks import sandbox as sandbox_module

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS", 0.01)
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 0.01)

    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    fake_telegram = FakeTelegramClient("fake-token")
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_telegram)
    get_settings.cache_clear()

    scheduler = SchedulerService(db_path=tmp_path / "scheduler.sqlite")
    app = create_app(
        agent_runtime=runtime,
        scheduler=scheduler,
        customer_profile_service=customer_profiles,
    )
    return app, fake_telegram


def test_telegram_business_intake_suppresses_stale_reply_from_webhook_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentulpa.api import app as app_module
    from opentulpa.tasks import sandbox as sandbox_module

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS", 0.01)
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 0.01)

    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    fake_telegram = FakeTelegramClient("fake-token")
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_telegram)
    get_settings.cache_clear()

    runtime = _DebounceRuntime()
    scheduler = SchedulerService(db_path=tmp_path / "scheduler.sqlite")
    app = create_app(agent_runtime=runtime, scheduler=scheduler)
    business_connection_id = "bc_e2e_debounce"
    owner_user_id = 321
    owner_chat_id = 654
    lead_chat_id = 777
    lead_user_id = 888
    customer_id = f"telegram_{owner_user_id}"

    with TestClient(app) as client:
        app.state.telegram_business.upsert_connection(
            {
                "id": business_connection_id,
                "user_chat_id": owner_chat_id,
                "is_enabled": True,
                "user": {"id": owner_user_id, "is_bot": False, "first_name": "Kim"},
                "rights": {"can_reply": True},
            }
        )
        workflow = app.state.intake_workflows.upsert_workflow(
            customer_id=customer_id,
            name="E2E Debounced Telegram Booking",
            channel="telegram_business_dm",
            provider="telegram_bot_api",
            source_config={"business_connection_id": business_connection_id},
            intent_description="Handle Telegram Business car wash booking requests.",
            required_fields=["car_model", "time"],
            assistant_instructions="Reply only to the newest coalesced lead context.",
            sink_type="local_csv",
            sink_config={"file_path": "tulpa_stuff/debounce.csv"},
        )

        first_status = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                message_id=10,
                text="Нужно помыть авто",
            ),
        )
        assert first_status.status_code == 200
        assert runtime.first_call_started.wait(timeout=5.0)

        second_status = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                message_id=11,
                text="Ролсройс кулинан",
            ),
        )
        assert second_status.status_code == 200

        assert _wait_until(
            lambda: len(
                [
                    item
                    for item in fake_telegram.sent_messages
                    if str(item.get("business_connection_id", "")).strip() == business_connection_id
                    and int(item.get("chat_id", 0)) == lead_chat_id
                ]
            )
            == 1,
            timeout_seconds=5.0,
        )
        time.sleep(0.1)

    lead_replies = [
        item
        for item in fake_telegram.sent_messages
        if str(item.get("business_connection_id", "")).strip() == business_connection_id
        and int(item.get("chat_id", 0)) == lead_chat_id
    ]
    assert len(lead_replies) == 1
    assert lead_replies[0]["reply_to_message_id"] == 11
    assert "Rolls-Royce Cullinan" in str(lead_replies[0]["text"])
    assert len(runtime.calls) == 2
    second_call_messages = runtime.calls[1]["conversation"]["recent_messages"]
    assert [item["text"] for item in second_call_messages if item["sender_role"] == "customer"] == [
        "Нужно помыть авто",
        "Ролсройс кулинан",
    ]
    assert any(item["event"] == "intake.conversation.stale" for item in runtime.behavior_events)
    assert workflow["workflow_id"]


def test_telegram_business_intake_restarts_with_three_unanswered_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _ThreeMessageRuntime()
    app, fake_telegram = _create_fake_telegram_app(tmp_path, monkeypatch, runtime)
    business_connection_id = "bc_e2e_three_messages"
    owner_user_id = 421
    owner_chat_id = 6541
    lead_chat_id = 778
    lead_user_id = 889
    customer_id = f"telegram_{owner_user_id}"

    with TestClient(app) as client:
        app.state.telegram_business.upsert_connection(
            {
                "id": business_connection_id,
                "user_chat_id": owner_chat_id,
                "is_enabled": True,
                "user": {"id": owner_user_id, "is_bot": False, "first_name": "Kim"},
                "rights": {"can_reply": True},
            }
        )
        app.state.intake_workflows.upsert_workflow(
            customer_id=customer_id,
            name="E2E Three Message Telegram Booking",
            channel="telegram_business_dm",
            provider="telegram_bot_api",
            source_config={"business_connection_id": business_connection_id},
            intent_description="Handle Telegram Business car wash booking requests.",
            required_fields=["car_model", "time"],
            assistant_instructions="Reply once using every unanswered customer message.",
            sink_type="local_csv",
            sink_config={"file_path": "tulpa_stuff/three_messages.csv"},
        )

        first_status = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                message_id=30,
                text="Нужно помыть авто",
            ),
        )
        assert first_status.status_code == 200
        assert runtime.first_call_started.wait(timeout=5.0)

        for message_id, text in (
            (31, "BMW sedan"),
            (32, "tomorrow at 10am"),
        ):
            status = client.post(
                "/webhook/telegram",
                headers={"x-telegram-bot-api-secret-token": "test-secret"},
                json=_business_message(
                    business_connection_id=business_connection_id,
                    lead_chat_id=lead_chat_id,
                    lead_user_id=lead_user_id,
                    message_id=message_id,
                    text=text,
                ),
            )
            assert status.status_code == 200

        assert _wait_until(
            lambda: len(
                [
                    item
                    for item in fake_telegram.sent_messages
                    if str(item.get("business_connection_id", "")).strip() == business_connection_id
                    and int(item.get("chat_id", 0)) == lead_chat_id
                ]
            )
            == 1,
            timeout_seconds=5.0,
        )
        time.sleep(0.1)

    lead_replies = [
        item
        for item in fake_telegram.sent_messages
        if str(item.get("business_connection_id", "")).strip() == business_connection_id
        and int(item.get("chat_id", 0)) == lead_chat_id
    ]
    assert len(lead_replies) == 1
    assert lead_replies[0]["reply_to_message_id"] == 32
    assert "BMW sedan" in str(lead_replies[0]["text"])
    assert len(runtime.calls) == 2
    restarted_messages = runtime.calls[1]["conversation"]["recent_messages"]
    assert [item["text"] for item in restarted_messages if item["sender_role"] == "customer"] == [
        "Нужно помыть авто",
        "BMW sedan",
        "tomorrow at 10am",
    ]
    assert any(item["event"] == "intake.conversation.stale" for item in runtime.behavior_events)


def test_telegram_business_intake_worker_does_not_block_unrelated_leads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _ParallelRuntime()
    app, fake_telegram = _create_fake_telegram_app(tmp_path, monkeypatch, runtime)
    first = {
        "business_connection_id": "bc_e2e_parallel_1",
        "owner_user_id": 521,
        "owner_chat_id": 6521,
        "lead_chat_id": 781,
        "lead_user_id": 891,
        "message_id": 40,
    }
    second = {
        "business_connection_id": "bc_e2e_parallel_2",
        "owner_user_id": 522,
        "owner_chat_id": 6522,
        "lead_chat_id": 782,
        "lead_user_id": 892,
        "message_id": 41,
    }

    with TestClient(app) as client:
        for index, case in enumerate((first, second), start=1):
            app.state.telegram_business.upsert_connection(
                {
                    "id": case["business_connection_id"],
                    "user_chat_id": case["owner_chat_id"],
                    "is_enabled": True,
                    "user": {
                        "id": case["owner_user_id"],
                        "is_bot": False,
                        "first_name": f"Owner {index}",
                    },
                    "rights": {"can_reply": True},
                }
            )
            app.state.intake_workflows.upsert_workflow(
                customer_id=f"telegram_{case['owner_user_id']}",
                name=f"E2E Parallel Telegram Booking {index}",
                channel="telegram_business_dm",
                provider="telegram_bot_api",
                source_config={"business_connection_id": case["business_connection_id"]},
                intent_description="Handle Telegram Business intake requests.",
                required_fields=["time"],
                assistant_instructions="Reply to this lead without waiting for other leads.",
                sink_type="local_csv",
                sink_config={"file_path": f"tulpa_stuff/parallel_{index}.csv"},
            )

        first_status = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_business_message(
                business_connection_id=first["business_connection_id"],
                lead_chat_id=first["lead_chat_id"],
                lead_user_id=first["lead_user_id"],
                message_id=first["message_id"],
                text="Need a wash tomorrow",
            ),
        )
        assert first_status.status_code == 200
        assert runtime.first_call_started.wait(timeout=5.0)

        second_status = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_business_message(
                business_connection_id=second["business_connection_id"],
                lead_chat_id=second["lead_chat_id"],
                lead_user_id=second["lead_user_id"],
                message_id=second["message_id"],
                text="Need detailing today",
            ),
        )
        assert second_status.status_code == 200

        assert _wait_until(
            lambda: len(
                [
                    item
                    for item in fake_telegram.sent_messages
                    if str(item.get("business_connection_id", "")).strip()
                    in {first["business_connection_id"], second["business_connection_id"]}
                    and int(item.get("chat_id", 0)) in {first["lead_chat_id"], second["lead_chat_id"]}
                ]
            )
            == 2,
            timeout_seconds=5.0,
        )

    lead_replies = [
        item
        for item in fake_telegram.sent_messages
        if str(item.get("business_connection_id", "")).strip()
        in {first["business_connection_id"], second["business_connection_id"]}
        and int(item.get("chat_id", 0)) in {first["lead_chat_id"], second["lead_chat_id"]}
    ]
    assert len(lead_replies) == 2
    assert {item["text"] for item in lead_replies} == {
        f"Reply for {first['lead_chat_id']}",
        f"Reply for {second['lead_chat_id']}",
    }
    assert runtime.max_active >= 2


def test_telegram_business_intake_uses_generic_user_id_after_late_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _ParallelRuntime()
    profiles = CustomerProfileService(tmp_path / "profiles.sqlite")
    app, fake_telegram = _create_fake_telegram_app(
        tmp_path,
        monkeypatch,
        runtime,
        customer_profiles=profiles,
    )
    generic_user_id = "usr_default"
    owner_user_id = 631
    owner_chat_id = 6631
    lead_chat_id = 783
    lead_user_id = 893
    business_connection_id = "bc_e2e_generic_late_bind"

    with TestClient(app) as client:
        connection = app.state.telegram_business.upsert_connection(
            {
                "id": business_connection_id,
                "user_chat_id": owner_chat_id,
                "is_enabled": True,
                "user": {
                    "id": owner_user_id,
                    "is_bot": False,
                    "first_name": "Generic Owner",
                },
                "rights": {"can_reply": True},
            }
        )
        assert connection["customer_id"] == f"telegram_{owner_user_id}"

        create = client.post(
            "/internal/intake/workflows/upsert",
            json={
                "customer_id": generic_user_id,
                "name": "Generic Telegram Intake",
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
                "source_config": {"business_connection_id": business_connection_id},
                "intent_description": "Handle Telegram Business intake requests for generic owner.",
                "required_fields": ["time"],
                "assistant_instructions": "Reply to the lead from generic owner storage.",
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/generic_late_bind.csv"},
            },
        )
        assert create.status_code == 200, create.text
        workflow = create.json()["workflow"]
        assert workflow["customer_id"] == generic_user_id

        bind = client.post(
            "/profiles/bind-telegram",
            json={"user_id": generic_user_id, "telegram_user_id": str(owner_user_id)},
        )
        assert bind.status_code == 200, bind.text
        assert profiles.resolve_customer_id(f"telegram_{owner_user_id}") == generic_user_id

        message_status = client.post(
            "/webhook/telegram",
            headers={"x-telegram-bot-api-secret-token": "test-secret"},
            json=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                message_id=50,
                text="Need detailing today",
            ),
        )
        assert message_status.status_code == 200

        assert _wait_until(
            lambda: len(
                [
                    item
                    for item in fake_telegram.sent_messages
                    if str(item.get("business_connection_id", "")).strip() == business_connection_id
                    and int(item.get("chat_id", 0)) == lead_chat_id
                ]
            )
            == 1,
            timeout_seconds=5.0,
        )

    assert runtime.calls
    assert runtime.calls[0]["customer_id"] == generic_user_id
    assert runtime.calls[0]["workflow"]["workflow_id"] == workflow["workflow_id"]
    assert runtime.calls[0]["conversation"]["summary"]["conversation_id"] == str(lead_chat_id)

    lead_replies = [
        item
        for item in fake_telegram.sent_messages
        if str(item.get("business_connection_id", "")).strip() == business_connection_id
        and int(item.get("chat_id", 0)) == lead_chat_id
    ]
    assert len(lead_replies) == 1
    assert lead_replies[0]["reply_to_message_id"] == 50
    assert lead_replies[0]["text"] == f"Reply for {lead_chat_id}"

    stored_connection = app.state.telegram_business.get_connection(business_connection_id)
    assert stored_connection is not None
    assert stored_connection["customer_id"] == generic_user_id
    conversation = app.state.telegram_business.get_conversation(
        customer_id=generic_user_id,
        business_connection_id=business_connection_id,
        conversation_id=str(lead_chat_id),
    )
    assert conversation["ok"] is True


@pytest.mark.live_llm
def test_live_llm_telegram_business_intake_suppresses_stale_split_reply(
    e2e_harness: E2EHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS", 0.01)
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 0.02)
    original_decide = e2e_harness.runtime.decide_intake_workflow

    async def _delayed_decide_intake_workflow(**kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(0.4)
        return await original_decide(**kwargs)

    monkeypatch.setattr(
        e2e_harness.runtime,
        "decide_intake_workflow",
        _delayed_decide_intake_workflow,
    )

    owner_user_id = 432
    owner_chat_id = 876
    customer_id = f"telegram_{owner_user_id}"
    business_connection_id = _seed_telegram_business_connection(
        e2e_harness,
        owner_user_id=owner_user_id,
        owner_chat_id=owner_chat_id,
        business_connection_id="bc_e2e_live_debounce",
    )
    lead_chat_id = 901
    lead_user_id = 902

    create = e2e_harness.client.post(
        "/internal/intake/workflows/upsert",
        json={
            "customer_id": customer_id,
            "name": "Live Debounced Telegram Intake",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
            "source_config": {"business_connection_id": business_connection_id},
            "intent_description": "Reply to Telegram Business car wash leads and collect booking details.",
            "required_fields": ["car_model", "car_type", "wash_type", "date", "time"],
            "assistant_instructions": (
                "Reply once to the newest coalesced lead context. "
                "If the customer splits details across adjacent messages, use all of them."
            ),
            "sink_type": "local_csv",
            "sink_config": {"file_path": "tulpa_stuff/e2e_live_debounce.csv"},
            "enabled": True,
        },
    )
    assert create.status_code == 200, create.text
    workflow = create.json()["workflow"]

    start_message_count = len(e2e_harness.telegram_client.sent_messages)
    assert (
        e2e_harness.post_telegram(
            body=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                message_id=20,
                text="Нужно помыть авто",
            )
        )
        == 200
    )
    assert _wait_until(
        lambda: any(
            item.get("event") == "intake.decision.start"
            and str(item.get("workflow_id", "")).strip() == workflow["workflow_id"]
            and str(item.get("conversation_id", "")).strip() == str(lead_chat_id)
            for item in _behavior_events(e2e_harness)
        ),
        timeout_seconds=30.0,
    )
    assert (
        e2e_harness.post_telegram(
            body=_business_message(
                business_connection_id=business_connection_id,
                lead_chat_id=lead_chat_id,
                lead_user_id=lead_user_id,
                message_id=21,
                text="BMW sedan, full wash, tomorrow at 10am.",
            )
        )
        == 200
    )

    assert _wait_until(
        lambda: len(
            [
                item
                for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
                if int(item.get("chat_id", 0)) == lead_chat_id
                and str(item.get("business_connection_id", "")).strip() == business_connection_id
            ]
        )
        == 1,
        timeout_seconds=90.0,
    )
    time.sleep(1.0)

    lead_replies = [
        item
        for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
        if int(item.get("chat_id", 0)) == lead_chat_id
        and str(item.get("business_connection_id", "")).strip() == business_connection_id
    ]
    assert len(lead_replies) == 1
    assert lead_replies[0]["reply_to_message_id"] == 21
    assert str(lead_replies[0].get("text", "")).strip()

    owner_errors = [
        item
        for item in e2e_harness.telegram_client.sent_messages[start_message_count:]
        if int(item.get("chat_id", 0)) == owner_chat_id
    ]
    assert owner_errors == []

    decision_starts = [
        item
        for item in _behavior_events(e2e_harness)
        if item.get("event") == "intake.decision.start"
        and str(item.get("workflow_id", "")).strip() == workflow["workflow_id"]
        and str(item.get("conversation_id", "")).strip() == str(lead_chat_id)
    ]
    assert len(decision_starts) == 2
    assert int(decision_starts[-1].get("recent_message_count") or 0) >= 2
    assert any(
        item.get("event") == "intake.conversation.stale"
        and str(item.get("workflow_id", "")).strip() == workflow["workflow_id"]
        and str(item.get("conversation_id", "")).strip() == str(lead_chat_id)
        for item in _behavior_events(e2e_harness)
    )
