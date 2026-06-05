from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from opentulpa.api import app as app_module
from opentulpa.core.config import get_settings
from opentulpa.intake import service as intake_service_module
from tests.test_intake_workflow_service import (
    _AlwaysFailingReplyComposio,
    _FakeComposio,
    _FakeRuntime,
    _instagram_conversation,
    _mk_service,
)


@pytest.fixture(autouse=True)
def _freeze_intake_now(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: datetime(2026, 4, 7, 8, 0, 30, tzinfo=UTC),
    )


def _summary(
    *,
    message_id: str,
    text: str,
    at: str = "2026-04-07T08:01:00+00:00",
    username: str = "alice",
) -> dict[str, str]:
    return {
        "conversation_id": "conv_1",
        "recipient_id": "cust_1",
        "latest_inbound_message_id": message_id,
        "latest_inbound_message_created_time": at,
        "latest_inbound_message_text_preview": text,
        "latest_inbound_sender_username": username,
    }


def _conversation_with_previous() -> dict[str, object]:
    return {
        "data": {
            "id": "conv_1",
            "participants": {"data": [{"id": "cust_1", "username": "alice"}]},
            "messages": {
                "data": [
                    {
                        "id": "msg_1",
                        "created_time": "2026-04-07T08:00:00+00:00",
                        "message": "I want ceramic coating tomorrow.",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                    {
                        "id": "msg_2",
                        "created_time": "2026-04-07T08:01:00+00:00",
                        "message": "Can you do 20% off?",
                        "from": {"id": "cust_1", "username": "alice"},
                    },
                ]
            },
        }
    }


def _handoff_decision(
    *,
    rule_id: str = "discount_approval",
    reason: str = "Customer asked for 20% off.",
    request: str = "Approve, counter, or decline 20% discount?",
    wait_reply: str = "Let me check that and get back to you.",
) -> dict[str, object]:
    return {
        "ok": True,
        "matches_workflow": True,
        "confidence": 0.95,
        "conversation_summary": "Lead needs owner approval.",
        "extracted_fields": {},
        "missing_fields": ["owner_approval"],
        "reply_action": "none",
        "reply_text": "",
        "ready_to_save": False,
        "booking_action": "create_new_booking",
        "save_payload": {},
        "handoff_action": "request_owner",
        "handoff_rule_id": rule_id,
        "handoff_reason": reason,
        "handoff_request": request,
        "customer_wait_reply": wait_reply,
    }


def _reply_decision(text: str) -> dict[str, object]:
    return {
        "ok": True,
        "matches_workflow": True,
        "confidence": 0.97,
        "conversation_summary": "Owner gave private guidance and agent wrote final reply.",
        "extracted_fields": {},
        "missing_fields": [],
        "reply_action": "send_reply",
        "reply_text": text,
        "ready_to_save": False,
        "booking_action": "ignore",
        "save_payload": {},
        "handoff_action": "none",
    }


def _upsert_handoff_workflow(service: Any, *, customer_id: str = "telegram_123") -> dict[str, Any]:
    return cast(dict[str, Any], service.upsert_workflow(
        customer_id=customer_id,
        name="Detailing",
        channel="instagram_dm",
        provider="composio",
        source_config={},
        intent_description="Book detailing jobs.",
        required_fields=["service", "owner_approval"],
        handoff_rules=[
            {
                "id": "discount_approval",
                "condition": "Customer asks for discount approval.",
                "owner_prompt": "Ask owner to approve or counter discount.",
                "customer_wait_reply": "Let me check that and get back to you.",
            }
        ],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    ))


@pytest.mark.asyncio
async def test_intake_handoff_opens_updates_and_pauses_automatic_apply(tmp_path: Path) -> None:
    runtime = _FakeRuntime([_handoff_decision()])
    composio = _FakeComposio(
        _summary(message_id="msg_2", text="Can you do 20% off?"),
        _conversation_with_previous(),
    )
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = _upsert_handoff_workflow(service)

    result = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    assert result["results"][0]["status"] == "owner_handoff_awaiting_owner"
    assert result["results"][0]["replied"] is True
    assert composio.execute_calls[0]["tool_slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"
    handoffs = service.handoffs.list_handoffs(customer_id="telegram_123")
    assert len(handoffs) == 1
    assert handoffs[0]["lead"]["username"] == "alice"
    assert handoffs[0]["messages"]["latest"][0]["text"] == "Can you do 20% off?"
    assert handoffs[0]["messages"]["previous"][0]["text"] == "I want ceramic coating tomorrow."


@pytest.mark.asyncio
async def test_intake_handoff_spam_updates_same_open_handoff(tmp_path: Path) -> None:
    runtime = _FakeRuntime(
        [
            _handoff_decision(reason="Discount requested.", request="Approve discount?", wait_reply="Let me check."),
            _handoff_decision(
                reason="Lead spammed another discount request.",
                request="Approve any discount?",
                wait_reply="Let me check.",
            ),
        ]
    )
    composio = _FakeComposio(
        _summary(message_id="msg_1", text="Need a manager discount.", at="2026-04-07T08:00:00+00:00"),
        _instagram_conversation(
            conversation_id="conv_1",
            latest_message_id="msg_1",
            latest_message_text="Need a manager discount.",
            latest_message_time="2026-04-07T08:00:00+00:00",
        ),
    )
    service, _, _, _, _ = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = _upsert_handoff_workflow(service)

    first = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])
    composio.summary.update(_summary(message_id="msg_2", text="Hello? 30%?"))
    composio.conversation = _instagram_conversation(
        conversation_id="conv_1",
        latest_message_id="msg_2",
        latest_message_text="Hello? 30%?",
        latest_message_time="2026-04-07T08:01:00+00:00",
    )
    second = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])

    handoffs = service.handoffs.list_handoffs(customer_id="telegram_123")
    assert first["results"][0]["handoff_id"] == second["results"][0]["handoff_id"]
    assert len(handoffs) == 1
    assert handoffs[0]["latest_customer_message_preview"] == "Hello? 30%?"
    assert len(composio.execute_calls) == 1


@pytest.mark.asyncio
async def test_owner_handoff_web_api_queues_and_pending_drain_resumes_reply_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    get_settings.cache_clear()
    runtime = _FakeRuntime(
        [
            _handoff_decision(),
            _reply_decision("I can do 10% off for ceramic coating tomorrow."),
        ]
    )
    composio = _FakeComposio(
        _summary(message_id="msg_2", text="Can you do 20% off?"),
        _conversation_with_previous(),
    )
    service, scheduler, skills, _, file_vault = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = _upsert_handoff_workflow(service)
    opened = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])
    handoff_id = str(opened["results"][0]["handoff_id"])
    app = app_module.create_app(
        agent_runtime=runtime,
        scheduler=scheduler,
        skill_store_service=skills,
        file_vault_service=file_vault,
        composio_service=composio,
        intake_workflow_service=service,
    )
    client = TestClient(app, client=("8.8.8.8", 50000))

    response = client.post(
        f"/web/intake/handoffs/{handoff_id}/respond?customer_id=telegram_123",
        headers={"authorization": "Bearer web-secret"},
        json={"owner_feedback": "Approve 10%, not 20%."},
    )
    drained = await service.drain_due_pending_runs(limit=5)
    detail = client.get(
        f"/web/intake/handoffs/{handoff_id}?customer_id=telegram_123",
        headers={"authorization": "Bearer web-secret"},
    )
    client.close()
    get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json()["queued"] is True
    assert response.json()["handoff"]["status"] == "owner_responded"
    assert drained == 1
    assert detail.status_code == 200
    assert detail.json()["handoff"]["status"] == "resolved"
    assert runtime.calls[1]["owner_handoff_feedback"]["owner_feedback"] == "Approve 10%, not 20%."
    assert [call["arguments"]["text"] for call in composio.execute_calls] == [
        "Let me check that and get back to you.",
        "I can do 10% off for ceramic coating tomorrow.",
    ]


@pytest.mark.asyncio
async def test_owner_handoff_pending_drain_marks_failed_reply_when_final_send_fails_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setattr(app_module, "PROJECT_ROOT", tmp_path)
    get_settings.cache_clear()
    runtime = _FakeRuntime(
        [
            _handoff_decision(
                rule_id="owner_approval",
                reason="Custom request needs owner approval.",
                request="Approve custom request?",
                wait_reply="",
            ),
            _reply_decision("Owner approved your custom request."),
        ]
    )
    composio = _AlwaysFailingReplyComposio(
        _summary(message_id="msg_1", text="Can you approve my custom request?"),
        _instagram_conversation(
            conversation_id="conv_1",
            latest_message_id="msg_1",
            latest_message_text="Can you approve my custom request?",
            latest_message_time="2026-04-07T08:01:00+00:00",
        ),
    )
    service, scheduler, skills, _, file_vault = _mk_service(tmp_path, runtime=runtime, composio=composio)
    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Detailing",
        channel="instagram_dm",
        provider="composio",
        source_config={},
        intent_description="Book detailing jobs.",
        required_fields=["owner_approval"],
        handoff_rules=[{"id": "owner_approval", "condition": "Owner approval needed."}],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )
    opened = await service.run_workflow(customer_id="telegram_123", workflow_id=workflow["workflow_id"])
    handoff_id = str(opened["results"][0]["handoff_id"])
    app = app_module.create_app(
        agent_runtime=runtime,
        scheduler=scheduler,
        skill_store_service=skills,
        file_vault_service=file_vault,
        composio_service=composio,
        intake_workflow_service=service,
    )
    client = TestClient(app, client=("8.8.8.8", 50000))

    response = client.post(
        f"/web/intake/handoffs/{handoff_id}/respond?customer_id=telegram_123",
        headers={"authorization": "Bearer web-secret"},
        json={"owner_feedback": "Approved."},
    )
    drained = await service.drain_due_pending_runs(limit=5)
    detail = client.get(
        f"/web/intake/handoffs/{handoff_id}?customer_id=telegram_123",
        headers={"authorization": "Bearer web-secret"},
    )
    client.close()
    get_settings.cache_clear()

    assert response.status_code == 200
    assert drained == 1
    assert detail.json()["handoff"]["status"] == "failed_reply"
    assert "temporary send failure" in detail.json()["handoff"]["failure_reason"]


def test_intake_workflow_upsert_persists_handoff_rules(tmp_path: Path) -> None:
    service, _, skills, _, _ = _mk_service(
        tmp_path,
        runtime=_FakeRuntime([]),
        composio=_FakeComposio(
            _summary(message_id="msg_1", text="Need a car wash tomorrow 3pm."),
            _instagram_conversation(
                conversation_id="conv_1",
                latest_message_id="msg_1",
                latest_message_text="Need a car wash tomorrow 3pm.",
                latest_message_time="2026-04-07T08:00:00+00:00",
            ),
        ),
    )

    workflow = service.upsert_workflow(
        customer_id="telegram_123",
        name="Car Wash Intake",
        intent_description="Handle Instagram DMs that ask to book a car wash service.",
        required_fields=["day", "time"],
        handoff_rules=[
            {
                "id": "discount_approval",
                "label": "Discount approval",
                "condition": "Customer asks for discount approval.",
                "owner_prompt": "Ask owner for discount approval before replying.",
                "customer_wait_reply": "Let me check that.",
            }
        ],
        sink_type="local_csv",
        sink_config={"file_path": "tulpa_stuff/bookings.csv"},
    )

    fetched = service.get_workflow(
        customer_id="telegram_123",
        workflow_id=str(workflow["workflow_id"]),
    )
    assert fetched is not None
    assert fetched["handoff_rules"] == [
        {
            "id": "discount_approval",
            "label": "Discount approval",
            "condition": "Customer asks for discount approval.",
            "owner_prompt": "Ask owner for discount approval before replying.",
            "customer_wait_reply": "Let me check that.",
            "enabled": True,
        }
    ]
    skill = skills.get_skill(
        customer_id="telegram_123",
        name=f"intake-workflow-{workflow['workflow_id']}",
        include_files=True,
        include_global=False,
    )
    assert skill is not None
    workflow_json = json.loads(skill["supporting_files"]["workflow.json"])
    assert workflow_json["handoff_rules"][0]["id"] == "discount_approval"
