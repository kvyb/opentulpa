from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from harness.runner import build_harness, close_harness
from mocks.composio_instagram import FakeComposioInstagramService
from mocks.telegram import FakeTelegramClient

from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.intake import service as intake_service_module
from opentulpa.scheduler.service import SchedulerService

pytestmark = [pytest.mark.e2e]


class _OwnerHandoffRuntime:
    def __init__(self, *, final_reply: str = "Owner approved 10%. I can do that for you.") -> None:
        self.calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []
        self.final_reply = final_reply

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        feedback = kwargs.get("owner_handoff_feedback")
        if isinstance(feedback, dict) and str(feedback.get("owner_feedback", "") or "").strip():
            return {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.97,
                "conversation_summary": "Owner gave private guidance and agent wrote final reply.",
                "extracted_fields": {"owner_guidance": "received"},
                "missing_fields": [],
                "reply_action": "send_reply",
                "reply_text": self.final_reply,
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "handoff_action": "none",
            }
        latest_text = _latest_customer_text(kwargs)
        return {
            "ok": True,
            "matches_workflow": True,
            "confidence": 0.94,
            "conversation_summary": f"Lead needs owner approval: {latest_text}",
            "extracted_fields": {"requested_exception": latest_text},
            "missing_fields": ["owner_approval"],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": False,
            "booking_action": "create_new_booking",
            "save_payload": {},
            "handoff_action": "request_owner",
            "handoff_rule_id": "owner_approval",
            "handoff_reason": "Lead requested an exception that owner must approve.",
            "handoff_request": "Approve, counter, or decline this request?",
            "customer_wait_reply": "Let me check with the owner and get back to you.",
        }

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        self.behavior_events.append({"event": event, "customer_id": customer_id, **fields})


class _NoReplyAfterOwnerRuntime(_OwnerHandoffRuntime):
    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        feedback = kwargs.get("owner_handoff_feedback")
        if isinstance(feedback, dict) and str(feedback.get("owner_feedback", "") or "").strip():
            return {
                "ok": True,
                "matches_workflow": True,
                "confidence": 0.8,
                "conversation_summary": "Owner advice says no customer reply should be sent.",
                "extracted_fields": {},
                "missing_fields": [],
                "reply_action": "none",
                "reply_text": "",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "handoff_action": "none",
            }
        return await super().decide_intake_workflow(**kwargs)


class _FailingFinalTelegramClient(FakeTelegramClient):
    def __init__(self, token: str) -> None:
        super().__init__(token)
        self.send_attempts = 0

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        self.send_attempts += 1
        if self.send_attempts >= 2:
            raise RuntimeError("temporary send failure")
        return cast(dict[str, Any], await super().send_message(**kwargs))


def _latest_customer_text(kwargs: dict[str, Any]) -> str:
    messages = kwargs.get("conversation", {}).get("recent_messages", [])
    if not isinstance(messages, list):
        return ""
    for item in reversed(messages):
        if not isinstance(item, dict):
            continue
        if str(item.get("sender_role", "") or "").strip() == "customer":
            return str(item.get("text", "") or "").strip()
    return ""


def _create_fake_telegram_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
    *,
    telegram_client: FakeTelegramClient | None = None,
) -> tuple[Any, FakeTelegramClient]:
    from opentulpa.api import app as app_module
    from opentulpa.tasks import sandbox as sandbox_module

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS", 0.01)
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 60.0)

    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    fake_telegram = telegram_client or FakeTelegramClient("fake-token")
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_telegram)
    get_settings.cache_clear()

    scheduler = SchedulerService(db_path=tmp_path / "scheduler.sqlite")
    app = create_app(agent_runtime=runtime, scheduler=scheduler)
    return app, fake_telegram


def _create_fake_instagram_app(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime: Any,
    composio: FakeComposioInstagramService,
) -> Any:
    from opentulpa.api import app as app_module
    from opentulpa.tasks import sandbox as sandbox_module

    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 60.0)

    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    get_settings.cache_clear()

    scheduler = SchedulerService(db_path=tmp_path / "scheduler.sqlite")
    return create_app(agent_runtime=runtime, scheduler=scheduler, composio_service=composio)


def _business_message(
    *,
    business_connection_id: str,
    lead_chat_id: int,
    lead_user_id: int,
    message_id: int,
    text: str,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000) + message_id,
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


def _unique_case(prefix: str) -> dict[str, Any]:
    suffix = uuid4().hex[:8]
    return {
        "business_connection_id": f"bc_{prefix}_{suffix}",
        "owner_user_id": int("7" + suffix[:5], 16) % 1_000_000 + 100_000,
        "owner_chat_id": int("8" + suffix[:5], 16) % 1_000_000 + 200_000,
        "lead_chat_id": int("9" + suffix[:5], 16) % 1_000_000 + 300_000,
        "lead_user_id": int("a" + suffix[:5], 16) % 1_000_000 + 400_000,
        "customer_id": "",
    }


def _seed_handoff_workflow(app: Any, case: dict[str, Any]) -> dict[str, Any]:
    customer_id = f"telegram_{case['owner_user_id']}"
    case["customer_id"] = customer_id
    app.state.telegram_business.upsert_connection(
        {
            "id": case["business_connection_id"],
            "user_chat_id": case["owner_chat_id"],
            "is_enabled": True,
            "user": {
                "id": case["owner_user_id"],
                "is_bot": False,
                "first_name": "Owner",
                "username": f"owner_{case['owner_user_id']}",
            },
            "rights": {"can_reply": True},
        }
    )
    return cast(dict[str, Any], app.state.intake_workflows.upsert_workflow(
        customer_id=customer_id,
        name="E2E Owner Handoff",
        channel="telegram_business_dm",
        provider="telegram_bot_api",
        source_config={"business_connection_id": case["business_connection_id"]},
        intent_description="Handle inbound leads and ask owner for approval on exceptions.",
        required_fields=["owner_approval"],
        assistant_instructions="Ask owner only for configured owner approval cases.",
        handoff_rules=[
            {
                "id": "owner_approval",
                "label": "Owner approval",
                "condition": "Lead asks for a discount, exception, custom slot, or approval.",
                "owner_prompt": "Approve, counter, or decline this request.",
                "customer_wait_reply": "Let me check with the owner and get back to you.",
            }
        ],
        sink_type="local_csv",
        sink_config={"file_path": f"tulpa_stuff/{uuid4().hex}.csv"},
    ))


def _unique_instagram_case(prefix: str) -> dict[str, Any]:
    suffix = uuid4().hex[:8]
    return {
        "customer_id": f"cust_ig_{prefix}_{suffix}",
        "connected_account_id": f"acct_ig_{prefix}_{suffix}",
        "conversation_id": f"conv_ig_{prefix}_{suffix}",
        "recipient_id": f"ig_user_{suffix}",
        "lead_username": f"ig_lead_{suffix}",
    }


def _seed_instagram_handoff_workflow(app: Any, case: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], app.state.intake_workflows.upsert_workflow(
        customer_id=str(case["customer_id"]),
        name="E2E Instagram Owner Handoff",
        channel="instagram_dm",
        provider="composio",
        source_config={
            "connected_account_id": case["connected_account_id"],
            "conversation_id": case["conversation_id"],
        },
        intent_description=(
            "Handle Instagram DMs for inbound leads. Ask the owner for approval when the "
            "lead requests a discount, exception, custom slot, or unclear manual decision."
        ),
        required_fields=["owner_approval"],
        assistant_instructions=(
            "If the lead asks for a discount, special approval, or custom slot, use the configured "
            "owner handoff rule. After owner feedback, write the customer reply yourself."
        ),
        handoff_rules=[
            {
                "id": "owner_approval",
                "label": "Owner approval",
                "condition": "Lead asks for a discount, exception, custom slot, or approval.",
                "owner_prompt": "Approve, counter, or decline this Instagram lead request.",
                "customer_wait_reply": "Let me check with the owner and get back to you.",
            }
        ],
        sink_type="local_csv",
        sink_config={"file_path": f"tulpa_stuff/{uuid4().hex}.csv"},
    ))


def _instagram_message(case: dict[str, Any], *, message_id: str, minutes: int, role: str, text: str) -> dict[str, Any]:
    sender_id = str(case["recipient_id"]) if role == "customer" else "page_1"
    username = str(case["lead_username"]) if role == "customer" else "biz_account"
    return {
        "id": message_id,
        "created_time": f"2026-04-14T10:{minutes:02d}:30+0000",
        "from": {"id": sender_id, "username": username},
        "to": {"data": [{"id": "page_1" if role == "customer" else case["recipient_id"]}]},
        "message": text,
    }


def _set_instagram_conversation(
    composio: FakeComposioInstagramService,
    case: dict[str, Any],
    messages: list[dict[str, Any]],
) -> None:
    assert messages
    recipient_id = str(case["recipient_id"])
    latest_message = messages[-1]
    inbound_messages = [
        item for item in messages if str(item.get("from", {}).get("id", "") or "").strip() == recipient_id
    ]
    outbound_messages = [
        item for item in messages if str(item.get("from", {}).get("id", "") or "").strip() != recipient_id
    ]
    assert inbound_messages
    latest_inbound = inbound_messages[-1]
    summary = {
        "conversation_id": str(case["conversation_id"]),
        "recipient_id": recipient_id,
        "username": str(case["lead_username"]),
        "latest_message_id": str(latest_message["id"]),
        "latest_message_created_time": str(latest_message["created_time"]),
        "latest_message_text_preview": str(latest_message["message"]),
        "latest_inbound_message_id": str(latest_inbound["id"]),
        "latest_inbound_message_created_time": str(latest_inbound["created_time"]),
        "latest_inbound_message_text_preview": str(latest_inbound["message"]),
        "latest_inbound_sender_username": str(case["lead_username"]),
        "message_count": len(messages),
    }
    if outbound_messages:
        latest_outbound = outbound_messages[-1]
        summary.update(
            {
                "latest_outbound_message_id": str(latest_outbound["id"]),
                "latest_outbound_message_created_time": str(latest_outbound["created_time"]),
            }
        )
    composio.conversations[str(case["conversation_id"])] = {
        "summary": summary,
        "conversation": {
            "id": str(case["conversation_id"]),
            "participants": {
                "data": [
                    {"id": "page_1", "username": "biz_account"},
                    {"id": recipient_id, "username": str(case["lead_username"])},
                ]
            },
            "messages": {"data": list(messages)},
        },
    }


def _post_lead(client: TestClient, case: dict[str, Any], *, message_id: int, text: str) -> None:
    response = client.post(
        "/webhook/telegram",
        headers={"x-telegram-bot-api-secret-token": "test-secret"},
        json=_business_message(
            business_connection_id=str(case["business_connection_id"]),
            lead_chat_id=int(case["lead_chat_id"]),
            lead_user_id=int(case["lead_user_id"]),
            message_id=message_id,
            text=text,
        ),
    )
    assert response.status_code == 200


def _handoffs(client: TestClient, case: dict[str, Any]) -> list[dict[str, Any]]:
    response = client.get(
        f"/web/intake/handoffs?customer_id={case['customer_id']}",
        headers={"authorization": "Bearer web-secret"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    return list(payload["handoffs"])


def _handoff_detail(client: TestClient, case: dict[str, Any], handoff_id: str) -> dict[str, Any]:
    response = client.get(
        f"/web/intake/handoffs/{handoff_id}?customer_id={case['customer_id']}",
        headers={"authorization": "Bearer web-secret"},
    )
    assert response.status_code == 200
    return dict(response.json()["handoff"])


def _respond_owner(client: TestClient, case: dict[str, Any], handoff_id: str, feedback: str) -> Any:
    return client.post(
        f"/web/intake/handoffs/{handoff_id}/respond?customer_id={case['customer_id']}",
        headers={"authorization": "Bearer web-secret"},
        json={"owner_feedback": feedback},
    )


def _sent_to_lead(fake_telegram: FakeTelegramClient, case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in fake_telegram.sent_messages
        if str(item.get("business_connection_id", "") or "").strip() == case["business_connection_id"]
        and int(item.get("chat_id", 0)) == int(case["lead_chat_id"])
    ]


def _run_intake_workflow(client: TestClient, *, customer_id: str, workflow_id: str) -> dict[str, Any]:
    response = client.post(
        "/internal/intake/workflows/run",
        json={
            "customer_id": customer_id,
            "workflow_id": workflow_id,
            "force": True,
            "event_type": "instagram_e2e",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    return cast(dict[str, Any], payload)


def _instagram_send_calls(
    composio: FakeComposioInstagramService,
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        item
        for item in composio.calls
        if item.get("method") == "execute_tool"
        and str(item.get("tool_slug", "") or "").upper() == "INSTAGRAM_SEND_TEXT_MESSAGE"
        and str(item.get("customer_id", "") or "").strip() == str(case["customer_id"])
    ]


def _drain_pending(app: Any, *, limit: int = 10) -> int:
    return int(asyncio.run(app.state.intake_workflows.drain_due_pending_runs(limit=limit)))


def _live_llm_key_configured() -> bool:
    get_settings.cache_clear()
    settings = get_settings()
    return bool(str(settings.openai_compatible_api_key or "").strip())


@pytest.mark.telegram
def test_owner_handoff_telegram_webhook_to_web_api_to_final_reply_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _OwnerHandoffRuntime(final_reply="Owner approved 10%. I can do that for you.")
    app, fake_telegram = _create_fake_telegram_app(tmp_path, monkeypatch, runtime)
    case = _unique_case("happy")

    with TestClient(app) as client:
        workflow = _seed_handoff_workflow(app, case)
        _post_lead(client, case, message_id=101, text="Can you approve 20% off?")
        assert _drain_pending(app) == 1

        assert _wait_until(lambda: len(_handoffs(client, case)) == 1)
        handoff = _handoffs(client, case)[0]
        assert handoff["status"] == "awaiting_owner"
        assert handoff["lead"]["username"] == f"lead_{case['lead_user_id']}"
        assert handoff["messages"]["latest"][0]["text"] == "Can you approve 20% off?"
        assert _sent_to_lead(fake_telegram, case)[0]["text"] == (
            "Let me check with the owner and get back to you."
        )

        response = _respond_owner(client, case, handoff["handoff_id"], "Approve 10%, not 20%.")
        assert response.status_code == 200
        assert response.json()["queued"] is True

        drained = _drain_pending(app)
        detail = _handoff_detail(client, case, handoff["handoff_id"])

    assert workflow["workflow_id"]
    assert drained == 1
    assert detail["status"] == "resolved"
    assert runtime.calls[-1]["owner_handoff_feedback"]["owner_feedback"] == "Approve 10%, not 20%."
    assert [item["text"] for item in _sent_to_lead(fake_telegram, case)] == [
        "Let me check with the owner and get back to you.",
        "Owner approved 10%. I can do that for you.",
    ]


@pytest.mark.telegram
def test_owner_handoff_spam_updates_one_open_handoff_and_preserves_resume_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _OwnerHandoffRuntime(final_reply="I saw your latest messages. Owner approved one custom slot.")
    app, fake_telegram = _create_fake_telegram_app(tmp_path, monkeypatch, runtime)
    case = _unique_case("spam")

    with TestClient(app) as client:
        workflow = _seed_handoff_workflow(app, case)
        _post_lead(client, case, message_id=201, text="Can you approve a custom slot?")
        assert _drain_pending(app) == 1
        assert _wait_until(lambda: len(_handoffs(client, case)) == 1)

        for message_id, text in (
            (202, "Also can it be tonight?"),
            (203, "And can you do 15% off?"),
            (204, "I need confirmation soon."),
            (205, "My username should be visible."),
        ):
            _post_lead(client, case, message_id=message_id, text=text)
            assert _drain_pending(app) == 1

        assert _wait_until(
            lambda: _handoffs(client, case)[0]["latest_customer_message_preview"]
            == "My username should be visible."
        )
        handoff = _handoffs(client, case)[0]
        assert len(_handoffs(client, case)) == 1
        assert len(_sent_to_lead(fake_telegram, case)) == 1
        assert [item["text"] for item in handoff["messages"]["latest"]] == [
            "My username should be visible.",
        ]
        assert [item["text"] for item in handoff["messages"]["previous"]] == [
            "Also can it be tonight?",
            "And can you do 15% off?",
            "I need confirmation soon.",
        ]

        response = _respond_owner(client, case, handoff["handoff_id"], "Approve tonight only.")
        assert response.status_code == 200
        app.state.intake_workflows._queue_pending_run(  # noqa: SLF001
            workflow=workflow,
            conversation_id=str(case["lead_chat_id"]),
            event_type="telegram_business_webhook",
            delay_seconds=0,
            last_inbound_message_id="206",
        )
        drained = _drain_pending(app)
        detail = _handoff_detail(client, case, handoff["handoff_id"])

    assert drained == 1
    assert detail["status"] == "resolved"
    assert runtime.calls[-1]["owner_handoff_feedback"]["owner_feedback"] == "Approve tonight only."
    assert _sent_to_lead(fake_telegram, case)[-1]["reply_to_message_id"] == 205


@pytest.mark.telegram
def test_owner_handoff_duplicate_owner_response_and_duplicate_drain_are_idempotent_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _OwnerHandoffRuntime(final_reply="Owner approved the smaller discount.")
    app, fake_telegram = _create_fake_telegram_app(tmp_path, monkeypatch, runtime)
    case = _unique_case("idempotent")

    with TestClient(app) as client:
        _seed_handoff_workflow(app, case)
        _post_lead(client, case, message_id=301, text="Can you approve a discount?")
        assert _drain_pending(app) == 1
        assert _wait_until(lambda: len(_handoffs(client, case)) == 1)
        handoff_id = str(_handoffs(client, case)[0]["handoff_id"])

        first = _respond_owner(client, case, handoff_id, "Approve 5%.")
        second = _respond_owner(client, case, handoff_id, "Actually approve 20%.")
        assert first.status_code == 200
        assert second.status_code == 409

        first_drain = _drain_pending(app)
        second_drain = _drain_pending(app)
        detail = _handoff_detail(client, case, handoff_id)

    assert first_drain == 1
    assert second_drain == 0
    assert detail["status"] == "resolved"
    assert len([call for call in runtime.calls if call.get("owner_handoff_feedback")]) == 1
    assert [item["text"] for item in _sent_to_lead(fake_telegram, case)] == [
        "Let me check with the owner and get back to you.",
        "Owner approved the smaller discount.",
    ]


@pytest.mark.telegram
def test_owner_handoff_final_reply_failure_marks_failed_and_stops_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _OwnerHandoffRuntime(final_reply="Owner approved the custom request.")
    failing_telegram = _FailingFinalTelegramClient("fake-token")
    app, fake_telegram = _create_fake_telegram_app(
        tmp_path,
        monkeypatch,
        runtime,
        telegram_client=failing_telegram,
    )
    case = _unique_case("failure")

    with TestClient(app) as client:
        _seed_handoff_workflow(app, case)
        _post_lead(client, case, message_id=401, text="Can owner approve a custom request?")
        assert _drain_pending(app) == 1
        assert _wait_until(lambda: len(_handoffs(client, case)) == 1)
        handoff_id = str(_handoffs(client, case)[0]["handoff_id"])

        response = _respond_owner(client, case, handoff_id, "Approved.")
        assert response.status_code == 200

        first_drain = _drain_pending(app)
        second_drain = _drain_pending(app)
        detail = _handoff_detail(client, case, handoff_id)

    assert first_drain == 1
    assert second_drain == 0
    assert detail["status"] == "failed_reply"
    assert "temporary send failure" in detail["failure_reason"]
    assert failing_telegram.send_attempts == 2
    assert [item["text"] for item in _sent_to_lead(fake_telegram, case)] == [
        "Let me check with the owner and get back to you."
    ]


@pytest.mark.ingress
def test_owner_handoff_instagram_dm_to_web_api_to_final_reply_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _OwnerHandoffRuntime(final_reply="Owner approved 10%. I can offer that for your booking.")
    composio = FakeComposioInstagramService(reply_fail_once_for_invalid_mid=False)
    app = _create_fake_instagram_app(tmp_path, monkeypatch, runtime, composio)
    case = _unique_instagram_case("happy")
    messages = [
        _instagram_message(case, message_id="mid_ig_out_1", minutes=10, role="assistant", text="Hi, how can I help?"),
        _instagram_message(
            case,
            message_id="mid_ig_in_1",
            minutes=11,
            role="customer",
            text="Can you ask owner to approve 20% off for detailing tomorrow?",
        ),
    ]
    _set_instagram_conversation(composio, case, messages)

    with TestClient(app) as client:
        workflow = _seed_instagram_handoff_workflow(app, case)
        run = _run_intake_workflow(
            client,
            customer_id=str(case["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
        )
        assert run["processed_conversations"] == 1

        handoffs = _handoffs(client, case)
        assert len(handoffs) == 1
        handoff = handoffs[0]
        assert handoff["status"] == "awaiting_owner"
        assert handoff["lead"]["username"] == case["lead_username"]
        assert handoff["messages"]["latest"][0]["message_id"] == "mid_ig_in_1"
        assert handoff["messages"]["latest"][0]["text"].startswith("Can you ask owner")
        assert _instagram_send_calls(composio, case)[0]["arguments"] == {
            "recipient_id": case["recipient_id"],
            "conversation_id": case["conversation_id"],
            "text": "Let me check with the owner and get back to you.",
            "reply_to_message_id": "mid_ig_in_1",
        }

        response = _respond_owner(client, case, handoff["handoff_id"], "Approve 10%, not 20%.")
        assert response.status_code == 200
        assert response.json()["queued"] is True
        drained = _drain_pending(app)
        detail = _handoff_detail(client, case, handoff["handoff_id"])

    assert drained == 1
    assert detail["status"] == "resolved"
    assert runtime.calls[-1]["owner_handoff_feedback"]["owner_feedback"] == "Approve 10%, not 20%."
    assert [item["arguments"]["text"] for item in _instagram_send_calls(composio, case)] == [
        "Let me check with the owner and get back to you.",
        "Owner approved 10%. I can offer that for your booking.",
    ]
    assert _instagram_send_calls(composio, case)[-1]["arguments"]["reply_to_message_id"] == "mid_ig_in_1"


@pytest.mark.ingress
def test_owner_handoff_instagram_spam_updates_existing_handoff_with_latest_context_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _OwnerHandoffRuntime(final_reply="I saw your updates. Owner approved tonight with 10% off.")
    composio = FakeComposioInstagramService(reply_fail_once_for_invalid_mid=False)
    app = _create_fake_instagram_app(tmp_path, monkeypatch, runtime, composio)
    case = _unique_instagram_case("spam")
    messages = [
        _instagram_message(case, message_id="mid_ig_out_1", minutes=10, role="assistant", text="Hi, how can I help?"),
        _instagram_message(
            case,
            message_id="mid_ig_in_1",
            minutes=11,
            role="customer",
            text="Can owner approve a custom slot?",
        ),
    ]
    _set_instagram_conversation(composio, case, messages)

    with TestClient(app) as client:
        workflow = _seed_instagram_handoff_workflow(app, case)
        _run_intake_workflow(
            client,
            customer_id=str(case["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
        )
        assert len(_handoffs(client, case)) == 1

        messages.extend(
            [
                _instagram_message(
                    case,
                    message_id="mid_ig_in_2",
                    minutes=12,
                    role="customer",
                    text="Also can it be tonight?",
                ),
                _instagram_message(
                    case,
                    message_id="mid_ig_in_3",
                    minutes=13,
                    role="customer",
                    text="Can you do 15% off too?",
                ),
                _instagram_message(
                    case,
                    message_id="mid_ig_in_4",
                    minutes=14,
                    role="customer",
                    text="Please confirm soon.",
                ),
                _instagram_message(
                    case,
                    message_id="mid_ig_in_5",
                    minutes=15,
                    role="customer",
                    text="My Instagram handle should show here.",
                ),
            ]
        )
        _set_instagram_conversation(composio, case, messages)
        _run_intake_workflow(
            client,
            customer_id=str(case["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
        )

        handoffs = _handoffs(client, case)
        assert len(handoffs) == 1
        handoff = handoffs[0]
        assert handoff["lead"]["username"] == case["lead_username"]
        assert handoff["latest_customer_message_preview"] == "My Instagram handle should show here."
        assert [item["text"] for item in handoff["messages"]["latest"]] == [
            "Also can it be tonight?",
            "Can you do 15% off too?",
            "Please confirm soon.",
            "My Instagram handle should show here.",
        ]
        assert [item["text"] for item in handoff["messages"]["previous"]] == [
            "Can owner approve a custom slot?",
        ]
        assert len(_instagram_send_calls(composio, case)) == 1

        response = _respond_owner(client, case, handoff["handoff_id"], "Approve tonight, but only 10% off.")
        assert response.status_code == 200
        drained = _drain_pending(app)
        detail = _handoff_detail(client, case, handoff["handoff_id"])

    assert drained == 1
    assert detail["status"] == "resolved"
    assert runtime.calls[-1]["owner_handoff_feedback"]["owner_feedback"] == "Approve tonight, but only 10% off."
    assert _instagram_send_calls(composio, case)[-1]["arguments"]["reply_to_message_id"] == "mid_ig_in_5"


@pytest.mark.live_llm
@pytest.mark.telegram
def test_live_llm_configured_handoff_simulated_lead_and_owner_advice_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _live_llm_key_configured():
        pytest.skip("OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY) required for live handoff e2e")
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_WEBHOOK_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(intake_service_module, "_TELEGRAM_BUSINESS_STALE_REQUEUE_SECONDS", 0.01)
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 60.0)
    get_settings.cache_clear()
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="live_owner_handoff",
    )
    case = {
        "business_connection_id": "bc_live_owner_handoff",
        "owner_user_id": 91501,
        "owner_chat_id": 92501,
        "lead_chat_id": 93501,
        "lead_user_id": 94501,
        "customer_id": "telegram_91501",
    }
    try:
        workflow = _seed_handoff_workflow(harness.client.app, case)
        _post_lead(
            harness.client,
            case,
            message_id=501,
            text=(
                "I want the premium detailing tomorrow at 10am, but I need a 20% discount. "
                "Please ask the owner if they can approve it."
            ),
        )
        assert _drain_pending(harness.client.app, limit=10) >= 1
        handoffs = _handoffs(harness.client, case)
        assert len(handoffs) == 1
        handoff = handoffs[0]
        assert handoff["status"] == "awaiting_owner"
        assert handoff["lead"]["username"] == f"lead_{case['lead_user_id']}"
        assert "20% discount" in handoff["latest_customer_message_preview"]

        owner_response = _respond_owner(
            harness.client,
            case,
            str(handoff["handoff_id"]),
            "Approve only 10% off. Do not mention that I rejected 20%; just offer 10% politely.",
        )
        assert owner_response.status_code == 200, owner_response.text
        assert _drain_pending(harness.client.app, limit=10) >= 1
        detail = _handoff_detail(harness.client, case, str(handoff["handoff_id"]))
        lead_replies = _sent_to_lead(harness.telegram_client, case)
        assert detail["status"] in {"resolved", "resolved_no_reply"}
        assert len(lead_replies) >= 1
        final_reply = str(lead_replies[-1].get("text", "") or "").strip()
        assert final_reply
        assert "backend error" not in final_reply.lower()

        report = harness.write_status_report(
            scenario="live_llm_configured_handoff_simulated_lead_and_owner_advice_e2e",
            ok=True,
            details={
                "customer_id": case["customer_id"],
                "workflow": {
                    "workflow_id": workflow["workflow_id"],
                    "handoff_rules": workflow["handoff_rules"],
                    "channel": workflow["channel"],
                    "provider": workflow["provider"],
                },
                "handoff": detail,
                "lead_replies": lead_replies,
                "final_reply": final_reply,
            },
        )
        assert report.exists()
    finally:
        close_harness(harness)


@pytest.mark.live_llm
@pytest.mark.ingress
def test_live_llm_instagram_owner_handoff_simulated_lead_and_owner_advice_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _live_llm_key_configured():
        pytest.skip("OPENAI_COMPATIBLE_API_KEY (or OPENROUTER_API_KEY) required for live handoff e2e")
    monkeypatch.setenv("OPENTULPA_WEB_TOKEN", "web-secret")
    monkeypatch.setattr(intake_service_module, "_PENDING_RUN_POLL_SECONDS", 60.0)
    get_settings.cache_clear()
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="live_instagram_owner_handoff",
    )
    case = {
        "customer_id": "cust_ig_live_owner_handoff",
        "connected_account_id": "acct_ig_live_owner_handoff",
        "conversation_id": "conv_ig_live_owner_handoff",
        "recipient_id": "ig_user_live_owner_handoff",
        "lead_username": "ig_lead_live_owner_handoff",
    }
    messages = [
        _instagram_message(
            case,
            message_id="mid_live_ig_out_1",
            minutes=10,
            role="assistant",
            text="Hi, how can I help with your booking?",
        ),
        _instagram_message(
            case,
            message_id="mid_live_ig_in_1",
            minutes=11,
            role="customer",
            text=(
                "I want the premium detailing tomorrow at 10am, but I need a 20% discount. "
                "Please ask the owner if they can approve it."
            ),
        ),
    ]
    try:
        harness.composio_service.reply_fail_once_for_invalid_mid = False
        _set_instagram_conversation(harness.composio_service, case, messages)
        workflow = _seed_instagram_handoff_workflow(harness.client.app, case)
        run = harness.run_workflow(
            customer_id=str(case["customer_id"]),
            workflow_id=str(workflow["workflow_id"]),
            event_type="instagram_live_handoff_e2e",
        )
        assert run["status_code"] == 200
        assert int(run["payload"]["processed_conversations"]) == 1
        handoffs = _handoffs(harness.client, case)
        assert len(handoffs) == 1
        handoff = handoffs[0]
        assert handoff["status"] == "awaiting_owner"
        assert handoff["lead"]["username"] == case["lead_username"]
        assert "20% discount" in handoff["latest_customer_message_preview"]

        response = _respond_owner(
            harness.client,
            case,
            str(handoff["handoff_id"]),
            "Approve only 10% off. Reply politely in the agent voice; do not mention rejected 20%.",
        )
        assert response.status_code == 200, response.text
        assert _drain_pending(harness.client.app, limit=10) >= 1
        detail = _handoff_detail(harness.client, case, str(handoff["handoff_id"]))
        instagram_replies = _instagram_send_calls(harness.composio_service, case)
        assert detail["status"] == "resolved"
        assert len(instagram_replies) >= 2
        final_reply = str(instagram_replies[-1]["arguments"].get("text", "") or "").strip()
        assert final_reply
        assert "backend error" not in final_reply.lower()
        assert instagram_replies[-1]["arguments"]["reply_to_message_id"] == "mid_live_ig_in_1"

        report = harness.write_status_report(
            scenario="live_llm_instagram_owner_handoff_simulated_lead_and_owner_advice_e2e",
            ok=True,
            details={
                "customer_id": case["customer_id"],
                "workflow": {
                    "workflow_id": workflow["workflow_id"],
                    "handoff_rules": workflow["handoff_rules"],
                    "channel": workflow["channel"],
                    "provider": workflow["provider"],
                },
                "handoff": detail,
                "instagram_replies": instagram_replies,
                "final_reply": final_reply,
            },
        )
        assert report.exists()
    finally:
        close_harness(harness)
