from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from harness.runner import build_harness, close_harness
from mocks.composio_instagram import FakeComposioInstagramService, build_instagram_conversation

from opentulpa.api.app import create_app
from opentulpa.intake import service as intake_service_module
from opentulpa.integrations.composio import ComposioService
from opentulpa.scheduler.service import SchedulerService

pytestmark = [pytest.mark.e2e, pytest.mark.ingress]


class _DeterministicDraftRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.behavior_events: list[dict[str, Any]] = []
        self._link_alias_service = None
        self._decisions = [
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 1.0,
                "conversation_summary": "Customer wants a same-day booking.",
                "extracted_fields": {},
                "missing_fields": ["vehicle"],
                "reply_action": "send_reply",
                "reply_text": "BROKEN_PAYLOAD",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "First draft deliberately exercises invalid Composio payload recovery.",
            },
            {
                "ok": True,
                "matches_workflow": True,
                "confidence": 1.0,
                "conversation_summary": "Retry after Composio rejected the approved draft payload.",
                "extracted_fields": {},
                "missing_fields": ["vehicle"],
                "reply_action": "send_reply",
                "reply_text": "Yes, we can do 17:00 today. What vehicle should we book?",
                "ready_to_save": False,
                "booking_action": "ignore",
                "save_payload": {},
                "reason": "Repair the rejected outbound payload with a valid customer-facing reply.",
            },
        ]

    async def decide_intake_workflow(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self._decisions:
            raise AssertionError("unexpected intake decision call")
        return self._decisions.pop(0)

    def record_observability_event(
        self,
        *,
        event: str,
        customer_id: str | None = None,
        **fields: Any,
    ) -> None:
        if customer_id:
            fields["customer_id"] = customer_id
        self.behavior_events.append({"event": event, **fields})


class _ComposioInstagramDraftE2E(ComposioService):
    __slots__ = ("conversation", "sdk_calls", "_reject_broken_payload_once", "_reject_next_send")

    def __init__(self, *, conversation: dict[str, Any], reject_next_send: bool = False) -> None:
        super().__init__(api_key="e2e-composio-key")
        self.conversation = conversation
        self.sdk_calls: list[dict[str, Any]] = []
        self._reject_broken_payload_once = True
        self._reject_next_send = bool(reject_next_send)

    def _sdk_execute_tool(  # type: ignore[override]
        self,
        *,
        slug: str,
        arguments: dict[str, Any],
        connected_account_id: str | None,
        user_id: str,
        text: str | None = None,
    ) -> dict[str, Any]:
        call = {
            "slug": slug,
            "arguments": dict(arguments),
            "connected_account_id": connected_account_id,
            "user_id": user_id,
            "text": text,
        }
        self.sdk_calls.append(call)
        if slug == "INSTAGRAM_GET_CONVERSATION":
            assert arguments["conversation_id"] == "conv_draft_e2e_1"
            assert user_id == "cust_e2e_drafts"
            return {"successful": True, "error": None, "data": self.conversation}
        if slug == "INSTAGRAM_LIST_ALL_CONVERSATIONS":
            return {
                "successful": True,
                "error": None,
                "data": {"data": [{"id": "conv_draft_e2e_1"}]},
            }
        if slug == "INSTAGRAM_SEND_TEXT_MESSAGE":
            assert arguments["recipient_id"] == "178900099"
            assert arguments["text"]
            if self._reject_next_send or (
                arguments["text"] == "BROKEN_PAYLOAD" and self._reject_broken_payload_once
            ):
                self._reject_next_send = False
                self._reject_broken_payload_once = False
                return {
                    "successful": False,
                    "error": "Invalid request data provided: field 'text' is not valid for Instagram send.",
                    "data": {"status_code": 400},
                }
            return {
                "successful": True,
                "error": None,
                "data": {"id": "mid_sent_repaired", "echo_text": arguments["text"]},
            }
        raise AssertionError(f"unexpected Composio tool slug: {slug}")


def _post_json(client: TestClient, path: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload.get("ok") is True
    return payload


def _scheduled_instagram_burst_conversation() -> dict[str, Any]:
    conversation_id = "conv_live_llm_burst_1"
    recipient_id = "178900177"
    created_latest = "2026-04-14T10:24:30+0000"
    return {
        "summary": {
            "conversation_id": conversation_id,
            "recipient_id": recipient_id,
            "latest_message_id": "mid_burst_5",
            "latest_message_created_time": created_latest,
            "latest_inbound_message_id": "mid_burst_5",
            "latest_inbound_message_created_time": created_latest,
            "latest_inbound_message_text_preview": "Pleeeeease",
            "message_count": 6,
        },
        "conversation": {
            "id": conversation_id,
            "participants": {
                "data": [
                    {"id": "page_live_llm_burst", "username": "salon_test"},
                    {"id": recipient_id, "username": "lead_burst_e2e"},
                ]
            },
            "messages": {
                "data": [
                    {
                        "id": "mid_prev_out_1",
                        "created_time": "2026-04-14T10:19:30+0000",
                        "from": {"id": "page_live_llm_burst", "username": "salon_test"},
                        "to": {"data": [{"id": recipient_id}]},
                        "message": "We can book your blow dry on June 8. What time works, and what phone number should we use?",
                    },
                    {
                        "id": "mid_burst_1",
                        "created_time": "2026-04-14T10:20:00+0000",
                        "from": {"id": recipient_id, "username": "lead_burst_e2e"},
                        "to": {"data": [{"id": "page_live_llm_burst"}]},
                        "message": "Reach ma ballls",
                    },
                    {
                        "id": "mid_burst_2",
                        "created_time": "2026-04-14T10:22:00+0000",
                        "from": {"id": recipient_id, "username": "lead_burst_e2e"},
                        "to": {"data": [{"id": "page_live_llm_burst"}]},
                        "message": "Can u do that for me",
                    },
                    {
                        "id": "mid_burst_3",
                        "created_time": "2026-04-14T10:23:30+0000",
                        "from": {"id": recipient_id, "username": "lead_burst_e2e"},
                        "to": {"data": [{"id": "page_live_llm_burst"}]},
                        "message": "?",
                    },
                    {
                        "id": "mid_burst_5",
                        "created_time": created_latest,
                        "from": {"id": recipient_id, "username": "lead_burst_e2e"},
                        "to": {"data": [{"id": "page_live_llm_burst"}]},
                        "message": "Pleeeeease",
                    },
                ]
            },
        },
    }


def test_instagram_intake_draft_approval_resumes_and_repairs_composio_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opentulpa.api import app as app_module
    from opentulpa.tasks import sandbox as sandbox_module

    project_root = tmp_path / "project_root"
    project_root.mkdir()
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)

    conversation_item = build_instagram_conversation(
        conversation_id="conv_draft_e2e_1",
        recipient_id="178900099",
        inbound_text="Can I book a wash today around 5?",
    )
    conversation_item["conversation"]["participants"] = {
        "data": [
            {"id": "page_draft_e2e", "username": "autospa"},
            {"id": "178900099", "username": "lead_draft_e2e"},
        ]
    }
    runtime = _DeterministicDraftRuntime()
    composio = _ComposioInstagramDraftE2E(conversation=conversation_item["conversation"])
    app = create_app(
        agent_runtime=runtime,
        scheduler=SchedulerService(db_path=tmp_path / "scheduler.sqlite"),
        composio_service=composio,
    )

    with TestClient(app) as client:
        workflow_payload = _post_json(
            client,
            "/internal/intake/workflows/upsert",
            {
                "customer_id": "cust_e2e_drafts",
                "name": "E2E Instagram Draft Approval",
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {
                    "connected_account_id": "acct_e2e_drafts",
                    "conversation_id": "conv_draft_e2e_1",
                },
                "intent_description": "Reply to Instagram DMs asking to book a car wash.",
                "required_fields": ["vehicle", "time"],
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/e2e/drafts.csv"},
                "reply_mode": "draft",
                "notify_user": False,
                "enabled": True,
            },
        )
        workflow = workflow_payload["workflow"]
        assert workflow["reply_mode"] == "draft"

        run_payload = _post_json(
            client,
            "/internal/intake/workflows/run",
            {
                "customer_id": "cust_e2e_drafts",
                "workflow_id": workflow["workflow_id"],
                "force": True,
                "event_type": "manual_e2e",
            },
        )
        assert run_payload["results"][0]["status"] == "approval_pending"
        assert not [
            call for call in composio.sdk_calls if call["slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"
        ]

        drafts_payload = _post_json(
            client,
            "/internal/intake/drafts/list",
            {
                "customer_id": "cust_e2e_drafts",
                "workflow_id": workflow["workflow_id"],
            },
        )
        drafts = drafts_payload["drafts"]
        assert len(drafts) == 1
        assert drafts[0]["reply_text"] == "BROKEN_PAYLOAD"
        assert drafts[0]["metadata"]["decision"]["reply_text"] == "BROKEN_PAYLOAD"
        assert drafts[0]["metadata"]["conversation"]["id"] == "conv_draft_e2e_1"

        approved_payload = _post_json(
            client,
            "/internal/intake/drafts/approve",
            {
                "customer_id": "cust_e2e_drafts",
                "draft_id": drafts[0]["draft_id"],
            },
        )

    approved = approved_payload["draft"]
    send_calls = [call for call in composio.sdk_calls if call["slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"]
    assert approved["status"] == "sent"
    assert approved["reply_text"] == "Yes, we can do 17:00 today. What vehicle should we book?"
    assert len(send_calls) == 2
    assert send_calls[0]["arguments"]["text"] == "BROKEN_PAYLOAD"
    assert send_calls[1]["arguments"]["text"] == approved["reply_text"]
    assert "conversation_id" not in send_calls[0]["arguments"]
    assert len(runtime.calls) == 2
    assert runtime.calls[1]["execution_feedback"][0]["phase"] == "reply_execution"
    assert "field 'text' is not valid" in runtime.calls[1]["execution_feedback"][0]["error"]
    assert any(
        event["event"] == "intake.reply.approval_pending"
        for event in runtime.behavior_events
    )


@pytest.mark.live_llm
def test_live_llm_instagram_intake_draft_approval_repairs_composio_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_item = build_instagram_conversation(
        conversation_id="conv_draft_e2e_1",
        recipient_id="178900099",
        inbound_text="Can I book a car wash today around 5pm?",
    )
    conversation_item["conversation"]["participants"] = {
        "data": [
            {"id": "page_draft_e2e", "username": "autospa"},
            {"id": "178900099", "username": "lead_draft_e2e"},
        ]
    }
    composio = _ComposioInstagramDraftE2E(
        conversation=conversation_item["conversation"],
        reject_next_send=True,
    )
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="instagram_intake_drafts_live_llm",
        composio_service=composio,
    )
    try:
        workflow_payload = _post_json(
            harness.client,
            "/internal/intake/workflows/upsert",
            {
                "customer_id": "cust_e2e_drafts",
                "name": "Live LLM E2E Instagram Draft Approval",
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {
                    "connected_account_id": "acct_e2e_drafts",
                    "conversation_id": "conv_draft_e2e_1",
                },
                "intent_description": (
                    "Reply to Instagram DMs asking to book a car wash. If the lead asks "
                    "for an appointment and the vehicle is missing, ask one concise "
                    "follow-up question for the vehicle before saving."
                ),
                "required_fields": ["vehicle", "time"],
                "field_guidance": {
                    "vehicle": "The customer's car model or vehicle type.",
                    "time": "Requested appointment time.",
                },
                "assistant_instructions": (
                    "Live E2E contract: when this Instagram lead asks for a booking, "
                    "reply with a concise customer-facing question asking what vehicle "
                    "they want to book. Do not save a booking yet. If execution_feedback "
                    "says the Instagram send failed, retry with a valid concise reply."
                ),
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/e2e/live_llm_drafts.csv"},
                "reply_mode": "draft",
                "notify_user": False,
                "enabled": True,
            },
        )
        workflow = workflow_payload["workflow"]

        run_payload = _post_json(
            harness.client,
            "/internal/intake/workflows/run",
            {
                "customer_id": "cust_e2e_drafts",
                "workflow_id": workflow["workflow_id"],
                "force": True,
                "event_type": "manual_live_llm_e2e",
            },
        )
        assert run_payload["results"], run_payload
        assert run_payload["results"][0]["status"] == "approval_pending"
        assert not [
            call for call in composio.sdk_calls if call["slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"
        ]

        drafts_payload = _post_json(
            harness.client,
            "/internal/intake/drafts/list",
            {
                "customer_id": "cust_e2e_drafts",
                "workflow_id": workflow["workflow_id"],
            },
        )
        drafts = drafts_payload["drafts"]
        assert len(drafts) == 1
        assert str(drafts[0]["reply_text"]).strip()

        approved = _post_json(
            harness.client,
            "/internal/intake/drafts/approve",
            {
                "customer_id": "cust_e2e_drafts",
                "draft_id": drafts[0]["draft_id"],
            },
        )["draft"]

        send_calls = [call for call in composio.sdk_calls if call["slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"]
        assert approved["status"] == "sent"
        assert str(approved["reply_text"]).strip()
        assert len(send_calls) >= 2
        assert send_calls[0]["arguments"]["text"] == drafts[0]["reply_text"]
        assert send_calls[-1]["arguments"]["text"] == approved["reply_text"]
        assert "conversation_id" not in send_calls[-1]["arguments"]
    finally:
        close_harness(harness)


@pytest.mark.live_llm
def test_live_llm_instagram_scheduled_burst_replies_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        intake_service_module,
        "_utc_now",
        lambda: intake_service_module.datetime(2026, 4, 14, 10, 24, 45, tzinfo=intake_service_module.UTC),
    )
    conversation_item = _scheduled_instagram_burst_conversation()
    composio = FakeComposioInstagramService(reply_fail_once_for_invalid_mid=False)
    composio.conversations["conv_live_llm_burst_1"] = conversation_item
    harness = build_harness(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        scenario_name="instagram_scheduled_burst_live_llm",
        composio_service=composio,
    )
    try:
        workflow_payload = _post_json(
            harness.client,
            "/internal/intake/workflows/upsert",
            {
                "customer_id": "cust_e2e_burst",
                "name": "Live LLM E2E Instagram Burst",
                "channel": "instagram_dm",
                "provider": "composio",
                "source_config": {
                    "connected_account_id": "acct_e2e_burst",
                    "scan_limit": 20,
                },
                "intent_description": (
                    "Reply to Instagram DMs for salon booking requests. Treat multiple "
                    "customer messages after the last assistant reply as one active turn."
                ),
                "required_fields": ["time", "phone"],
                "field_guidance": {
                    "time": "Requested appointment time on June 8.",
                    "phone": "Customer phone number.",
                },
                "assistant_instructions": (
                    "Live E2E contract: answer as the salon. Use the whole active customer "
                    "turn, not only the latest short message. Since time and phone are still "
                    "missing, send one concise reply asking for both. Do not save a booking."
                ),
                "sink_type": "local_csv",
                "sink_config": {"file_path": "tulpa_stuff/e2e/live_llm_burst.csv"},
                "reply_mode": "auto",
                "notify_user": False,
                "enabled": True,
            },
        )
        workflow = workflow_payload["workflow"]
        assert workflow["schedule"] == "*/2 * * * *"

        first = _post_json(
            harness.client,
            "/internal/intake/workflows/run",
            {
                "customer_id": "cust_e2e_burst",
                "workflow_id": workflow["workflow_id"],
                "event_type": "scheduled",
            },
        )
        second = _post_json(
            harness.client,
            "/internal/intake/workflows/run",
            {
                "customer_id": "cust_e2e_burst",
                "workflow_id": workflow["workflow_id"],
                "event_type": "scheduled",
            },
        )

        send_calls = [
            call
            for call in composio.calls
            if call["method"] == "execute_tool"
            and call["tool_slug"] == "INSTAGRAM_SEND_TEXT_MESSAGE"
        ]
        assert first["processed_conversations"] == 1
        assert first["matched_conversations"] == 1
        assert first["results"][0]["replied"] is True
        assert second["processed_conversations"] == 0
        assert len(send_calls) == 1
        reply_text = str(send_calls[0]["arguments"]["text"])
        assert reply_text.strip()
        assert send_calls[0]["arguments"]["recipient_id"] == "178900177"
        assert send_calls[0]["arguments"]["reply_to_message_id"] == "mid_burst_5"
    finally:
        close_harness(harness)
