from __future__ import annotations

import pytest

from opentulpa.handoffs import (
    HANDOFF_PLAN_NO_HANDOFF,
    HANDOFF_PLAN_REQUESTED,
    OWNER_HANDOFF_WEB_EVENT_KIND,
    HandoffDecisionContext,
    IntakeHandoffService,
    IntakeHandoffStore,
    normalize_handoff_rules,
    plan_handoff_request,
)


def _context() -> HandoffDecisionContext:
    return HandoffDecisionContext.from_runtime_inputs(
        workflow={
            "customer_id": "owner_1",
            "workflow_id": "iwf_1",
            "channel": "instagram_dm",
            "provider": "composio",
        },
        conversation_summary={
            "conversation_id": "conv_1",
            "latest_inbound_message_id": "msg_1",
            "latest_inbound_message_created_time": "2026-06-04T04:00:00+00:00",
            "latest_inbound_message_text_preview": "Can you do 20% off?",
            "incoming_user_id": "lead_1",
            "username": "alice",
        },
    )


def test_handoff_context_falls_back_to_latest_inbound_sender_username() -> None:
    context = HandoffDecisionContext.from_runtime_inputs(
        workflow={
            "customer_id": "owner_1",
            "workflow_id": "iwf_1",
            "channel": "telegram_business_dm",
            "provider": "telegram_bot_api",
        },
        conversation_summary={
            "conversation_id": "conv_1",
            "latest_inbound_sender_username": "lead_only",
        },
    )

    assert context.sender_username == "lead_only"
    assert context.lead_context(decision={})["sender_username"] == "lead_only"


def test_normalize_handoff_rules_makes_owner_rules_typed_and_stable() -> None:
    rules = normalize_handoff_rules(
        [
            {
                "id": "Discount Approval!",
                "label": "Discount approval",
                "condition": "Customer asks for a discount or price exception.",
                "owner_prompt_template": "Ask owner whether to approve or counter-offer.",
                "customer_wait_reply": "Let me check that.",
            }
        ]
    )

    assert rules == [
        {
            "id": "discount_approval",
            "label": "Discount approval",
            "condition": "Customer asks for a discount or price exception.",
            "owner_prompt": "Ask owner whether to approve or counter-offer.",
            "customer_wait_reply": "Let me check that.",
            "enabled": True,
        }
    ]


def test_normalize_handoff_rules_rejects_loose_string_rules() -> None:
    with pytest.raises(ValueError, match=r"handoff_rules\[0\] must be an object"):
        normalize_handoff_rules(["ask owner for discounts"])


def test_plan_handoff_request_is_noop_without_rules() -> None:
    plan = plan_handoff_request(
        workflow={"handoff_rules": []},
        decision={"handoff_action": "request_owner", "handoff_rule_id": "discount_approval"},
        context=_context(),
    )

    assert plan.kind == HANDOFF_PLAN_NO_HANDOFF
    assert plan.reason_code == "no_enabled_handoff_rules"


def test_plan_handoff_request_requires_explicit_configured_rule() -> None:
    workflow = {
        "handoff_rules": normalize_handoff_rules(
            [
                {
                    "id": "discount_approval",
                    "label": "Discount approval",
                    "condition": "Customer asks for discount approval.",
                }
            ]
        )
    }

    plan = plan_handoff_request(
        workflow=workflow,
        decision={"handoff_action": "request_owner", "handoff_rule_id": "unknown"},
        context=_context(),
    )

    assert plan.kind == HANDOFF_PLAN_NO_HANDOFF
    assert plan.reason_code == "unknown_handoff_rule"


def test_plan_handoff_request_returns_web_event_contract() -> None:
    workflow = {
        "handoff_rules": normalize_handoff_rules(
            [
                {
                    "id": "discount_approval",
                    "label": "Discount approval",
                    "condition": "Customer asks for discount approval.",
                    "owner_prompt": "Ask owner whether to approve a discount.",
                    "customer_wait_reply": "Let me check with the owner.",
                }
            ]
        )
    }
    decision = {
        "handoff_action": "request_owner",
        "handoff_rule_id": "discount_approval",
        "handoff_reason": "Customer asked for 20% off.",
        "handoff_request": "Need owner decision: approve 20%, counter, or decline.",
        "extracted_fields": {"requested_discount": "20%"},
        "missing_fields": ["owner_approval"],
        "conversation_summary": "Lead wants a discount before booking.",
    }

    plan = plan_handoff_request(
        workflow=workflow,
        decision=decision,
        context=_context(),
    )
    contract = plan.to_web_event_contract()

    assert plan.kind == HANDOFF_PLAN_REQUESTED
    assert plan.handoff_id_hint.startswith("hnd_")
    assert contract["kind"] == OWNER_HANDOFF_WEB_EVENT_KIND
    assert contract["thread_id"] == f"handoff:{plan.handoff_id_hint}"
    assert contract["metadata"]["trigger_rule_id"] == "discount_approval"
    assert contract["metadata"]["status"] == "awaiting_owner"
    assert contract["metadata"]["lead_context"]["extracted_fields"] == {"requested_discount": "20%"}


@pytest.mark.asyncio
async def test_handoff_service_response_is_single_use_and_queues_resume(tmp_path) -> None:
    service = IntakeHandoffService(store=IntakeHandoffStore(db_path=tmp_path / "handoffs.db"))
    calls: list[object] = []

    def _queue(handoff: object) -> dict[str, object]:
        calls.append(handoff)
        return {"ok": True, "queued": True}

    service.set_queue_callback(_queue)
    workflow = {
        "customer_id": "owner_1",
        "workflow_id": "iwf_1",
        "name": "Bookings",
        "channel": "instagram_dm",
        "provider": "composio",
        "handoff_rules": normalize_handoff_rules(
            [
                {
                    "id": "discount_approval",
                    "condition": "Customer asks for discount approval.",
                    "owner_prompt": "Approve discount?",
                }
            ]
        ),
    }
    decision = {
        "handoff_action": "request_owner",
        "handoff_rule_id": "discount_approval",
        "handoff_reason": "Customer asks for discount.",
        "handoff_request": "Approve discount?",
    }
    plan = plan_handoff_request(workflow=workflow, decision=decision, context=_context())
    opened = service.open_or_update(
        workflow=workflow,
        plan=plan,
        context=_context(),
        messages=[
            {
                "id": "msg_1",
                "sender_role": "customer",
                "text": "Can you do 20% off?",
                "created_time": "2026-06-04T04:00:00+00:00",
                "sender_username": "alice",
            }
        ],
    )

    first = await service.respond(
        customer_id="owner_1",
        handoff_id=opened["handoff_id"],
        owner_feedback="Approve 10%, not 20%.",
    )
    second = await service.respond(
        customer_id="owner_1",
        handoff_id=opened["handoff_id"],
        owner_feedback="Approve 20%.",
    )

    assert first["ok"] is True
    assert first["queued"] is True
    assert first["handoff"]["status"] == "owner_responded"
    assert second["ok"] is False
    assert second["status"] == "conflict"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_handoff_status_transitions_are_compare_and_swap(tmp_path) -> None:
    service = IntakeHandoffService(store=IntakeHandoffStore(db_path=tmp_path / "handoffs.db"))
    service.set_queue_callback(lambda handoff: {"ok": True, "queued": True, "handoff": handoff.to_api_dict()})
    workflow = {
        "customer_id": "owner_1",
        "workflow_id": "iwf_1",
        "name": "Bookings",
        "channel": "instagram_dm",
        "provider": "composio",
        "handoff_rules": normalize_handoff_rules(
            [{"id": "discount_approval", "condition": "Customer asks for discount approval."}]
        ),
    }
    decision = {
        "handoff_action": "request_owner",
        "handoff_rule_id": "discount_approval",
        "handoff_reason": "Customer asks for discount.",
        "handoff_request": "Approve discount?",
    }
    plan = plan_handoff_request(workflow=workflow, decision=decision, context=_context())
    opened = service.open_or_update(
        workflow=workflow,
        plan=plan,
        context=_context(),
        messages=[
            {
                "id": "msg_1",
                "sender_role": "customer",
                "text": "Can you do 20% off?",
                "created_time": "2026-06-04T04:00:00+00:00",
            }
        ],
    )
    await service.respond(
        customer_id="owner_1",
        handoff_id=opened["handoff_id"],
        owner_feedback="Approve 10%.",
    )
    owner_responded = service.get_handoff_record(customer_id="owner_1", handoff_id=opened["handoff_id"])
    assert owner_responded is not None

    running = service.mark_resuming(owner_responded)
    assert running is not None
    resolved = service.mark_resolved(running, no_reply=False)
    assert resolved is not None

    assert service.mark_resuming(resolved) is None
    assert service.mark_failed_reply(resolved, "late duplicate failure") is None
