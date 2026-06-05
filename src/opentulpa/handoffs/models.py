"""Typed owner handoff rule, storage, and runtime contracts."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

HANDOFF_ACTION_REQUEST_OWNER = "request_owner"
HANDOFF_PLAN_NO_HANDOFF = "no_handoff"
HANDOFF_PLAN_REQUESTED = "handoff_requested"
OWNER_HANDOFF_RESUME_EVENT_TYPE = "owner_handoff_resume"
OWNER_HANDOFF_WEB_EVENT_KIND = "handoff.updated"
OWNER_HANDOFF_WEB_EVENT_SOURCE = "intake_workflow"
HANDOFF_STATUS_AWAITING_OWNER = "awaiting_owner"
HANDOFF_STATUS_OWNER_RESPONDED = "owner_responded"
HANDOFF_STATUS_RESUMING = "resuming"
HANDOFF_STATUS_RESOLVED = "resolved"
HANDOFF_STATUS_RESOLVED_NO_REPLY = "resolved_no_reply"
HANDOFF_STATUS_FAILED_REPLY = "failed_reply"
HANDOFF_STATUS_EXPIRED = "expired"
HANDOFF_STATUS_CANCELED_BY_CUSTOMER_UPDATE = "canceled_by_customer_update"
HANDOFF_LIFECYCLE_STATUSES = (
    HANDOFF_STATUS_AWAITING_OWNER,
    HANDOFF_STATUS_OWNER_RESPONDED,
    HANDOFF_STATUS_RESUMING,
    HANDOFF_STATUS_RESOLVED,
    HANDOFF_STATUS_RESOLVED_NO_REPLY,
    HANDOFF_STATUS_FAILED_REPLY,
    HANDOFF_STATUS_EXPIRED,
    HANDOFF_STATUS_CANCELED_BY_CUSTOMER_UPDATE,
)
HANDOFF_NON_TERMINAL_STATUSES = (
    HANDOFF_STATUS_AWAITING_OWNER,
    HANDOFF_STATUS_OWNER_RESPONDED,
    HANDOFF_STATUS_RESUMING,
)

HandoffStatus = Literal[
    "awaiting_owner",
    "owner_responded",
    "resuming",
    "resolved",
    "resolved_no_reply",
    "failed_reply",
    "expired",
    "canceled_by_customer_update",
]
HandoffPlanKind = Literal["no_handoff", "handoff_requested"]

_MAX_HANDOFF_RULES = 20
_MAX_RULE_LABEL_CHARS = 120
_MAX_RULE_CONDITION_CHARS = 1000
_MAX_OWNER_PROMPT_CHARS = 1200
_MAX_WAIT_REPLY_CHARS = 500
_MAX_CONTEXT_ITEMS = 20
_MAX_CONTEXT_VALUE_CHARS = 500


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FrozenContract(_Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().casefold()
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    return default


def _trim_text(value: Any, *, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def _safe_rule_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:64].strip("_")


def _stable_rule_id(*, raw_id: Any, label: str, condition: str) -> str:
    explicit = _safe_rule_token(raw_id)
    if explicit:
        return explicit
    base = _safe_rule_token(label) or _safe_rule_token(condition[:80]) or "handoff"
    digest = sha256(f"{label}|{condition}".encode()).hexdigest()[:8]
    return f"{base[:40].rstrip('_')}_{digest}"


def _compact_context_dict(value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for raw_key, raw_value in list(_safe_dict(value).items())[:_MAX_CONTEXT_ITEMS]:
        key = str(raw_key or "").strip()
        if not key:
            continue
        if isinstance(raw_value, (bool, int, float)) or raw_value is None:
            out[key] = raw_value
            continue
        if isinstance(raw_value, (dict, list)):
            rendered = json.dumps(raw_value, ensure_ascii=False, sort_keys=True)
            out[key] = _trim_text(rendered, limit=_MAX_CONTEXT_VALUE_CHARS)
            continue
        out[key] = _trim_text(raw_value, limit=_MAX_CONTEXT_VALUE_CHARS)
    return out


class HandoffRule(_FrozenContract):
    """Owner-configured condition that permits intake to ask for human guidance."""

    id: str
    label: str
    condition: str
    owner_prompt: str
    customer_wait_reply: str = ""
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_rule(self) -> HandoffRule:
        assert self.id == self.id.strip()
        assert self.condition == self.condition.strip()
        assert self.condition
        assert self.owner_prompt == self.owner_prompt.strip()
        return self

    @classmethod
    def from_mapping(cls, value: dict[str, Any], *, index: int) -> HandoffRule:
        label = _trim_text(
            value.get("label") or value.get("title"),
            limit=_MAX_RULE_LABEL_CHARS,
        )
        condition = _trim_text(value.get("condition"), limit=_MAX_RULE_CONDITION_CHARS)
        if not condition:
            raise ValueError(f"handoff_rules[{index}].condition is required")
        if not label:
            label = _trim_text(condition, limit=80)
        owner_prompt = _trim_text(
            value.get("owner_prompt")
            or value.get("owner_prompt_template")
            or value.get("prompt")
            or condition,
            limit=_MAX_OWNER_PROMPT_CHARS,
        )
        rule = cls(
            id=_stable_rule_id(
                raw_id=value.get("id") or value.get("rule_id"),
                label=label,
                condition=condition,
            ),
            label=label,
            condition=condition,
            owner_prompt=owner_prompt,
            customer_wait_reply=_trim_text(
                value.get("customer_wait_reply"),
                limit=_MAX_WAIT_REPLY_CHARS,
            ),
            enabled=_as_bool(value.get("enabled"), default=True),
        )
        assert rule.id
        assert len(rule.condition) <= _MAX_RULE_CONDITION_CHARS
        return rule

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class HandoffDecisionContext(_FrozenContract):
    """Runtime context needed to create a handoff request contract."""

    customer_id: str
    workflow_id: str
    conversation_id: str
    source_channel: str
    source_provider: str
    latest_inbound_message_id: str = ""
    latest_inbound_message_at: str = ""
    latest_inbound_text: str = ""
    sender_id: str = ""
    sender_username: str = ""

    @model_validator(mode="after")
    def _validate_context(self) -> HandoffDecisionContext:
        assert self.customer_id == self.customer_id.strip()
        assert self.workflow_id == self.workflow_id.strip()
        assert self.conversation_id == self.conversation_id.strip()
        assert self.customer_id
        assert self.workflow_id
        assert self.conversation_id
        return self

    @classmethod
    def from_runtime_inputs(
        cls,
        *,
        workflow: dict[str, Any],
        conversation_summary: dict[str, Any],
    ) -> HandoffDecisionContext:
        summary = _safe_dict(conversation_summary)
        return cls(
            customer_id=str(workflow.get("customer_id", "") or "").strip(),
            workflow_id=str(workflow.get("workflow_id", "") or "").strip(),
            conversation_id=str(summary.get("conversation_id", "") or "").strip(),
            source_channel=str(workflow.get("channel", "") or "").strip(),
            source_provider=str(workflow.get("provider", "") or "").strip(),
            latest_inbound_message_id=str(summary.get("latest_inbound_message_id", "") or "").strip(),
            latest_inbound_message_at=str(summary.get("latest_inbound_message_created_time", "") or "").strip(),
            latest_inbound_text=str(summary.get("latest_inbound_message_text_preview", "") or "").strip(),
            sender_id=str(
                summary.get("incoming_user_id") or summary.get("latest_inbound_sender_id") or ""
            ).strip(),
            sender_username=str(
                summary.get("username") or summary.get("latest_inbound_sender_username") or ""
            ).strip(),
        )

    def lead_context(self, *, decision: dict[str, Any]) -> dict[str, Any]:
        context = {
            "customer_id": self.customer_id,
            "workflow_id": self.workflow_id,
            "conversation_id": self.conversation_id,
            "source_channel": self.source_channel,
            "source_provider": self.source_provider,
            "latest_inbound_message_id": self.latest_inbound_message_id,
            "latest_inbound_message_at": self.latest_inbound_message_at,
            "latest_inbound_text": _trim_text(self.latest_inbound_text, limit=_MAX_CONTEXT_VALUE_CHARS),
            "sender_id": self.sender_id,
            "sender_username": self.sender_username,
            "conversation_summary": _trim_text(
                decision.get("conversation_summary"),
                limit=_MAX_CONTEXT_VALUE_CHARS,
            ),
            "extracted_fields": _compact_context_dict(decision.get("extracted_fields")),
            "missing_fields": [
                _trim_text(item, limit=80)
                for item in _safe_list(decision.get("missing_fields"))[:_MAX_CONTEXT_ITEMS]
                if str(item or "").strip()
            ],
        }
        return {key: value for key, value in context.items() if value not in ("", [], {})}


class HandoffMessage(_Contract):
    message_id: str
    direction: Literal["inbound", "outbound"]
    text: str
    created_at: str = ""
    sender_username: str = ""
    sender_id: str = ""


class HandoffMessages(_Contract):
    latest: list[HandoffMessage] = Field(default_factory=list)
    previous: list[HandoffMessage] = Field(default_factory=list)

    def to_api_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "latest": [item.model_dump() for item in self.latest],
            "previous": [item.model_dump() for item in self.previous],
        }


class HandoffLead(_Contract):
    username: str = ""
    display_name: str = ""
    platform_user_id: str = ""


class HandoffTrigger(_Contract):
    rule_id: str
    rule_label: str
    reason: str
    owner_prompt: str
    customer_wait_reply: str = ""


class HandoffOpenRequest(_Contract):
    handoff_id: str
    customer_id: str
    workflow_id: str
    workflow_name: str = ""
    source_channel: str
    source_provider: str
    conversation_id: str
    trigger: HandoffTrigger
    lead: HandoffLead
    messages: HandoffMessages
    latest_inbound_message_id: str = ""
    latest_customer_message_preview: str = ""

    @model_validator(mode="after")
    def _validate_open_request(self) -> HandoffOpenRequest:
        assert self.handoff_id.startswith("hnd_")
        assert self.customer_id and self.workflow_id and self.conversation_id
        assert self.trigger.owner_prompt
        return self


class HandoffRecord(_Contract):
    handoff_id: str
    status: HandoffStatus
    customer_id: str
    workflow_id: str
    workflow_name: str = ""
    source_channel: str
    source_provider: str
    conversation_id: str
    lead: HandoffLead
    trigger: HandoffTrigger
    messages: HandoffMessages
    latest_inbound_message_id: str = ""
    latest_customer_message_preview: str = ""
    owner_feedback: str = ""
    failure_reason: str = ""
    created_at: str
    updated_at: str
    responded_at: str = ""
    resolved_at: str = ""
    created: bool | None = None

    def to_api_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"created"})
        payload["lead"] = self.lead.model_dump()
        payload["trigger"] = self.trigger.model_dump()
        payload["messages"] = self.messages.to_api_dict()
        if self.created is not None:
            payload["created"] = self.created
        return payload


class HandoffPlan(_FrozenContract):
    """Pure handoff planning result, before persistence or delivery."""

    kind: HandoffPlanKind
    reason_code: str
    customer_id: str = ""
    workflow_id: str = ""
    conversation_id: str = ""
    latest_inbound_message_id: str = ""
    source_channel: str = ""
    source_provider: str = ""
    rule: HandoffRule | None = None
    trigger_reason: str = ""
    owner_prompt: str = ""
    customer_wait_reply: str = ""
    handoff_id_hint: str = ""
    lead_context: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_plan(self) -> HandoffPlan:
        assert self.kind in {HANDOFF_PLAN_NO_HANDOFF, HANDOFF_PLAN_REQUESTED}
        assert self.reason_code == self.reason_code.strip()
        assert self.reason_code
        if self.kind == HANDOFF_PLAN_NO_HANDOFF:
            assert self.rule is None
            return self
        assert self.rule is not None
        assert self.customer_id and self.workflow_id and self.conversation_id
        assert self.owner_prompt == self.owner_prompt.strip()
        assert self.owner_prompt
        assert self.handoff_id_hint.startswith("hnd_")
        return self

    @classmethod
    def no_handoff(cls, reason_code: str) -> HandoffPlan:
        return cls(kind="no_handoff", reason_code=reason_code)

    @classmethod
    def requested(
        cls,
        *,
        rule: HandoffRule,
        context: HandoffDecisionContext,
        trigger_reason: str,
        owner_prompt: str,
        customer_wait_reply: str,
        lead_context: dict[str, Any],
    ) -> HandoffPlan:
        request_key = "|".join(
            [
                context.customer_id,
                context.workflow_id,
                context.conversation_id,
                context.latest_inbound_message_id or "latest",
                rule.id,
            ]
        )
        handoff_id_hint = "hnd_" + sha256(request_key.encode()).hexdigest()[:16]
        return cls(
            kind="handoff_requested",
            reason_code=HANDOFF_ACTION_REQUEST_OWNER,
            customer_id=context.customer_id,
            workflow_id=context.workflow_id,
            conversation_id=context.conversation_id,
            latest_inbound_message_id=context.latest_inbound_message_id,
            source_channel=context.source_channel,
            source_provider=context.source_provider,
            rule=rule,
            trigger_reason=trigger_reason,
            owner_prompt=owner_prompt,
            customer_wait_reply=customer_wait_reply,
            handoff_id_hint=handoff_id_hint,
            lead_context=lead_context,
        )

    def to_web_event_contract(self) -> dict[str, Any]:
        if self.kind == HANDOFF_PLAN_NO_HANDOFF:
            return {"kind": HANDOFF_PLAN_NO_HANDOFF, "reason_code": self.reason_code}
        assert self.rule is not None
        metadata = {
            "handoff_id_hint": self.handoff_id_hint,
            "status": HANDOFF_STATUS_AWAITING_OWNER,
            "workflow_id": self.workflow_id,
            "conversation_id": self.conversation_id,
            "latest_inbound_message_id": self.latest_inbound_message_id,
            "trigger_rule_id": self.rule.id,
            "trigger_rule_label": self.rule.label,
            "trigger_reason": self.trigger_reason,
            "owner_prompt": self.owner_prompt,
            "customer_wait_reply": self.customer_wait_reply,
            "lead_context": self.lead_context,
        }
        return {
            "source": OWNER_HANDOFF_WEB_EVENT_SOURCE,
            "kind": OWNER_HANDOFF_WEB_EVENT_KIND,
            "thread_id": f"handoff:{self.handoff_id_hint}",
            "text": self.owner_prompt,
            "metadata": metadata,
        }


def parse_handoff_rules(value: Any) -> list[HandoffRule]:
    if value is None:
        return []
    items = _safe_list(value)
    if not isinstance(value, list):
        raise ValueError("handoff_rules must be a list")
    if len(items) > _MAX_HANDOFF_RULES:
        raise ValueError(f"handoff_rules must contain at most {_MAX_HANDOFF_RULES} entries")
    rules: list[HandoffRule] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"handoff_rules[{index}] must be an object")
        rule = HandoffRule.from_mapping(item, index=index)
        if rule.id in seen:
            raise ValueError(f"handoff_rules duplicate id: {rule.id}")
        seen.add(rule.id)
        rules.append(rule)
    assert len(rules) <= _MAX_HANDOFF_RULES
    return rules


def normalize_handoff_rules(value: Any) -> list[dict[str, Any]]:
    return [rule.to_dict() for rule in parse_handoff_rules(value)]


def plan_handoff_request(
    *,
    workflow: dict[str, Any],
    decision: dict[str, Any],
    context: HandoffDecisionContext,
) -> HandoffPlan:
    rules = [rule for rule in parse_handoff_rules(workflow.get("handoff_rules")) if rule.enabled]
    if not rules:
        return HandoffPlan.no_handoff("no_enabled_handoff_rules")

    action = str(decision.get("handoff_action", "") or "none").strip().lower()
    if action in {"", "none", "no_handoff"}:
        return HandoffPlan.no_handoff("decision_did_not_request_handoff")
    if action != HANDOFF_ACTION_REQUEST_OWNER:
        return HandoffPlan.no_handoff("unsupported_handoff_action")

    requested_rule_id = _safe_rule_token(decision.get("handoff_rule_id"))
    rule_by_id = {rule.id: rule for rule in rules}
    rule = rule_by_id.get(requested_rule_id)
    if rule is None:
        return HandoffPlan.no_handoff("unknown_handoff_rule")

    trigger_reason = _trim_text(
        decision.get("handoff_reason") or decision.get("reason") or rule.condition,
        limit=_MAX_OWNER_PROMPT_CHARS,
    )
    owner_prompt = _trim_text(
        decision.get("handoff_request")
        or decision.get("owner_prompt")
        or rule.owner_prompt
        or trigger_reason,
        limit=_MAX_OWNER_PROMPT_CHARS,
    )
    if not owner_prompt:
        return HandoffPlan.no_handoff("empty_owner_prompt")
    customer_wait_reply = _trim_text(
        decision.get("customer_wait_reply") or rule.customer_wait_reply,
        limit=_MAX_WAIT_REPLY_CHARS,
    )
    return HandoffPlan.requested(
        rule=rule,
        context=context,
        trigger_reason=trigger_reason,
        owner_prompt=owner_prompt,
        customer_wait_reply=customer_wait_reply,
        lead_context=context.lead_context(decision=decision),
    )
