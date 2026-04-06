"""Approval policy/intention evaluator used by the approval broker."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from opentulpa.approvals.models import ActionIntent, GateAction, GateDecision, RecipientScope

EXTERNAL_DEFAULT_ACTIONS: set[str] = {
    "whatsapp_send",
    "email_send",
}

_SENSITIVE_KEY_PARTS = {"key", "token", "secret", "password", "authorization", "api"}


def _parse_impact(value: str, default: str = "read") -> str:
    raw = str(value or "").strip().lower()
    if raw in {"read", "write", "purchase", "costly"}:
        return raw
    return default


def _parse_scope(value: str, default: RecipientScope = "unknown") -> RecipientScope:
    raw = str(value or "").strip().lower()
    if raw in {"self", "external", "unknown"}:
        return raw  # type: ignore[return-value]
    return default


def _parse_gate(value: str, default: GateAction = "allow") -> GateAction:
    raw = str(value or "").strip().lower()
    if raw in {"allow", "require_approval", "deny"}:
        return raw  # type: ignore[return-value]
    return default


def _mask_sensitive(action_args: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (action_args or {}).items():
        key_text = str(key).strip()
        key_lower = key_text.lower()
        if any(part in key_lower for part in _SENSITIVE_KEY_PARTS):
            out[key_text] = "***"
            continue
        if isinstance(value, str):
            if key_lower in {"command", "script", "implementation_command", "code"}:
                out[key_text] = value[:12000]
            else:
                out[key_text] = value[:400]
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key_text] = value
        elif isinstance(value, list):
            out[key_text] = [str(item)[:120] for item in value[:10]]
        elif isinstance(value, dict):
            out[key_text] = {str(k)[:40]: str(v)[:120] for k, v in list(value.items())[:10]}
        else:
            out[key_text] = str(value)[:200]
    return out


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


class ApprovalEvaluator:
    """Compute intent + LLM gate decision for one tool action."""

    def __init__(self, *, runtime: Any | None = None) -> None:
        self._runtime = runtime

    @staticmethod
    def is_background_thread(thread_id: str) -> bool:
        raw = str(thread_id or "").strip().lower()
        return (
            raw.startswith("wake_")
            or raw.startswith("wake-")
            or raw.startswith("routine_")
            or raw.startswith("routine-")
        )

    @staticmethod
    def _resolve_recipient_scope(
        *,
        action_name: str,
        action_args: dict[str, Any],
        origin_conversation_id: str,
        origin_user_id: str,
    ) -> RecipientScope:
        if action_name in {"uploaded_file_send", "tulpa_file_send", "web_image_send", "tulpa_write_file"}:
            return "self"
        if action_name in EXTERNAL_DEFAULT_ACTIONS:
            return "external"

        destination = ""
        for key in (
            "chat_id",
            "channel_id",
            "destination",
            "recipient",
            "to",
            "email",
            "phone",
            "conversation_id",
            "target_user_id",
            "user_id",
        ):
            value = action_args.get(key)
            if value is None:
                continue
            destination = str(value).strip()
            if destination:
                break
        if not destination:
            return "unknown"
        if destination in {str(origin_conversation_id).strip(), str(origin_user_id).strip()}:
            return "self"
        return "external"

    async def _llm_intent_hint(
        self,
        *,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> dict[str, Any]:
        if self._runtime is None or not hasattr(self._runtime, "classify_guardrail_intent"):
            return {"ok": False, "error": "runtime_unavailable"}
        try:
            result = await self._runtime.classify_guardrail_intent(
                action_name=action_name,
                action_args=_mask_sensitive(action_args),
                action_note=str(action_note or "").strip()[:2000],
            )
        except Exception as exc:
            return {"ok": False, "error": f"classifier_error:{exc}"}
        if not isinstance(result, dict):
            return {"ok": False, "error": "classifier_invalid_payload"}
        if not bool(result.get("ok", True)):
            return {"ok": False, "error": str(result.get("error", "classifier_not_ok"))}
        return {"ok": True, **result}

    @staticmethod
    def summarize_action(action_name: str, action_args: dict[str, Any]) -> str:
        if action_name == "uploaded_file_send":
            return f"send file_id={str(action_args.get('file_id', '')).strip()[:60]}"
        if action_name == "tulpa_file_send":
            return f"send local file path={str(action_args.get('path', '')).strip()[:120]}"
        if action_name == "web_image_send":
            return f"send image from url={str(action_args.get('url', '')).strip()[:100]}"
        if action_name == "browser_use_run":
            return f"run browser task: {str(action_args.get('task', '')).strip()[:160]}"
        if action_name == "browser_use_task_control":
            return (
                f"browser task control task_id={str(action_args.get('task_id', '')).strip()[:50]} "
                f"action={str(action_args.get('action', 'stop_task_and_session')).strip()[:30]}"
            )
        if action_name == "tulpa_run_terminal":
            return f"run terminal command: {str(action_args.get('command', '')).strip()[:180]}"
        if action_name == "routine_create":
            routine_name = str(action_args.get("name", "")).strip()[:80]
            schedule = str(action_args.get("schedule", "")).strip()[:60]
            impl_cmd = str(action_args.get("implementation_command", "")).strip()
            if not impl_cmd and isinstance(action_args.get("execution"), dict):
                impl_cmd = str((action_args.get("execution") or {}).get("command", "")).strip()
            if impl_cmd:
                return (
                    f"create routine name={routine_name or 'unnamed'} schedule={schedule or 'unspecified'} "
                    f"run={impl_cmd[:140]}"
                )
            return (
                f"create routine name={routine_name or 'unnamed'} "
                f"schedule={schedule or 'unspecified'}"
            )
        return f"execute {action_name}"

    async def build_intent(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        origin_interface: str,
        origin_user_id: str,
        origin_conversation_id: str,
        action_note: str | None = None,
    ) -> ActionIntent:
        safe_action = str(action_name or "").strip()
        safe_note = str(action_note or "").strip()[:2000] or None

        recipient_scope = self._resolve_recipient_scope(
            action_name=safe_action,
            action_args=action_args,
            origin_conversation_id=origin_conversation_id,
            origin_user_id=origin_user_id,
        )
        impact = "read"
        gate: GateAction = "allow"
        reason = "llm_guardrail_default_allow"
        confidence = 0.0
        llm_uncertain = False

        hint = await self._llm_intent_hint(
            action_name=safe_action,
            action_args=action_args,
            action_note=safe_note,
        )
        if hint.get("ok"):
            gate = _parse_gate(hint.get("gate"), default=gate)
            impact = _parse_impact(hint.get("impact_type"), default=impact)
            llm_scope = _parse_scope(hint.get("recipient_scope"), default=recipient_scope)
            if llm_scope != "unknown" or recipient_scope == "unknown":
                recipient_scope = llm_scope
            reason = _first_non_empty(hint.get("reason"), reason)[:500]
            with_conf = hint.get("confidence", confidence)
            try:
                confidence = float(with_conf)
            except Exception:
                confidence = 0.6
            confidence = max(0.0, min(confidence, 1.0))
        else:
            reason = str(hint.get("error", "guardrail_classifier_failed"))[:500]
            llm_uncertain = True
            confidence = 0.0

        summary = self.summarize_action(safe_action, action_args)
        return ActionIntent(
            customer_id=str(customer_id or "").strip(),
            thread_id=str(thread_id or "").strip(),
            action_name=safe_action,
            action_args=action_args if isinstance(action_args, dict) else {},
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
            recipient_scope=recipient_scope,
            impact_type=_parse_impact(impact),  # type: ignore[arg-type]
            summary=summary,
            reason=reason,
            confidence=confidence,
            llm_gate=gate,
            llm_uncertain=llm_uncertain,
        )

    async def evaluate(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        origin_interface: str,
        origin_user_id: str,
        origin_conversation_id: str,
        action_note: str | None = None,
    ) -> tuple[ActionIntent, GateDecision]:
        intent = await self.build_intent(
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
            action_note=action_note,
        )
        llm_gate = str(intent.llm_gate or "").strip().lower()
        if llm_gate in {"allow", "require_approval", "deny"}:
            gate: GateAction = llm_gate  # type: ignore[assignment]
        else:
            gate = "allow"
        read_only_override = False
        if intent.impact_type == "read" and gate == "require_approval":
            gate = "allow"
            read_only_override = True
        if read_only_override:
            reason = "read_only_no_approval"
        elif gate == "allow" and intent.llm_uncertain:
            reason = "classifier_uncertain_allow"
        else:
            reason = str(intent.reason or "").strip()[:500] or "llm_guardrail_decision"
        decision = GateDecision(
            gate=gate,
            reason=reason,
            summary=intent.summary,
            confidence=intent.confidence,
            recipient_scope=intent.recipient_scope,
            impact_type=intent.impact_type,
        )
        return intent, decision

    @staticmethod
    def as_dict(decision: GateDecision) -> dict[str, Any]:
        return asdict(decision)
