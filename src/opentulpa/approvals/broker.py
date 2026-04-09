"""Approval broker for external-impact side effects."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from opentulpa.approvals.adapters.base import ApprovalAdapter
from opentulpa.approvals.store import PendingApprovalStore
from opentulpa.core.ids import new_short_id
from opentulpa.policy.evaluator import ApprovalEvaluator


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _execution_succeeded(result: Any) -> bool:
    """Interpret tool execution payloads conservatively for approval lifecycle."""
    if not isinstance(result, dict):
        # Non-dict payloads from older executors are treated as success.
        return True
    if bool(result.get("ok", True)) is False:
        return False
    if str(result.get("error", "")).strip():
        return False
    nested = result.get("result")
    if isinstance(nested, dict):
        if bool(nested.get("ok", True)) is False:
            return False
        if str(nested.get("error", "")).strip():
            return False
    return True


class ApprovalBroker:
    """Orchestrates approvals, persistence, and execution for side-effecting actions.

    Decision matrix (high-level):
    - Gate decisions (`allow`/`require_approval`/`deny`) come from LLM guardrail evaluator output.
    - Instant external actions that evaluate to `require_approval` create a challenge.
    - Only one pending approval is allowed per thread at a time; repeated guarded actions reuse the
      newest pending approval instead of creating a new challenge.
    - Scheduled external automations evaluate at routine creation time.
    - Background (`wake_*` / `routine_*`) runs are treated as pre-authorized scheduled execution.
      No per-run approval prompts/checks.
    - One approval record represents one executable action intent.
    """

    def __init__(
        self,
        *,
        store: PendingApprovalStore,
        runtime: Any | None = None,
        approval_ttl_minutes: int = 10,
        adapters: dict[str, ApprovalAdapter] | None = None,
        origin_resolver: Callable[[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._evaluator = ApprovalEvaluator(runtime=runtime)
        self._ttl_seconds = max(60, min(int(approval_ttl_minutes), 120)) * 60
        self._adapters = adapters or {}
        self._origin_resolver = origin_resolver

    def _resolve_origin(
        self,
        *,
        customer_id: str,
        thread_id: str,
        origin_interface: str | None,
        origin_user_id: str | None,
        origin_conversation_id: str | None,
    ) -> tuple[str, str, str]:
        interface = str(origin_interface or "").strip()
        user_id = str(origin_user_id or "").strip()
        conversation_id = str(origin_conversation_id or "").strip()
        if interface and user_id and conversation_id:
            return interface, user_id, conversation_id
        if self._origin_resolver is not None:
            resolved = self._origin_resolver(customer_id, thread_id)
            if isinstance(resolved, dict):
                interface = _first_non_empty(interface, resolved.get("origin_interface"))
                user_id = _first_non_empty(user_id, resolved.get("origin_user_id"))
                conversation_id = _first_non_empty(
                    conversation_id, resolved.get("origin_conversation_id")
                )
        interface = interface or "unknown"
        user_id = user_id or ""
        conversation_id = conversation_id or ""
        return interface, user_id, conversation_id

    async def _deliver_challenge(
        self,
        record: Any,
        *,
        defer_delivery: bool = False,
    ) -> str | None:
        adapter = self._adapters.get(record.origin_interface)
        if adapter is not None:
            if defer_delivery:
                queue_fn = getattr(adapter, "queue_challenge", None)
                if callable(queue_fn):
                    try:
                        queued = await queue_fn(record)
                    except Exception:
                        queued = False
                    if queued:
                        return f"{adapter.name}_deferred"
            try:
                sent = await adapter.send_challenge(record)
            except Exception:
                sent = False
            if sent:
                return adapter.name

        return None

    async def evaluate_action(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
        origin_interface: str | None = None,
        origin_user_id: str | None = None,
        origin_conversation_id: str | None = None,
        defer_challenge_delivery: bool = False,
    ) -> dict[str, Any]:
        interface, user_id, conversation_id = self._resolve_origin(
            customer_id=customer_id,
            thread_id=thread_id,
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
        )
        intent, decision = await self._evaluator.evaluate(
            customer_id=customer_id,
            thread_id=thread_id,
            action_name=action_name,
            action_args=action_args,
            origin_interface=interface,
            origin_user_id=user_id,
            origin_conversation_id=conversation_id,
            action_note=action_note,
        )
        if decision.gate != "require_approval":
            return self._evaluator.as_dict(decision)

        args_json = json.dumps(intent.action_args, sort_keys=True)
        is_background_thread = self._evaluator.is_background_thread(intent.thread_id)
        if is_background_thread and intent.action_name != "routine_create":
            decision.gate = "allow"
            decision.reason = "background_preauthorized_execution"
            decision.status = "approved"
            decision.approval_id = None
            decision.delivery_mode = None
            return self._evaluator.as_dict(decision)

        duplicate = self._store.find_pending_duplicate(
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            action_name=intent.action_name,
            action_args_json=args_json,
        )
        if duplicate is not None:
            decision.approval_id = duplicate.id
            decision.status = duplicate.status
            decision.expires_at = duplicate.expires_at
            decision.delivery_mode = "existing_pending"
            return self._evaluator.as_dict(decision)

        same_action_pending = self._store.find_latest_pending_for_customer_thread(
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            action_name=intent.action_name,
        )
        if same_action_pending is not None:
            decision.approval_id = same_action_pending.id
            decision.status = same_action_pending.status
            decision.expires_at = same_action_pending.expires_at
            decision.delivery_mode = "existing_pending_thread_action"
            decision.reason = "pending_approval_already_exists_for_action"
            decision.summary = same_action_pending.summary
            return self._evaluator.as_dict(decision)

        thread_pending = self._store.find_latest_pending_for_customer_thread(
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
        )
        if thread_pending is not None:
            decision.approval_id = thread_pending.id
            decision.status = thread_pending.status
            decision.expires_at = thread_pending.expires_at
            decision.delivery_mode = "existing_pending_thread"
            decision.reason = "pending_approval_already_exists_for_thread"
            decision.summary = thread_pending.summary
            return self._evaluator.as_dict(decision)

        approval_id = new_short_id("apr")
        record = self._store.create_pending(
            approval_id=approval_id,
            customer_id=intent.customer_id,
            thread_id=intent.thread_id,
            origin_interface=intent.origin_interface,
            origin_user_id=intent.origin_user_id,
            origin_conversation_id=intent.origin_conversation_id,
            action_name=intent.action_name,
            action_args=intent.action_args,
            recipient_scope=intent.recipient_scope,
            impact_type=intent.impact_type,
            summary=intent.summary,
            reason=intent.reason,
            confidence=float(intent.confidence),
            ttl_seconds=self._ttl_seconds,
        )
        delivery_mode = await self._deliver_challenge(
            record,
            defer_delivery=defer_challenge_delivery,
        )
        decision.approval_id = record.id
        decision.status = record.status
        decision.expires_at = record.expires_at
        decision.delivery_mode = delivery_mode
        return self._evaluator.as_dict(decision)

    async def flush_deferred_challenges(
        self,
        *,
        origin_interface: str,
        origin_conversation_id: str,
    ) -> int:
        interface = str(origin_interface or "").strip()
        conversation_id = str(origin_conversation_id or "").strip()
        if not interface or not conversation_id:
            return 0
        adapter = self._adapters.get(interface)
        if adapter is None:
            return 0
        flush_fn = getattr(adapter, "flush_challenges", None)
        if not callable(flush_fn):
            return 0
        try:
            count = await flush_fn(chat_id=conversation_id)
        except Exception:
            return 0
        try:
            return max(0, int(count))
        except Exception:
            return 0

    def get(self, approval_id: str) -> dict[str, Any] | None:
        return self._store.as_dict(self._store.get(approval_id))

    def list_pending_for_origin(
        self,
        *,
        origin_interface: str,
        origin_user_id: str,
        origin_conversation_id: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        records = self._store.list_pending_for_origin(
            origin_interface=origin_interface,
            origin_user_id=origin_user_id,
            origin_conversation_id=origin_conversation_id,
            limit=limit,
        )
        out: list[dict[str, Any]] = []
        for record in records:
            payload = self._store.as_dict(record)
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def has_pending_for_customer_thread(self, *, customer_id: str, thread_id: str) -> bool:
        return bool(
            self._store.has_pending_for_customer_thread(
                customer_id=customer_id,
                thread_id=thread_id,
            )
        )

    async def decide(
        self,
        *,
        approval_id: str,
        decision: str,
        actor_interface: str,
        actor_id: str,
    ) -> dict[str, Any]:
        record = self._store.get(approval_id)
        if record is None:
            return {"ok": False, "status": "denied", "reason": "approval_not_found"}
        if record.status != "pending":
            return {
                "ok": False,
                "status": "denied",
                "reason": f"approval_not_pending:{record.status}",
                "approval_id": record.id,
            }
        if str(actor_interface or "").strip() != str(record.origin_interface or "").strip():
            return {
                "ok": False,
                "status": "denied",
                "reason": "wrong_interface",
                "approval_id": record.id,
            }
        expected_actor = str(record.origin_user_id or "").strip()
        provided_actor = str(actor_id or "").strip()
        if not expected_actor or provided_actor != expected_actor:
            return {
                "ok": False,
                "status": "denied",
                "reason": "unauthorized_actor",
                "approval_id": record.id,
            }
        choice = str(decision or "").strip().lower()
        if choice not in {"approve", "deny"}:
            return {
                "ok": False,
                "status": "denied",
                "reason": "invalid_decision",
                "approval_id": record.id,
            }
        updated = self._store.set_decision(
            approval_id=record.id,
            decision=choice,
            actor_id=provided_actor,
        )
        payload = self._store.as_dict(updated)
        if payload is None:
            return {"ok": False, "status": "denied", "reason": "decision_persist_failed"}
        payload["ok"] = True
        payload["decision"] = choice
        return payload

    async def execute_approved_action(
        self,
        *,
        approval_id: str,
        customer_id: str,
        executor: Callable[[str, dict[str, Any], str], Awaitable[Any]],
    ) -> dict[str, Any]:
        record = self._store.get(approval_id)
        if record is None:
            return {"ok": False, "error": "approval_not_found"}
        if str(record.customer_id) != str(customer_id):
            return {"ok": False, "error": "customer_mismatch"}
        if record.status == "executed":
            # Idempotent replay handling: a duplicate execute call should be safe.
            return {
                "ok": True,
                "approval_id": record.id,
                "status": "executed",
                "action_name": record.action_name,
                "already_executed": True,
            }
        if record.status != "approved":
            return {"ok": False, "error": f"approval_not_executable:{record.status}"}

        try:
            action_args = json.loads(record.action_args_json)
        except Exception:
            action_args = {}
        if not isinstance(action_args, dict):
            action_args = {}

        result = await executor(record.action_name, action_args, record.customer_id)
        execution_ok = _execution_succeeded(result)
        updated = self._store.mark_executed(record.id) if execution_ok else self._store.get(record.id)
        status = updated.status if updated else ("executed" if execution_ok else "approved")
        return {
            "ok": True,
            "approval_id": record.id,
            "status": status,
            "action_name": record.action_name,
            "result": result,
            "execution_ok": execution_ok,
            "retryable": not execution_ok,
        }
