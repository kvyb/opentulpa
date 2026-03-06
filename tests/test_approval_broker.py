from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from opentulpa.approvals.broker import ApprovalBroker
from opentulpa.approvals.store import PendingApprovalStore


class _CaptureAdapter:
    name = "capture"
    interactive = True

    def __init__(self) -> None:
        self.sent: list[object] = []

    async def send_challenge(self, approval) -> bool:  # type: ignore[no-untyped-def]
        self.sent.append(approval)
        return True


class _DeferredCaptureAdapter:
    name = "capture"
    interactive = True

    def __init__(self) -> None:
        self.sent: list[object] = []
        self.queued: list[object] = []

    async def send_challenge(self, approval) -> bool:  # type: ignore[no-untyped-def]
        self.sent.append(approval)
        return True

    async def queue_challenge(self, approval) -> bool:  # type: ignore[no-untyped-def]
        self.queued.append(approval)
        return True

    async def flush_challenges(self, *, chat_id: str) -> int:
        _ = chat_id
        count = len(self.queued)
        self.sent.extend(self.queued)
        self.queued = []
        return count


class _ClassifierRuntime:
    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        action_note: str | None = None,
    ) -> dict[str, object]:
        _ = action_note
        if action_name == "browser_use_run":
            task = str(action_args.get("task", "")).strip().lower()
            if "submit" in task or "buy" in task:
                return {
                    "ok": True,
                    "gate": "require_approval",
                    "impact_type": "write",
                    "recipient_scope": "external",
                    "confidence": 0.9,
                    "reason": "task includes write intent",
                }
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "unknown",
                "confidence": 0.9,
                "reason": "task is read-only",
            }
        if action_name == "routine_create":
            combined = (
                f"{str(action_args.get('name', ''))} "
                f"{str(action_args.get('message', ''))}"
            ).strip().lower()
            if "post to x" in combined or "autopost" in combined:
                return {
                    "ok": True,
                    "gate": "require_approval",
                    "impact_type": "write",
                    "recipient_scope": "external",
                    "confidence": 0.9,
                    "reason": "routine targets external posting",
                }
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "self",
                "confidence": 0.85,
                "reason": "routine is internal research/summarization",
            }
        if action_name in {"slack_post", "email_send"}:
            return {
                "ok": True,
                "gate": "require_approval",
                "impact_type": "write",
                "recipient_scope": "external",
                "confidence": 0.9,
                "reason": "external side effects",
            }
        if action_name == "uploaded_file_send":
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "write",
                "recipient_scope": "self",
                "confidence": 0.9,
                "reason": "self-targeted send",
            }
        if action_name in {"routine_delete", "automation_delete"}:
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "self",
                "confidence": 0.9,
                "reason": "internal action",
            }
        if action_name == "tulpa_run_terminal":
            return {"ok": False, "error": "classifier_down"}
        return {
            "ok": True,
            "gate": "allow",
            "impact_type": "read",
            "recipient_scope": "unknown",
            "confidence": 0.7,
            "reason": "default",
        }


class _ParanoidClassifierRuntime:
    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        action_note: str | None = None,
    ) -> dict[str, object]:
        _ = action_name
        _ = action_args
        _ = action_note
        return {
            "ok": True,
            "gate": "require_approval",
            "impact_type": "write",
            "recipient_scope": "external",
            "confidence": 0.95,
            "reason": "paranoid-default",
        }


class _ReadParanoidClassifierRuntime:
    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        action_note: str | None = None,
    ) -> dict[str, object]:
        _ = action_name
        _ = action_args
        _ = action_note
        return {
            "ok": True,
            "gate": "require_approval",
            "impact_type": "read",
            "recipient_scope": "external",
            "confidence": 0.95,
            "reason": "paranoid-read-default",
        }


def _origin_resolver(customer_id: str, thread_id: str) -> dict[str, str]:
    assert customer_id
    assert thread_id
    return {
        "origin_interface": "telegram",
        "origin_user_id": "42",
        "origin_conversation_id": "4242",
    }


@pytest.fixture
def broker_factory(tmp_path: Path) -> Callable[..., tuple[ApprovalBroker, _CaptureAdapter]]:
    def _make(
        *,
        adapter: _CaptureAdapter | None = None,
        runtime: Any | None = None,
    ) -> tuple[ApprovalBroker, _CaptureAdapter]:
        resolved_adapter = adapter or _CaptureAdapter()
        broker = ApprovalBroker(
            store=PendingApprovalStore(db_path=tmp_path / "approvals.db"),
            runtime=runtime or _ClassifierRuntime(),
            adapters={"telegram": resolved_adapter},
            origin_resolver=_origin_resolver,
        )
        return broker, resolved_adapter

    return _make


@pytest.mark.asyncio
async def test_self_target_send_is_not_gated(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="uploaded_file_send",
        action_args={"file_id": "file_abc"},
    )
    assert result["gate"] == "allow"
    assert result.get("approval_id") is None


@pytest.mark.asyncio
async def test_routine_delete_is_internal_no_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="routine_delete",
        action_args={"routine_id": "rtn_abc", "customer_id": "telegram_42"},
    )
    assert result["gate"] == "allow"
    assert result.get("approval_id") is None
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_skill_upsert_follows_classifier_gate_when_classifier_is_paranoid(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory(runtime=_ParanoidClassifierRuntime())
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="skill_upsert",
        action_args={
            "customer_id": "telegram_42",
            "name": "growth-skill",
            "description": "desc",
            "instructions": "search and summarize",
        },
    )
    assert result["gate"] == "require_approval"
    assert result.get("approval_id")
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_external_read_never_requires_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory(runtime=_ReadParanoidClassifierRuntime())
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={"task": "Read docs from https://example.com and summarize"},
    )
    assert result["gate"] == "allow"
    assert result["reason"] == "read_only_no_approval"
    assert result.get("approval_id") is None
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_external_action_requires_approval_and_reuses_pending(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    request = {
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "slack_post",
        "action_args": {"channel_id": "C123", "text": "hello"},
    }
    first = await broker.evaluate_action(**request)
    second = await broker.evaluate_action(**request)
    assert first["gate"] == "require_approval"
    assert second["gate"] == "require_approval"
    assert first["approval_id"] == second["approval_id"]
    assert len(adapter.sent) == 1
    assert second.get("delivery_mode") == "existing_pending"


@pytest.mark.asyncio
async def test_external_action_can_defer_challenge_delivery(tmp_path: Path) -> None:
    adapter = _DeferredCaptureAdapter()
    broker = ApprovalBroker(
        store=PendingApprovalStore(db_path=tmp_path / "approvals_defer.db"),
        runtime=_ClassifierRuntime(),
        adapters={"telegram": adapter},
        origin_resolver=_origin_resolver,
    )
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="slack_post",
        action_args={"channel_id": "C123", "text": "hello"},
        defer_challenge_delivery=True,
    )

    assert result["gate"] == "require_approval"
    assert str(result.get("delivery_mode", "")).endswith("_deferred")
    assert len(adapter.queued) == 1
    assert len(adapter.sent) == 0

    flushed = await broker.flush_deferred_challenges(
        origin_interface="telegram",
        origin_conversation_id="4242",
    )
    assert flushed == 1
    assert len(adapter.queued) == 0
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_browser_read_vs_write_intent(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    read_result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={"task": "Open docs and summarize the page"},
    )
    write_result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={"task": "Open this site and submit the form"},
    )
    assert read_result["gate"] == "allow"
    assert write_result["gate"] == "require_approval"


@pytest.mark.asyncio
async def test_classifier_failure_allows_when_external_write_not_inferred(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="tulpa_run_terminal",
        action_args={"command": "curl -X POST https://example.com -d 'x=1'"},
    )
    assert result["gate"] == "allow"
    assert result["reason"] == "classifier_uncertain_allow"


@pytest.mark.asyncio
async def test_local_terminal_command_allows_when_external_write_not_inferred(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="tulpa_run_terminal",
        action_args={"command": "python3 gmail_setup.py"},
    )
    assert result["gate"] == "allow"
    assert result["reason"] == "classifier_uncertain_allow"


@pytest.mark.asyncio
async def test_exact_duplicate_after_executed_requests_new_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    request = {
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "slack_post",
        "action_args": {"channel_id": "C123", "text": "hello"},
    }

    first = await broker.evaluate_action(**request)
    approval_id = str(first.get("approval_id", "")).strip()
    assert first["gate"] == "require_approval"
    assert approval_id

    decided = await broker.decide(
        approval_id=approval_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert decided["ok"] is True
    assert decided["status"] == "approved"

    async def _executor(
        action_name: str,
        action_args: dict[str, object],
        customer_id: str,
    ) -> dict[str, object]:
        return {
            "ok": True,
            "action_name": action_name,
            "action_args": action_args,
            "customer_id": customer_id,
        }

    executed = await broker.execute_approved_action(
        approval_id=approval_id,
        customer_id="telegram_42",
        executor=_executor,
    )
    assert executed["ok"] is True

    again = await broker.evaluate_action(**request)
    assert again["gate"] == "require_approval"
    assert str(again.get("approval_id", "")).strip()
    assert str(again.get("approval_id", "")).strip() != approval_id
    assert len(adapter.sent) == 2


@pytest.mark.asyncio
async def test_failed_execution_requests_new_approval_on_retry(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    request = {
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "slack_post",
        "action_args": {"channel_id": "C123", "text": "hello"},
    }

    first = await broker.evaluate_action(**request)
    approval_id = str(first.get("approval_id", "")).strip()
    assert first["gate"] == "require_approval"
    assert approval_id

    decided = await broker.decide(
        approval_id=approval_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert decided["ok"] is True
    assert decided["status"] == "approved"

    async def _executor_fail(
        action_name: str,
        action_args: dict[str, object],
        customer_id: str,
    ) -> dict[str, object]:
        _ = action_name
        _ = action_args
        _ = customer_id
        return {"ok": False, "error": "transient network failure"}

    failed = await broker.execute_approved_action(
        approval_id=approval_id,
        customer_id="telegram_42",
        executor=_executor_fail,
    )
    assert failed["ok"] is True
    assert failed["status"] == "approved"
    assert failed["execution_ok"] is False
    assert failed["retryable"] is True

    again = await broker.evaluate_action(**request)
    assert again["gate"] == "require_approval"
    assert str(again.get("approval_id", "")).strip()
    assert str(again.get("approval_id", "")).strip() != approval_id
    assert len(adapter.sent) == 2


@pytest.mark.asyncio
async def test_background_thread_external_actions_allow_without_per_run_prompt(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    first = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake_abcd12",
        action_name="slack_post",
        action_args={"channel_id": "C123", "text": "hello"},
    )
    assert first["gate"] == "allow"
    assert first["reason"] == "background_preauthorized_execution"
    assert first.get("approval_id") is None
    assert len(adapter.sent) == 0

    again = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake_abcd12",
        action_name="email_send",
        action_args={"to": "a@example.com", "subject": "x", "text": "y"},
    )
    assert again["gate"] == "allow"
    assert again["reason"] == "background_preauthorized_execution"
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_background_thread_legacy_wake_prefix_allows_without_prompt(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake-legacy01",
        action_name="email_send",
        action_args={"to": "a@example.com", "subject": "x", "text": "y"},
    )
    assert result["gate"] == "allow"
    assert result["reason"] == "background_preauthorized_execution"
    assert result.get("approval_id") is None
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_background_thread_routine_prefix_allows_without_prompt(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="routine_rtn_legacy01",
        action_name="email_send",
        action_args={"to": "a@example.com", "subject": "x", "text": "y"},
    )
    assert result["gate"] == "allow"
    assert result["reason"] == "background_preauthorized_execution"
    assert result.get("approval_id") is None
    assert len(adapter.sent) == 0


@pytest.mark.asyncio
async def test_background_thread_external_routine_creation_still_requires_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="wake_abcd12",
        action_name="routine_create",
        action_args={
            "name": "Autopost to X",
            "schedule": "0 */2 * * *",
            "message": "Post to X every 2 hours",
            "customer_id": "telegram_42",
            "notify_user": True,
        },
    )
    assert result["gate"] == "require_approval"
    assert result.get("approval_id")
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
async def test_routine_create_internal_schedule_does_not_require_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="routine_create",
        action_args={
            "name": "Morning research digest",
            "schedule": "0 9 * * *",
            "message": "Scan AI news and summarize top 5 points here",
            "customer_id": "telegram_42",
            "notify_user": True,
        },
    )
    assert result["gate"] == "allow"


@pytest.mark.asyncio
async def test_routine_create_external_schedule_requires_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, _ = broker_factory()
    result = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="routine_create",
        action_args={
            "name": "Autopost to X",
            "schedule": "0 */2 * * *",
            "message": "Post to X every 2 hours with a short market reflection",
            "customer_id": "telegram_42",
            "notify_user": True,
        },
    )
    assert result["gate"] == "require_approval"


@pytest.mark.asyncio
async def test_browser_task_duplicate_after_executed_requests_new_approval(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    first = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={
            "task": "Go to agentmail.to and submit registration",
            "max_steps": 18,
            "sessionId": "sess_a",
        },
    )
    approval_id = str(first.get("approval_id", "")).strip()
    assert first["gate"] == "require_approval"
    assert approval_id

    decided = await broker.decide(
        approval_id=approval_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert decided["ok"] is True

    async def _executor(
        action_name: str,
        action_args: dict[str, object],
        customer_id: str,
    ) -> dict[str, object]:
        return {"ok": True}

    executed = await broker.execute_approved_action(
        approval_id=approval_id,
        customer_id="telegram_42",
        executor=_executor,
    )
    assert executed["ok"] is True

    again = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="browser_use_run",
        action_args={
            "task": "Go to agentmail.to and submit registration",
            "max_steps": 25,
            "sessionId": "sess_b",
        },
    )
    assert again["gate"] == "require_approval"
    assert str(again.get("approval_id", "")).strip()
    assert str(again.get("approval_id", "")).strip() != approval_id
    assert len(adapter.sent) == 2


@pytest.mark.asyncio
async def test_multiple_approvals_are_independent(
    broker_factory: Callable[..., tuple[ApprovalBroker, _CaptureAdapter]],
) -> None:
    broker, adapter = broker_factory()
    first = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="slack_post",
        action_args={"channel_id": "C1", "text": "one"},
    )
    second = await broker.evaluate_action(
        customer_id="telegram_42",
        thread_id="chat-42",
        action_name="email_send",
        action_args={"to": "a@example.com", "text": "two"},
    )
    first_id = str(first.get("approval_id", "")).strip()
    second_id = str(second.get("approval_id", "")).strip()
    assert first_id and second_id and first_id != second_id
    assert len(adapter.sent) == 2

    first_decision = await broker.decide(
        approval_id=first_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert first_decision["ok"] is True
    assert first_decision["status"] == "approved"

    second_decision = await broker.decide(
        approval_id=second_id,
        decision="approve",
        actor_interface="telegram",
        actor_id="42",
    )
    assert second_decision["ok"] is True
    assert second_decision["status"] == "approved"
