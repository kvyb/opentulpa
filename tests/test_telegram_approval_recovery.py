from __future__ import annotations

import pytest

from opentulpa.api.routes.telegram_webhook import (
    _run_post_approval_execution_flow,
    _run_post_denial_iteration_flow,
)


class _FakeContextEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def add_event(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(kwargs)
        return len(self.events)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def execute_tool(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        customer_id: str | None = None,
        inject_customer_id: bool = False,
    ):
        self.calls.append(
            {
                "kind": "execute",
                "action_name": action_name,
                "action_args": action_args,
                "customer_id": customer_id,
                "inject_customer_id": inject_customer_id,
            }
        )
        return {
            "ok": True,
            "approval_id": "apr_test",
            "status": "executed",
            "action_name": "tulpa_write_file",
            "result": {"error": 'write failed: {"detail":"router missing"}'},
            "error": 'write failed: {"detail":"router missing"}',
        }

    async def ainvoke_text(  # type: ignore[no-untyped-def]
        self,
        *,
        thread_id: str,
        customer_id: str,
        text: str,
        turn_mode: str = "interactive",
        include_pending_context: bool = True,
        recursion_limit_override: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "kind": "ainvoke",
                "thread_id": thread_id,
                "customer_id": customer_id,
                "turn_mode": turn_mode,
                "include_pending_context": include_pending_context,
                "recursion_limit_override": recursion_limit_override,
                "text": text,
            }
        )
        return "I fixed the script path and re-ran setup. Here is your auth link."


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.edited_markup: list[dict[str, object]] = []

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
    ) -> bool:
        self.sent.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
            }
        )
        return True

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        reply_markup: dict[str, object] | None = None,
    ) -> bool:
        self.edited_markup.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_markup": reply_markup or {},
            }
        )
        return True

    async def send_chat_action(self, *, chat_id: int | str, action: str = "typing") -> bool:
        _ = (chat_id, action)
        return True


class _FakeApprovals:
    def __init__(self) -> None:
        self.flush_calls: list[dict[str, str]] = []

    async def flush_deferred_challenges(
        self,
        *,
        origin_interface: str,
        origin_conversation_id: str,
    ) -> int:
        self.flush_calls.append(
            {
                "origin_interface": origin_interface,
                "origin_conversation_id": origin_conversation_id,
            }
        )
        return 1


class _FakeApprovalExecutionOrchestrator:
    async def execute_group_and_merge(
        self,
        *,
        approval_ids: list[str],
        decision_payload: dict[str, object],
        chat_id: int,
    ) -> str:
        _ = (approval_ids, decision_payload, chat_id)
        return "Completed approved actions:\n\ndone"


@pytest.mark.asyncio
async def test_post_approval_flow_flushes_deferred_challenges() -> None:
    client = _FakeTelegramClient()
    approvals = _FakeApprovals()
    orchestrator = _FakeApprovalExecutionOrchestrator()

    await _run_post_approval_execution_flow(
        get_telegram_client=lambda: client,
        get_approvals=lambda: approvals,
        get_approval_execution_orchestrator=lambda: orchestrator,
        approval_ids=["apr_test"],
        decision_payload={"customer_id": "telegram_42", "thread_id": "chat-42"},
        chat_id=42,
        approval_message_id=99,
    )

    assert approvals.flush_calls == [
        {
            "origin_interface": "telegram",
            "origin_conversation_id": "42",
        }
    ]
    assert any("Completed approved actions" in str(item.get("text", "")) for item in client.sent)


@pytest.mark.asyncio
async def test_post_denial_flow_flushes_deferred_challenges() -> None:
    runtime = _FakeRuntime()
    client = _FakeTelegramClient()
    approvals = _FakeApprovals()
    decision_payload = {
        "id": "apr_test",
        "customer_id": "telegram_42",
        "thread_id": "chat-42",
        "action_name": "tulpa_run_terminal",
        "summary": "post",
        "action_args": {"command": "curl -X POST https://example.com"},
    }

    await _run_post_denial_iteration_flow(
        get_telegram_client=lambda: client,
        get_agent_runtime=lambda: runtime,
        get_approvals=lambda: approvals,
        decision_payload=decision_payload,
        chat_id=42,
    )

    assert approvals.flush_calls == [
        {
            "origin_interface": "telegram",
            "origin_conversation_id": "42",
        }
    ]
    ainvoke_calls = [c for c in runtime.calls if c.get("kind") == "ainvoke"]
    assert len(ainvoke_calls) == 1
    assert ainvoke_calls[0].get("turn_mode") == "event_notification"
