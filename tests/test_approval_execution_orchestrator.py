from __future__ import annotations

import pytest

from opentulpa.application.approval_execution import (
    ApprovalExecutionOrchestrator,
    _extract_execution_error_text,
)


def test_extract_execution_error_text_top_level_error() -> None:
    payload = {"ok": False, "error": "approval_not_found"}
    assert _extract_execution_error_text(payload) == "approval_not_found"


def test_extract_execution_error_text_nested_error() -> None:
    payload = {
        "ok": True,
        "approval_id": "apr_123",
        "status": "executed",
        "result": {"ok": False, "error": "terminal failed: working_dir invalid"},
    }
    assert _extract_execution_error_text(payload) == "terminal failed: working_dir invalid"


def test_extract_execution_error_text_nested_stderr_when_ok_false() -> None:
    payload = {
        "ok": True,
        "result": {
            "ok": False,
            "returncode": 2,
            "stderr": "python3: can't open file",
        },
    }
    assert _extract_execution_error_text(payload) == "python3: can't open file"


class _FakeContextEvents:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def add_event(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(dict(kwargs))


class _RetryRuntime:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, object]] = []
        self.ainvoke_calls: list[dict[str, object]] = []
        self._n = 0

    async def execute_tool(self, **kwargs):  # type: ignore[no-untyped-def]
        self.execute_calls.append(dict(kwargs))
        self._n += 1
        if self._n == 1:
            return {
                "ok": True,
                "approval_id": "apr_x",
                "status": "approved",
                "action_name": "tulpa_run_terminal",
                "result": {
                    "ok": False,
                    "returncode": 2,
                    "stderr": "python3: can't open file '/tmp/tulpa_stuff/tulpa_stuff/tg_login.py'",
                },
                "execution_ok": False,
                "retryable": True,
            }
        return {
            "ok": True,
            "approval_id": "apr_x",
            "status": "executed",
            "action_name": "tulpa_run_terminal",
            "result": {"ok": True, "returncode": 0, "stdout": "done", "stderr": ""},
            "execution_ok": True,
            "retryable": False,
        }

    async def ainvoke_text(self, **kwargs):  # type: ignore[no-untyped-def]
        self.ainvoke_calls.append(dict(kwargs))
        return "should-not-be-called"


class _RecoveryThreadRuntime:
    def __init__(self) -> None:
        self.execute_calls: list[dict[str, object]] = []
        self.ainvoke_calls: list[dict[str, object]] = []

    async def execute_tool(self, **kwargs):  # type: ignore[no-untyped-def]
        self.execute_calls.append(dict(kwargs))
        return {
            "ok": True,
            "approval_id": "apr_y",
            "status": "approved",
            "action_name": "tulpa_run_terminal",
            "result": {"ok": False, "stderr": "python3: can't open file"},
            "execution_ok": False,
            "retryable": False,
        }

    async def ainvoke_text(self, **kwargs):  # type: ignore[no-untyped-def]
        self.ainvoke_calls.append(dict(kwargs))
        return "I retried and fixed it."


@pytest.mark.asyncio
async def test_orchestrator_retries_retryable_execution_once_before_llm_recovery() -> None:
    runtime = _RetryRuntime()
    events = _FakeContextEvents()
    orchestrator = ApprovalExecutionOrchestrator(
        get_agent_runtime=lambda: runtime,
        get_context_events=lambda: events,
    )
    out = await orchestrator.execute_approved_action_and_summarize(
        approval_id="apr_x",
        decision_payload={
            "customer_id": "telegram_1",
            "thread_id": "chat_1",
            "action_name": "tulpa_run_terminal",
            "action_args": {"command": "python3 tulpa_stuff/tg_login.py", "working_dir": "tulpa_stuff"},
        },
        chat_id=1,
    )
    assert "succeeded automatically on retry" in out
    assert len(runtime.execute_calls) == 2
    assert runtime.execute_calls[0]["action_name"] == "guardrail_execute_approved_action"
    assert runtime.execute_calls[1]["action_name"] == "guardrail_execute_approved_action"
    assert runtime.ainvoke_calls == []
    assert any(str(evt.get("event_type")) == "executed_retry_success" for evt in events.events)


@pytest.mark.asyncio
async def test_orchestrator_uses_isolated_recovery_thread_for_llm_repair() -> None:
    runtime = _RecoveryThreadRuntime()
    events = _FakeContextEvents()
    orchestrator = ApprovalExecutionOrchestrator(
        get_agent_runtime=lambda: runtime,
        get_context_events=lambda: events,
    )
    out = await orchestrator.execute_approved_action_and_summarize(
        approval_id="apr_y",
        decision_payload={
            "customer_id": "telegram_2",
            "thread_id": "chat_main",
            "action_name": "tulpa_run_terminal",
            "action_args": {"command": "python3 tulpa_stuff/tg_login.py", "working_dir": "tulpa_stuff"},
        },
        chat_id=2,
    )
    assert "retried and fixed" in out.lower()
    assert len(runtime.ainvoke_calls) >= 1
    assert str(runtime.ainvoke_calls[0]["thread_id"]).startswith("chat_main::approval-recovery::apr_y")
    assert runtime.ainvoke_calls[0]["turn_mode"] == "approval_recovery"
