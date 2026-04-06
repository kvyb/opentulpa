from __future__ import annotations

from typing import Any

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | list[Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else str(payload)
        self.content = b"" if payload is None else b"x"

    def json(self) -> dict[str, Any] | list[Any]:
        return self._payload if self._payload is not None else {}


class _DummyRuntime:
    def __init__(
        self,
        responses: list[_Response],
        *,
        guard_result: dict[str, Any] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._guard_result = guard_result or {"gate": "allow", "reason": "ok", "summary": "execute"}
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.guard_calls: list[dict[str, Any]] = []
        self._active_customer_id = "telegram_1"

    async def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> _Response:
        self.calls.append((method, path, kwargs))
        if not self._responses:
            raise RuntimeError("unexpected internal API call")
        return self._responses.pop(0)

    async def evaluate_tool_guardrail(
        self,
        *,
        customer_id: str,
        thread_id: str,
        action_name: str,
        action_args: dict[str, Any],
        action_note: str | None = None,
    ) -> dict[str, Any]:
        self.guard_calls.append(
            {
                "customer_id": customer_id,
                "thread_id": thread_id,
                "action_name": action_name,
                "action_args": action_args,
                "action_note": action_note,
            }
        )
        return dict(self._guard_result)


@pytest.mark.asyncio
async def test_terminal_interactive_requires_approval_before_execution() -> None:
    runtime = _DummyRuntime(
        [],
        guard_result={
            "gate": "require_approval",
            "approval_id": "apr_boundary_1",
            "summary": "run terminal command",
            "reason": "external write side effect",
        },
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 tulpa_stuff/scripts/post_update.py",
            "working_dir": "tulpa_stuff",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["status"] == "approval_pending"
    assert result["approval_id"] == "apr_boundary_1"
    assert len(runtime.guard_calls) == 1
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_terminal_require_approval_without_approval_id_returns_guardrail_unavailable() -> None:
    runtime = _DummyRuntime(
        [],
        guard_result={
            "gate": "require_approval",
            "approval_id": None,
            "summary": "run terminal command",
            "reason": "guardrail_request_error:ReadTimeout",
        },
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 tg_scan_work.py",
            "working_dir": "tulpa_stuff",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["status"] == "guardrail_unavailable"
    assert result["approval_id"] is None
    assert result["gate"] == "require_approval"
    assert result["retryable"] is True
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_terminal_scheduled_execution_skips_guardrail() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "stdout": "done", "stderr": "", "returncode": 0})],
        guard_result={"gate": "require_approval", "approval_id": "apr_should_not_happen"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 tulpa_stuff/scripts/digest.py",
            "working_dir": "tulpa_stuff",
            "thread_id": "wake_abc123",
            "execution_origin": "scheduled",
        }
    )
    assert result["ok"] is True
    assert result["execution_origin"] == "scheduled"
    assert len(runtime.guard_calls) == 0
    assert len(runtime.calls) == 1
    assert runtime.calls[0][1] == "/internal/tulpa/run_terminal"


@pytest.mark.asyncio
async def test_terminal_routine_thread_prefix_is_treated_as_scheduled() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "stdout": "done", "stderr": "", "returncode": 0})],
        guard_result={"gate": "require_approval", "approval_id": "apr_should_not_happen"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 scripts/digest.py",
            "working_dir": "tulpa_stuff",
            "thread_id": "routine_rtn_abc123",
        }
    )
    assert result["ok"] is True
    assert result["execution_origin"] == "scheduled"
    assert len(runtime.guard_calls) == 0
    assert len(runtime.calls) == 1
    assert runtime.calls[0][1] == "/internal/tulpa/run_terminal"


@pytest.mark.asyncio
async def test_terminal_normalizes_redundant_working_dir_prefix_before_guard_and_execution() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "stdout": "done", "stderr": "", "returncode": 0})],
        guard_result={"gate": "allow", "reason": "ok", "summary": "execute"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["tulpa_run_terminal"].ainvoke(
        {
            "command": "python3 tulpa_stuff/tg_login.py",
            "working_dir": "tulpa_stuff",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["ok"] is True
    assert len(runtime.guard_calls) == 1
    assert runtime.guard_calls[0]["action_args"]["command"] == "python3 tg_login.py"
    assert len(runtime.calls) == 1
    sent = runtime.calls[0][2]["json_body"]
    assert sent["command"] == "python3 tg_login.py"
    assert sent["working_dir"] == "tulpa_stuff"


@pytest.mark.asyncio
async def test_routine_create_saves_schedule_without_execution_artifact_metadata() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "id": "rtn_1"})],
        guard_result={"gate": "allow", "reason": "internal plan", "summary": "create routine"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Daily Digest",
            "schedule": "0 9 * * *",
            "instruction": (
                "You must run scripts/digest.py for the daily digest. First read market inputs "
                "using file tulpa_stuff/input.json and API key source NEWS_API_KEY from env. "
                "Then append concise bullets to tulpa_stuff/digest.md. "
                "If the API fails or the file is missing, log error and return failure summary."
            ),
            "implementation_command": "python3 tulpa_stuff/scripts/digest.py",
            "implementation_working_dir": "tulpa_stuff",
            "implementation_timeout_seconds": 120,
            "notify_user": True,
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["ok"] is True
    assert result["id"] == "rtn_1"
    assert len(runtime.guard_calls) == 1
    assert len(runtime.calls) == 1
    sent = runtime.calls[0][2]["json_body"]
    assert "execution" not in sent["payload"]
    assert sent["payload"]["instruction"].startswith("You must run scripts/digest.py")
    assert "message" not in sent["payload"]


@pytest.mark.asyncio
async def test_routine_create_normalizes_implementation_command_prefix_for_guardrail() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "id": "rtn_3"})],
        guard_result={"gate": "allow", "reason": "ok", "summary": "create routine"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Login Refresh",
            "schedule": "0 */6 * * *",
            "instruction": "You must run scripts/tg_login.py and report result.",
            "implementation_command": "python3 tulpa_stuff/tg_login.py",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["ok"] is True
    assert len(runtime.guard_calls) == 1
    assert runtime.guard_calls[0]["action_args"]["implementation_command"] == "python3 tg_login.py"


@pytest.mark.asyncio
async def test_routine_create_accepts_instruction_without_legacy_message() -> None:
    runtime = _DummyRuntime(
        [_Response(200, {"ok": True, "id": "rtn_2"})],
        guard_result={"gate": "allow", "reason": "internal plan", "summary": "create routine"},
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Silent Timelog",
            "schedule": "0 */3 * * *",
            "instruction": (
                "You must run scripts/logtime.py to keep timelog fresh. "
                "First read existing file tulpa_stuff/logtimes.md if present. "
                "Then append ISO-8601 UTC timestamp to tulpa_stuff/logtimes.md. "
                "If file access fails, log error and return failure summary."
            ),
            "implementation_command": "python3 scripts/logtime.py",
            "notify_user": False,
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["ok"] is True
    assert len(runtime.calls) == 1
    sent = runtime.calls[0][2]["json_body"]
    assert sent["payload"]["instruction"].startswith("You must run scripts/logtime.py")
    assert "message" not in sent["payload"]


@pytest.mark.asyncio
async def test_routine_create_requires_non_empty_instruction() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "id": "rtn_unexpected"})])
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Daily Digest",
            "schedule": "0 9 * * *",
            "instruction": "   ",
            "implementation_command": "python3 tulpa_stuff/scripts/digest.py",
            "notify_user": True,
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert str(result.get("error", "")).startswith("routine_create failed: instruction is required")
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_routine_create_pending_approval_does_not_save_schedule() -> None:
    runtime = _DummyRuntime(
        [],
        guard_result={
            "gate": "require_approval",
            "approval_id": "apr_routine_1",
            "summary": "create external write routine",
            "reason": "external write side effect",
        },
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Auto Post",
            "schedule": "0 */2 * * *",
            "instruction": (
                "You must run scripts/post_agentx.py for recurring market post. "
                "First read source file tulpa_stuff/post_context.md and API key source POST_API_KEY from env. "
                "Then post summary to https://mockapi.io/api/v1/posts. "
                "If request fails or key is missing, log error and return failure summary."
            ),
            "implementation_command": "python3 tulpa_stuff/scripts/post_agentx.py",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["status"] == "approval_pending"
    assert result["approval_id"] == "apr_routine_1"
    assert len(runtime.guard_calls) == 1
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_routine_create_require_approval_without_approval_id_returns_guardrail_unavailable() -> None:
    runtime = _DummyRuntime(
        [],
        guard_result={
            "gate": "require_approval",
            "approval_id": "None",
            "summary": "create external write routine",
            "reason": "guardrail_request_error:ReadTimeout",
        },
    )
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Auto Post",
            "schedule": "0 */2 * * *",
            "instruction": (
                "You must run scripts/post_agentx.py for recurring market post. "
                "Then post summary to https://mockapi.io/api/v1/posts."
            ),
            "implementation_command": "python3 scripts/post_agentx.py",
            "thread_id": "chat-1",
            "execution_origin": "interactive",
        }
    )
    assert result["status"] == "guardrail_unavailable"
    assert result["approval_id"] is None
    assert result["gate"] == "require_approval"
    assert result["retryable"] is True
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_routine_create_requires_non_empty_implementation_command() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "id": "rtn_unexpected"})])
    tools = register_runtime_tools(runtime)
    result = await tools["routine_create"].ainvoke(
        {
            "name": "Auto Post",
            "schedule": "0 */2 * * *",
            "instruction": (
                "You must run scripts/post_agentx.py for recurring market post. "
                "First read source file tulpa_stuff/post_context.md and API key source POST_API_KEY from env. "
                "Then post summary to https://mockapi.io/api/v1/posts. "
                "If request fails or key is missing, log error and return failure summary."
            ),
            "implementation_command": "   ",
        }
    )
    assert str(result.get("error", "")).startswith("ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED")
    assert runtime.calls == []
    assert runtime.guard_calls == []
@pytest.mark.asyncio
async def test_composio_tools_call_internal_routes_with_customer_scope() -> None:
    runtime = _DummyRuntime(
        [
            _Response(
                200,
                {
                    "ok": True,
                    "enabled": True,
                    "callback_url_configured": True,
                    "default_callback_url": "https://example.com/callback",
                },
            ),
            _Response(
                200,
                {
                    "ok": True,
                    "customer_id": "telegram_1",
                    "toolkit": "instagram",
                    "connection_id": "conn_1",
                    "redirect_url": "https://connect.example.com/instagram",
                    "message_for_user": "Connect your instagram account here: https://connect.example.com/instagram",
                },
            ),
            _Response(200, {"ok": True, "connection": {"id": "conn_1", "status": "ACTIVE"}}),
            _Response(200, {"ok": True, "connected_account": {"id": "acct_1", "disabled": True}}),
            _Response(200, {"ok": True, "connected_account": {"id": "acct_2", "deleted": True}}),
            _Response(
                200,
                {
                    "ok": True,
                    "matched": True,
                    "conversation_id": "conv_1",
                    "recipient_id": "rcp_1",
                    "recipient_id_verified": True,
                    "latest_inbound_message_created_time": "2026-04-06T11:14:00+0000",
                },
            ),
            _Response(
                200,
                {
                    "ok": True,
                    "tool_slug": "INSTAGRAM_LIST_ALL_MESSAGES",
                    "successful": True,
                    "data": {"items": []},
                },
            ),
        ]
    )
    tools = register_runtime_tools(runtime)

    status = await tools["composio_status"].ainvoke({})
    auth = await tools["composio_authorize_toolkit"].ainvoke({"toolkit": "instagram"})
    connected = await tools["composio_wait_for_connection"].ainvoke({"connection_id": "conn_1"})
    disabled = await tools["composio_disable_connected_account"].ainvoke({"connected_account_id": "acct_1"})
    deleted = await tools["composio_delete_connected_account"].ainvoke({"connected_account_id": "acct_2"})
    precheck = await tools["composio_instagram_reply_precheck"].ainvoke(
        {
            "recipient_id": "rcp_1",
            "connected_account_id": "acct_1",
        }
    )
    executed = await tools["composio_tool_execute"].ainvoke(
        {
            "tool_slug": "INSTAGRAM_LIST_ALL_MESSAGES",
            "arguments": {"conversation_id": "conv_1"},
            "connected_account_id": "acct_1",
            "text": "list messages",
        }
    )

    assert status["enabled"] is True
    assert auth["redirect_url"] == "https://connect.example.com/instagram"
    assert "Open this authorization link" in auth["message"]
    assert connected["status"] == "ACTIVE"
    assert disabled["disabled"] is True
    assert deleted["deleted"] is True
    assert precheck["recipient_id_verified"] is True
    assert executed["successful"] is True
    assert runtime.calls[0][1] == "/internal/composio/status"
    assert runtime.calls[1][1] == "/internal/composio/authorize"
    assert runtime.calls[1][2]["json_body"]["customer_id"] == "telegram_1"
    assert runtime.calls[2][1] == "/internal/composio/wait_for_connection"
    assert runtime.calls[3][1] == "/internal/composio/connected_accounts/disable"
    assert runtime.calls[4][1] == "/internal/composio/connected_accounts/delete"
    assert runtime.calls[5][1] == "/internal/composio/instagram/reply_precheck"
    assert runtime.calls[5][2]["json_body"]["customer_id"] == "telegram_1"
    assert runtime.calls[6][1] == "/internal/composio/tools/execute"
    assert runtime.calls[6][2]["json_body"]["customer_id"] == "telegram_1"
