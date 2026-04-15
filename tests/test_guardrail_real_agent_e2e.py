from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from opentulpa.agent import runtime as runtime_module
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.scheduler.service import SchedulerService

TEST_CUSTOMER_ID = "cust_e2e_guardrail"
E2E_LOG_DIR = Path(".opentulpa/logs/e2e").resolve()


class _JsonlRecorder:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def add(self, kind: str, **payload: Any) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": str(kind or "").strip(),
            **payload,
        }
        self.entries.append(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def count_kind(self, kind: str) -> int:
        return len([item for item in self.entries if item.get("kind") == kind])

    def kind_slice(self, kind: str, start: int) -> list[dict[str, Any]]:
        items = [item for item in self.entries if item.get("kind") == kind]
        return items[start:]


class _FakeTelegramClient:
    def __init__(self, _token: str) -> None:
        self.callback_answers: list[dict[str, Any]] = []
        self.sent_messages: list[dict[str, Any]] = []
        self.edited_messages: list[dict[str, Any]] = []

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> bool:
        self.callback_answers.append(
            {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": bool(show_alert),
            }
        )
        return True

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.sent_messages.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup or {},
            }
        )
        return {"ok": True}

    async def edit_message_text(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        self.edited_messages.append(
            {
                "chat_id": chat_id,
                "message_id": int(message_id),
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup or {},
            }
        )
        return True

    async def edit_message_reply_markup(
        self,
        *,
        chat_id: int | str,
        message_id: int,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        self.edited_messages.append(
            {
                "chat_id": chat_id,
                "message_id": int(message_id),
                "text": "",
                "parse_mode": None,
                "reply_markup": reply_markup or {},
            }
        )
        return True

    async def send_chat_action(self, *, chat_id: int | str, action: str = "typing") -> bool:
        _ = (chat_id, action)
        return True


class _ScriptedAgentModel:
    def __init__(self, *, recorder: _JsonlRecorder) -> None:
        self._recorder = recorder
        self._bound_tools: list[str] = []

    def bind_tools(self, tools: list[Any]) -> _ScriptedAgentModel:
        names: list[str] = []
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            if name:
                names.append(name)
        self._bound_tools = names
        self._recorder.add("model_bind_tools", tools=names)
        return self

    @staticmethod
    def _msg_text(message: Any) -> str:
        return str(getattr(message, "content", "") or "").strip()

    @staticmethod
    def _last_text(messages: list[Any], cls: type[Any]) -> str:
        for item in reversed(messages):
            if isinstance(item, cls):
                return str(getattr(item, "content", "") or "").strip()
        return ""

    @staticmethod
    def _extract_approval_id(text: str) -> str:
        match = re.search(r"\bapr_[a-z0-9_-]{6,40}\b", str(text or ""), flags=re.IGNORECASE)
        return str(match.group(0)).strip() if match else ""

    @staticmethod
    def _extract_action_context(human_text: str) -> tuple[str, dict[str, Any]]:
        raw = str(human_text or "")
        match = re.search(
            r"action_name=(?P<name>[^\n]+)\naction_args=(?P<args>\{.*?\})(?:\naction_note=|$)",
            raw,
            flags=re.DOTALL,
        )
        if not match:
            return "", {}
        action_name = str(match.group("name") or "").strip()
        args_text = str(match.group("args") or "").strip()
        try:
            parsed = json.loads(args_text)
        except Exception:
            parsed = {}
        return action_name, parsed if isinstance(parsed, dict) else {}

    def _guardrail_classifier_reply(self, human_text: str) -> str:
        action_name, action_args = self._extract_action_context(human_text)
        command = ""
        if action_name == "tulpa_run_terminal":
            command = str(action_args.get("command", "")).strip()
        elif action_name == "routine_create":
            command = str(action_args.get("implementation_command", "")).strip()
        cmd = command.lower()
        is_mockapi = "mockapi.io" in cmd
        is_external_write = any(
            token in cmd
            for token in (
                "-x post",
                "--request post",
                " -d ",
                "--data",
                "-x put",
                "--request put",
                "-x patch",
                "--request patch",
                "-x delete",
                "--request delete",
            )
        )
        if is_mockapi and is_external_write:
            payload = {
                "ok": True,
                "gate": "require_approval",
                "impact_type": "write",
                "recipient_scope": "external",
                "confidence": 0.97,
                "reason": "external write side effect",
            }
        else:
            payload = {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "self",
                "confidence": 0.8,
                "reason": "no external write",
            }
        return json.dumps(payload, ensure_ascii=False)

    def _claim_check_reply(self) -> str:
        return json.dumps(
            {
                "ok": True,
                "applies": False,
                "mismatch": False,
                "confidence": 0.95,
                "reason": "no_immediate_claim",
                "repair_instruction": "",
            },
            ensure_ascii=False,
        )

    def _wake_classifier_reply(self) -> str:
        return json.dumps({"notify_user": True, "reason": "deliver update"}, ensure_ascii=False)

    def _agent_reply(self, messages: list[Any], *, user_text: str, tool_text: str) -> AIMessage:
        lower_user = str(user_text or "").lower()
        lower_tool = str(tool_text or "").lower()
        tool_payload: dict[str, Any] | None = None
        if tool_text:
            try:
                parsed = json.loads(tool_text)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                tool_payload = parsed
        last_turn_message: Any | None = None
        for item in reversed(messages):
            if isinstance(item, (HumanMessage, ToolMessage, AIMessage)):
                last_turn_message = item
                break

        if isinstance(last_turn_message, ToolMessage) and tool_text:
            approval_id = self._extract_approval_id(tool_text)
            if "approval_pending" in lower_tool or "approval pending" in lower_tool:
                safe_id = approval_id or "unknown"
                return AIMessage(
                    content=(
                        "Approval required before external write. "
                        f"approval_id={safe_id}"
                    )
                )
            routine_id = ""
            if isinstance(tool_payload, dict):
                routine_id = str(tool_payload.get("id", "")).strip()
            if not routine_id:
                routine_match = re.search(r'"id"\s*:\s*"(rtn_[^"]+)"', tool_text)
                routine_id = routine_match.group(1) if routine_match else ""
            if routine_id.startswith("rtn_"):
                return AIMessage(content=f"Routine created successfully. routine_id={routine_id}")
            if '"status": "executed"' in lower_tool and "routine_create" in lower_tool:
                routine_match = re.search(r'"id"\s*:\s*"(rtn_[^"]+)"', tool_text)
                routine_id = routine_match.group(1) if routine_match else "rtn_unknown"
                return AIMessage(content=f"Routine created successfully. routine_id={routine_id}")
            if (
                (isinstance(tool_payload, dict) and str(tool_payload.get("execution_origin", "")).strip() == "scheduled")
                or '"execution_origin": "scheduled"' in lower_tool
            ):
                return AIMessage(content="Scheduled run executed without a new approval prompt.")
            if (isinstance(tool_payload, dict) and "returncode" in tool_payload) or '"returncode"' in lower_tool:
                if "external read" in lower_user:
                    return AIMessage(content="External read action completed with no approval prompt.")
                if "scheduled wake read" in lower_user:
                    return AIMessage(content="Scheduled read run completed without requesting approval.")
                return AIMessage(content="Scheduled run completed without requesting approval.")
            if (isinstance(tool_payload, dict) and "error" in tool_payload) or '"error"' in lower_tool:
                return AIMessage(content="Run finished in scheduled mode and did not request approval.")

        if "previously approved action has just been executed" in lower_user:
            return AIMessage(content="Approved action executed successfully.")
        if "previously approved action execution failed" in lower_user:
            return AIMessage(content="Approved action failed and needs a repair step.")

        if "run external write now" in lower_user:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_immediate_write_1",
                        "name": "tulpa_run_terminal",
                        "args": {
                            "command": (
                                "curl -X POST https://mockapi.io/api/v1/posts "
                                "-H 'Content-Type: application/json' "
                                "-d '{\"source\":\"e2e\",\"kind\":\"immediate\"}'"
                            ),
                            "working_dir": "tulpa_stuff",
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "run external read now" in lower_user:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_immediate_read_1",
                        "name": "tulpa_run_terminal",
                        "args": {
                            "command": (
                                "python3 -c \"print('read from https://mockapi.io/api/v1/posts')\""
                            ),
                            "working_dir": "tulpa_stuff",
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "create recurring schedule" in lower_user:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_schedule_create_1",
                        "name": "routine_create",
                        "args": {
                            "name": "Market Writer",
                            "schedule": "0 * * * *",
                            "instruction": (
                                "You must post market reflection update. First read required context from "
                                "tulpa_stuff/post_context.md and API key source from environment. "
                                "Then call the configured endpoint to publish update. "
                                "If request fails, log error and return failure summary."
                            ),
                            "implementation_command": (
                                "curl -X POST https://mockapi.io/api/v1/posts "
                                "-H 'Content-Type: application/json' "
                                "-d '{\"source\":\"e2e\",\"kind\":\"schedule\"}'"
                            ),
                            "notify_user": True,
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "create recurring read schedule" in lower_user:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_schedule_read_create_1",
                        "name": "routine_create",
                        "args": {
                            "name": "Read Poller",
                            "schedule": "0 * * * *",
                            "instruction": (
                                "You must read latest updates from API. First read endpoint config file "
                                "tulpa_stuff/api_source.txt and API key source from environment. "
                                "Then fetch data and write summary to tulpa_stuff/read_poller.log. "
                                "If fetch fails, log error and return failure summary."
                            ),
                            "implementation_command": (
                                "python3 -c \"print('scheduled read from "
                                "https://mockapi.io/api/v1/posts')\""
                            ),
                            "notify_user": True,
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "approve schedule" in lower_user:
            approval_id = self._extract_approval_id(user_text) or "apr_missing"
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_schedule_approve_1",
                        "name": "guardrail_execute_approved_action",
                        "args": {
                            "approval_id": approval_id,
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "simulate scheduled wake read run now" in lower_user:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_wake_read_run_1",
                        "name": "tulpa_run_terminal",
                        "args": {
                            "command": (
                                "python3 -c \"print('scheduled wake read "
                                "https://mockapi.io/api/v1/posts')\""
                            ),
                            "working_dir": "tulpa_stuff",
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "simulate scheduled wake run now" in lower_user:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_wake_run_1",
                        "name": "tulpa_run_terminal",
                        "args": {
                            "command": (
                                "python3 -c \"print('scheduled write simulation "
                                "https://mockapi.io/api/v1/posts')\""
                            ),
                            "working_dir": "tulpa_stuff",
                        },
                        "type": "tool_call",
                    }
                ],
            )
        if "denied your planned action" in lower_user:
            return AIMessage(
                content=(
                    "Understood. You denied that action. Tell me what to change, "
                    "and I will revise and resubmit for approval."
                )
            )
        return AIMessage(content="Ready for the next step.")

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        first_system = self._msg_text(messages[0]) if messages and isinstance(messages[0], SystemMessage) else ""
        user_text = self._last_text(messages, HumanMessage)
        tool_text = self._last_text(messages, ToolMessage)
        self._recorder.add(
            "model_input",
            first_system=first_system[:500],
            user_text=user_text[:1200],
            tool_text=tool_text[:1200],
        )
        if "Classify action safety intent for an approval gate." in first_system:
            payload = self._guardrail_classifier_reply(user_text)
            self._recorder.add("model_output", mode="guardrail_classifier", content=payload)
            return AIMessage(content=payload)
        if "You verify assistant execution claims against tool evidence." in first_system:
            payload = self._claim_check_reply()
            self._recorder.add("model_output", mode="claim_check_classifier", content=payload)
            return AIMessage(content=payload)
        if "You classify background assistant events." in first_system:
            payload = self._wake_classifier_reply()
            self._recorder.add("model_output", mode="wake_classifier", content=payload)
            return AIMessage(content=payload)
        reply = self._agent_reply(messages, user_text=user_text, tool_text=tool_text)
        self._recorder.add(
            "model_output",
            mode="agent",
            content=str(reply.content or "")[:1200],
            tool_calls=getattr(reply, "tool_calls", []),
        )
        return reply


@dataclass
class _Harness:
    client: TestClient
    runtime: OpenTulpaLangGraphRuntime
    recorder: _JsonlRecorder
    log_path: Path
    telegram_client: _FakeTelegramClient | None


def _extract_approval_id(text: str) -> str:
    match = re.search(r"\bapr_[a-z0-9_-]{6,40}\b", str(text or ""), flags=re.IGNORECASE)
    return str(match.group(0)).strip() if match else ""


def _patch_runtime_internal_api(
    *,
    runtime: OpenTulpaLangGraphRuntime,
    app: Any,
    recorder: _JsonlRecorder,
) -> None:
    async def _request_with_backoff(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> httpx.Response:
        attempts = max(0, int(retries)) + 1
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.request(
                        method=method,
                        url=path,
                        params=params,
                        json=json_body,
                        timeout=timeout,
                    )
                recorder.add(
                    "internal_api_call",
                    method=str(method).upper(),
                    path=path,
                    params=params or {},
                    json_body=json_body or {},
                    status_code=int(response.status_code),
                    response_text=str(response.text or "")[:2000],
                )
                return response
            except Exception as exc:  # pragma: no cover - defensive retry path
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(0.05 * (attempt + 1))
        raise RuntimeError(f"internal request failed: {last_exc}")

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]


def _build_harness(
    *,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    log_name: str,
    enable_telegram: bool,
) -> _Harness:
    log_path = E2E_LOG_DIR / f"{log_name}.jsonl"
    recorder = _JsonlRecorder(log_path)
    model = _ScriptedAgentModel(recorder=recorder)
    monkeypatch.setattr(runtime_module, "init_chat_model", lambda *args, **kwargs: model)
    monkeypatch.setenv("APPROVALS_DB_PATH", str(tmp_path / f"{log_name}_approvals.db"))
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / f"{log_name}_links.db"))
    if enable_telegram:
        fake_tg = _FakeTelegramClient("token")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
        monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_tg)
    else:
        fake_tg = None
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_WEBHOOK_SECRET", raising=False)
    get_settings.cache_clear()

    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://testserver",
        openrouter_api_key="test-key",
        model_name="fake/test-model",
        wake_classifier_model_name="fake/test-model",
        guardrail_classifier_model_name="fake/test-model",
        checkpoint_db_path=str(tmp_path / f"{log_name}_checkpoints.sqlite"),
        behavior_log_path=str(tmp_path / f"{log_name}_behavior.jsonl"),
        behavior_log_enabled=True,
    )
    scheduler = SchedulerService(db_path=tmp_path / f"{log_name}_scheduler.db")
    app = create_app(agent_runtime=runtime, scheduler=scheduler)
    _patch_runtime_internal_api(runtime=runtime, app=app, recorder=recorder)
    client = TestClient(app)
    client.__enter__()
    return _Harness(
        client=client,
        runtime=runtime,
        recorder=recorder,
        log_path=log_path,
        telegram_client=fake_tg,
    )


@pytest.fixture()
def real_agent_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Harness:
    harness = _build_harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        log_name="guardrail_real_agent",
        enable_telegram=False,
    )
    try:
        yield harness
    finally:
        harness.client.__exit__(None, None, None)
        get_settings.cache_clear()


@pytest.fixture()
def real_agent_telegram_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Harness:
    harness = _build_harness(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        log_name="guardrail_telegram_deny",
        enable_telegram=True,
    )
    try:
        yield harness
    finally:
        harness.client.__exit__(None, None, None)
        get_settings.cache_clear()


def _chat_turn(*, harness: _Harness, thread_id: str, text: str) -> dict[str, Any]:
    harness.recorder.add("user_turn", thread_id=thread_id, text=text)
    response = harness.client.post(
        "/internal/chat",
        json={
            "customer_id": TEST_CUSTOMER_ID,
            "thread_id": thread_id,
            "text": text,
        },
    )
    payload = response.json()
    harness.recorder.add(
        "agent_turn",
        thread_id=thread_id,
        status_code=int(response.status_code),
        payload=payload,
    )
    assert response.status_code == 200
    assert payload.get("ok") is True
    return payload


def _wait_until(predicate: Any, timeout_seconds: float = 3.0) -> bool:
    deadline = time.time() + max(0.1, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.05)
    return bool(predicate())


def _post_telegram_approval_callback(
    *,
    harness: _Harness,
    approval_id: str,
    decision: str,
    callback_id: str,
    chat_id: int = 777,
    user_id: int = 999,
    message_id: int = 55,
) -> None:
    response = harness.client.post(
        "/webhook/telegram",
        headers={"x-telegram-bot-api-secret-token": "test-secret"},
        json={
            "callback_query": {
                "id": callback_id,
                "from": {"id": int(user_id)},
                "message": {"message_id": int(message_id), "chat": {"id": int(chat_id)}},
                "data": f"approval:{approval_id}:{decision}",
            }
        },
    )
    assert response.status_code == 200


def _post_telegram_text(
    *,
    harness: _Harness,
    text: str,
    chat_id: int = 777,
    user_id: int = 999,
    message_id: int = 77,
) -> None:
    response = harness.client.post(
        "/webhook/telegram",
        headers={"x-telegram-bot-api-secret-token": "test-secret"},
        json={
            "message": {
                "message_id": int(message_id),
                "chat": {"id": int(chat_id)},
                "from": {"id": int(user_id)},
                "text": str(text),
            }
        },
    )
    assert response.status_code == 200


def _seed_telegram_session(*, customer_id: str, thread_id: str, chat_id: int = 777, user_id: int = 999) -> None:
    from opentulpa.interfaces.telegram.chat_service import STATE_STORE

    now_utc = datetime.now(timezone.utc).isoformat()

    def _mutate(state: dict[str, Any]) -> None:
        sessions = state.get("sessions")
        if not isinstance(sessions, dict):
            sessions = {}
        sessions[str(chat_id)] = {
            "user_id": int(user_id),
            "customer_id": customer_id,
            "thread_id": thread_id,
            "wake_thread_id": "wake_seeded_guardrail",
            "last_user_message_at": now_utc,
            "last_assistant_message_at": now_utc,
        }
        state["sessions"] = sessions

    STATE_STORE.update(_mutate)


def test_e2e_immediate_external_write_requires_approval(real_agent_harness: _Harness) -> None:
    start = real_agent_harness.recorder.count_kind("internal_api_call")
    payload = _chat_turn(
        harness=real_agent_harness,
        thread_id="chat-guardrail-e2e-1",
        text=(
            "Run external write now: post a test payload to "
            "https://mockapi.io/api/v1/posts"
        ),
    )
    approval_id = _extract_approval_id(str(payload.get("text", "")))
    assert approval_id
    internal_calls = real_agent_harness.recorder.kind_slice("internal_api_call", start)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in internal_calls)
    assert not any(item.get("path") == "/internal/tulpa/run_terminal" for item in internal_calls)
    assert real_agent_harness.log_path.exists()
    assert real_agent_harness.log_path.read_text(encoding="utf-8").strip()


def test_e2e_immediate_external_read_runs_without_approval(real_agent_harness: _Harness) -> None:
    start = real_agent_harness.recorder.count_kind("internal_api_call")
    payload = _chat_turn(
        harness=real_agent_harness,
        thread_id="chat-guardrail-e2e-read-1",
        text="Run external read now from https://mockapi.io/api/v1/posts",
    )
    lower_text = str(payload.get("text", "")).lower()
    assert "approval_id=" not in lower_text
    internal_calls = real_agent_harness.recorder.kind_slice("internal_api_call", start)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in internal_calls)
    assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in internal_calls)


def test_e2e_immediate_external_write_approve_executes(real_agent_telegram_harness: _Harness) -> None:
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-guardrail-write-approve-1",
        chat_id=777,
        user_id=999,
    )
    start = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    first = _chat_turn(
        harness=real_agent_telegram_harness,
        thread_id="chat-guardrail-write-approve-1",
        text="Run external write now: post a test payload to https://mockapi.io/api/v1/posts",
    )
    approval_id = _extract_approval_id(str(first.get("text", "")))
    assert approval_id
    _post_telegram_approval_callback(
        harness=real_agent_telegram_harness,
        approval_id=approval_id,
        decision="approve",
        callback_id="cbq_write_approve_1",
    )
    tg_client = real_agent_telegram_harness.telegram_client
    assert tg_client is not None
    assert _wait_until(
        lambda: any(
            "approved action executed successfully" in str(item.get("text", "")).lower()
            for item in tg_client.sent_messages
        ),
        timeout_seconds=3.0,
    )
    assert any(
        str(item.get("text", "")).strip() == "Working on the task."
        for item in tg_client.callback_answers
    )
    new_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start)
    assert any(item.get("path") == "/internal/approvals/execute" for item in new_calls)
    assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in new_calls)


def test_e2e_plaintext_sure_does_not_decide_pending_approval(
    real_agent_telegram_harness: _Harness,
) -> None:
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-guardrail-write-sure-approve-1",
        chat_id=777,
        user_id=999,
    )
    start = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    first = _chat_turn(
        harness=real_agent_telegram_harness,
        thread_id="chat-guardrail-write-sure-approve-1",
        text="Run external write now: post a test payload to https://mockapi.io/api/v1/posts",
    )
    approval_id = _extract_approval_id(str(first.get("text", "")))
    assert approval_id

    _post_telegram_text(
        harness=real_agent_telegram_harness,
        text="sure",
        chat_id=777,
        user_id=999,
        message_id=88,
    )

    tg_client = real_agent_telegram_harness.telegram_client
    assert tg_client is not None
    assert not _wait_until(
        lambda: any(
            "approved action executed successfully" in str(item.get("text", "")).lower()
            for item in tg_client.sent_messages
        ),
        timeout_seconds=1.2,
    )
    new_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start)
    assert not any(item.get("path") == "/internal/approvals/execute" for item in new_calls)


def test_e2e_schedule_creation_then_wake_run_skips_guard(
    real_agent_telegram_harness: _Harness,
) -> None:
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-guardrail-e2e-2",
        chat_id=777,
        user_id=999,
    )

    start_create = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    create_payload = _chat_turn(
        harness=real_agent_telegram_harness,
        thread_id="chat-guardrail-e2e-2",
        text=(
            "Create recurring schedule to publish updates to "
            "https://mockapi.io/api/v1/posts every hour."
        ),
    )
    approval_id = _extract_approval_id(str(create_payload.get("text", "")))
    assert approval_id
    create_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start_create)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in create_calls)
    assert not any(item.get("path") == "/internal/scheduler/routine" for item in create_calls)

    start_approve = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    _post_telegram_approval_callback(
        harness=real_agent_telegram_harness,
        approval_id=approval_id,
        decision="approve",
        callback_id="cbq_schedule_approve_1",
    )
    tg_client = real_agent_telegram_harness.telegram_client
    assert tg_client is not None
    assert _wait_until(
        lambda: any(
            "approved action executed successfully" in str(item.get("text", "")).lower()
            for item in tg_client.sent_messages
        ),
        timeout_seconds=3.0,
    )
    approve_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start_approve)
    assert any(item.get("path") == "/internal/approvals/execute" for item in approve_calls)
    assert any(item.get("path") == "/internal/scheduler/routine" for item in approve_calls)

    def _has_market_writer() -> bool:
        routines = real_agent_telegram_harness.client.get(
            "/internal/scheduler/routines",
            params={"customer_id": TEST_CUSTOMER_ID},
        )
        if routines.status_code != 200:
            return False
        routines_payload = routines.json()
        return any(
            str(item.get("name", "")).strip() == "Market Writer"
            for item in routines_payload.get("routines", [])
        )

    assert _wait_until(_has_market_writer, timeout_seconds=3.0)
    routines = real_agent_telegram_harness.client.get(
        "/internal/scheduler/routines", params={"customer_id": TEST_CUSTOMER_ID}
    )
    assert routines.status_code == 200
    routines_payload = routines.json()
    assert any(
        str(item.get("name", "")).strip() == "Market Writer"
        for item in routines_payload.get("routines", [])
    )

    start_wake = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    wake_payload = _chat_turn(
        harness=real_agent_telegram_harness,
        thread_id="wake_guardrail_e2e_2",
        text="Simulate scheduled wake run now.",
    )
    wake_text = str(wake_payload.get("text", "")).lower()
    assert "approval_id=" not in wake_text
    wake_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start_wake)
    assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in wake_calls)
    assert not any(item.get("path") == "/internal/approvals/evaluate" for item in wake_calls)


def test_e2e_schedule_read_create_and_wake_run_without_approval(
    real_agent_harness: _Harness,
) -> None:
    start_create = real_agent_harness.recorder.count_kind("internal_api_call")
    create_payload = _chat_turn(
        harness=real_agent_harness,
        thread_id="chat-guardrail-read-schedule-1",
        text="Create recurring read schedule from https://mockapi.io/api/v1/posts every hour.",
    )
    create_text = str(create_payload.get("text", "")).lower()
    assert "routine created successfully" in create_text
    assert "approval_id=" not in create_text
    create_calls = real_agent_harness.recorder.kind_slice("internal_api_call", start_create)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in create_calls)
    assert any(item.get("path") == "/internal/scheduler/routine" for item in create_calls)

    start_wake = real_agent_harness.recorder.count_kind("internal_api_call")
    wake_payload = _chat_turn(
        harness=real_agent_harness,
        thread_id="wake_guardrail_read_schedule_1",
        text="Simulate scheduled wake read run now.",
    )
    wake_text = str(wake_payload.get("text", "")).lower()
    assert "approval_id=" not in wake_text
    wake_calls = real_agent_harness.recorder.kind_slice("internal_api_call", start_wake)
    assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in wake_calls)
    assert not any(item.get("path") == "/internal/approvals/evaluate" for item in wake_calls)


def test_e2e_telegram_deny_triggers_agent_iteration(
    real_agent_telegram_harness: _Harness,
) -> None:
    assert real_agent_telegram_harness.telegram_client is not None
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-guardrail-deny-1",
        chat_id=777,
        user_id=999,
    )
    pending = _chat_turn(
        harness=real_agent_telegram_harness,
        thread_id="chat-guardrail-deny-1",
        text="Run external write now: post a test payload to https://mockapi.io/api/v1/posts",
    )
    approval_id = _extract_approval_id(str(pending.get("text", "")))
    assert approval_id

    _post_telegram_approval_callback(
        harness=real_agent_telegram_harness,
        approval_id=approval_id,
        decision="deny",
        callback_id="cbq_write_deny_1",
    )

    tg_client = real_agent_telegram_harness.telegram_client
    assert _wait_until(
        lambda: any(
            "resubmit for approval" in str(item.get("text", "")).lower()
            for item in tg_client.sent_messages
        ),
        timeout_seconds=3.0,
    )

    assert any(
        str(item.get("text", "")).strip() == "Action denied."
        for item in tg_client.callback_answers
    )
    assert any(
        "resubmit for approval" in str(item.get("text", "")).lower()
        for item in tg_client.sent_messages
    )
    assert any(
        "denied your planned action" in str(item.get("user_text", "")).lower()
        for item in real_agent_telegram_harness.recorder.entries
        if item.get("kind") == "model_input"
    )
    assert real_agent_telegram_harness.log_path.exists()
    assert real_agent_telegram_harness.log_path.read_text(encoding="utf-8").strip()


def test_e2e_schedule_write_deny_prevents_creation_and_iterates(
    real_agent_telegram_harness: _Harness,
) -> None:
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-guardrail-schedule-deny-1",
        chat_id=777,
        user_id=999,
    )
    start = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    create_payload = _chat_turn(
        harness=real_agent_telegram_harness,
        thread_id="chat-guardrail-schedule-deny-1",
        text=(
            "Create recurring schedule to publish updates to "
            "https://mockapi.io/api/v1/posts every hour."
        ),
    )
    approval_id = _extract_approval_id(str(create_payload.get("text", "")))
    assert approval_id
    create_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in create_calls)
    assert not any(item.get("path") == "/internal/scheduler/routine" for item in create_calls)

    start_deny = real_agent_telegram_harness.recorder.count_kind("internal_api_call")
    _post_telegram_approval_callback(
        harness=real_agent_telegram_harness,
        approval_id=approval_id,
        decision="deny",
        callback_id="cbq_schedule_deny_1",
    )
    tg_client = real_agent_telegram_harness.telegram_client
    assert tg_client is not None
    assert _wait_until(
        lambda: any(
            "resubmit for approval" in str(item.get("text", "")).lower()
            for item in tg_client.sent_messages
        ),
        timeout_seconds=3.0,
    )
    deny_calls = real_agent_telegram_harness.recorder.kind_slice("internal_api_call", start_deny)
    assert not any(item.get("path") == "/internal/approvals/execute" for item in deny_calls)
    assert not any(item.get("path") == "/internal/scheduler/routine" for item in deny_calls)
