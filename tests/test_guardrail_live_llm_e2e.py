from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime
from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.scheduler.service import SchedulerService

LIVE_E2E_FLAG = "OPENTULPA_ENABLE_LIVE_LLM_E2E"
TEST_CUSTOMER_ID = "cust_live_llm_guardrail"
E2E_LOG_DIR = Path(".opentulpa/logs/e2e").resolve()


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        return float(default)
    return max(1.0, value)


LIVE_E2E_POST_DEADLINE_SECONDS = _env_float("OPENTULPA_LIVE_E2E_POST_DEADLINE_SECONDS", 180.0)
LIVE_E2E_WAIT_TIMEOUT_SECONDS = _env_float("OPENTULPA_LIVE_E2E_WAIT_TIMEOUT_SECONDS", 45.0)

if str(os.getenv(LIVE_E2E_FLAG, "")).strip().lower() not in {"1", "true", "yes"}:
    pytest.skip(
        f"set {LIVE_E2E_FLAG}=1 to run live OpenRouter guardrail e2e tests",
        allow_module_level=True,
    )

_settings_probe = get_settings()
if not str(_settings_probe.openrouter_api_key or "").strip():
    pytest.skip("OPENROUTER_API_KEY is required for live_llm_e2e", allow_module_level=True)


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

    def slice_kind(self, kind: str, start: int) -> list[dict[str, Any]]:
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


@dataclass
class _Harness:
    client: TestClient
    app: Any
    runtime: OpenTulpaLangGraphRuntime
    recorder: _JsonlRecorder
    log_path: Path
    telegram_client: _FakeTelegramClient


def _extract_approval_id(text: str) -> str:
    match = re.search(r"\bapr_[a-z0-9_-]{6,40}\b", str(text or ""), flags=re.IGNORECASE)
    return str(match.group(0)).strip() if match else ""


def _decode_json(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(str(text or "").strip())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _wait_until(predicate: Any, timeout_seconds: float = LIVE_E2E_WAIT_TIMEOUT_SECONDS) -> bool:
    deadline = time.time() + max(0.2, float(timeout_seconds))
    while time.time() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.1)
    return bool(predicate())


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
            "wake_thread_id": "wake_live_llm_seeded",
            "last_user_message_at": now_utc,
            "last_assistant_message_at": now_utc,
        }
        state["sessions"] = sessions

    STATE_STORE.update(_mutate)


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
                    response_text=str(response.text or "")[:4000],
                )
                return response
            except Exception as exc:  # pragma: no cover
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
        raise RuntimeError(f"internal request failed: {last_exc}")

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]


@pytest.fixture()
def live_llm_harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Harness:
    settings = get_settings()
    fake_telegram = _FakeTelegramClient("live-llm-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "live-llm-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "live-llm-secret")
    monkeypatch.setenv("APPROVALS_DB_PATH", str(tmp_path / "live_llm_approvals.db"))
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "live_llm_links.db"))
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_telegram)
    get_settings.cache_clear()
    settings = get_settings()

    log_path = E2E_LOG_DIR / "guardrail_live_llm.jsonl"
    recorder = _JsonlRecorder(log_path)
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://testserver",
        openrouter_api_key=str(settings.openrouter_api_key or "").strip(),
        model_name=settings.llm_model,
        wake_classifier_model_name=settings.wake_classifier_model,
        guardrail_classifier_model_name=settings.guardrail_classifier_model,
        checkpoint_db_path=str(tmp_path / "live_llm_checkpoints.sqlite"),
        behavior_log_path=str(E2E_LOG_DIR / "guardrail_live_llm_behavior.jsonl"),
        behavior_log_enabled=True,
    )
    scheduler = SchedulerService(db_path=tmp_path / "live_llm_scheduler.db")
    app = create_app(agent_runtime=runtime, scheduler=scheduler)
    _patch_runtime_internal_api(runtime=runtime, app=app, recorder=recorder)
    client = TestClient(app)
    client.__enter__()
    harness = _Harness(
        client=client,
        app=app,
        runtime=runtime,
        recorder=recorder,
        log_path=log_path,
        telegram_client=fake_telegram,
    )
    try:
        yield harness
    finally:
        client.__exit__(None, None, None)
        get_settings.cache_clear()


def _chat_turn(*, harness: _Harness, thread_id: str, text: str) -> dict[str, Any]:
    harness.recorder.add("user_turn", thread_id=thread_id, text=text)
    response = _post_json_with_deadline(
        harness=harness,
        path="/internal/chat",
        body={
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


def _calls_since(harness: _Harness, start: int) -> list[dict[str, Any]]:
    return harness.recorder.slice_kind("internal_api_call", start)


def _post_json_with_deadline(
    *,
    harness: _Harness,
    path: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    deadline_seconds: float = LIVE_E2E_POST_DEADLINE_SECONDS,
) -> httpx.Response:
    timeout_s = max(1.0, float(deadline_seconds))
    alarm_enabled = bool(hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"))
    alarm_fired = False
    old_handler: Any = None

    def _handle_alarm(_signum: int, _frame: Any) -> None:
        nonlocal alarm_fired
        alarm_fired = True
        raise TimeoutError(f"request timed out for {path}")

    if alarm_enabled:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _handle_alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout_s)

    try:
        return harness.client.post(
            path,
            json=body,
            headers=headers,
        )
    except Exception as exc:
        if alarm_fired:
            harness.recorder.add(
                "client_post_timeout",
                path=path,
                deadline_seconds=float(timeout_s),
                body=body or {},
            )
            raise AssertionError(f"request timed out for {path} after {timeout_s:.1f}s") from exc
        raise
    finally:
        if alarm_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old_handler)


def _latest_approval_id_from_calls(calls: list[dict[str, Any]], *, action_name: str) -> str:
    for item in reversed(calls):
        if str(item.get("path", "")).strip() != "/internal/approvals/evaluate":
            continue
        json_body = item.get("json_body", {})
        if not isinstance(json_body, dict):
            continue
        if str(json_body.get("action_name", "")).strip() != action_name:
            continue
        payload = _decode_json(str(item.get("response_text", "")))
        approval_id = str(payload.get("approval_id", "")).strip()
        if approval_id:
            return approval_id
    return ""


def _post_approval_callback(
    *,
    harness: _Harness,
    approval_id: str,
    decision: str,
    callback_id: str,
    chat_id: int = 777,
    user_id: int = 999,
) -> None:
    response = _post_json_with_deadline(
        harness=harness,
        path="/webhook/telegram",
        headers={"x-telegram-bot-api-secret-token": "live-llm-secret"},
        body={
            "callback_query": {
                "id": callback_id,
                "from": {"id": int(user_id)},
                "message": {"message_id": 55, "chat": {"id": int(chat_id)}},
                "data": f"approval:{approval_id}:{decision}",
            }
        },
    )
    assert response.status_code == 200


def test_live_llm_action_read_write_and_subflows(live_llm_harness: _Harness) -> None:
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-live-guardrail-actions",
        chat_id=777,
        user_id=999,
    )

    # action_read
    start_read = live_llm_harness.recorder.count_kind("internal_api_call")
    _chat_turn(
        harness=live_llm_harness,
        thread_id="chat-live-guardrail-actions",
        text=(
            "Execute a read-only external command now using tulpa_run_terminal:\n"
            "curl https://mockapi.io/\n"
            "Then report what happened."
        ),
    )
    read_calls = _calls_since(live_llm_harness, start_read)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in read_calls)
    assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in read_calls)

    # action_write -> deny
    start_write_deny = live_llm_harness.recorder.count_kind("internal_api_call")
    first_write = _chat_turn(
        harness=live_llm_harness,
        thread_id="chat-live-guardrail-actions",
        text=(
            "Execute an external write now using tulpa_run_terminal exactly once:\n"
            "curl -X POST https://mockapi.io/api/v1/posts "
            "-H \"Content-Type: application/json\" "
            "-d '{\"source\":\"live_llm\",\"kind\":\"action_write_deny\"}'"
        ),
    )
    write_deny_calls = _calls_since(live_llm_harness, start_write_deny)
    approval_id = _extract_approval_id(str(first_write.get("text", ""))) or _latest_approval_id_from_calls(
        write_deny_calls,
        action_name="tulpa_run_terminal",
    )
    assert approval_id
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in write_deny_calls)
    assert not any(item.get("path") == "/internal/tulpa/run_terminal" for item in write_deny_calls)

    _post_approval_callback(
        harness=live_llm_harness,
        approval_id=approval_id,
        decision="deny",
        callback_id="cbq_live_action_deny",
    )
    assert _wait_until(
        lambda: any(
            "resubmit" in str(item.get("text", "")).lower()
            for item in live_llm_harness.telegram_client.sent_messages
        ),
        timeout_seconds=LIVE_E2E_WAIT_TIMEOUT_SECONDS,
    )

    # action_write -> approve
    start_write_approve = live_llm_harness.recorder.count_kind("internal_api_call")
    second_write = _chat_turn(
        harness=live_llm_harness,
        thread_id="chat-live-guardrail-actions",
        text=(
            "Retry the external write with the same command and request approval if needed:\n"
            "curl -X POST https://mockapi.io/api/v1/posts "
            "-H \"Content-Type: application/json\" "
            "-d '{\"source\":\"live_llm\",\"kind\":\"action_write_approve\"}'"
        ),
    )
    write_approve_calls = _calls_since(live_llm_harness, start_write_approve)
    approval_id_2 = _extract_approval_id(str(second_write.get("text", ""))) or _latest_approval_id_from_calls(
        write_approve_calls,
        action_name="tulpa_run_terminal",
    )
    assert approval_id_2
    start_after_approve = live_llm_harness.recorder.count_kind("internal_api_call")
    _post_approval_callback(
        harness=live_llm_harness,
        approval_id=approval_id_2,
        decision="approve",
        callback_id="cbq_live_action_approve",
    )
    assert _wait_until(
        lambda: any(
            item.get("path") == "/internal/approvals/execute"
            for item in _calls_since(live_llm_harness, start_after_approve)
        ),
        timeout_seconds=LIVE_E2E_WAIT_TIMEOUT_SECONDS,
    )
    post_approve_calls = _calls_since(live_llm_harness, start_after_approve)
    assert any(item.get("path") == "/internal/approvals/execute" for item in post_approve_calls)
    assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in post_approve_calls)


def test_live_llm_schedule_read_write_and_subflows(live_llm_harness: _Harness) -> None:
    _seed_telegram_session(
        customer_id=TEST_CUSTOMER_ID,
        thread_id="chat-live-guardrail-schedules",
        chat_id=777,
        user_id=999,
    )

    # schedule_read create (allow)
    start_schedule_read = live_llm_harness.recorder.count_kind("internal_api_call")
    _chat_turn(
        harness=live_llm_harness,
        thread_id="chat-live-guardrail-schedules",
        text=(
            "Create recurring read schedule now with routine_create.\n"
            "name: LiveReadSchedule\n"
            "schedule: 0 * * * *\n"
            "instruction: You must call tulpa_run_terminal with `curl https://mockapi.io/` at run time and report concise status.\n"
            "implementation_command: curl https://mockapi.io/\n"
            "notify_user: true"
        ),
    )
    schedule_read_calls = _calls_since(live_llm_harness, start_schedule_read)
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in schedule_read_calls)
    assert any(item.get("path") == "/internal/scheduler/routine" for item in schedule_read_calls)

    # schedule_write create (require approval) then deny
    start_schedule_write_deny = live_llm_harness.recorder.count_kind("internal_api_call")
    write_schedule = _chat_turn(
        harness=live_llm_harness,
        thread_id="chat-live-guardrail-schedules",
        text=(
            "Create recurring write schedule now with routine_create.\n"
            "name: LiveWriteSchedule\n"
            "schedule: 0 * * * *\n"
            "instruction: You must call tulpa_run_terminal with a POST to mockapi at run time and report concise status.\n"
            "implementation_command: curl -X POST https://mockapi.io/api/v1/posts "
            "-H \"Content-Type: application/json\" -d '{\"source\":\"live_llm\",\"kind\":\"schedule_write_deny\"}'\n"
            "notify_user: true"
        ),
    )
    schedule_write_deny_calls = _calls_since(live_llm_harness, start_schedule_write_deny)
    approval_id = _extract_approval_id(str(write_schedule.get("text", ""))) or _latest_approval_id_from_calls(
        schedule_write_deny_calls,
        action_name="routine_create",
    )
    assert approval_id
    assert any(item.get("path") == "/internal/approvals/evaluate" for item in schedule_write_deny_calls)
    assert not any(item.get("path") == "/internal/scheduler/routine" for item in schedule_write_deny_calls)

    _post_approval_callback(
        harness=live_llm_harness,
        approval_id=approval_id,
        decision="deny",
        callback_id="cbq_live_schedule_deny",
    )
    assert _wait_until(
        lambda: any(
            "resubmit" in str(item.get("text", "")).lower()
            for item in live_llm_harness.telegram_client.sent_messages
        ),
        timeout_seconds=LIVE_E2E_WAIT_TIMEOUT_SECONDS,
    )

    # schedule_write create (require approval) then approve
    start_schedule_write_approve = live_llm_harness.recorder.count_kind("internal_api_call")
    write_schedule_2 = _chat_turn(
        harness=live_llm_harness,
        thread_id="chat-live-guardrail-schedules",
        text=(
            "Retry recurring write schedule creation with routine_create.\n"
            "name: LiveWriteScheduleApproved\n"
            "schedule: 0 * * * *\n"
            "instruction: You must call tulpa_run_terminal with POST to mockapi at run time and report concise status.\n"
            "implementation_command: curl -X POST https://mockapi.io/api/v1/posts "
            "-H \"Content-Type: application/json\" -d '{\"source\":\"live_llm\",\"kind\":\"schedule_write_approve\"}'\n"
            "notify_user: true"
        ),
    )
    schedule_write_approve_calls = _calls_since(live_llm_harness, start_schedule_write_approve)
    approval_id_2 = _extract_approval_id(str(write_schedule_2.get("text", ""))) or _latest_approval_id_from_calls(
        schedule_write_approve_calls,
        action_name="routine_create",
    )
    assert approval_id_2

    start_after_approve = live_llm_harness.recorder.count_kind("internal_api_call")
    _post_approval_callback(
        harness=live_llm_harness,
        approval_id=approval_id_2,
        decision="approve",
        callback_id="cbq_live_schedule_approve",
    )
    assert _wait_until(
        lambda: any(
            item.get("path") == "/internal/scheduler/routine"
            for item in _calls_since(live_llm_harness, start_after_approve)
        ),
        timeout_seconds=LIVE_E2E_WAIT_TIMEOUT_SECONDS,
    )
    approved_calls = _calls_since(live_llm_harness, start_after_approve)
    assert any(item.get("path") == "/internal/approvals/execute" for item in approved_calls)
    assert any(item.get("path") == "/internal/scheduler/routine" for item in approved_calls)

    # scheduled read/write wake behavior: execute without runtime approval prompt
    wake_payloads = [
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "routine_id": "rtn_live_read_wake",
            "routine_name": "LiveReadScheduleWake",
            "customer_id": TEST_CUSTOMER_ID,
            "notify_user": True,
            "payload": {
                "customer_id": TEST_CUSTOMER_ID,
                "notify_user": True,
                "instruction": (
                    "Scheduled run instruction: call tulpa_run_terminal with command "
                    "`curl https://mockapi.io/` and send concise status."
                ),
            },
        },
        {
            "type": "routine_event",
            "event_type": "scheduled",
            "routine_id": "rtn_live_write_wake",
            "routine_name": "LiveWriteScheduleWake",
            "customer_id": TEST_CUSTOMER_ID,
            "notify_user": True,
            "payload": {
                "customer_id": TEST_CUSTOMER_ID,
                "notify_user": True,
                "instruction": (
                    "Scheduled run instruction: call tulpa_run_terminal with command "
                    "`curl -X POST https://mockapi.io/api/v1/posts "
                    "-H \"Content-Type: application/json\" "
                    "-d '{\"source\":\"live_llm\",\"kind\":\"wake_write\"}'` "
                    "and then send concise status."
                ),
            },
        },
    ]
    for idx, wake_body in enumerate(wake_payloads, start=1):
        start_wake = live_llm_harness.recorder.count_kind("internal_api_call")
        wake_response = _post_json_with_deadline(
            harness=live_llm_harness,
            path="/internal/wake",
            body=wake_body,
        )
        assert wake_response.status_code == 200
        assert _wait_until(
            lambda _start_wake=start_wake: any(
                item.get("path") == "/internal/tulpa/run_terminal"
                for item in _calls_since(live_llm_harness, _start_wake)
            ),
            timeout_seconds=max(LIVE_E2E_WAIT_TIMEOUT_SECONDS, 60.0),
        ), f"scheduled wake {idx} did not execute tulpa_run_terminal"
        wake_calls = _calls_since(live_llm_harness, start_wake)
        assert any(item.get("path") == "/internal/tulpa/run_terminal" for item in wake_calls)
        wake_terminal_evals = [
            item
            for item in wake_calls
            if item.get("path") == "/internal/approvals/evaluate"
            and str((item.get("json_body") or {}).get("action_name", "")).strip() == "tulpa_run_terminal"
        ]
        assert not wake_terminal_evals

    assert live_llm_harness.log_path.exists()
    assert live_llm_harness.log_path.read_text(encoding="utf-8").strip()
