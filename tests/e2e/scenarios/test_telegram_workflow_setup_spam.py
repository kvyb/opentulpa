from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from mocks.telegram import FakeTelegramClient

from opentulpa.api import app as app_module
from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings
from opentulpa.interfaces.telegram import attachments as attachments_module
from opentulpa.interfaces.telegram import chat_service as chat_module
from opentulpa.interfaces.telegram import relay as relay_module
from opentulpa.interfaces.telegram.state_store import TelegramStateStore
from opentulpa.scheduler.service import SchedulerService
from opentulpa.tasks import sandbox as sandbox_module

pytestmark = [pytest.mark.e2e, pytest.mark.telegram]


def _telegram_message(
    *,
    chat_id: int,
    user_id: int,
    text: str,
    message_id: int,
    date: int | None = None,
) -> dict[str, Any]:
    return {
        "update_id": int(time.time() * 1000) + int(message_id),
        "message": {
            "message_id": int(message_id),
            "date": int(date if date is not None else time.time()),
            "chat": {"id": chat_id, "type": "private"},
            "from": {"id": user_id, "is_bot": False, "username": f"user_{user_id}"},
            "text": text,
        },
    }


def _wait_until(predicate: Any, timeout_seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    while time.monotonic() < deadline:
        if bool(predicate()):
            return True
        time.sleep(0.05)
    return bool(predicate())


def _owner_thread_id(*, chat_id: int) -> str:
    state = chat_module.STATE_STORE.load()
    sessions = state.get("sessions") if isinstance(state, dict) else {}
    slot = sessions.get(str(chat_id)) if isinstance(sessions, dict) else {}
    return str(slot.get("thread_id", "") or "").strip() if isinstance(slot, dict) else ""


class _SlowWorkflowSetupRuntime:
    def __init__(self) -> None:
        self.released = False
        self.ainvoke_calls: list[dict[str, Any]] = []
        self.classifier_calls: list[dict[str, Any]] = []

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def ainvoke_text(self, **kwargs: Any) -> str:
        self.ainvoke_calls.append(dict(kwargs))
        if len(self.ainvoke_calls) == 1:
            while not self.released:
                await asyncio.sleep(0.01)
            return "Old workflow proposal: should not be delivered after queued edit."
        return "Updated workflow proposal: includes the new required fields. Please confirm to activate it."

    async def classify_workflow_setup_interruption(self, **kwargs: Any) -> dict[str, Any]:
        self.classifier_calls.append(dict(kwargs))
        text = str(kwargs.get("text", "") or "").lower()
        if "работ" in text or "what" in text or "status" in text:
            return {
                "ok": True,
                "kind": "status_nudge",
                "confidence": 0.99,
                "status_reply": str(kwargs["status"]["reply_if_status_nudge"]),
                "reason": "Progress nudge.",
            }
        return {
            "ok": True,
            "kind": "setup_input",
            "confidence": 0.98,
            "status_reply": "",
            "reason": "Substantive workflow setup input.",
        }


def test_telegram_workflow_setup_spam_does_not_duplicate_runs_and_applies_queued_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tg = FakeTelegramClient("fake-token")
    runtime = _SlowWorkflowSetupRuntime()
    project_root = tmp_path / "project_root"
    project_root.mkdir(parents=True)

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-bot-token")
    monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USER_IDS", "")
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERNAMES", "")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "test-key")
    monkeypatch.setenv("LINK_ALIAS_DB_PATH", str(tmp_path / "links.sqlite"))
    monkeypatch.setattr(relay_module, "WORKFLOW_SETUP_FINAL_REPLY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(sandbox_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_module, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(
        chat_module,
        "STATE_STORE",
        TelegramStateStore(project_root / ".opentulpa" / "telegram_state.json"),
    )
    monkeypatch.setattr(app_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(attachments_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(chat_module, "TelegramClient", lambda _token: fake_tg)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda _token: fake_tg)
    get_settings.cache_clear()
    relay_module._WORKFLOW_SETUP_RUNS.clear()

    app = create_app(
        agent_runtime=runtime,
        scheduler=SchedulerService(db_path=tmp_path / "scheduler.sqlite"),
    )
    owner_chat_id = 771
    owner_user_id = 1771
    customer_id = f"telegram_{owner_user_id}"

    try:
        with TestClient(app) as client:
            fresh = client.post(
                "/webhook/telegram",
                headers={"x-telegram-bot-api-secret-token": "test-secret"},
                json=_telegram_message(
                    chat_id=owner_chat_id,
                    user_id=owner_user_id,
                    text="/fresh",
                    message_id=1,
                    date=1_800_000_001,
                ),
            )
            assert fresh.status_code == 200
            assert _wait_until(lambda: bool(_owner_thread_id(chat_id=owner_chat_id)))
            thread_id = _owner_thread_id(chat_id=owner_chat_id)
            setup = app.state.intake_workflow_setup.begin_session(
                customer_id=customer_id,
                thread_id=thread_id,
                mode="create",
            )
            setup_session_id = str(setup["session_id"])

            start = client.post(
                "/webhook/telegram",
                headers={"x-telegram-bot-api-secret-token": "test-secret"},
                json=_telegram_message(
                    chat_id=owner_chat_id,
                    user_id=owner_user_id,
                    text="Start configuring my intake workflow.",
                    message_id=2,
                    date=1_800_000_002,
                ),
            )
            assert start.status_code == 200
            assert _wait_until(lambda: len(runtime.ainvoke_calls) == 1), [
                item.get("text") for item in fake_tg.sent_messages
            ]
            assert _wait_until(
                lambda: any("still working" in str(item.get("text", "")).lower() for item in fake_tg.sent_messages)
            )

            nudge = client.post(
                "/webhook/telegram",
                headers={"x-telegram-bot-api-secret-token": "test-secret"},
                json=_telegram_message(
                    chat_id=owner_chat_id,
                    user_id=owner_user_id,
                    text="работаешь?",
                    message_id=3,
                    date=1_800_000_003,
                ),
            )
            assert nudge.status_code == 200
            assert _wait_until(lambda: len(runtime.classifier_calls) == 1)
            assert len(runtime.ainvoke_calls) == 1

            edit = client.post(
                "/webhook/telegram",
                headers={"x-telegram-bot-api-secret-token": "test-secret"},
                json=_telegram_message(
                    chat_id=owner_chat_id,
                    user_id=owner_user_id,
                    text="Required fields: client, phone, service, day, time.",
                    message_id=4,
                    date=1_800_000_004,
                ),
            )
            assert edit.status_code == 200
            assert _wait_until(lambda: len(runtime.classifier_calls) == 2)
            assert len(runtime.ainvoke_calls) == 1
            assert any(
                "latest note" in str(item.get("text", "")).lower()
                for item in fake_tg.sent_messages
            )

            runtime.released = True
            assert _wait_until(lambda: len(runtime.ainvoke_calls) == 2)
            assert "Required fields: client" in str(runtime.ainvoke_calls[1]["text"])
            assert _wait_until(
                lambda: any(
                    "Updated workflow proposal" in str(item.get("text", ""))
                    for item in fake_tg.sent_messages
                ),
                timeout_seconds=5.0,
            )

            owner_messages = [
                str(item.get("text", ""))
                for item in fake_tg.sent_messages
                if int(item.get("chat_id", 0)) == owner_chat_id
            ]
            assert not any("Old workflow proposal" in text for text in owner_messages)
            assert len(runtime.ainvoke_calls) == 2
            assert app.state.intake_workflow_setup.get_thread_session(
                customer_id=customer_id,
                thread_id=thread_id,
                include_paused=True,
            )["session_id"] == setup_session_id
            assert app.state.intake_workflows.list_workflows(
                customer_id=customer_id,
                include_disabled=True,
            ) == []
    finally:
        runtime.released = True
        relay_module._WORKFLOW_SETUP_RUNS.clear()
