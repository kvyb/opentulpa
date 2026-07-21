from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

from opentulpa.capability_workers.agent_api import (
    AgentAPIError,
    AgentEvent,
    AgentNotification,
    AgentNotificationApproval,
)
from opentulpa.capability_workers.state import TelegramWorkerState
from opentulpa.capability_workers.telegram_api import TelegramAPIError, TelegramAttachment
from opentulpa.capability_workers.telegram_worker import (
    TelegramInterfaceWorker,
    parse_edited_arguments,
)


def _event(
    event_type: str,
    sequence: int,
    data: Mapping[str, Any],
    *,
    run_id: str = "run_1",
) -> AgentEvent:
    return AgentEvent(
        type=event_type,
        run_id=run_id,
        sequence=sequence,
        timestamp="2026-07-20T00:00:00Z",
        data=data,
    )


class _Telegram:
    def __init__(self, updates: list[dict[str, Any]] | None = None) -> None:
        self.updates = updates or []
        self.messages: list[dict[str, Any]] = []
        self.callbacks: list[dict[str, Any]] = []
        self.downloads: list[TelegramAttachment] = []
        self.webhook_deletes = 0

    async def delete_webhook(self) -> None:
        self.webhook_deletes += 1

    async def get_me(self) -> dict[str, Any]:
        return {"id": 99, "username": "open_tulpa_bot"}

    async def get_updates(self, *, offset: int, timeout_seconds: int) -> list[dict[str, Any]]:
        assert 1 <= timeout_seconds <= 50
        return [update for update in self.updates if int(update["update_id"]) >= offset]

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        self.messages.append(
            {"chat_id": chat_id, "text": text, "reply_markup": reply_markup}
        )

    async def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str,
        show_alert: bool = False,
    ) -> None:
        self.callbacks.append(
            {
                "callback_query_id": callback_query_id,
                "text": text,
                "show_alert": show_alert,
            }
        )

    async def download_attachment(self, attachment: TelegramAttachment) -> bytes:
        self.downloads.append(attachment)
        return b"attachment bytes"


class _Agent:
    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.uploads: list[dict[str, Any]] = []
        self.resumes: list[dict[str, Any]] = []
        self.replays: list[dict[str, Any]] = []
        self.notifications: list[AgentNotification] = []
        self.notification_requests: list[dict[str, Any]] = []
        self.notification_acks: list[int] = []
        self.fail_notification_ack = False
        self.start_events = [
            _event("run.started", 1, {}),
            _event("message.delta", 2, {"text": "hello"}),
            _event("run.completed", 3, {"text": "hello"}),
        ]
        self.resume_events = [_event("run.completed", 4, {"text": "approved"})]
        self.replay_events = [_event("run.completed", 3, {"text": "recovered"})]

    async def upload_file(self, **kwargs: Any) -> str:
        self.uploads.append(kwargs)
        return f"file_{len(self.uploads)}"

    async def start_run(self, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        self.starts.append(kwargs)
        for event in self.start_events:
            yield event

    async def resume_run(
        self,
        *,
        run_id: str,
        approval_id: str,
        decision: Literal["approve", "edit", "reject"],
        source_event_id: str,
        edited_arguments: Mapping[str, Any] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        self.resumes.append(
            {
                "run_id": run_id,
                "approval_id": approval_id,
                "decision": decision,
                "source_event_id": source_event_id,
                "edited_arguments": edited_arguments,
            }
        )
        for event in self.resume_events:
            yield event

    async def replay_run(self, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        self.replays.append(kwargs)
        for event in self.replay_events:
            yield event

    async def list_notifications(self, **kwargs: Any) -> list[AgentNotification]:
        self.notification_requests.append(kwargs)
        return [
            item
            for item in self.notifications
            if item.id > int(kwargs.get("after_id") or 0)
        ]

    async def acknowledge_notification(self, notification_id: int) -> None:
        if self.fail_notification_ack:
            raise AgentAPIError("ack unavailable")
        self.notification_acks.append(notification_id)


def _message(update_id: int, *, user_id: int = 7, chat_id: int = 9, text: str) -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "message_id": update_id,
            "from": {"id": user_id},
            "chat": {"id": chat_id},
            "text": text,
        },
    }


@pytest.mark.asyncio
async def test_one_time_pairing_then_routes_only_owner_messages(tmp_path: Path) -> None:
    telegram = _Telegram(
        [
            _message(1, text="/start pair-code"),
            _message(2, user_id=8, text="steal this bot"),
            _message(3, text="hello"),
        ]
    )
    agent = _Agent()
    state = TelegramWorkerState(tmp_path / "worker.json")
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code="pair-code",
        poll_timeout_seconds=1,
    )

    assert await worker.poll_once() == 3
    assert state.paired_identity() == (7, 9)
    assert len(agent.starts) == 1
    assert agent.starts[0]["text"] == "hello"
    assert agent.starts[0]["source_event_id"] == "telegram:99:3"
    assert [item["text"] for item in telegram.messages] == [
        "OpenTulpa is paired with this Telegram account. Send a request or attach files.",
        "Unauthorized.",
        "hello",
    ]
    assert state.next_update_id == 4

    await worker.poll_once()
    assert len(agent.starts) == 1


@pytest.mark.asyncio
async def test_attachment_is_downloaded_and_uploaded_before_agent_run(tmp_path: Path) -> None:
    update = _message(4, text="inspect")
    update["message"].pop("text")
    update["message"]["caption"] = "inspect"
    update["message"]["document"] = {
        "file_id": "tg_file",
        "file_name": "../notes.txt",
        "mime_type": "text/plain",
        "file_size": 100,
    }
    telegram = _Telegram([update])
    agent = _Agent()
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()

    assert telegram.downloads[0].file_id == "tg_file"
    assert agent.uploads[0]["filename"] == "notes.txt"
    assert agent.uploads[0]["source_event_id"] == "telegram:99:4:file:0"
    assert agent.starts[0]["file_ids"] == ["file_1"]


@pytest.mark.asyncio
async def test_approval_buttons_are_durable_and_callback_resumes_via_api(tmp_path: Path) -> None:
    telegram = _Telegram([_message(1, text="send the email")])
    agent = _Agent()
    agent.start_events = [
        _event("run.started", 1, {}),
        _event(
            "approval.required",
            2,
            {
                "approval_id": "approval_1",
                "tool_name": "integration_invoke",
                "description": "Send email",
                "allowed_decisions": ["approve", "edit", "reject"],
            },
        ),
    ]
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()
    buttons = telegram.messages[-1]["reply_markup"]["inline_keyboard"][0]
    assert [button["text"] for button in buttons] == ["Approve", "Edit", "Reject"]
    callback_data = buttons[0]["callback_data"]
    token = callback_data.split(":")[1]
    assert state.approval(token) is not None

    telegram.updates.append(
        {
            "update_id": 2,
            "callback_query": {
                "id": "callback_1",
                "from": {"id": 7},
                "message": {"chat": {"id": 9}},
                "data": callback_data,
            },
        }
    )
    await worker.poll_once()

    assert agent.resumes == [
        {
            "run_id": "run_1",
            "approval_id": "approval_1",
            "decision": "approve",
            "source_event_id": "telegram:99:2",
            "edited_arguments": None,
        }
    ]
    assert telegram.callbacks[-1]["text"] == "Processing approval..."
    assert telegram.messages[-1]["text"] == "approved"
    assert state.approval(token) is None
    assert state.next_update_id == 3


@pytest.mark.asyncio
async def test_recovery_replays_from_durable_sequence_without_new_run(tmp_path: Path) -> None:
    telegram = _Telegram()
    agent = _Agent()
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    state.save_pending_run(
        source_event_id="telegram:99:5",
        update_id=5,
        run_id="run_1",
        chat_id=9,
        sequence=2,
        accumulated_text="partial",
    )
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.recover()

    assert agent.replays == [{"run_id": "run_1", "after_sequence": 2}]
    assert agent.starts == []
    assert telegram.messages[-1]["text"] == "recovered"
    assert state.source_seen("telegram:99:5")
    assert state.pending_runs() == []


@pytest.mark.asyncio
async def test_background_notifications_and_approvals_deliver_to_paired_owner(
    tmp_path: Path,
) -> None:
    telegram = _Telegram()
    agent = _Agent()
    agent.notifications = [
        AgentNotification(
            id=7,
            kind="approval.required",
            text="The scheduled run is waiting.",
            status="interrupted",
            thread_id="trigger:daily",
            run_id="run-background",
            approvals=(
                AgentNotificationApproval(
                    approval_id="approval-background",
                    tool_name="integration_invoke",
                    description="Send an external message.",
                    allowed_decisions=("approve", "reject"),
                ),
            ),
            created_at="2026-07-20T00:00:00+00:00",
        )
    ]
    state_path = tmp_path / "worker.json"
    state = TelegramWorkerState(state_path)
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()

    assert telegram.messages[0]["text"] == "The scheduled run is waiting."
    assert telegram.messages[1]["text"].startswith("Approval required")
    assert agent.notification_acks == [7]
    assert state.notification_cursor == 7
    approval = next(iter(state.snapshot()["approvals"].values()))
    assert approval["run_id"] == "run-background"
    assert approval["approval_id"] == "approval-background"

    restarted = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=TelegramWorkerState(state_path),
        pairing_code=None,
        poll_timeout_seconds=1,
    )
    await restarted.recover()
    assert [item["text"] for item in telegram.messages].count(
        "The scheduled run is waiting."
    ) == 1


@pytest.mark.asyncio
async def test_notification_ack_retries_without_redelivering_telegram_message(
    tmp_path: Path,
) -> None:
    telegram = _Telegram()
    agent = _Agent()
    agent.notifications = [
        AgentNotification(
            id=3,
            kind="evolution.candidate.failed",
            text="Candidate evaluation failed; the current release was retained.",
            status="failed",
            thread_id="thread-1",
            run_id=None,
            approvals=(),
            created_at="2026-07-20T00:00:00+00:00",
        )
    ]
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )
    agent.fail_notification_ack = True

    with pytest.raises(AgentAPIError, match="ack unavailable"):
        await worker.poll_once()

    assert state.notification_cursor == 3
    assert state.pending_notification_acks() == [3]
    assert [item["text"] for item in telegram.messages] == [
        "Candidate evaluation failed; the current release was retained."
    ]

    agent.fail_notification_ack = False
    await worker.poll_once()
    assert agent.notification_acks == [3]
    assert state.pending_notification_acks() == []
    assert len(telegram.messages) == 1


@pytest.mark.asyncio
async def test_wrong_owner_callback_cannot_consume_approval(tmp_path: Path) -> None:
    telegram = _Telegram()
    agent = _Agent()
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    state.save_approval(
        token="approvaltoken",
        run_id="run_1",
        approval_id="approval_1",
        chat_id=9,
        user_id=7,
        allowed_decisions=["approve"],
        tool_name="integration_invoke",
        description="Send email",
    )
    telegram.updates.append(
        {
            "update_id": 1,
            "callback_query": {
                "id": "callback_bad",
                "from": {"id": 8},
                "message": {"chat": {"id": 9}},
                "data": "ot:approvaltoken:approve",
            },
        }
    )
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()

    assert agent.resumes == []
    assert state.approval("approvaltoken") is not None
    assert telegram.callbacks[-1]["text"] == "Unauthorized"


def test_edited_arguments_are_bounded_and_reject_hidden_context_recursively() -> None:
    assert parse_edited_arguments('{"recipient":"owner@example.com"}') == {
        "recipient": "owner@example.com"
    }
    with pytest.raises(ValueError, match="context"):
        parse_edited_arguments('{"nested":{"tenant_id":"other"}}')
    with pytest.raises(ValueError, match="non-finite"):
        parse_edited_arguments('{"amount":1e10000}')


@pytest.mark.asyncio
async def test_invalid_bot_identity_fails_startup_before_readiness(tmp_path: Path) -> None:
    class _InvalidTokenTelegram(_Telegram):
        async def get_me(self) -> dict[str, Any]:
            raise TelegramAPIError("Telegram getMe rejected the request.")

    telegram = _InvalidTokenTelegram()
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=_Agent(),
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )
    readiness: list[bool] = []

    with pytest.raises(TelegramAPIError, match="getMe"):
        await worker.run(asyncio.Event(), on_ready=lambda: readiness.append(True))

    assert readiness == []
    assert telegram.webhook_deletes == 0


@pytest.mark.asyncio
async def test_run_cancels_inflight_long_poll_on_shutdown(tmp_path: Path) -> None:
    entered = asyncio.Event()

    class _BlockingTelegram(_Telegram):
        async def get_updates(
            self,
            *,
            offset: int,
            timeout_seconds: int,
        ) -> list[dict[str, Any]]:
            del offset, timeout_seconds
            entered.set()
            await asyncio.Future()
            return []

    telegram = _BlockingTelegram()
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=_Agent(),
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    await asyncio.wait_for(entered.wait(), timeout=1)
    stop.set()

    await asyncio.wait_for(task, timeout=1)
    assert telegram.webhook_deletes == 1
