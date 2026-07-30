from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, Literal

import pytest

import opentulpa.capability_workers.telegram_worker as telegram_worker_module
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
        self.actions: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.webhook_deletes = 0
        self.commands: list[dict[str, str]] = []

    async def delete_webhook(self) -> None:
        self.webhook_deletes += 1

    async def get_me(self) -> dict[str, Any]:
        return {"id": 99, "username": "open_tulpa_bot"}

    async def set_my_commands(self, commands: list[dict[str, str]]) -> None:
        self.commands = list(commands)

    async def get_updates(self, *, offset: int, timeout_seconds: int) -> list[dict[str, Any]]:
        assert 1 <= timeout_seconds <= 50
        return [update for update in self.updates if int(update["update_id"]) >= offset]

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> list[int]:
        self.messages.append(
            {"chat_id": chat_id, "text": text, "reply_markup": reply_markup}
        )
        return [len(self.messages)]

    async def send_chat_action(self, *, chat_id: int, action: str = "typing") -> None:
        self.actions.append({"chat_id": chat_id, "action": action})

    async def edit_message_text(
        self,
        *,
        chat_id: int,
        message_id: int,
        text: str,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "reply_markup": reply_markup,
            }
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
        self.ensured_threads: list[str] = []
        self.inference_updates: list[dict[str, Any]] = []
        self.cancelled_threads: list[str] = []
        self.inference = {
            "revision": 0,
            "effective": {
                "provider": "api",
                "model": "openai/gpt-5.2",
                "reasoning_effort": "high",
            },
        }
        self.models = [
            {
                "id": "openai/gpt-5.2",
                "reasoning_efforts": ["low", "medium", "high"],
            }
        ]
        self.codex_status: dict[str, Any] = {"codex": {"connected": False}}
        self.codex_login = {
            "login_id": "login-1",
            "verification_url": "https://auth.openai.com/codex/device",
            "user_code": "ABCD-EFGH",
            "status": "pending",
        }
        self.start_events = [
            _event("run.started", 1, {}),
            _event("message.delta", 2, {"text": "hello"}),
            _event("run.completed", 3, {"text": "hello"}),
        ]
        self.resume_events = [_event("run.completed", 4, {"text": "approved"})]
        self.replay_events = [_event("run.completed", 3, {"text": "recovered"})]

    async def ensure_thread(self, thread_id: str) -> None:
        self.ensured_threads.append(thread_id)

    async def get_thread_inference(self, thread_id: str) -> dict[str, Any]:
        del thread_id
        return dict(self.inference)

    async def update_thread_inference(
        self,
        thread_id: str,
        *,
        expected_revision: int,
        selection: Mapping[str, Any],
    ) -> dict[str, Any]:
        self.inference_updates.append(
            {
                "thread_id": thread_id,
                "expected_revision": expected_revision,
                "selection": dict(selection),
            }
        )
        self.inference = {
            "revision": expected_revision + 1,
            "effective": dict(selection),
        }
        return dict(self.inference)

    async def inference_status(self) -> dict[str, Any]:
        return dict(self.codex_status)

    async def list_models(self, **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return list(self.models)

    async def start_codex_login(self) -> dict[str, Any]:
        return dict(self.codex_login)

    async def get_codex_login(self, login_id: str) -> dict[str, Any]:
        assert login_id == self.codex_login["login_id"]
        return dict(self.codex_login)

    async def cancel_thread(self, thread_id: str) -> dict[str, Any]:
        self.cancelled_threads.append(thread_id)
        return {"status": "cancelled"}

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


class _UnavailableAgent(_Agent):
    async def list_notifications(self, **kwargs: Any) -> list[AgentNotification]:
        del kwargs
        raise AgentAPIError("agent API is starting")


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
async def test_worker_becomes_ready_while_local_agent_api_is_starting(tmp_path: Path) -> None:
    telegram = _Telegram()
    stop = asyncio.Event()
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=_UnavailableAgent(),
        state=TelegramWorkerState(tmp_path / "worker.json"),
        pairing_code="pair-code",
        poll_timeout_seconds=1,
    )

    await worker.run(stop, on_ready=stop.set)

    assert stop.is_set()
    assert telegram.webhook_deletes == 1
    assert [item["command"] for item in telegram.commands] == [
        "start",
        "fresh",
        "regenerate",
        "model",
        "models",
        "reasoning",
        "codex",
        "cancel",
    ]


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
async def test_rapid_text_messages_start_one_agent_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        telegram_worker_module,
        "_TEXT_COALESCE_QUIET_SECONDS",
        0.02,
    )
    monkeypatch.setattr(
        telegram_worker_module,
        "_TEXT_COALESCE_MAX_SECONDS",
        0.1,
    )
    telegram = _Telegram([_message(index, text=f"part {index}") for index in range(1, 4)])
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

    poll = asyncio.create_task(worker.poll_once())
    while len(state.pending_updates()) < 3:
        await asyncio.sleep(0)
    await asyncio.sleep(0.01)
    for index in range(4, 7):
        state.accept_update(
            update_id=index,
            update=_message(index, text=f"part {index}"),
        )
    await poll

    assert len(agent.starts) == 1
    assert agent.starts[0]["text"] == "\n\n".join(
        f"part {index}" for index in range(1, 7)
    )
    assert agent.starts[0]["source_event_id"] == "telegram:99:1"
    assert state.pending_updates() == []
    assert state.next_update_id == 7


def test_coalesced_updates_survive_worker_restart(tmp_path: Path) -> None:
    state_path = tmp_path / "worker.json"
    state = TelegramWorkerState(state_path)
    state.accept_update(update_id=1, update=_message(1, text="first"))
    state.accept_update(update_id=2, update=_message(2, text="second"))
    merged = _message(1, text="first\n\nsecond")

    state.coalesce_updates(update_ids=(1, 2), merged_update=merged)

    assert TelegramWorkerState(state_path).pending_updates() == [(1, merged)]
    assert TelegramWorkerState(state_path).next_update_id == 3


@pytest.mark.asyncio
async def test_inference_and_codex_commands_use_agent_api_without_starting_runs(
    tmp_path: Path,
) -> None:
    telegram = _Telegram(
        [
            _message(1, text="/model"),
            _message(2, text="/reasoning ultra"),
            _message(3, text="/codex login"),
        ]
    )
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

    assert await worker.poll_once() == 3

    thread_id = state.thread_id(9)
    assert agent.ensured_threads == [thread_id, thread_id, thread_id]
    assert agent.starts == []
    assert agent.inference_updates[0]["selection"]["reasoning_effort"] == "ultra"
    assert state.codex_login(9) == "login-1"
    assert [message["text"].splitlines()[0] for message in telegram.messages] == [
        "Current model:",
        "Reasoning updated:",
        "Connect Codex:",
    ]


@pytest.mark.parametrize("status", ["expired", "failed"])
@pytest.mark.asyncio
async def test_terminal_codex_login_status_starts_over(
    tmp_path: Path,
    status: str,
) -> None:
    telegram = _Telegram([_message(1, text="/codex status")])
    agent = _Agent()
    agent.codex_login["status"] = status
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    state.set_codex_login(9, "login-1")
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    assert await worker.poll_once() == 1

    assert state.codex_login(9) == ""
    assert telegram.messages[-1]["text"] == (
        f"Codex login {status}. Run /codex login to start again."
    )


@pytest.mark.asyncio
async def test_fresh_conversation_preserves_explicit_model_selection(tmp_path: Path) -> None:
    telegram = _Telegram([_message(1, text="/fresh")])
    agent = _Agent()
    selection = {
        "provider": "codex",
        "model": "gpt-5.5",
        "reasoning_effort": "high",
    }
    agent.inference = {
        "revision": 3,
        "selection": selection,
        "effective": selection,
    }
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    current_thread_id = state.thread_id(9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()

    replacement_thread_id = state.thread_id(9)
    assert replacement_thread_id != current_thread_id
    assert agent.ensured_threads == [current_thread_id, replacement_thread_id]
    assert agent.inference_updates == [
        {
            "thread_id": replacement_thread_id,
            "expected_revision": 0,
            "selection": selection,
        }
    ]
    assert telegram.messages[-1]["text"] == "Started a fresh conversation."


@pytest.mark.asyncio
async def test_cancel_bypasses_a_long_running_message(tmp_path: Path) -> None:
    run_finished = asyncio.Event()
    stop = asyncio.Event()

    class _LongAgent(_Agent):
        async def start_run(self, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            self.starts.append(kwargs)
            yield _event("run.started", 1, {})
            await run_finished.wait()
            yield _event(
                "run.failed",
                2,
                {"message": "The agent run was cancelled before completion."},
            )

        async def cancel_thread(self, thread_id: str) -> dict[str, Any]:
            self.cancelled_threads.append(thread_id)
            run_finished.set()
            return {"status": "cancelled"}

    class _LiveTelegram(_Telegram):
        async def get_updates(
            self,
            *,
            offset: int,
            timeout_seconds: int,
        ) -> list[dict[str, Any]]:
            del timeout_seconds
            if offset <= 1:
                return [
                    _message(1, text="work for a long time"),
                    _message(2, text="/cancel"),
                ]
            await run_finished.wait()
            stop.set()
            return []

    telegram = _LiveTelegram()
    agent = _LongAgent()
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await asyncio.wait_for(worker.run(stop), timeout=2)

    assert agent.cancelled_threads == [state.thread_id(9)]
    assert "Cancellation requested." in [message["text"] for message in telegram.messages]


@pytest.mark.asyncio
async def test_message_deltas_stream_by_editing_one_telegram_message(
    tmp_path: Path,
) -> None:
    telegram = _Telegram([_message(4, text="hello")])
    agent = _Agent()
    agent.start_events = [
        _event("run.started", 1, {}),
        _event("message.delta", 2, {"text": "he"}),
        _event("message.delta", 3, {"text": "llo"}),
        _event("run.completed", 4, {"text": "hello"}),
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

    assert telegram.actions == [{"chat_id": 9, "action": "typing"}]
    assert [item["text"] for item in telegram.messages] == ["he"]
    assert telegram.edits == [
        {"chat_id": 9, "message_id": 1, "text": "hello", "reply_markup": None}
    ]


@pytest.mark.asyncio
async def test_transient_edit_failure_does_not_send_duplicate_message() -> None:
    class _TransientEditFailure(_Telegram):
        async def edit_message_text(self, **kwargs: Any) -> None:
            del kwargs
            raise TelegramAPIError("Telegram editMessageText transport failed.")

    telegram = _TransientEditFailure()
    streamer = telegram_worker_module._TelegramResponseStreamer(  # noqa: SLF001
        telegram=telegram,
        chat_id=9,
        message_id=41,
        rendered_text="before",
    )

    with pytest.raises(TelegramAPIError, match="transport failed"):
        await streamer.update("after")

    assert telegram.messages == []


@pytest.mark.asyncio
async def test_missing_edit_target_is_replaced_once() -> None:
    class _MissingEditTarget(_Telegram):
        async def edit_message_text(self, **kwargs: Any) -> None:
            del kwargs
            raise TelegramAPIError(
                "Telegram editMessageText rejected the request.",
                edit_target_unavailable=True,
            )

    telegram = _MissingEditTarget()
    streamer = telegram_worker_module._TelegramResponseStreamer(  # noqa: SLF001
        telegram=telegram,
        chat_id=9,
        message_id=41,
        rendered_text="before",
    )

    await streamer.update("after")

    assert [item["text"] for item in telegram.messages] == ["after"]
    assert streamer.message_id == 1


@pytest.mark.asyncio
async def test_tool_progress_flushes_throttled_text_and_finishes_cleanly(
    tmp_path: Path,
) -> None:
    telegram = _Telegram([_message(4, text="hello")])
    agent = _Agent()
    agent.start_events = [
        _event("run.started", 1, {}),
        _event("message.delta", 2, {"text": "Good"}),
        _event("message.delta", 3, {"text": " question."}),
        _event("tool.started", 4, {"name": "task"}),
        _event("tool.completed", 5, {"name": "task"}),
        _event("run.completed", 6, {"text": "Finished."}),
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

    assert telegram.messages[0]["text"] == "Good"
    assert telegram.edits[0]["text"] == (
        "Good question.\n\nWorking: Delegating work (0s)..."
    )
    assert telegram.edits[1]["text"] == (
        "Good question.\n\nWorking: Finishing response (0s)..."
    )
    assert telegram.edits[-1]["text"] == "Finished."


@pytest.mark.asyncio
async def test_tool_progress_refreshes_during_a_silent_tool_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowToolAgent(_Agent):
        async def start_run(self, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            self.starts.append(kwargs)
            yield _event("run.started", 1, {})
            yield _event("tool.started", 2, {"name": "web_search"})
            await asyncio.sleep(0.035)
            yield _event("run.completed", 3, {"text": "done"})

    elapsed = 0

    def _elapsed_label(_: float) -> str:
        nonlocal elapsed
        elapsed += 1
        return f"{elapsed}s"

    monkeypatch.setattr(
        "opentulpa.capability_workers.telegram_worker._STREAM_TYPING_REFRESH_SECONDS",
        0.01,
    )
    monkeypatch.setattr(
        "opentulpa.capability_workers.telegram_worker._format_progress_elapsed",
        _elapsed_label,
    )
    telegram = _Telegram([_message(4, text="hello")])
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=_SlowToolAgent(),
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()

    progress = [
        item["text"]
        for item in [*telegram.messages, *telegram.edits]
        if "Working: Searching the web" in item["text"]
    ]
    assert len(progress) >= 3
    assert len(set(progress)) >= 3
    assert telegram.edits[-1]["text"] == "done"


@pytest.mark.asyncio
async def test_typing_indicator_refreshes_while_agent_has_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowAgent(_Agent):
        async def start_run(self, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            self.starts.append(kwargs)
            yield _event("run.started", 1, {})
            await asyncio.sleep(0.035)
            yield _event("run.completed", 2, {"text": "done"})

    monkeypatch.setattr(
        "opentulpa.capability_workers.telegram_worker._STREAM_TYPING_REFRESH_SECONDS",
        0.01,
    )
    telegram = _Telegram([_message(4, text="hello")])
    state = TelegramWorkerState(tmp_path / "worker.json")
    state.pair(user_id=7, chat_id=9)
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=_SlowAgent(),
        state=state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.poll_once()

    assert len(telegram.actions) >= 3
    assert telegram.messages[-1]["text"] == "done"


@pytest.mark.asyncio
async def test_regenerate_command_routes_through_the_shared_agent_api(tmp_path: Path) -> None:
    telegram = _Telegram([_message(4, text="/regenerate@open_tulpa_bot")])
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

    assert agent.starts[0]["text"] == "/regenerate"
    assert agent.starts[0]["file_ids"] == []
    assert agent.starts[0]["source_event_id"] == "telegram:99:4"


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
        response_message_id=41,
        rendered_text="partial",
    )
    restarted_state = TelegramWorkerState(tmp_path / "worker.json")
    worker = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=restarted_state,
        pairing_code=None,
        poll_timeout_seconds=1,
    )

    await worker.recover()

    assert agent.replays == [{"run_id": "run_1", "after_sequence": 2}]
    assert agent.starts == []
    assert telegram.messages == []
    assert telegram.edits[-1] == {
        "chat_id": 9,
        "message_id": 41,
        "text": "recovered",
        "reply_markup": None,
    }
    assert restarted_state.source_seen("telegram:99:5")
    assert restarted_state.pending_runs() == []


@pytest.mark.asyncio
async def test_recovery_processes_an_update_accepted_before_worker_restart(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "worker.json"
    state = TelegramWorkerState(state_path)
    state.pair(user_id=7, chat_id=9)
    assert state.accept_update(
        update_id=5,
        update=_message(5, text="continue durable work"),
    )

    telegram = _Telegram()
    agent = _Agent()
    restarted = TelegramInterfaceWorker(
        telegram=telegram,
        agent=agent,
        state=TelegramWorkerState(state_path),
        pairing_code=None,
        poll_timeout_seconds=1,
    )
    await restarted.recover()

    assert agent.starts[0]["text"] == "continue durable work"
    assert TelegramWorkerState(state_path).pending_updates() == []


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
