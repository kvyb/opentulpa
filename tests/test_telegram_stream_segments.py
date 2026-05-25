from __future__ import annotations

import asyncio

import pytest

from opentulpa.agent.runtime import (
    STREAM_PROGRESS_PREFIX,
    STREAM_WAIT_SIGNAL,
)
from opentulpa.interfaces.telegram import relay as relay_module


class _SegmentedRuntime:
    async def astream_text(self, **kwargs):
        yield "I have access to your inbox. I will check it now."
        yield STREAM_WAIT_SIGNAL
        await asyncio.sleep(0.02)
        yield "I checked your inbox. 3 priority emails found."


class _ToolFirstRuntime:
    async def astream_text(self, **kwargs):
        yield STREAM_WAIT_SIGNAL
        await asyncio.sleep(0.02)
        yield "I checked the inbox. 3 priority emails found."


class _TraceContextRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._langfuse_tracer = self

    def trace_context(self, **kwargs):
        self.calls.append(kwargs)
        return relay_module.nullcontext()


class _DelayedSingleReplyRuntime:
    async def astream_text(self, **kwargs):
        await asyncio.sleep(0.4)
        yield "I checked your inbox. 3 priority emails found."


class _ResultThenWaitRuntime:
    async def astream_text(self, **kwargs):
        yield "Here is the finished result."
        yield STREAM_WAIT_SIGNAL


class _FinalOnlyWorkflowSetupRuntime:
    def __init__(self) -> None:
        self.ainvoke_calls: list[dict[str, object]] = []
        self.astream_called = False

    async def astream_text(self, **kwargs):
        self.astream_called = True
        yield "This pre-tool setup narration should not be sent."

    async def ainvoke_text(self, **kwargs):
        self.ainvoke_calls.append(kwargs)
        return "Draft updated. Please confirm this workflow before I save it."


class _ActiveWorkflowSetupService:
    def get_thread_session(self, **kwargs):
        del kwargs
        return {"status": "active"}


class _PromotedWorkflowSetupStreamRuntime:
    workflow_setup_service = _ActiveWorkflowSetupService()

    async def astream_text(self, **kwargs):
        yield "Workflow proposal: ready. Please confirm to activate it."

    async def ainvoke_text(self, **kwargs):
        return "Workflow proposal: ready. Please confirm to activate it."


class _SlowWorkflowSetupRuntime:
    def __init__(self) -> None:
        self.ainvoke_calls: list[dict[str, object]] = []
        self.classifier_calls: list[dict[str, object]] = []
        self.release = asyncio.Event()

    async def astream_text(self, **kwargs):
        yield "This should not stream in workflow setup mode."

    async def ainvoke_text(self, **kwargs):
        self.ainvoke_calls.append(kwargs)
        await self.release.wait()
        return "Workflow proposal: ready. Please confirm to activate it."

    async def classify_workflow_setup_interruption(self, **kwargs):
        self.classifier_calls.append(kwargs)
        return {
            "ok": True,
            "kind": "status_nudge",
            "confidence": 0.99,
            "status_reply": str(kwargs["status"]["reply_if_status_nudge"]),
            "reason": "User asked for progress only.",
        }


class _QueuedWorkflowSetupRuntime:
    def __init__(self) -> None:
        self.ainvoke_calls: list[dict[str, object]] = []
        self.classifier_calls: list[dict[str, object]] = []
        self.release_first = asyncio.Event()

    async def astream_text(self, **kwargs):
        yield "This should not stream in workflow setup mode."

    async def ainvoke_text(self, **kwargs):
        self.ainvoke_calls.append(kwargs)
        if len(self.ainvoke_calls) == 1:
            await self.release_first.wait()
            return "Old workflow proposal: please confirm to activate it."
        return "Updated workflow proposal: includes the new fields. Please confirm to activate it."

    async def classify_workflow_setup_interruption(self, **kwargs):
        self.classifier_calls.append(kwargs)
        return {
            "ok": True,
            "kind": "setup_input",
            "confidence": 0.98,
            "status_reply": "",
            "reason": "Message contains workflow fields.",
        }


class _UpdatingProgressRuntime:
    async def astream_text(self, **kwargs):
        yield f"{STREAM_PROGRESS_PREFIX}Searching the web…"
        await asyncio.sleep(0.01)
        yield f"{STREAM_PROGRESS_PREFIX}Fetching a webpage…"
        await asyncio.sleep(0.01)
        yield "Here is the result."


class _RapidChunkRuntime:
    async def astream_text(self, **kwargs):
        yield "H"
        yield "He"
        yield "Hel"
        yield "Hell"
        yield "Hello"
        yield "Hello "
        yield "Hello w"
        yield "Hello wo"
        yield "Hello wor"
        yield "Hello worl"
        yield "Hello world"


class _WordByWordRuntime:
    async def astream_text(self, **kwargs):
        yield "This is a"
        yield "This is a slightly"
        yield "This is a slightly longer"
        yield "This is a slightly longer streamed"
        yield "This is a slightly longer streamed reply"
        yield "This is a slightly longer streamed reply with"
        yield "This is a slightly longer streamed reply with enough"
        yield "This is a slightly longer streamed reply with enough words."


class _DraftThenLongPauseRuntime:
    async def astream_text(self, **kwargs):
        yield "This first visible draft is long enough to publish immediately."
        await asyncio.sleep(4.2)
        yield "This first visible draft is long enough to publish immediately. And here is the completed follow-up chunk."


class _PacedChunkRuntime:
    async def astream_text(self, **kwargs):
        yield "Chunk one."
        await asyncio.sleep(1.0)
        yield "Chunk one. Chunk two."
        await asyncio.sleep(1.0)
        yield "Chunk one. Chunk two. Chunk three."


class _TaskStableStreamRuntime:
    async def astream_text(self, **kwargs):
        owner_task = asyncio.current_task()
        yield "First stable chunk."
        await asyncio.sleep(0.01)
        assert asyncio.current_task() is owner_task
        yield "First stable chunk. Second stable chunk."


class _AlwaysPendingInteractiveSession:
    async def has_pending_items(self) -> bool:
        return True


class _FakeTelegramClient:
    def __init__(self, bot_token: str, *, draft_ok: bool = True) -> None:
        self.bot_token = bot_token
        self.draft_ok = draft_ok
        self.draft_calls: list[tuple[int | str, int, str, str | None, int | None]] = []
        self.message_calls: list[tuple[int | str, str, str | None]] = []
        self.chat_actions: list[tuple[int | str, str]] = []
        self.deleted_messages: list[tuple[int | str, int]] = []

    async def send_message_draft(
        self,
        *,
        chat_id: int | str,
        draft_id: int,
        text: str,
        message_thread_id: int | None = None,
        parse_mode: str | None = None,
    ) -> bool:
        self.draft_calls.append((chat_id, draft_id, text, parse_mode, message_thread_id))
        return self.draft_ok

    async def send_message(
        self,
        *,
        chat_id: int | str,
        text: str,
        parse_mode: str | None = "HTML",
        reply_markup=None,
    ) -> bool:
        del reply_markup
        self.message_calls.append((chat_id, text, parse_mode))
        return True

    async def send_chat_action(
        self,
        *,
        chat_id: int | str,
        action: str = "typing",
    ) -> bool:
        self.chat_actions.append((chat_id, action))
        return True

    async def delete_message(self, *, chat_id: int | str, message_id: int) -> bool:
        self.deleted_messages.append((chat_id, message_id))
        return True


async def _cancel_workflow_setup_runs() -> None:
    runs = list(relay_module._WORKFLOW_SETUP_RUNS.values())
    relay_module._WORKFLOW_SETUP_RUNS.clear()
    tasks: list[asyncio.Task] = []
    for run in runs:
        tasks.append(run.task)
        if run.delivery_task is not None:
            tasks.append(run.delivery_task)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _wait_for_message(fake_client: _FakeTelegramClient, needle: str) -> None:
    for _ in range(100):
        if any(needle in text for _, text, _ in fake_client.message_calls):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"message containing {needle!r} was not sent")


@pytest.mark.asyncio
async def test_stream_replaces_live_draft_with_final_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_DelayedSingleReplyRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="check inbox",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "priority emails" in str(final or "").lower()
    assert [text for _, _, text, _, _ in fake_client.draft_calls] == [
        "I checked your inbox. 3 priority emails found."
    ]
    assert len({draft_id for _, draft_id, _, _, _ in fake_client.draft_calls}) == 1
    assert fake_client.message_calls == [(1, "I checked your inbox. 3 priority emails found.", "HTML")]
    assert fake_client.chat_actions
    assert not fake_client.deleted_messages


@pytest.mark.asyncio
async def test_wait_signal_does_not_emit_visible_progress_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ToolFirstRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="check inbox",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "priority emails" in str(final or "").lower()
    assert not any("working on it" in text.lower() for _, _, text, _, _ in fake_client.draft_calls)
    assert fake_client.message_calls == [(1, "I checked the inbox. 3 priority emails found.", "HTML")]


@pytest.mark.asyncio
async def test_workflow_setup_turn_uses_final_reply_without_draft_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    runtime = _FinalOnlyWorkflowSetupRuntime()
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=runtime,
        thread_id="chat-setup",
        customer_id="telegram_setup",
        text="update workflow draft",
        bot_token="dummy",
        chat_id=1,
        turn_mode="workflow_setup",
    )

    assert suppressed is False
    assert final == "Draft updated. Please confirm this workflow before I save it."
    assert runtime.astream_called is False
    assert runtime.ainvoke_calls[0]["turn_mode"] == "workflow_setup"
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [
        (1, "Draft updated. Please confirm this workflow before I save it.", "HTML")
    ]


@pytest.mark.asyncio
async def test_workflow_setup_final_reply_is_not_suppressed_by_interactive_pending_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    runtime = _FinalOnlyWorkflowSetupRuntime()
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=runtime,
        thread_id="chat-setup-pending",
        customer_id="telegram_setup_pending",
        text="update workflow draft",
        bot_token="dummy",
        chat_id=1,
        turn_mode="workflow_setup",
        interactive_session=_AlwaysPendingInteractiveSession(),
    )

    assert suppressed is False
    assert final == "Draft updated. Please confirm this workflow before I save it."
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [
        (1, "Draft updated. Please confirm this workflow before I save it.", "HTML")
    ]


@pytest.mark.asyncio
async def test_promoted_workflow_setup_final_reply_is_not_suppressed_by_pending_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    runtime = _PromotedWorkflowSetupStreamRuntime()
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=runtime,
        thread_id="chat-promoted-setup",
        customer_id="telegram_promoted_setup",
        text="add these workflow fields",
        bot_token="dummy",
        chat_id=1,
        turn_mode="interactive",
        interactive_session=_AlwaysPendingInteractiveSession(),
    )

    assert suppressed is False
    assert final == "Workflow proposal: ready. Please confirm to activate it."
    assert fake_client.message_calls == [
        (1, "Workflow proposal: ready. Please confirm to activate it.", "HTML")
    ]


@pytest.mark.asyncio
async def test_slow_workflow_setup_turn_backgrounds_and_status_nudges_do_not_duplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _cancel_workflow_setup_runs()
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    runtime = _SlowWorkflowSetupRuntime()
    delivered_replies: list[str] = []
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "WORKFLOW_SETUP_FINAL_REPLY_TIMEOUT_SECONDS", 0.01)

    try:
        final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=runtime,
            thread_id="chat-setup-slow",
            customer_id="telegram_setup_slow",
            text="build workflow",
            bot_token="dummy",
            chat_id=1,
            turn_mode="workflow_setup",
            final_reply_callback=delivered_replies.append,
        )
        follow_up, follow_up_suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=runtime,
            thread_id="chat-setup-slow",
            customer_id="telegram_setup_slow",
            text="so what's up?",
            bot_token="dummy",
            chat_id=1,
            turn_mode="workflow_setup",
        )

        assert suppressed is False
        assert follow_up_suppressed is False
        assert final == relay_module.WORKFLOW_SETUP_BUSY_REPLY
        assert follow_up == relay_module.WORKFLOW_SETUP_BUSY_REPLY
        assert len(runtime.ainvoke_calls) == 1
        assert len(runtime.classifier_calls) == 1
        assert runtime.classifier_calls[0]["status"]["state"] == "workflow_setup_running"
        assert runtime.classifier_calls[0]["user_text"] == "so what's up?"

        runtime.release.set()
        await _wait_for_message(fake_client, "Workflow proposal: ready.")

        assert len(runtime.ainvoke_calls) == 1
        assert fake_client.message_calls[-1] == (
            1,
            "Workflow proposal: ready. Please confirm to activate it.",
            "HTML",
        )
        assert delivered_replies == ["Workflow proposal: ready. Please confirm to activate it."]
    finally:
        await _cancel_workflow_setup_runs()


@pytest.mark.asyncio
async def test_slow_workflow_setup_turn_applies_substantive_follow_up_before_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _cancel_workflow_setup_runs()
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    runtime = _QueuedWorkflowSetupRuntime()
    delivered_replies: list[str] = []
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)
    monkeypatch.setattr(relay_module, "WORKFLOW_SETUP_FINAL_REPLY_TIMEOUT_SECONDS", 0.01)

    try:
        final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=runtime,
            thread_id="chat-setup-queued",
            customer_id="telegram_setup_queued",
            text="build workflow",
            bot_token="dummy",
            chat_id=1,
            turn_mode="workflow_setup",
            final_reply_callback=delivered_replies.append,
        )
        queued, queued_suppressed = await relay_module.stream_langgraph_reply_to_telegram(
            agent_runtime=runtime,
            thread_id="chat-setup-queued",
            customer_id="telegram_setup_queued",
            text="Required fields: client, phone, service, day, time.",
            bot_token="dummy",
            chat_id=1,
            turn_mode="workflow_setup",
        )

        assert suppressed is False
        assert queued_suppressed is False
        assert final == relay_module.WORKFLOW_SETUP_BUSY_REPLY
        assert queued == relay_module.WORKFLOW_SETUP_QUEUED_REPLY
        assert len(runtime.ainvoke_calls) == 1
        assert len(runtime.classifier_calls) == 1
        assert runtime.classifier_calls[0]["status"]["state"] == "workflow_setup_running"

        runtime.release_first.set()
        await _wait_for_message(fake_client, "Updated workflow proposal:")

        assert len(runtime.ainvoke_calls) == 2
        assert "Required fields: client" in str(runtime.ainvoke_calls[1]["text"])
        assert not any("Old workflow proposal" in text for _, text, _ in fake_client.message_calls)
        assert delivered_replies == [
            "Updated workflow proposal: includes the new fields. Please confirm to activate it."
        ]
    finally:
        await _cancel_workflow_setup_runs()


@pytest.mark.asyncio
async def test_progress_signals_stay_in_typing_only_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_UpdatingProgressRuntime(),
        thread_id="chat-1",
        customer_id="telegram_1",
        text="search",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Here is the result."
    assert not any(
        "searching the web" in text.lower() for _, _, text, _, _ in fake_client.draft_calls
    )
    assert not any(
        "fetching a webpage" in text.lower() for _, _, text, _, _ in fake_client.draft_calls
    )
    assert fake_client.message_calls == [(1, "Here is the result.", "HTML")]


@pytest.mark.asyncio
async def test_stream_coalesces_rapid_partial_updates_until_final_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_RapidChunkRuntime(),
        thread_id="chat-rapid",
        customer_id="telegram_rapid",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Hello world"
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [(1, "Hello world", "HTML")]


@pytest.mark.asyncio
async def test_stream_paces_draft_updates_by_time_not_by_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_PacedChunkRuntime(),
        thread_id="chat-wordy",
        customer_id="telegram_wordy",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Chunk one. Chunk two. Chunk three."
    assert [text for _, _, text, _, _ in fake_client.draft_calls] == [
        "Chunk one. Chunk two.",
        "Chunk one. Chunk two. Chunk three.",
    ]
    assert fake_client.message_calls == [(1, "Chunk one. Chunk two. Chunk three.", "HTML")]


@pytest.mark.asyncio
async def test_stream_drives_runtime_generator_from_one_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_TaskStableStreamRuntime(),
        thread_id="chat-stable",
        customer_id="telegram_stable",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "First stable chunk. Second stable chunk."
    assert fake_client.message_calls == [(1, "First stable chunk. Second stable chunk.", "HTML")]


@pytest.mark.asyncio
async def test_draft_failure_falls_back_to_typing_and_final_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ResultThenWaitRuntime(),
        thread_id="chat-fallback",
        customer_id="telegram_fallback",
        text="finish",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert final == "Here is the finished result."
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [(1, "Here is the finished result.", "HTML")]
    assert fake_client.chat_actions


@pytest.mark.asyncio
async def test_successful_draft_stream_stops_typing_loop_after_first_publish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_DraftThenLongPauseRuntime(),
        thread_id="chat-draft-stop",
        customer_id="telegram_draft_stop",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "completed follow-up chunk" in str(final or "")
    assert fake_client.draft_calls
    assert 1 <= len(fake_client.chat_actions) <= 2
    assert fake_client.message_calls == [
        (
            1,
            "This first visible draft is long enough to publish immediately. "
            "And here is the completed follow-up chunk.",
            "HTML",
        )
    ]


@pytest.mark.asyncio
async def test_failed_draft_stream_also_stops_typing_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy", draft_ok=False)
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_DraftThenLongPauseRuntime(),
        thread_id="chat-draft-fail-stop",
        customer_id="telegram_draft_fail_stop",
        text="hello",
        bot_token="dummy",
        chat_id=1,
    )

    assert suppressed is False
    assert "completed follow-up chunk" in str(final or "")
    assert fake_client.draft_calls
    assert fake_client.message_calls == [
        (
            1,
            "This first visible draft is long enough to publish immediately. "
            "And here is the completed follow-up chunk.",
            "HTML",
        )
    ]
    assert 1 <= len(fake_client.chat_actions) <= 2


@pytest.mark.asyncio
async def test_non_private_chat_bypasses_draft_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ResultThenWaitRuntime(),
        thread_id="chat-group",
        customer_id="telegram_group",
        text="finish",
        bot_token="dummy",
        chat_id=-100123456,
    )

    assert suppressed is False
    assert final == "Here is the finished result."
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == [(-100123456, "Here is the finished result.", "HTML")]


def test_telegram_observability_context_maps_customer_and_thread_to_langfuse_kwargs() -> None:
    runtime = _TraceContextRuntime()

    with relay_module._telegram_observability_context(
        agent_runtime=runtime,
        thread_id="chat_1",
        customer_id="telegram_1",
        text="hello",
        chat_id=123,
        turn_mode="interactive",
    ):
        pass

    assert runtime.calls == [
        {
            "name": "opentulpa.interactive.turn",
            "trace_id": None,
            "user_id": "telegram_1",
            "session_id": "chat_1",
            "input": {"text": "hello", "chat_id": 123, "mode": "telegram"},
            "metadata": {"turn_mode": "interactive", "chat_id": 123},
            "tags": ["interactive", "telegram"],
        }
    ]



@pytest.mark.asyncio
async def test_interactive_pending_items_suppress_final_visible_send(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeTelegramClient("dummy")
    monkeypatch.setattr(relay_module, "TelegramClient", lambda token: fake_client)

    final, suppressed = await relay_module.stream_langgraph_reply_to_telegram(
        agent_runtime=_ResultThenWaitRuntime(),
        thread_id="chat-interactive-pending",
        customer_id="telegram_interactive_pending",
        text="finish",
        bot_token="dummy",
        chat_id=1,
        interactive_session=_AlwaysPendingInteractiveSession(),
    )

    assert suppressed is True
    assert final is None
    assert fake_client.draft_calls == []
    assert fake_client.message_calls == []
