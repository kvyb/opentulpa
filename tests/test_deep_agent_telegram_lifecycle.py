from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.deep_agent.contracts import AgentRunEvent, AgentRunRequest, ApprovalDecision
from opentulpa.interfaces.telegram.deep_agent_relay import DeepAgentTelegramRelay
from opentulpa.interfaces.telegram.state_store import TelegramStateStore


class _Profiles:
    def resolve_customer_id(self, value: str) -> str:
        return value

    def resolve_telegram_customer_id(self, value: int) -> str:
        return f"telegram_{value}"


class _Files:
    def ingest_file(self, **_: Any) -> dict[str, str]:
        return {"id": "file_1"}


class _Client:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        self.messages.append(kwargs)
        return {"ok": True, "result": {"message_id": len(self.messages)}}

    async def edit_message_text(self, **kwargs: Any) -> bool:
        self.edits.append(kwargs)
        return True

    async def delete_message(self, **kwargs: Any) -> bool:
        self.deleted.append(kwargs)
        return True

    async def set_my_commands(self, **_: Any) -> bool:
        return True

    async def download_file(self, **_: Any) -> dict[str, bytes]:
        return {"raw_bytes": b"file"}


class _Agent:
    def __init__(self) -> None:
        self.requests: list[AgentRunRequest] = []
        self.cancelled_threads: list[tuple[str, str]] = []

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield AgentRunEvent("run.started", "run_1", 1, "now", {})
        yield AgentRunEvent("run.completed", "run_1", 2, "now", {"text": request.text})

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        del decision
        yield AgentRunEvent("run.started", run_id, 1, "now", {})
        yield AgentRunEvent("run.completed", run_id, 2, "now", {"text": "resumed"})

    async def cancel_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> Any | None:
        self.cancelled_threads.append((tenant_id, thread_id))
        return None


class _ApprovalAgent(_Agent):
    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield AgentRunEvent("run.started", "run_approval", 1, "now", {})
        yield AgentRunEvent(
            "message.delta",
            "run_approval",
            2,
            "now",
            {"text": "Preparing the change"},
        )
        yield AgentRunEvent(
            "approval.required",
            "run_approval",
            3,
            "now",
            {
                "approval_id": "approval_1",
                "tool_name": "source_release",
                "description": "Release the candidate",
                "allowed_decisions": ["approve", "reject"],
            },
        )


class _BlockingAgent(_Agent):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield AgentRunEvent("run.started", "run_blocked", 1, "now", {})
        yield AgentRunEvent("message.delta", "run_blocked", 2, "now", {"text": "Working"})
        self.started.set()
        await self.release.wait()
        if self.cancelled:
            yield AgentRunEvent(
                "run.failed",
                "run_blocked",
                3,
                "now",
                {"message": "Agent run was cancelled."},
            )
        else:
            yield AgentRunEvent("run.completed", "run_blocked", 3, "now", {"text": "Finished"})

    async def cancel_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> Any:
        self.cancelled_threads.append((tenant_id, thread_id))
        self.cancelled = True
        self.release.set()
        return SimpleNamespace(run_id="run_blocked")


class _ConcurrentAgent(_Agent):
    def __init__(self) -> None:
        super().__init__()
        self.first_started = asyncio.Event()
        self.both_started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        self.first_started.set()
        if len(self.requests) == 2:
            self.both_started.set()
        run_id = f"run_{len(self.requests)}"
        yield AgentRunEvent("run.started", run_id, 1, "now", {})
        await self.release.wait()
        yield AgentRunEvent("run.completed", run_id, 2, "now", {"text": request.text})


class _FailingAgent(_Agent):
    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        raise RuntimeError("worker stopped")
        yield  # pragma: no cover


def _relay(
    tmp_path: Path,
    *,
    agent: _Agent,
    state: TelegramStateStore | None = None,
) -> tuple[DeepAgentTelegramRelay, _Client, TelegramStateStore]:
    client = _Client()
    state = state or TelegramStateStore(tmp_path / "telegram.json")
    relay = DeepAgentTelegramRelay(
        agent=agent,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        state=state,
        profiles=_Profiles(),  # type: ignore[arg-type]
        files=_Files(),  # type: ignore[arg-type]
        bot_token="token",
        owner_tenant_id="tenant-a",
        allowed_user_ids="7",
        allowed_usernames=None,
    )
    return relay, client, state


def _update(update_id: int, *, chat_id: int = 9, text: str = "work") -> dict[str, Any]:
    return {
        "update_id": update_id,
        "message": {
            "chat": {"id": chat_id},
            "from": {"id": 7, "username": "owner"},
            "text": text,
        },
    }


@pytest.mark.asyncio
async def test_approval_replaces_transient_progress_message(tmp_path: Path) -> None:
    relay, client, state = _relay(tmp_path, agent=_ApprovalAgent())

    await relay.handle_update(_update(4, text="release"))

    assert [message["text"] for message in client.messages] == [
        "Preparing the change",
        "Approval required for source_release: Release the candidate",
    ]
    assert client.deleted == [{"chat_id": 9, "message_id": 1}]
    assert len(state.load()["pending_approvals"]) == 1


@pytest.mark.asyncio
async def test_duplicate_owner_processing_claims_once_and_sets_run_idempotency(
    tmp_path: Path,
) -> None:
    agent = _BlockingAgent()
    relay, _, state = _relay(tmp_path, agent=agent)
    accepted = await relay.accept_update(_update(42, text="change source"))

    first = asyncio.create_task(relay.process_update(accepted))
    await agent.started.wait()
    duplicate = asyncio.create_task(relay.process_update(accepted))
    await duplicate

    assert len(agent.requests) == 1
    assert agent.requests[0].idempotency_key == "telegram:update:42"
    assert accepted.ingress_key is not None
    preparation = state.owner_update_preparation(accepted.ingress_key)
    assert preparation is not None
    assert preparation["thread_id"] == agent.requests[0].context.thread_id

    agent.release.set()
    await first
    assert state.owner_update(accepted.ingress_key) is None


@pytest.mark.asyncio
async def test_retry_reuses_durable_run_preparation(tmp_path: Path) -> None:
    failed = _FailingAgent()
    relay, _, state = _relay(tmp_path, agent=failed)
    accepted = await relay.accept_update(_update(43, text="survive restart"))

    with pytest.raises(RuntimeError, match="worker stopped"):
        await relay.process_update(accepted)
    assert accepted.ingress_key is not None
    preparation = state.owner_update_preparation(accepted.ingress_key)
    assert preparation is not None

    recovered = _Agent()
    recovered_relay, _, _ = _relay(tmp_path, agent=recovered, state=state)
    await recovered_relay.process_update(accepted)

    assert len(failed.requests) == len(recovered.requests) == 1
    original = failed.requests[0]
    retry = recovered.requests[0]
    assert retry.idempotency_key == original.idempotency_key
    assert retry.context.thread_id == original.context.thread_id
    assert retry.context.agent_spec == original.context.agent_spec
    assert retry.file_ids == original.file_ids


@pytest.mark.asyncio
async def test_owner_runs_are_concurrent_across_chats_and_ordered_within_chat(
    tmp_path: Path,
) -> None:
    concurrent = _ConcurrentAgent()
    relay, _, _ = _relay(tmp_path, agent=concurrent)

    await relay.accept_update(_update(51, text="first chat"))
    await relay.accept_update(_update(52, chat_id=10, text="second chat"))
    await relay.start()
    try:
        await asyncio.wait_for(concurrent.both_started.wait(), timeout=1)
        concurrent.release.set()
        for _ in range(50):
            if not relay._owner_updates_inflight:  # noqa: SLF001
                break
            await asyncio.sleep(0.01)
    finally:
        await relay.shutdown()
    assert {request.text for request in concurrent.requests} == {"first chat", "second chat"}

    ordered = _ConcurrentAgent()
    ordered_relay, _, _ = _relay(tmp_path / "ordered", agent=ordered)
    first = asyncio.create_task(ordered_relay.handle_update(_update(53, text="first")))
    await ordered.first_started.wait()
    second = asyncio.create_task(ordered_relay.handle_update(_update(54, text="second")))
    await asyncio.sleep(0.02)
    assert [request.text for request in ordered.requests] == ["first"]
    ordered.release.set()
    await asyncio.gather(first, second)
    assert [request.text for request in ordered.requests] == ["first", "second"]


@pytest.mark.asyncio
async def test_cancel_bypasses_active_chat_run_and_cleans_approval_state(tmp_path: Path) -> None:
    agent = _BlockingAgent()
    relay, client, state = _relay(tmp_path, agent=agent)
    running = asyncio.create_task(relay.handle_update(_update(61, text="keep working")))
    await agent.started.wait()
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_blocked",
        approval={"approval_id": "approval_1", "tool_name": "source_release"},
    )

    await asyncio.wait_for(relay.handle_update(_update(62, text="/cancel")), timeout=1)
    await running

    assert len(agent.cancelled_threads) == 1
    assert state.load()["pending_approvals"] == {}
    assert any(message["text"] == "Cancelled the active run." for message in client.messages)
