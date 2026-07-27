from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from opentulpa.deep_agent.contracts import (
    AgentApproval,
    AgentRunEvent,
    AgentRunRequest,
    ApprovalDecision,
)
from opentulpa.inference import InferenceModel
from opentulpa.interfaces.telegram.business import TelegramBusinessService
from opentulpa.interfaces.telegram.deep_agent_relay import DeepAgentTelegramRelay
from opentulpa.interfaces.telegram.delivery import TelegramOwnerDelivery
from opentulpa.interfaces.telegram.state_store import TelegramStateStore


class _Profiles:
    def resolve_customer_id(self, value: str) -> str:
        return value

    def resolve_telegram_customer_id(self, value: int) -> str:
        return f"telegram_{value}"


class _Files:
    def ingest_file(self, **kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"id": "file_1"}


class _Client:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.edits: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self.commands: list[dict[str, str]] = []
        self.callbacks: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any) -> dict[str, Any]:
        self.messages.append(kwargs)
        return {"ok": True, "result": {"message_id": len(self.messages)}}

    async def edit_message_text(self, **kwargs: Any) -> bool:
        self.edits.append(kwargs)
        return True

    async def set_my_commands(self, *, commands: list[dict[str, str]]) -> bool:
        self.commands = commands
        return True

    async def delete_message(self, **kwargs: Any) -> bool:
        self.deleted.append(kwargs)
        return True

    async def answer_callback_query(self, **kwargs: Any) -> bool:
        self.callbacks.append(kwargs)
        return True

    async def download_file(self, *, file_id: str) -> dict[str, Any]:
        del file_id
        return {"raw_bytes": b"hello"}


class _Agent:
    def __init__(self) -> None:
        self.requests: list[AgentRunRequest] = []
        self.decisions: list[tuple[str, ApprovalDecision]] = []
        self.threads: set[tuple[str, str]] = set()
        self.inference = {
            "revision": 0,
            "selection": None,
            "effective": {
                "provider": "api",
                "model": "api/default",
                "reasoning_effort": "low",
                "service_tier": None,
                "fallback_to_api": False,
            },
        }

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield AgentRunEvent("run.started", "run_1", 1, "now", {})
        yield AgentRunEvent("message.delta", "run_1", 2, "now", {"text": "hello"})
        yield AgentRunEvent("run.completed", "run_1", 3, "now", {"text": "hello"})

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        self.decisions.append((run_id, decision))
        yield AgentRunEvent("run.started", run_id, 3, "now", {"resumed": True})
        yield AgentRunEvent("run.completed", run_id, 4, "now", {"text": "approved"})

    async def ensure_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        channel: str,
    ) -> None:
        assert channel == "telegram"
        self.threads.add((tenant_id, thread_id))

    async def get_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        return self.inference if (tenant_id, thread_id) in self.threads else None

    async def update_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        expected_revision: int,
        selection: Any,
    ) -> dict[str, Any] | None:
        if (tenant_id, thread_id) not in self.threads:
            return None
        assert expected_revision == self.inference["revision"]
        self.inference = {
            "revision": expected_revision + 1,
            "selection": selection.model_dump(mode="json"),
            "effective": selection.model_dump(mode="json"),
        }
        return self.inference

class _Inference:
    def __init__(self) -> None:
        self.connected = False

    def codex_connected(self, tenant_id: str) -> bool:
        assert tenant_id == "tenant-a"
        return self.connected

    async def status(self, tenant_id: str) -> dict[str, Any]:
        return {"codex": {"connected": self.codex_connected(tenant_id)}}

    async def models(
        self,
        tenant_id: str,
        provider: str,
        *,
        query: str = "",
    ) -> tuple[InferenceModel, ...]:
        assert tenant_id == "tenant-a"
        model = InferenceModel(
            provider=provider,
            id="gpt-5.3-codex" if provider == "codex" else "api/default",
            reasoning_efforts=("low", "high", "ultra"),
            default_reasoning_effort="low",
        )
        return (model,) if not query or query in model.id else ()

    async def start_device_login(self, tenant_id: str) -> dict[str, Any]:
        assert tenant_id == "tenant-a"
        return {
            "id": "login_1",
            "status": "pending",
            "verification_url": "https://auth.openai.com/codex/device",
            "user_code": "ABCD-EFGH",
        }

    async def get_device_login(
        self,
        tenant_id: str,
        login_id: str,
    ) -> dict[str, Any] | None:
        assert tenant_id == "tenant-a"
        assert login_id == "login_1"
        self.connected = True
        return {"id": login_id, "status": "authorized"}


class _GatedAgent(_Agent):
    def __init__(self) -> None:
        super().__init__()
        self.resume_entered = asyncio.Event()
        self.resume_accepted = asyncio.Event()

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        self.decisions.append((run_id, decision))
        self.resume_entered.set()
        await self.resume_accepted.wait()
        yield AgentRunEvent("run.started", run_id, 3, "now", {"resumed": True})
        yield AgentRunEvent("run.completed", run_id, 4, "now", {"text": "approved"})


class _ProgressAgent(_Agent):
    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        self.requests.append(request)
        yield AgentRunEvent("run.started", "run_1", 1, "now", {})
        yield AgentRunEvent("message.delta", "run_1", 2, "now", {"text": "Inspecting"})
        yield AgentRunEvent(
            "tool.started",
            "run_1",
            3,
            "now",
            {"name": "source_shell"},
        )
        await asyncio.sleep(0.02)
        yield AgentRunEvent(
            "tool.completed",
            "run_1",
            4,
            "now",
            {"name": "source_shell"},
        )
        yield AgentRunEvent("message.delta", "run_1", 5, "now", {"text": "Finished"})
        yield AgentRunEvent("run.completed", "run_1", 6, "now", {"text": "All done"})


class _FailOnceAgent(_Agent):
    def __init__(self) -> None:
        super().__init__()
        self.resume_attempts = 0

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        self.resume_attempts += 1
        if self.resume_attempts == 1:
            raise sqlite3.OperationalError("database is locked")
        self.decisions.append((run_id, decision))
        yield AgentRunEvent("run.started", run_id, 3, "now", {"resumed": True})
        yield AgentRunEvent("run.completed", run_id, 4, "now", {"text": "approved"})


class _IntakeWorkflows:
    def __init__(self) -> None:
        self.fail_enqueue = False
        self.enqueued: list[dict[str, Any]] = []
        self.drains = 0

    def list_workflows(
        self,
        *,
        customer_id: str,
        include_disabled: bool,
    ) -> list[dict[str, Any]]:
        assert customer_id == "tenant-a"
        assert include_disabled is False
        return [
            {
                "workflow_id": "workflow-1",
                "channel": "telegram_business_dm",
                "provider": "telegram_bot_api",
            }
        ]

    def _source_matches_workflow(self, **_: Any) -> bool:
        return True

    async def enqueue_telegram_business_workflow_run(self, **kwargs: Any) -> dict[str, Any]:
        self.enqueued.append(kwargs)
        if self.fail_enqueue:
            return {
                "ok": False,
                "summary": "provider body token=private-secret /srv/private/.env",
            }
        return {"ok": True, "queued": True}

    async def drain_due_pending_runs(self, *, limit: int) -> int:
        assert limit == 10
        self.drains += 1
        return 0


def _relay(
    tmp_path: Path,
    *,
    agent: _Agent | None = None,
    owner_tenant_id: str | None = "tenant-a",
    allowed_user_ids: str = "7",
    inference: _Inference | None = None,
) -> tuple[DeepAgentTelegramRelay, _Agent, _Client, TelegramStateStore]:
    agent = agent or _Agent()
    client = _Client()
    state = TelegramStateStore(tmp_path / "telegram.json")
    relay = DeepAgentTelegramRelay(
        agent=agent,
        client=client,  # type: ignore[arg-type]
        state=state,
        profiles=_Profiles(),  # type: ignore[arg-type]
        files=_Files(),  # type: ignore[arg-type]
        bot_token="token",
        owner_tenant_id=owner_tenant_id,
        allowed_user_ids=allowed_user_ids,
        allowed_usernames=None,
        inference=inference,
    )
    return relay, agent, client, state


def test_edited_arguments_reject_exponent_overflow() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        DeepAgentTelegramRelay._parse_edited_arguments('{"amount":1e10000}')  # noqa: SLF001


@pytest.mark.asyncio
async def test_health_tracks_durable_owner_inbox_dispatcher(tmp_path: Path) -> None:
    relay, _, client, _ = _relay(tmp_path)

    assert relay.healthy() is False
    await relay.start()
    assert relay.healthy() is True
    assert {command["command"] for command in client.commands} == {
        "fresh",
        "model",
        "models",
        "reasoning",
        "codex",
        "cancel",
    }
    await relay.shutdown()
    assert relay.healthy() is False


@pytest.mark.asyncio
async def test_owner_update_injects_tenant_context_and_reuses_thread(tmp_path: Path) -> None:
    relay, agent, client, _ = _relay(tmp_path)
    body = {
        "update_id": 1,
        "message": {
            "chat": {"id": 9},
            "from": {"id": 7, "username": "owner"},
            "text": "hello",
        }
    }
    await relay.handle_update(body)
    await relay.handle_update({**body, "update_id": 2})

    assert [request.context.tenant_id for request in agent.requests] == ["tenant-a", "tenant-a"]
    assert agent.requests[0].context.thread_id == agent.requests[1].context.thread_id
    assert agent.requests[0].context.actor_id == "telegram:7"
    assert [message["text"] for message in client.messages] == [
        "hello",
        "hello",
        "hello",
        "hello",
    ]
    assert [item["message_id"] for item in client.deleted] == [1, 3]


@pytest.mark.asyncio
async def test_owner_progress_edits_one_message_without_concatenating_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "opentulpa.interfaces.telegram.run_output._EDIT_MIN_INTERVAL_SECONDS",
        0.01,
    )
    relay, _, client, _ = _relay(tmp_path, agent=_ProgressAgent())

    await relay.handle_update(
        {
            "update_id": 1,
            "message": {
                "chat": {"id": 9},
                "from": {"id": 7, "username": "owner"},
                "text": "change the source",
            },
        }
    )

    assert [message["text"] for message in client.messages] == ["Inspecting", "All done"]
    assert client.edits[0]["text"] == (
        "Inspecting\n\nWorking: Working on OpenTulpa source..."
    )
    assert client.deleted == [{"chat_id": 9, "message_id": 1}]
    assert all("InspectingFinished" not in edit["text"] for edit in client.edits)


@pytest.mark.asyncio
async def test_telegram_codex_login_model_and_reasoning_commands(tmp_path: Path) -> None:
    inference = _Inference()
    relay, agent, client, _ = _relay(tmp_path, inference=inference)

    async def send(update_id: int, text: str) -> None:
        await relay.handle_update(
            {
                "update_id": update_id,
                "message": {
                    "chat": {"id": 9},
                    "from": {"id": 7, "username": "owner"},
                    "text": text,
                },
            }
        )

    await send(1, "/codex login")
    assert "ABCD-EFGH" in client.messages[-1]["text"]
    await send(2, "/codex status")
    assert client.messages[-1]["text"].startswith("Codex is connected")
    await send(3, "/models codex")
    assert "gpt-5.3-codex" in client.messages[-1]["text"]
    await send(4, "/model codex gpt-5.3-codex high")
    assert agent.inference["effective"]["provider"] == "codex"
    assert agent.inference["effective"]["reasoning_effort"] == "high"
    await send(5, "/reasoning ultra")
    assert agent.inference["effective"]["reasoning_effort"] == "ultra"
    assert agent.requests == []


@pytest.mark.asyncio
async def test_owner_update_recovers_from_durable_inbox_after_restart(tmp_path: Path) -> None:
    relay, _, _, state = _relay(tmp_path)
    accepted = await relay.accept_update(
        {
            "update_id": 17,
            "message": {
                "chat": {"id": 9},
                "from": {"id": 7, "username": "owner"},
                "text": "recover me",
            },
        }
    )
    assert accepted.ingress_key is not None
    assert state.owner_update(accepted.ingress_key) is not None

    recovered, agent, client, recovered_state = _relay(tmp_path)
    await recovered.start()
    try:
        for _ in range(50):
            if agent.requests:
                break
            await asyncio.sleep(0.02)
    finally:
        await recovered.shutdown()

    assert len(agent.requests) == 1
    assert client.messages[-1]["text"] == "hello"
    assert recovered_state.owner_update(accepted.ingress_key) is None


@pytest.mark.asyncio
async def test_approval_callback_is_bound_to_tenant_and_chat(tmp_path: Path) -> None:
    relay, agent, client, state = _relay(tmp_path)
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_1",
        approval={
            "approval_id": "approval_1",
            "tool_name": "integration_invoke",
            "description": "Send email",
        },
    )
    token = next(iter(state.load()["pending_approvals"]))
    await relay.handle_update(
        {
            "callback_query": {
                "id": "callback_1",
                "data": f"ot:{token}:approve",
                "from": {"id": 7, "username": "owner"},
                "message": {"chat": {"id": 9}},
            }
        }
    )

    assert agent.decisions == [
        ("run_1", ApprovalDecision(approval_id="approval_1", decision="approve"))
    ]
    assert client.callbacks[-1]["text"] == "Approved"
    assert client.messages[-1]["text"] == "approved"


@pytest.mark.asyncio
async def test_scheduled_approval_uses_relay_buttons_and_resume_path(tmp_path: Path) -> None:
    relay, agent, client, state = _relay(tmp_path)
    relay._reset_session(  # noqa: SLF001
        chat_id=9,
        user_id=7,
        username="owner",
        tenant_id="tenant-a",
    )
    approval = AgentApproval(
        id="approval_1",
        tool_name="integration_invoke",
        description="Send the report",
        arguments={"recipient": "owner@example.com"},
        allowed_decisions=("approve", "edit", "reject"),
    )
    delivery = TelegramOwnerDelivery(
        client=client,  # type: ignore[arg-type]
        state=state,
        deliver_approval=relay.deliver_approval,
    )

    await delivery.deliver_text(
        tenant_id="tenant-a",
        title="Daily report",
        text="Scheduled agent job is waiting for owner approval.",
        run_id="run_1",
        approval_required=True,
        approvals=(approval,),
    )

    assert agent.decisions == []
    assert client.messages[0]["text"] == (
        "Daily report\n\nScheduled agent job is waiting for owner approval."
    )
    buttons = client.messages[1]["reply_markup"]["inline_keyboard"][0]
    assert [button["text"] for button in buttons] == ["Approve", "Edit", "Reject"]
    token = next(iter(state.load()["pending_approvals"]))

    await relay.handle_update(
        {
            "callback_query": {
                "id": "scheduled_approval",
                "data": f"ot:{token}:approve",
                "from": {"id": 7, "username": "owner"},
                "message": {"chat": {"id": 9}},
            }
        }
    )

    assert agent.decisions == [
        ("run_1", ApprovalDecision(approval_id="approval_1", decision="approve"))
    ]
    assert token not in state.load()["pending_approvals"]


@pytest.mark.asyncio
async def test_approval_edit_accepts_safe_json_without_echoing_arguments(tmp_path: Path) -> None:
    relay, agent, client, state = _relay(tmp_path)
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_1",
        approval={
            "approval_id": "approval_1",
            "tool_name": "integration_invoke",
            "description": "Send email",
            "allowed_decisions": ["approve", "edit", "reject"],
        },
    )
    token = next(iter(state.load()["pending_approvals"]))
    buttons = client.messages[-1]["reply_markup"]["inline_keyboard"][0]
    assert [button["text"] for button in buttons] == ["Approve", "Edit", "Reject"]

    await relay.handle_update(
        {
            "callback_query": {
                "id": "callback_edit",
                "data": f"ot:{token}:edit",
                "from": {"id": 7, "username": "owner"},
                "message": {"chat": {"id": 9}},
            }
        }
    )

    assert agent.decisions == []
    assert token in state.load()["pending_approvals"]
    assert client.callbacks[-1]["text"] == "Send the replacement arguments as JSON."

    await relay.handle_update(
        {
            "message": {
                "chat": {"id": 9},
                "from": {"id": 7, "username": "owner"},
                "text": '{"tenant_id":"do-not-echo"}',
            }
        }
    )
    assert agent.decisions == []
    assert all("do-not-echo" not in message["text"] for message in client.messages)

    await relay.handle_update(
        {
            "message": {
                "chat": {"id": 9},
                "from": {"id": 7, "username": "owner"},
                "text": '{"recipient":"other@example.com"}',
            }
        }
    )

    assert agent.decisions == [
        (
            "run_1",
            ApprovalDecision(
                approval_id="approval_1",
                decision="edit",
                edited_arguments={"recipient": "other@example.com"},
            ),
        )
    ]
    final_state = state.load()
    assert token not in final_state["pending_approvals"]
    assert final_state.get("pending_approval_edits") == {}
    assert all("other@example.com" not in message["text"] for message in client.messages)


@pytest.mark.asyncio
async def test_mismatched_callback_never_consumes_approval(tmp_path: Path) -> None:
    relay, agent, client, state = _relay(
        tmp_path,
        owner_tenant_id=None,
        allowed_user_ids="7,8",
    )
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="telegram_7",
        run_id="run_1",
        approval={"approval_id": "approval_1", "tool_name": "job_cancel"},
    )
    token = next(iter(state.load()["pending_approvals"]))

    for callback_id, user_id, chat_id in (
        ("wrong_tenant", 8, 9),
        ("wrong_chat", 7, 10),
    ):
        await relay.handle_update(
            {
                "callback_query": {
                    "id": callback_id,
                    "data": f"ot:{token}:approve",
                    "from": {"id": user_id, "username": "owner"},
                    "message": {"chat": {"id": chat_id}},
                }
            }
        )
        assert token in state.load()["pending_approvals"]
        assert agent.decisions == []
        assert client.callbacks[-1]["text"] == "Approval not found."

    await relay.handle_update(
        {
            "callback_query": {
                "id": "valid",
                "data": f"ot:{token}:approve",
                "from": {"id": 7, "username": "owner"},
                "message": {"chat": {"id": 9}},
            }
        }
    )
    assert len(agent.decisions) == 1
    assert token not in state.load()["pending_approvals"]


@pytest.mark.asyncio
async def test_callback_is_not_approved_until_resume_is_durably_accepted(tmp_path: Path) -> None:
    gated = _GatedAgent()
    relay, _, client, state = _relay(tmp_path, agent=gated)
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_1",
        approval={"approval_id": "approval_1", "tool_name": "job_cancel"},
    )
    token = next(iter(state.load()["pending_approvals"]))
    callback = {
        "callback_query": {
            "id": "callback_1",
            "data": f"ot:{token}:approve",
            "from": {"id": 7, "username": "owner"},
            "message": {"chat": {"id": 9}},
        }
    }

    task = asyncio.create_task(relay.handle_update(callback))
    await gated.resume_entered.wait()
    assert client.callbacks == []
    assert token in state.load()["pending_approvals"]

    gated.resume_accepted.set()
    await task
    assert client.callbacks[-1]["text"] == "Approved"
    assert token not in state.load()["pending_approvals"]


@pytest.mark.asyncio
async def test_concurrent_callbacks_claim_approval_once(tmp_path: Path) -> None:
    relay, agent, client, state = _relay(tmp_path)
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_1",
        approval={"approval_id": "approval_1", "tool_name": "job_cancel"},
    )
    token = next(iter(state.load()["pending_approvals"]))

    def callback(callback_id: str) -> dict[str, Any]:
        return {
            "callback_query": {
                "id": callback_id,
                "data": f"ot:{token}:approve",
                "from": {"id": 7, "username": "owner"},
                "message": {"chat": {"id": 9}},
            }
        }

    await asyncio.gather(
        relay.handle_update(callback("callback_1")),
        relay.handle_update(callback("callback_2")),
    )

    assert len(agent.decisions) == 1
    assert token not in state.load()["pending_approvals"]
    assert sorted(item["text"] for item in client.callbacks) == ["Approval not found.", "Approved"]


@pytest.mark.asyncio
async def test_resume_db_failure_leaves_callback_retryable(tmp_path: Path) -> None:
    flaky = _FailOnceAgent()
    relay, _, client, state = _relay(tmp_path, agent=flaky)
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_1",
        approval={"approval_id": "approval_1", "tool_name": "job_cancel"},
    )
    token = next(iter(state.load()["pending_approvals"]))
    callback = {
        "callback_query": {
            "id": "callback_1",
            "data": f"ot:{token}:approve",
            "from": {"id": 7, "username": "owner"},
            "message": {"chat": {"id": 9}},
        }
    }

    await relay.handle_update(callback)
    assert client.callbacks[-1]["text"] == "Could not resume. Try again."
    assert token in state.load()["pending_approvals"]
    assert flaky.decisions == []

    callback["callback_query"]["id"] = "callback_2"
    await relay.handle_update(callback)
    assert client.callbacks[-1]["text"] == "Approved"
    assert token not in state.load()["pending_approvals"]
    assert len(flaky.decisions) == 1


@pytest.mark.asyncio
async def test_accepted_resume_is_not_repeated_when_handle_cleanup_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relay, agent, client, state = _relay(tmp_path)
    await relay._send_approval(  # noqa: SLF001
        chat_id=9,
        tenant_id="tenant-a",
        run_id="run_1",
        approval={"approval_id": "approval_1", "tool_name": "job_cancel"},
    )
    token = next(iter(state.load()["pending_approvals"]))
    original_update = state.update
    failed = False

    def fail_first_consume(mutator: Any) -> Any:
        nonlocal failed
        if not failed and getattr(mutator, "__name__", "") == "consume":
            failed = True
            raise sqlite3.OperationalError("state database is locked")
        return original_update(mutator)

    monkeypatch.setattr(state, "update", fail_first_consume)

    def callback(callback_id: str) -> dict[str, Any]:
        return {
            "callback_query": {
                "id": callback_id,
                "data": f"ot:{token}:approve",
                "from": {"id": 7, "username": "owner"},
                "message": {"chat": {"id": 9}},
            }
        }

    await relay.handle_update(callback("callback_1"))
    assert client.callbacks[-1]["text"] == "Approved"
    assert token in state.load()["pending_approvals"]
    assert len(agent.decisions) == 1

    await relay.handle_update(callback("callback_2"))
    assert client.callbacks[-1]["text"] == "Already approved"
    assert token not in state.load()["pending_approvals"]
    assert len(agent.decisions) == 1


@pytest.mark.asyncio
async def test_business_ingress_retries_pending_dispatch_without_duplicate_queue(
    tmp_path: Path,
) -> None:
    agent = _Agent()
    client = _Client()
    business = TelegramBusinessService(
        db_path=tmp_path / "telegram-business.db",
        owner_customer_id="tenant-a",
    )
    business.upsert_connection(
        {
            "id": "bc_123",
            "user_chat_id": 777,
            "is_enabled": True,
            "user": {"id": 123, "is_bot": False},
            "rights": {"can_reply": True},
        }
    )
    intake = _IntakeWorkflows()
    relay = DeepAgentTelegramRelay(
        agent=agent,
        client=client,  # type: ignore[arg-type]
        state=TelegramStateStore(tmp_path / "telegram.json"),
        profiles=_Profiles(),  # type: ignore[arg-type]
        files=_Files(),  # type: ignore[arg-type]
        bot_token="token",
        owner_tenant_id="tenant-a",
        allowed_user_ids="7",
        allowed_usernames=None,
        telegram_business=business,
        intake_workflows=intake,
    )
    body = {
        "update_id": 1001,
        "business_message": {
            "business_connection_id": "bc_123",
            "message_id": 10,
            "date": 1_775_552_400,
            "chat": {"id": 555, "type": "private"},
            "from": {"id": 999, "is_bot": False},
            "text": "Can I book?",
        },
    }

    intake.fail_enqueue = True
    with pytest.raises(RuntimeError, match="could not be queued"):
        await relay.accept_update(body)
    pending = business.ingest_update(body)
    assert pending["duplicate"] is True
    assert pending["dispatch_pending"] is True

    intake.fail_enqueue = False
    accepted = await relay.accept_update(body)
    assert accepted.business_result is not None
    assert accepted.business_result["dispatch_pending"] is False
    queue_count = len(intake.enqueued)
    await relay.accept_update(body)
    assert len(intake.enqueued) == queue_count
    await relay.process_update(accepted)
    assert intake.drains == 1
    conversations = business.list_conversations(
        customer_id="tenant-a",
        business_connection_id="bc_123",
    )
    assert len(conversations["items"]) == 1


@pytest.mark.asyncio
async def test_owner_chat_never_enters_business_ingestion_twice(tmp_path: Path) -> None:
    class CountingBusiness:
        def __init__(self) -> None:
            self.calls = 0

        def ingest_update(self, body: dict[str, Any]) -> dict[str, Any]:
            del body
            self.calls += 1
            return {"handled": False}

    agent = _Agent()
    client = _Client()
    business = CountingBusiness()
    relay = DeepAgentTelegramRelay(
        agent=agent,
        client=client,  # type: ignore[arg-type]
        state=TelegramStateStore(tmp_path / "telegram.json"),
        profiles=_Profiles(),  # type: ignore[arg-type]
        files=_Files(),  # type: ignore[arg-type]
        bot_token="token",
        owner_tenant_id="tenant-a",
        allowed_user_ids="7",
        allowed_usernames=None,
        telegram_business=business,
    )
    owner_body = {
        "message": {
            "chat": {"id": 9},
            "from": {"id": 7, "username": "owner"},
            "text": "hello",
        }
    }

    accepted = await relay.accept_update(owner_body)
    await relay.process_update(accepted)

    assert business.calls == 0
    assert len(agent.requests) == 1
