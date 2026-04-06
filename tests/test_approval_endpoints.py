from __future__ import annotations

from pathlib import Path
import sys
from typing import Any
import types

import pytest
from fastapi.testclient import TestClient

apscheduler_module = types.ModuleType("apscheduler")
schedulers_module = types.ModuleType("apscheduler.schedulers")
asyncio_module = types.ModuleType("apscheduler.schedulers.asyncio")
triggers_module = types.ModuleType("apscheduler.triggers")
cron_module = types.ModuleType("apscheduler.triggers.cron")
date_module = types.ModuleType("apscheduler.triggers.date")
mem0_module = types.ModuleType("mem0")


class _DummyAsyncIOScheduler:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


class _DummyCronTrigger:
    @classmethod
    def from_crontab(cls, value: str) -> "_DummyCronTrigger":
        _ = value
        return cls()


class _DummyDateTrigger:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


class _DummyMemory:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs


asyncio_module.AsyncIOScheduler = _DummyAsyncIOScheduler
cron_module.CronTrigger = _DummyCronTrigger
date_module.DateTrigger = _DummyDateTrigger
apscheduler_module.schedulers = schedulers_module
apscheduler_module.triggers = triggers_module
schedulers_module.asyncio = asyncio_module
triggers_module.cron = cron_module
triggers_module.date = date_module
mem0_module.Memory = _DummyMemory
sys.modules.setdefault("apscheduler", apscheduler_module)
sys.modules.setdefault("apscheduler.schedulers", schedulers_module)
sys.modules.setdefault("apscheduler.schedulers.asyncio", asyncio_module)
sys.modules.setdefault("apscheduler.triggers", triggers_module)
sys.modules.setdefault("apscheduler.triggers.cron", cron_module)
sys.modules.setdefault("apscheduler.triggers.date", date_module)
sys.modules.setdefault("mem0", mem0_module)

from opentulpa.api.app import create_app
from opentulpa.core.config import get_settings


class _DummyTool:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(args)
        return {"ok": True, "echo": args}


class _DummyRuntime:
    def __init__(self) -> None:
        self.started = 0
        self.tool = _DummyTool()
        self._tools = {"dummy_action": self.tool}

    async def start(self) -> None:
        self.started += 1

    async def shutdown(self) -> None:
        return None

    def healthy(self) -> bool:
        return True

    async def classify_guardrail_intent(
        self,
        *,
        action_name: str,
        action_args: dict[str, object],
        action_note: str | None = None,
    ) -> dict[str, object]:
        _ = action_note
        if action_name == "dummy_read_action":
            return {
                "ok": True,
                "gate": "require_approval",
                "impact_type": "read",
                "recipient_scope": "external",
                "confidence": 0.9,
                "reason": "paranoid read",
            }
        if action_name == "routine_create":
            combined = (
                f"{str(action_args.get('name', ''))} "
                f"{str(action_args.get('message', ''))}"
            ).strip().lower()
            if "post to x" in combined or "autopost" in combined:
                return {
                    "ok": True,
                    "gate": "require_approval",
                    "impact_type": "write",
                    "recipient_scope": "external",
                    "confidence": 0.9,
                    "reason": "external publish routine",
                }
            return {
                "ok": True,
                "gate": "allow",
                "impact_type": "read",
                "recipient_scope": "self",
                "confidence": 0.85,
                "reason": "internal routine",
            }
        return {
            "ok": True,
            "gate": "require_approval",
            "impact_type": "write",
            "recipient_scope": "external",
            "confidence": 0.8,
            "reason": "default",
        }


@pytest.fixture
def approvals_client(
    tmp_path: Path,
    monkeypatch: Any,
) -> tuple[TestClient, _DummyRuntime]:
    monkeypatch.setenv("APPROVALS_DB_PATH", str(tmp_path / "pending_approvals_test.db"))
    get_settings.cache_clear()
    runtime = _DummyRuntime()
    app = create_app(agent_runtime=runtime)
    with TestClient(app) as client:
        yield client, runtime
    get_settings.cache_clear()


def test_approval_endpoints_lifecycle(approvals_client: tuple[TestClient, _DummyRuntime]) -> None:
    client, runtime = approvals_client
    evaluate = client.post(
        "/internal/approvals/evaluate",
        json={
            "customer_id": "cust_9",
            "thread_id": "chat-9",
            "action_name": "dummy_action",
            "action_args": {"x": 1},
            "origin_interface": "unknown",
            "origin_user_id": "99",
            "origin_conversation_id": "",
        },
    )
    assert evaluate.status_code == 200
    payload = evaluate.json()
    assert payload["gate"] == "require_approval"
    approval_id = payload["approval_id"]

    fetched = client.get(f"/internal/approvals/{approval_id}")
    assert fetched.status_code == 200
    assert fetched.json()["approval"]["status"] == "pending"

    wrong_actor = client.post(
        "/internal/approvals/decide",
        json={
            "approval_id": approval_id,
            "decision": "approve",
            "actor_interface": "unknown",
            "actor_id": "100",
        },
    )
    assert wrong_actor.status_code == 200
    assert wrong_actor.json()["ok"] is False
    assert wrong_actor.json()["reason"] == "unauthorized_actor"

    approved = client.post(
        "/internal/approvals/decide",
        json={
            "approval_id": approval_id,
            "decision": "approve",
            "actor_interface": "unknown",
            "actor_id": "99",
        },
    )
    assert approved.status_code == 200
    assert approved.json()["ok"] is True
    assert approved.json()["status"] == "approved"

    executed = client.post(
        "/internal/approvals/execute",
        json={"approval_id": approval_id, "customer_id": "cust_9"},
    )
    assert executed.status_code == 200
    assert executed.json()["ok"] is True
    assert runtime.tool.calls == [{"x": 1}]

    replay = client.post(
        "/internal/approvals/execute",
        json={"approval_id": approval_id, "customer_id": "cust_9"},
    )
    assert replay.status_code == 200
    replay_payload = replay.json()
    assert replay_payload["ok"] is True
    assert replay_payload["already_executed"] is True


def test_background_actions_are_preauthorized_without_runtime_grant_lookup(
    approvals_client: tuple[TestClient, _DummyRuntime],
) -> None:
    client, _ = approvals_client
    allowed = client.post(
        "/internal/approvals/evaluate",
        json={
            "customer_id": "cust_9",
            "thread_id": "wake_123abc",
            "action_name": "email_send",
            "action_args": {"to": "a@example.com", "text": "hi"},
            "origin_interface": "unknown",
            "origin_user_id": "99",
            "origin_conversation_id": "",
        },
    )
    assert allowed.status_code == 200
    allowed_payload = allowed.json()
    assert allowed_payload["gate"] == "allow"
    assert allowed_payload["reason"] == "background_preauthorized_execution"
    assert allowed_payload.get("approval_id") is None


def test_external_read_is_allow_even_if_classifier_requests_approval(
    approvals_client: tuple[TestClient, _DummyRuntime],
) -> None:
    client, _ = approvals_client
    evaluate = client.post(
        "/internal/approvals/evaluate",
        json={
            "customer_id": "cust_9",
            "thread_id": "chat-9",
            "action_name": "dummy_read_action",
            "action_args": {"url": "https://example.com"},
            "origin_interface": "unknown",
            "origin_user_id": "99",
            "origin_conversation_id": "",
        },
    )
    assert evaluate.status_code == 200
    payload = evaluate.json()
    assert payload["gate"] == "allow"
    assert payload["reason"] == "read_only_no_approval"
    assert payload.get("approval_id") is None


def test_external_routine_creation_requires_one_time_approval(
    approvals_client: tuple[TestClient, _DummyRuntime],
) -> None:
    client, _ = approvals_client

    evaluate = client.post(
        "/internal/approvals/evaluate",
        json={
            "customer_id": "cust_9",
            "thread_id": "chat-9",
            "action_name": "routine_create",
            "action_args": {
                "name": "Autopost X",
                "schedule": "0 */2 * * *",
                "message": "Post to X every 2 hours",
                "customer_id": "cust_9",
                "notify_user": True,
            },
            "origin_interface": "unknown",
            "origin_user_id": "99",
            "origin_conversation_id": "",
        },
    )
    assert evaluate.status_code == 200
    approval_id = str(evaluate.json().get("approval_id", "")).strip()
    assert approval_id

    decided = client.post(
        "/internal/approvals/decide",
        json={
            "approval_id": approval_id,
            "decision": "approve",
            "actor_interface": "unknown",
            "actor_id": "99",
        },
    )
    assert decided.status_code == 200
    decided_payload = decided.json()
    assert decided_payload["ok"] is True
    assert decided_payload["status"] == "approved"

    reevaluate = client.post(
        "/internal/approvals/evaluate",
        json={
            "customer_id": "cust_9",
            "thread_id": "wake_123abc",
            "action_name": "email_send",
            "action_args": {"to": "a@example.com", "text": "x"},
            "origin_interface": "unknown",
            "origin_user_id": "99",
            "origin_conversation_id": "",
        },
    )
    assert reevaluate.status_code == 200
    reevaluate_payload = reevaluate.json()
    assert reevaluate_payload["gate"] == "allow"
    assert reevaluate_payload["reason"] == "background_preauthorized_execution"


def test_background_external_routine_creation_still_needs_approval(
    approvals_client: tuple[TestClient, _DummyRuntime],
) -> None:
    client, _ = approvals_client
    evaluate = client.post(
        "/internal/approvals/evaluate",
        json={
            "customer_id": "cust_9",
            "thread_id": "wake_123abc",
            "action_name": "routine_create",
            "action_args": {
                "name": "Autopost X",
                "schedule": "0 */2 * * *",
                "message": "Post to X every 2 hours",
                "customer_id": "cust_9",
                "notify_user": True,
            },
            "origin_interface": "unknown",
            "origin_user_id": "99",
            "origin_conversation_id": "",
        },
    )
    assert evaluate.status_code == 200
    payload = evaluate.json()
    assert payload["gate"] == "require_approval"
    assert payload.get("approval_id")
