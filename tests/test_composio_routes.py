from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from opentulpa.core.config import get_settings

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
        self.started = False
        self.jobs: dict[str, dict[str, Any]] = {}

    def add_job(self, func: Any, trigger: Any, *, id: str, args: list[Any], **kwargs: Any) -> None:
        self.jobs[str(id)] = {
            "func": func,
            "trigger": trigger,
            "args": list(args),
            **kwargs,
        }

    def remove_job(self, job_id: str) -> None:
        self.jobs.pop(str(job_id), None)

    def start(self) -> None:
        self.started = True

    def shutdown(self, wait: bool = True) -> None:
        _ = wait
        self.started = False


class _DummyCronTrigger:
    @classmethod
    def from_crontab(cls, value: str) -> _DummyCronTrigger:
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

from opentulpa.api.app import create_app  # noqa: E402
from opentulpa.integrations.composio import ComposioService  # noqa: E402
from opentulpa.skills.service import SkillStoreService  # noqa: E402


class _FakeComposioService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def status(self) -> dict[str, Any]:
        return {
            "ok": True,
            "enabled": True,
            "callback_url_configured": True,
            "default_callback_url": "https://example.com/composio/callback",
        }

    def authorize_toolkit(
        self,
        *,
        customer_id: str,
        toolkit: str,
        callback_url: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "authorize_toolkit",
                {
                    "customer_id": customer_id,
                    "toolkit": toolkit,
                    "callback_url": callback_url,
                },
            )
        )
        return {
            "ok": True,
            "customer_id": customer_id,
            "toolkit": toolkit,
            "connection_id": "conn_123",
            "redirect_url": "https://connect.example.com/auth",
            "callback_url": callback_url,
            "message_for_user": "Connect your instagram account here: https://connect.example.com/auth",
        }

    def wait_for_connection(self, *, connection_id: str, timeout_seconds: float = 60.0) -> dict[str, Any]:
        self.calls.append(
            (
                "wait_for_connection",
                {"connection_id": connection_id, "timeout_seconds": timeout_seconds},
            )
        )
        return {"id": connection_id, "status": "ACTIVE", "toolkit_slug": "instagram"}

    def list_toolkits(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        is_connected: bool | None = None,
        limit: int = 50,
        search: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "list_toolkits",
                {
                    "customer_id": customer_id,
                    "toolkits": toolkits,
                    "is_connected": is_connected,
                    "limit": limit,
                    "search": search,
                },
            )
        )
        return {
            "ok": True,
            "customer_id": customer_id,
            "items": [{"slug": "instagram", "is_connected": True}],
            "next_cursor": None,
            "total_pages": 1,
        }

    def list_connected_accounts(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "list_connected_accounts",
                {
                    "customer_id": customer_id,
                    "toolkits": toolkits,
                    "statuses": statuses,
                    "limit": limit,
                },
            )
        )
        return {"ok": True, "customer_id": customer_id, "items": [{"id": "acct_1"}], "next_cursor": None}

    def disable_connected_account(self, *, connected_account_id: str) -> dict[str, Any]:
        self.calls.append(("disable_connected_account", {"connected_account_id": connected_account_id}))
        return {"ok": True, "connected_account": {"id": connected_account_id, "disabled": True}}

    def delete_connected_account(self, *, connected_account_id: str) -> dict[str, Any]:
        self.calls.append(("delete_connected_account", {"connected_account_id": connected_account_id}))
        return {"ok": True, "connected_account": {"id": connected_account_id, "deleted": True}}

    def search_tools(
        self,
        *,
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "search_tools",
                {"query": query, "toolkits": toolkits, "limit": limit},
            )
        )
        return {"ok": True, "items": [{"slug": "INSTAGRAM_LIST_ALL_MESSAGES"}]}

    def get_tool_schema(self, *, tool_slug: str) -> dict[str, Any]:
        self.calls.append(("get_tool_schema", {"tool_slug": tool_slug}))
        return {"ok": True, "tool": {"slug": tool_slug, "input_schema": {"type": "object"}}}

    def inspect_instagram_reply_target(
        self,
        *,
        customer_id: str,
        recipient_id: str | None = None,
        conversation_id: str | None = None,
        connected_account_id: str | None = None,
        scan_limit: int = 10,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "inspect_instagram_reply_target",
                {
                    "customer_id": customer_id,
                    "recipient_id": recipient_id,
                    "conversation_id": conversation_id,
                    "connected_account_id": connected_account_id,
                    "scan_limit": scan_limit,
                },
            )
        )
        return {
            "ok": True,
            "matched": True,
            "recipient_id_verified": True,
            "conversation_id": conversation_id or "conv_123",
            "recipient_id": recipient_id or "rcp_123",
            "latest_inbound_message_created_time": "2026-04-06T11:14:00+0000",
            "reply_window_status": "unconfirmed",
        }

    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "execute_tool",
                {
                    "customer_id": customer_id,
                    "tool_slug": tool_slug,
                    "arguments": arguments,
                    "connected_account_id": connected_account_id,
                    "text": text,
                },
            )
        )
        return {"ok": True, "tool_slug": tool_slug, "successful": True, "data": {"items": []}}


class _FakeComposioTransientError(Exception):
    status_code = 500
    body = {"error": {"status": 500, "message": "Bad Gateway"}}


class _RetryOnceComposioService(_FakeComposioService):
    def execute_tool(
        self,
        *,
        customer_id: str,
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "execute_tool",
                {
                    "customer_id": customer_id,
                    "tool_slug": tool_slug,
                    "arguments": arguments,
                    "connected_account_id": connected_account_id,
                    "text": text,
                },
            )
        )
        if len(self.calls) == 1:
            raise _FakeComposioTransientError("temporary")
        return {"ok": True, "tool_slug": tool_slug, "successful": True, "data": {"items": []}}


def _mk_client(tmp_path: Path) -> tuple[TestClient, _FakeComposioService]:
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    composio = _FakeComposioService()
    app = create_app(
        skill_store_service=skills,
        composio_service=composio,
    )
    return TestClient(app), composio


def _mk_client_with_composio(tmp_path: Path, composio: Any) -> TestClient:
    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(
        skill_store_service=skills,
        composio_service=composio,
    )
    return TestClient(app)


def test_composio_status_route_exposes_callback_configuration(tmp_path: Path) -> None:
    client, _composio = _mk_client(tmp_path)

    with client:
        response = client.get("/internal/composio/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["enabled"] is True
    assert payload["callback_url_configured"] is True
    assert payload["default_callback_url"] == "https://example.com/composio/callback"


def test_composio_instagram_reply_precheck_route(tmp_path: Path) -> None:
    client, composio = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/composio/instagram/reply_precheck",
            json={
                "customer_id": "cust_123",
                "recipient_id": "rcp_123",
                "connected_account_id": "acct_1",
                "scan_limit": 7,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["recipient_id_verified"] is True
    assert composio.calls[0] == (
        "inspect_instagram_reply_target",
        {
            "customer_id": "cust_123",
            "recipient_id": "rcp_123",
            "conversation_id": None,
            "connected_account_id": "acct_1",
            "scan_limit": 7,
        },
    )


def test_composio_disable_connected_account_route(tmp_path: Path) -> None:
    client, composio = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/composio/connected_accounts/disable",
            json={"connected_account_id": "acct_1"},
        )

    assert response.status_code == 200
    assert response.json()["connected_account"]["disabled"] is True
    assert composio.calls[0] == ("disable_connected_account", {"connected_account_id": "acct_1"})


def test_composio_delete_connected_account_route(tmp_path: Path) -> None:
    client, composio = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/composio/connected_accounts/delete",
            json={"connected_account_id": "acct_1"},
        )

    assert response.status_code == 200
    assert response.json()["connected_account"]["deleted"] is True
    assert composio.calls[0] == ("delete_connected_account", {"connected_account_id": "acct_1"})


def test_composio_authorize_route_returns_redirect_payload(tmp_path: Path) -> None:
    client, composio = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/composio/authorize",
            json={
                "customer_id": "cust_123",
                "toolkit": "instagram",
                "callback_url": "https://example.com/callback",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["redirect_url"] == "https://connect.example.com/auth"
    assert payload["message_for_user"].startswith("Connect your instagram account here:")
    assert composio.calls[0] == (
        "authorize_toolkit",
        {
            "customer_id": "cust_123",
            "toolkit": "instagram",
            "callback_url": "https://example.com/callback",
        },
    )


def test_composio_callback_landing_page_is_public(tmp_path: Path) -> None:
    client, _composio = _mk_client(tmp_path)

    with client:
        response = client.get(
            "/webhook/composio/callback",
            params={"toolkit": "instagram", "connection_id": "conn_123"},
        )

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "connecting instagram" in response.text.lower()
    assert "conn_123" in response.text


def test_composio_service_status_uses_public_base_url_as_callback(monkeypatch) -> None:
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://example.com/")
    monkeypatch.delenv("RAILWAY_PUBLIC_DOMAIN", raising=False)
    service = ComposioService(api_key="test-key", default_callback_url=None)

    payload = service.status()

    assert payload["callback_url_configured"] is True
    assert payload["default_callback_url"] is None
    assert payload["resolved_callback_url"] == "https://example.com/webhook/composio/callback"


def test_composio_status_route_returns_disabled_when_api_key_unset(tmp_path: Path, monkeypatch) -> None:
    settings = get_settings().model_copy(
        update={
            "composio_api_key": None,
            "composio_default_callback_url": None,
        }
    )
    monkeypatch.setattr("opentulpa.api.app.get_settings", lambda: settings)

    skills = SkillStoreService(
        db_path=tmp_path / "skills.db",
        root_dir=tmp_path / "skills",
    )
    app = create_app(skill_store_service=skills)

    with TestClient(app) as client:
        response = client.get("/internal/composio/status")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "enabled": False,
        "callback_url_configured": False,
        "default_callback_url": None,
        "resolved_callback_url": None,
    }


def test_composio_execute_route_passes_customer_and_arguments(tmp_path: Path) -> None:
    client, composio = _mk_client(tmp_path)

    with client:
        response = client.post(
            "/internal/composio/tools/execute",
            json={
                "customer_id": "cust_123",
                "tool_slug": "INSTAGRAM_LIST_ALL_MESSAGES",
                "arguments": {"conversation_id": "conv_1"},
                "connected_account_id": "acct_1",
                "text": "list the messages",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["successful"] is True
    assert composio.calls[0] == (
        "execute_tool",
        {
            "customer_id": "cust_123",
            "tool_slug": "INSTAGRAM_LIST_ALL_MESSAGES",
            "arguments": {"conversation_id": "conv_1"},
            "connected_account_id": "acct_1",
            "text": "list the messages",
        },
    )


def test_composio_execute_route_does_not_retry_transient_errors(tmp_path: Path) -> None:
    composio = _RetryOnceComposioService()
    client = _mk_client_with_composio(tmp_path, composio)

    with client:
        response = client.post(
            "/internal/composio/tools/execute",
            json={
                "customer_id": "cust_123",
                "tool_slug": "INSTAGRAM_LIST_ALL_MESSAGES",
                "arguments": {"conversation_id": "conv_1"},
            },
        )

    assert response.status_code == 500
    assert response.json()["error"]["message"] == "Bad Gateway"
    assert [call[0] for call in composio.calls] == ["execute_tool"]
