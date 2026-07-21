from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from opentulpa.application.product_tools import ProductToolApplication
from opentulpa.integrations import tenant_composio
from opentulpa.integrations.tenant_composio import (
    ComposioProviderError,
    IntegrationConnectionNotFoundError,
    TenantComposioService,
)
from opentulpa.jobs import Job, JobHandlerRegistry, JobService
from opentulpa.persistence.idempotency import (
    IdempotencyConflictError,
    IdempotencyPendingError,
    IdempotencyStore,
)
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling.adapters import ProductToolInvocation
from opentulpa.tooling.contract import (
    TOOL_SPEC_BY_NAME,
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
)


class _Provider:
    enabled = True

    def __init__(self) -> None:
        self.accounts: list[dict[str, Any]] = [
            {
                "id": "connection-own",
                "status": "ACTIVE",
                "user_id": "tenant-a",
                "toolkit_slug": "gmail",
                "toolkit_name": "Gmail",
            },
            {
                "id": "connection-foreign",
                "status": "ACTIVE",
                "user_id": "tenant-b",
                "toolkit_slug": "slack",
                "toolkit_name": "Slack",
            },
        ]
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.worker_threads: list[int] = []
        self.fail_authorize = False

    def _record(self, name: str, **kwargs: Any) -> None:
        self.calls.append((name, kwargs))
        self.worker_threads.append(threading.get_ident())

    def list_toolkits(self, **kwargs: Any) -> dict[str, Any]:
        self._record("list_toolkits", **kwargs)
        return {
            "items": [
                {"slug": "gmail", "name": "Gmail", "is_no_auth": False},
                {"slug": "slack", "name": "Slack", "is_no_auth": False},
            ]
        }

    def authorize_toolkit(self, **kwargs: Any) -> dict[str, Any]:
        self._record("authorize_toolkit", **kwargs)
        if self.fail_authorize:
            raise RuntimeError("api_key=provider-secret")
        return {
            "customer_id": kwargs["customer_id"],
            "connection_id": f"pending-{kwargs['toolkit']}",
            "redirect_url": "https://auth.example/connect?token=provider-secret",
            "callback_url": "https://private.example/callback",
            "api_key": "provider-secret",
        }

    def list_connected_accounts(self, **kwargs: Any) -> dict[str, Any]:
        self._record("list_connected_accounts", **kwargs)
        toolkits = {str(item).casefold() for item in kwargs.get("toolkits") or []}
        items = [
            dict(item)
            for item in self.accounts
            if not toolkits or str(item["toolkit_slug"]).casefold() in toolkits
        ]
        return {"items": items, "access_token": "provider-secret"}

    def delete_connected_account(self, **kwargs: Any) -> dict[str, Any]:
        self._record("delete_connected_account", **kwargs)
        connection_id = str(kwargs["connected_account_id"])
        self.accounts = [item for item in self.accounts if item["id"] != connection_id]
        return {
            "connected_account": {
                "id": connection_id,
                "deleted": True,
                "refresh_token": "provider-secret",
            }
        }

    def search_tools(self, **kwargs: Any) -> dict[str, Any]:
        self._record("search_tools", **kwargs)
        return {"items": [self._tool("GMAIL_SEND_EMAIL")]}

    def get_tool_schema(self, **kwargs: Any) -> dict[str, Any]:
        self._record("get_tool_schema", **kwargs)
        return {"tool": self._tool(str(kwargs["tool_slug"]))}

    def execute_tool(self, **kwargs: Any) -> dict[str, Any]:
        self._record("execute_tool", **kwargs)
        return {
            "successful": True,
            "error": None,
            "data": {
                "message_id": "message-1",
                "access_token": "provider-secret",
                "note": "api_key=provider-secret",
            },
        }

    @staticmethod
    def _tool(name: str) -> dict[str, Any]:
        integration = "slack" if name.startswith("SLACK") else "gmail"
        return {
            "slug": name,
            "name": "Send email",
            "description": "Send one email",
            "toolkit_slug": integration,
            "toolkit_name": integration.title(),
            "input_schema": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "access_token": {"type": "string", "default": "provider-secret"},
                },
            },
        }


class _UnusedPort:
    def __getattr__(self, name: str) -> Callable[..., Any]:
        def unused(**kwargs: Any) -> Any:
            raise AssertionError(f"unexpected {name} call: {kwargs}")

        return unused


def _adapter(tmp_path: Path) -> tuple[TenantComposioService, _Provider, IdempotencyStore]:
    provider = _Provider()
    store = IdempotencyStore(tmp_path / "effects.sqlite")
    return TenantComposioService(provider=provider, idempotency=store), provider, store


def _invocation(name: str, arguments: dict[str, Any], key: str) -> ProductToolInvocation:
    return ProductToolInvocation(
        spec=TOOL_SPEC_BY_NAME[name],
        context=AgentRunContext(
            tenant_id="tenant-a",
            actor_id="owner-1",
            thread_id="thread-1",
            channel=AgentChannel.WEB,
            run_kind=AgentRunKind.OWNER,
            correlation_id="correlation-1",
            origin=OriginRef(interface="web", source_id="test"),
            agent_spec=AgentSpecRef(tenant_id="tenant-a", spec_id="owner", revision=1),
            trust_class="owner",
        ),
        arguments=arguments,
        idempotency_key=key,
        audit_id="audit-1",
    )


async def _wait_terminal(
    jobs: JobService,
    *,
    tenant_id: str,
    job_id: str,
) -> Job:
    async with asyncio.timeout(2):
        while True:
            job = jobs.get(tenant_id=tenant_id, job_id=job_id)
            if job.status in {"succeeded", "failed", "cancelled"}:
                return job
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_connections_are_filtered_and_foreign_mutations_fail_closed(tmp_path: Path) -> None:
    service, provider, _ = _adapter(tmp_path)

    listed = await service.list_connections(tenant_id="tenant-a", integration_id=None)

    listed_items = listed["items"]
    assert isinstance(listed_items, list)
    assert [item["id"] for item in listed_items if isinstance(item, dict)] == [
        "connection-own"
    ]
    with pytest.raises(IntegrationConnectionNotFoundError, match="connection not found"):
        await service.get_connection(
            tenant_id="tenant-a",
            connection_id="connection-foreign",
        )
    with pytest.raises(IntegrationConnectionNotFoundError, match="connection not found"):
        await service.disconnect(
            tenant_id="tenant-a",
            connection_id="connection-foreign",
            idempotency_key="disconnect-foreign",
        )
    assert [name for name, _ in provider.calls].count("delete_connected_account") == 0


@pytest.mark.asyncio
async def test_connect_replays_mismatches_and_pending_claims_fail_closed(tmp_path: Path) -> None:
    service, provider, store = _adapter(tmp_path)
    main_thread = threading.get_ident()
    first = await service.connect(
        tenant_id="tenant-a",
        actor_id="owner-1",
        integration_id="gmail",
        redirect_url=None,
        idempotency_key="connect-1",
    )
    replay = await service.connect(
        tenant_id="tenant-a",
        actor_id="owner-1",
        integration_id="gmail",
        redirect_url=None,
        idempotency_key="connect-1",
    )

    assert first == replay
    assert first["user_id"] == "tenant-a"
    assert "api_key" not in first
    assert [name for name, _ in provider.calls].count("authorize_toolkit") == 1
    assert provider.worker_threads and all(thread != main_thread for thread in provider.worker_threads)
    with pytest.raises(IdempotencyConflictError, match="different request"):
        await service.connect(
            tenant_id="tenant-a",
            actor_id="owner-1",
            integration_id="slack",
            redirect_url=None,
            idempotency_key="connect-1",
        )

    pending_key = "connect-pending"
    arguments = {
        "actor_id": "owner-1",
        "integration_id": "gmail",
        "callback_url": None,
    }
    store.claim(
        tenant_id="tenant-a",
        idempotency_key=tenant_composio._provider_effect_key(  # noqa: SLF001
            "integration_connect", pending_key
        ),
        operation="integration_connect",
        arguments={
            "request_hash": tenant_composio._request_hash(  # noqa: SLF001
                "integration_connect", arguments
            )
        },
    )
    with pytest.raises(IdempotencyPendingError, match="indeterminate"):
        await service.connect(
            tenant_id="tenant-a",
            actor_id="owner-1",
            integration_id="gmail",
            redirect_url=None,
            idempotency_key=pending_key,
        )
    assert [name for name, _ in provider.calls].count("authorize_toolkit") == 1


@pytest.mark.asyncio
async def test_provider_errors_are_sanitized_and_failed_keys_do_not_retry(tmp_path: Path) -> None:
    service, provider, _ = _adapter(tmp_path)
    provider.fail_authorize = True

    with pytest.raises(ComposioProviderError) as error:
        await service.connect(
            tenant_id="tenant-a",
            actor_id="owner-1",
            integration_id="gmail",
            redirect_url=None,
            idempotency_key="provider-failure",
        )

    assert "provider-secret" not in str(error.value)
    with pytest.raises(IdempotencyConflictError, match="previous external effect attempt failed"):
        await service.connect(
            tenant_id="tenant-a",
            actor_id="owner-1",
            integration_id="gmail",
            redirect_url=None,
            idempotency_key="provider-failure",
        )
    assert [name for name, _ in provider.calls].count("authorize_toolkit") == 1


@pytest.mark.asyncio
async def test_disconnect_replays_without_a_second_provider_delete(tmp_path: Path) -> None:
    service, provider, _ = _adapter(tmp_path)

    first = await service.disconnect(
        tenant_id="tenant-a",
        connection_id="connection-own",
        idempotency_key="disconnect-1",
    )
    replay = await service.disconnect(
        tenant_id="tenant-a",
        connection_id="connection-own",
        idempotency_key="disconnect-1",
    )

    assert first == replay
    assert first["deleted"] is True
    assert [name for name, _ in provider.calls].count("delete_connected_account") == 1
    with pytest.raises(IdempotencyConflictError, match="different request"):
        await service.disconnect(
            tenant_id="tenant-a",
            connection_id="connection-foreign",
            idempotency_key="disconnect-1",
        )


@pytest.mark.asyncio
async def test_product_application_and_provider_idempotency_namespaces_coexist(
    tmp_path: Path,
) -> None:
    service, provider, store = _adapter(tmp_path)
    unused = _UnusedPort()
    application = ProductToolApplication(
        profiles=unused,
        files=unused,
        artifacts=unused,
        knowledge=unused,
        research=unused,
        browser=unused,
        integrations=service,
        intake=unused,
        schedules=unused,
        jobs=unused,
        idempotency=store,
    )
    invocation = _invocation(
        "integration_connect",
        {"integration_id": "gmail", "redirect_url": None},
        "shared-raw-key",
    )

    first = await application.integration_connect(invocation)
    replay = await application.integration_connect(invocation)

    assert first == replay
    assert first.data["connection_id"] == "pending-gmail"
    assert first.data["oauth_url"] == "https://auth.example/connect?token=provider-secret"
    assert [name for name, _ in provider.calls].count("authorize_toolkit") == 1


@pytest.mark.asyncio
async def test_invoke_job_revalidates_owner_and_sanitizes_result(tmp_path: Path) -> None:
    service, provider, _ = _adapter(tmp_path)
    registry = JobHandlerRegistry()
    service.register_handlers(registry)
    jobs = JobService(tmp_path / "jobs.sqlite", registry=registry)
    assert registry.names() == ("integration_invoke",)

    queued = await jobs.create(
        tenant_id="tenant-a",
        handler_name="integration_invoke",
        arguments={
            "connection_id": "connection-own",
            "action_name": "GMAIL_SEND_EMAIL",
            "parameters": {"to": "lead@example.com"},
        },
        idempotency_key="invoke-owner-race",
    )
    provider.accounts[0]["user_id"] = "tenant-b"
    await jobs.start()
    rejected = await _wait_terminal(jobs, tenant_id="tenant-a", job_id=queued.id)

    assert rejected.status == "failed"
    assert [name for name, _ in provider.calls].count("execute_tool") == 0

    provider.accounts[0]["user_id"] = "tenant-a"
    accepted = await jobs.create(
        tenant_id="tenant-a",
        handler_name="integration_invoke",
        arguments={
            "connection_id": "connection-own",
            "action_name": "GMAIL_SEND_EMAIL",
            "parameters": {"to": "lead@example.com"},
        },
        idempotency_key="invoke-success",
    )
    completed = await _wait_terminal(jobs, tenant_id="tenant-a", job_id=accepted.id)
    await jobs.shutdown()

    assert completed.status == "succeeded"
    assert completed.result is not None
    action_result = completed.result.data["result"]
    assert isinstance(action_result, dict)
    assert action_result["access_token"] == "[redacted]"
    assert action_result["note"] == "api_key=[redacted]"
    execute_call = next(kwargs for name, kwargs in provider.calls if name == "execute_tool")
    assert execute_call["customer_id"] == "tenant-a"
    assert execute_call["connected_account_id"] == "connection-own"
