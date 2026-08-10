from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from opentulpa.bootstrap.evolution_api import EvolutionClient, register_evolution_control_api


class _EvolutionService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def source_status(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("status", kwargs))
        return {"active_release_id": "release-current", "workspace_head": "a" * 40}

    async def source_read(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("read", kwargs))
        return {"path": kwargs["path"], "content": "hello\n"}

    async def source_write(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("write", kwargs))
        return {"path": kwargs["path"], "bytes_written": len(kwargs["content"])}

    async def source_edit(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("edit", kwargs))
        return {"path": kwargs["path"], "replacements": 1}

    async def source_bash(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("bash", kwargs))
        return {"exit_code": 0, "output": "ok"}

    async def source_activate(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("activate", kwargs))
        return {"status": "preparing", "activation_id": "activation-1"}

    async def source_rollback(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("rollback", kwargs))
        return {"status": "preparing", "activation_id": "activation-2"}

    async def source_runtime_env_get(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("runtime-env-read", kwargs))
        return {"available": True, "variables": [], "count": 0}

    async def source_set_runtime_env(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("runtime-env", kwargs))
        return {"status": "updated", "name": kwargs["name"], "value": "[set]"}


@pytest.mark.asyncio
async def test_evolution_control_client_is_authenticated_and_source_only() -> None:
    service = _EvolutionService()
    app = FastAPI()
    token = "t" * 48
    register_evolution_control_api(app, service=service, token=token)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        client = EvolutionClient(
            base_url="http://bootstrap/bootstrap/internal/v1/evolution",
            token=token,
            client=http_client,
        )
        await client.start()
        audit = {"tenant_id": "owner", "thread_id": "thread-1"}

        assert (await client.source_status(audit_context=audit))["active_release_id"] == (
            "release-current"
        )
        assert (await client.source_read(path="README.md", audit_context=audit))["content"] == (
            "hello\n"
        )
        await client.source_write(path="README.md", content="hello\n", audit_context=audit)
        await client.source_edit(
            path="README.md",
            old_text="hello",
            new_text="hi",
            audit_context=audit,
        )
        await client.source_bash(command="git status", audit_context=audit)
        await client.source_activate(
            idempotency_key="activate-1",
            message="Improve source",
            audit_context=audit,
        )
        await client.source_rollback(
            idempotency_key="rollback-1",
            expected_active_release_id="release-current",
            audit_context=audit,
        )
        await client.source_runtime_env_get(audit_context=audit)
        await client.source_set_runtime_env(
            name="TELEGRAM_BOT_TOKEN",
            value="raw-token-value",
            idempotency_key="source-env-1",
            audit_context=audit,
        )
        unauthorized = await http_client.post(
            "http://bootstrap/bootstrap/internal/v1/evolution/source/status",
            json={"audit_context": {}},
        )

    assert unauthorized.status_code == 401
    assert [name for name, _ in service.calls] == [
        "status",
        "read",
        "write",
        "edit",
        "bash",
        "activate",
        "rollback",
        "runtime-env-read",
        "runtime-env",
    ]
    assert service.calls[1][1] == {
        "path": "README.md",
        "offset": 1,
        "limit": 2_000,
        "audit_context": audit,
    }
    assert service.calls[5][1]["idempotency_key"] == "activate-1"
    assert service.calls[6][1]["expected_active_release_id"] == "release-current"
