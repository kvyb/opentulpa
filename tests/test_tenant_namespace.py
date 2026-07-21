from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.store.sqlite.aio import AsyncSqliteStore

from opentulpa.deep_agent import AgentRunContext, AgentRunRequest, DeepAgentService
from opentulpa.persistence.tenant_namespace import (
    tenant_namespace_label,
    tenant_store_namespace,
)
from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling import AgentChannel, AgentRunKind


class _ToolCapableModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _ToolCapableModel:
        del tools, tool_choice, kwargs
        return self


def _context(*, tenant_id: str, thread_id: str) -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="owner",
        thread_id=thread_id,
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id=thread_id,
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
        trust_class="owner",
    )


def test_tenant_namespace_is_valid_deterministic_and_collision_resistant() -> None:
    dotted = tenant_namespace_label("tenant.alpha")
    slug_collision = tenant_namespace_label("tenant-alpha")

    assert dotted == tenant_namespace_label("tenant.alpha")
    assert "." not in dotted
    assert "." not in slug_collision
    assert dotted != slug_collision
    assert tenant_namespace_label(" customer@example.com ") == tenant_namespace_label(
        "customer@example.com"
    )
    with pytest.raises(ValueError, match="tenant_id is required"):
        tenant_namespace_label("  ")


@pytest.mark.asyncio
async def test_deep_agent_memory_store_isolates_dotted_tenant_ids(tmp_path: Path) -> None:
    model = _ToolCapableModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {
                            "file_path": "/memories/note.md",
                            "content": "dotted tenant memory",
                        },
                        "id": "write-dotted",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Stored dotted tenant memory."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {
                            "file_path": "/memories/note.md",
                            "content": "slug collision tenant memory",
                        },
                        "id": "write-slug-collision",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Stored the other tenant memory."),
        ]
    )
    store_path = tmp_path / "store.sqlite3"
    service = DeepAgentService(
        api_key="",
        base_url="",
        model_name="test-model",
        checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
        store_db_path=store_path,
        runs_db_path=tmp_path / "runs.sqlite3",
        workspaces_root=tmp_path / "workspaces",
        model=model,
    )
    await service.start()
    try:
        first_events = [
            event
            async for event in service.stream(
                AgentRunRequest(
                    context=_context(tenant_id="tenant.alpha", thread_id="thread-dotted"),
                    text="Remember this",
                )
            )
        ]
        second_events = [
            event
            async for event in service.stream(
                AgentRunRequest(
                    context=_context(tenant_id="tenant-alpha", thread_id="thread-slug"),
                    text="Remember this separately",
                )
            )
        ]
    finally:
        await service.shutdown()

    assert first_events[-1].type == "run.completed"
    assert second_events[-1].type == "run.completed"
    async with AsyncSqliteStore.from_conn_string(str(store_path)) as store:
        await store.setup()
        dotted = await store.aget(
            tenant_store_namespace("tenant.alpha", "memory"),
            "/note.md",
        )
        slug_collision = await store.aget(
            tenant_store_namespace("tenant-alpha", "memory"),
            "/note.md",
        )
    assert dotted is not None
    assert dotted.value["content"] == "dotted tenant memory"
    assert slug_collision is not None
    assert slug_collision.value["content"] == "slug collision tenant memory"
