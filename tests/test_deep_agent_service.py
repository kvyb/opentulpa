from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from deepagents.middleware.summarization import compute_summarization_defaults
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.tools import BaseTool, tool

from opentulpa.deep_agent import (
    AgentRunContext,
    AgentRunRequest,
    ApprovalDecision,
    DeepAgentService,
    TenantDynamicToolRegistry,
)
from opentulpa.deep_agent.contracts import (
    AgentRunCheckpointConflictError,
    AgentRunIdempotencyConflictError,
)
from opentulpa.deep_agent.sandbox import TenantSandboxBackend
from opentulpa.deep_agent.service import (
    _ProviderFallbackMiddleware,
    _shell_command_requires_approval,
    _with_deepagents_context_budget,
    build_openrouter_chat_model,
)
from opentulpa.specs import AgentSpecRef, AgentSpecStore, AgentSpecWrite, OriginRef
from opentulpa.tooling import AgentChannel, AgentRunKind


class _ToolCapableTextModel(FakeListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _ToolCapableTextModel:
        del tools, tool_choice, kwargs
        return self


class _ToolCapableMessageModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> _ToolCapableMessageModel:
        del tools, tool_choice, kwargs
        return self


def _destructive_shell_tool(calls: list[str]) -> BaseTool:
    @tool("source_shell")
    def source_shell(command: str) -> dict[str, str]:
        """Run a source command after destructive-shell approval."""

        calls.append(command)
        return {"command": command}

    return source_shell


class _AttachmentResolver:
    def __init__(self, records: dict[tuple[str, str], tuple[dict[str, Any], bytes]]) -> None:
        self.records = records
        self.reads: list[tuple[str, str]] = []

    def get_file(self, tenant_id: str, file_id: str) -> dict[str, Any] | None:
        resolved = self.records.get((tenant_id, file_id))
        return dict(resolved[0]) if resolved is not None else None

    def read_file_bytes(self, tenant_id: str, file_id: str) -> bytes | None:
        self.reads.append((tenant_id, file_id))
        resolved = self.records.get((tenant_id, file_id))
        return resolved[1] if resolved is not None else None


class _FailingGraph:
    def __init__(self, error: str) -> None:
        self._error = error

    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        if False:
            yield {}
        raise RuntimeError(self._error)


class BadRequestResponseError(RuntimeError):
    pass


class _ProviderRejectedGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        if False:
            yield {}
        raise BadRequestResponseError("private provider response")


class _ProviderRejectedAfterOutputGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        yield {
            "type": "messages",
            "data": (AIMessage(content="usable streamed response"), {}),
        }
        raise BadRequestResponseError("private provider response")


class _ProviderUnavailableGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        if False:
            yield {}
        raise ValueError("OpenRouter API returned an error: no endpoints available")


class _CodexOverloadedGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        if False:
            yield {}
        raise RuntimeError("Our servers are currently overloaded. Please try again later.")


class _ToolErrorThenCodexOverloadedGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        yield {
            "type": "messages",
            "data": (
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repository_publish_pr",
                            "args": {},
                            "id": "call-publish",
                            "type": "tool_call",
                        }
                    ],
                ),
                {},
            ),
        }
        yield {
            "type": "messages",
            "data": (
                ToolMessage(
                    name="repository_publish_pr",
                    tool_call_id="call-publish",
                    content=json.dumps(
                        {
                            "status": "error",
                            "data": None,
                            "error": {
                                "code": "repository_workspace_error",
                                "message": "GitHub publishing is not configured.",
                                "retryable": False,
                            },
                        }
                    ),
                ),
                {},
            ),
        }
        raise RuntimeError("Our servers are currently overloaded. Please try again later.")


class _MiddlewareRequest:
    def __init__(self, model: Any) -> None:
        self.model = model

    def override(self, *, model: Any) -> _MiddlewareRequest:
        return _MiddlewareRequest(model)


class _BlockingGraph:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        self.entered.set()
        await asyncio.Event().wait()
        if False:
            yield {}


class _SerializedGraph:
    def __init__(self) -> None:
        self.first_entered = asyncio.Event()
        self.release_first = asyncio.Event()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        self.calls += 1
        call_number = self.calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if call_number == 1:
                self.first_entered.set()
                await self.release_first.wait()
            yield {
                "type": "messages",
                "data": (AIMessage(content=f"response-{call_number}"), {}),
            }
        finally:
            self.active -= 1


class _SummarizingGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        yield {
            "type": "messages",
            "data": (AIMessage(content="internal conversation summary"), {"lc_source": "summarization"}),
        }
        yield {
            "type": "messages",
            "data": (AIMessage(content="visible answer"), {}),
        }


class _ChunkedToolGraph:
    async def astream(self, graph_input: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        del graph_input, kwargs
        chunks = (
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "web_search",
                        "args": "",
                        "id": "call-search",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
            ),
            AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": None,
                        "args": '{"query":"cats"}',
                        "id": None,
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
                chunk_position="last",
            ),
            ToolMessage(
                name="web_search",
                tool_call_id="call-search",
                content='{"status":"ok"}',
            ),
            AIMessageChunk(content="Found cats.", chunk_position="last"),
        )
        for chunk in chunks:
            yield {"type": "messages", "data": (chunk, {})}


class _SerializedIntakeGraph:
    def __init__(self) -> None:
        self.first_entered = asyncio.Event()
        self.release_first = asyncio.Event()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def ainvoke(self, graph_input: Any, **kwargs: Any) -> dict[str, Any]:
        del graph_input, kwargs
        self.calls += 1
        call_number = self.calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if call_number == 1:
                self.first_entered.set()
                await self.release_first.wait()
            return {
                "structured_response": {
                    "action": "ignore",
                    "evidence_source_ids": [],
                }
            }
        finally:
            self.active -= 1


class _BlockingIntakeGraph:
    async def ainvoke(self, graph_input: Any, **kwargs: Any) -> dict[str, Any]:
        del graph_input, kwargs
        await asyncio.Event().wait()
        return {}


class _RecordingTracer:
    def __init__(self) -> None:
        self.errors: list[Exception] = []
        self.active = False
        self.callback_contexts: list[bool] = []

    def build_callbacks(self, **kwargs: Any) -> list[Any]:
        del kwargs
        self.callback_contexts.append(self.active)
        return []

    @contextmanager
    def trace_context(self, **kwargs: Any) -> Iterator[None]:
        del kwargs
        self.active = True
        try:
            yield
        except Exception as exc:
            self.errors.append(exc)
            raise
        finally:
            self.active = False


def _context() -> AgentRunContext:
    return AgentRunContext(
        tenant_id="tenant-1",
        actor_id="owner-1",
        thread_id="thread-1",
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id="correlation-1",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id="tenant-1", spec_id="owner", revision=1),
        trust_class="owner",
    )


def test_openrouter_model_uses_sdk_timeout_units_without_hidden_retries() -> None:
    model = build_openrouter_chat_model(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model_name="moonshotai/kimi-k3",
    )
    sdk_config = model.client.sdk_configuration

    try:
        assert model.request_timeout == 60_000
        assert sdk_config.timeout_ms == 60_000
        assert model.max_retries == 0
    finally:
        sdk_config.client.close()
        asyncio.run(sdk_config.async_client.aclose())


def test_openrouter_model_restricts_provider_order_before_model_fallback() -> None:
    model = build_openrouter_chat_model(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model_name="z-ai/glm-5.2",
        provider_order=("z-ai/fp8", "fireworks", "deepinfra/fp4"),
    )

    try:
        assert model.openrouter_provider == {
            "order": ["z-ai/fp8", "fireworks", "deepinfra/fp4"],
            "allow_fallbacks": False,
        }
    finally:
        model.client.sdk_configuration.client.close()
        asyncio.run(model.client.sdk_configuration.async_client.aclose())


def test_deepagents_context_budget_preserves_stricter_model_profiles() -> None:
    unknown = _ToolCapableTextModel(responses=["ok"])
    budgeted = _with_deepagents_context_budget(unknown)

    assert budgeted is unknown
    assert budgeted.profile == {"max_input_tokens": 50_000}
    assert compute_summarization_defaults(budgeted) == {
        "trigger": ("fraction", 0.85),
        "keep": ("fraction", 0.1),
        "truncate_args_settings": {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.1),
        },
    }

    smaller = _ToolCapableTextModel(
        responses=["ok"],
        profile={"max_input_tokens": 32_000, "tool_calling": True},
    )
    unchanged = _with_deepagents_context_budget(smaller)

    assert unchanged is smaller
    assert unchanged.profile == {
        "max_input_tokens": 32_000,
        "tool_calling": True,
    }

    codex = _ToolCapableTextModel(responses=["ok"])
    codex_budgeted = _with_deepagents_context_budget(codex, provider="codex")

    assert codex_budgeted is codex
    assert codex_budgeted.profile == {"max_input_tokens": 300_000}
    assert compute_summarization_defaults(codex_budgeted) == {
        "trigger": ("fraction", 0.85),
        "keep": ("fraction", 0.1),
        "truncate_args_settings": {
            "trigger": ("fraction", 0.85),
            "keep": ("fraction", 0.1),
        },
    }

    smaller_codex = _ToolCapableTextModel(
        responses=["ok"],
        profile={"max_input_tokens": 200_000, "tool_calling": True},
    )
    _with_deepagents_context_budget(smaller_codex, provider="codex")

    assert smaller_codex.profile == {
        "max_input_tokens": 200_000,
        "tool_calling": True,
    }

    larger_codex = _ToolCapableTextModel(
        responses=["ok"],
        profile={"max_input_tokens": 400_000, "tool_calling": True},
    )
    _with_deepagents_context_budget(larger_codex, provider="codex")

    assert larger_codex.profile == {
        "max_input_tokens": 300_000,
        "tool_calling": True,
    }


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf build",
        "rm -fr build",
        "rm -r -f build",
        "sudo /bin/rm --recursive --force build",
        "cd /workspace && rm -Rf build",
        "bash -c 'rm -rf build'",
    ],
)
def test_shell_policy_requires_approval_for_recursive_forced_removal(command: str) -> None:
    request = SimpleNamespace(tool_call={"args": {"command": command}})

    assert _shell_command_requires_approval(request) is True


@pytest.mark.parametrize(
    "command",
    [
        "rm build.log",
        "rm -r build",
        "rm -f build.log",
        "find build -delete",
        "python cleanup.py",
    ],
)
def test_shell_policy_auto_approves_other_commands(command: str) -> None:
    request = SimpleNamespace(tool_call={"args": {"command": command}})

    assert _shell_command_requires_approval(request) is False


@pytest.mark.asyncio
async def test_shell_policy_allows_harmless_rm_reference_without_approval(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)
    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "grep -R 'rm -rf' ."},
                        "id": "call-shell",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Search completed."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    try:
        events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Find destructive commands")
            )
        ]
    finally:
        await service.shutdown()

    assert all(event.type != "approval.required" for event in events)
    assert calls == ["grep -R 'rm -rf' ."]


@pytest.mark.asyncio
async def test_shell_policy_rejects_ambiguous_command_before_execution(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)
    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm${IFS}-rf build"},
                        "id": "call-shell",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The ambiguous command was rejected."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    try:
        events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Run the ambiguous command")
            )
        ]
    finally:
        await service.shutdown()

    assert all(event.type != "approval.required" for event in events)
    assert events[-1].type == "run.completed"
    assert calls == []


def test_job_artifact_tool_output_emits_public_artifact_event() -> None:
    message = ToolMessage(
        name="job_artifacts",
        tool_call_id="call-artifacts",
        content=json.dumps(
            {
                "status": "ok",
                "data": [
                    {
                        "id": "artifact-1",
                        "tenant_id": "tenant-1",
                        "job_id": "job-1",
                        "name": "report.pdf",
                        "media_type": "application/pdf",
                        "uri": "/private/tenant/path/report.pdf",
                        "size_bytes": 1234,
                    }
                ],
                "audit_id": "audit-1",
            }
        ),
    )

    events = DeepAgentService._message_events(message)

    assert events == [
        (
            "tool.completed",
            {
                "name": "job_artifacts",
                "call_id": "call-artifacts",
                "ok": True,
                "result": {
                    "status": "ok",
                    "data": [
                        {
                            "id": "artifact-1",
                            "tenant_id": "[redacted]",
                            "job_id": "job-1",
                            "name": "report.pdf",
                            "media_type": "application/pdf",
                            "uri": "[redacted-path]",
                            "size_bytes": 1234,
                        }
                    ],
                    "audit_id": "audit-1",
                },
            },
            "",
        ),
        (
            "artifact.ready",
            {
                "artifact_id": "artifact-1",
                "job_id": "job-1",
                "name": "report.pdf",
                "media_type": "application/pdf",
                "size_bytes": 1234,
            },
            "",
        ),
    ]
    assert "/private/tenant/path" not in str(events)


def test_partial_tool_call_chunks_do_not_emit_malformed_tool_events() -> None:
    partial = AIMessageChunk(
        content="",
        tool_calls=[
            {
                "name": "",
                "args": {"query": "cats"},
                "id": None,
                "type": "tool_call",
            }
        ],
    )

    assert DeepAgentService._message_events(partial) == []


def test_accumulated_tool_call_chunks_emit_complete_tool_information() -> None:
    start = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": "web_search",
                "args": "",
                "id": "call-search",
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
    )
    finish = AIMessageChunk(
        content="",
        tool_call_chunks=[
            {
                "name": None,
                "args": '{"query":"cats"}',
                "id": None,
                "index": 0,
                "type": "tool_call_chunk",
            }
        ],
        chunk_position="last",
    )

    assert DeepAgentService._tool_start_events(start + finish) == [
        (
            "tool.started",
            {
                "name": "web_search",
                "call_id": "call-search",
                "arguments": {"query": "cats"},
            },
            "",
        )
    ]


def _service(
    root: Path,
    model: Any,
    *,
    tools: Sequence[BaseTool] = (),
    langfuse_tracer: Any | None = None,
    run_observer: Any | None = None,
    agent_specs: AgentSpecStore | None = None,
    dynamic_tools: TenantDynamicToolRegistry | None = None,
    attachment_resolver: Any | None = None,
    execution_backend: Any | None = None,
    workspace_backend: Any | None = None,
) -> DeepAgentService:
    return DeepAgentService(
        api_key="",
        base_url="",
        model_name="test-model",
        checkpoint_db_path=root / "checkpoints.sqlite3",
        store_db_path=root / "store.sqlite3",
        runs_db_path=root / "runs.sqlite3",
        workspaces_root=root / "workspaces",
        model=model,
        tools=tools,
        langfuse_tracer=langfuse_tracer,
        run_observer=run_observer,
        agent_specs=agent_specs,
        dynamic_tools=dynamic_tools,
        attachment_resolver=attachment_resolver,
        execution_backend=execution_backend,
        workspace_backend=workspace_backend,
    )


def test_owner_uses_one_workspace_backend_for_shell_and_files(tmp_path: Path) -> None:
    workspace = TenantSandboxBackend(
        workspaces_root=tmp_path / "workspace",
        persistent_files=True,
    )
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["unused"]),
        execution_backend=workspace,
        workspace_backend=workspace,
    )

    backend = service._owner_backend()  # noqa: SLF001

    assert backend.default is workspace
    assert backend.routes["/workspace/"] is workspace


@pytest.mark.asyncio
async def test_threads_project_sanitized_requests_events_and_tenant_ownership(
    tmp_path: Path,
) -> None:
    resolver = _AttachmentResolver(
        {
            ("tenant-1", "file-1"): (
                {
                    "kind": "image",
                    "original_filename": "screen.png",
                    "mime_type": "image/png",
                    "size_bytes": 9,
                },
                b"png-bytes",
            )
        }
    )
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["Hello from Tulpa"]),
        attachment_resolver=resolver,
    )
    await service.start()
    try:
        request = AgentRunRequest(
            context=_context(),
            text="Plan the launch",
            file_ids=("file-1",),
        )
        result = await service.run(request)
        listed = await service.list_threads(tenant_id="tenant-1")
        timeline = await service.thread_timeline(
            tenant_id="tenant-1",
            thread_id=request.context.thread_id,
        )
        missing = await service.thread_timeline(
            tenant_id="tenant-2",
            thread_id=request.context.thread_id,
        )
    finally:
        await service.shutdown()

    assert listed["threads"][0]["title"] == "Plan the launch"
    assert timeline is not None
    assert timeline["entries"][0] == {
        "id": f"{result.run_id}:user",
        "type": "user",
        "run_id": result.run_id,
        "timestamp": result.created_at,
        "text": "Plan the launch",
        "file_ids": ["file-1"],
        "attachments": [
            {
                "id": "file-1",
                "kind": "image",
                "original_filename": "screen.png",
                "mime_type": "image/png",
                "size_bytes": 9,
                "available": True,
            }
        ],
    }
    assert timeline["entries"][-1]["type"] == "assistant"
    assert timeline["entries"][-1]["text"] == "Hello from Tulpa"
    assert any(
        entry.get("event_type") == "message.delta" for entry in timeline["entries"]
    )
    assert missing is None


def test_request_content_inlines_tenant_owned_images_and_text(tmp_path: Path) -> None:
    resolver = _AttachmentResolver(
        {
            ("tenant-1", "image-1"): (
                {
                    "original_filename": "screen.png",
                    "mime_type": "image/png",
                    "text_excerpt": "",
                },
                b"png-bytes",
            ),
            ("tenant-1", "text-1"): (
                {
                    "original_filename": "notes.txt",
                    "mime_type": "text/plain",
                    "text_excerpt": "exact file text",
                },
                b"exact file text",
            ),
        }
    )
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["unused"]),
        attachment_resolver=resolver,
    )

    content = service._request_content(  # noqa: SLF001
        AgentRunRequest(
            context=_context(),
            text="Inspect these",
            file_ids=("image-1", "text-1", "other-tenant-file"),
        )
    )

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "not filesystem paths" in content[0]["text"]
    image_blocks = [block for block in content if block["type"] == "image_url"]
    assert image_blocks == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,cG5nLWJ5dGVz"},
        }
    ]
    assert any("exact file text" in block.get("text", "") for block in content)
    assert any("unavailable to this tenant" in block.get("text", "") for block in content)
    assert resolver.reads == [("tenant-1", "image-1")]


def test_request_content_inlines_pdf_as_native_file(tmp_path: Path) -> None:
    resolver = _AttachmentResolver(
        {
            ("tenant-1", "pdf-1"): (
                {
                    "original_filename": "scan.pdf",
                    "mime_type": "application/pdf",
                    "text_excerpt": "",
                },
                b"%PDF-smoke",
            )
        }
    )
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["unused"]),
        attachment_resolver=resolver,
    )

    content = service._request_content(  # noqa: SLF001
        AgentRunRequest(context=_context(), text="Read it", file_ids=("pdf-1",))
    )

    assert isinstance(content, list)
    assert content[-1] == {
        "type": "file",
        "file": {
            "filename": "scan.pdf",
            "file_data": "data:application/pdf;base64,JVBERi1zbW9rZQ==",
        },
    }


def test_exact_owner_regenerate_command_becomes_side_effect_safe_instruction(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    request = AgentRunRequest(context=_context(), text="/regenerate")

    text = service._request_text(request)  # noqa: SLF001

    assert "immediately preceding\nowner request" in text
    assert "do not repeat tool calls, approvals, or\nexternal side effects" in text
    assert service._request_text(  # noqa: SLF001
        AgentRunRequest(context=_context(), text="/regenerate now")
    ) == "/regenerate now"
    assert service._request_text(  # noqa: SLF001
        AgentRunRequest(context=_context(), text="/regenerate", file_ids=("file-1",))
    ).startswith("/regenerate\n\nAttached tenant file IDs")


@pytest.mark.asyncio
async def test_stream_persists_ordered_events_and_snapshot(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["hello"]))
    await service.start()
    try:
        events = [
            event async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
        snapshot = await service.get_run(events[0].run_id)
    finally:
        await service.shutdown()

    restarted = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await restarted.start()
    try:
        replayed = [
            event
            async for event in restarted.events(
                events[0].run_id,
                after_sequence=1,
            )
        ]
        trace = await restarted.trace_get(
            tenant_id="tenant-1",
            run_id=events[0].run_id,
        )
    finally:
        await restarted.shutdown()

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].type == "run.started"
    assert events[-1].type == "run.completed"
    assert events[-1].data == {"text": "hello"}
    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.final_text == "hello"
    assert snapshot.context == _context()
    assert replayed == events[1:]
    assert trace is not None
    assert trace["status"] == "completed"
    assert [event["type"] for event in trace["events"]] == ["run.started", "run.completed"]
    assert trace["events"][0]["data"]["input_summary"] == "Hi"


@pytest.mark.asyncio
async def test_stream_builds_callbacks_inside_root_trace(tmp_path: Path) -> None:
    tracer = _RecordingTracer()
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["hello"]),
        langfuse_tracer=tracer,
    )
    await service.start()
    try:
        events = [
            event async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
    finally:
        await service.shutdown()

    assert events[-1].type == "run.completed"
    assert tracer.callback_contexts == [True]


@pytest.mark.asyncio
async def test_trace_reads_are_tenant_scoped_and_include_sanitized_tool_details(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    context = _context()
    other_context = context.model_copy(
        update={
            "tenant_id": "tenant-2",
            "actor_id": "owner-2",
            "thread_id": "thread-2",
            "correlation_id": "correlation-2",
            "agent_spec": AgentSpecRef(
                tenant_id="tenant-2",
                spec_id="owner",
                revision=1,
            ),
        }
    )
    try:
        await service._insert_run(  # noqa: SLF001
            "trace-run-1",
            service._checkpoint_thread_id(context),  # noqa: SLF001
            context,
        )
        await service._append_event(  # noqa: SLF001
            run_id="trace-run-1",
            type="run.started",
            data={"thread_id": context.thread_id, "resumed": False},
        )
        messages = (
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "content_fetch",
                        "args": {
                            "url": "https://example.com/?token=url-secret",
                            "authorization": "Bearer header-secret",
                        },
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                name="content_fetch",
                tool_call_id="call-1",
                status="error",
                content=json.dumps(
                    {
                        "token": "result-secret",
                        "message": "Authorization: Bearer provider-secret",
                    }
                ),
            ),
        )
        for message in messages:
            for event_type, event_data, _ in service._message_events(message):  # noqa: SLF001
                await service._append_event(  # noqa: SLF001
                    run_id="trace-run-1",
                    type=event_type,
                    data=event_data,
                )
        await service._transition_with_event(  # noqa: SLF001
            "trace-run-1",
            allowed_statuses={"running"},
            status="completed",
            event_type="run.completed",
            event_data={"text": "finished"},
            final_text="finished",
        )
        await service._insert_run(  # noqa: SLF001
            "trace-run-other",
            service._checkpoint_thread_id(other_context),  # noqa: SLF001
            other_context,
        )
        listed = await service.trace_list(tenant_id="tenant-1", status="completed")
        detail = await service.trace_get(tenant_id="tenant-1", run_id="trace-run-1")
        persisted = [event async for event in service.events("trace-run-1")]
        cross_tenant = await service.trace_get(
            tenant_id="tenant-2",
            run_id="trace-run-1",
        )
    finally:
        await service.shutdown()

    assert [item["run_id"] for item in listed] == ["trace-run-1"]
    assert listed[0]["tool_count"] == 1
    assert listed[0]["failed_tool_count"] == 1
    assert detail is not None
    assert detail["tool_count"] == 1
    assert detail["failed_tool_count"] == 1
    assert detail["events"][-1]["data"] == {}
    serialized = json.dumps(detail, sort_keys=True)
    persisted_serialized = json.dumps(
        [{"type": event.type, "data": event.data} for event in persisted],
        sort_keys=True,
    )
    assert "url-secret" not in serialized
    assert "header-secret" not in serialized
    assert "result-secret" not in serialized
    assert "provider-secret" not in serialized
    assert "url-secret" not in persisted_serialized
    assert "header-secret" not in persisted_serialized
    assert "result-secret" not in persisted_serialized
    assert "provider-secret" not in persisted_serialized
    assert context.thread_id not in serialized
    assert serialized.count("[redacted]") >= 4
    assert cross_tenant is None


@pytest.mark.asyncio
async def test_trace_list_pages_by_tenant_scoped_run_cursor(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    context = _context()
    try:
        for run_id, created_at in (
            ("trace-run-older", "2026-01-01T00:00:00+00:00"),
            ("trace-run-newer", "2026-01-02T00:00:00+00:00"),
        ):
            await service._insert_run(  # noqa: SLF001
                run_id,
                service._checkpoint_thread_id(context),  # noqa: SLF001
                context,
            )
            await service._transition_with_event(  # noqa: SLF001
                run_id,
                allowed_statuses={"running"},
                status="completed",
                event_type="run.completed",
                event_data={"text": run_id},
                final_text=run_id,
            )
            db = service._require_runs_db()  # noqa: SLF001
            await db.execute(
                "UPDATE agent_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
                (created_at, created_at, run_id),
            )
            await db.commit()

        first_page = await service.trace_list(
            tenant_id=context.tenant_id,
            status="completed",
            limit=1,
        )
        second_page = await service.trace_list(
            tenant_id=context.tenant_id,
            status="completed",
            limit=1,
            before_run_id=first_page[-1]["run_id"],
        )
        unavailable_cursor = await service.trace_list(
            tenant_id=context.tenant_id,
            before_run_id="another-tenant-run",
        )
    finally:
        await service.shutdown()

    assert [item["run_id"] for item in first_page] == ["trace-run-newer"]
    assert [item["run_id"] for item in second_page] == ["trace-run-older"]
    assert unavailable_cursor == []


@pytest.mark.asyncio
async def test_exact_agent_specs_isolate_same_thread_history_and_unresolved_runs(
    tmp_path: Path,
) -> None:
    specs = AgentSpecStore(tmp_path / "agent-specs.db")
    first_spec = specs.create_revision(
        tenant_id="tenant-1",
        spec_id="routine-a",
        write=AgentSpecWrite(
            name="Routine A",
            runtime_profile="routine",
            instructions="Run A.",
            isolation="private",
            tool_policy="allowlist",
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=None,
        created_by="owner-1",
    )
    second_spec = specs.create_revision(
        tenant_id="tenant-1",
        spec_id="routine-b",
        write=AgentSpecWrite(
            name="Routine B",
            runtime_profile="routine",
            instructions="Run B.",
            isolation="private",
            tool_policy="allowlist",
            memory_scope="none",
            workspace_scope="none",
        ),
        expected_revision=None,
        created_by="owner-1",
    )

    def context(spec: AgentSpecRef) -> AgentRunContext:
        return AgentRunContext(
            tenant_id="tenant-1",
            actor_id="scheduler",
            thread_id="same-logical-thread",
            channel="routine",
            run_kind="routine",
            correlation_id=f"correlation:{spec.spec_id}",
            origin=OriginRef(interface="routine", source_id="scheduler"),
            agent_spec=spec,
            trust_class="background",
        )

    first_context = context(first_spec.ref)
    second_context = context(second_spec.ref)
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["answer-a", "answer-b"]),
        agent_specs=specs,
    )
    await service.start()
    try:
        await service.run(AgentRunRequest(context=first_context, text="history-a"))
        await service.run(AgentRunRequest(context=second_context, text="history-b"))
        first_graph = service._graph_for_context(first_context)  # noqa: SLF001
        second_graph = service._graph_for_context(second_context)  # noqa: SLF001
        first_state = await first_graph.aget_state(
            service._run_config(  # noqa: SLF001
                first_context,
                service._checkpoint_thread_id(first_context),  # noqa: SLF001
            )
        )
        second_state = await second_graph.aget_state(
            service._run_config(  # noqa: SLF001
                second_context,
                service._checkpoint_thread_id(second_context),  # noqa: SLF001
            )
        )
        first_history = "\n".join(
            service._message_text(message)  # noqa: SLF001
            for message in first_state.values["messages"]
        )
        second_history = "\n".join(
            service._message_text(message)  # noqa: SLF001
            for message in second_state.values["messages"]
        )

        await service._insert_run(  # noqa: SLF001
            "run-routine-a-pending",
            service._checkpoint_thread_id(first_context),  # noqa: SLF001
            first_context,
        )
        with pytest.raises(AgentRunCheckpointConflictError):
            await service._prepare_run(  # noqa: SLF001
                AgentRunRequest(context=first_context, text="blocked")
            )
        prepared_other = await service._prepare_run(  # noqa: SLF001
            AgentRunRequest(context=second_context, text="not blocked")
        )
    finally:
        await service.shutdown()

    assert "history-a" in first_history and "history-b" not in first_history
    assert "history-b" in second_history and "history-a" not in second_history
    assert prepared_other.created is True
    assert service._checkpoint_thread_id(first_context) != service._checkpoint_thread_id(
        second_context
    )


@pytest.mark.asyncio
async def test_terminal_run_observer_sees_persisted_snapshot(tmp_path: Path) -> None:
    observed: list[Any] = []

    async def observe(snapshot: Any) -> None:
        observed.append(snapshot)

    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["hello"]),
        run_observer=observe,
    )
    await service.start()
    try:
        events = [
            event async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
    finally:
        await service.shutdown()

    assert len(observed) == 1
    assert observed[0].run_id == events[0].run_id
    assert observed[0].status == "completed"
    assert observed[0].final_text == "hello"


@pytest.mark.asyncio
async def test_summarization_tokens_are_not_exposed_as_agent_output(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _SummarizingGraph()  # noqa: SLF001
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
        snapshot = await service.get_run(events[0].run_id)
    finally:
        await service.shutdown()

    assert [event.type for event in events] == [
        "run.started",
        "message.delta",
        "run.completed",
    ]
    assert events[1].data["text"] == "visible answer"
    assert snapshot is not None
    assert snapshot.final_text == "visible answer"


@pytest.mark.asyncio
async def test_chunked_tool_calls_emit_one_complete_start_event(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _ChunkedToolGraph()  # noqa: SLF001
    context = _context()
    try:
        events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=context, text="Find cats")
            )
        ]
        timeline = await service.thread_timeline(
            tenant_id=context.tenant_id,
            thread_id=context.thread_id,
        )
    finally:
        await service.shutdown()

    assert [event.type for event in events] == [
        "run.started",
        "tool.started",
        "tool.completed",
        "message.delta",
        "run.completed",
    ]
    assert events[1].data == {
        "name": "web_search",
        "call_id": "call-search",
        "arguments": {"query": "cats"},
    }
    assert events[2].data["call_id"] == "call-search"
    assert events[-1].data["text"] == "Found cats."
    assert timeline is not None
    assert [
        entry["event_type"]
        for entry in timeline["entries"]
        if entry.get("event_type")
    ] == ["run.started", "tool.started", "tool.completed", "message.delta", "run.completed"]
    started = next(
        entry for entry in timeline["entries"] if entry.get("event_type") == "run.started"
    )
    assert started["data"]["provider"] == "api"
    assert started["data"]["model"] == "test-model"


@pytest.mark.asyncio
async def test_run_failure_is_sanitized_for_events_and_persisted_status(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_error = "provider token sk-secret leaked from /srv/private/.env"
    prompt_secret = "prompt-password"
    tracer = _RecordingTracer()
    service = _service(
        tmp_path,
        _ToolCapableTextModel(responses=["unused"]),
        langfuse_tracer=tracer,
    )
    await service.start()
    service._graphs["owner"] = _FailingGraph(private_error)  # noqa: SLF001
    caplog.set_level(logging.ERROR, logger="opentulpa.deep_agent.service")
    try:
        events = [
            event
            async for event in service.stream(
                AgentRunRequest(
                    context=_context(),
                    text=f"Diagnose this; password is {prompt_secret}",
                )
            )
        ]
        snapshot = await service.get_run(events[0].run_id)
    finally:
        await service.shutdown()

    assert events[-1].type == "run.failed"
    assert events[-1].data["code"] == "agent_run_failed"
    assert events[-1].data["message"] == "The agent run could not be completed."
    assert events[-1].data["retryable"] is False
    assert events[-1].data["diagnostic"]["error_type"] == "RuntimeError"
    assert events[-1].data["diagnostic"]["phase"] == "agent_loop"
    assert events[-1].data["diagnostic"]["location"].startswith(
        "test_deep_agent_service.py:astream:"
    )
    assert "[redacted]" in events[-1].data["diagnostic"]["cause"]
    assert "[redacted-path]" in events[-1].data["diagnostic"]["cause"]
    assert len(events[-1].data["diagnostic"]["fingerprint"]) == 16
    assert snapshot is not None
    assert snapshot.status == "failed"
    assert snapshot.error == "The agent run could not be completed."
    assert private_error not in str(events[-1].data)
    assert "/srv/private/.env" not in str(events[-1].data)
    assert prompt_secret not in str(events[0].data)
    assert "[redacted]" in events[0].data["input_summary"]
    assert private_error not in snapshot.error
    assert private_error in caplog.text
    assert [str(error) for error in tracer.errors] == [private_error]


@pytest.mark.asyncio
async def test_provider_bad_request_has_actionable_public_failure(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _ProviderRejectedGraph()  # noqa: SLF001
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
        snapshot = await service.get_run(events[0].run_id)
    finally:
        await service.shutdown()

    assert events[-1].type == "run.failed"
    assert events[-1].data["code"] == "model_provider_rejected"
    assert events[-1].data["message"] == (
        "The model provider rejected this conversation. Start a new thread or use a different "
        "model/provider."
    )
    assert events[-1].data["retryable"] is False
    assert snapshot is not None
    assert snapshot.error == events[-1].data["message"]
    assert "private provider response" not in snapshot.error


@pytest.mark.asyncio
async def test_exhausted_provider_routes_have_specific_public_failure(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _ProviderUnavailableGraph()  # noqa: SLF001
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
    finally:
        await service.shutdown()

    assert events[-1].type == "run.failed"
    assert events[-1].data["code"] == "model_provider_failed"
    assert events[-1].data["message"] == (
        "No configured model provider could complete this request. Try again later."
    )
    assert events[-1].data["retryable"] is True


@pytest.mark.asyncio
async def test_codex_overload_has_specific_retryable_public_failure(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _CodexOverloadedGraph()  # noqa: SLF001
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
    finally:
        await service.shutdown()

    assert events[-1].type == "run.failed"
    assert events[-1].data["code"] == "model_provider_failed"
    assert events[-1].data["message"] == (
        "No configured model provider could complete this request. Try again later."
    )
    assert events[-1].data["retryable"] is True


@pytest.mark.asyncio
async def test_provider_failure_preserves_latest_actionable_tool_error(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _ToolErrorThenCodexOverloadedGraph()  # noqa: SLF001
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Publish"))
        ]
        snapshot = await service.get_run(events[0].run_id)
    finally:
        await service.shutdown()

    assert events[-1].type == "run.failed"
    assert events[-1].data["last_tool_error"] == {
        "tool_name": "repository_publish_pr",
        "message": "GitHub publishing is not configured.",
    }
    assert events[-1].data["message"].endswith(
        "The last operation also failed: GitHub publishing is not configured."
    )
    assert snapshot is not None
    assert snapshot.error == events[-1].data["message"]


@pytest.mark.asyncio
async def test_provider_rejection_after_output_preserves_completed_response(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["owner"] = _ProviderRejectedAfterOutputGraph()  # noqa: SLF001
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Hi"))
        ]
        snapshot = await service.get_run(events[0].run_id)
    finally:
        await service.shutdown()

    assert [event.type for event in events] == [
        "run.started",
        "message.delta",
        "run.completed",
    ]
    assert events[-1].data["text"] == "usable streamed response"
    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.final_text == "usable streamed response"
    assert snapshot.error == ""


@pytest.mark.asyncio
async def test_provider_rejection_retries_same_model_call_with_fallback() -> None:
    calls: list[Any] = []

    async def handler(request: _MiddlewareRequest) -> str:
        calls.append(request.model)
        if len(calls) == 1:
            raise BadRequestResponseError("rejected")
        return "fallback response"

    middleware = _ProviderFallbackMiddleware(["glm-5.2"])
    result = await middleware.awrap_model_call(_MiddlewareRequest("kimi-k3"), handler)

    assert result == "fallback response"
    assert calls == ["kimi-k3", "glm-5.2"]


@pytest.mark.asyncio
async def test_provider_failure_advances_through_entire_fallback_chain() -> None:
    calls: list[Any] = []

    async def handler(request: _MiddlewareRequest) -> str:
        calls.append(request.model)
        if len(calls) < 3:
            raise ValueError("OpenRouter API returned an error: provider unavailable")
        return "second fallback response"

    middleware = _ProviderFallbackMiddleware(["glm-5.2", "gemini-3.1-pro"])
    result = await middleware.awrap_model_call(_MiddlewareRequest("kimi-k3"), handler)

    assert result == "second fallback response"
    assert calls == ["kimi-k3", "glm-5.2", "gemini-3.1-pro"]


@pytest.mark.asyncio
async def test_non_provider_failure_does_not_advance_fallback_chain() -> None:
    calls: list[Any] = []

    async def handler(request: _MiddlewareRequest) -> str:
        calls.append(request.model)
        raise RuntimeError("application bug")

    middleware = _ProviderFallbackMiddleware(["glm-5.2", "gemini-3.1-pro"])
    with pytest.raises(RuntimeError, match="application bug"):
        await middleware.awrap_model_call(_MiddlewareRequest("kimi-k3"), handler)

    assert calls == ["kimi-k3"]


@pytest.mark.asyncio
async def test_streaming_content_rejection_also_uses_fallback() -> None:
    calls: list[Any] = []

    async def handler(request: _MiddlewareRequest) -> str:
        calls.append(request.model)
        if len(calls) == 1:
            raise ValueError(
                "OpenRouter API returned an error during streaming: "
                "Output data may contain inappropriate content"
            )
        return "fallback response"

    middleware = _ProviderFallbackMiddleware(["glm-5.2"])
    result = await middleware.awrap_model_call(_MiddlewareRequest("kimi-k3"), handler)

    assert result == "fallback response"
    assert calls == ["kimi-k3", "glm-5.2"]


@pytest.mark.asyncio
async def test_closing_stream_does_not_cancel_detached_run(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    graph = _SerializedGraph()
    service._graphs["owner"] = graph  # noqa: SLF001
    events = cast(
        AsyncGenerator[Any, None],
        service.stream(AgentRunRequest(context=_context(), text="Hi")),
    )
    try:
        started = await anext(events)
        await asyncio.wait_for(graph.first_entered.wait(), timeout=1)
        await events.aclose()
        disconnected = await service.get_run(started.run_id)
        graph.release_first.set()
        replayed = [
            event
            async for event in service.events(started.run_id, after_sequence=started.sequence)
        ]
        completed = await service.get_run(started.run_id)
    finally:
        graph.release_first.set()
        await service.shutdown()

    assert started.type == "run.started"
    assert disconnected is not None and disconnected.status == "running"
    assert completed is not None and completed.status == "completed"
    assert replayed[-1].type == "run.completed"
    assert replayed[-1].data == {"text": "response-1"}


@pytest.mark.asyncio
async def test_cancel_race_cannot_append_completion_after_terminal_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["done"]))
    await service.start()
    entered_completion = asyncio.Event()
    release_completion = asyncio.Event()
    original_transition = service._transition_with_event

    async def delayed_transition(run_id: str, **kwargs: Any) -> Any:
        if kwargs.get("status") == "completed":
            entered_completion.set()
            await release_completion.wait()
        return await original_transition(run_id, **kwargs)

    monkeypatch.setattr(service, "_transition_with_event", delayed_transition)
    stream = cast(
        AsyncGenerator[Any, None],
        service.stream(AgentRunRequest(context=_context(), text="Finish")),
    )
    try:
        started = await anext(stream)

        async def collect_tail() -> list[Any]:
            return [event async for event in stream]

        tail_task = asyncio.create_task(collect_tail())
        await asyncio.wait_for(entered_completion.wait(), timeout=1)
        await service.cancel(started.run_id)
        release_completion.set()
        tail = await asyncio.wait_for(tail_task, timeout=1)
        snapshot = await service.get_run(started.run_id)
        replayed = [event async for event in service.events(started.run_id)]
    finally:
        release_completion.set()
        await stream.aclose()
        await service.shutdown()

    assert snapshot is not None and snapshot.status == "cancelled"
    assert all(event.type != "run.completed" for event in tail)
    assert [event.type for event in replayed][-1] == "run.failed"
    assert sum(event.type in {"run.completed", "run.failed"} for event in replayed) == 1


@pytest.mark.asyncio
async def test_run_idempotency_replays_the_original_durable_event_stream(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["Only once"]))
    request = AgentRunRequest(
        context=_context(),
        text="Perform one interface event",
        idempotency_key="telegram:update:42",
    )
    await service.start()
    try:
        first = [event async for event in service.stream(request)]
        retry = AgentRunRequest(
            context=request.context.model_copy(update={"correlation_id": "retry-correlation"}),
            text=request.text,
            file_ids=request.file_ids,
            idempotency_key=request.idempotency_key,
        )
        replay = [event async for event in service.stream(retry)]
    finally:
        await service.shutdown()

    assert first
    assert replay == first
    assert len({event.run_id for event in (*first, *replay)}) == 1


@pytest.mark.asyncio
async def test_run_idempotency_rejects_changed_request_identity_or_payload(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["Only once"]))
    context = _context()
    original = AgentRunRequest(
        context=context,
        text="Original",
        file_ids=("file-1",),
        idempotency_key="interface:event:7",
    )
    await service.start()
    try:
        await service.run(original)
        variants = (
            original.context.model_copy(update={"actor_id": "other-actor"}),
            original.context.model_copy(update={"thread_id": "other-thread"}),
            original.context.model_copy(
                update={
                    "agent_spec": AgentSpecRef(
                        tenant_id=context.tenant_id,
                        spec_id="owner",
                        revision=2,
                    )
                }
            ),
        )
        changed_requests = [
            AgentRunRequest(
                context=variant,
                text=original.text,
                file_ids=original.file_ids,
                idempotency_key=original.idempotency_key,
            )
            for variant in variants
        ]
        changed_requests.extend(
            (
                AgentRunRequest(
                    context=context,
                    text="Changed",
                    file_ids=original.file_ids,
                    idempotency_key=original.idempotency_key,
                ),
                AgentRunRequest(
                    context=context,
                    text=original.text,
                    file_ids=("file-2",),
                    idempotency_key=original.idempotency_key,
                ),
            )
        )
        for changed in changed_requests:
            with pytest.raises(AgentRunIdempotencyConflictError):
                await service.open_stream(changed)
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_stream_consumer_cancellation_requires_explicit_run_cancel(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    graph = _BlockingGraph()
    service._graphs["owner"] = graph  # noqa: SLF001
    events = cast(
        AsyncGenerator[Any, None],
        service.stream(AgentRunRequest(context=_context(), text="Hi")),
    )
    try:
        started = await anext(events)

        async def advance() -> Any:
            return await anext(events)

        pending = asyncio.create_task(advance())
        await asyncio.wait_for(graph.entered.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        disconnected = await service.get_run(started.run_id)
        await events.aclose()
        cancelled = await service.cancel(started.run_id)
    finally:
        await service.shutdown()

    assert disconnected is not None and disconnected.status == "running"
    assert cancelled.status == "cancelled"


@pytest.mark.asyncio
async def test_cancel_thread_stops_the_owned_active_run_without_a_run_id(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    graph = _BlockingGraph()
    service._graphs["owner"] = graph  # noqa: SLF001
    events = cast(
        AsyncGenerator[Any, None],
        service.stream(AgentRunRequest(context=_context(), text="Hi")),
    )
    try:
        started = await anext(events)
        await asyncio.wait_for(graph.entered.wait(), timeout=1)
        cancelled = await service.cancel_thread(
            tenant_id=_context().tenant_id,
            thread_id=_context().thread_id,
        )
        missing = await service.cancel_thread(
            tenant_id="other-tenant",
            thread_id=_context().thread_id,
        )
    finally:
        await events.aclose()
        await service.shutdown()

    assert cancelled is not None
    assert cancelled.run_id == started.run_id
    assert cancelled.status == "cancelled"
    assert missing is None


@pytest.mark.asyncio
async def test_cancelled_run_can_continue_immediately_on_the_same_checkpoint(
    tmp_path: Path,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    graph = _SerializedGraph()
    service._graphs["owner"] = graph  # noqa: SLF001
    first = cast(
        AsyncGenerator[Any, None],
        await service.open_stream(AgentRunRequest(context=_context(), text="Original")),
    )
    try:
        started = await anext(first)
        await asyncio.wait_for(graph.first_entered.wait(), timeout=1)
        cancelled = await service.cancel(started.run_id)
        steered = cast(
            AsyncGenerator[Any, None],
            await service.open_stream(
                AgentRunRequest(context=_context(), text="Use this direction instead")
            ),
        )
        steered_events = [event async for event in steered]
    finally:
        graph.release_first.set()
        await first.aclose()
        await service.shutdown()

    assert cancelled.status == "cancelled"
    assert steered_events[-1].type == "run.completed"
    assert steered_events[-1].data == {"text": "response-2"}
    assert graph.calls == 2
    assert graph.max_active == 1


@pytest.mark.asyncio
async def test_checkpoint_fence_rejects_overlap_until_prior_run_finishes(tmp_path: Path) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    graph = _SerializedGraph()
    service._graphs["owner"] = graph  # noqa: SLF001
    first = cast(
        AsyncGenerator[Any, None],
        await service.open_stream(AgentRunRequest(context=_context(), text="First")),
    )

    async def collect(events: AsyncGenerator[Any, None]) -> list[Any]:
        return [event async for event in events]

    try:
        first_started = await anext(first)
        first_tail_task = asyncio.create_task(collect(first))
        await asyncio.wait_for(graph.first_entered.wait(), timeout=1)
        with pytest.raises(AgentRunCheckpointConflictError):
            await service.open_stream(AgentRunRequest(context=_context(), text="Second"))
        graph.release_first.set()
        first_tail = await asyncio.wait_for(first_tail_task, timeout=1)
        second = cast(
            AsyncGenerator[Any, None],
            await service.open_stream(AgentRunRequest(context=_context(), text="Second")),
        )
        second_events = await asyncio.wait_for(collect(second), timeout=1)
        first_snapshot = await service.get_run(first_started.run_id)
        second_snapshot = await service.get_run(second_events[0].run_id)
    finally:
        graph.release_first.set()
        await first.aclose()
        await service.shutdown()

    assert graph.calls == 2
    assert graph.max_active == 1
    assert first_tail[-1].type == "run.completed"
    assert second_events[-1].type == "run.completed"
    assert first_snapshot is not None and first_snapshot.status == "completed"
    assert second_snapshot is not None and second_snapshot.status == "completed"


@pytest.mark.asyncio
async def test_startup_reconciles_orphaned_running_runs(tmp_path: Path) -> None:
    context = _context()
    first = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await first.start()
    await first._insert_run(  # noqa: SLF001
        "run-orphaned",
        first._checkpoint_thread_id(context),  # noqa: SLF001
        context,
    )
    await first.shutdown()

    second = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await second.start()
    try:
        snapshot = await second.get_run("run-orphaned")
        events = [event async for event in second.events("run-orphaned")]
    finally:
        await second.shutdown()

    assert snapshot is not None
    assert snapshot.status == "cancelled"
    assert snapshot.error == "The agent run was cancelled before completion."
    assert [event.type for event in events] == ["run.failed"]
    assert events[0].sequence == 1
    assert events[0].data["code"] == "agent_run_cancelled"


@pytest.mark.asyncio
async def test_approval_interrupt_survives_restart_and_resumes(tmp_path: Path) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Schedule deleted."),
        ]
    )
    first = _service(tmp_path, model, tools=[source_shell])
    await first.start()
    events = [
        event async for event in first.stream(AgentRunRequest(context=_context(), text="Delete it"))
    ]
    run_id = events[0].run_id
    interrupted = await first.get_run(run_id)
    await first.shutdown()

    assert interrupted is not None
    assert interrupted.status == "interrupted"
    assert len(interrupted.approvals) == 1
    assert calls == []

    second = _service(tmp_path, model, tools=[source_shell])
    await second.start()
    try:
        resumed = [
            event
            async for event in second.resume(
                run_id,
                ApprovalDecision(
                    approval_id=interrupted.approvals[0].id,
                    decision="approve",
                ),
            )
        ]
        completed = await second.get_run(run_id)
    finally:
        await second.shutdown()

    assert resumed[0].type == "run.started"
    assert resumed[0].data["resumed"] is True
    assert resumed[-1].type == "run.completed"
    assert completed is not None
    assert completed.status == "completed"
    assert completed.final_text == "Schedule deleted."
    assert calls == ["rm -rf schedule-1"]


@pytest.mark.asyncio
async def test_new_message_cannot_discard_a_pending_same_thread_approval(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Schedule deleted."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    try:
        events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Delete the schedule")
            )
        ]
        interrupted = await service.get_run(events[0].run_id)
        assert interrupted is not None
        with pytest.raises(AgentRunCheckpointConflictError):
            await service.open_stream(
                AgentRunRequest(context=_context(), text="Ignore that and do something else")
            )
        resumed = [
            event
            async for event in service.resume(
                interrupted.run_id,
                ApprovalDecision(
                    approval_id=interrupted.approvals[0].id,
                    decision="approve",
                ),
            )
        ]
    finally:
        await service.shutdown()

    assert calls == ["rm -rf schedule-1"]
    assert resumed[-1].type == "run.completed"


@pytest.mark.asyncio
async def test_claimed_approval_resume_recovers_after_process_restart(tmp_path: Path) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Recovered after restart."),
        ]
    )
    first = _service(tmp_path, model, tools=[source_shell])
    await first.start()
    interrupted_events = [
        event
        async for event in first.stream(
            AgentRunRequest(context=_context(), text="Delete after approval")
        )
    ]
    run_id = interrupted_events[0].run_id
    interrupted = await first.get_run(run_id)
    assert interrupted is not None
    decision = ApprovalDecision(
        approval_id=interrupted.approvals[0].id,
        decision="approve",
    )
    await first._claim_approval_decision(run_id, decision)  # noqa: SLF001
    claimed = await first.get_run(run_id)
    assert claimed is not None and claimed.status == "resume_pending"
    await first.shutdown()

    second = _service(tmp_path, model, tools=[source_shell])
    await second.start()
    try:
        for _ in range(100):
            completed = await second.get_run(run_id)
            if completed is not None and completed.status in {
                "completed",
                "failed",
                "cancelled",
            }:
                break
            await asyncio.sleep(0.01)
        replayed = [event async for event in second.events(run_id)]
    finally:
        await second.shutdown()

    assert completed is not None
    assert completed.status == "completed"
    assert completed.final_text == "Recovered after restart."
    assert replayed[-1].type == "run.completed"
    assert calls == ["rm -rf schedule-1"]


@pytest.mark.asyncio
async def test_source_release_hands_off_to_restarted_runtime(
    tmp_path: Path,
) -> None:
    release_started = asyncio.Event()
    release_responses: dict[str, dict[str, str]] = {}
    release_calls: list[str] = []

    @tool("source_release")
    async def source_release(
        idempotency_key: str,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        message: str,
    ) -> dict[str, str]:
        """Activate an exact, owner-approved source candidate."""

        del expected_diff_sha256, message
        existing = release_responses.get(idempotency_key)
        if existing is not None:
            return existing
        result = {
            "candidate_id": expected_candidate_id,
            "status": "queued",
        }
        release_responses[idempotency_key] = result
        release_calls.append(idempotency_key)
        release_started.set()
        await asyncio.Event().wait()
        return result

    digest = "a" * 64
    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_release",
                        "args": {
                            "idempotency_key": "release-key-1",
                            "expected_candidate_id": "candidate-1",
                            "expected_diff_sha256": digest,
                            "message": "Deploy the tested source",
                        },
                        "id": "call-release",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The new source release is active."),
        ]
    )
    first = _service(tmp_path, model, tools=[source_release])
    await first.start()
    events = await first.open_stream(
        AgentRunRequest(context=_context(), text="Release the tested change")
    )
    started = await anext(events)
    run_id = started.run_id
    await asyncio.wait_for(release_started.wait(), timeout=1)
    await events.aclose()
    await first.shutdown()

    second = _service(tmp_path, model, tools=[source_release])
    await second.start()
    try:
        for _ in range(200):
            completed = await second.get_run(run_id)
            if completed is not None and completed.status in {
                "completed",
                "failed",
                "cancelled",
            }:
                break
            await asyncio.sleep(0.01)
        replayed = [event async for event in second.events(run_id)]
    finally:
        await second.shutdown()

    assert completed is not None
    assert completed.status == "completed"
    assert completed.final_text == "The new source release is active."
    assert replayed[-1].type == "run.completed"
    assert all(event.type != "approval.required" for event in replayed)
    assert (
        sum(event.type == "run.started" and event.data.get("resumed") is True for event in replayed)
        == 2
    )
    assert release_calls == ["release-key-1"]


@pytest.mark.asyncio
async def test_source_release_does_not_surface_alongside_shell_approval(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    @tool("source_release")
    def source_release(message: str) -> dict[str, str]:
        """Activate a source candidate without a user-facing approval."""

        calls.append(message)
        return {"message": message}

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_release",
                        "args": {"message": "release candidate-1"},
                        "id": "call-release",
                        "type": "tool_call",
                    },
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf build"},
                        "id": "call-shell",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="Release completed."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_release, source_shell])
    await service.start()
    try:
        events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Release and clean up")
            )
        ]
        interrupted = await service.get_run(events[0].run_id)
        assert interrupted is not None
        pending = [approval for approval in interrupted.approvals if approval.status == "pending"]
        resumed = [
            event
            async for event in service.resume(
                interrupted.run_id,
                ApprovalDecision(
                    approval_id=pending[0].id,
                    decision="approve",
                ),
            )
        ]
    finally:
        await service.shutdown()

    surfaced = [event.data["tool_name"] for event in events if event.type == "approval.required"]
    assert surfaced == ["source_shell"]
    assert [approval.tool_name for approval in pending] == ["source_shell"]
    assert resumed[-1].type == "run.completed"
    assert calls == ["release candidate-1", "rm -rf build"]


@pytest.mark.asyncio
async def test_closed_resume_stream_continues_from_durable_decision(tmp_path: Path) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Continued without the SSE client."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    resume_events: AsyncGenerator[Any, None] | None = None
    try:
        interrupted_events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Delete after approval")
            )
        ]
        run_id = interrupted_events[0].run_id
        interrupted = await service.get_run(run_id)
        assert interrupted is not None
        resume_events = cast(
            AsyncGenerator[Any, None],
            service.resume(
                run_id,
                ApprovalDecision(
                    approval_id=interrupted.approvals[0].id,
                    decision="approve",
                ),
            ),
        )
        started = await anext(resume_events)
        assert started.type == "run.started"
        await resume_events.aclose()
        for _ in range(100):
            completed = await service.get_run(run_id)
            if completed is not None and completed.status == "completed":
                break
            await asyncio.sleep(0.01)
        replayed = [event async for event in service.events(run_id)]
    finally:
        if resume_events is not None:
            await resume_events.aclose()
        await service.shutdown()

    assert completed is not None and completed.status == "completed"
    assert completed.final_text == "Continued without the SSE client."
    assert replayed[-1].type == "run.completed"
    assert calls == ["rm -rf schedule-1"]


@pytest.mark.asyncio
async def test_unconsumed_resume_stream_continues_from_durable_decision(tmp_path: Path) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Continued before the SSE body started."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    resume_events: AsyncIterator[Any] | None = None
    try:
        interrupted_events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Delete after approval")
            )
        ]
        run_id = interrupted_events[0].run_id
        interrupted = await service.get_run(run_id)
        assert interrupted is not None
        resume_events = await service.open_resume(
            run_id,
            ApprovalDecision(
                approval_id=interrupted.approvals[0].id,
                decision="approve",
            ),
        )
        for _ in range(100):
            completed = await service.get_run(run_id)
            if completed is not None and completed.status == "completed":
                break
            await asyncio.sleep(0.01)
        replayed = [event async for event in service.events(run_id)]
    finally:
        if resume_events is not None:
            close = getattr(resume_events, "aclose", None)
            if callable(close):
                await close()
        await service.shutdown()

    assert completed is not None and completed.status == "completed"
    assert completed.final_text == "Continued before the SSE body started."
    assert replayed[-1].type == "run.completed"
    assert calls == ["rm -rf schedule-1"]


@pytest.mark.asyncio
async def test_concurrent_resume_claims_one_approval_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Schedule deleted."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    try:
        interrupted_events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Delete it"))
        ]
        run_id = interrupted_events[0].run_id
        interrupted = await service.get_run(run_id)
        assert interrupted is not None

        original_get_run = service.get_run
        both_read = asyncio.Event()
        read_count = 0

        async def synchronized_get_run(candidate: str) -> Any:
            nonlocal read_count
            snapshot = await original_get_run(candidate)
            if candidate == run_id and snapshot is not None and snapshot.status == "interrupted":
                read_count += 1
                if read_count == 2:
                    both_read.set()
                await asyncio.wait_for(both_read.wait(), timeout=1)
            return snapshot

        monkeypatch.setattr(service, "get_run", synchronized_get_run)
        decision = ApprovalDecision(
            approval_id=interrupted.approvals[0].id,
            decision="approve",
        )

        async def consume_resume() -> list[Any]:
            return [event async for event in service.resume(run_id, decision)]

        results = await asyncio.gather(
            consume_resume(),
            consume_resume(),
            return_exceptions=True,
        )
        completed = await original_get_run(run_id)
    finally:
        await service.shutdown()

    successes = [result for result in results if isinstance(result, list)]
    conflicts = [result for result in results if isinstance(result, ValueError)]
    assert len(successes) == 1
    assert successes[0][-1].type == "run.completed"
    assert len(conflicts) == 1
    assert "approval state changed" in str(conflicts[0])
    assert completed is not None
    assert completed.status == "completed"
    assert calls == ["rm -rf schedule-1"]


@pytest.mark.asyncio
async def test_reject_skips_external_effect(tmp_path: Path) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf job-1"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Cancellation rejected."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    try:
        events = [
            event
            async for event in service.stream(AgentRunRequest(context=_context(), text="Cancel"))
        ]
        snapshot = await service.get_run(events[0].run_id)
        assert snapshot is not None
        resumed = [
            event
            async for event in service.resume(
                snapshot.run_id,
                ApprovalDecision(
                    approval_id=snapshot.approvals[0].id,
                    decision="reject",
                    message="Keep it running",
                ),
            )
        ]
    finally:
        await service.shutdown()

    assert resumed[-1].type == "run.completed"
    assert calls == []


@pytest.mark.asyncio
async def test_sequential_interrupts_have_distinct_approval_bindings(tmp_path: Path) -> None:
    calls: list[str] = []
    source_shell = _destructive_shell_tool(calls)

    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-a"},
                        "id": "call-a",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "source_shell",
                        "args": {"command": "rm -rf schedule-b"},
                        "id": "call-b",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Both approved."),
        ]
    )
    service = _service(tmp_path, model, tools=[source_shell])
    await service.start()
    try:
        first_events = [
            event
            async for event in service.stream(
                AgentRunRequest(context=_context(), text="Delete both schedules")
            )
        ]
        first = await service.get_run(first_events[0].run_id)
        assert first is not None
        first_approval = first.approvals[0]
        first_resume = [
            event
            async for event in service.resume(
                first.run_id,
                ApprovalDecision(approval_id=first_approval.id, decision="approve"),
            )
        ]
        second = await service.get_run(first.run_id)
        assert second is not None
        second_approval = second.approvals[0]
        with pytest.raises(KeyError, match="approval not found"):
            await service.open_resume(
                first.run_id,
                ApprovalDecision(approval_id=first_approval.id, decision="approve"),
            )
        final_events = [
            event
            async for event in service.resume(
                first.run_id,
                ApprovalDecision(approval_id=second_approval.id, decision="approve"),
            )
        ]
    finally:
        await service.shutdown()

    assert first_approval.id != second_approval.id
    assert first_resume[-1].type == "approval.required"
    assert calls == ["rm -rf schedule-a", "rm -rf schedule-b"]
    assert final_events[-1].type == "run.completed"


@pytest.mark.asyncio
async def test_intake_profile_returns_typed_decision_without_product_writes(
    tmp_path: Path,
) -> None:
    model = _ToolCapableMessageModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntakeDecision",
                        "args": {
                            "action": "request_fields",
                            "reply_text": "What day works for you?",
                            "booking_patch": {
                                "fields": {"service": "consultation"},
                                "missing_fields": ["date"],
                            },
                            "evidence_source_ids": ["message-1"],
                        },
                        "id": "decision-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    service = _service(tmp_path, model)
    await service.start()
    try:
        context = _context().model_copy(
            update={
                "actor_id": "intake",
                "thread_id": "intake:workflow-1:conversation-1",
                "channel": AgentChannel.INTAKE,
                "run_kind": AgentRunKind.INTAKE,
            }
        )
        decision = await service.decide_intake(
            context=context,
            decision_input={"conversation": [{"id": "message-1", "text": "Consultation"}]},
        )
    finally:
        await service.shutdown()

    assert decision.action == "request_fields"
    assert decision.booking_patch is not None
    assert decision.booking_patch.missing_fields == ["date"]
    assert decision.evidence_source_ids == ["message-1"]


@pytest.mark.asyncio
async def test_intake_decisions_serialize_on_the_checkpoint_thread(tmp_path: Path) -> None:
    graph = _SerializedIntakeGraph()
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["intake"] = graph  # noqa: SLF001
    context = _context().model_copy(
        update={
            "actor_id": "intake",
            "channel": AgentChannel.INTAKE,
            "run_kind": AgentRunKind.INTAKE,
        }
    )
    first = asyncio.create_task(
        service.decide_intake(context=context, decision_input={"message": "one"})
    )
    try:
        await graph.first_entered.wait()
        second = asyncio.create_task(
            service.decide_intake(context=context, decision_input={"message": "two"})
        )
        await asyncio.sleep(0.02)
        assert graph.calls == 1
        graph.release_first.set()
        decisions = await asyncio.gather(first, second)
    finally:
        graph.release_first.set()
        if not first.done():
            first.cancel()
        await service.shutdown()

    assert [decision.action for decision in decisions] == ["ignore", "ignore"]
    assert graph.max_active == 1


@pytest.mark.asyncio
async def test_intake_decision_honors_agent_runtime_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service(tmp_path, _ToolCapableTextModel(responses=["unused"]))
    await service.start()
    service._graphs["intake"] = _BlockingIntakeGraph()  # noqa: SLF001
    monkeypatch.setattr(service, "_runtime_limit_for_context", lambda _context: 0.01)
    context = _context().model_copy(
        update={
            "actor_id": "intake",
            "channel": AgentChannel.INTAKE,
            "run_kind": AgentRunKind.INTAKE,
        }
    )
    try:
        with pytest.raises(TimeoutError):
            await service.decide_intake(context=context, decision_input={"message": "wait"})
    finally:
        await service.shutdown()
