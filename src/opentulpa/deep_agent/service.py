"""Application lifecycle, persistence, and streaming around the Deep Agents graph."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import nullcontext, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, cast
from weakref import WeakValueDictionary

import aiosqlite
import httpx
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.middleware.filesystem import FilesystemPermission
from langchain.agents.middleware import (
    AgentMiddleware,
    InterruptOnConfig,
    ModelCallLimitMiddleware,
)
from langchain_core.language_models import ModelProfile
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Command
from pydantic import SecretStr

from opentulpa.core.ids import new_short_id
from opentulpa.deep_agent.contracts import (
    AgentApproval,
    AgentApprovalStatus,
    AgentRunCapabilityConflictError,
    AgentRunCheckpointConflictError,
    AgentRunContext,
    AgentRunEvent,
    AgentRunEventType,
    AgentRunIdempotencyConflictError,
    AgentRunRequest,
    AgentRunSnapshot,
    AgentRunStatus,
    ApprovalDecision,
    utc_now_iso,
)
from opentulpa.deep_agent.dynamic_tools import DynamicToolProvider, DynamicToolSnapshot
from opentulpa.deep_agent.prompts import INTAKE_PROMPT, OWNER_PROMPT, ROUTINE_PROMPT
from opentulpa.deep_agent.sandbox import (
    TenantContainerPolicy,
    TenantExecutionProvider,
    TenantSandboxBackend,
)
from opentulpa.deep_agent.shell_policy import (
    ShellCommandDisposition,
    classify_shell_command,
)
from opentulpa.inference.codex import is_transient as is_codex_transient
from opentulpa.inference.codex import is_unauthorized as is_codex_unauthorized
from opentulpa.inference.models import (
    InferenceProvider,
    InferenceSelection,
    ResolvedInferencePlan,
)
from opentulpa.inference.service import (
    InferenceConflictError,
    InferenceService,
    InferenceUnavailableError,
    ResolvedModel,
)
from opentulpa.intake.decision import IntakeDecision
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.persistence.tenant_namespace import (
    tenant_namespace_label,
    tenant_store_namespace,
)
from opentulpa.specs import AgentSpec, AgentSpecRef, AgentSpecStore, OriginRef
from opentulpa.tooling import TOOL_SPEC_BY_NAME, AgentRunKind

logger = logging.getLogger(__name__)

_PUBLIC_RUN_FAILURE_MESSAGE = "The agent run could not be completed."
_PUBLIC_PROVIDER_REJECTION_MESSAGE = (
    "The model provider rejected this conversation. Start a new thread or use a different "
    "model/provider."
)
_PUBLIC_PROVIDER_FAILURE_MESSAGE = (
    "No configured model provider could complete this request. Try again later."
)
_DEEPAGENTS_API_CONTEXT_BUDGET_TOKENS = 50_000
_DEEPAGENTS_CODEX_CONTEXT_BUDGET_TOKENS = 300_000
_REGENERATE_COMMAND = "/regenerate"
_REGENERATE_INSTRUCTION = """Regenerate your latest attempted response to the immediately preceding
owner request. Produce a fresh answer rather than discussing this command or merely repeating the
previous answer. If the previous attempt failed before producing an answer, answer that preceding
request now. Reuse existing conversation and tool results; do not repeat tool calls, approvals, or
external side effects that may already have executed. If there is no preceding owner request, say
that briefly."""
_PUBLIC_RUN_CANCELLED_MESSAGE = "The agent run was cancelled before completion."
_PUBLIC_CAPABILITY_CHANGED_MESSAGE = (
    "The approved capability changed before the agent run could continue."
)
_ACTIVE_RUN_STATUSES = frozenset({"running", "interrupted", "resume_pending"})
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})
_RUN_STATUSES = _ACTIVE_RUN_STATUSES | _TERMINAL_RUN_STATUSES
_TRACE_EVENT_LIMIT = 500
_TRACE_LIST_LIMIT = 100
_TRACE_TEXT_PREVIEW_CHARS = 500
_TRACE_TOOL_VALUE_CHARS = 4_000
_TRACE_FAILURE_CAUSE_CHARS = 500
_TRACE_HIDDEN_KEYS = frozenset(
    {"actor_id", "checkpoint_thread_id", "customer_id", "tenant_id", "thread_id"}
)
_TRACE_PATH_KEYS = frozenset({"path", "uri"})
_TRACE_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w:/])(?:[A-Za-z]:\\[^\s\"']+|/(?:[^\s/\"']+/)*[^\s/\"']+)"
)
_INLINE_IMAGE_MIME_TYPES = frozenset({"image/gif", "image/jpeg", "image/png", "image/webp"})
_MAX_INLINE_ATTACHMENT_BYTES = 10 * 1024 * 1024
_MAX_INLINE_ATTACHMENTS_BYTES = 20 * 1024 * 1024

_OWNER_PRODUCT_TOOL_NAMES = frozenset(TOOL_SPEC_BY_NAME)
_ROUTINE_PRODUCT_TOOL_NAMES = frozenset(
    {
        "artifact_deliver",
        "browser_act",
        "browser_get",
        "browser_start",
        "connection_list",
        "content_fetch",
        "file_analyze",
        "file_get",
        "file_inspect",
        "file_search",
        "integration_action_search",
        "integration_invoke",
        "integration_list",
        "job_artifacts",
        "job_events",
        "job_get",
        "knowledge_find",
        "knowledge_list",
        "knowledge_query",
        "profile_get",
        "schedule_list",
        "web_search",
    }
)
_INTAKE_PRODUCT_TOOL_NAMES = frozenset(
    {
        "knowledge_find",
        "knowledge_list",
        "knowledge_query",
    }
)

_FILESYSTEM_TOOL_NAMES = frozenset(
    {"edit_file", "execute", "glob", "grep", "ls", "read_file", "write_file"}
)
_OWNER_DECISIONS: tuple[
    Literal["approve"],
    Literal["edit"],
    Literal["reject"],
] = ("approve", "edit", "reject")


def _bounded_trace_value(value: Any) -> Any:
    safe_value = _sanitize_trace_value(redact_for_langfuse(value))
    encoded = json.dumps(safe_value, ensure_ascii=False, sort_keys=True, default=str)
    if len(encoded) <= _TRACE_TOOL_VALUE_CHARS:
        return safe_value
    return {
        "preview": encoded[:_TRACE_TOOL_VALUE_CHARS],
        "truncated": True,
    }


def _sanitize_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized = key.strip().casefold()
            if normalized in _TRACE_HIDDEN_KEYS:
                sanitized[key] = "[redacted]"
            elif (
                normalized in _TRACE_PATH_KEYS
                and isinstance(raw_value, str)
                and (raw_value.startswith("/") or raw_value.startswith("file:"))
            ):
                sanitized[key] = "[redacted-path]"
            else:
                sanitized[key] = _sanitize_trace_value(raw_value)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_trace_value(item) for item in value]
    return value


def _trace_text_preview(value: Any) -> str:
    safe_value = str(redact_for_langfuse(str(value or "")))
    if len(safe_value) <= _TRACE_TEXT_PREVIEW_CHARS:
        return safe_value
    return f"{safe_value[:_TRACE_TEXT_PREVIEW_CHARS]}...[truncated]"


def _trace_input_summary(graph_input: Any) -> str:
    if not isinstance(graph_input, dict):
        return ""
    messages = graph_input.get("messages")
    if not isinstance(messages, list | tuple):
        return ""
    parts: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            parts.append(content.strip())
        elif isinstance(content, list | tuple):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    return _trace_text_preview("\n".join(parts))


def _safe_failure_cause(error: BaseException) -> str:
    cause = str(redact_for_langfuse(str(error or type(error).__name__)))
    cause = _TRACE_ABSOLUTE_PATH_RE.sub("[redacted-path]", cause)
    if len(cause) > _TRACE_FAILURE_CAUSE_CHARS:
        return f"{cause[:_TRACE_FAILURE_CAUSE_CHARS]}...[truncated]"
    return cause


def _public_text(value: Any) -> str:
    return str(redact_for_langfuse(str(value or "")))


def _failure_diagnostic(error: BaseException, *, phase: str) -> dict[str, str]:
    traceback = error.__traceback__
    location = "unknown"
    while traceback is not None:
        frame = traceback.tb_frame
        location = (
            f"{Path(frame.f_code.co_filename).name}:{frame.f_code.co_name}:{traceback.tb_lineno}"
        )
        traceback = traceback.tb_next
    error_type = type(error).__name__[:100] or "Exception"
    material = f"{type(error).__module__}.{type(error).__qualname__}:{location}"
    return {
        "error_type": error_type,
        "phase": str(phase or "unknown")[:100],
        "location": location[:300],
        "cause": _safe_failure_cause(error),
        "fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest()[:16],
    }


def _public_run_failure(error: BaseException) -> tuple[str, str, bool]:
    if _is_provider_rejection(error):
        return "model_provider_rejected", _PUBLIC_PROVIDER_REJECTION_MESSAGE, False
    if _is_model_provider_failure(error):
        return "model_provider_failed", _PUBLIC_PROVIDER_FAILURE_MESSAGE, True
    return "agent_run_failed", _PUBLIC_RUN_FAILURE_MESSAGE, False


def _is_provider_rejection(error: BaseException) -> bool:
    if type(error).__name__ == "BadRequestResponseError":
        return True
    message = str(error or "").casefold()
    return "openrouter api returned an error" in message and (
        "inappropriate content" in message or "provider returned error" in message
    )


def _is_provider_fallback_error(error: BaseException) -> bool:
    if _is_provider_rejection(error) or isinstance(error, httpx.TransportError):
        return True
    error_type = type(error)
    if error_type.__module__.startswith("openrouter.errors"):
        status_code = int(getattr(error, "status_code", 0) or 0)
        return status_code not in {401, 402, 403}
    message = str(error or "").casefold()
    return "openrouter api returned an error" in message or (
        "openrouter api returned a response with no choices" in message
    )


def _is_model_provider_failure(error: BaseException) -> bool:
    return _is_provider_fallback_error(error) or is_codex_transient(error)


def _shell_command_requires_approval(request: Any) -> bool:
    """Interrupt only shell calls classified as recursive forced removal."""

    raw_arguments = request.tool_call.get("args", {})
    if not isinstance(raw_arguments, dict):
        return False
    command = raw_arguments.get("command")
    if not isinstance(command, str):
        return False
    return classify_shell_command(command) is ShellCommandDisposition.REQUIRE_APPROVAL


class AgentRunObserver(Protocol):
    """Receive redacted persisted terminal snapshots outside the agent loop."""

    async def __call__(self, snapshot: AgentRunSnapshot) -> None: ...


class AgentAttachmentResolver(Protocol):
    """Resolve only tenant-owned uploads for native model attachment blocks."""

    def get_file(self, tenant_id: str, file_id: str) -> dict[str, Any] | None: ...

    def read_file_bytes(self, tenant_id: str, file_id: str) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class _PreparedRun:
    run_id: str
    checkpoint_thread_id: str
    created: bool
    inference_plan: ResolvedInferencePlan


class _DenyToolsMiddleware(AgentMiddleware):
    """Hide and reject built-ins a restricted AgentSpec is not allowed to use."""

    def __init__(self, denied: frozenset[str]) -> None:
        self._denied = denied

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        tools = [tool for tool in request.tools if tool.name not in self._denied]
        return await handler(request.override(tools=tools))

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        name = str(request.tool_call.get("name", ""))
        if name in self._denied:
            return ToolMessage(
                content="This AgentSpec is not permitted to use that tool.",
                tool_call_id=str(request.tool_call.get("id", "denied-tool-call")),
                name=name,
                status="error",
            )
        return await handler(request)


class _ShellCommandPolicyMiddleware(AgentMiddleware):
    """Reject shell calls whose destructive behavior cannot be classified safely."""

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = request.tool_call
        name = str(tool_call.get("name", ""))
        arguments = tool_call.get("args", {})
        command = arguments.get("command") if isinstance(arguments, dict) else None
        if (
            name in {"execute", "source_shell"}
            and isinstance(command, str)
            and classify_shell_command(command) is ShellCommandDisposition.REJECT
        ):
            return ToolMessage(
                content=(
                    "This shell command uses syntax that cannot be classified safely. "
                    "Use a literal command without dynamic executable or option construction."
                ),
                tool_call_id=str(tool_call.get("id", "rejected-shell-call")),
                name=name,
                status="error",
            )
        return await handler(request)


class _ProviderFallbackMiddleware(AgentMiddleware):
    """Try an ordered model chain for provider-level failures in one model call."""

    def __init__(
        self,
        fallback_models: Sequence[Any],
        *,
        eligible: Callable[[BaseException], bool] = _is_provider_fallback_error,
        allow_request: Callable[[Any], bool] | None = None,
    ) -> None:
        self._fallback_models = tuple(fallback_models)
        self._eligible = eligible
        self._allow_request = allow_request or (lambda _: True)

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        candidates = (request.model, *self._fallback_models)
        for index, model in enumerate(candidates):
            candidate_request = request if index == 0 else request.override(model=model)
            try:
                return await handler(candidate_request)
            except Exception as exc:
                if (
                    not self._eligible(exc)
                    or not self._allow_request(request)
                    or index == len(candidates) - 1
                ):
                    raise
                logger.warning(
                    "Model provider failed; trying fallback %d of %d",
                    index + 1,
                    len(self._fallback_models),
                )
        raise RuntimeError("model fallback chain was empty")


class _CodexAuthRetryMiddleware(AgentMiddleware):
    """Force one token refresh for a single unauthorized Codex model call."""

    def __init__(self, resolved: ResolvedModel) -> None:
        self._model = resolved.model
        self._provider = resolved.token_provider

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        try:
            return await handler(request)
        except Exception as exc:
            if (
                self._provider is None
                or request.model is not self._model
                or not is_codex_unauthorized(exc)
            ):
                raise
            await self._provider.aforce_refresh()
            return await handler(request)


class _InferenceMessageMiddleware(AgentMiddleware):
    """Remove provider-specific reasoning blocks when a thread switches providers."""

    def __init__(self, provider: Literal["api", "codex"]) -> None:
        self._provider = provider

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        messages: list[Any] = []
        changed = False
        for message in request.messages:
            if not isinstance(message, AIMessage) or not isinstance(message.content, list):
                messages.append(message)
                continue
            content: list[Any] = []
            message_changed = False
            for block in message.content:
                if not isinstance(block, dict) or block.get("type") != "reasoning":
                    content.append(block)
                    continue
                if self._provider == "codex" and block.get("encrypted_content"):
                    content.append(block)
                else:
                    changed = True
                    message_changed = True
            messages.append(
                message.model_copy(update={"content": content}) if message_changed else message
            )
        return (
            await handler(request.override(messages=messages))
            if changed
            else await handler(request)
        )


def _before_current_run_activity(request: Any) -> bool:
    """Allow cross-provider fallback only on the first model call of this turn."""

    messages = tuple(getattr(request, "messages", ()) or ())
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return True
        if isinstance(message, AIMessage | ToolMessage):
            return False
    return True


_RUN_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    checkpoint_thread_id TEXT NOT NULL,
    channel TEXT NOT NULL,
    run_kind TEXT NOT NULL,
    origin_json TEXT NOT NULL DEFAULT '{}',
    agent_spec_id TEXT NOT NULL DEFAULT 'owner',
    agent_spec_revision INTEGER NOT NULL DEFAULT 1,
    trust_class TEXT NOT NULL DEFAULT 'owner',
    correlation_id TEXT NOT NULL,
    idempotency_key TEXT,
    status TEXT NOT NULL,
    final_text TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    approvals_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0,
    request_digest TEXT NOT NULL DEFAULT '',
    dynamic_generation INTEGER NOT NULL DEFAULT 0,
    dynamic_digest TEXT NOT NULL DEFAULT '',
    request_text TEXT NOT NULL DEFAULT '',
    file_ids_json TEXT NOT NULL DEFAULT '[]',
    inference_json TEXT NOT NULL DEFAULT ''
)
"""

_RUN_EVENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_run_events (
    run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    data_json TEXT NOT NULL,
    PRIMARY KEY (run_id, sequence),
    FOREIGN KEY (run_id) REFERENCES agent_runs(run_id) ON DELETE CASCADE
)
"""

_THREAD_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_threads (
    tenant_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    title TEXT NOT NULL,
    channel TEXT NOT NULL,
    archived INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, thread_id)
)
"""

_THREAD_INFERENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_inference_preferences (
    tenant_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    revision INTEGER NOT NULL DEFAULT 0,
    selection_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tenant_id, thread_id),
    FOREIGN KEY (tenant_id, thread_id)
        REFERENCES agent_threads(tenant_id, thread_id) ON DELETE CASCADE
)
"""

_TENANT_INFERENCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS tenant_inference_preferences (
    tenant_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL DEFAULT 0,
    selection_json TEXT,
    updated_at TEXT NOT NULL
)
"""


def build_openrouter_chat_model(
    *,
    api_key: str,
    base_url: str,
    model_name: str,
    reasoning_effort: str | None = "high",
    max_completion_tokens: int | None = None,
    provider_order: Sequence[str] = (),
) -> ChatOpenRouter:
    """Build the model adapter shared by OpenTulpa agent profiles."""

    safe_key = str(api_key or "").strip()
    safe_model = str(model_name or "").strip()
    if not safe_key:
        raise RuntimeError("OPENAI_COMPATIBLE_API_KEY is required")
    if not safe_model:
        raise RuntimeError("LLM_MODEL is required")
    effort = str(reasoning_effort or "").strip() or None
    reasoning = {"effort": effort, "exclude": False} if effort else None
    providers = [str(provider).strip() for provider in provider_order if str(provider).strip()]
    return ChatOpenRouter(
        model=safe_model,
        api_key=SecretStr(safe_key),
        base_url=str(base_url or "").strip() or None,
        app_url="https://github.com/kvyb/opentulpa",
        app_title="OpenTulpa",
        reasoning=reasoning,
        max_completion_tokens=max_completion_tokens,
        openrouter_provider=(
            {
                "order": providers,
                "allow_fallbacks": False,
            }
            if providers
            else None
        ),
        streaming=True,
        max_retries=0,
        # langchain-openrouter forwards this value to the SDK as milliseconds.
        timeout=60_000,
    )


def _with_deepagents_context_budget(
    model: Any,
    *,
    provider: InferenceProvider = "api",
) -> Any:
    """Give Deep Agents the selected provider's working-context budget."""

    if not isinstance(model, BaseChatModel):
        return model
    budget = (
        _DEEPAGENTS_CODEX_CONTEXT_BUDGET_TOKENS
        if provider == "codex"
        else _DEEPAGENTS_API_CONTEXT_BUDGET_TOKENS
    )
    profile = cast("ModelProfile", dict(model.profile or {}))
    advertised_limit = profile.get("max_input_tokens")
    if type(advertised_limit) is int and advertised_limit > 0:
        effective_limit = min(advertised_limit, budget)
    else:
        effective_limit = budget
    if advertised_limit == effective_limit:
        return model
    profile["max_input_tokens"] = effective_limit
    model.profile = profile
    return model


class DeepAgentService:
    """Configure and invoke Deep Agents without adding another agent runtime."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        checkpoint_db_path: str | Path,
        store_db_path: str | Path,
        runs_db_path: str | Path,
        workspaces_root: str | Path,
        tools: Sequence[BaseTool] = (),
        dynamic_tools: DynamicToolProvider | None = None,
        reasoning_effort: str | None = "high",
        max_completion_tokens: int | None = None,
        langfuse_tracer: Any | None = None,
        model: Any | None = None,
        agent_specs: AgentSpecStore | None = None,
        model_resolver: Callable[[str], Any] | None = None,
        container_policy: TenantContainerPolicy | None = None,
        container_cli: str = "docker",
        execution_provider: TenantExecutionProvider | None = None,
        execution_backend: Any | None = None,
        workspace_backend: Any | None = None,
        run_observer: AgentRunObserver | None = None,
        attachment_resolver: AgentAttachmentResolver | None = None,
        provider_fallback_models: Sequence[Any] = (),
        inference_service: InferenceService | None = None,
        graph_cache_limit: int = 48,
    ) -> None:
        self._api_key = str(api_key or "").strip()
        self._base_url = str(base_url or "").strip()
        self._model_name = str(model_name or "").strip()
        self._checkpoint_db_path = Path(checkpoint_db_path).expanduser().resolve()
        self._store_db_path = Path(store_db_path).expanduser().resolve()
        self._runs_db_path = Path(runs_db_path).expanduser().resolve()
        self._workspaces_root = Path(workspaces_root).expanduser().resolve()
        self._tools = self._validate_product_tools(tools)
        if dynamic_tools is not None and agent_specs is None:
            raise ValueError("dynamic tools require revisioned AgentSpecs")
        self._dynamic_tools = dynamic_tools
        self._reasoning_effort = str(reasoning_effort or "").strip() or None
        self._max_completion_tokens = max_completion_tokens
        self._langfuse_tracer = langfuse_tracer
        self._provided_model = model
        self._agent_specs = agent_specs
        self._model_resolver = model_resolver
        self._container_policy = container_policy or TenantContainerPolicy()
        self._container_cli = str(container_cli or "docker").strip() or "docker"
        self._execution_provider = execution_provider
        self._execution_backend = execution_backend
        self._workspace_backend = workspace_backend
        self._run_observer = run_observer
        self._attachment_resolver = attachment_resolver
        self._provider_fallback_models = tuple(provider_fallback_models)
        self._inference = inference_service
        self._graph_cache_limit = max(8, int(graph_cache_limit))
        self._checkpoint_conn: aiosqlite.Connection | None = None
        self._store_cm: Any | None = None
        self._checkpointer: AsyncSqliteSaver | None = None
        self._store: AsyncSqliteStore | None = None
        self._runs_db: aiosqlite.Connection | None = None
        self._run_event_lock = asyncio.Lock()
        self._run_event_conditions: WeakValueDictionary[str, asyncio.Condition] = (
            WeakValueDictionary()
        )
        self._active_run_tasks: dict[str, asyncio.Task[None]] = {}
        self._graphs: dict[str, Any] = {}
        self._spec_graphs: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._checkpoint_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
        self._pending_resume_tasks: set[asyncio.Task[None]] = set()
        self._pending_resume_ids: set[str] = set()
        self._shutting_down = False

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def langfuse_tracer(self) -> Any | None:
        return self._langfuse_tracer

    @property
    def started(self) -> bool:
        return bool(self._graphs and self._runs_db is not None)

    def healthy(self) -> bool:
        return self.started

    async def start(self, *, recover_pending_resumes: bool = True) -> None:
        if self.started:
            return
        self._shutting_down = False
        for path in (self._checkpoint_db_path, self._store_db_path, self._runs_db_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._workspaces_root.mkdir(parents=True, exist_ok=True)

        self._checkpoint_conn = await aiosqlite.connect(str(self._checkpoint_db_path))
        self._checkpointer = AsyncSqliteSaver(
            self._checkpoint_conn,
            serde=JsonPlusSerializer(pickle_fallback=False),
        )
        await self._checkpointer.setup()
        self._store_cm = AsyncSqliteStore.from_conn_string(str(self._store_db_path))
        self._store = await self._store_cm.__aenter__()
        await self._store.setup()
        self._runs_db = await aiosqlite.connect(str(self._runs_db_path))
        self._runs_db.row_factory = aiosqlite.Row
        await self._runs_db.execute("PRAGMA foreign_keys=ON")
        await self._runs_db.execute("PRAGMA journal_mode=WAL")
        await self._runs_db.execute(_RUN_SCHEMA)
        await self._runs_db.execute(_RUN_EVENT_SCHEMA)
        await self._runs_db.execute(_THREAD_SCHEMA)
        await self._runs_db.execute(_THREAD_INFERENCE_SCHEMA)
        await self._runs_db.execute(_TENANT_INFERENCE_SCHEMA)
        schema_cursor = await self._runs_db.execute("PRAGMA table_info(agent_runs)")
        columns = {str(row[1]) for row in await schema_cursor.fetchall()}
        await schema_cursor.close()
        if "last_sequence" not in columns:
            await self._runs_db.execute(
                "ALTER TABLE agent_runs ADD COLUMN last_sequence INTEGER NOT NULL DEFAULT 0"
            )
        for column, definition in (
            ("origin_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("agent_spec_id", "TEXT NOT NULL DEFAULT 'owner'"),
            ("agent_spec_revision", "INTEGER NOT NULL DEFAULT 1"),
            ("trust_class", "TEXT NOT NULL DEFAULT 'owner'"),
            ("idempotency_key", "TEXT"),
            ("request_digest", "TEXT NOT NULL DEFAULT ''"),
            ("dynamic_generation", "INTEGER NOT NULL DEFAULT 0"),
            ("dynamic_digest", "TEXT NOT NULL DEFAULT ''"),
            ("request_text", "TEXT NOT NULL DEFAULT ''"),
            ("file_ids_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("inference_json", "TEXT NOT NULL DEFAULT ''"),
        ):
            if column not in columns:
                await self._runs_db.execute(
                    f"ALTER TABLE agent_runs ADD COLUMN {column} {definition}"
                )
        await self._runs_db.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_runs_tenant_idempotency
            ON agent_runs (tenant_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL
            """
        )
        default_plan_json = self._default_inference_plan().model_dump_json()
        await self._runs_db.execute(
            """
            UPDATE agent_runs
            SET inference_json = ?
            WHERE inference_json IS NULL OR inference_json = ''
            """,
            (default_plan_json,),
        )
        await self._runs_db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_run_events_run_type
            ON agent_run_events (run_id, event_type)
            """
        )
        await self._runs_db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_agent_threads_tenant_updated
            ON agent_threads (tenant_id, archived, updated_at DESC, thread_id DESC)
            """
        )
        await self._runs_db.execute(
            """
            INSERT OR IGNORE INTO agent_threads (
                tenant_id, thread_id, title, channel, archived, created_at, updated_at
            )
            SELECT
                tenant_id,
                thread_id,
                'Previous session',
                MIN(channel),
                0,
                MIN(created_at),
                MAX(updated_at)
            FROM agent_runs
            GROUP BY tenant_id, thread_id
            """
        )
        await self._runs_db.execute(
            """
            INSERT OR IGNORE INTO tenant_inference_preferences (
                tenant_id, revision, selection_json, updated_at
            )
            SELECT
                preference.tenant_id,
                1,
                preference.selection_json,
                preference.updated_at
            FROM thread_inference_preferences AS preference
            WHERE preference.selection_json IS NOT NULL
              AND TRIM(preference.selection_json) != ''
              AND preference.rowid = (
                  SELECT candidate.rowid
                  FROM thread_inference_preferences AS candidate
                  WHERE candidate.tenant_id = preference.tenant_id
                    AND candidate.selection_json IS NOT NULL
                    AND TRIM(candidate.selection_json) != ''
                  ORDER BY candidate.updated_at DESC, candidate.rowid DESC
                  LIMIT 1
              )
            """
        )
        await self._runs_db.commit()
        self._graphs = (
            {"__agent_specs__": True} if self._agent_specs is not None else self._build_graphs()
        )
        await self._reconcile_stale_running_runs()
        if recover_pending_resumes:
            await self.recover_pending_resumes()

    async def recover_pending_resumes(self) -> None:
        """Continue durable approval decisions after capability restoration."""

        self._require_started()
        await self._recover_pending_resumes()

    async def shutdown(self) -> None:
        self._shutting_down = True
        active_tasks = tuple(self._active_run_tasks.values())
        for task in active_tasks:
            task.cancel()
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)
        self._active_run_tasks.clear()
        recovery_tasks = tuple(self._pending_resume_tasks)
        for task in recovery_tasks:
            task.cancel()
        if recovery_tasks:
            await asyncio.gather(*recovery_tasks, return_exceptions=True)
        self._pending_resume_tasks.clear()
        self._pending_resume_ids.clear()
        self._graphs.clear()
        self._spec_graphs.clear()
        if self._runs_db is not None:
            await self._runs_db.close()
            self._runs_db = None
        if self._store_cm is not None:
            await self._store_cm.__aexit__(None, None, None)
            self._store_cm = None
            self._store = None
        if self._checkpoint_conn is not None:
            await self._checkpoint_conn.close()
            self._checkpoint_conn = None
            self._checkpointer = None

    async def run(self, request: AgentRunRequest) -> AgentRunSnapshot:
        run_id = ""
        async for event in self.stream(request):
            run_id = event.run_id
        if not run_id:
            raise RuntimeError("agent run did not start")
        snapshot = await self.get_run(run_id)
        if snapshot is None:
            raise RuntimeError(f"agent run {run_id} was not persisted")
        return snapshot

    async def open_stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        """Persist and validate a run before an HTTP server commits SSE headers."""

        self._require_started()
        prepared = await self._prepare_run(request)
        if prepared.created:
            self._schedule_run(request, prepared)
        return self.events(prepared.run_id)

    async def stream(self, request: AgentRunRequest) -> AsyncIterator[AgentRunEvent]:
        events = await self.open_stream(request)
        try:
            async for event in events:
                yield event
        finally:
            close = getattr(events, "aclose", None)
            if callable(close):
                await close()

    async def _prepare_run(self, request: AgentRunRequest) -> _PreparedRun:
        run_id = new_short_id("run", suffix_chars=10)
        checkpoint_thread_id = self._checkpoint_thread_id(request.context)
        dynamic = self._dynamic_snapshot(request.context.tenant_id)
        inference_plan = await self._resolve_inference_plan(request.context)
        persisted_run_id = await self._insert_run(
            run_id,
            checkpoint_thread_id,
            request.context,
            request_text=request.text,
            file_ids=request.file_ids,
            idempotency_key=request.idempotency_key,
            request_digest=self._request_digest(request),
            dynamic_generation=dynamic.generation,
            dynamic_digest=self._dynamic_digest(dynamic),
            inference_plan=inference_plan,
        )
        if persisted_run_id != run_id:
            existing = await self.get_run(persisted_run_id)
            if existing is not None and existing.inference_plan is not None:
                inference_plan = existing.inference_plan
        return _PreparedRun(
            run_id=persisted_run_id,
            checkpoint_thread_id=checkpoint_thread_id,
            created=persisted_run_id == run_id,
            inference_plan=inference_plan,
        )

    def _schedule_run(
        self,
        request: AgentRunRequest,
        prepared: _PreparedRun,
    ) -> None:
        if self._shutting_down:
            return
        current = self._active_run_tasks.get(prepared.run_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._run_prepared(request, prepared),
            name=f"opentulpa-agent-run:{prepared.run_id}",
        )
        self._active_run_tasks[prepared.run_id] = task
        task.add_done_callback(lambda completed: self._run_task_done(prepared.run_id, completed))

    async def _run_prepared(
        self,
        request: AgentRunRequest,
        prepared: _PreparedRun,
    ) -> None:
        graph_input = {
            "messages": [
                {
                    "role": "user",
                    "content": self._request_content(request),
                }
            ]
        }
        try:
            async with self._checkpoint_lock(prepared.checkpoint_thread_id):
                async for _ in self._stream_graph(
                    run_id=prepared.run_id,
                    context=request.context,
                    checkpoint_thread_id=prepared.checkpoint_thread_id,
                    graph_input=graph_input,
                    resumed=False,
                    inference_plan=prepared.inference_plan,
                ):
                    pass
        finally:
            await self._finalize_abandoned_run(prepared.run_id)

    def _run_task_done(
        self,
        run_id: str,
        task: asyncio.Task[None],
    ) -> None:
        if self._active_run_tasks.get(run_id) is task:
            self._active_run_tasks.pop(run_id, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Detached Deep Agent run stopped: run_id=%s error_type=%s",
                run_id,
                type(error).__name__,
            )

    async def open_resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        """Claim an approval before an HTTP server commits SSE headers."""

        self._require_started()
        snapshot = await self.get_run(run_id)
        if snapshot is None:
            raise KeyError(f"agent run not found: {run_id}")
        if snapshot.status not in {"interrupted", "resume_pending"}:
            raise ValueError(f"agent run is not awaiting approval: {snapshot.status}")
        dynamic = await self._verified_dynamic_snapshot(run_id, snapshot.context)
        claimed_snapshot, approvals = await self._claim_approval_decision(run_id, decision)
        pending = [approval for approval in approvals if approval.status == "pending"]
        if pending:
            return self._pending_approval_events(run_id, pending)
        cursor = await self._last_sequence(run_id)
        events = self._resume_with_recovery(
            run_id=run_id,
            snapshot=claimed_snapshot,
            approvals=approvals,
            dynamic=dynamic,
            cursor=cursor,
        )
        # The decision is durable before SSE headers are committed. Start recovery now so
        # a disconnect before the response iterator begins cannot strand resume_pending.
        self._schedule_pending_resume(run_id)
        return events

    async def resume(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> AsyncIterator[AgentRunEvent]:
        events = await self.open_resume(run_id, decision)
        try:
            async for event in events:
                yield event
        finally:
            close = getattr(events, "aclose", None)
            if callable(close):
                await close()

    async def _pending_approval_events(
        self,
        run_id: str,
        approvals: Sequence[AgentApproval],
    ) -> AsyncIterator[AgentRunEvent]:
        for approval in approvals:
            event = await self._append_event(
                run_id=run_id,
                type="approval.required",
                data=self._public_approval(approval),
            )
            if event is None:
                return
            yield event

    async def _resume_with_recovery(
        self,
        *,
        run_id: str,
        snapshot: AgentRunSnapshot,
        approvals: Sequence[AgentApproval],
        dynamic: DynamicToolSnapshot,
        cursor: int,
    ) -> AsyncIterator[AgentRunEvent]:
        try:
            async for event in self._resume_claimed_run(
                run_id=run_id,
                snapshot=snapshot,
                approvals=approvals,
                dynamic=dynamic,
                cursor=cursor,
            ):
                yield event
        finally:
            self._schedule_pending_resume(run_id)

    async def _resume_claimed_run(
        self,
        *,
        run_id: str,
        snapshot: AgentRunSnapshot,
        approvals: Sequence[AgentApproval],
        dynamic: DynamicToolSnapshot,
        cursor: int | None = None,
    ) -> AsyncIterator[AgentRunEvent]:
        checkpoint_thread_id = self._checkpoint_thread_id(snapshot.context)
        if cursor is None:
            cursor = await self._last_sequence(run_id)
        async with self._checkpoint_lock(checkpoint_thread_id):
            current = await self.get_run(run_id)
            if current is None:
                raise KeyError(f"agent run not found: {run_id}")
            if current.status in {"completed", "failed", "cancelled", "interrupted"}:
                async for event in self.events(run_id, after_sequence=cursor):
                    yield event
                return
            if current.status != "resume_pending":
                raise ValueError(f"agent run cannot resume from status: {current.status}")
            current_dynamic = await self._verified_dynamic_snapshot(run_id, current.context)
            if self._dynamic_binding(current_dynamic) != self._dynamic_binding(dynamic):
                raise AgentRunCapabilityConflictError(_PUBLIC_CAPABILITY_CHANGED_MESSAGE)
            inference_plan = current.inference_plan or self._default_inference_plan()
            graph = self._graph_for_context(
                current.context,
                dynamic=current_dynamic,
                inference_plan=inference_plan,
            )
            config = self._run_config(
                current.context,
                checkpoint_thread_id,
                inference_plan=inference_plan,
            )
            graph_input, handled = await self._resume_input_from_checkpoint(
                run_id=run_id,
                snapshot=current,
                graph=graph,
                config=config,
                approvals=approvals,
            )
            if handled:
                async for event in self.events(run_id, after_sequence=cursor):
                    yield event
                return
            async for event in self._stream_graph(
                run_id=run_id,
                context=current.context,
                checkpoint_thread_id=checkpoint_thread_id,
                graph_input=graph_input,
                resumed=True,
                inference_plan=inference_plan,
            ):
                yield event

    async def _resume_input_from_checkpoint(
        self,
        *,
        run_id: str,
        snapshot: AgentRunSnapshot,
        graph: Any,
        config: dict[str, Any],
        approvals: Sequence[AgentApproval],
    ) -> tuple[Any, bool]:
        state = await graph.aget_state(config)
        state_interrupts = [
            interrupt
            for task in getattr(state, "tasks", ()) or ()
            for interrupt in getattr(task, "interrupts", ()) or ()
        ]
        checkpoint_approvals = self._approvals_from_stream(
            {"__interrupt__": state_interrupts},
            run_id,
        )
        decided_ids = {approval.id for approval in approvals}
        checkpoint_ids = {approval.id for approval in checkpoint_approvals}
        if checkpoint_approvals and checkpoint_ids != decided_ids:
            updated = await self._update_run(
                run_id,
                status="interrupted",
                approvals=checkpoint_approvals,
                final_text=snapshot.final_text,
                allowed_statuses={"resume_pending"},
            )
            if not updated:
                return None, True
            for approval in checkpoint_approvals:
                event = await self._append_event(
                    run_id=run_id,
                    type="approval.required",
                    data=self._public_approval(approval),
                )
                if event is None:
                    return None, True
            return None, True
        if not tuple(getattr(state, "next", ()) or ()):
            final_text = self._last_ai_text(state.values.get("messages", []))
            event = await self._transition_with_event(
                run_id,
                allowed_statuses={"resume_pending"},
                status="completed",
                event_type="run.completed",
                event_data={"text": final_text or snapshot.final_text},
                final_text=final_text or snapshot.final_text,
                approvals=[],
            )
            if event is not None:
                await self._observe_run(run_id)
            return None, True
        if checkpoint_approvals:
            return Command(
                resume={"decisions": [self._resume_decision(item) for item in approvals]}
            ), False
        return None, False

    async def _claim_approval_decision(
        self,
        run_id: str,
        decision: ApprovalDecision,
    ) -> tuple[AgentRunSnapshot, list[AgentApproval]]:
        snapshot = await self.get_run(run_id)
        if snapshot is None:
            raise KeyError(f"agent run not found: {run_id}")
        if snapshot.status not in {"interrupted", "resume_pending"}:
            raise ValueError(f"agent run is not awaiting approval: {snapshot.status}")

        approvals = list(snapshot.approvals)
        for index, approval in enumerate(approvals):
            if approval.id != decision.approval_id:
                continue
            if approval.status != "pending":
                if not self._approval_matches_decision(approval, decision):
                    raise ValueError(f"approval already decided differently: {approval.id}")
                return snapshot, approvals
            if decision.decision not in approval.allowed_decisions:
                raise ValueError(
                    f"decision {decision.decision!r} is not allowed for {approval.tool_name}"
                )
            approvals[index] = AgentApproval(
                id=approval.id,
                tool_name=approval.tool_name,
                description=approval.description,
                arguments=approval.arguments,
                allowed_decisions=approval.allowed_decisions,
                status=decision.decision,
                edited_arguments=decision.edited_arguments,
                message=decision.message,
            )
            break
        else:
            raise KeyError(f"approval not found: {decision.approval_id}")

        next_status = (
            "interrupted"
            if any(approval.status == "pending" for approval in approvals)
            else "resume_pending"
        )
        db = self._require_runs_db()
        async with self._run_event_lock:
            cursor = await db.execute(
                """
                UPDATE agent_runs
                SET approvals_json = ?, status = ?, updated_at = ?
                WHERE run_id = ? AND status = 'interrupted' AND approvals_json = ?
                """,
                (
                    self._serialize_approvals(approvals),
                    next_status,
                    utc_now_iso(),
                    run_id,
                    self._serialize_approvals(snapshot.approvals),
                ),
            )
            claimed = cursor.rowcount == 1
            await cursor.close()
            await db.commit()
        if not claimed:
            raise ValueError("approval state changed; reload the run before retrying")
        return snapshot, approvals

    @staticmethod
    def _approval_matches_decision(
        approval: AgentApproval,
        decision: ApprovalDecision,
    ) -> bool:
        return (
            approval.status == decision.decision
            and approval.edited_arguments == decision.edited_arguments
            and approval.message == decision.message
        )

    async def get_run(self, run_id: str) -> AgentRunSnapshot | None:
        db = self._require_runs_db()
        cursor = await db.execute("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
        row = await cursor.fetchone()
        await cursor.close()
        return self._snapshot_from_row(row) if row is not None else None

    async def trace_list(
        self,
        *,
        tenant_id: str,
        status: str | None = None,
        limit: int = 20,
        before_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List bounded run summaries visible to one tenant."""

        safe_tenant_id = str(tenant_id or "").strip()
        if not safe_tenant_id:
            raise ValueError("tenant_id is required")
        safe_status = str(status or "").strip() or None
        if safe_status is not None and safe_status not in _RUN_STATUSES:
            raise ValueError(f"unknown agent run status: {safe_status}")
        safe_limit = min(_TRACE_LIST_LIMIT, max(1, int(limit)))
        safe_before_run_id = str(before_run_id or "").strip() or None
        db = self._require_runs_db()
        before_created_at: str | None = None
        if safe_before_run_id is not None:
            before_cursor = await db.execute(
                """
                SELECT created_at
                FROM agent_runs
                WHERE tenant_id = ? AND run_id = ?
                """,
                (safe_tenant_id, safe_before_run_id),
            )
            before_row = await before_cursor.fetchone()
            await before_cursor.close()
            if before_row is None:
                return []
            before_created_at = str(before_row["created_at"])
        conditions = ["tenant_id = ?"]
        parameters: list[Any] = [safe_tenant_id]
        if safe_status is not None:
            conditions.append("status = ?")
            parameters.append(safe_status)
        if safe_before_run_id is not None and before_created_at is not None:
            conditions.append("(created_at < ? OR (created_at = ? AND run_id < ?))")
            parameters.extend((before_created_at, before_created_at, safe_before_run_id))
        parameters.append(safe_limit)
        cursor = await db.execute(
            f"""
            SELECT run_id, status, channel, run_kind, final_text, created_at, updated_at
            FROM agent_runs
            WHERE {" AND ".join(conditions)}
            ORDER BY created_at DESC, run_id DESC
            LIMIT ?
            """,
            parameters,
        )
        rows = await cursor.fetchall()
        await cursor.close()
        counts = {str(row["run_id"]): [0, 0] for row in rows}
        if counts:
            placeholders = ", ".join("?" for _ in counts)
            event_cursor = await db.execute(
                f"""
                SELECT
                    event.run_id,
                    SUM(CASE WHEN event.event_type = 'tool.started' THEN 1 ELSE 0 END)
                        AS tool_count,
                    SUM(
                        CASE
                            WHEN event.event_type = 'tool.completed'
                             AND json_extract(event.data_json, '$.ok') = 0
                            THEN 1 ELSE 0
                        END
                    ) AS failed_tool_count
                FROM agent_run_events AS event
                WHERE event.run_id IN ({placeholders})
                  AND event.event_type IN ('tool.started', 'tool.completed')
                GROUP BY event.run_id
                """,
                tuple(counts),
            )
            event_rows = await event_cursor.fetchall()
            await event_cursor.close()
            for event_row in event_rows:
                counts[str(event_row["run_id"])] = [
                    int(event_row["tool_count"] or 0),
                    int(event_row["failed_tool_count"] or 0),
                ]
        return [
            self._trace_run_summary(
                row,
                tool_count=counts[str(row["run_id"])][0],
                failed_tool_count=counts[str(row["run_id"])][1],
            )
            for row in rows
        ]

    async def trace_get(
        self,
        *,
        tenant_id: str,
        run_id: str,
        after_sequence: int = 0,
        limit: int = 200,
        include_messages: bool = False,
    ) -> dict[str, Any] | None:
        """Read one tenant-scoped durable trace without exposing runtime ownership data."""

        safe_tenant_id = str(tenant_id or "").strip()
        safe_run_id = str(run_id or "").strip()
        if not safe_tenant_id or not safe_run_id:
            raise ValueError("tenant_id and run_id are required")
        safe_after = max(0, int(after_sequence))
        safe_limit = min(_TRACE_EVENT_LIMIT, max(1, int(limit)))
        db = self._require_runs_db()
        cursor = await db.execute(
            """
            SELECT run_id, status, channel, run_kind, final_text, created_at, updated_at
            FROM agent_runs
            WHERE tenant_id = ? AND run_id = ?
            """,
            (safe_tenant_id, safe_run_id),
        )
        run_row = await cursor.fetchone()
        await cursor.close()
        if run_row is None:
            return None
        if include_messages:
            event_cursor = await db.execute(
                """
                SELECT event.sequence, event.event_type, event.timestamp, event.data_json
                FROM agent_run_events AS event
                JOIN agent_runs AS run ON run.run_id = event.run_id
                WHERE run.tenant_id = ? AND event.run_id = ? AND event.sequence > ?
                ORDER BY event.sequence ASC
                LIMIT ?
                """,
                (safe_tenant_id, safe_run_id, safe_after, safe_limit + 1),
            )
        else:
            event_cursor = await db.execute(
                """
                SELECT event.sequence, event.event_type, event.timestamp, event.data_json
                FROM agent_run_events AS event
                JOIN agent_runs AS run ON run.run_id = event.run_id
                WHERE run.tenant_id = ? AND event.run_id = ? AND event.sequence > ?
                  AND event.event_type != 'message.delta'
                ORDER BY event.sequence ASC
                LIMIT ?
                """,
                (safe_tenant_id, safe_run_id, safe_after, safe_limit + 1),
            )
        event_rows = list(await event_cursor.fetchall())
        await event_cursor.close()
        count_cursor = await db.execute(
            """
            SELECT
                SUM(CASE WHEN event.event_type = 'tool.started' THEN 1 ELSE 0 END)
                    AS tool_count,
                SUM(
                    CASE
                        WHEN event.event_type = 'tool.completed'
                         AND json_extract(event.data_json, '$.ok') = 0
                        THEN 1 ELSE 0
                    END
                ) AS failed_tool_count
            FROM agent_run_events AS event
            WHERE event.run_id = ?
              AND event.event_type IN ('tool.started', 'tool.completed')
            """,
            (safe_run_id,),
        )
        count_row = await count_cursor.fetchone()
        await count_cursor.close()
        tool_count = int(count_row["tool_count"] or 0) if count_row is not None else 0
        failed_tool_count = int(count_row["failed_tool_count"] or 0) if count_row is not None else 0
        has_more = len(event_rows) > safe_limit
        event_rows = event_rows[:safe_limit]
        events: list[dict[str, Any]] = []
        for event_row in event_rows:
            try:
                raw_data = json.loads(str(event_row["data_json"]))
            except (TypeError, ValueError):
                raw_data = {"code": "invalid_trace_event"}
            data = raw_data if isinstance(raw_data, dict) else {"value": raw_data}
            if not include_messages and str(event_row["event_type"]) == "run.completed":
                data.pop("text", None)
            events.append(
                {
                    "sequence": int(event_row["sequence"]),
                    "type": str(event_row["event_type"]),
                    "timestamp": str(event_row["timestamp"]),
                    "data": _bounded_trace_value(data),
                }
            )
        trace = self._trace_run_summary(
            run_row,
            tool_count=tool_count,
            failed_tool_count=failed_tool_count,
        )
        trace["events"] = events
        trace["has_more"] = has_more
        trace["next_after_sequence"] = int(event_rows[-1]["sequence"]) if event_rows else safe_after
        return trace

    @staticmethod
    def _trace_run_summary(
        row: aiosqlite.Row,
        *,
        tool_count: int = 0,
        failed_tool_count: int = 0,
    ) -> dict[str, Any]:
        return {
            "run_id": str(row["run_id"]),
            "status": str(row["status"]),
            "channel": str(row["channel"]),
            "run_kind": str(row["run_kind"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "final_text_preview": _trace_text_preview(row["final_text"]),
            "tool_count": max(0, int(tool_count)),
            "failed_tool_count": max(0, int(failed_tool_count)),
        }

    async def events(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[AgentRunEvent]:
        """Replay durable events, then follow a live run without owning its execution."""

        sequence = max(0, int(after_sequence))
        condition = self._run_event_condition(run_id)
        while True:
            async with condition:
                async with self._run_event_lock:
                    db = self._require_runs_db()
                    cursor = await db.execute(
                        """
                        SELECT sequence, event_type, timestamp, data_json
                        FROM agent_run_events
                        WHERE run_id = ? AND sequence > ?
                        ORDER BY sequence ASC
                        """,
                        (run_id, sequence),
                    )
                    rows = await cursor.fetchall()
                    await cursor.close()
                    status = await self._run_status(run_id) if not rows else ""
                if not rows:
                    if status not in {"running", "resume_pending"}:
                        return
                    await condition.wait()
                    continue
            for row in rows:
                sequence = int(row["sequence"])
                yield AgentRunEvent(
                    type=cast(AgentRunEventType, str(row["event_type"])),
                    run_id=run_id,
                    sequence=sequence,
                    timestamp=str(row["timestamp"]),
                    data=dict(json.loads(str(row["data_json"]))),
                )

    def _checkpoint_lock(self, checkpoint_thread_id: str) -> asyncio.Lock:
        lock = self._checkpoint_locks.get(checkpoint_thread_id)
        if lock is None:
            lock = asyncio.Lock()
            self._checkpoint_locks[checkpoint_thread_id] = lock
        return lock

    def _run_event_condition(self, run_id: str) -> asyncio.Condition:
        condition = self._run_event_conditions.get(run_id)
        if condition is None:
            condition = asyncio.Condition()
            self._run_event_conditions[run_id] = condition
        return condition

    async def _notify_run_event(self, run_id: str) -> None:
        condition = self._run_event_condition(run_id)
        async with condition:
            condition.notify_all()

    async def cancel(self, run_id: str) -> AgentRunSnapshot:
        snapshot = await self.get_run(run_id)
        if snapshot is None:
            raise KeyError(f"agent run not found: {run_id}")
        if snapshot.status in {"completed", "failed", "cancelled"}:
            return snapshot
        task = self._active_run_tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if await self._run_status(run_id) not in _TERMINAL_RUN_STATUSES:
            await self._cancel_with_event(
                run_id,
                allowed_statuses={"running", "interrupted", "resume_pending"},
            )
        updated = await self.get_run(run_id)
        assert updated is not None
        return updated

    async def cancel_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> AgentRunSnapshot | None:
        """Cancel the newest active run when an interface has not received its run ID yet."""

        db = self._require_runs_db()
        cursor = await db.execute(
            """
            SELECT run_id
            FROM agent_runs
            WHERE tenant_id = ?
              AND thread_id = ?
              AND status IN ('running', 'interrupted', 'resume_pending')
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (tenant_id, thread_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return await self.cancel(str(row["run_id"]))

    async def create_thread(
        self,
        *,
        tenant_id: str,
        channel: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Create durable client metadata without creating a Deep Agents checkpoint."""

        self._require_started()
        thread_id = new_short_id("thread", suffix_chars=12)
        now = utc_now_iso()
        safe_title = self._normalize_thread_title(title) or "New session"
        db = self._require_runs_db()
        await db.execute(
            """
            INSERT INTO agent_threads (
                tenant_id, thread_id, title, channel, archived, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 0, ?, ?)
            """,
            (tenant_id, thread_id, safe_title, channel, now, now),
        )
        await db.commit()
        return {
            "thread_id": thread_id,
            "title": safe_title,
            "channel": channel,
            "archived": False,
            "created_at": now,
            "updated_at": now,
            "last_run_id": None,
            "status": "idle",
            "preview": "",
        }

    async def ensure_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        channel: str,
    ) -> None:
        """Persist caller-owned thread metadata before a run or preference update."""

        self._require_started()
        now = utc_now_iso()
        db = self._require_runs_db()
        async with self._run_event_lock:
            await db.execute(
                """
                INSERT INTO agent_threads (
                    tenant_id, thread_id, title, channel, archived, created_at, updated_at
                ) VALUES (?, ?, 'New session', ?, 0, ?, ?)
                ON CONFLICT (tenant_id, thread_id) DO NOTHING
                """,
                (tenant_id, thread_id, channel, now, now),
            )
            await db.commit()

    async def list_threads(
        self,
        *,
        tenant_id: str,
        cursor: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        self._require_started()
        safe_limit = min(max(1, int(limit)), 100)
        cursor_updated, cursor_thread = self._decode_thread_cursor(cursor)
        where = "thread.tenant_id = ? AND thread.archived = 0"
        values: list[Any] = [tenant_id]
        if cursor_updated and cursor_thread:
            where += " AND (thread.updated_at, thread.thread_id) < (?, ?)"
            values.extend((cursor_updated, cursor_thread))
        values.append(safe_limit + 1)
        db = self._require_runs_db()
        query = f"""
            SELECT
                thread.thread_id,
                thread.title,
                thread.channel,
                thread.archived,
                thread.created_at,
                thread.updated_at,
                latest.run_id AS last_run_id,
                COALESCE(latest.status, 'idle') AS status,
                COALESCE(NULLIF(latest.final_text, ''), latest.request_text, '') AS preview
            FROM agent_threads AS thread
            LEFT JOIN agent_runs AS latest
              ON latest.run_id = (
                  SELECT run.run_id
                  FROM agent_runs AS run
                  WHERE run.tenant_id = thread.tenant_id
                    AND run.thread_id = thread.thread_id
                  ORDER BY run.created_at DESC, run.run_id DESC
                  LIMIT 1
              )
            WHERE {where}
            ORDER BY thread.updated_at DESC, thread.thread_id DESC
            LIMIT ?
        """
        result = await db.execute(query, values)
        rows = list(await result.fetchall())
        await result.close()
        has_more = len(rows) > safe_limit
        rows = rows[:safe_limit]
        items = [self._thread_row(row) for row in rows]
        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = self._encode_thread_cursor(
                str(last["updated_at"]), str(last["thread_id"])
            )
        return {"threads": items, "next_cursor": next_cursor}

    async def thread_timeline(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        cursor: int = 0,
        limit: int = 30,
    ) -> dict[str, Any] | None:
        self._require_started()
        db = self._require_runs_db()
        thread_cursor = await db.execute(
            """
            SELECT
                thread.thread_id,
                thread.title,
                thread.channel,
                thread.archived,
                thread.created_at,
                thread.updated_at,
                latest.run_id AS last_run_id,
                COALESCE(latest.status, 'idle') AS status,
                COALESCE(NULLIF(latest.final_text, ''), latest.request_text, '') AS preview
            FROM agent_threads AS thread
            LEFT JOIN agent_runs AS latest
              ON latest.run_id = (
                  SELECT run.run_id
                  FROM agent_runs AS run
                  WHERE run.tenant_id = thread.tenant_id
                    AND run.thread_id = thread.thread_id
                  ORDER BY run.created_at DESC, run.run_id DESC
                  LIMIT 1
              )
            WHERE thread.tenant_id = ? AND thread.thread_id = ?
            """,
            (tenant_id, thread_id),
        )
        thread = await thread_cursor.fetchone()
        await thread_cursor.close()
        if thread is None:
            return None
        safe_cursor = max(0, int(cursor))
        safe_limit = min(max(1, int(limit)), 100)
        runs_cursor = await db.execute(
            """
            SELECT *
            FROM agent_runs
            WHERE tenant_id = ? AND thread_id = ?
            ORDER BY created_at, run_id
            LIMIT ? OFFSET ?
            """,
            (tenant_id, thread_id, safe_limit + 1, safe_cursor),
        )
        runs = list(await runs_cursor.fetchall())
        await runs_cursor.close()
        has_more = len(runs) > safe_limit
        runs = runs[:safe_limit]
        entries: list[dict[str, Any]] = []
        for run in runs:
            run_id = str(run["run_id"])
            request_text = str(run["request_text"] or "")
            try:
                file_ids = json.loads(str(run["file_ids_json"] or "[]"))
            except ValueError:
                file_ids = []
            safe_file_ids = (
                [str(file_id) for file_id in file_ids if str(file_id).strip()]
                if isinstance(file_ids, list)
                else []
            )
            if request_text or safe_file_ids:
                entries.append(
                    {
                        "id": f"{run_id}:user",
                        "type": "user",
                        "run_id": run_id,
                        "timestamp": str(run["created_at"]),
                        "text": request_text,
                        "file_ids": safe_file_ids,
                        "attachments": self._timeline_attachments(
                            tenant_id=tenant_id,
                            file_ids=safe_file_ids,
                        ),
                    }
                )
            event_cursor = await db.execute(
                """
                SELECT sequence, event_type, timestamp, data_json
                FROM agent_run_events
                WHERE run_id = ?
                ORDER BY sequence
                """,
                (run_id,),
            )
            for event in await event_cursor.fetchall():
                event_type = str(event["event_type"])
                try:
                    data = json.loads(str(event["data_json"] or "{}"))
                except ValueError:
                    data = {}
                entries.append(
                    {
                        "id": f"{run_id}:{int(event['sequence'])}",
                        "type": self._timeline_type(event_type),
                        "event_type": event_type,
                        "run_id": run_id,
                        "sequence": int(event["sequence"]),
                        "timestamp": str(event["timestamp"]),
                        "data": data if isinstance(data, dict) else {},
                    }
                )
            await event_cursor.close()
            final_text = str(run["final_text"] or "")
            if final_text:
                entries.append(
                    {
                        "id": f"{run_id}:assistant",
                        "type": "assistant",
                        "run_id": run_id,
                        "timestamp": str(run["updated_at"]),
                        "text": final_text,
                    }
                )
        summary = self._thread_row(thread)
        return {
            "thread": summary,
            "entries": entries,
            "next_cursor": safe_cursor + safe_limit if has_more else None,
        }

    def _timeline_attachments(
        self,
        *,
        tenant_id: str,
        file_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        resolver = self._attachment_resolver
        attachments: list[dict[str, Any]] = []
        for file_id in file_ids:
            record = resolver.get_file(tenant_id, file_id) if resolver is not None else None
            attachments.append(
                {
                    "id": file_id,
                    "kind": str((record or {}).get("kind") or "file"),
                    "original_filename": str((record or {}).get("original_filename") or file_id),
                    "mime_type": (str((record or {}).get("mime_type") or "").strip() or None),
                    "size_bytes": max(0, int((record or {}).get("size_bytes") or 0)),
                    "available": record is not None,
                }
            )
        return attachments

    async def update_thread(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        title: str | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any] | None:
        self._require_started()
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now_iso()]
        if title is not None:
            normalized = self._normalize_thread_title(title)
            if not normalized:
                raise ValueError("thread title cannot be empty")
            fields.append("title = ?")
            values.append(normalized)
        if archived is not None:
            fields.append("archived = ?")
            values.append(1 if archived else 0)
        values.extend((tenant_id, thread_id))
        db = self._require_runs_db()
        result = await db.execute(
            f"UPDATE agent_threads SET {', '.join(fields)} WHERE tenant_id = ? AND thread_id = ?",
            values,
        )
        found = result.rowcount == 1
        await result.close()
        await db.commit()
        if not found:
            return None
        cursor = await db.execute(
            """
            SELECT thread_id, title, channel, archived, created_at, updated_at
            FROM agent_threads WHERE tenant_id = ? AND thread_id = ?
            """,
            (tenant_id, thread_id),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return self._thread_row(row) if row is not None else None

    async def get_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
    ) -> dict[str, Any] | None:
        self._require_started()
        db = self._require_runs_db()
        thread = await db.execute(
            "SELECT 1 FROM agent_threads WHERE tenant_id = ? AND thread_id = ?",
            (tenant_id, thread_id),
        )
        exists = await thread.fetchone()
        await thread.close()
        if exists is None:
            return None
        return await self.get_owner_inference(tenant_id=tenant_id)

    async def update_thread_inference(
        self,
        *,
        tenant_id: str,
        thread_id: str,
        expected_revision: int,
        selection: InferenceSelection | None,
    ) -> dict[str, Any] | None:
        self._require_started()
        db = self._require_runs_db()
        thread = await db.execute(
            "SELECT 1 FROM agent_threads WHERE tenant_id = ? AND thread_id = ?",
            (tenant_id, thread_id),
        )
        exists = await thread.fetchone()
        await thread.close()
        if exists is None:
            return None
        return await self.update_owner_inference(
            tenant_id=tenant_id,
            expected_revision=expected_revision,
            selection=selection,
        )

    async def get_owner_inference(self, *, tenant_id: str) -> dict[str, Any]:
        self._require_started()
        db = self._require_runs_db()
        cursor = await db.execute(
            """
            SELECT revision, selection_json
            FROM tenant_inference_preferences
            WHERE tenant_id = ?
            """,
            (tenant_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        revision = int(row["revision"] or 0) if row is not None else 0
        raw_selection = str(row["selection_json"] or "") if row is not None else ""
        selection = (
            InferenceSelection.model_validate_json(raw_selection) if raw_selection.strip() else None
        )
        effective = selection or self._default_inference_plan().primary
        return {
            "scope": "owner",
            "revision": revision,
            "selection": selection.model_dump(mode="json") if selection is not None else None,
            "effective": effective.model_dump(mode="json"),
        }

    async def update_owner_inference(
        self,
        *,
        tenant_id: str,
        expected_revision: int,
        selection: InferenceSelection | None,
    ) -> dict[str, Any]:
        self._require_started()
        if selection is not None and self._inference is not None:
            selection = await self._inference.validate_selection(tenant_id, selection)
        elif selection is not None and selection.provider != "api":
            raise InferenceUnavailableError("Codex inference is unavailable")
        db = self._require_runs_db()
        async with self._run_event_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """
                    SELECT revision FROM tenant_inference_preferences
                    WHERE tenant_id = ?
                    """,
                    (tenant_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                revision = int(row["revision"] or 0) if row is not None else 0
                if revision != int(expected_revision):
                    await db.rollback()
                    raise InferenceConflictError("owner inference preference changed")
                next_revision = revision + 1
                await db.execute(
                    """
                    INSERT INTO tenant_inference_preferences (
                        tenant_id, revision, selection_json, updated_at
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT (tenant_id) DO UPDATE SET
                        revision = excluded.revision,
                        selection_json = excluded.selection_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        tenant_id,
                        next_revision,
                        selection.model_dump_json() if selection is not None else None,
                        utc_now_iso(),
                    ),
                )
                await db.commit()
            except BaseException:
                with suppress(Exception):
                    await db.rollback()
                raise
        return await self.get_owner_inference(tenant_id=tenant_id)

    async def codex_preference_count(self, tenant_id: str) -> int:
        db = self._require_runs_db()
        cursor = await db.execute(
            """
            SELECT COUNT(*)
            FROM tenant_inference_preferences
            WHERE tenant_id = ?
              AND json_extract(selection_json, '$.provider') = 'codex'
            """,
            (tenant_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0] or 0) if row is not None else 0

    async def reset_codex_preferences(self, tenant_id: str) -> int:
        db = self._require_runs_db()
        async with self._run_event_lock:
            global_cursor = await db.execute(
                """
                UPDATE tenant_inference_preferences
                SET revision = revision + 1, selection_json = NULL, updated_at = ?
                WHERE tenant_id = ?
                  AND json_extract(selection_json, '$.provider') = 'codex'
                """,
                (utc_now_iso(), tenant_id),
            )
            changed = global_cursor.rowcount
            await global_cursor.close()
            legacy_cursor = await db.execute(
                """
                UPDATE thread_inference_preferences
                SET revision = revision + 1, selection_json = NULL, updated_at = ?
                WHERE tenant_id = ?
                  AND json_extract(selection_json, '$.provider') = 'codex'
                """,
                (utc_now_iso(), tenant_id),
            )
            await legacy_cursor.close()
            await db.commit()
        return max(0, int(changed))

    def _build_graphs(self) -> dict[str, Any]:
        assert self._checkpointer is not None
        assert self._store is not None
        model = _with_deepagents_context_budget(
            self._provided_model or self._build_model(),
            provider="api",
        )
        owner_tools = [tool for tool in self._tools if tool.name in _OWNER_PRODUCT_TOOL_NAMES]
        owner_interrupt_on = self._owner_interrupt_for_tools(
            {tool.name for tool in owner_tools} | {"execute"}
        )
        routine_tools = [tool for tool in owner_tools if tool.name in _ROUTINE_PRODUCT_TOOL_NAMES]
        intake_tools = [tool for tool in owner_tools if tool.name in _INTAKE_PRODUCT_TOOL_NAMES]
        middleware = [
            _ShellCommandPolicyMiddleware(),
            *self._provider_fallback_middleware(),
        ]

        return {
            "owner": create_deep_agent(
                model=model,
                name="opentulpa_owner",
                tools=owner_tools,
                system_prompt=OWNER_PROMPT,
                skills=["/skills/"],
                memory=["/memories/AGENTS.md"],
                backend=self._owner_backend(),
                interrupt_on=owner_interrupt_on,
                context_schema=AgentRunContext,
                checkpointer=self._checkpointer,
                store=self._store,
                middleware=middleware,
            ),
            "routine": create_deep_agent(
                model=model,
                name="opentulpa_routine",
                tools=routine_tools,
                system_prompt=ROUTINE_PROMPT,
                backend=StateBackend(),
                interrupt_on=None,
                context_schema=AgentRunContext,
                checkpointer=self._checkpointer,
                middleware=self._provider_fallback_middleware(),
            ),
            "intake": create_deep_agent(
                model=model,
                name="opentulpa_intake",
                tools=intake_tools,
                system_prompt=INTAKE_PROMPT,
                backend=StateBackend(),
                response_format=IntakeDecision,
                context_schema=AgentRunContext,
                checkpointer=self._checkpointer,
                middleware=self._provider_fallback_middleware(),
            ),
        }

    def _graph_for_context(
        self,
        context: AgentRunContext,
        *,
        dynamic: DynamicToolSnapshot | None = None,
        inference_plan: ResolvedInferencePlan | None = None,
    ) -> Any:
        active_plan = inference_plan or self._default_inference_plan()
        if self._agent_specs is None:
            if active_plan.primary != self._default_inference_plan().primary:
                raise RuntimeError("custom inference requires revisioned AgentSpecs")
            graph = self._graphs.get(str(context.run_kind))
            if graph is None:
                raise RuntimeError("the requested agent profile is unavailable")
            return graph

        spec = self._agent_specs.get_revision(context.agent_spec)
        if spec is None:
            raise RuntimeError("the requested AgentSpec revision does not exist")
        self._validate_context_spec(context, spec)
        active_plan = self._effective_spec_plan(spec, active_plan)
        active_dynamic = dynamic or self._dynamic_snapshot(spec.tenant_id)
        key = (
            spec.tenant_id,
            spec.id,
            spec.revision,
            spec.content_digest,
            active_dynamic.generation,
            self._dynamic_digest(active_dynamic),
            *self._inference_cache_key(spec.tenant_id, active_plan),
        )
        graph = self._spec_graphs.get(key)
        if graph is None:
            graph = self._compile_spec_graph(
                spec,
                dynamic=active_dynamic,
                inference_plan=active_plan,
            )
            self._cache_spec_graph(key, graph)
        else:
            self._spec_graphs.move_to_end(key)
        return graph

    def _compile_spec_graph(
        self,
        spec: AgentSpec,
        *,
        dynamic: DynamicToolSnapshot | None = None,
        inference_plan: ResolvedInferencePlan | None = None,
    ) -> Any:
        assert self._checkpointer is not None
        assert self._store is not None
        active_dynamic = dynamic or self._dynamic_snapshot(spec.tenant_id)
        tools = self._tools_for_spec(spec, dynamic=active_dynamic)
        tool_names = {tool.name for tool in tools}
        interrupt_on: dict[str, bool | InterruptOnConfig] = {}
        if spec.runtime_profile == AgentRunKind.OWNER.value:
            owner_tool_names = set(tool_names)
            if spec.workspace_scope == "read_write":
                owner_tool_names.add("execute")
            interrupt_on = self._owner_interrupt_for_tools(owner_tool_names)
        backend, uses_store = self._backend_for_spec(spec)
        denied: set[str] = set()
        permissions: list[FilesystemPermission] = []
        if not spec.allow_delegation:
            denied.add("task")
        if spec.isolation == "external":
            denied.update(_FILESYSTEM_TOOL_NAMES)
            denied.add("write_todos")
            permissions.append(
                FilesystemPermission(
                    operations=["read", "write"],
                    paths=["/**"],
                    mode="deny",
                )
            )
        elif spec.workspace_scope == "none":
            denied.add("execute")
            permissions.append(
                FilesystemPermission(
                    operations=["read", "write"],
                    paths=["/workspace/**"],
                    mode="deny",
                )
            )
        elif spec.workspace_scope == "read_only":
            denied.update({"edit_file", "execute", "write_file"})
            permissions.append(
                FilesystemPermission(
                    operations=["write"],
                    paths=["/workspace/**"],
                    mode="deny",
                )
            )

        active_plan = self._effective_spec_plan(
            spec,
            inference_plan or self._default_inference_plan(),
        )
        resolved_model = self._model_for_spec(spec, active_plan)
        model = _with_deepagents_context_budget(
            resolved_model.model,
            provider=active_plan.primary.provider,
        )
        middleware: list[Any] = [
            ModelCallLimitMiddleware(
                run_limit=spec.max_model_calls,
                exit_behavior="error",
            ),
            _ShellCommandPolicyMiddleware(),
        ]
        middleware.extend(self._middleware_for_plan(active_plan, resolved_model))
        if denied:
            middleware.append(_DenyToolsMiddleware(frozenset(denied)))
        kwargs: dict[str, Any] = {
            "model": model,
            "name": f"opentulpa_{spec.id}_r{spec.revision}",
            "tools": tools,
            "system_prompt": self._prompt_for_spec(spec),
            "backend": backend,
            "interrupt_on": interrupt_on or None,
            "context_schema": AgentRunContext,
            "checkpointer": self._checkpointer,
            "middleware": middleware,
            "permissions": permissions or None,
        }
        if uses_store:
            kwargs["store"] = self._store
        if uses_store:
            kwargs["memory"] = ["/memories/AGENTS.md"]
            kwargs["skills"] = ["/skills/"]
        if spec.runtime_profile == AgentRunKind.INTAKE.value:
            kwargs["response_format"] = IntakeDecision
        elif spec.output_schema is not None:
            kwargs["response_format"] = spec.output_schema
        return create_deep_agent(**kwargs)

    def _provider_fallback_middleware(self) -> list[AgentMiddleware]:
        if not self._provider_fallback_models:
            return []
        return [_ProviderFallbackMiddleware(self._provider_fallback_models)]

    def _middleware_for_plan(
        self,
        plan: ResolvedInferencePlan,
        resolved: ResolvedModel,
    ) -> list[AgentMiddleware]:
        middleware: list[AgentMiddleware] = [_InferenceMessageMiddleware(plan.primary.provider)]
        if plan.primary.provider == "api":
            middleware.extend(self._provider_fallback_middleware())
            return middleware
        if plan.primary.fallback_to_api:
            api_primary = self._provided_model or self._build_model()
            middleware.append(
                _ProviderFallbackMiddleware(
                    (api_primary, *self._provider_fallback_models),
                    eligible=is_codex_transient,
                    allow_request=_before_current_run_activity,
                )
            )
        middleware.append(_CodexAuthRetryMiddleware(resolved))
        return middleware

    def _model_for_spec(
        self,
        spec: AgentSpec,
        plan: ResolvedInferencePlan,
    ) -> ResolvedModel:
        if spec.model_alias != "default":
            return ResolvedModel(model=self._resolve_spec_model(spec.model_alias))
        if plan.primary.provider == "codex":
            if self._inference is None:
                raise RuntimeError("Codex inference is unavailable")
            return self._inference.resolve_model(spec.tenant_id, plan.primary)
        if (
            plan.primary.model == self._model_name
            and plan.primary.reasoning_effort == self._reasoning_effort
            and plan.primary.service_tier is None
        ):
            return ResolvedModel(model=self._provided_model or self._build_model())
        return ResolvedModel(
            model=build_openrouter_chat_model(
                api_key=self._api_key,
                base_url=self._base_url,
                model_name=plan.primary.model,
                reasoning_effort=plan.primary.reasoning_effort,
                max_completion_tokens=self._max_completion_tokens,
            )
        )

    def _default_inference_plan(self) -> ResolvedInferencePlan:
        selection = (
            self._inference.default_selection
            if self._inference is not None
            else InferenceSelection(
                provider="api",
                model=self._model_name,
                reasoning_effort=self._reasoning_effort,
            )
        )
        return ResolvedInferencePlan.resolve(selection, preference_revision=0)

    async def _resolve_inference_plan(
        self,
        context: AgentRunContext,
    ) -> ResolvedInferencePlan:
        default = self._default_inference_plan()
        db = self._require_runs_db()
        cursor = await db.execute(
            """
            SELECT revision, selection_json
            FROM tenant_inference_preferences
            WHERE tenant_id = ?
            """,
            (context.tenant_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        plan = default
        if row is not None:
            raw_selection = str(row["selection_json"] or "")
            selection = (
                InferenceSelection.model_validate_json(raw_selection)
                if raw_selection.strip()
                else default.primary
            )
            plan = ResolvedInferencePlan.resolve(
                selection,
                preference_revision=int(row["revision"] or 0),
            )
        if self._agent_specs is not None:
            spec = self._agent_specs.get_revision(context.agent_spec)
            if spec is not None and spec.model_alias != "default":
                return self._effective_spec_plan(spec, plan)
        return plan

    def _effective_spec_plan(
        self,
        spec: AgentSpec,
        plan: ResolvedInferencePlan,
    ) -> ResolvedInferencePlan:
        if spec.model_alias == "default":
            return plan
        return ResolvedInferencePlan.resolve(
            InferenceSelection(
                provider="api",
                model=spec.model_alias,
                reasoning_effort=self._reasoning_effort,
            ),
            preference_revision=plan.preference_revision,
        )

    def _inference_cache_key(
        self,
        tenant_id: str,
        plan: ResolvedInferencePlan,
    ) -> tuple[Any, ...]:
        credential_revision = (
            self._inference.credential_revision(tenant_id)
            if self._inference is not None and plan.primary.provider == "codex"
            else 0
        )
        return (
            plan.primary.provider,
            plan.primary.model,
            plan.primary.reasoning_effort,
            plan.primary.service_tier,
            plan.primary.fallback_to_api,
            credential_revision,
        )

    def _cache_spec_graph(self, key: tuple[Any, ...], graph: Any) -> None:
        self._spec_graphs[key] = graph
        self._spec_graphs.move_to_end(key)
        while len(self._spec_graphs) > self._graph_cache_limit:
            self._spec_graphs.popitem(last=False)

    def preflight_agent_spec(self, spec: AgentSpec) -> None:
        """Compile an exact revision before its active pointer can change."""

        self._require_started()
        required_profiles = {
            AgentRunKind.OWNER.value: (AgentRunKind.OWNER.value, "private"),
            AgentRunKind.ROUTINE.value: (AgentRunKind.ROUTINE.value, "private"),
            AgentRunKind.INTAKE.value: (AgentRunKind.INTAKE.value, "external"),
        }
        expected = required_profiles.get(spec.id)
        if expected is not None and (spec.runtime_profile, spec.isolation) != expected:
            raise ValueError(
                f"the {spec.id} AgentSpec must retain its runtime profile and isolation boundary"
            )
        dynamic = self._dynamic_snapshot(spec.tenant_id)
        key = (
            spec.tenant_id,
            spec.id,
            spec.revision,
            spec.content_digest,
            dynamic.generation,
            self._dynamic_digest(dynamic),
            *self._inference_cache_key(spec.tenant_id, self._default_inference_plan()),
        )
        if key in self._spec_graphs:
            return
        try:
            graph = self._compile_spec_graph(
                spec,
                dynamic=dynamic,
                inference_plan=self._default_inference_plan(),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("the AgentSpec cannot compile in the active runtime") from exc
        self._cache_spec_graph(key, graph)

    def _resolve_spec_model(self, alias: str) -> Any:
        if self._model_resolver is not None:
            return self._model_resolver(alias)
        if alias not in {"default", self._model_name}:
            raise RuntimeError("the AgentSpec model alias is not configured")
        return self._provided_model or self._build_model()

    def _tools_for_spec(
        self,
        spec: AgentSpec,
        *,
        dynamic: DynamicToolSnapshot | None = None,
    ) -> list[BaseTool]:
        active_dynamic = dynamic or self._dynamic_snapshot(spec.tenant_id)
        dynamic_tools = () if spec.isolation == "external" else active_dynamic.tools
        dynamic_names = frozenset(tool.name for tool in dynamic_tools)
        all_tools = (*self._tools, *dynamic_tools)
        available = {tool.name: tool for tool in all_tools}
        if len(available) != len(all_tools):
            raise RuntimeError("active capability tools collide with kernel tools")
        if spec.tool_policy == "profile_default":
            available_names = frozenset(available)
            names = {
                AgentRunKind.OWNER.value: available_names,
                AgentRunKind.ROUTINE.value: _ROUTINE_PRODUCT_TOOL_NAMES & available_names,
                AgentRunKind.INTAKE.value: _INTAKE_PRODUCT_TOOL_NAMES & available_names,
            }.get(spec.runtime_profile)
            if names is None:
                raise RuntimeError("custom AgentSpecs must declare an explicit tool allowlist")
        else:
            names = frozenset(spec.tools)
        boundary = (
            _INTAKE_PRODUCT_TOOL_NAMES
            if spec.isolation == "external"
            else None
            if spec.id == "owner"
            else _ROUTINE_PRODUCT_TOOL_NAMES | dynamic_names
        )
        unsafe = sorted(set(names) - boundary) if boundary is not None else []
        if unsafe:
            raise RuntimeError(
                f"AgentSpec {spec.id!r} exceeds its execution boundary: " + ", ".join(unsafe)
            )
        missing = sorted(set(names) - set(available))
        if missing:
            raise RuntimeError("AgentSpec tools are unavailable: " + ", ".join(missing))
        return [tool for tool in all_tools if tool.name in names]

    def _dynamic_snapshot(self, tenant_id: str) -> DynamicToolSnapshot:
        if self._dynamic_tools is None:
            return DynamicToolSnapshot(generation=0, tools=(), interrupt_on={})
        return self._dynamic_tools.snapshot(tenant_id)

    @staticmethod
    def _dynamic_digest(dynamic: DynamicToolSnapshot) -> str:
        tools: list[dict[str, Any]] = []
        for tool in sorted(dynamic.tools, key=lambda item: item.name):
            try:
                schema_source = tool.tool_call_schema
                if isinstance(schema_source, dict):
                    schema = schema_source
                else:
                    render_schema = getattr(schema_source, "model_json_schema", None)
                    if not callable(render_schema):
                        render_schema = schema_source.schema
                    schema = render_schema()
            except Exception:
                schema = {"unavailable": True}
            tools.append(
                {
                    "name": tool.name,
                    "description": str(tool.description or ""),
                    "schema": schema,
                    "metadata": redact_for_langfuse(dict(tool.metadata or {})),
                    "implementation": {
                        "class": f"{type(tool).__module__}.{type(tool).__qualname__}",
                        "callable": DeepAgentService._tool_callable_identity(tool),
                    },
                }
            )
        canonical = json.dumps(
            {
                "tools": tools,
                "interrupt_on": sorted(
                    (str(name), bool(required)) for name, required in dynamic.interrupt_on.items()
                ),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _tool_callable_identity(tool: BaseTool) -> dict[str, str]:
        target = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
        if target is None:
            return {}
        code = getattr(target, "__code__", None)
        code_digest = ""
        if code is not None:
            payload = b"\0".join(
                (
                    bytes(code.co_code),
                    repr(code.co_consts).encode("utf-8", errors="replace"),
                    repr(code.co_names).encode("utf-8", errors="replace"),
                )
            )
            code_digest = hashlib.sha256(payload).hexdigest()
        return {
            "module": str(getattr(target, "__module__", "") or ""),
            "qualname": str(getattr(target, "__qualname__", "") or ""),
            "code_digest": code_digest,
        }

    @classmethod
    def _dynamic_binding(cls, dynamic: DynamicToolSnapshot) -> tuple[int, str]:
        return dynamic.generation, cls._dynamic_digest(dynamic)

    async def _verified_dynamic_snapshot(
        self,
        run_id: str,
        context: AgentRunContext,
    ) -> DynamicToolSnapshot:
        db = self._require_runs_db()
        cursor = await db.execute(
            "SELECT dynamic_generation, dynamic_digest FROM agent_runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise KeyError(f"agent run not found: {run_id}")
        current = self._dynamic_snapshot(context.tenant_id)
        expected = (int(row["dynamic_generation"] or 0), str(row["dynamic_digest"] or ""))
        if expected != self._dynamic_binding(current):
            raise AgentRunCapabilityConflictError(_PUBLIC_CAPABILITY_CHANGED_MESSAGE)
        return current

    def _backend_for_spec(self, spec: AgentSpec) -> tuple[Any, bool]:
        if spec.isolation == "external":
            return StateBackend(), False

        default: Any = StateBackend()
        routes: dict[str, Any] = {}
        uses_store = spec.memory_scope != "none"
        if spec.workspace_scope == "read_write":
            if self._workspace_backend is not None and spec.runtime_profile == "owner":
                default = self._execution_backend or self._workspace_backend
                routes["/workspace/"] = self._workspace_backend
            else:
                default = TenantSandboxBackend(
                    workspaces_root=self._workspaces_root,
                    policy=self._container_policy,
                    container_cli=self._container_cli,
                    persistent_execution_workspace=True,
                    execution_provider=self._execution_provider,
                )
                routes["/workspace/"] = TenantSandboxBackend(
                    workspaces_root=self._workspaces_root,
                    policy=self._container_policy,
                    container_cli=self._container_cli,
                    persistent_files=True,
                    execution_provider=self._execution_provider,
                )
        elif spec.workspace_scope == "read_only":
            routes["/workspace/"] = TenantSandboxBackend(
                workspaces_root=self._workspaces_root,
                policy=self._container_policy,
                container_cli=self._container_cli,
                persistent_files=True,
                execution_provider=self._execution_provider,
            )
        if uses_store:
            routes["/memories/"] = StoreBackend(
                store=self._store,
                namespace=self._store_namespace_for_spec(spec, "memory"),
            )
            routes["/skills/"] = StoreBackend(
                store=self._store,
                namespace=self._store_namespace_for_spec(spec, "skills"),
            )
        if routes:
            return CompositeBackend(default=default, routes=routes), uses_store
        return default, False

    @staticmethod
    def _store_namespace_for_spec(
        spec: AgentSpec,
        kind: Literal["memory", "skills"],
    ) -> Callable[[Any], tuple[str, ...]]:
        def namespace(runtime: Any) -> tuple[str, ...]:
            tenant_id = runtime.context.tenant_id
            if spec.memory_scope == "owner":
                return tenant_store_namespace(tenant_id, kind)
            return (
                "tenant",
                tenant_namespace_label(tenant_id),
                "agent-spec",
                spec.id,
                kind,
            )

        return namespace

    @staticmethod
    def _prompt_for_spec(spec: AgentSpec) -> str:
        base = {
            AgentRunKind.OWNER.value: OWNER_PROMPT,
            AgentRunKind.ROUTINE.value: ROUTINE_PROMPT,
            AgentRunKind.INTAKE.value: INTAKE_PROMPT,
        }.get(spec.runtime_profile, "You are an OpenTulpa agent configured by the owner.")
        return f"{base}\n\nActive AgentSpec instructions:\n{spec.instructions}"

    @staticmethod
    def _validate_context_spec(context: AgentRunContext, spec: AgentSpec) -> None:
        if context.run_kind != spec.runtime_profile:
            raise RuntimeError(
                "AgentSpec runtime profile does not match the authenticated run kind"
            )
        authenticated_external = context.trust_class == "external"
        if authenticated_external != (spec.isolation == "external"):
            raise RuntimeError("AgentSpec isolation does not match the authenticated origin")
        if context.trust_class == "background" and spec.id == "owner":
            raise RuntimeError("background runs cannot use the owner AgentSpec")

    async def decide_intake(
        self,
        *,
        context: AgentRunContext,
        decision_input: dict[str, Any],
    ) -> IntakeDecision:
        """Ask the isolated intake profile for typed advice without applying side effects."""

        self._require_started()
        if context.run_kind != AgentRunKind.INTAKE.value:
            raise ValueError("intake decisions require run_kind=intake")
        inference_plan = await self._resolve_inference_plan(context)
        graph = self._graph_for_context(context, inference_plan=inference_plan)
        checkpoint_thread_id = self._checkpoint_thread_id(context)
        graph_input = {
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        decision_input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
            ]
        }
        async with self._checkpoint_lock(checkpoint_thread_id):
            with self._trace_context(
                context,
                graph_input,
                inference_plan=inference_plan,
            ):
                async with asyncio.timeout(self._runtime_limit_for_context(context)):
                    state = await graph.ainvoke(
                        graph_input,
                        config=self._run_config(
                            context,
                            checkpoint_thread_id,
                            inference_plan=inference_plan,
                        ),
                        context=context,
                    )
        response = state.get("structured_response") if isinstance(state, dict) else None
        if isinstance(response, IntakeDecision):
            return response
        if isinstance(response, dict):
            return IntakeDecision.model_validate(response)
        raise RuntimeError("intake agent did not return IntakeDecision")

    def _build_model(self) -> ChatOpenRouter:
        return build_openrouter_chat_model(
            api_key=self._api_key,
            base_url=self._base_url,
            model_name=self._model_name,
            reasoning_effort=self._reasoning_effort,
            max_completion_tokens=self._max_completion_tokens,
        )

    def _owner_backend(self) -> CompositeBackend:
        workspace = self._workspace_backend
        if workspace is None:
            workspace = TenantSandboxBackend(
                workspaces_root=self._workspaces_root,
                policy=self._container_policy,
                container_cli=self._container_cli,
                persistent_files=True,
                execution_provider=self._execution_provider,
            )
        execution = self._execution_backend
        if execution is None:
            execution = TenantSandboxBackend(
                workspaces_root=self._workspaces_root,
                policy=self._container_policy,
                container_cli=self._container_cli,
                persistent_execution_workspace=True,
                execution_provider=self._execution_provider,
            )
        return CompositeBackend(
            default=execution,
            routes={
                "/memories/": StoreBackend(
                    store=self._store,
                    namespace=lambda rt: tenant_store_namespace(rt.context.tenant_id, "memory"),
                ),
                "/skills/": StoreBackend(
                    store=self._store,
                    namespace=lambda rt: tenant_store_namespace(rt.context.tenant_id, "skills"),
                ),
                "/workspace/": workspace,
            },
        )

    @staticmethod
    def _owner_interrupt_for_tools(
        tool_names: set[str],
    ) -> dict[str, bool | InterruptOnConfig]:
        policy: dict[str, bool | InterruptOnConfig] = {
            name: InterruptOnConfig(
                allowed_decisions=list(_OWNER_DECISIONS),
                when=_shell_command_requires_approval,
            )
            for name in {"execute", "source_shell"}.intersection(tool_names)
        }
        if "source_release" in tool_names:
            # Keep release handoff restart-safe without surfacing an approval.
            policy["source_release"] = InterruptOnConfig(allowed_decisions=list(_OWNER_DECISIONS))
        return policy

    @staticmethod
    def _validate_product_tools(tools: Sequence[BaseTool]) -> tuple[BaseTool, ...]:
        validated = tuple(tools)
        names = [tool.name for tool in validated]
        unknown = sorted(set(names) - _OWNER_PRODUCT_TOOL_NAMES)
        if unknown:
            raise ValueError(f"unknown product tools: {', '.join(unknown)}")
        duplicates = sorted(name for name in set(names) if names.count(name) > 1)
        if duplicates:
            raise ValueError(f"duplicate product tools: {', '.join(duplicates)}")
        return validated

    async def _stream_graph(
        self,
        *,
        run_id: str,
        context: AgentRunContext,
        checkpoint_thread_id: str,
        graph_input: Any,
        resumed: bool,
        inference_plan: ResolvedInferencePlan,
    ) -> AsyncIterator[AgentRunEvent]:
        final_parts: list[str] = []
        pending_ai_chunk: AIMessageChunk | None = None
        interrupted = False
        started_data: dict[str, Any] = {
            "thread_id": context.thread_id,
            "resumed": resumed,
            "provider": inference_plan.primary.provider,
            "model": inference_plan.primary.model,
            "reasoning_effort": inference_plan.primary.reasoning_effort,
            "service_tier": inference_plan.primary.service_tier,
            "inference_plan_digest": inference_plan.digest,
            "preference_revision": inference_plan.preference_revision,
        }
        if not resumed:
            started_data["input_summary"] = _trace_input_summary(graph_input)
        started_event = await self._append_event(
            run_id=run_id,
            type="run.started",
            data=started_data,
        )
        if started_event is None:
            return
        yield started_event
        trace_context = self._trace_context(context, graph_input, inference_plan=inference_plan)
        failure_phase = "trace_setup"
        try:
            with trace_context:
                config = self._run_config(
                    context,
                    checkpoint_thread_id,
                    inference_plan=inference_plan,
                )
                failure_phase = "capability_resolution"
                dynamic = await self._verified_dynamic_snapshot(run_id, context)
                graph = self._graph_for_context(
                    context,
                    dynamic=dynamic,
                    inference_plan=inference_plan,
                )
                runtime_limit = self._runtime_limit_for_context(context)
                failure_phase = "agent_loop"
                async with asyncio.timeout(runtime_limit):
                    async for part in graph.astream(
                        graph_input,
                        config=config,
                        context=context,
                        stream_mode=["messages", "updates", "custom"],
                        version="v2",
                    ):
                        if await self._run_status(run_id) in _TERMINAL_RUN_STATUSES:
                            return
                        part_type = str(part.get("type", "")) if isinstance(part, dict) else ""
                        data = part.get("data") if isinstance(part, dict) else None
                        if part_type == "messages":
                            message = data[0] if isinstance(data, tuple | list) and data else data
                            metadata = (
                                data[1]
                                if isinstance(data, tuple | list)
                                and len(data) > 1
                                and isinstance(data[1], dict)
                                else {}
                            )
                            if metadata.get("lc_source") == "summarization":
                                continue
                            message_events = self._message_events(message)
                            if isinstance(message, AIMessageChunk):
                                pending_ai_chunk = (
                                    message
                                    if pending_ai_chunk is None
                                    else pending_ai_chunk + message
                                )
                                if message.chunk_position == "last":
                                    message_events.extend(self._tool_start_events(pending_ai_chunk))
                                    pending_ai_chunk = None
                            elif isinstance(message, ToolMessage) and pending_ai_chunk is not None:
                                message_events = [
                                    *self._tool_start_events(pending_ai_chunk),
                                    *message_events,
                                ]
                                pending_ai_chunk = None
                            for event_type, event_data, text in message_events:
                                if text:
                                    final_parts.append(text)
                                event = await self._append_event(
                                    run_id=run_id,
                                    type=event_type,
                                    data=event_data,
                                )
                                if event is None:
                                    return
                                yield event
                        approvals = self._approvals_from_stream(data, run_id)
                        if approvals:
                            interrupted = True
                            if context.run_kind == AgentRunKind.OWNER.value and any(
                                approval.tool_name == "source_release" for approval in approvals
                            ):
                                decided = [
                                    replace(approval, status="approve")
                                    if approval.tool_name == "source_release"
                                    else approval
                                    for approval in approvals
                                ]
                                pending = [
                                    approval for approval in decided if approval.status == "pending"
                                ]
                                updated = await self._update_run(
                                    run_id,
                                    status="interrupted" if pending else "resume_pending",
                                    approvals=decided,
                                    final_text="".join(final_parts).strip(),
                                    allowed_statuses={"running", "resume_pending"},
                                )
                                if not updated:
                                    return
                                if not pending:
                                    self._schedule_pending_resume(run_id)
                                for approval in pending:
                                    event = await self._append_event(
                                        run_id=run_id,
                                        type="approval.required",
                                        data=self._public_approval(approval),
                                    )
                                    if event is None:
                                        return
                                    yield event
                                return
                            updated = await self._update_run(
                                run_id,
                                status="interrupted",
                                approvals=approvals,
                                final_text="".join(final_parts).strip(),
                                allowed_statuses={"running", "resume_pending"},
                            )
                            if not updated:
                                return
                            for approval in approvals:
                                event = await self._append_event(
                                    run_id=run_id,
                                    type="approval.required",
                                    data=self._public_approval(approval),
                                )
                                if event is None:
                                    return
                                yield event
                if interrupted:
                    return
                if await self._run_status(run_id) in _TERMINAL_RUN_STATUSES:
                    return
                failure_phase = "finalization"
                final_text = "".join(final_parts).strip()
                if not final_text:
                    state = await graph.aget_state(config)
                    final_text = self._last_ai_text(state.values.get("messages", []))
                event = await self._transition_with_event(
                    run_id,
                    allowed_statuses={"running", "resume_pending"},
                    status="completed",
                    event_type="run.completed",
                    event_data={"text": final_text},
                    final_text=final_text,
                    approvals=[],
                )
                if event is None:
                    return
                await self._observe_run(run_id)
                yield event
        except Exception as exc:
            final_text = "".join(final_parts).strip()
            if _is_model_provider_failure(exc) and final_text:
                logger.warning(
                    "Model provider failed after emitting output; preserving response: run_id=%s",
                    run_id,
                )
                event = await self._transition_with_event(
                    run_id,
                    allowed_statuses={"running", "resume_pending"},
                    status="completed",
                    event_type="run.completed",
                    event_data={"text": final_text},
                    final_text=final_text,
                    approvals=[],
                )
                if event is None:
                    return
                await self._observe_run(run_id)
                yield event
                return
            logger.exception("Deep Agent run failed: run_id=%s", run_id)
            failure_code, failure_message, retryable = _public_run_failure(exc)
            last_tool_error = (
                await self._latest_tool_error(run_id)
                if failure_code == "model_provider_failed"
                else None
            )
            if last_tool_error is not None:
                failure_message = (
                    f"{failure_message} The last operation also failed: "
                    f"{last_tool_error['message']}"
                )
            failure_event_data: dict[str, Any] = {
                "code": failure_code,
                "message": failure_message,
                "retryable": retryable,
                "diagnostic": _failure_diagnostic(exc, phase=failure_phase),
            }
            if last_tool_error is not None:
                failure_event_data["last_tool_error"] = last_tool_error
            event = await self._transition_with_event(
                run_id,
                allowed_statuses={"running", "resume_pending"},
                status="failed",
                event_type="run.failed",
                event_data=failure_event_data,
                error=failure_message,
            )
            if event is None:
                return
            await self._observe_run(run_id)
            yield event

    async def _latest_tool_error(self, run_id: str) -> dict[str, str] | None:
        db = self._require_runs_db()
        cursor = await db.execute(
            """
            SELECT data_json
            FROM agent_run_events
            WHERE run_id = ? AND event_type = 'tool.completed'
            ORDER BY sequence DESC
            LIMIT 1
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        try:
            data = json.loads(str(row["data_json"] or "{}"))
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        result = data.get("result")
        raw_error: Any = data.get("error") if data.get("ok") is False else None
        if isinstance(result, dict) and result.get("status") == "error":
            raw_error = result.get("error")
        if isinstance(raw_error, dict):
            message = str(raw_error.get("message") or "").strip()
        else:
            message = str(raw_error or "").strip()
        if not message:
            return None
        safe_message = str(redact_for_langfuse(message))[:600]
        return {
            "tool_name": str(data.get("name") or "tool")[:200],
            "message": safe_message,
        }

    def _run_config(
        self,
        context: AgentRunContext,
        checkpoint_thread_id: str,
        *,
        inference_plan: ResolvedInferencePlan | None = None,
    ) -> dict[str, Any]:
        callbacks: list[Any] = []
        tracer = self._langfuse_tracer
        if tracer is not None and hasattr(tracer, "build_callbacks"):
            callbacks = tracer.build_callbacks(
                user_id=context.tenant_id,
                trace_id=context.correlation_id,
                session_id=context.thread_id,
                metadata={
                    "run_kind": str(context.run_kind),
                    "channel": str(context.channel),
                    **self._inference_trace_metadata(inference_plan),
                },
                tags=["deepagents", str(context.run_kind)],
            )
        return {
            "configurable": {"thread_id": checkpoint_thread_id},
            "callbacks": callbacks,
            "metadata": {
                "tenant_id": context.tenant_id,
                "actor_id": context.actor_id,
                "correlation_id": context.correlation_id,
                "run_kind": str(context.run_kind),
                "channel": str(context.channel),
                **self._inference_trace_metadata(inference_plan),
            },
        }

    def _runtime_limit_for_context(self, context: AgentRunContext) -> float | None:
        if self._agent_specs is None:
            return None
        spec = self._agent_specs.get_revision(context.agent_spec)
        return float(spec.max_runtime_seconds) if spec is not None else None

    def _trace_context(
        self,
        context: AgentRunContext,
        graph_input: Any,
        *,
        inference_plan: ResolvedInferencePlan | None = None,
    ) -> Any:
        tracer = self._langfuse_tracer
        if tracer is None or not hasattr(tracer, "trace_context"):
            return nullcontext()
        return tracer.trace_context(
            name=f"deepagent.{context.run_kind}",
            trace_id=context.correlation_id,
            user_id=context.tenant_id,
            session_id=context.thread_id,
            input=graph_input,
            metadata={
                "channel": str(context.channel),
                "actor_id": context.actor_id,
                **self._inference_trace_metadata(inference_plan),
            },
            tags=["deepagents", str(context.run_kind)],
        )

    @staticmethod
    def _inference_trace_metadata(
        plan: ResolvedInferencePlan | None,
    ) -> dict[str, Any]:
        if plan is None:
            return {}
        return {
            "inference_provider": plan.primary.provider,
            "inference_model": plan.primary.model,
            "reasoning_effort": plan.primary.reasoning_effort,
            "service_tier": plan.primary.service_tier,
            "inference_plan_digest": plan.digest,
            "preference_revision": plan.preference_revision,
        }

    @staticmethod
    def _message_events(message: Any) -> list[tuple[AgentRunEventType, dict[str, Any], str]]:
        events: list[tuple[AgentRunEventType, dict[str, Any], str]] = []
        if isinstance(message, AIMessage | AIMessageChunk):
            text = DeepAgentService._message_text(message)
            if text:
                events.append(("message.delta", {"text": text}, text))
            if isinstance(message, AIMessage) and not isinstance(message, AIMessageChunk):
                events.extend(DeepAgentService._tool_start_events(message))
        elif isinstance(message, ToolMessage):
            ok = str(getattr(message, "status", "success")) != "error"
            result = message.content
            if isinstance(result, str):
                with suppress(TypeError, ValueError):
                    result = json.loads(result)
            result_key = "result" if ok else "error"
            events.append(
                (
                    "tool.completed",
                    {
                        "name": str(getattr(message, "name", "") or "")[:200],
                        "call_id": str(getattr(message, "tool_call_id", "") or "")[:200],
                        "ok": ok,
                        result_key: _bounded_trace_value(result),
                    },
                    "",
                )
            )
            events.extend(DeepAgentService._artifact_events(message))
        return events

    @staticmethod
    def _tool_start_events(
        message: AIMessage | AIMessageChunk,
    ) -> list[tuple[AgentRunEventType, dict[str, Any], str]]:
        events: list[tuple[AgentRunEventType, dict[str, Any], str]] = []
        for call in message.tool_calls:
            name = str(call.get("name") or "").strip()
            call_id = str(call.get("id") or "").strip()
            if not name or not call_id or call_id.casefold() == "none":
                continue
            events.append(
                (
                    "tool.started",
                    {
                        "name": name[:200],
                        "call_id": call_id[:200],
                        "arguments": _bounded_trace_value(call.get("args", {})),
                    },
                    "",
                )
            )
        return events

    @staticmethod
    def _artifact_events(
        message: ToolMessage,
    ) -> list[tuple[AgentRunEventType, dict[str, Any], str]]:
        """Surface artifacts when the native graph observes job event/artifact tool output."""

        tool_name = str(getattr(message, "name", "") or "")
        if tool_name not in {"job_events", "job_artifacts"}:
            return []
        content = message.content
        if not isinstance(content, str):
            return []
        try:
            envelope = json.loads(content)
        except (TypeError, ValueError):
            return []
        if not isinstance(envelope, dict) or envelope.get("status") != "ok":
            return []
        raw_items = envelope.get("data")
        if not isinstance(raw_items, list):
            return []
        artifacts: list[dict[str, Any]] = []
        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            if tool_name == "job_events":
                if raw.get("event_type") != "artifact.ready":
                    continue
                payload = raw.get("payload")
                if not isinstance(payload, dict):
                    continue
                raw = (
                    payload.get("artifact")
                    if isinstance(payload.get("artifact"), dict)
                    else payload
                )
            artifact_id = str(raw.get("id") or raw.get("artifact_id") or "").strip()
            job_id = str(raw.get("job_id") or "").strip()
            name = str(raw.get("name") or "").strip()
            if not artifact_id or not job_id or not name:
                continue
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "job_id": job_id,
                    "name": name[:300],
                    "media_type": str(raw.get("media_type") or "")[:200],
                    "size_bytes": (
                        raw.get("size_bytes") if isinstance(raw.get("size_bytes"), int) else None
                    ),
                }
            )
        return [("artifact.ready", artifact, "") for artifact in artifacts]

    @staticmethod
    def _message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text") or ""))
        return "".join(parts)

    @classmethod
    def _last_ai_text(cls, messages: Sequence[Any]) -> str:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                text = cls._message_text(message).strip()
                if text:
                    return text
        return ""

    @staticmethod
    def _approvals_from_stream(data: Any, run_id: str) -> list[AgentApproval]:
        interrupts: list[Any] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key == "__interrupt__":
                        if isinstance(item, list | tuple):
                            interrupts.extend(item)
                        else:
                            interrupts.append(item)
                    else:
                        collect(item)
            elif isinstance(value, list | tuple):
                for item in value:
                    collect(item)

        collect(data)
        approvals: list[AgentApproval] = []
        seen_ids: set[str] = set()
        for interrupt in interrupts:
            value = getattr(interrupt, "value", interrupt)
            if not isinstance(value, dict):
                continue
            interrupt_id = str(getattr(interrupt, "id", "") or "").strip()
            actions = value.get("action_requests") or []
            configs = value.get("review_configs") or []
            for index, action in enumerate(actions):
                if not isinstance(action, dict):
                    continue
                config = (
                    configs[index]
                    if index < len(configs) and isinstance(configs[index], dict)
                    else {}
                )
                tool_name = str(action.get("name") or config.get("action_name") or "").strip()
                binding = json.dumps(
                    {
                        "interrupt_id": interrupt_id,
                        "index": index,
                        "tool_name": tool_name,
                        "arguments": action.get("args") or {},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                    default=str,
                )
                call_key = hashlib.sha256(binding.encode("utf-8")).hexdigest()[:32]
                approval_id = f"approval_{run_id}_{call_key}"
                if approval_id in seen_ids:
                    continue
                seen_ids.add(approval_id)
                approvals.append(
                    AgentApproval(
                        id=approval_id,
                        tool_name=tool_name,
                        description=str(action.get("description") or "Approval required"),
                        arguments=dict(action.get("args") or {}),
                        allowed_decisions=tuple(
                            config.get("allowed_decisions") or ("approve", "edit", "reject")
                        ),
                    )
                )
        return approvals

    @staticmethod
    def _public_approval(approval: AgentApproval) -> dict[str, Any]:
        return {
            "approval_id": approval.id,
            "tool_name": approval.tool_name,
            "description": approval.description,
            "arguments": redact_for_langfuse(approval.arguments),
            "allowed_decisions": list(approval.allowed_decisions),
        }

    @staticmethod
    def _resume_decision(approval: AgentApproval) -> dict[str, Any]:
        if approval.status == "approve":
            return {"type": "approve"}
        if approval.status == "edit":
            return {
                "type": "edit",
                "edited_action": {
                    "name": approval.tool_name,
                    "args": approval.edited_arguments or {},
                },
            }
        if approval.status == "reject":
            payload: dict[str, Any] = {"type": "reject"}
            if approval.message:
                payload["message"] = approval.message
            return payload
        raise ValueError(f"approval decision is incomplete: {approval.id}")

    async def _insert_run(
        self,
        run_id: str,
        checkpoint_thread_id: str,
        context: AgentRunContext,
        *,
        request_text: str = "",
        file_ids: Sequence[str] = (),
        idempotency_key: str | None = None,
        request_digest: str = "",
        dynamic_generation: int = 0,
        dynamic_digest: str = "",
        inference_plan: ResolvedInferencePlan | None = None,
    ) -> str:
        db = self._require_runs_db()
        now = utc_now_iso()
        public_request_text = _public_text(request_text)
        async with self._run_event_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                if idempotency_key is not None:
                    existing_cursor = await db.execute(
                        """
                        SELECT run_id, request_digest FROM agent_runs
                        WHERE tenant_id = ? AND idempotency_key = ?
                        """,
                        (context.tenant_id, idempotency_key),
                    )
                    existing = await existing_cursor.fetchone()
                    await existing_cursor.close()
                    if existing is not None:
                        await db.rollback()
                        if str(existing["request_digest"] or "") != request_digest:
                            raise AgentRunIdempotencyConflictError(
                                "the idempotency key belongs to a different agent run request"
                            )
                        return str(existing["run_id"])
                active_cursor = await db.execute(
                    """
                    SELECT run_id FROM agent_runs
                    WHERE checkpoint_thread_id = ?
                      AND status IN ('running', 'interrupted', 'resume_pending')
                    LIMIT 1
                    """,
                    (checkpoint_thread_id,),
                )
                active = await active_cursor.fetchone()
                await active_cursor.close()
                if active is not None:
                    await db.rollback()
                    raise AgentRunCheckpointConflictError(
                        "the checkpoint thread already has an unresolved agent run"
                    )
                await db.execute(
                    """
                    INSERT INTO agent_threads (
                        tenant_id, thread_id, title, channel, archived, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 0, ?, ?)
                    ON CONFLICT (tenant_id, thread_id) DO UPDATE SET
                        title = CASE
                            WHEN agent_threads.title IN ('New session', 'Previous session')
                              AND excluded.title != 'New session'
                            THEN excluded.title
                            ELSE agent_threads.title
                        END,
                        channel = excluded.channel,
                        archived = 0,
                        updated_at = excluded.updated_at
                    """,
                    (
                        context.tenant_id,
                        context.thread_id,
                        self._thread_title(public_request_text),
                        str(context.channel),
                        now,
                        now,
                    ),
                )
                await db.execute(
                    """
                    INSERT INTO agent_runs (
                        run_id, tenant_id, actor_id, thread_id, checkpoint_thread_id,
                        channel, run_kind, origin_json, agent_spec_id, agent_spec_revision,
                        trust_class, correlation_id, idempotency_key, status, created_at,
                        updated_at, request_digest, dynamic_generation, dynamic_digest,
                        request_text, file_ids_json, inference_json
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        run_id,
                        context.tenant_id,
                        context.actor_id,
                        context.thread_id,
                        checkpoint_thread_id,
                        str(context.channel),
                        str(context.run_kind),
                        context.origin.model_dump_json(),
                        context.agent_spec.spec_id,
                        context.agent_spec.revision,
                        context.trust_class,
                        context.correlation_id,
                        idempotency_key,
                        now,
                        now,
                        request_digest,
                        dynamic_generation,
                        dynamic_digest,
                        public_request_text,
                        json.dumps(list(file_ids), ensure_ascii=False),
                        (inference_plan or self._default_inference_plan()).model_dump_json(),
                    ),
                )
                await db.commit()
            except BaseException:
                with suppress(Exception):
                    await db.rollback()
                raise
        return run_id

    @staticmethod
    def _normalize_thread_title(value: str | None) -> str:
        title = " ".join(str(value or "").split())
        return title[:120]

    @classmethod
    def _thread_title(cls, request_text: str) -> str:
        text = " ".join(str(request_text or "").split())
        if not text or text == _REGENERATE_COMMAND:
            return "New session"
        return cls._normalize_thread_title(text[:80])

    @staticmethod
    def _encode_thread_cursor(updated_at: str, thread_id: str) -> str:
        payload = json.dumps([updated_at, thread_id], separators=(",", ":"))
        return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_thread_cursor(value: str | None) -> tuple[str, str]:
        if not value:
            return "", ""
        try:
            raw = str(value)
            raw += "=" * (-len(raw) % 4)
            payload = json.loads(base64.urlsafe_b64decode(raw).decode("utf-8"))
            if not isinstance(payload, list) or len(payload) != 2:
                return "", ""
            return str(payload[0]), str(payload[1])
        except (ValueError, UnicodeDecodeError):
            return "", ""

    @staticmethod
    def _thread_row(row: aiosqlite.Row) -> dict[str, Any]:
        keys = set(row.keys())
        return {
            "thread_id": str(row["thread_id"]),
            "title": str(row["title"]),
            "channel": str(row["channel"]),
            "archived": bool(row["archived"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "last_run_id": (
                str(row["last_run_id"]) if "last_run_id" in keys and row["last_run_id"] else None
            ),
            "status": str(row["status"]) if "status" in keys else "idle",
            "preview": str(row["preview"] or "")[:240] if "preview" in keys else "",
        }

    @staticmethod
    def _timeline_type(event_type: str) -> str:
        if event_type.startswith("tool."):
            return "tool"
        if event_type == "approval.required":
            return "approval"
        if event_type == "artifact.ready":
            return "artifact"
        if event_type == "run.failed":
            return "error"
        return "event"

    async def _reconcile_stale_running_runs(self) -> None:
        db = self._require_runs_db()
        cursor = await db.execute(
            "SELECT run_id FROM agent_runs WHERE status = 'running' ORDER BY created_at"
        )
        run_ids = [str(row["run_id"]) for row in await cursor.fetchall()]
        await cursor.close()
        for run_id in run_ids:
            await self._cancel_with_event(run_id, allowed_statuses={"running"})
        if run_ids:
            logger.warning("Cancelled %s stale Deep Agent run(s) during startup", len(run_ids))

    async def _recover_pending_resumes(self) -> None:
        db = self._require_runs_db()
        cursor = await db.execute(
            "SELECT run_id FROM agent_runs WHERE status = 'resume_pending' ORDER BY created_at"
        )
        run_ids = [str(row["run_id"]) for row in await cursor.fetchall()]
        await cursor.close()
        for run_id in run_ids:
            self._schedule_pending_resume(run_id)

    def _schedule_pending_resume(self, run_id: str) -> None:
        if self._shutting_down or run_id in self._pending_resume_ids or self._runs_db is None:
            return
        task = asyncio.create_task(
            self._recover_pending_resume(run_id),
            name=f"opentulpa-resume-recovery:{run_id}",
        )
        self._pending_resume_ids.add(run_id)
        self._pending_resume_tasks.add(task)
        task.add_done_callback(lambda completed: self._pending_resume_done(run_id, completed))

    async def _recover_pending_resume(self, run_id: str) -> None:
        snapshot = await self.get_run(run_id)
        if snapshot is None or snapshot.status != "resume_pending":
            return
        if any(approval.status == "pending" for approval in snapshot.approvals):
            await self._update_run(
                run_id,
                status="interrupted",
                allowed_statuses={"resume_pending"},
            )
            return
        try:
            dynamic = await self._verified_dynamic_snapshot(run_id, snapshot.context)
            async for _ in self._resume_claimed_run(
                run_id=run_id,
                snapshot=snapshot,
                approvals=snapshot.approvals,
                dynamic=dynamic,
            ):
                pass
        except asyncio.CancelledError:
            raise
        except AgentRunCapabilityConflictError:
            event = await self._transition_with_event(
                run_id,
                allowed_statuses={"resume_pending"},
                status="failed",
                event_type="run.failed",
                event_data={
                    "code": "agent_capability_changed",
                    "message": _PUBLIC_CAPABILITY_CHANGED_MESSAGE,
                    "retryable": False,
                },
                error=_PUBLIC_CAPABILITY_CHANGED_MESSAGE,
            )
            if event is not None:
                await self._observe_run(run_id)
        except Exception:
            logger.exception("Pending Deep Agent resume recovery failed: run_id=%s", run_id)

    def _pending_resume_done(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._pending_resume_ids.discard(run_id)
        self._pending_resume_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Pending Deep Agent resume recovery stopped: error_type=%s",
                type(error).__name__,
            )

    async def _finalize_abandoned_run(self, run_id: str) -> None:
        cleanup = asyncio.create_task(self._cancel_running_run(run_id))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(cleanup)
            except Exception:
                logger.exception("Failed to persist cancelled Deep Agent run: run_id=%s", run_id)
            raise
        except Exception:
            logger.exception("Failed to persist cancelled Deep Agent run: run_id=%s", run_id)

    async def _cancel_running_run(self, run_id: str) -> None:
        await self._cancel_with_event(run_id, allowed_statuses={"running"})

    async def _update_run(
        self,
        run_id: str,
        *,
        status: str | None = None,
        final_text: str | None = None,
        error: str | None = None,
        approvals: Sequence[AgentApproval] | None = None,
        allowed_statuses: set[str],
    ) -> bool:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now_iso()]
        safe_final_text = _public_text(final_text) if final_text is not None else None
        for column, value in (
            ("status", status),
            ("final_text", safe_final_text),
            ("error", error),
        ):
            if value is not None:
                fields.append(f"{column} = ?")
                values.append(value)
        if approvals is not None:
            fields.append("approvals_json = ?")
            values.append(self._serialize_approvals(approvals))
        statuses = sorted(allowed_statuses)
        placeholders = ", ".join("?" for _ in statuses)
        values.append(run_id)
        values.extend(statuses)
        db = self._require_runs_db()
        async with self._run_event_lock:
            cursor = await db.execute(
                f"UPDATE agent_runs SET {', '.join(fields)} "
                f"WHERE run_id = ? AND status IN ({placeholders})",
                values,
            )
            updated = cursor.rowcount == 1
            await cursor.close()
            await db.commit()
        return updated

    async def _cancel_with_event(self, run_id: str, *, allowed_statuses: set[str]) -> None:
        event = await self._transition_with_event(
            run_id,
            allowed_statuses=allowed_statuses,
            status="cancelled",
            event_type="run.failed",
            event_data={
                "code": "agent_run_cancelled",
                "message": _PUBLIC_RUN_CANCELLED_MESSAGE,
                "retryable": False,
            },
            error=_PUBLIC_RUN_CANCELLED_MESSAGE,
        )
        if event is not None:
            await self._observe_run(run_id)

    async def _transition_with_event(
        self,
        run_id: str,
        *,
        allowed_statuses: set[str],
        status: AgentRunStatus,
        event_type: AgentRunEventType,
        event_data: dict[str, Any],
        final_text: str | None = None,
        error: str | None = None,
        approvals: Sequence[AgentApproval] | None = None,
    ) -> AgentRunEvent | None:
        """Atomically persist one conditional state transition and its terminal event."""

        timestamp = utc_now_iso()
        safe_data = cast("dict[str, Any]", redact_for_langfuse(event_data))
        safe_final_text = _public_text(final_text) if final_text is not None else None
        db = self._require_runs_db()
        async with self._run_event_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "SELECT status, last_sequence FROM agent_runs WHERE run_id = ?",
                    (run_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise KeyError(f"agent run not found: {run_id}")
                if str(row["status"]) not in allowed_statuses:
                    await db.rollback()
                    return None
                sequence = int(row["last_sequence"] or 0) + 1
                fields = ["status = ?", "last_sequence = ?", "updated_at = ?"]
                values: list[Any] = [status, sequence, timestamp]
                if final_text is not None:
                    fields.append("final_text = ?")
                    values.append(safe_final_text)
                if error is not None:
                    fields.append("error = ?")
                    values.append(error)
                if approvals is not None:
                    fields.append("approvals_json = ?")
                    values.append(self._serialize_approvals(approvals))
                values.append(run_id)
                await db.execute(
                    f"UPDATE agent_runs SET {', '.join(fields)} WHERE run_id = ?",
                    values,
                )
                await db.execute(
                    """
                    UPDATE agent_threads
                    SET updated_at = ?
                    WHERE (tenant_id, thread_id) = (
                        SELECT tenant_id, thread_id FROM agent_runs WHERE run_id = ?
                    )
                    """,
                    (timestamp, run_id),
                )
                await db.execute(
                    """
                    INSERT INTO agent_run_events (
                        run_id, sequence, event_type, timestamp, data_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        sequence,
                        event_type,
                        timestamp,
                        json.dumps(safe_data, ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )
                await db.commit()
            except BaseException:
                with suppress(Exception):
                    await db.rollback()
                raise
        event = AgentRunEvent(
            type=event_type,
            run_id=run_id,
            sequence=sequence,
            timestamp=timestamp,
            data=safe_data,
        )
        await self._notify_run_event(run_id)
        return event

    async def _observe_run(self, run_id: str) -> None:
        observer = self._run_observer
        if observer is None:
            return
        snapshot = await self.get_run(run_id)
        if snapshot is None or snapshot.status not in {"completed", "failed", "cancelled"}:
            return
        try:
            await observer(snapshot)
        except Exception:
            logger.exception("Deep Agent run observer failed: run_id=%s", run_id)

    async def _last_sequence(self, run_id: str) -> int:
        db = self._require_runs_db()
        cursor = await db.execute(
            "SELECT last_sequence FROM agent_runs WHERE run_id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise KeyError(f"agent run not found: {run_id}")
        return int(row["last_sequence"] or 0)

    async def _append_event(
        self,
        *,
        run_id: str,
        type: AgentRunEventType,
        data: dict[str, Any],
    ) -> AgentRunEvent | None:
        """Allocate a monotonic cursor and persist an event before publishing it."""

        timestamp = utc_now_iso()
        safe_data = cast("dict[str, Any]", redact_for_langfuse(data))
        db = self._require_runs_db()
        async with self._run_event_lock:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "SELECT status, last_sequence FROM agent_runs WHERE run_id = ?",
                    (run_id,),
                )
                row = await cursor.fetchone()
                await cursor.close()
                if row is None:
                    raise KeyError(f"agent run not found: {run_id}")
                if str(row["status"]) in _TERMINAL_RUN_STATUSES:
                    await db.rollback()
                    return None
                sequence = int(row["last_sequence"] or 0) + 1
                await db.execute(
                    """
                    INSERT INTO agent_run_events (
                        run_id, sequence, event_type, timestamp, data_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        sequence,
                        type,
                        timestamp,
                        json.dumps(safe_data, ensure_ascii=False, sort_keys=True, default=str),
                    ),
                )
                await db.execute(
                    "UPDATE agent_runs SET last_sequence = ?, updated_at = ? WHERE run_id = ?",
                    (sequence, timestamp, run_id),
                )
                await db.execute(
                    """
                    UPDATE agent_threads
                    SET updated_at = ?
                    WHERE (tenant_id, thread_id) = (
                        SELECT tenant_id, thread_id FROM agent_runs WHERE run_id = ?
                    )
                    """,
                    (timestamp, run_id),
                )
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        event = AgentRunEvent(
            type=type,
            run_id=run_id,
            sequence=sequence,
            timestamp=timestamp,
            data=safe_data,
        )
        await self._notify_run_event(run_id)
        return event

    async def _run_status(self, run_id: str) -> str:
        db = self._require_runs_db()
        cursor = await db.execute("SELECT status FROM agent_runs WHERE run_id = ?", (run_id,))
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            raise KeyError(f"agent run not found: {run_id}")
        return str(row["status"])

    @staticmethod
    def _approval_json(approval: AgentApproval) -> dict[str, Any]:
        return {
            "id": approval.id,
            "tool_name": approval.tool_name,
            "description": approval.description,
            "arguments": approval.arguments,
            "allowed_decisions": list(approval.allowed_decisions),
            "status": approval.status,
            "edited_arguments": approval.edited_arguments,
            "message": approval.message,
        }

    @classmethod
    def _serialize_approvals(cls, approvals: Sequence[AgentApproval]) -> str:
        return json.dumps([cls._approval_json(item) for item in approvals], sort_keys=True)

    @staticmethod
    def _snapshot_from_row(row: aiosqlite.Row) -> AgentRunSnapshot:
        raw_approvals = json.loads(str(row["approvals_json"] or "[]"))
        approvals = tuple(
            AgentApproval(
                id=str(item["id"]),
                tool_name=str(item["tool_name"]),
                description=str(item["description"]),
                arguments=dict(item.get("arguments") or {}),
                allowed_decisions=tuple(item.get("allowed_decisions") or ()),
                status=cast(AgentApprovalStatus, str(item.get("status") or "pending")),
                edited_arguments=item.get("edited_arguments"),
                message=item.get("message"),
            )
            for item in raw_approvals
        )
        context = AgentRunContext(
            tenant_id=str(row["tenant_id"]),
            actor_id=str(row["actor_id"]),
            thread_id=str(row["thread_id"]),
            channel=str(row["channel"]),
            run_kind=str(row["run_kind"]),
            correlation_id=str(row["correlation_id"]),
            origin=(
                OriginRef.model_validate_json(str(row["origin_json"]))
                if str(row["origin_json"] or "").strip() not in {"", "{}"}
                else OriginRef(interface=str(row["channel"]), source_id="legacy-run")
            ),
            agent_spec=AgentSpecRef(
                tenant_id=str(row["tenant_id"]),
                spec_id=str(row["agent_spec_id"] or row["run_kind"]),
                revision=max(1, int(row["agent_spec_revision"] or 1)),
            ),
            trust_class=cast(
                "Literal['owner', 'background', 'external']",
                str(row["trust_class"] or "owner"),
            ),
        )
        return AgentRunSnapshot(
            run_id=str(row["run_id"]),
            context=context,
            status=cast(AgentRunStatus, str(row["status"])),
            final_text=str(row["final_text"] or ""),
            error=str(row["error"] or ""),
            approvals=approvals,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            inference_plan=(
                ResolvedInferencePlan.model_validate_json(str(row["inference_json"]))
                if str(row["inference_json"] or "").strip()
                else None
            ),
        )

    @staticmethod
    def _checkpoint_thread_id(context: AgentRunContext) -> str:
        thread_digest = hashlib.sha256(context.thread_id.encode("utf-8")).hexdigest()[:32]
        spec = context.agent_spec
        owner_shared = (
            context.trust_class == "owner"
            and context.run_kind == AgentRunKind.OWNER.value
            and spec.spec_id == "owner"
        )
        if owner_shared:
            authority_scope = "owner-shared"
        else:
            authority = "\0".join(
                (
                    context.actor_id,
                    context.origin.interface,
                    context.origin.source_id,
                    context.trust_class,
                )
            )
            authority_digest = hashlib.sha256(authority.encode("utf-8")).hexdigest()[:16]
            authority_scope = f"{context.trust_class}-{authority_digest}"
        return (
            f"{tenant_namespace_label(context.tenant_id)}:spec-{spec.spec_id}"
            f"-r{spec.revision}:{authority_scope}:{context.run_kind}:{thread_digest}"
        )

    @staticmethod
    def _request_digest(request: AgentRunRequest) -> str:
        context = request.context
        canonical = json.dumps(
            {
                "actor_id": context.actor_id,
                "thread_id": context.thread_id,
                "channel": str(context.channel),
                "run_kind": str(context.run_kind),
                "origin": context.origin.model_dump(mode="json"),
                "agent_spec": context.agent_spec.model_dump(mode="json"),
                "trust_class": context.trust_class,
                "text": request.text,
                "file_ids": list(request.file_ids),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _request_text(request: AgentRunRequest) -> str:
        text = str(request.text).strip()
        if (
            text == _REGENERATE_COMMAND
            and request.context.run_kind == AgentRunKind.OWNER.value
            and not request.file_ids
        ):
            return _REGENERATE_INSTRUCTION
        if request.file_ids:
            attached = ", ".join(request.file_ids)
            text = (
                f"{text}\n\nAttached tenant file IDs: {attached}\n"
                "These IDs are not filesystem paths. Their supported contents are attached "
                "to this message; use file_get or file_inspect for additional metadata and "
                "extracted text. Do not pass a file ID to read_file."
            )
        return text

    def _request_content(self, request: AgentRunRequest) -> str | list[dict[str, Any]]:
        text = self._request_text(request)
        resolver = self._attachment_resolver
        if resolver is None or not request.file_ids:
            return text

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        total_inline_bytes = 0
        for file_id in request.file_ids:
            record = resolver.get_file(request.context.tenant_id, file_id)
            if record is None:
                content.append(
                    {
                        "type": "text",
                        "text": f"Attachment {file_id} is unavailable to this tenant.",
                    }
                )
                continue

            filename = str(record.get("original_filename") or file_id)
            mime_type = str(record.get("mime_type") or "application/octet-stream").lower()
            text_excerpt = str(record.get("text_excerpt") or "").strip()
            raw_bytes: bytes | None = None
            if mime_type in _INLINE_IMAGE_MIME_TYPES or mime_type == "application/pdf":
                raw_bytes = resolver.read_file_bytes(request.context.tenant_id, file_id)

            can_inline = (
                raw_bytes is not None
                and len(raw_bytes) <= _MAX_INLINE_ATTACHMENT_BYTES
                and total_inline_bytes + len(raw_bytes) <= _MAX_INLINE_ATTACHMENTS_BYTES
            )
            label = f"Attachment {file_id}: {filename} ({mime_type})"
            if can_inline and raw_bytes is not None:
                encoded = base64.b64encode(raw_bytes).decode("ascii")
                total_inline_bytes += len(raw_bytes)
                content.append({"type": "text", "text": label})
                if mime_type in _INLINE_IMAGE_MIME_TYPES:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        }
                    )
                else:
                    content.append(
                        {
                            "type": "file",
                            "file": {
                                "filename": filename,
                                "file_data": f"data:{mime_type};base64,{encoded}",
                            },
                        }
                    )
                continue

            if text_excerpt:
                content.append(
                    {
                        "type": "text",
                        "text": f"{label}\n<file-content>\n{text_excerpt}\n</file-content>",
                    }
                )
            else:
                content.append(
                    {
                        "type": "text",
                        "text": (
                            f"{label}\nNative content was not inlined. Use the file tools "
                            "for available metadata or extracted content."
                        ),
                    }
                )
        return content

    def _require_started(self) -> None:
        if not self.started:
            raise RuntimeError("DeepAgentService is not started")

    def _require_runs_db(self) -> aiosqlite.Connection:
        if self._runs_db is None:
            raise RuntimeError("DeepAgentService is not started")
        return self._runs_db
