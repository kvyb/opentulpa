"""LangChain adapters for the explicit OpenTulpa product-tool contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol, cast
from uuid import uuid4

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

from opentulpa.inference.models import ResolvedInferencePlan
from opentulpa.tooling.arguments import OPERATION_ARGUMENT_SCHEMAS
from opentulpa.tooling.contract import (
    TOOL_SPEC_BY_NAME,
    TOOL_SPECS,
    AgentRunContext,
    ApprovalMode,
    ExecutionMode,
    IdempotencyMode,
    ToolError,
    ToolResult,
    ToolSpec,
    ToolStatus,
)

logger = logging.getLogger(__name__)

_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|authorization|password|secret|token)\s*[:=]\s*\S+"
)


@dataclass(frozen=True, slots=True)
class ProductToolInvocation:
    """Trusted request passed directly to one application-service operation."""

    spec: ToolSpec
    context: AgentRunContext
    arguments: Mapping[str, Any]
    idempotency_key: str | None
    audit_id: str
    inference_plan: ResolvedInferencePlan | None = None


@dataclass(frozen=True, slots=True)
class ProductToolOutput:
    """Application result before the adapter applies the public result envelope."""

    data: Any = None
    job_id: str | None = None


class ProductToolApplicationError(Exception):
    """Expected application error with an explicitly public, sanitized message."""

    def __init__(
        self,
        code: str,
        public_message: str,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.public_message = public_message
        self.retryable = retryable


class ProductToolApplication(Protocol):
    """Explicit application port; every method must enforce tenant ownership."""

    async def profile_get(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def profile_update(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def file_search(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def file_get(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def file_analyze(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def file_inspect(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def artifact_deliver(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def knowledge_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def knowledge_find(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def knowledge_attach(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def knowledge_archive(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def knowledge_reindex(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def knowledge_query(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def web_search(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def content_fetch(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def browser_start(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def browser_get(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def browser_act(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def browser_stop(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def integration_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def integration_connect(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def connection_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def connection_disconnect(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def integration_action_search(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def integration_invoke(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def intake_workflow_list(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def intake_workflow_get(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def intake_draft_save(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def intake_draft_prepare(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def intake_draft_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def intake_workflow_delete(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def intake_workflow_test(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def schedule_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def schedule_save(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def schedule_delete(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def agent_spec_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def agent_spec_save(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def agent_spec_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def agent_spec_rollback(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def trigger_spec_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def trigger_spec_save(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def trigger_spec_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def trigger_spec_rollback(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def secret_handle_list(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def secret_handle_revoke(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def sandbox_ssh_diagnostic(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def capability_list(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def capability_seed_bundled(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def capability_test(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def capability_activate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def capability_rollback(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def capability_deactivate(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def job_get(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def job_events(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def job_artifacts(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def job_cancel(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def repository_open(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def repository_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def repository_status(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def repository_close(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def repository_publish_pr(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def source_status(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_read(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_write(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_edit(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_bash(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_activate(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_rollback(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def source_runtime_env_get(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def source_set_runtime_env(
        self,
        invocation: ProductToolInvocation,
    ) -> ProductToolOutput: ...

    async def trace_list(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...

    async def trace_get(self, invocation: ProductToolInvocation) -> ProductToolOutput: ...


type ProductToolHandler = Callable[[ProductToolInvocation], Awaitable[ProductToolOutput]]


def _safe_error_code(value: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if _ERROR_CODE_RE.fullmatch(candidate) else "operation_failed"


def _safe_public_message(value: str) -> str:
    message = " ".join(str(value or "").split())
    message = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}=[redacted]", message)
    return message[:500] or "The operation could not be completed."


def _derived_idempotency_key(
    spec: ToolSpec,
    context: AgentRunContext,
    arguments: Mapping[str, Any],
    *,
    tool_call_id: str,
) -> str:
    canonical = json.dumps(
        {
            "tenant_id": context.tenant_id,
            "operation": spec.name,
            "version": spec.version,
            "tool_call_id": tool_call_id,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"derived_{digest}"


def _error_result(
    *,
    audit_id: str,
    idempotency_key: str | None,
    code: str,
    message: str,
    retryable: bool,
) -> dict[str, Any]:
    return ToolResult[Any](
        status=ToolStatus.ERROR,
        error=ToolError(
            code=_safe_error_code(code),
            message=_safe_public_message(message),
            retryable=retryable,
        ),
        idempotency_key=idempotency_key,
        audit_id=audit_id,
    ).model_dump(mode="json")


async def _execute_product_tool(
    *,
    application: ProductToolApplication,
    spec: ToolSpec,
    context: AgentRunContext,
    raw_arguments: Mapping[str, Any],
    tool_call_id: str | None = None,
    inference_plan: ResolvedInferencePlan | None = None,
) -> dict[str, Any]:
    """Validate policy, invoke one direct port method, and sanitize its result."""

    audit_id = f"audit_{uuid4().hex}"
    schema = OPERATION_ARGUMENT_SCHEMAS[spec.name]
    validated = schema.model_validate(dict(raw_arguments)).model_dump(
        mode="json", exclude_none=True
    )
    supplied_key = cast("str | None", validated.pop("idempotency_key", None))
    if spec.idempotency is IdempotencyMode.REQUIRED:
        idempotency_key = supplied_key
        if not idempotency_key:
            return _error_result(
                audit_id=audit_id,
                idempotency_key=None,
                code="idempotency_key_required",
                message="An idempotency key is required for this operation.",
                retryable=False,
            )
    elif spec.idempotency is IdempotencyMode.DERIVED:
        idempotency_key = _derived_idempotency_key(
            spec,
            context,
            validated,
            tool_call_id=str(tool_call_id or f"direct_{audit_id}"),
        )
    else:
        idempotency_key = None

    handler = cast("ProductToolHandler", getattr(application, spec.name))
    invocation = ProductToolInvocation(
        spec=spec,
        context=context,
        arguments=validated,
        idempotency_key=idempotency_key,
        audit_id=audit_id,
        inference_plan=inference_plan,
    )
    try:
        async with asyncio.timeout(spec.timeout_seconds):
            output = await handler(invocation)
        if not isinstance(output, ProductToolOutput):
            raise TypeError("application handler returned an invalid result")
        if spec.execution is ExecutionMode.JOB and not str(output.job_id or "").strip():
            return _error_result(
                audit_id=audit_id,
                idempotency_key=idempotency_key,
                code="invalid_service_response",
                message="The background job was not accepted.",
                retryable=False,
            )
        status = ToolStatus.ACCEPTED if spec.execution is ExecutionMode.JOB else ToolStatus.OK
        result = ToolResult[Any](
            status=status,
            data=output.data,
            job_id=output.job_id,
            idempotency_key=idempotency_key,
            audit_id=audit_id,
        )
        logger.info(
            "product tool completed: audit_id=%s operation=%s tenant=%s status=%s",
            audit_id,
            spec.name,
            context.tenant_id,
            status.value,
        )
        return result.model_dump(mode="json")
    except TimeoutError:
        logger.warning(
            "product tool timed out: audit_id=%s operation=%s tenant=%s",
            audit_id,
            spec.name,
            context.tenant_id,
        )
        return _error_result(
            audit_id=audit_id,
            idempotency_key=idempotency_key,
            code="timeout",
            message="The operation timed out.",
            retryable=True,
        )
    except ProductToolApplicationError as exc:
        logger.info(
            "product tool rejected: audit_id=%s operation=%s tenant=%s code=%s",
            audit_id,
            spec.name,
            context.tenant_id,
            _safe_error_code(exc.code),
        )
        return _error_result(
            audit_id=audit_id,
            idempotency_key=idempotency_key,
            code=exc.code,
            message=exc.public_message,
            retryable=exc.retryable,
        )
    except Exception as exc:
        logger.error(
            "product tool failed: audit_id=%s operation=%s tenant=%s exception=%s",
            audit_id,
            spec.name,
            context.tenant_id,
            type(exc).__name__,
        )
        return _error_result(
            audit_id=audit_id,
            idempotency_key=idempotency_key,
            code="operation_failed",
            message="The operation could not be completed.",
            retryable=False,
        )


def _runtime_schema(spec: ToolSpec) -> type[BaseModel]:
    return _runtime_schema_for_name(spec.name)


@cache
def _runtime_schema_for_name(name: str) -> type[BaseModel]:
    arguments_schema = OPERATION_ARGUMENT_SCHEMAS[name]
    return create_model(
        f"{arguments_schema.__name__}Runtime",
        __base__=arguments_schema,
        runtime=(ToolRuntime[AgentRunContext], Field(exclude=True)),
    )


def _description(spec: ToolSpec) -> str:
    description = {
        "web_search": (
            "Search the public web through the configured provider. Use content_fetch to read "
            "authoritative result pages."
        ),
        "content_fetch": (
            "Fetch and extract a public HTTP(S) URL. When web_search is unavailable, it can "
            "fetch a search-engine results URL such as "
            "https://www.bing.com/search?q=<URL-encoded query> for discovery. Fetch the relevant "
            "result pages before answering."
        ),
        "source_status": (
            "Inspect the persistent OpenTulpa source worktree, active release, previous release, "
            "runtime status, and latest activation."
        ),
        "source_read": (
            "Read UTF-8 text from the persistent OpenTulpa source worktree."
        ),
        "source_write": (
            "Create or replace a UTF-8 file in the persistent OpenTulpa source worktree."
        ),
        "source_edit": (
            "Replace exact text in a file in the persistent OpenTulpa source worktree."
        ),
        "source_bash": (
            "Run a bounded shell command directly in the persistent OpenTulpa source worktree "
            "for Git operations, inspection, edits, or tests."
        ),
        "source_activate": (
            "Commit current source edits and queue host-owned dependency and compile checks, "
            "then child-owned import, contract, health, probation, and rollback checks."
        ),
        "source_set_runtime_env": (
            "Set one live runtime .env variable through the stable host, restart the child, "
            "and restore the previous .env if the restart fails. For credentials, pass the "
            "opaque secret handle as secret_id; never pass plaintext. Never returns the value."
        ),
        "source_runtime_env_get": (
            "Inspect the live runtime .env through the stable host. Owner-only. Returns which "
            "variable names are set, never their values."
        ),
        "sandbox_ssh_diagnostic": (
            "Run one SSH command from the sandbox using an opaque stored private-key or password "
            "secret handle. Never provide plaintext credentials."
        ),
    }.get(spec.name)
    if description is None:
        action = spec.name.replace("_", " ")
        description = f"{action.capitalize()} for the authenticated OpenTulpa tenant."
    approval = "policy" if spec.approval is ApprovalMode.POLICY else spec.approval.value
    return (
        f"{description} "
        f"Effect: {spec.effect.value}; approval: {approval}; "
        f"execution: {spec.execution.value}. "
        "Resource ownership is always resolved from trusted run context."
    )


def _build_tool(application: ProductToolApplication, spec: ToolSpec) -> BaseTool:
    async def execute(
        *,
        runtime: ToolRuntime[AgentRunContext],
        **arguments: Any,
    ) -> dict[str, Any]:
        context = runtime.context
        if not isinstance(context, AgentRunContext):
            audit_id = f"audit_{uuid4().hex}"
            return _error_result(
                audit_id=audit_id,
                idempotency_key=None,
                code="missing_run_context",
                message="Trusted agent run context is unavailable.",
                retryable=False,
            )
        return await _execute_product_tool(
            application=application,
            spec=spec,
            context=context,
            raw_arguments=arguments,
            tool_call_id=runtime.tool_call_id,
            inference_plan=_runtime_inference_plan(runtime),
        )

    execute.__name__ = f"execute_{spec.name}"
    # StructuredTool inspects raw signature annotations rather than resolved type hints.
    execute.__annotations__["runtime"] = ToolRuntime[AgentRunContext]
    return StructuredTool.from_function(
        coroutine=execute,
        name=spec.name,
        description=_description(spec),
        args_schema=_runtime_schema(spec),
    )


def _runtime_inference_plan(
    runtime: ToolRuntime[AgentRunContext],
) -> ResolvedInferencePlan | None:
    configurable = runtime.config.get("configurable", {})
    raw = configurable.get("inference_plan") if isinstance(configurable, dict) else None
    return ResolvedInferencePlan.model_validate(raw) if isinstance(raw, dict) else None


def build_product_tools(
    application: ProductToolApplication,
    *,
    names: Sequence[str] | None = None,
) -> tuple[BaseTool, ...]:
    """Build explicit LangChain tools backed by direct application-service methods."""

    schema_names = set(OPERATION_ARGUMENT_SCHEMAS)
    registry_names = set(TOOL_SPEC_BY_NAME)
    if schema_names != registry_names:
        missing = sorted(registry_names - schema_names)
        extra = sorted(schema_names - registry_names)
        raise RuntimeError(f"tool argument registry mismatch: missing={missing}, extra={extra}")

    selected = set(names) if names is not None else registry_names
    unknown = sorted(selected - registry_names)
    if unknown:
        raise ValueError(f"unknown product tools: {', '.join(unknown)}")
    missing_handlers = sorted(
        spec.name
        for spec in TOOL_SPECS
        if spec.name in selected and not callable(getattr(application, spec.name, None))
    )
    if missing_handlers:
        raise TypeError(f"application port is missing handlers: {', '.join(missing_handlers)}")
    return tuple(_build_tool(application, spec) for spec in TOOL_SPECS if spec.name in selected)


__all__ = [
    "ProductToolApplication",
    "ProductToolApplicationError",
    "ProductToolInvocation",
    "ProductToolOutput",
    "build_product_tools",
]
