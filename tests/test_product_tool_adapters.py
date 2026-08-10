from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from opentulpa.specs import AgentSpecRef, OriginRef
from opentulpa.tooling.adapters import (
    ProductToolApplication,
    ProductToolApplicationError,
    ProductToolInvocation,
    ProductToolOutput,
    _execute_product_tool,
    build_product_tools,
)
from opentulpa.tooling.arguments import OPERATION_ARGUMENT_SCHEMAS
from opentulpa.tooling.contract import (
    TOOL_SPEC_BY_NAME,
    TOOL_SPECS,
    AgentChannel,
    AgentRunContext,
    AgentRunKind,
)


class _RecordingApplication:
    def __init__(
        self,
        behavior: Callable[[ProductToolInvocation], Any] | None = None,
    ) -> None:
        self.invocations: list[ProductToolInvocation] = []
        self.behavior = behavior

    def __getattr__(self, name: str) -> Any:
        if name not in TOOL_SPEC_BY_NAME:
            raise AttributeError(name)

        async def handle(invocation: ProductToolInvocation) -> ProductToolOutput:
            self.invocations.append(invocation)
            if self.behavior is not None:
                result = self.behavior(invocation)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, Exception):
                    raise result
                return cast("ProductToolOutput", result)
            job_id = f"job_{invocation.spec.name}" if invocation.spec.execution == "job" else None
            return ProductToolOutput(
                data={"operation": invocation.spec.name},
                job_id=job_id,
            )

        return handle


def _application(
    behavior: Callable[[ProductToolInvocation], Any] | None = None,
) -> tuple[ProductToolApplication, _RecordingApplication]:
    recording = _RecordingApplication(behavior)
    return cast("ProductToolApplication", recording), recording


def _context(*, tenant_id: str = "tenant-a") -> AgentRunContext:
    return AgentRunContext(
        tenant_id=tenant_id,
        actor_id="actor-1",
        thread_id="thread-1",
        channel=AgentChannel.WEB,
        run_kind=AgentRunKind.OWNER,
        correlation_id="correlation-1",
        origin=OriginRef(interface="web", source_id="test"),
        agent_spec=AgentSpecRef(tenant_id=tenant_id, spec_id="owner", revision=1),
        trust_class="owner",
    )


async def _invoke_through_tool_node(
    tool: Any,
    *,
    context: AgentRunContext,
    arguments: dict[str, Any],
    call_id: str = "call-1",
) -> dict[str, Any]:
    builder = StateGraph(MessagesState, context_schema=AgentRunContext)
    builder.add_node("tools", ToolNode([tool]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    result = await graph.ainvoke(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool.name,
                            "args": arguments,
                            "id": call_id,
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        },
        context=context,
    )
    message = result["messages"][-1]
    assert isinstance(message, ToolMessage)
    assert isinstance(message.content, str)
    return cast("dict[str, Any]", json.loads(message.content))


def test_factory_covers_exact_registry_and_hides_all_host_context_fields() -> None:
    application, _ = _application()
    tools = build_product_tools(application)
    rebuilt_tools = build_product_tools(application)

    assert tuple(tool.name for tool in tools) == tuple(spec.name for spec in TOOL_SPECS)
    assert tuple(tool.args_schema for tool in rebuilt_tools) == tuple(
        tool.args_schema for tool in tools
    )
    assert set(OPERATION_ARGUMENT_SCHEMAS) == set(TOOL_SPEC_BY_NAME)
    for tool in tools:
        schema = tool.tool_call_schema.model_json_schema()
        properties = schema.get("properties", {})
        spec = TOOL_SPEC_BY_NAME[tool.name]
        assert f"Effect: {spec.effect.value}" in tool.description
        approval = "policy" if spec.approval.value == "policy" else "auto"
        assert f"approval: {approval}" in tool.description
        assert f"execution: {spec.execution.value}" in tool.description
        assert "runtime" not in properties
        assert "context" not in properties
        assert "tenant_id" not in properties
        assert "actor_id" not in properties
        assert "thread_id" not in properties
        assert "correlation_id" not in properties

    deliver = next(tool for tool in tools if tool.name == "artifact_deliver")
    assert "idempotency_key" in deliver.tool_call_schema.model_json_schema()["required"]
    activate = next(tool for tool in tools if tool.name == "intake_draft_activate")
    activate_properties = activate.tool_call_schema.model_json_schema()["properties"]
    assert "confirmation_handle" in activate_properties
    assert "confirmation_token" not in activate_properties
    capability_activate = next(tool for tool in tools if tool.name == "capability_activate")
    capability_schema = capability_activate.tool_call_schema.model_json_schema()
    assert {
        "capability_name",
        "revision",
        "config",
        "secret_handles",
        "idempotency_key",
    } <= set(capability_schema["properties"])
    assert "idempotency_key" in capability_schema["required"]
    assert "tenant_id" not in capability_schema["properties"]
    serialized_capability_schema = json.dumps(capability_schema).lower()
    assert '"plaintext"' not in serialized_capability_schema
    assert '"secret_value"' not in serialized_capability_schema
    profile_get = next(tool for tool in tools if tool.name == "profile_get")
    assert profile_get.tool_call_schema.model_json_schema()["properties"] == {}
    web_search = next(tool for tool in tools if tool.name == "web_search")
    assert "configured provider" in web_search.description
    assert "content_fetch" in web_search.description
    content_fetch = next(tool for tool in tools if tool.name == "content_fetch")
    assert "https://www.bing.com/search?q=<URL-encoded query>" in content_fetch.description
    assert "result pages before answering" in content_fetch.description
    source_status = next(tool for tool in tools if tool.name == "source_status")
    assert source_status.tool_call_schema.model_json_schema()["properties"] == {}
    assert "persistent OpenTulpa source worktree" in source_status.description
    source_read = next(tool for tool in tools if tool.name == "source_read")
    assert set(source_read.tool_call_schema.model_json_schema()["properties"]) == {
        "path",
        "offset",
        "limit",
    }
    source_write = next(tool for tool in tools if tool.name == "source_write")
    assert set(source_write.tool_call_schema.model_json_schema()["properties"]) == {
        "path",
        "content",
    }
    source_edit = next(tool for tool in tools if tool.name == "source_edit")
    assert set(source_edit.tool_call_schema.model_json_schema()["properties"]) == {
        "path",
        "old_text",
        "new_text",
        "replace_all",
    }
    source_bash = next(tool for tool in tools if tool.name == "source_bash")
    assert set(source_bash.tool_call_schema.model_json_schema()["properties"]) == {
        "command",
        "timeout_seconds",
    }
    source_activate = next(tool for tool in tools if tool.name == "source_activate")
    assert set(source_activate.tool_call_schema.model_json_schema()["properties"]) == {
        "idempotency_key",
        "message",
        "reason",
    }
    source_rollback = next(tool for tool in tools if tool.name == "source_rollback")
    assert set(source_rollback.tool_call_schema.model_json_schema()["properties"]) == {
        "expected_active_release_id",
        "idempotency_key",
        "reason",
    }
    source_env_get = next(tool for tool in tools if tool.name == "source_runtime_env_get")
    assert source_env_get.tool_call_schema.model_json_schema()["properties"] == {}
    assert "never their values" in source_env_get.description
    source_env = next(tool for tool in tools if tool.name == "source_set_runtime_env")
    assert set(source_env.tool_call_schema.model_json_schema()["properties"]) == {
        "name",
        "value",
        "secret_id",
        "idempotency_key",
    }
    assert "opaque secret handle" in source_env.description
    assert "Never returns the value" in source_env.description


def test_runtime_env_arguments_require_exactly_one_value_source() -> None:
    schema = OPERATION_ARGUMENT_SCHEMAS["source_set_runtime_env"]

    secret = schema.model_validate(
        {
            "name": "COMPOSIO_API_KEY",
            "secret_id": "composio_api_key",
            "idempotency_key": "runtime-env-secret-1",
        }
    )
    assert secret.secret_id == "composio_api_key"
    assert secret.value is None
    with pytest.raises(ValueError, match="exactly one"):
        schema.model_validate(
            {
                "name": "COMPOSIO_API_KEY",
                "idempotency_key": "runtime-env-missing-1",
            }
        )
    with pytest.raises(ValueError, match="exactly one"):
        schema.model_validate(
            {
                "name": "COMPOSIO_API_KEY",
                "value": "plaintext",
                "secret_id": "composio_api_key",
                "idempotency_key": "runtime-env-both-1",
            }
        )


@pytest.mark.asyncio
async def test_tool_node_injects_trusted_context_and_adapter_calls_direct_port() -> None:
    application, recording = _application()
    tool = build_product_tools(application, names=["file_get"])[0]

    result = await _invoke_through_tool_node(
        tool,
        context=_context(tenant_id="trusted-tenant"),
        arguments={"file_id": "file-1"},
    )

    assert result["status"] == "ok"
    assert result["data"] == {"operation": "file_get"}
    assert result["audit_id"].startswith("audit_")
    invocation = recording.invocations[0]
    assert invocation.context.tenant_id == "trusted-tenant"
    assert invocation.context.actor_id == "actor-1"
    assert invocation.arguments == {"file_id": "file-1"}
    assert invocation.idempotency_key is None


@pytest.mark.asyncio
async def test_derived_idempotency_is_tool_call_stable_and_tenant_scoped() -> None:
    application, recording = _application()
    spec = TOOL_SPEC_BY_NAME["profile_update"]

    first = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(tenant_id="tenant-a"),
        raw_arguments={"updates": {"locale": "en"}},
        tool_call_id="call-1",
    )
    second = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(tenant_id="tenant-a"),
        raw_arguments={"updates": {"locale": "en"}},
        tool_call_id="call-1",
    )
    later_intention = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(tenant_id="tenant-a"),
        raw_arguments={"updates": {"locale": "en"}},
        tool_call_id="call-2",
    )
    other_tenant = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(tenant_id="tenant-b"),
        raw_arguments={"updates": {"locale": "en"}},
        tool_call_id="call-1",
    )

    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["idempotency_key"].startswith("derived_")
    assert later_intention["idempotency_key"] != first["idempotency_key"]
    assert other_tenant["idempotency_key"] != first["idempotency_key"]
    assert all("idempotency_key" not in item.arguments for item in recording.invocations)


@pytest.mark.asyncio
async def test_separate_identical_intentions_do_not_replay_an_old_derived_effect() -> None:
    application, _ = _application()
    tool = build_product_tools(application, names=["profile_update"])[0]
    context = _context()

    first = await _invoke_through_tool_node(
        tool,
        context=context,
        arguments={"updates": {"locale": "en"}},
        call_id="call-en-first",
    )
    intervening = await _invoke_through_tool_node(
        tool,
        context=context,
        arguments={"updates": {"locale": "ru"}},
        call_id="call-ru",
    )
    later = await _invoke_through_tool_node(
        tool,
        context=context,
        arguments={"updates": {"locale": "en"}},
        call_id="call-en-later",
    )

    assert len(
        {
            first["idempotency_key"],
            intervening["idempotency_key"],
            later["idempotency_key"],
        }
    ) == 3


@pytest.mark.asyncio
async def test_required_idempotency_is_forwarded_separately_from_arguments() -> None:
    application, recording = _application()
    spec = TOOL_SPEC_BY_NAME["artifact_deliver"]

    result = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(),
        raw_arguments={"artifact_id": "artifact-1", "idempotency_key": "delivery-1"},
    )

    assert result["status"] == "ok"
    assert result["idempotency_key"] == "delivery-1"
    assert recording.invocations[0].idempotency_key == "delivery-1"
    assert recording.invocations[0].arguments == {"artifact_id": "artifact-1"}


@pytest.mark.asyncio
async def test_job_tools_require_job_acceptance_and_return_accepted_envelope() -> None:
    application, _ = _application()
    spec = TOOL_SPEC_BY_NAME["file_analyze"]
    accepted = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(),
        raw_arguments={"file_id": "file-1", "instruction": "Summarize"},
    )
    assert accepted["status"] == "accepted"
    assert accepted["job_id"] == "job_file_analyze"

    invalid_application, _ = _application(lambda invocation: ProductToolOutput(data={}))
    rejected = await _execute_product_tool(
        application=invalid_application,
        spec=spec,
        context=_context(),
        raw_arguments={"file_id": "file-1", "instruction": "Summarize"},
    )
    assert rejected["status"] == "error"
    assert rejected["error"]["code"] == "invalid_service_response"


@pytest.mark.asyncio
async def test_results_are_model_visible_and_expected_errors_are_sanitized() -> None:
    output_application, _ = _application(
        lambda invocation: ProductToolOutput(
            data={"token": "secret-value", "result": "safe"},
        )
    )
    output = await _execute_product_tool(
        application=output_application,
        spec=TOOL_SPEC_BY_NAME["file_get"],
        context=_context(),
        raw_arguments={"file_id": "file-1"},
    )
    assert output["data"] == {"token": "secret-value", "result": "safe"}

    public_error_application, _ = _application(
        lambda invocation: ProductToolApplicationError(
            "not_authorized",
            "Access denied token=must-not-leak",
        )
    )
    public_error = await _execute_product_tool(
        application=public_error_application,
        spec=TOOL_SPEC_BY_NAME["file_get"],
        context=_context(),
        raw_arguments={"file_id": "file-1"},
    )
    assert public_error["error"] == {
        "code": "not_authorized",
        "message": "Access denied token=[redacted]",
        "retryable": False,
    }

    unknown_error_application, _ = _application(lambda invocation: RuntimeError("raw secret"))
    unknown_error = await _execute_product_tool(
        application=unknown_error_application,
        spec=TOOL_SPEC_BY_NAME["file_get"],
        context=_context(),
        raw_arguments={"file_id": "file-1"},
    )
    assert unknown_error["error"]["message"] == "The operation could not be completed."
    assert "raw secret" not in json.dumps(unknown_error)


@pytest.mark.asyncio
async def test_registry_timeout_is_enforced_as_retryable_error() -> None:
    async def slow_handler(invocation: ProductToolInvocation) -> ProductToolOutput:
        await asyncio.sleep(0.05)
        return ProductToolOutput(data={})

    application, _ = _application(slow_handler)
    spec = TOOL_SPEC_BY_NAME["file_get"].model_copy(update={"timeout_seconds": 0.001})

    result = await _execute_product_tool(
        application=application,
        spec=spec,
        context=_context(),
        raw_arguments={"file_id": "file-1"},
    )

    assert result["status"] == "error"
    assert result["error"] == {
        "code": "timeout",
        "message": "The operation timed out.",
        "retryable": True,
    }


def test_factory_rejects_unknown_tools_and_missing_direct_handlers() -> None:
    application, _ = _application()
    with pytest.raises(ValueError, match="unknown product tools"):
        build_product_tools(application, names=["unknown_product_tool"])

    with pytest.raises(TypeError, match="missing handlers"):
        build_product_tools(cast("ProductToolApplication", object()), names=["profile_get"])
