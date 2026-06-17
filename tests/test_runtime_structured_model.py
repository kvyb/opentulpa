from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import tool as lc_tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict

from opentulpa.agent.runtime import (
    INTERACTIVE_NATIVE_TOOL_NAMES,
    ROUTINE_WAKE_NATIVE_TOOL_NAMES,
    OpenTulpaLangGraphRuntime,
)
from opentulpa.agent.tools.tool_gateway_tools import (
    TOOL_GATEWAY_TOOL_NAMES,
    TOOL_GROUP_DEFINITIONS,
)
from opentulpa.agent.turn_plan import build_turn_plan_prompt_context


class _Schema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = False
    reason: str = ""


class _StructuredRunner:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.messages: object | None = None

    async def ainvoke(self, _messages: object, **_: Any) -> object:
        self.messages = _messages
        return self._payload


class _StructuredModel:
    def __init__(self, payload: object) -> None:
        self._payload = payload
        self.runner: _StructuredRunner | None = None

    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        self.runner = _StructuredRunner(self._payload)
        return self.runner


class _RecordingTracer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def trace_context(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return nullcontext()


class _FallbackResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FallbackModel:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: object) -> _FallbackResponse:
        return _FallbackResponse(self._content)


class _SequenceFallbackModel:
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls: list[object] = []

    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        raise RuntimeError("structured_unavailable")

    async def ainvoke(self, messages: object) -> _FallbackResponse:
        self.calls.append(messages)
        if not self._contents:
            return _FallbackResponse("")
        return _FallbackResponse(self._contents.pop(0))


class _BrokenStructuredThenFallbackModel(_FallbackModel):
    def with_structured_output(self, _schema: type[BaseModel]) -> _StructuredRunner:
        raise RuntimeError("structured_unavailable")


def test_tools_for_routine_wake_excludes_interactive_owner_update_tool() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    send_owner_update = object()
    turn_plan = object()
    server_time = object()
    gateway = object()
    runtime._tools = {
        "send_owner_update": send_owner_update,
        "turn_plan": turn_plan,
        "server_time": server_time,
        "tool_group_exec": gateway,
    }

    assert runtime.tools_for_turn_mode("interactive") == [
        send_owner_update,
        turn_plan,
        server_time,
        gateway,
    ]
    assert runtime.tools_for_turn_mode("routine_wake") == [server_time, gateway]
    assert runtime.tools_for_turn_mode("workflow_setup") == [
        send_owner_update,
        turn_plan,
        server_time,
        gateway,
    ]


def test_turn_plan_context_includes_completed_items_as_done() -> None:
    context = build_turn_plan_prompt_context(
        {
            "turn_plan": [
                {"id": "done", "content": "Already done", "status": "completed"},
                {
                    "id": "next",
                    "content": "Use gathered evidence",
                    "status": "in_progress",
                },
            ]
        }
    )

    assert "CURRENT_TURN_PLAN" in context
    assert "Use gathered evidence" in context
    assert "[x] done: Already done (completed)" in context


@lc_tool
def _runtime_structured_test_tool() -> str:
    """Tiny runtime structured model test helper."""

    return "ok"


def _tool_schema_chars(tools: list[Any]) -> int:
    total = 0
    for tool in tools:
        schema = convert_to_openai_tool(tool)
        total += len(
            json.dumps(
                schema,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return total


def test_tools_for_workflow_setup_uses_task_specific_profile() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._tools = {
        "send_owner_update": _runtime_structured_test_tool,
        "server_time": _runtime_structured_test_tool,
        "tool_group_list": _runtime_structured_test_tool,
        "tool_group_describe": _runtime_structured_test_tool,
        "tool_group_exec": _runtime_structured_test_tool,
        "intake_workflow_setup_begin": _runtime_structured_test_tool,
        "intake_workflow_setup_update": _runtime_structured_test_tool,
        "intake_workflow_setup_finalize_confirmation": _runtime_structured_test_tool,
        "telegram_business_status": _runtime_structured_test_tool,
        "business_knowledge_query": _runtime_structured_test_tool,
        "intake_workflow_upsert": _runtime_structured_test_tool,
        "browser_use_run": _runtime_structured_test_tool,
        "routine_create": _runtime_structured_test_tool,
        "tulpa_run_terminal": _runtime_structured_test_tool,
    }

    workflow_setup_tools = runtime.tools_for_turn_mode("workflow_setup")

    assert workflow_setup_tools == [
        _runtime_structured_test_tool,
        _runtime_structured_test_tool,
        _runtime_structured_test_tool,
        _runtime_structured_test_tool,
        _runtime_structured_test_tool,
    ]


def test_real_workflow_setup_profile_removes_large_irrelevant_schemas(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    interactive_tools = runtime.tools_for_turn_mode("interactive")
    workflow_tools = runtime.tools_for_turn_mode("workflow_setup")
    workflow_tool_ids = {
        tool_id
        for tool_id, tool in runtime._tools.items()
        if any(tool is selected for selected in workflow_tools)
    }

    assert workflow_tool_ids == {
        "send_owner_update",
        "server_time",
        "tool_group_list",
        "tool_group_describe",
        "tool_group_exec",
    }
    assert "intake_workflow_upsert" not in workflow_tool_ids
    assert "browser_use_run" not in workflow_tool_ids
    assert "routine_create" not in workflow_tool_ids
    assert "tulpa_run_terminal" not in workflow_tool_ids
    assert len(workflow_tools) == len(interactive_tools) - 1
    assert _tool_schema_chars(workflow_tools) < 2500


@pytest.mark.asyncio
async def test_tool_group_gateway_describes_and_executes_commands(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    group_list = await runtime._tools["tool_group_list"].ainvoke({})
    groups = {item["group"] for item in group_list["groups"]}
    assert {"web", "browser", "intake", "composio", "memory"}.issubset(groups)

    description = await runtime._tools["tool_group_describe"].ainvoke(
        {"group": "web", "command": "web_search"}
    )
    assert description["command"] == "web_search"
    assert description["call_pattern"]["tool"] == "tool_group_exec"

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {"group": "memory", "command": "server_time", "args_json": "{}"}
    )
    assert result["ok"] is True
    assert result["command"] == "server_time"
    assert "server_time_utc_iso" in result["result"]


def test_web_search_tool_schema_is_query_only_without_exa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "k")
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    schema = runtime._tools["web_search"].args_schema.model_json_schema()

    assert set(schema["properties"]) == {"query"}
    assert schema["required"] == ["query"]


def test_web_search_tool_schema_exposes_exa_filters_with_exa(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "k")
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    schema = runtime._tools["web_search"].args_schema.model_json_schema()

    assert set(schema["properties"]) == {"query", "search_type", "category"}
    assert schema["required"] == ["query"]


def test_tool_group_gateway_covers_registered_tools_exactly_once(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    core_bound = set(INTERACTIVE_NATIVE_TOOL_NAMES) | set(ROUTINE_WAKE_NATIVE_TOOL_NAMES)
    intentionally_core = core_bound | TOOL_GATEWAY_TOOL_NAMES
    grouped_by_tool: dict[str, list[str]] = {}
    for group, definition in TOOL_GROUP_DEFINITIONS.items():
        commands = definition.get("commands")
        assert isinstance(commands, set)
        for command in commands:
            grouped_by_tool.setdefault(str(command), []).append(group)

    registered = set(runtime._tools)
    missing = sorted(registered - set(grouped_by_tool) - intentionally_core)
    unknown_configured = sorted(set(grouped_by_tool) - registered)
    duplicated = {
        tool_name: groups
        for tool_name, groups in sorted(grouped_by_tool.items())
        if len(groups) != 1
    }

    assert missing == []
    assert unknown_configured == []
    assert duplicated == {}


@pytest.mark.asyncio
async def test_tool_group_exec_runs_hidden_customer_scoped_tools(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    captured: dict[str, Any] = {}

    class _Response:
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, Any]:
            return {"skills": [{"name": "customer-skill"}]}

    async def _request_with_backoff(*args: Any, **kwargs: Any) -> _Response:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Response()

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]
    runtime._active_customer_id = "cust_gateway"
    runtime._register_tools()

    interactive_tool_names = {
        str(getattr(tool, "name", "") or "").strip()
        for tool in runtime.tools_for_turn_mode("interactive")
    }
    assert "skill_list" not in interactive_tool_names
    assert "tool_group_exec" in interactive_tool_names

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {"group": "skills", "command": "skill_list", "args_json": {}}
    )

    assert result == {
        "group": "skills",
        "command": "skill_list",
        "ok": True,
        "result": [{"name": "customer-skill"}],
    }
    assert captured["args"][:2] == ("POST", "/internal/skills/list")
    assert captured["kwargs"]["json_body"]["customer_id"] == "cust_gateway"


@pytest.mark.asyncio
async def test_tool_group_gateway_repairs_common_intake_update_shape(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    captured: dict[str, Any] = {}

    class _Response:
        status_code = 200
        text = "{}"

        def json(self) -> dict[str, Any]:
            return {"session": {"ok": True}}

    async def _request_with_backoff(*args: Any, **kwargs: Any) -> _Response:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _Response()

    runtime._request_with_backoff = _request_with_backoff  # type: ignore[method-assign]
    runtime._active_customer_id = "cust_1"
    runtime._active_thread_id = "thread_1"
    runtime._register_tools()

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {
            "group": "intake",
            "command": "intake_workflow_setup_update",
            "args_json": {
                "name": "E2E Telegram Car Wash",
                "provider": "telegram",
                "channel": "telegram_business_dm",
            },
        }
    )

    assert result["ok"] is True
    json_body = captured["kwargs"]["json_body"]
    assert json_body["draft_patch"] == {
        "name": "E2E Telegram Car Wash",
        "provider": "telegram_bot_api",
        "channel": "telegram_business_dm",
    }

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {
            "group": "intake",
            "command": "intake_workflow_setup_update",
            "args_json": {
                "draft_upsert": {
                    "name": "E2E Telegram Car Wash",
                    "provider": "telegram_business",
                    "channel": "telegram_business_dm",
                }
            },
        }
    )

    assert result["ok"] is True
    assert captured["kwargs"]["json_body"]["draft_patch"] == {
        "name": "E2E Telegram Car Wash",
        "provider": "telegram_bot_api",
        "channel": "telegram_business_dm",
    }

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {
            "group": "intake",
            "command": "intake_workflow_setup_update",
            "args_json": {
                "name": "E2E Telegram Car Wash",
                "channel": "telegram_business_dm",
            },
        }
    )

    assert result["ok"] is True
    assert captured["kwargs"]["json_body"]["draft_patch"]["provider"] == "telegram_bot_api"

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {
            "group": "intake",
            "command": "intake_workflow_setup_update",
            "args_json": {
                "draft_patch": {
                    "draft": {
                        "name": "E2E Telegram Car Wash",
                        "provider": "telegram_business",
                        "channel": "telegram_business_dm",
                    }
                }
            },
        }
    )

    assert result["ok"] is True
    assert captured["kwargs"]["json_body"]["draft_patch"] == {
        "name": "E2E Telegram Car Wash",
        "provider": "telegram_bot_api",
        "channel": "telegram_business_dm",
    }


@pytest.mark.asyncio
async def test_tool_group_exec_returns_compact_repair_hint_for_missing_args(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {"group": "browser", "command": "browser_use_run", "args_json": {}}
    )

    assert result["error"] == "missing required args"
    assert result["missing_args"] == ["task"]
    hint = result["repair_hint"]
    assert hint["expected_args"]["args"]["task"]["required"] is True
    assert hint["example_call"] == {
        "tool": "tool_group_exec",
        "group": "browser",
        "command": "browser_use_run",
        "args_json": "JSON object with the expected args above",
    }
    assert "tool_group_exec directly" in hint["next_step"]
    assert "schema" not in hint


@pytest.mark.asyncio
async def test_tool_group_exec_returns_compact_repair_hint_for_bad_args_json(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    result = await runtime._tools["tool_group_exec"].ainvoke(
        {"group": "web", "command": "web_search", "args_json": "query=openai"}
    )

    assert result["error"] == "args_json must be a JSON object"
    assert result["received_type"] == "str"
    hint = result["repair_hint"]
    assert hint["expected_args"]["args"]["query"]["required"] is True
    assert hint["example_call"]["group"] == "web"
    assert hint["example_call"]["command"] == "web_search"
    assert "tool_group_describe" in hint["next_step"]


def test_registered_tools_have_searchable_descriptions(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="z-ai/glm-5.1",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._register_tools()

    missing = [
        name
        for name, tool in runtime._tools.items()
        if not str(getattr(tool, "description", "") or "").strip()
    ]
    assert missing == []

    required_terms = {
        "intake_workflow_setup_begin": ("workflow", "setup"),
        "intake_workflow_setup_update": ("workflow", "draft"),
        "intake_workflow_setup_preflight": ("workflow", "preflight"),
        "business_knowledge_query": ("business", "knowledge"),
        "uploaded_file_search": ("uploaded", "file"),
        "user_context_query": ("user", "context"),
        "composio_tool_search": ("composio", "tool"),
        "browser_use_run": ("browser",),
    }
    for name, terms in required_terms.items():
        description = str(getattr(runtime._tools[name], "description", "") or "").lower()
        assert all(term in description for term in terms), name


class _ProviderAwareStructuredRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, _messages: object, **kwargs: Any) -> object:
        self.calls.append({"messages": _messages, "kwargs": kwargs})
        return {"ok": True, "reason": "default_route"}


class _ProviderAwareStructuredModel:
    def __init__(self) -> None:
        self.runners: list[_ProviderAwareStructuredRunner] = []
        self.fallback_calls: list[dict[str, Any]] = []

    def with_structured_output(self, _schema: type[BaseModel]) -> _ProviderAwareStructuredRunner:
        runner = _ProviderAwareStructuredRunner()
        self.runners.append(runner)
        return runner

    async def ainvoke(self, messages: object, **kwargs: Any) -> _FallbackResponse:
        self.fallback_calls.append({"messages": messages, "kwargs": kwargs})
        return _FallbackResponse('{"ok": true, "reason": "fallback_route"}')


@pytest.mark.asyncio
async def test_invoke_structured_model_prefers_native_structured_output() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _StructuredModel(_Schema(ok=True, reason="native"))

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "native"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_uses_strict_json_fallback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('{"ok": true, "reason": "fallback"}')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fallback"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_accepts_fenced_json_in_fallback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('```json\n{"ok": true, "reason": "fenced"}\n```')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fenced"
    assert error is None


@pytest.mark.asyncio
async def test_invoke_structured_model_rejects_wrapped_non_json_text() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _BrokenStructuredThenFallbackModel('prefix {"ok": true, "reason": "x"} suffix')

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert parsed is None
    assert isinstance(error, str)
    assert "ValidationError" in error


@pytest.mark.asyncio
async def test_invoke_structured_model_repairs_invalid_json_fallback_once() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    model = _SequenceFallbackModel(
        [
            '{"ok": true, "reason": ',
            '{"ok": true, "reason": "repaired"}',
        ]
    )

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "repaired"
    assert error is None
    assert len(model.calls) == 2


@pytest.mark.asyncio
async def test_invoke_structured_model_skips_deepseek_v4_pro_native_structured_output() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.openrouter_base_url = "https://openrouter.ai/api/v1"
    runtime.model_name = "deepseek/deepseek-v4-pro"
    runtime._reasoning_effort = "medium"
    runtime._prompt_caching_enabled = False
    runtime._prompt_cache_ttl_1h = False
    model = _ProviderAwareStructuredModel()

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
        model_name="deepseek/deepseek-v4-pro",
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fallback_route"
    assert error is None
    assert model.runners == []
    assert len(model.fallback_calls) == 1
    assert model.fallback_calls[0]["kwargs"] == {}


@pytest.mark.asyncio
async def test_invoke_structured_model_omits_legacy_deepseek_disable_payload_for_openrouter_adapter() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.openrouter_base_url = "https://openrouter.ai/api/v1"
    runtime.model_name = "deepseek/deepseek-v4-pro"
    runtime._reasoning_effort = None
    runtime._prompt_caching_enabled = False
    runtime._prompt_cache_ttl_1h = False
    model = _ProviderAwareStructuredModel()

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
        model_name="deepseek/deepseek-v4-pro",
    )

    assert isinstance(parsed, _Schema)
    assert parsed.ok is True
    assert parsed.reason == "fallback_route"
    assert error is None
    assert model.runners == []
    assert len(model.fallback_calls) == 1
    assert model.fallback_calls[0]["kwargs"] == {}


@pytest.mark.asyncio
async def test_invoke_structured_model_records_single_llm_call_trace_on_success(tmp_path: Path) -> None:
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
    )
    runtime._llm_call_trace_path = tmp_path / "llm_call_traces.jsonl"
    model = _StructuredModel({"ok": True, "reason": "native"})

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
        call_context={
            "call_site": "intake_workflow_decision",
            "trace_id": "intake_trace_test",
            "thread_id": "intake_decision_iwf_conv",
            "customer_id": "telegram_123",
            "turn_mode": "routine_wake",
            "prompt_mode": "structured_intake",
        },
    )

    assert isinstance(parsed, _Schema)
    assert error is None
    records = [
        json.loads(line)
        for line in runtime._llm_call_trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["call_site"] == "intake_workflow_decision"
    assert records[0]["trace_id"] == "intake_trace_test"
    assert "native" in records[0]["response_text"]
    assert records[0]["response_content"]["reason"] == "native"


@pytest.mark.asyncio
async def test_invoke_structured_model_logs_preprovider_behavior_events(tmp_path: Path) -> None:
    behavior_log = tmp_path / "agent_behavior.jsonl"
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key="k",
        model_name="google/gemini-3-flash-preview",
        checkpoint_db_path=str(tmp_path / "checkpoint.sqlite"),
        behavior_log_enabled=True,
        behavior_log_path=str(behavior_log),
    )
    model = _StructuredModel({"ok": True, "reason": "native"})

    parsed, error = await runtime._invoke_structured_model(
        model=model,
        messages=[],
        schema=_Schema,
        call_context={
            "call_site": "intake_workflow_decision",
            "trace_id": "intake_trace_test",
            "thread_id": "intake_decision_iwf_conv",
            "customer_id": "telegram_123",
            "turn_mode": "routine_wake",
            "prompt_mode": "structured_intake",
        },
    )

    assert isinstance(parsed, _Schema)
    assert error is None
    events = [
        json.loads(line)
        for line in behavior_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_names = [str(item.get("event", "")) for item in events]
    assert "llm.invoke.start" in event_names
    assert "llm.invoke.runner_ready" in event_names
    assert "llm.invoke.await_provider" in event_names
    assert "llm.invoke.finish" in event_names
    assert event_names.index("llm.invoke.start") < event_names.index("llm.invoke.await_provider")
    assert event_names.index("llm.invoke.await_provider") < event_names.index("llm.invoke.finish")


@pytest.mark.asyncio
async def test_decide_intake_workflow_uses_stronger_policy_prompt() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    model = _StructuredModel(
        {
            "matches_workflow": False,
            "confidence": 0.2,
            "conversation_summary": "unrelated chat",
            "extracted_fields": {},
            "missing_fields": [],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": False,
            "booking_action": "ignore",
            "save_payload": {},
            "reason": "not a booking",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"
    tracer = _RecordingTracer()
    runtime._langfuse_tracer = tracer

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": "interior, exterior, or both"},
            "sink_type": "local_csv",
        },
        conversation={
            "summary": {"conversation_id": "conv_1"},
            "recent_messages": [{"sender_role": "customer", "text": "thanks"}],
            "unanswered_customer_messages": [{"sender_role": "customer", "text": "thanks"}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "reply_execution",
                "error": "Invalid request data provided",
                "prior_decision": {"reply_action": "send_reply"},
            }
        ],
    )

    assert decision["ok"] is True
    assert model.runner is not None
    messages = model.runner.messages
    assert isinstance(messages, list)
    system_text = str(messages[0].content)
    assert isinstance(messages[0].content, list)
    assert messages[0].content[0]["cache_control"] == {"type": "ephemeral"}
    assert isinstance(messages[1].content, str)
    assert "Default mode is not an intent filter" in system_text
    assert "conversation.unanswered_customer_messages as the active customer turn" in system_text
    assert "workflow.intent_match_required is true" in system_text
    assert "If customer messages conflict, prefer the latest customer-provided value" in system_text
    assert "Ask at most one compact question at a time" in system_text
    assert "When ready_to_save=true, save_payload must contain the merged final field set" in system_text
    assert "Booking-state fast path" in system_text
    assert "do not require workflow.knowledge_answer or business_knowledge_query" in system_text
    assert "Never ask the customer to confirm a booking or change that you are saving now" in system_text
    assert "needs_business_knowledge=true" in system_text
    assert "If workflow.knowledge_file_ids is empty, never set needs_business_knowledge=true" in system_text
    assert "business_knowledge_query to one concise natural language query" in system_text
    assert tracer.calls[0]["name"] == "opentulpa.intake.turn"
    assert tracer.calls[0]["input"] == {
        "workflow_id": "iwf_123",
        "conversation_id": "conv_1",
        "incoming_id": "latest",
    }
    assert tracer.calls[0]["metadata"]["incoming_id"] == "latest"


@pytest.mark.asyncio
async def test_decide_intake_workflow_prefers_tool_runtime_first_for_composio_sinks() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    runtime._graph = object()
    runtime._wake_execution_model_with_tools = object()
    model = _StructuredModel(
        {
            "matches_workflow": True,
            "confidence": 0.95,
            "conversation_summary": "Customer wants a car wash tomorrow at 4pm.",
            "extracted_fields": {"day": "tomorrow"},
            "missing_fields": ["time"],
            "reply_action": "send_reply",
            "reply_text": "What time works best?",
            "ready_to_save": False,
            "booking_action": "create_new_booking",
            "save_payload": {},
            "reason": "Need time before save.",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"
    captured: dict[str, Any] = {"called": False}

    async def _fake_ainvoke_text(**kwargs: Any) -> str:
        captured.update(kwargs)
        captured["called"] = True
        return (
            '{"matches_workflow": true, "confidence": 0.95, "conversation_summary": '
            '"Customer wants a car wash tomorrow at 4pm.", "extracted_fields": {"day": "tomorrow"}, '
            '"missing_fields": ["time"], "reply_action": "send_reply", "reply_text": "What time works best?", '
            '"ready_to_save": false, "booking_action": "create_new_booking", "save_payload": {}, '
            '"sink_arguments": {}, "reason": "Need time before save."}'
        )

    runtime.ainvoke_text = _fake_ainvoke_text

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": "interior, exterior, or both"},
            "sink_type": "google_sheets_composio",
            "sink_config": {"tool_slug": "GOOGLESHEETS_ADD_ROW", "field_mapping": {"day": "Date"}},
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
            },
            "recent_messages": [{"sender_role": "customer", "text": "Need a wash tomorrow."}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "reply_execution",
                "error": "Invalid request data provided",
                "prior_decision": {"reply_action": "send_reply"},
            }
        ],
    )

    assert decision["ok"] is True
    assert captured["called"] is True
    assert captured["thread_id"] == "wake_intake_iwf_123_conv_1_msg_1"
    assert captured["turn_mode"] == "routine_wake"
    assert captured["include_pending_context"] is False
    assert captured["prompt_mode_override"] == "literal_chat"
    assert model.runner is None


@pytest.mark.asyncio
async def test_decide_intake_workflow_does_not_use_tool_runtime_for_bound_knowledge_files() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    runtime._graph = object()
    runtime._wake_execution_model_with_tools = object()
    model = _StructuredModel(
        {
            "matches_workflow": False,
            "confidence": 0.2,
            "conversation_summary": "Fallback structured model should not run.",
            "extracted_fields": {},
            "missing_fields": [],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": False,
            "booking_action": "ignore",
            "save_payload": {},
            "needs_business_knowledge": True,
            "business_knowledge_query": "2 phase wash price",
            "reason": "Needs source-backed price.",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"

    async def _fake_ainvoke_text(**kwargs: Any) -> str:
        raise AssertionError("bound knowledge alone should not force tool runtime")

    runtime.ainvoke_text = _fake_ainvoke_text

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_knowledge",
            "name": "Autospa Intake",
            "intent_description": "Handle autospa bookings.",
            "required_fields": ["wash_type", "time"],
            "field_guidance": {},
            "knowledge_file_ids": ["file_prepared"],
            "sink_type": "local_csv",
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
            },
            "recent_messages": [{"sender_role": "customer", "text": "How much is 2 phase wash?"}],
        },
        active_booking=None,
        recent_completed_booking=None,
    )

    assert decision["ok"] is True
    assert decision["needs_business_knowledge"] is True
    assert decision["business_knowledge_query"] == "2 phase wash price"
    assert model.runner is not None


@pytest.mark.asyncio
async def test_decide_intake_workflow_escalates_to_tool_runtime_after_structured_failure_with_feedback() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    runtime._graph = object()
    runtime._wake_execution_model_with_tools = object()
    runtime._model = _BrokenStructuredThenFallbackModel("not json")
    runtime._wake_execution_model = runtime._model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"
    captured: dict[str, Any] = {}

    async def _fake_ainvoke_text(**kwargs: Any) -> str:
        captured.update(kwargs)
        return (
            '{"matches_workflow": true, "confidence": 0.95, "conversation_summary": '
            '"Customer wants a car wash tomorrow at 4pm.", "extracted_fields": {"day": "tomorrow"}, '
            '"missing_fields": ["time"], "reply_action": "send_reply", "reply_text": "What time works best?", '
            '"ready_to_save": false, "booking_action": "create_new_booking", "save_payload": {}, '
            '"sink_arguments": {}, '
            '"reason": "Need time before save."}'
        )

    runtime.ainvoke_text = _fake_ainvoke_text

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": "interior, exterior, or both"},
            "sink_type": "google_sheets_composio",
            "sink_config": {"tool_slug": "GOOGLESHEETS_ADD_ROW", "field_mapping": {"day": "Date"}},
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
            },
            "recent_messages": [{"sender_role": "customer", "text": "Need a wash tomorrow."}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "reply_execution",
                "error": "Invalid request data provided",
                "prior_decision": {"reply_action": "send_reply"},
            }
        ],
    )

    assert decision["ok"] is True
    assert captured["thread_id"] == "wake_intake_iwf_123_conv_1_msg_1"
    assert captured["turn_mode"] == "routine_wake"
    assert captured["include_pending_context"] is False
    assert captured["prompt_mode_override"] == "literal_chat"
    prompt = str(captured["text"])
    assert "Operate like a real OpenTulpa background execution turn and use tools when needed." in prompt
    assert "composio_tool_search" in prompt
    assert "execution_feedback=" in prompt
    assert "Invalid request data provided" in prompt
    assert "sink_arguments" in prompt
    assert "A write sink is not an availability source by default." in prompt
    assert "do not check availability" in prompt


@pytest.mark.asyncio
async def test_decide_intake_workflow_compacts_prompt_payload() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    long_text = "x" * 1000
    model = _StructuredModel(
        {
            "matches_workflow": False,
            "confidence": 0.2,
            "conversation_summary": "unrelated chat",
            "extracted_fields": {},
            "missing_fields": [],
            "reply_action": "none",
            "reply_text": "",
            "ready_to_save": False,
            "booking_action": "ignore",
            "save_payload": {},
            "reason": "not a booking",
        }
    )
    runtime._model = model
    runtime._wake_execution_model = model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"

    recent_messages = [
        {"id": f"m{i}", "created_time": f"2026-04-08T08:0{i}:00+00:00", "sender_role": "customer", "text": long_text}
        for i in range(8)
    ]
    await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": long_text,
            "required_fields": ["day", "time", "car_type", "wash_type"],
            "field_guidance": {"wash_type": long_text},
            "business_facts": {
                "prices": {"basic_wash": "1000 RUB"},
                "long_note": long_text,
            },
            "workflow_skill": "Owner-Provided Business Facts\nbasic_wash costs 1000 RUB\n" + long_text,
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "tool_slug": "GOOGLESHEETS_ADD_ROW",
                "field_mapping": {"day": "Date"},
                "static_arguments": {"spreadsheet_id": long_text},
            },
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
                "latest_inbound_message_text_preview": long_text,
            },
            "recent_messages": recent_messages,
        },
        active_booking={"booking_id": "bkg_1", "status": "active", "extracted_fields": {"notes": long_text}},
        recent_completed_booking=None,
        execution_feedback=[
            {"phase": "sink_execution", "error": long_text, "prior_decision": {"reply_action": "send_reply"}},
            {"phase": "reply_execution", "error": long_text, "prior_decision": {"reply_action": "send_reply"}},
            {"phase": "ignored", "error": long_text, "prior_decision": {"reply_action": "send_reply"}},
        ],
    )

    assert model.runner is not None
    messages = model.runner.messages
    assert isinstance(messages, list)
    human_text = str(messages[1].content)
    assert human_text.count('"sender_role"') == 6
    assert ('"text": "' + ("x" * 301)) not in human_text
    assert human_text.count('"phase"') == 2
    assert '"business_facts": {"prices":' in human_text
    assert "1000 RUB" in human_text
    assert "Owner-Provided Business Facts" in human_text
    assert '"static_argument_keys": ["spreadsheet_id"]' in human_text
    assert '"static_arguments": {"spreadsheet_id": "' in human_text


@pytest.mark.asyncio
async def test_decide_intake_workflow_returns_sink_arguments_from_tool_runtime() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime.model_name = "google/gemini-3-flash-preview"
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False
    runtime._graph = object()
    runtime._wake_execution_model_with_tools = object()
    runtime._model = _BrokenStructuredThenFallbackModel("not json")
    runtime._wake_execution_model = runtime._model
    runtime._wake_execution_model_name = "google/gemini-3-flash-preview"

    async def _fake_ainvoke_text(**_: Any) -> str:
        return (
            '{"matches_workflow": true, "confidence": 0.97, "conversation_summary": '
            '"Recovered by finding the correct sheet.", "extracted_fields": {"day": "tomorrow"}, '
            '"missing_fields": [], "reply_action": "send_reply", "reply_text": "Booked.", '
            '"ready_to_save": true, "booking_action": "update_active", "save_payload": {"day": "tomorrow"}, '
            '"sink_arguments": {"sheetName": "Лист1"}, "reason": "Use the discovered tab."}'
        )

    runtime.ainvoke_text = _fake_ainvoke_text

    decision = await runtime.decide_intake_workflow(
        customer_id="telegram_123",
        workflow={
            "workflow_id": "iwf_123",
            "name": "Car Wash Intake",
            "intent_description": "Handle booking requests from Instagram DMs.",
            "required_fields": ["day"],
            "field_guidance": {},
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "tool_slug": "GOOGLESHEETS_ADD_ROW",
                "field_mapping": {"day": "Date"},
                "static_arguments": {"spreadsheet_id": "sheet_123"},
            },
        },
        conversation={
            "summary": {
                "conversation_id": "conv_1",
                "latest_inbound_message_id": "msg_1",
            },
            "recent_messages": [{"sender_role": "customer", "text": "Tomorrow works."}],
        },
        active_booking=None,
        recent_completed_booking=None,
        execution_feedback=[
            {
                "phase": "sink_execution",
                "error": "Following fields are missing: {'sheetName'}",
                "prior_decision": {"ready_to_save": True, "sink_arguments": {}},
            }
        ],
    )

    assert decision["ok"] is True
    assert decision["sink_arguments"] == {"sheetName": "Лист1"}


def test_prompt_cache_profile_uses_openrouter_standard_modes() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._prompt_caching_enabled = True
    runtime._prompt_cache_ttl_1h = False

    anth = runtime.prompt_cache_profile(model_name="anthropic/claude-sonnet-4.6")
    gemini = runtime.prompt_cache_profile(model_name="google/gemini-3-flash-preview")
    auto = runtime.prompt_cache_profile(model_name="openai/gpt-4.1")
    zai = runtime.prompt_cache_profile(model_name="z-ai/glm-5.2")
    qwen = runtime.prompt_cache_profile(model_name="qwen/qwen3.7-max")
    minimax = runtime.prompt_cache_profile(model_name="minimax/minimax-m3")

    assert anth["strategy"] == "top_level"
    assert gemini["strategy"] == "breakpoint"
    assert auto["strategy"] == "automatic"
    assert zai["strategy"] == "automatic"
    assert qwen["strategy"] == "implicit_stable_prefix"
    assert minimax["strategy"] == "implicit_stable_prefix"
