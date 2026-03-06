from __future__ import annotations

from typing import Any

import pytest

from opentulpa.agent.tools_registry import register_runtime_tools


class _Response:
    def __init__(self, status_code: int, payload: dict[str, Any] | list[Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else str(payload)
        self.content = b"" if payload is None else b"x"

    def json(self) -> dict[str, Any] | list[Any]:
        return self._payload if self._payload is not None else {}


class _DummyRuntime:
    def __init__(self, responses: list[_Response], *, customer_id: str = "telegram_123") -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._active_customer_id = customer_id

    async def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> _Response:
        self.calls.append((method, path, kwargs))
        if not self._responses:
            raise RuntimeError("unexpected internal API call")
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_routine_list_passes_customer_scope() -> None:
    runtime = _DummyRuntime([_Response(200, {"routines": [{"id": "rtn_abc"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["routine_list"].ainvoke({})
    assert result == [{"id": "rtn_abc"}]
    assert runtime.calls[0][0] == "GET"
    assert runtime.calls[0][1] == "/internal/scheduler/routines"
    assert runtime.calls[0][2].get("params") == {"customer_id": "telegram_123"}


@pytest.mark.asyncio
async def test_routine_delete_verifies_removed() -> None:
    runtime = _DummyRuntime(
        [
            _Response(200, {"ok": True}),
            _Response(200, {"routines": []}),
        ]
    )
    tools = register_runtime_tools(runtime)

    result = await tools["routine_delete"].ainvoke(
        {"routine_id": "rtn_deadbeef"}
    )
    assert result["ok"] is True
    assert result["verified_removed"] is True


@pytest.mark.asyncio
async def test_automation_delete_calls_delete_with_assets() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "deleted_routines": [{"id": "rtn_1"}]})])
    tools = register_runtime_tools(runtime)

    result = await tools["automation_delete"].ainvoke(
        {
            "routine_id": "rtn_1",
            "delete_files": True,
            "cleanup_paths": ["tulpa_stuff/scripts/weather.py"],
        }
    )
    assert result["ok"] is True
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/scheduler/routine/delete_with_assets"
    sent = runtime.calls[0][2]["json_body"]
    assert sent["routine_id"] == "rtn_1"
    assert sent["delete_files"] is True
    assert sent["cleanup_paths"] == ["tulpa_stuff/scripts/weather.py"]


@pytest.mark.asyncio
async def test_lessons_learnt_get_action() -> None:
    runtime = _DummyRuntime([_Response(200, {"customer_id": "telegram_123", "lessons_learnt": "foo"})])
    tools = register_runtime_tools(runtime)

    result = await tools["lessons_learnt"].ainvoke(
        {"action": "get"}
    )
    assert result["lessons_learnt"] == "foo"
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/lessons_learnt/get"


@pytest.mark.asyncio
async def test_lessons_learnt_append_action() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "lessons_learnt": "old\nnew"})])
    tools = register_runtime_tools(runtime)

    result = await tools["lessons_learnt"].ainvoke(
        {"action": "append", "lesson": "new"}
    )
    assert result["ok"] is True
    assert runtime.calls[0][0] == "POST"
    assert runtime.calls[0][1] == "/internal/lessons_learnt/append"
    sent = runtime.calls[0][2]["json_body"]
    assert sent["lesson"] == "new"
    assert sent["max_chars"] == 20000


@pytest.mark.asyncio
async def test_lessons_learnt_set_action() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "lessons_learnt": "rewritten"})])
    tools = register_runtime_tools(runtime)

    result = await tools["lessons_learnt"].ainvoke(
        {"action": "set", "lesson": "rewritten"}
    )
    assert result["ok"] is True
    assert runtime.calls[0][1] == "/internal/lessons_learnt/set"
    sent = runtime.calls[0][2]["json_body"]
    assert sent["lessons_learnt"] == "rewritten"


@pytest.mark.asyncio
async def test_lessons_learnt_clear_action() -> None:
    runtime = _DummyRuntime([_Response(200, {"ok": True, "cleared": True})])
    tools = register_runtime_tools(runtime)

    result = await tools["lessons_learnt"].ainvoke(
        {"action": "clear"}
    )
    assert result["ok"] is True
    assert runtime.calls[0][1] == "/internal/lessons_learnt/clear"


@pytest.mark.asyncio
async def test_lessons_learnt_rejects_invalid_action() -> None:
    runtime = _DummyRuntime([])
    tools = register_runtime_tools(runtime)

    result = await tools["lessons_learnt"].ainvoke(
        {"action": "destroy"}
    )
    assert "action must be one of get|append|set|clear" in str(result.get("error", ""))
    assert runtime.calls == []


@pytest.mark.asyncio
async def test_customer_scoped_tool_fails_closed_without_customer_context() -> None:
    runtime = _DummyRuntime([_Response(200, {"routines": []})], customer_id="")
    tools = register_runtime_tools(runtime)
    with pytest.raises(RuntimeError, match="customer_id is missing"):
        await tools["routine_list"].ainvoke({})
