from __future__ import annotations

from typing import Any

import pytest

from opentulpa.agent.tools.internal_http import InternalToolHTTPClient


class _Response:
    def __init__(self, *, status_code: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _Runtime:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    async def _request_with_backoff(self, method: str, path: str, **kwargs: Any) -> _Response:
        self.calls.append((method, path, kwargs))
        return self.response


@pytest.mark.asyncio
async def test_internal_tool_http_success_projects_item() -> None:
    runtime = _Runtime(_Response(status_code=200, payload={"items": [{"id": "one"}]}))
    result = await InternalToolHTTPClient(runtime).request_item(
        "tool_list",
        "GET",
        "/internal/items",
        "items",
        default=[],
        params={"limit": 1},
    )

    assert result == [{"id": "one"}]
    assert runtime.calls == [("GET", "/internal/items", {"params": {"limit": 1}, "timeout": 20.0})]


@pytest.mark.asyncio
async def test_internal_tool_http_non_200_marks_retryable() -> None:
    runtime = _Runtime(_Response(status_code=503, text="try later"))
    result = await InternalToolHTTPClient(runtime).request("tool_run", "POST", "/internal/run")

    assert result["error"] == "tool_run failed: try later"
    assert result["status_code"] == 503
    assert result["retryable"] is True


@pytest.mark.asyncio
async def test_internal_tool_http_invalid_json_is_error() -> None:
    runtime = _Runtime(_Response(status_code=200, payload=ValueError("bad"), text="not-json"))
    result = await InternalToolHTTPClient(runtime).request("tool_run", "GET", "/internal/run")

    assert result["error"] == "tool_run failed: invalid JSON response"
    assert result["response"] == "not-json"
