from __future__ import annotations

import pytest

from opentulpa.agent.tools.browser_tools import _normalize_allowed_domains
from opentulpa.agent.tools_registry import register_runtime_tools


class _DummyRuntime:
    def __init__(self, manager: object | None = None) -> None:
        self._active_customer_id = "u_1"
        self._manager = manager

    async def _request_with_backoff(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise RuntimeError("unexpected internal API call")

    def get_browser_use_local_manager(self) -> object | None:
        return self._manager


class _DummyBrowserManager:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self.last_screenshot: dict[str, object] | None = None

    async def start_task(
        self,
        *,
        task: str,
        max_steps: int,
        llm: str,
        allowed_domains: list[str] | None = None,
        start_url: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        task_id = "task_123"
        sid = session_id or "bses_1"
        payload = {
            "id": task_id,
            "sessionId": sid,
            "status": "finished",
            "isSuccess": True,
            "startedAt": "2026-01-01T00:00:00+00:00",
            "finishedAt": "2026-01-01T00:00:01+00:00",
            "task": task,
            "llm": llm,
            "output": "done",
            "outputFiles": [],
            "steps": [
                {
                    "number": 1,
                    "url": start_url or "https://example.com",
                    "nextGoal": "",
                    "actions": ["search(query)"],
                    "screenshotUrl": None,
                }
            ],
        }
        self.tasks[task_id] = payload
        return payload

    async def get_task(self, task_id: str) -> dict | None:
        return self.tasks.get(task_id)

    async def control_task(self, *, task_id: str, action: str) -> dict:
        payload = self.tasks.get(task_id)
        if payload is None:
            return {"error": "task not found"}
        payload["status"] = "stopped" if action.startswith("stop") else "running"
        return payload

    async def capture_screenshot(self, *, task_id: str, full_page: bool = True) -> dict:
        self.last_screenshot = {"task_id": task_id, "full_page": full_page}
        return {
            "ok": True,
            "task_id": task_id,
            "session_id": "bses_1",
            "path": f"tulpa_stuff/screenshots/browser_use/{task_id}.png",
            "file_name": f"{task_id}.png",
        }

    async def list_sessions(self) -> list[dict]:
        return [
            {
                "session_id": "bses_1",
                "reusable": True,
                "active_task_ids": [],
                "latest_task_id": "task_123",
                "latest_status": "finished",
                "last_url": "https://example.com",
                "last_used_seconds": 12,
            }
        ]


def test_normalize_allowed_domains_filters_invalid_values() -> None:
    values = _normalize_allowed_domains(
        [
            "https://example.com/path",
            "docs.python.org",
            "localhost",
            "bad domain",
            "https://example.com/other",
            "",
        ]
    )
    assert values == ["example.com", "docs.python.org"]


@pytest.mark.asyncio
async def test_browser_use_run_uses_local_manager() -> None:
    tools = register_runtime_tools(_DummyRuntime(_DummyBrowserManager()))

    result = await tools["browser_use_run"].ainvoke(
        {"task": "open docs", "start_url": "https://example.com"}
    )
    assert result.get("task_id") == "task_123"
    assert result.get("status") == "finished"
    assert result.get("output") == "done"


@pytest.mark.asyncio
async def test_browser_use_session_list_returns_sessions() -> None:
    tools = register_runtime_tools(_DummyRuntime(_DummyBrowserManager()))

    result = await tools["browser_use_session_list"].ainvoke({})
    assert result["sessions"][0]["session_id"] == "bses_1"
    assert result["sessions"][0]["reusable"] is True


@pytest.mark.asyncio
async def test_browser_use_run_errors_when_manager_missing() -> None:
    tools = register_runtime_tools(_DummyRuntime(None))

    result = await tools["browser_use_run"].ainvoke({"task": "open docs"})
    assert "error" in result
    assert "manager is None" in str(result["error"])


@pytest.mark.asyncio
async def test_browser_use_task_get_not_found() -> None:
    tools = register_runtime_tools(_DummyRuntime(_DummyBrowserManager()))
    result = await tools["browser_use_task_get"].ainvoke({"task_id": "task_missing"})
    assert "error" in result
    assert "task not found" in str(result["error"])


@pytest.mark.asyncio
async def test_browser_use_task_control_validates_action() -> None:
    tools = register_runtime_tools(_DummyRuntime(_DummyBrowserManager()))

    result = await tools["browser_use_task_control"].ainvoke(
        {"task_id": "task_123", "action": "explode"}
    )
    assert "error" in result
    assert "invalid action" in str(result["error"])


@pytest.mark.asyncio
async def test_browser_use_task_screenshot_returns_local_path() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager))

    result = await tools["browser_use_task_screenshot"].ainvoke(
        {"task_id": "task_123", "full_page": False}
    )
    assert result.get("path") == "tulpa_stuff/screenshots/browser_use/task_123.png"
    assert manager.last_screenshot == {"task_id": "task_123", "full_page": False}
