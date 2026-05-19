from __future__ import annotations

import pytest

from opentulpa.agent.tools.browser_tools import _build_browser_use_task, _normalize_allowed_domains
from opentulpa.agent.tools_registry import register_runtime_tools
from opentulpa.integrations.browser_use_local import BrowserUseLocalManager


class _DummyRuntime:
    def __init__(self, manager: object | None = None, turn_mode: str = "interactive") -> None:
        self._active_customer_id = "u_1"
        self._manager = manager
        self._turn_mode = turn_mode

    async def _request_with_backoff(self, *args, **kwargs):  # pragma: no cover - not used in tests
        raise RuntimeError("unexpected internal API call")

    def get_browser_use_local_manager(self) -> object | None:
        return self._manager

    def get_active_turn_mode(self) -> str:
        return self._turn_mode


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
        customer_id: str | None = None,
        allow_owner_input: bool = True,
    ) -> dict:
        self.last_allow_owner_input = allow_owner_input
        self.last_customer_id = customer_id
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
            "imageCandidates": [
                {
                    "url": "https://images.example.com/chipmunk.jpg",
                    "source": "img_src",
                    "alt": "chipmunk on a rock",
                    "width": 800,
                    "height": 533,
                    "natural_width": 1600,
                    "natural_height": 1066,
                    "page_url": start_url or "https://example.com",
                }
            ],
            "networkImageResources": [
                {
                    "url": "https://cdn.example.com/chipmunk.webp",
                    "source": "network_resource",
                    "initiator_type": "img",
                    "transfer_size": 12345,
                    "decoded_body_size": 45678,
                    "page_url": start_url or "https://example.com",
                }
            ],
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

    async def get_task(self, task_id: str, *, customer_id: str | None = None) -> dict | None:
        self.last_get_customer_id = customer_id
        return self.tasks.get(task_id)

    async def control_task(
        self,
        *,
        task_id: str,
        action: str,
        customer_id: str | None = None,
    ) -> dict:
        self.last_control_customer_id = customer_id
        payload = self.tasks.get(task_id)
        if payload is None:
            return {"error": "task not found"}
        payload["status"] = "stopped" if action.startswith("stop") else "running"
        return payload

    async def capture_screenshot(
        self,
        *,
        task_id: str,
        full_page: bool = True,
        customer_id: str | None = None,
    ) -> dict:
        self.last_screenshot = {
            "task_id": task_id,
            "full_page": full_page,
            "customer_id": customer_id,
        }
        return {
            "ok": True,
            "task_id": task_id,
            "session_id": "bses_1",
            "path": f"tulpa_stuff/screenshots/browser_use/{task_id}.png",
            "file_name": f"{task_id}.png",
        }

    async def submit_owner_input(
        self,
        *,
        task_id: str,
        owner_input: str,
        customer_id: str | None = None,
    ) -> dict:
        self.last_submit_customer_id = customer_id
        payload = self.tasks.get(task_id)
        if payload is None:
            return {"error": "task not found"}
        if payload.get("status") != "waiting_for_owner":
            return {"error": "not waiting"}
        payload["status"] = "running"
        payload["output"] = f"owner input submitted: {owner_input}"
        payload["ownerInputPrompt"] = None
        payload["ownerInputType"] = None
        payload["ownerInputRequestedAt"] = None
        return payload

    async def list_sessions(self, *, customer_id: str | None = None) -> list[dict]:
        self.last_list_customer_id = customer_id
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


def test_browser_use_tool_descriptions_include_login_session_and_secret_boundaries() -> None:
    tools = register_runtime_tools(_DummyRuntime(_DummyBrowserManager()))

    session_description = str(getattr(tools["browser_use_session_list"], "description", ""))
    run_description = str(getattr(tools["browser_use_run"], "description", ""))
    normalized_run_description = " ".join(run_description.split())

    assert "persisted\nbrowser profile state" in session_description
    assert "Browser Use-backed browser session" in normalized_run_description
    assert "OpenTulpa-captured page evidence" in normalized_run_description
    assert "image_candidates" in normalized_run_description
    assert "Do not ask the owner to paste credentials into durable memory" in normalized_run_description


def test_browser_use_image_resource_normalizer_bounds_and_deduplicates() -> None:
    items = BrowserUseLocalManager._normalize_image_resource_items(
        [
            {"url": "data:image/png;base64,abc", "source": "img_src"},
            {"url": "https://images.example.com/a.jpg#fragment", "source": "img_src"},
            {"url": "https://images.example.com/a.jpg", "source": "network_resource"},
            {"url": "https://images.example.com/b.jpg", "source": "img_src", "width": 640.4},
        ],
        page_url="https://example.com/search",
        max_items=2,
    )

    assert items == [
        {
            "url": "https://images.example.com/a.jpg#fragment",
            "source": "img_src",
            "page_url": "https://example.com/search",
        },
        {
            "url": "https://images.example.com/b.jpg",
            "source": "img_src",
            "page_url": "https://example.com/search",
            "width": 640,
        },
    ]


def test_build_browser_use_task_adds_operator_instruction() -> None:
    task = _build_browser_use_task("Find the source page")

    assert task.startswith("Use the browser like a careful human operator.")
    assert "Prefer visible page evidence over guesses." in task
    assert "prefer returned image_candidates or" in task
    assert "Do not keep browsing just to be exhaustive." in task
    assert task.endswith("Task:\nFind the source page")


@pytest.mark.asyncio
async def test_browser_use_run_uses_local_manager() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager))

    result = await tools["browser_use_run"].ainvoke(
        {"task": "open docs", "start_url": "https://example.com"}
    )
    assert result.get("task_id") == "task_123"
    assert result.get("status") == "finished"
    assert result.get("output") == "done"
    assert result["image_candidates"] == [
        {
            "url": "https://images.example.com/chipmunk.jpg",
            "source": "img_src",
            "alt": "chipmunk on a rock",
            "page_url": "https://example.com",
            "width": 800,
            "height": 533,
            "natural_width": 1600,
            "natural_height": 1066,
        }
    ]
    assert result["network_image_resources"][0]["url"] == "https://cdn.example.com/chipmunk.webp"
    assert result["network_image_resources"][0]["initiator_type"] == "img"
    assert manager.last_customer_id == "u_1"
    assert manager.tasks["task_123"]["task"].startswith(
        "Use the browser like a careful human operator."
    )
    assert manager.tasks["task_123"]["task"].endswith("Task:\nopen docs")


@pytest.mark.asyncio
async def test_browser_use_session_list_returns_sessions() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager))

    result = await tools["browser_use_session_list"].ainvoke({})
    assert result["sessions"][0]["session_id"] == "bses_1"
    assert result["sessions"][0]["reusable"] is True
    assert manager.last_list_customer_id == "u_1"


@pytest.mark.asyncio
async def test_browser_use_run_errors_when_manager_missing() -> None:
    tools = register_runtime_tools(_DummyRuntime(None))

    result = await tools["browser_use_run"].ainvoke({"task": "open docs"})
    assert "error" in result
    assert "manager is None" in str(result["error"])


@pytest.mark.asyncio
async def test_browser_use_task_get_not_found() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager))
    result = await tools["browser_use_task_get"].ainvoke({"task_id": "task_missing"})
    assert "error" in result
    assert "task not found" in str(result["error"])
    assert manager.last_get_customer_id == "u_1"


@pytest.mark.asyncio
async def test_browser_use_task_control_validates_action() -> None:
    tools = register_runtime_tools(_DummyRuntime(_DummyBrowserManager()))

    result = await tools["browser_use_task_control"].ainvoke(
        {"task_id": "task_123", "action": "explode"}
    )
    assert "error" in result
    assert "invalid action" in str(result["error"])


@pytest.mark.asyncio
async def test_browser_use_task_control_passes_customer_scope() -> None:
    manager = _DummyBrowserManager()
    await manager.start_task(task="open docs", max_steps=5, llm="browser-use-llm")
    tools = register_runtime_tools(_DummyRuntime(manager))

    result = await tools["browser_use_task_control"].ainvoke(
        {"task_id": "task_123", "action": "stop"}
    )
    assert result.get("status") == "stopped"
    assert manager.last_control_customer_id == "u_1"


@pytest.mark.asyncio
async def test_browser_use_task_screenshot_returns_local_path() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager))

    result = await tools["browser_use_task_screenshot"].ainvoke(
        {"task_id": "task_123", "full_page": False}
    )
    assert result.get("path") == "tulpa_stuff/screenshots/browser_use/task_123.png"
    assert manager.last_screenshot == {
        "task_id": "task_123",
        "full_page": False,
        "customer_id": "u_1",
    }


@pytest.mark.asyncio
async def test_browser_use_run_returns_when_waiting_for_owner() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager))

    async def start_waiting_task(**kwargs) -> dict:
        payload = await _DummyBrowserManager.start_task(manager, **kwargs)
        payload["status"] = "waiting_for_owner"
        payload["ownerInputPrompt"] = "Enter the email code."
        payload["ownerInputType"] = "email_code"
        payload["ownerInputRequestedAt"] = "2026-01-01T00:00:00+00:00"
        return payload

    manager.start_task = start_waiting_task  # type: ignore[method-assign]

    result = await tools["browser_use_run"].ainvoke({"task": "log in"})
    assert result.get("status") == "waiting_for_owner"
    assert result.get("owner_input_prompt") == "Enter the email code."


@pytest.mark.asyncio
async def test_browser_use_owner_input_submit_resumes_waiting_task() -> None:
    manager = _DummyBrowserManager()
    task = await manager.start_task(task="login", max_steps=5, llm="browser-use-llm")
    task["status"] = "waiting_for_owner"
    task["ownerInputPrompt"] = "Enter the SMS code."
    tools = register_runtime_tools(_DummyRuntime(manager))

    result = await tools["browser_use_owner_input_submit"].ainvoke(
        {"task_id": "task_123", "owner_input": "123456"}
    )
    assert result.get("status") == "running"
    assert result.get("output") == "owner input submitted: 123456"
    assert manager.last_submit_customer_id == "u_1"


@pytest.mark.asyncio
async def test_browser_use_run_disables_owner_input_outside_interactive_turn() -> None:
    manager = _DummyBrowserManager()
    tools = register_runtime_tools(_DummyRuntime(manager, turn_mode="routine_wake"))

    result = await tools["browser_use_run"].ainvoke({"task": "open docs"})
    assert result.get("status") == "finished"
    assert manager.last_allow_owner_input is False
