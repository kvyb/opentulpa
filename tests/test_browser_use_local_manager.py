from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from opentulpa.integrations.browser_use_local import (
    BrowserUseLocalManager,
    _BrowserUseTaskState,
)


class _FakeChatOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeBrowserSession:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True

    async def take_screenshot(
        self,
        path: str | None = None,
        full_page: bool = False,  # noqa: ARG002
        format: str = "png",  # noqa: ARG002
        quality: int | None = None,  # noqa: ARG002
        clip: dict | None = None,  # noqa: ARG002
    ) -> bytes:
        raw = b"fake-png"
        if path:
            Path(path).write_bytes(raw)
        return raw


@dataclass
class _FakeModelAction:
    name: str

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:  # noqa: ARG002
        return {self.name: {"query": "OpenAI"}}


class _FakeModelOutput:
    def __init__(self, action_name: str = "search_google") -> None:
        self.action = [_FakeModelAction(action_name)]


class _FakeBrowserState:
    def __init__(self, url: str) -> None:
        self.url = url


class _FakeHistory:
    def __init__(self, *, success: bool = True, output: str = "done", has_error: bool = False) -> None:
        self._success = success
        self._output = output
        self._has_error = has_error

    def final_result(self) -> str:
        return self._output

    def is_successful(self) -> bool:
        return self._success

    def errors(self) -> list[str | None]:
        if self._has_error:
            return ["browser failed"]
        return []

    def urls(self) -> list[str]:
        return ["https://www.google.com/search?q=openai"]

    def model_actions(self) -> list[dict[str, Any]]:
        return [{"search_google": {"query": "openai"}}]


class _FakeAgent:
    run_delay_seconds = 0.0

    def __init__(
        self,
        *,
        task: str,
        llm: Any,
        browser_session: Any,
        register_new_step_callback: Any,
        controller: Any | None = None,
        directly_open_url: bool = True,  # noqa: ARG002
    ) -> None:
        self.task = task
        self.llm = llm
        self.browser_session = browser_session
        self.controller = controller
        self._callback = register_new_step_callback
        self._paused = False
        self._stopped = False

    async def run(self, max_steps: int = 20) -> _FakeHistory:  # noqa: ARG002
        for idx in range(2):
            if self._stopped:
                break
            while self._paused and not self._stopped:
                await asyncio.sleep(0.001)
            await self._callback(
                _FakeBrowserState(f"https://example.com/{idx + 1}"),
                _FakeModelOutput(),
                idx + 1,
            )
            if self.run_delay_seconds > 0:
                await asyncio.sleep(self.run_delay_seconds)
        if self._stopped:
            return _FakeHistory(success=False, output="stopped", has_error=False)
        return _FakeHistory(success=True, output="finished", has_error=False)

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._stopped = True


async def _no_preflight() -> str | None:
    return None


@pytest.mark.asyncio
async def test_local_manager_start_task_finishes_and_uses_default_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    created = await manager.start_task(
        task="Search Google for OpenAI",
        max_steps=5,
        llm="browser-use-llm",
        session_id="sess_1",
    )
    assert created.get("id")
    task_id = str(created["id"])

    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("task did not finish in time")

    assert payload is not None
    assert payload["status"] == "finished"
    assert payload["llm"] == "google/gemini-3-flash-preview"
    assert payload["isSuccess"] is True
    assert payload["steps"]
    state = manager._tasks[task_id]
    assert state.agent.llm.kwargs["reasoning_effort"] == "medium"
    assert state.agent.controller is not None
    assert "solve_captcha_with_capsolver" not in state.agent.task


@pytest.mark.asyncio
async def test_local_manager_reuses_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    first = await manager.start_task(task="first", max_steps=2, llm="", session_id="sess_shared")
    first_task_id = str(first["id"])
    for _ in range(50):
        payload = await manager.get_task(first_task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)
    await manager.start_task(task="second", max_steps=2, llm="", session_id="sess_shared")
    assert len(manager._sessions) == 1
    assert manager._sessions["default/sess_shared"].session.kwargs["keep_alive"] is True


@pytest.mark.asyncio
async def test_local_manager_uses_default_persistent_session_without_session_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        user_data_dir=tmp_path / "browser_profiles",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    first = await manager.start_task(task="first", max_steps=2, llm="")
    first_task_id = str(first["id"])
    for _ in range(50):
        payload = await manager.get_task(first_task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)

    second = await manager.start_task(task="second", max_steps=2, llm="")

    assert first["sessionId"] == "default"
    assert second["sessionId"] == "default"
    assert len(manager._sessions) == 1
    assert manager._sessions["default/default"].session.kwargs["user_data_dir"] == str(
        tmp_path / "browser_profiles" / "default" / "default"
    )


@pytest.mark.asyncio
async def test_local_manager_uses_persistent_profile_dir_per_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        user_data_dir=tmp_path / "browser_profiles",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    created = await manager.start_task(
        task="first",
        max_steps=2,
        llm="",
        session_id="owner/google login",
    )
    task_id = str(created["id"])
    assert created["sessionId"] == "owner_google_login"
    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)

    session = manager._sessions["default/owner_google_login"].session
    profile_path = Path(session.kwargs["user_data_dir"])
    assert profile_path == tmp_path / "browser_profiles" / "default" / "owner_google_login"
    assert profile_path.exists()


@pytest.mark.asyncio
async def test_local_manager_lists_persisted_profile_dirs(tmp_path: Path) -> None:
    profile_root = tmp_path / "browser_profiles"
    (profile_root / "u_1" / "default").mkdir(parents=True)
    (profile_root / "u_1" / "owner_google").mkdir()
    (profile_root / "u_2" / "other").mkdir(parents=True)
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        user_data_dir=profile_root,
    )

    sessions = await manager.list_sessions(customer_id="u_1")

    by_id = {item["session_id"]: item for item in sessions}
    assert by_id["default"]["persisted"] is True
    assert by_id["default"]["reusable"] is True
    assert by_id["owner_google"]["persisted"] is True
    assert by_id["owner_google"]["active_task_ids"] == []
    assert "other" not in by_id


@pytest.mark.asyncio
async def test_local_manager_deletes_profiles_unused_over_fourteen_days(tmp_path: Path) -> None:
    profile_root = tmp_path / "browser_profiles"
    stale = profile_root / "u_1" / "stale"
    fresh = profile_root / "u_1" / "fresh"
    stale.mkdir(parents=True)
    fresh.mkdir(parents=True)
    stale_metadata = stale / "profile.json"
    stale_metadata.write_text('{"lastUsedAt":"2020-01-01T00:00:00+00:00"}', encoding="utf-8")
    fresh_metadata = fresh / "profile.json"
    fresh_metadata.write_text('{"lastUsedAt":"2999-01-01T00:00:00+00:00"}', encoding="utf-8")

    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        user_data_dir=profile_root,
    )

    async with manager._lock:
        manager._cleanup_locked()

    assert not stale.exists()
    assert fresh.exists()


@pytest.mark.asyncio
async def test_local_manager_control_stop_marks_task_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    _FakeAgent.run_delay_seconds = 0.05
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    created = await manager.start_task(task="slow task", max_steps=10, llm="", session_id="sess_stop")
    task_id = str(created["id"])

    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"running", "paused"}:
            break
        await asyncio.sleep(0.005)

    controlled = await manager.control_task(task_id=task_id, action="stop")
    assert controlled["status"] == "stopped"

    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"stopped", "failed", "finished"}:
            break
        await asyncio.sleep(0.01)
    assert payload is not None
    assert payload["status"] in {"stopped", "finished"}
    _FakeAgent.run_delay_seconds = 0.0


@pytest.mark.asyncio
async def test_local_manager_cleanup_removes_expired_terminal_tasks() -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        task_retention_seconds=120,
    )
    stale = _BrowserUseTaskState(
        task_id="task_old",
        session_id="sess_old",
        task="old",
        llm="model",
        status="finished",
    )
    stale.updated_monotonic = time.monotonic() - 500
    manager._tasks["task_old"] = stale

    async with manager._lock:
        manager._cleanup_locked()
    assert "task_old" not in manager._tasks


@pytest.mark.asyncio
async def test_local_manager_capture_screenshot_writes_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import opentulpa.integrations.browser_use_local as browser_use_local

    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )
    monkeypatch.setattr(browser_use_local, "TULPA_STUFF_DIR", tmp_path / "tulpa_stuff")

    created = await manager.start_task(task="first", max_steps=2, llm="", session_id="sess_shot")
    task_id = str(created["id"])

    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("task did not finish in time")

    shot = await manager.capture_screenshot(task_id=task_id, full_page=False)
    assert shot["ok"] is True
    assert shot["path"].startswith("tulpa_stuff/screenshots/browser_use/")
    assert (tmp_path / shot["path"]).exists()

    payload = await manager.get_task(task_id)
    assert payload is not None
    assert payload["outputFiles"]
    assert payload["steps"][-1]["screenshotUrl"] == shot["path"]


@pytest.mark.asyncio
async def test_local_manager_waits_for_owner_input_and_resumes_same_task() -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    state = _BrowserUseTaskState(
        task_id="task_mfa",
        session_id="sess_mfa",
        task="log in",
        llm="model",
        status="running",
    )
    manager._tasks[state.task_id] = state

    waiter = asyncio.create_task(
        manager.request_owner_input(
            task_id="task_mfa",
            prompt="Enter the email code.",
            input_type="email_code",
        )
    )
    for _ in range(50):
        payload = await manager.get_task("task_mfa")
        if payload and payload.get("status") == "waiting_for_owner":
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("task did not start waiting for owner input")

    assert payload["ownerInputPrompt"] == "Enter the email code."
    submitted = await manager.submit_owner_input(task_id="task_mfa", owner_input="123456")
    assert submitted["status"] == "running"
    assert await waiter == "123456"

    payload = await manager.get_task("task_mfa")
    assert payload is not None
    assert payload["status"] == "running"
    assert payload["ownerInputPrompt"] is None


@pytest.mark.asyncio
async def test_local_manager_lists_sessions_and_expires_idle_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    created = await manager.start_task(task="first", max_steps=2, llm="", session_id="sess_idle")
    task_id = str(created["id"])
    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)
    sessions = await manager.list_sessions()
    assert sessions[0]["session_id"] == "sess_idle"
    assert sessions[0]["reusable"] is True

    session = manager._sessions["default/sess_idle"].session
    manager._sessions["default/sess_idle"].updated_monotonic = time.monotonic() - 3700
    async with manager._lock:
        manager._cleanup_locked()
    await asyncio.sleep(0)
    assert "default/sess_idle" not in manager._sessions
    assert session.stopped is True


@pytest.mark.asyncio
async def test_local_manager_background_cleanup_expires_idle_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentulpa.integrations.browser_use_local as browser_use_local

    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(browser_use_local, "_SESSION_CLEANUP_POLL_SECONDS", 0.01)

    session = _FakeBrowserSession()
    manager._sessions["default/sess_bg"] = browser_use_local._BrowserUseSessionState(
        session=session,
        customer_id="default",
        session_id="sess_bg",
    )
    manager._sessions["default/sess_bg"].updated_monotonic = time.monotonic() - 3700

    async with manager._lock:
        manager._ensure_cleanup_task_locked()

    for _ in range(50):
        if "default/sess_bg" not in manager._sessions:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("idle session was not cleaned up by background loop")

    assert session.stopped is True
    await manager.shutdown()


@pytest.mark.asyncio
async def test_local_manager_allows_twenty_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    for idx in range(20):
        session_id = f"sess_{idx}"
        created = await manager.start_task(task=session_id, max_steps=2, llm="", session_id=session_id)
        assert created.get("sessionId") == session_id
        task_id = str(created["id"])
        for _ in range(50):
            payload = await manager.get_task(task_id)
            if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
                break
            await asyncio.sleep(0.01)

    assert len(manager._sessions) == 20

    other_customer = await manager.start_task(
        task="other customer",
        max_steps=2,
        llm="",
        session_id="sess_0",
        customer_id="u_2",
    )
    assert other_customer.get("sessionId") == "sess_0"
    assert len(manager._sessions) == 21


@pytest.mark.asyncio
async def test_local_manager_rejects_twenty_first_explicit_session_at_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )

    for idx in range(20):
        session_id = f"sess_{idx}"
        created = await manager.start_task(task=session_id, max_steps=2, llm="", session_id=session_id)
        task_id = str(created["id"])
        for _ in range(50):
            payload = await manager.get_task(task_id)
            if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
                break
            await asyncio.sleep(0.01)

    blocked = await manager.start_task(task="extra", max_steps=2, llm="", session_id="sess_extra")
    assert "error" in blocked
    assert "session capacity reached" in str(blocked["error"])
    assert blocked["sessionLimit"] == 20
    assert len(manager._sessions) == 20


@pytest.mark.asyncio
async def test_local_manager_attaches_capsolver_controller_when_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import opentulpa.integrations.browser_use_captcha as captcha_module

    controller = object()
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        capsolver_api_key="cap-key",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        lambda: (_FakeAgent, _FakeChatOpenAI, _FakeBrowserSession),
    )
    monkeypatch.setattr(
        captcha_module,
        "register_capsolver_action",
        lambda base_controller, client: controller,
    )

    created = await manager.start_task(task="blocked by captcha", max_steps=2, llm="", session_id="sess_cap")
    task_id = str(created["id"])
    for _ in range(50):
        payload = await manager.get_task(task_id)
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("task did not finish in time")

    assert manager._tasks[task_id].agent.controller is controller
    assert "solve_captcha_with_capsolver" in manager._tasks[task_id].agent.task
