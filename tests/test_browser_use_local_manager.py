from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from opentulpa.integrations.browser_use_local import (
    BrowserUseLocalManager,
    _BrowserUseTaskState,
)
from opentulpa.integrations.browser_use_session_registry import (
    BrowserUseSessionRegistry,
    BrowserUseSessionState,
)


class _FakeBrowserSession:
    state_delay_seconds = 0.0
    fail_navigate = False

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.stopped = False
        self.started = False
        self.navigated_urls: list[str] = []
        self.current_url = ""

    async def start(self) -> None:
        self.started = True

    async def navigate_to(self, url: str) -> None:
        if self.fail_navigate:
            raise RuntimeError("navigation failed")
        self.current_url = url
        self.navigated_urls.append(url)

    async def get_current_page_url(self) -> str:
        return self.current_url

    async def get_current_page_title(self) -> str:
        return "Fake Browser Page"

    async def get_state_as_text(self) -> str:
        if self.state_delay_seconds > 0:
            await asyncio.sleep(self.state_delay_seconds)
        return "Fake visible page state"

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


class _FakeBrowserUseCloudClient:
    def __init__(self) -> None:
        self.created_profiles: list[str] = []
        self.created_sessions: list[str] = []
        self.stopped_sessions: list[str] = []

    async def create_profile(self, *, name: str) -> str:
        self.created_profiles.append(name)
        return "prof_123"

    async def create_browser_session(self, *, profile_id: str) -> Any:
        self.created_sessions.append(profile_id)
        return SimpleNamespace(
            id="bs_123",
            cdp_url="wss://secret-cdp.example/session",
            profile_id=profile_id,
            live_url="https://live.browser-use.example/session",
        )

    async def stop_browser_session(self, session_id: str) -> None:
        self.stopped_sessions.append(session_id)


async def _no_preflight() -> str | None:
    return None


def _fake_browser_use_components() -> tuple[None, None, type[_FakeBrowserSession]]:
    return None, None, _FakeBrowserSession


def test_browser_use_session_registry_tracks_active_and_reusable_sessions() -> None:
    registry = BrowserUseSessionRegistry()
    registry.set_session(
        BrowserUseSessionState(
            session=object(),
            customer_id="cust_1",
            session_id="shared",
            updated_monotonic=10.0,
        )
    )
    registry.set_task(
        _BrowserUseTaskState(
            task_id="task_running",
            customer_id="cust_1",
            session_id="shared",
            task="open page",
            llm="model",
            status="running",
            updated_monotonic=11.0,
        )
    )

    active_task = registry.active_task_for_session(customer_id="cust_1", session_id="shared")

    assert active_task is not None
    assert active_task.task_id == "task_running"
    assert registry.pick_reusable_session_id("cust_1") is None
    registry.tasks["task_running"].status = "finished"

    assert registry.active_task_for_session(customer_id="cust_1", session_id="shared") is None
    assert registry.pick_reusable_session_id("cust_1") == "shared"


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
        _fake_browser_use_components,
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
    assert not hasattr(state, "agent")
    assert "Browser snapshot captured by OpenTulpa" in str(payload.get("output") or "")
    assert "Fake visible page state" in str(payload.get("output") or "")


@pytest.mark.asyncio
async def test_local_manager_rejects_model_override_from_tool_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="z-ai/glm-5.1",
        model_override="google/gemini-3-flash-preview",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        _fake_browser_use_components,
    )

    created = await manager.start_task(
        task="Search Google for OpenAI",
        max_steps=5,
        llm="openai/gpt-4o",
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


@pytest.mark.asyncio
async def test_local_manager_normalizes_disallowed_browser_model_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="z-ai/glm-5.1",
        model_override="bu-mini",
    )
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        _fake_browser_use_components,
    )

    created = await manager.start_task(
        task="Search Google for OpenAI",
        max_steps=5,
        llm="browser-use-llm",
        session_id="sess_1",
    )
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
        _fake_browser_use_components,
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
    assert manager._sessions[manager._session_key("default", "sess_shared")].session.kwargs["keep_alive"] is True


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
        _fake_browser_use_components,
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
    assert manager._sessions[manager._session_key("default", "default")].session.kwargs["user_data_dir"] == str(
        tmp_path / "browser_profiles" / "default" / "default"
    )


@pytest.mark.asyncio
async def test_local_manager_implicit_run_uses_fallback_profile_when_default_busy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        user_data_dir=tmp_path / "browser_profiles",
    )
    _FakeBrowserSession.state_delay_seconds = 0.2
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        _fake_browser_use_components,
    )

    try:
        first = await manager.start_task(task="first slow task", max_steps=2, llm="")
        first_task_id = str(first["id"])
        for _ in range(50):
            payload = await manager.get_task(first_task_id)
            if payload and str(payload.get("status")) == "running":
                break
            await asyncio.sleep(0.01)
        else:  # pragma: no cover
            raise AssertionError("first task did not start running in time")

        second = await manager.start_task(task="second unrelated task", max_steps=2, llm="")

        assert first["sessionId"] == "default"
        assert not second.get("error")
        assert str(second["sessionId"]).startswith("bses_")
        assert second["sessionId"] != "default"
        assert len(manager._sessions) == 2
    finally:
        _FakeBrowserSession.state_delay_seconds = 0.0
        await manager.shutdown()


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
        _fake_browser_use_components,
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

    session = manager._sessions[manager._session_key("default", "owner_google_login")].session
    profile_path = Path(session.kwargs["user_data_dir"])
    assert profile_path == tmp_path / "browser_profiles" / "default" / "owner_google_login"
    assert profile_path.exists()


@pytest.mark.asyncio
async def test_local_manager_cloud_browser_uses_cdp_and_redacts_cdp_url(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cloud_client = _FakeBrowserUseCloudClient()
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
        user_data_dir=tmp_path / "browser_profiles",
        browser_use_api_key="bu-test",
    )
    manager._browser_use_cloud_client = cloud_client
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        _fake_browser_use_components,
    )

    created = await manager.start_task(
        task="open example",
        max_steps=2,
        llm="",
        session_id="owner/reddit",
        customer_id="owner_1",
    )
    task_id = str(created["id"])
    for _ in range(50):
        payload = await manager.get_task(task_id, customer_id="owner_1")
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)
    else:  # pragma: no cover
        raise AssertionError("task did not finish in time")

    assert payload is not None
    assert payload["backend"] == "browser-use-cloud"
    assert payload["liveUrl"] == "https://live.browser-use.example/session"
    assert payload["browserUseProfileId"] == "prof_123"
    assert payload["browserUseBrowserSessionId"] == "bs_123"
    assert "cdp" not in str(payload).lower()
    assert cloud_client.created_profiles == ["opentulpa-owner_1-owner_reddit"]
    assert cloud_client.created_sessions == ["prof_123"]
    session = manager._sessions[manager._session_key("owner_1", "owner_reddit")].session
    assert session.kwargs["cdp_url"] == "wss://secret-cdp.example/session"
    assert session.kwargs["captcha_solver"] is True
    assert "user_data_dir" not in session.kwargs

    sessions = await manager.list_sessions(customer_id="owner_1")
    assert sessions[0]["backend"] == "browser-use-cloud"
    assert sessions[0]["live_url"] == "https://live.browser-use.example/session"
    assert "cdp" not in str(sessions).lower()

    await manager.shutdown()
    assert cloud_client.stopped_sessions == ["bs_123"]


@pytest.mark.asyncio
async def test_local_manager_keeps_customer_identity_raw_for_access_checks(
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
        _fake_browser_use_components,
    )

    created = await manager.start_task(
        task="first",
        max_steps=2,
        llm="",
        session_id="shared",
        customer_id="acme/foo",
    )
    task_id = str(created["id"])
    for _ in range(50):
        payload = await manager.get_task(task_id, customer_id="acme/foo")
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)

    assert created["customerId"] == "acme/foo"
    assert await manager.get_task(task_id, customer_id="acme_foo") is None
    assert await manager.list_sessions(customer_id="acme_foo") == []
    session = manager._sessions[manager._session_key("acme/foo", "shared")].session
    profile_path = Path(session.kwargs["user_data_dir"])
    assert profile_path.parent.name.startswith("acme_foo-")
    assert profile_path.parent.name != "acme_foo"


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
    _FakeBrowserSession.state_delay_seconds = 0.05
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        _fake_browser_use_components,
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
    _FakeBrowserSession.state_delay_seconds = 0.0


@pytest.mark.asyncio
async def test_local_manager_fails_task_when_browser_navigation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = BrowserUseLocalManager(
        openrouter_api_key="sk-test",
        openrouter_base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-3-flash-preview",
    )
    _FakeBrowserSession.fail_navigate = True
    monkeypatch.setattr(manager, "preflight", _no_preflight)
    monkeypatch.setattr(
        manager,
        "_import_browser_use_components",
        _fake_browser_use_components,
    )

    try:
        created = await manager.start_task(
            task="open https://example.com",
            max_steps=10,
            llm="",
        )
        task_id = str(created["id"])
        payload = None
        for _ in range(150):
            payload = await manager.get_task(task_id)
            if payload and str(payload.get("status")) in {"failed", "finished", "stopped"}:
                break
            await asyncio.sleep(0.01)
        assert payload is not None
        assert payload["status"] == "failed"
        assert "navigation failed" in str(payload.get("error") or "")
    finally:
        _FakeBrowserSession.fail_navigate = False


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
        _fake_browser_use_components,
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
    assert submitted["ownerInputPrompt"] is None
    assert await waiter == "123456"

    payload = await manager.get_task("task_mfa")
    assert payload is not None
    assert payload["status"] == "running"
    assert payload["ownerInputPrompt"] is None


@pytest.mark.asyncio
async def test_local_manager_rejects_task_access_for_wrong_customer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
        _fake_browser_use_components,
    )
    monkeypatch.setattr(
        "opentulpa.integrations.browser_use_local.TULPA_STUFF_DIR",
        tmp_path / "tulpa_stuff",
    )

    created = await manager.start_task(
        task="owner login",
        max_steps=2,
        llm="",
        session_id="sess_mfa",
        customer_id="u_1",
    )
    task_id = str(created["id"])
    for _ in range(50):
        payload = await manager.get_task(task_id, customer_id="u_1")
        if payload and str(payload.get("status")) in {"finished", "failed", "stopped"}:
            break
        await asyncio.sleep(0.01)

    assert await manager.get_task(task_id, customer_id="u_2") is None

    control = await manager.control_task(
        task_id=task_id,
        action="stop",
        customer_id="u_2",
    )
    assert "task not found" in str(control.get("error"))
    assert manager._tasks[task_id].status == "finished"

    screenshot = await manager.capture_screenshot(
        task_id=task_id,
        customer_id="u_2",
    )
    assert "task not found" in str(screenshot.get("error"))
    assert manager._tasks[task_id].customer_id == "u_1"

    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    manager._tasks[task_id].status = "waiting_for_owner"
    manager._tasks[task_id].owner_input_future = future
    submit = await manager.submit_owner_input(
        task_id=task_id,
        owner_input="123456",
        customer_id="u_2",
    )
    assert "task not found" in str(submit.get("error"))
    assert not future.done()

    submit = await manager.submit_owner_input(
        task_id=task_id,
        owner_input="123456",
        customer_id="u_1",
    )
    assert submit["status"] == "running"
    assert await future == "123456"


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
        _fake_browser_use_components,
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

    session_key = manager._session_key("default", "sess_idle")
    session = manager._sessions[session_key].session
    manager._sessions[session_key].updated_monotonic = time.monotonic() - 3700
    async with manager._lock:
        manager._cleanup_locked()
    await asyncio.sleep(0)
    assert session_key not in manager._sessions
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
    session_key = manager._session_key("default", "sess_bg")
    manager._sessions[session_key] = browser_use_local._BrowserUseSessionState(
        session=session,
        customer_id="default",
        session_id="sess_bg",
    )
    manager._sessions[session_key].updated_monotonic = time.monotonic() - 3700

    async with manager._lock:
        manager._ensure_cleanup_task_locked()

    for _ in range(50):
        if session_key not in manager._sessions:
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
        _fake_browser_use_components,
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
        _fake_browser_use_components,
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
async def test_local_manager_does_not_delegate_to_browser_use_agent_when_capsolver_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
        _fake_browser_use_components,
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

    assert payload is not None
    assert payload["status"] == "finished"
    assert not hasattr(manager._tasks[task_id], "agent")
