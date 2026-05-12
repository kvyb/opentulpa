"""Local Browser Use task manager with headless-first defaults."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.core.ids import new_short_id
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"finished", "stopped", "failed"}
_OWNER_WAITING_STATUS = "waiting_for_owner"
_OWNER_INPUT_TIMEOUT_SECONDS = 24 * 60 * 60
_SESSION_IDLE_TIMEOUT_SECONDS = 3600
_CLOUD_SESSION_IDLE_TIMEOUT_SECONDS = 10 * 60
_CLOUD_AGENT_POLL_SECONDS = 15.0
_CLOUD_AGENT_STUCK_REPEAT_POLLS = 8
_CLOUD_AGENT_MAX_STUCK_INTERVENTIONS = 2
_SESSION_CLEANUP_POLL_SECONDS = 60.0
_MAX_BROWSER_USE_SESSIONS = 20
_DEFAULT_SESSION_ID = "default"
_DEFAULT_CUSTOMER_ID = "default"
_PROFILE_RETENTION_SECONDS = 14 * 24 * 60 * 60
_PROFILE_METADATA_FILE = "profile.json"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class _BrowserUseTaskState:
    task_id: str
    session_id: str | None
    task: str
    llm: str
    customer_id: str = _DEFAULT_CUSTOMER_ID
    status: str = "queued"
    is_success: bool | None = None
    started_at: str | None = None
    finished_at: str | None = None
    output: str | None = None
    output_files: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cloud_agent_session_id: str | None = None
    created_monotonic: float = field(default_factory=time.monotonic)
    updated_monotonic: float = field(default_factory=time.monotonic)
    runner: asyncio.Task[Any] | None = None
    agent: Any = None
    browser_session: Any = None
    allow_owner_input: bool = True
    owner_input_prompt: str | None = None
    owner_input_type: str | None = None
    owner_input_requested_at: str | None = None
    owner_input_future: asyncio.Future[str] | None = None
    stop_requested: bool = False
    close_session_when_done: bool = False


@dataclass(slots=True)
class _BrowserUseSessionState:
    session: Any
    customer_id: str
    session_id: str
    backend: str = "local"
    cloud_profile_id: str | None = None
    cloud_browser_session_id: str | None = None
    cloud_model: str | None = None
    live_url: str | None = None
    recording_url: str | None = None
    updated_monotonic: float = field(default_factory=time.monotonic)


class BrowserUseLocalManager:
    """Manage local Browser Use runs with in-memory task/session state."""

    def __init__(
        self,
        *,
        openrouter_api_key: str,
        openrouter_base_url: str,
        default_model: str,
        model_override: str | None = None,
        reasoning_effort: str | None = "medium",
        headless: bool = True,
        max_concurrent_tasks: int = 2,
        task_retention_seconds: int = 1800,
        user_data_dir: str | Path | None = None,
        capsolver_api_key: str | None = None,
        browser_use_api_key: str | None = None,
        browser_use_cloud_proxy_country_code: str | None = "us",
        browser_use_cloud_timeout_minutes: int = 15,
    ) -> None:
        self._openrouter_api_key = str(openrouter_api_key or "").strip()
        self._openrouter_base_url = str(openrouter_base_url or "").strip().rstrip("/")
        self._default_model = str(default_model or "").strip()
        self._model_override = str(model_override or "").strip()
        self._reasoning_effort = str(reasoning_effort or "").strip() or None
        self._headless = bool(headless)
        self._task_retention_seconds = max(60, int(task_retention_seconds))
        self._user_data_dir = self._resolve_user_data_dir(user_data_dir)
        self._capsolver_api_key = str(capsolver_api_key or "").strip()
        self._browser_use_api_key = str(browser_use_api_key or "").strip()
        self._browser_use_cloud_proxy_country_code = str(
            browser_use_cloud_proxy_country_code or ""
        ).strip()
        self._browser_use_cloud_timeout_minutes = max(
            1, min(int(browser_use_cloud_timeout_minutes), 240)
        )
        self._browser_use_cloud_client: Any | None = None
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_tasks)))
        self._lock = asyncio.Lock()
        self._tasks: dict[str, _BrowserUseTaskState] = {}
        self._sessions: dict[str, _BrowserUseSessionState] = {}
        self._cloud_session_ids_by_browser_session: dict[int, str] = {}
        self._preflight_checked = False
        self._preflight_error: str | None = None
        self._cleanup_task: asyncio.Task[Any] | None = None

    async def preflight(self) -> str | None:
        if self._preflight_checked:
            return self._preflight_error
        self._preflight_checked = True

        if self._browser_use_cloud_enabled():
            if self._user_data_dir is None:
                self._preflight_error = (
                    "browser_use cloud backend unavailable: BROWSER_USE_USER_DATA_DIR "
                    "is required to remember Browser Use Cloud profile ids"
                )
                return self._preflight_error
            try:
                self._get_browser_use_cloud_client()
            except Exception as exc:
                self._preflight_error = (
                    "browser_use cloud backend unavailable: Browser Use Cloud SDK import failed "
                    f"({exc}). Install dependencies with `uv sync`."
                )
                return self._preflight_error
            self._preflight_error = None
            return None

        try:
            self._import_browser_use_components()
        except Exception as exc:
            self._preflight_error = (
                "browser_use local backend unavailable: package import failed "
                f"({exc}). Install dependencies with `uv sync`."
            )
            return self._preflight_error

        try:
            from playwright.async_api import async_playwright
        except Exception as exc:
            self._preflight_error = (
                "browser_use local backend unavailable: Playwright import failed "
                f"({exc}). Install with `uv sync`."
            )
            return self._preflight_error

        try:
            async with async_playwright() as playwright:
                chromium_path = str(getattr(playwright.chromium, "executable_path", "") or "").strip()
        except Exception as exc:
            self._preflight_error = (
                "browser_use local backend preflight failed while probing Playwright Chromium: "
                f"{exc}. Install browser binaries with `uv run playwright install chromium`."
            )
            return self._preflight_error

        if not chromium_path or not Path(chromium_path).exists():
            self._preflight_error = (
                "browser_use local backend unavailable: Playwright Chromium binary not found. "
                "Install with `uv run playwright install chromium` "
                "(Docker: `uv run playwright install --with-deps chromium`)."
            )
            return self._preflight_error

        self._preflight_error = None
        return None

    def get_preflight_error(self) -> str | None:
        return self._preflight_error

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
    ) -> dict[str, Any]:
        preflight_error = await self.preflight()
        if preflight_error:
            return {"error": preflight_error}
        if not self._openrouter_api_key and not self._browser_use_cloud_enabled():
            return {
                "error": (
                    "browser_use_run unavailable: OPENAI_COMPATIBLE_API_KEY missing "
                    "(OPENROUTER_API_KEY also accepted)"
                )
            }

        task_text = str(task or "").strip()
        if not task_text:
            return {"error": "browser_use_run requires a non-empty task"}

        resolved_model = self._resolve_model(llm=llm)
        if not resolved_model:
            return {
                "error": (
                    "browser_use_run unavailable: no model resolved. "
                    "Set BROWSER_USE_MODEL or LLM_MODEL."
                )
            }

        explicit_session_id = str(session_id or "").strip()
        has_explicit_session_id = bool(explicit_session_id)
        safe_session_id = (
            self._safe_profile_name(explicit_session_id)
            if has_explicit_session_id
            else _DEFAULT_SESSION_ID
        )
        safe_customer_id = self._normalize_customer_id(customer_id)
        safe_max_steps = max(1, min(int(max_steps), 120))
        safe_start_url = str(start_url or "").strip()
        safe_domains = self._sanitize_domains(allowed_domains)

        if self._browser_use_cloud_enabled():
            return await self._start_cloud_agent_task(
                task_text=task_text,
                resolved_model=resolved_model,
                safe_start_url=safe_start_url,
                safe_domains=safe_domains,
                safe_session_id=safe_session_id,
                safe_customer_id=safe_customer_id,
                has_explicit_session_id=has_explicit_session_id,
                allow_owner_input=allow_owner_input,
            )

        session_to_close: Any | None = None
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            active_task = self._active_task_for_session_locked(safe_customer_id, safe_session_id)
            if active_task is not None and not has_explicit_session_id:
                reusable_session_id = self._pick_reusable_session_id_locked(safe_customer_id)
                safe_session_id = reusable_session_id or new_short_id("bses")
                active_task = self._active_task_for_session_locked(safe_customer_id, safe_session_id)
            if active_task is not None:
                return {
                    "error": (
                        "browser_use_run session busy: "
                        f"active task {active_task.task_id} is still {active_task.status}"
                    ),
                    "sessionId": safe_session_id,
                    "customerId": safe_customer_id,
                    "activeTaskId": active_task.task_id,
                }
            session_key = self._session_key(safe_customer_id, safe_session_id)
            session_state = self._sessions.get(session_key)
            if (
                session_state is None
                and self._live_session_count_for_customer_locked(safe_customer_id)
                >= _MAX_BROWSER_USE_SESSIONS
            ):
                return {
                    "error": (
                        "browser_use_run session capacity reached: "
                        f"maximum {_MAX_BROWSER_USE_SESSIONS} sessions. "
                        "Reuse an existing profile for this user or stop one first."
                    ),
                    "sessionLimit": _MAX_BROWSER_USE_SESSIONS,
                        "sessions": self._session_summaries_locked(safe_customer_id),
                }
            if session_state is None:
                browser_session, browser_info = await self._new_browser_session(
                    allowed_domains=safe_domains,
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
                )
                session_state = _BrowserUseSessionState(
                    session=browser_session,
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
                    backend=browser_info.get("backend", "local"),
                    cloud_profile_id=browser_info.get("profile_id"),
                    cloud_browser_session_id=browser_info.get("session_id"),
                    live_url=browser_info.get("live_url"),
                    recording_url=browser_info.get("recording_url"),
                )
                if session_state.cloud_browser_session_id:
                    self._cloud_session_ids_by_browser_session[id(browser_session)] = (
                        session_state.cloud_browser_session_id
                    )
                self._sessions[session_key] = session_state
                self._write_profile_metadata(
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
                    status="idle",
                )
            else:
                session_state.updated_monotonic = time.monotonic()
                self._write_profile_metadata(
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
                    status="idle",
                )

            task_id = new_short_id("task")
            state = _BrowserUseTaskState(
                task_id=task_id,
                customer_id=safe_customer_id,
                session_id=safe_session_id,
                task=task_text,
                llm=resolved_model,
                status="queued",
                browser_session=session_state.session,
                allow_owner_input=bool(allow_owner_input),
            )
            runner = asyncio.create_task(
                self._run_task(
                    task_id=task_id,
                    max_steps=safe_max_steps,
                    start_url=safe_start_url,
                    allowed_domains=safe_domains,
                ),
                name=f"browser_use_local:{task_id}",
            )
            state.runner = runner
            self._tasks[task_id] = state
            self._write_profile_metadata(
                customer_id=safe_customer_id,
                session_id=safe_session_id,
                status="running",
                task_id=task_id,
            )
            payload = self._state_to_payload(state)

        if session_to_close is not None:
            await self._close_session(session_to_close)
        return payload

    async def get_task(
        self,
        task_id: str,
        *,
        customer_id: str | None = None,
    ) -> dict[str, Any] | None:
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return None
        safe_customer_id = self._normalize_optional_customer_id(customer_id)
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return None
            if safe_customer_id is not None and state.customer_id != safe_customer_id:
                return None
            self._touch_session_locked(state.customer_id, state.session_id)
            return self._state_to_payload(state)

    async def list_sessions(self, *, customer_id: str | None = None) -> list[dict[str, Any]]:
        safe_customer_id = self._normalize_customer_id(customer_id)
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            now = time.monotonic()
            out: list[dict[str, Any]] = []
            for _, session_state in self._sessions.items():
                if session_state.customer_id != safe_customer_id:
                    continue
                related_tasks = [
                    state
                    for state in self._tasks.values()
                    if state.customer_id == safe_customer_id
                    and str(state.session_id or "").strip() == session_state.session_id
                ]
                related_tasks.sort(
                    key=lambda item: float(item.updated_monotonic or item.created_monotonic),
                    reverse=True,
                )
                latest = related_tasks[0] if related_tasks else None
                active_task_ids = [
                    state.task_id for state in related_tasks if state.status not in _TERMINAL_STATUSES
                ][:3]
                last_url = None
                if latest is not None and latest.steps:
                    last_url = str(latest.steps[-1].get("url", "")).strip() or None
                out.append(
                    {
                        "session_id": session_state.session_id,
                        "customer_id": safe_customer_id,
                        "backend": session_state.backend,
                        "reusable": not active_task_ids,
                        "persisted": self._profile_dir_exists(safe_customer_id, session_state.session_id),
                        "live_url": session_state.live_url,
                        "recording_url": session_state.recording_url,
                        "cloud_profile_id": session_state.cloud_profile_id,
                        "cloud_browser_session_id": session_state.cloud_browser_session_id,
                        "cloud_model": session_state.cloud_model,
                        "active_task_ids": active_task_ids,
                        "latest_task_id": latest.task_id if latest is not None else None,
                        "latest_status": latest.status if latest is not None else None,
                        "owner_input_prompt": latest.owner_input_prompt if latest is not None else None,
                        "last_url": last_url,
                        "last_used_seconds": max(
                            0,
                            int(now - float(session_state.updated_monotonic or now)),
                        ),
                    }
                )
            seen = {str(item.get("session_id", "")).strip() for item in out}
            for session_id, profile_dir, metadata in self._persisted_profile_dirs(safe_customer_id):
                if session_id in seen:
                    continue
                out.append(
                    {
                        "session_id": session_id,
                        "customer_id": safe_customer_id,
                        "backend": metadata.get("backend") or "local",
                        "reusable": (
                            self._live_session_count_for_customer_locked(safe_customer_id)
                            < _MAX_BROWSER_USE_SESSIONS
                        ),
                        "persisted": True,
                        "live_url": metadata.get("liveUrl") or None,
                        "recording_url": metadata.get("recordingUrl") or None,
                        "cloud_profile_id": metadata.get("browserUseProfileId") or None,
                        "cloud_browser_session_id": metadata.get("browserUseBrowserSessionId") or None,
                        "cloud_model": metadata.get("browserUseModel") or None,
                        "active_task_ids": [],
                        "latest_task_id": None,
                        "latest_status": None,
                        "owner_input_prompt": None,
                        "last_url": metadata.get("lastUrl") or None,
                        "last_used_seconds": max(
                            0,
                            int(now - self._profile_last_used_timestamp(profile_dir, metadata)),
                        ),
                    }
                )
            out.sort(key=lambda item: (item["last_used_seconds"], item["session_id"]))
            return out

    async def control_task(
        self,
        *,
        task_id: str,
        action: str,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        safe_action = str(action or "").strip().lower()
        if not safe_task_id:
            return {"error": "browser_use_task_control requires task_id"}
        safe_customer_id = self._normalize_optional_customer_id(customer_id)

        session_to_close: Any | None = None
        cloud_task_session_id: str | None = None
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return {"error": f"browser_use_task_control task not found: {safe_task_id}"}
            if safe_customer_id is not None and state.customer_id != safe_customer_id:
                return {"error": f"browser_use_task_control task not found: {safe_task_id}"}

            agent = state.agent
            if safe_action == "pause":
                if agent is not None and hasattr(agent, "pause"):
                    with suppress(Exception):
                        agent.pause()
                if state.status == "running":
                    state.status = "paused"
                    state.updated_monotonic = time.monotonic()
                    self._touch_session_locked(state.customer_id, state.session_id)
            elif safe_action == "resume":
                if agent is not None and hasattr(agent, "resume"):
                    with suppress(Exception):
                        agent.resume()
                if state.status in {"paused", "queued"}:
                    state.status = "running"
                    state.updated_monotonic = time.monotonic()
                    self._touch_session_locked(state.customer_id, state.session_id)
            elif safe_action in {"stop", "stop_task_and_session"}:
                state.stop_requested = True
                if self._session_is_cloud_locked(state.customer_id, state.session_id):
                    cloud_task_session_id = state.cloud_agent_session_id
                if state.owner_input_future is not None and not state.owner_input_future.done():
                    state.owner_input_future.cancel()
                if agent is not None and hasattr(agent, "stop"):
                    with suppress(Exception):
                        agent.stop()
                state.status = "stopped"
                state.is_success = False
                if not state.finished_at:
                    state.finished_at = _utc_now_iso()
                state.updated_monotonic = time.monotonic()
                if safe_action == "stop_task_and_session":
                    state.close_session_when_done = True
                    if state.runner is None or state.runner.done():
                        session_to_close = self._detach_session_if_unused_locked(
                            state.customer_id, state.session_id
                        )
            else:
                return {
                    "error": (
                        "browser_use_task_control invalid action. "
                        "Use one of: stop, pause, resume, stop_task_and_session"
                    )
                }

            payload = self._state_to_payload(state)

        if cloud_task_session_id and safe_action == "stop":
            with suppress(Exception):
                await self._get_browser_use_cloud_client().stop_agent_session(
                    cloud_task_session_id,
                    strategy="task",
                )
        if session_to_close is not None:
            await self._close_session(session_to_close)
        return payload

    async def capture_screenshot(
        self,
        *,
        task_id: str,
        full_page: bool = True,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_screenshot requires task_id"}
        safe_customer_id = self._normalize_optional_customer_id(customer_id)

        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return {"error": f"browser_use_task_screenshot task not found: {safe_task_id}"}
            if safe_customer_id is not None and state.customer_id != safe_customer_id:
                return {"error": f"browser_use_task_screenshot task not found: {safe_task_id}"}
            if self._session_is_cloud_locked(state.customer_id, state.session_id):
                screenshot_url = None
                for step in reversed(state.steps):
                    screenshot_url = str(step.get("screenshotUrl") or "").strip()
                    if screenshot_url:
                        break
                if screenshot_url:
                    return {
                        "ok": True,
                        "task_id": safe_task_id,
                        "session_id": state.session_id,
                        "screenshot_url": screenshot_url,
                    }
                return {
                    "error": (
                        "browser_use_task_screenshot unavailable: Browser Use Cloud Agent "
                        "has not returned a screenshot URL yet"
                    )
                }
            browser_session = state.browser_session
            session_id = str(state.session_id or "").strip() or None
            self._touch_session_locked(state.customer_id, state.session_id)

        if browser_session is None or not hasattr(browser_session, "take_screenshot"):
            return {"error": "browser_use_task_screenshot unavailable: browser session missing"}

        screenshot_dir = (TULPA_STUFF_DIR / "screenshots" / "browser_use").resolve()
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = (screenshot_dir / f"{safe_task_id}_{timestamp}.png").resolve()
        try:
            raw_bytes = await browser_session.take_screenshot(
                path=str(target),
                full_page=bool(full_page),
                format="png",
            )
        except Exception as exc:
            return {"error": f"browser_use_task_screenshot failed: {exc}"}

        if not target.exists() and isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
            target.write_bytes(bytes(raw_bytes))
        if not target.exists():
            return {"error": "browser_use_task_screenshot failed: screenshot file not created"}

        rel_path = str(target.relative_to(TULPA_STUFF_DIR.parent))
        file_entry = {
            "id": new_short_id("shot"),
            "fileName": target.name,
            "path": rel_path,
        }
        async with self._lock:
            state = self._tasks.get(safe_task_id)
            if (
                state is not None
                and (safe_customer_id is None or state.customer_id == safe_customer_id)
            ):
                state.output_files = [
                    item
                    for item in state.output_files
                    if str(item.get("path", "")).strip() != rel_path
                ]
                state.output_files.append(file_entry)
                if state.steps:
                    state.steps[-1]["screenshotUrl"] = rel_path
                state.updated_monotonic = time.monotonic()
                self._touch_session_locked(state.customer_id, state.session_id)
        return {
            "ok": True,
            "task_id": safe_task_id,
            "session_id": session_id,
            "path": rel_path,
            "file_name": target.name,
        }

    async def request_owner_input(
        self,
        *,
        task_id: str,
        prompt: str,
        input_type: str = "text",
    ) -> str:
        safe_task_id = str(task_id or "").strip()
        safe_prompt = str(prompt or "").strip()
        safe_input_type = str(input_type or "").strip() or "text"
        if not safe_task_id:
            raise ValueError("request_owner_input requires task_id")
        if not safe_prompt:
            safe_prompt = "Owner input is required to continue the browser task."

        loop = asyncio.get_running_loop()
        async with self._lock:
            state = self._tasks.get(safe_task_id)
            if state is None:
                raise ValueError(f"request_owner_input task not found: {safe_task_id}")
            if state.status in _TERMINAL_STATUSES:
                raise ValueError(f"request_owner_input task is already {state.status}")
            if state.owner_input_future is not None and not state.owner_input_future.done():
                raise ValueError("request_owner_input is already waiting for owner input")

            future: asyncio.Future[str] = loop.create_future()
            state.owner_input_future = future
            state.owner_input_prompt = safe_prompt
            state.owner_input_type = safe_input_type
            state.owner_input_requested_at = _utc_now_iso()
            state.status = _OWNER_WAITING_STATUS
            state.updated_monotonic = time.monotonic()
            self._touch_session_locked(state.customer_id, state.session_id)

        try:
            return await asyncio.wait_for(future, timeout=_OWNER_INPUT_TIMEOUT_SECONDS)
        finally:
            async with self._lock:
                state = self._tasks.get(safe_task_id)
                if state is not None and state.owner_input_future is future:
                    state.owner_input_future = None
                    state.owner_input_prompt = None
                    state.owner_input_type = None
                    state.owner_input_requested_at = None
                    if state.status == _OWNER_WAITING_STATUS:
                        state.status = "running"
                    state.updated_monotonic = time.monotonic()
                    self._touch_session_locked(state.customer_id, state.session_id)

    async def submit_owner_input(
        self,
        *,
        task_id: str,
        owner_input: str,
        customer_id: str | None = None,
    ) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        safe_owner_input = str(owner_input or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_owner_input_submit requires task_id"}
        if not safe_owner_input:
            return {"error": "browser_use_owner_input_submit requires owner_input"}
        safe_customer_id = self._normalize_optional_customer_id(customer_id)

        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return {"error": f"browser_use_owner_input_submit task not found: {safe_task_id}"}
            if safe_customer_id is not None and state.customer_id != safe_customer_id:
                return {"error": f"browser_use_owner_input_submit task not found: {safe_task_id}"}
            if state.status != _OWNER_WAITING_STATUS:
                return {
                    "error": (
                        "browser_use_owner_input_submit requires a task waiting for owner input; "
                        f"current status is {state.status}"
                    )
                }
            if self._session_is_cloud_locked(state.customer_id, state.session_id):
                state.task = (
                    f"{state.task}\n\n"
                    f"The owner completed the requested live-browser handoff and replied: "
                    f"{safe_owner_input}. Continue from the current browser state."
                )
                state.owner_input_future = None
                state.owner_input_prompt = None
                state.owner_input_type = None
                state.owner_input_requested_at = None
                state.status = "queued"
                state.finished_at = None
                state.updated_monotonic = time.monotonic()
                runner = asyncio.create_task(
                    self._run_cloud_agent_task(
                        task_id=safe_task_id,
                        start_url="",
                        allowed_domains=[],
                    ),
                    name=f"browser_use_cloud_agent:{safe_task_id}:resume",
                )
                state.runner = runner
                self._touch_session_locked(state.customer_id, state.session_id)
                return self._state_to_payload(state)
            future = state.owner_input_future
            if future is None or future.done():
                return {"error": "browser_use_owner_input_submit has no pending owner input request"}
            future.set_result(safe_owner_input)
            state.owner_input_future = None
            state.owner_input_prompt = None
            state.owner_input_type = None
            state.owner_input_requested_at = None
            state.status = "running"
            state.updated_monotonic = time.monotonic()
            self._touch_session_locked(state.customer_id, state.session_id)
            return self._state_to_payload(state)

    async def shutdown(self) -> None:
        async with self._lock:
            task_states = list(self._tasks.values())
            sessions = [item.session for item in self._sessions.values()]
            cleanup_task = self._cleanup_task
            self._cleanup_task = None
            self._tasks.clear()
            self._sessions.clear()

        if cleanup_task is not None:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task

        runners: list[asyncio.Task[Any]] = []
        for state in task_states:
            state.stop_requested = True
            if state.agent is not None and hasattr(state.agent, "stop"):
                with suppress(Exception):
                    state.agent.stop()
            if state.runner is not None and not state.runner.done():
                runners.append(state.runner)

        for runner in runners:
            with suppress(Exception):
                await asyncio.wait_for(runner, timeout=2.0)

        for session in sessions:
            await self._close_session(session)

    async def _start_cloud_agent_task(
        self,
        *,
        task_text: str,
        resolved_model: str,
        safe_start_url: str,
        safe_domains: list[str],
        safe_session_id: str,
        safe_customer_id: str,
        has_explicit_session_id: bool,
        allow_owner_input: bool,
    ) -> dict[str, Any]:
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            active_task = self._active_task_for_session_locked(safe_customer_id, safe_session_id)
            if active_task is not None and not has_explicit_session_id:
                reusable_session_id = self._pick_reusable_session_id_locked(safe_customer_id)
                safe_session_id = reusable_session_id or new_short_id("bses")
                active_task = self._active_task_for_session_locked(safe_customer_id, safe_session_id)
            if active_task is not None:
                return {
                    "error": (
                        "browser_use_run session busy: "
                        f"active task {active_task.task_id} is still {active_task.status}"
                    ),
                    "sessionId": safe_session_id,
                    "customerId": safe_customer_id,
                    "activeTaskId": active_task.task_id,
                }
            if (
                self._sessions.get(self._session_key(safe_customer_id, safe_session_id)) is None
                and self._live_session_count_for_customer_locked(safe_customer_id)
                >= _MAX_BROWSER_USE_SESSIONS
            ):
                return {
                    "error": (
                        "browser_use_run session capacity reached: "
                        f"maximum {_MAX_BROWSER_USE_SESSIONS} sessions. "
                        "Reuse an existing profile for this user or stop one first."
                    ),
                    "sessionLimit": _MAX_BROWSER_USE_SESSIONS,
                    "sessions": self._session_summaries_locked(safe_customer_id),
                }

            task_id = new_short_id("task")
            state = _BrowserUseTaskState(
                task_id=task_id,
                customer_id=safe_customer_id,
                session_id=safe_session_id,
                task=task_text,
                llm=resolved_model,
                status="queued",
                allow_owner_input=bool(allow_owner_input),
            )
            runner = asyncio.create_task(
                self._run_cloud_agent_task(
                    task_id=task_id,
                    start_url=safe_start_url,
                    allowed_domains=safe_domains,
                ),
                name=f"browser_use_cloud_agent:{task_id}",
            )
            state.runner = runner
            self._tasks[task_id] = state
            self._write_profile_metadata(
                customer_id=safe_customer_id,
                session_id=safe_session_id,
                status="running",
                backend="browser-use-cloud-agent",
                task_id=task_id,
            )
            return self._state_to_payload(state)

    async def _run_cloud_agent_task(
        self,
        *,
        task_id: str,
        start_url: str,
        allowed_domains: list[str],
    ) -> None:
        await self._semaphore.acquire()
        try:
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                state.status = "running"
                state.started_at = state.started_at or _utc_now_iso()
                state.updated_monotonic = time.monotonic()
                customer_id = state.customer_id
                session_id = str(state.session_id or "")
                model_name = state.llm
                task_text = state.task
                allow_owner_input = state.allow_owner_input

            profile_id = self._browser_use_cloud_profile_id(customer_id, session_id)
            if not profile_id:
                profile_id = await self._get_browser_use_cloud_client().create_profile(
                    name=self._browser_use_cloud_profile_name(customer_id, session_id)
                )
            composed_task = self._compose_cloud_agent_task(
                task_text=task_text,
                start_url=start_url,
                allowed_domains=allowed_domains,
                allow_owner_input=allow_owner_input,
            )
            cloud_model = self._browser_use_cloud_model(model_name)
            existing_session_id = await self._cloud_agent_session_id(customer_id, session_id)
            cloud_session = await self._create_cloud_agent_session(
                task=composed_task,
                model=cloud_model,
                profile_id=profile_id,
                session_id=existing_session_id,
            )
            recording_url = (cloud_session.recording_urls or [None])[-1]
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                state.cloud_agent_session_id = cloud_session.id
                state.browser_session = cloud_session.id
                session_state = _BrowserUseSessionState(
                    session=cloud_session.id,
                    customer_id=customer_id,
                    session_id=session_id,
                    backend="browser-use-cloud-agent",
                    cloud_profile_id=profile_id,
                    cloud_browser_session_id=cloud_session.id,
                    cloud_model=cloud_model,
                    live_url=cloud_session.live_url,
                    recording_url=recording_url,
                )
                self._sessions[self._session_key(customer_id, session_id)] = session_state
                self._write_profile_metadata(
                    customer_id=customer_id,
                    session_id=session_id,
                    status="running",
                    backend="browser-use-cloud-agent",
                    task_id=task_id,
                    cloud_profile_id=profile_id,
                    cloud_browser_session_id=cloud_session.id,
                    cloud_model=cloud_model,
                    live_url=cloud_session.live_url,
                    recording_url=recording_url,
                )

            last_status = ""
            repeated_signature = ""
            repeated_count = 0
            stuck_interventions = 0
            for _ in range(max(1, int(self._browser_use_cloud_timeout_minutes * 30))):
                async with self._lock:
                    state = self._tasks.get(task_id)
                    if state is None or state.stop_requested:
                        break
                cloud_session = await self._get_browser_use_cloud_client().get_agent_session(
                    cloud_session.id
                )
                last_status = str(cloud_session.status or "").strip().lower()
                await self._apply_cloud_agent_session(task_id, cloud_session)
                if last_status in {"idle", "stopped", "timed_out", "error"}:
                    break
                progress_signature = self._cloud_agent_progress_signature(cloud_session)
                if progress_signature and progress_signature == repeated_signature:
                    repeated_count += 1
                else:
                    repeated_signature = progress_signature
                    repeated_count = 1 if progress_signature else 0
                if repeated_count >= _CLOUD_AGENT_STUCK_REPEAT_POLLS:
                    if stuck_interventions >= _CLOUD_AGENT_MAX_STUCK_INTERVENTIONS:
                        await self._get_browser_use_cloud_client().stop_agent_session(
                            cloud_session.id,
                            strategy="task",
                        )
                        async with self._lock:
                            state = self._tasks.get(task_id)
                            if state is not None:
                                state.error = (
                                    "Browser Use Cloud Agent stopped because it repeated "
                                    f"the same progress signal: {progress_signature[:300]}"
                                )
                                state.is_success = False
                        last_status = "error"
                        break
                    await self._get_browser_use_cloud_client().stop_agent_session(
                        cloud_session.id,
                        strategy="task",
                    )
                    stuck_interventions += 1
                    recovery_task = self._compose_cloud_agent_recovery_task(
                        original_task=composed_task,
                        repeated_signal=progress_signature,
                        attempt=stuck_interventions,
                    )
                    cloud_session = await self._create_cloud_agent_session(
                        task=recovery_task,
                        model=cloud_model,
                        profile_id=profile_id,
                        session_id=cloud_session.id,
                    )
                    async with self._lock:
                        state = self._tasks.get(task_id)
                        if state is not None:
                            state.cloud_agent_session_id = cloud_session.id
                            state.browser_session = cloud_session.id
                            recovery_session_state = self._sessions.get(
                                self._session_key(state.customer_id, str(state.session_id or ""))
                            )
                            if recovery_session_state is not None:
                                recovery_session_state.cloud_browser_session_id = cloud_session.id
                                recovery_session_state.cloud_model = cloud_model
                                recovery_session_state.live_url = (
                                    cloud_session.live_url or recovery_session_state.live_url
                                )
                                recovery_session_state.updated_monotonic = time.monotonic()
                    repeated_count = 0
                    repeated_signature = ""
                await asyncio.sleep(_CLOUD_AGENT_POLL_SECONDS)

            await self._finish_cloud_agent_task(task_id, last_status=last_status)
        except Exception as exc:
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is not None:
                    state.status = "failed"
                    state.is_success = False
                    state.error = str(exc)[:2000]
                    state.finished_at = _utc_now_iso()
                    state.updated_monotonic = time.monotonic()
                    self._write_profile_metadata(
                        customer_id=state.customer_id,
                        session_id=str(state.session_id or ""),
                        status="failed",
                        backend="browser-use-cloud-agent",
                        task_id=state.task_id,
                    )
        finally:
            self._semaphore.release()

    async def _run_task(
        self,
        *,
        task_id: str,
        max_steps: int,
        start_url: str,
        allowed_domains: list[str],
    ) -> None:
        await self._semaphore.acquire()
        try:
            agent_cls, chat_openai_cls, _ = self._import_browser_use_components()
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                state.status = "running"
                state.started_at = state.started_at or _utc_now_iso()
                state.updated_monotonic = time.monotonic()
                self._touch_session_locked(state.customer_id, state.session_id)
                model_name = state.llm
                task_text = state.task
                browser_session = state.browser_session

            llm = chat_openai_cls(
                model=model_name,
                api_key=self._openrouter_api_key,
                base_url=self._openrouter_base_url,
                reasoning_effort=self._reasoning_effort or "none",
            )
            composed_task = task_text
            if start_url:
                composed_task = (
                    f"First navigate to this URL: {start_url}. "
                    f"Then complete this task: {task_text}"
                )
            if self._capsolver_api_key:
                composed_task = (
                    f"{composed_task}\n\n"
                    "If a supported CAPTCHA blocks progress, use the "
                    "solve_captcha_with_capsolver action before continuing. "
                    "Supported challenges are reCAPTCHA v2, reCAPTCHA v3, and Cloudflare Turnstile."
                )
            if state is not None and state.allow_owner_input:
                composed_task = (
                    f"{composed_task}\n\n"
                    "Owner handoff rule: if progress requires login, sign-in, account selection, "
                    "CAPTCHA that the registered solver cannot solve, MFA, email/SMS code, "
                    "authenticator approval, credentials, or any other owner-only verification, "
                    "use the request_owner_input action immediately before failing or trying a "
                    "different browser/session. Tell the owner exactly what to do in the live browser "
                    "session, for example: 'Open the live browser link and finish login, then reply "
                    "done.' After the owner confirms, continue in this same browser session."
                )

            agent_kwargs: dict[str, Any] = {
                "task": composed_task,
                "llm": llm,
                "browser_session": browser_session,
                "register_new_step_callback": self._step_callback(task_id),
                "directly_open_url": True,
            }
            allow_owner_input = bool(state.allow_owner_input) if state is not None else True
            controller = self._new_controller(task_id=task_id, allow_owner_input=allow_owner_input)
            if controller is not None:
                agent_kwargs["controller"] = controller
            agent = agent_cls(**agent_kwargs)
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is not None:
                    state.agent = agent

            history = await agent.run(max_steps=max_steps)
            history_payload = self._history_to_payload(history)

            completed_session_to_close: Any | None = None
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                state.output = history_payload["output"]
                state.is_success = history_payload["is_success"]
                state.steps = history_payload["steps"] or state.steps
                state.error = history_payload["error"]
                state.finished_at = _utc_now_iso()
                state.updated_monotonic = time.monotonic()

                if state.status == _OWNER_WAITING_STATUS and not state.stop_requested:
                    state.finished_at = None
                    return
                if state.stop_requested:
                    state.status = "stopped"
                    state.is_success = False
                    if not state.output:
                        state.output = "Task stopped by user."
                elif history_payload["failed"]:
                    state.status = "failed"
                    if state.is_success is None:
                        state.is_success = False
                else:
                    state.status = "finished"
                    if state.is_success is None:
                        state.is_success = not bool(state.error)
                self._write_profile_metadata(
                    customer_id=state.customer_id,
                    session_id=str(state.session_id or ""),
                    status=state.status,
                    task_id=state.task_id,
                    last_url=self._latest_step_url(state),
                )

                if state.close_session_when_done or self._session_is_cloud_locked(
                    state.customer_id, state.session_id
                ):
                    completed_session_to_close = self._detach_session_if_unused_locked(
                        state.customer_id, state.session_id
                    )

            if completed_session_to_close is not None:
                await self._close_session(completed_session_to_close)
        except Exception as exc:
            failed_session_to_close: Any | None = None
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is not None:
                    state.status = "failed"
                    state.is_success = False
                    state.error = str(exc)[:2000]
                    state.finished_at = _utc_now_iso()
                    state.updated_monotonic = time.monotonic()
                    self._write_profile_metadata(
                        customer_id=state.customer_id,
                        session_id=str(state.session_id or ""),
                        status="failed",
                        task_id=state.task_id,
                        last_url=self._latest_step_url(state),
                    )
                    if state.close_session_when_done or self._session_is_cloud_locked(
                        state.customer_id, state.session_id
                    ):
                        failed_session_to_close = self._detach_session_if_unused_locked(
                            state.customer_id, state.session_id
                        )
            if failed_session_to_close is not None:
                await self._close_session(failed_session_to_close)
        finally:
            self._semaphore.release()

    def _step_callback(self, task_id: str) -> Any:
        async def _callback(browser_state_summary: Any, model_output: Any, n_steps: int) -> None:
            step_number = max(1, int(n_steps))
            url = str(getattr(browser_state_summary, "url", "") or "").strip() or None
            actions = self._extract_actions(model_output)
            step = {
                "number": step_number,
                "url": url,
                "nextGoal": "",
                "actions": actions[:5],
                "screenshotUrl": None,
            }
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                replaced = False
                for idx, existing in enumerate(state.steps):
                    if int(existing.get("number", 0) or 0) == step_number:
                        state.steps[idx] = step
                        replaced = True
                        break
                if not replaced:
                    state.steps.append(step)
                    state.steps.sort(key=lambda item: int(item.get("number", 0) or 0))
                state.updated_monotonic = time.monotonic()
                self._touch_session_locked(state.customer_id, state.session_id)
                self._write_profile_metadata(
                    customer_id=state.customer_id,
                    session_id=str(state.session_id or ""),
                    status=state.status,
                    task_id=state.task_id,
                    last_url=url,
                )

        return _callback

    def _history_to_payload(self, history: Any) -> dict[str, Any]:
        output = ""
        is_success: bool | None = None
        error_lines: list[str] = []
        failed = False
        steps = self._extract_steps_from_history(history)

        with suppress(Exception):
            out = history.final_result()
            if out is not None:
                output = str(out).strip()

        with suppress(Exception):
            value = history.is_successful()
            if isinstance(value, bool):
                is_success = value

        with suppress(Exception):
            errors = history.errors()
            if isinstance(errors, list):
                error_lines = [str(item).strip() for item in errors if str(item or "").strip()]
                if error_lines:
                    failed = True

        if not output and error_lines:
            output = "\n".join(error_lines[:6])[:12000]

        return {
            "output": output or None,
            "is_success": is_success,
            "error": "\n".join(error_lines[:6])[:1200] if error_lines else None,
            "failed": bool(failed),
            "steps": steps,
        }

    def _extract_steps_from_history(self, history: Any) -> list[dict[str, Any]]:
        urls: list[str | None] = []
        actions_by_step: list[Any] = []
        with suppress(Exception):
            values = history.urls()
            if isinstance(values, list):
                urls = [str(v).strip() if v is not None else None for v in values]

        with suppress(Exception):
            values = history.model_actions()
            if isinstance(values, list):
                actions_by_step = values

        step_count = max(len(urls), len(actions_by_step))
        out: list[dict[str, Any]] = []
        for idx in range(step_count):
            step_actions: list[str] = []
            if idx < len(actions_by_step):
                raw = actions_by_step[idx]
                if isinstance(raw, dict):
                    for key, value in list(raw.items())[:5]:
                        name = str(key or "").strip()
                        if not name:
                            continue
                        if isinstance(value, dict) and value:
                            arg_preview = ", ".join(str(k) for k in list(value.keys())[:3])
                            step_actions.append(f"{name}({arg_preview})")
                        else:
                            step_actions.append(name)
                elif raw is not None:
                    step_actions.append(str(raw)[:200])
            out.append(
                {
                    "number": idx + 1,
                    "url": urls[idx] if idx < len(urls) else None,
                    "nextGoal": "",
                    "actions": step_actions[:5],
                    "screenshotUrl": None,
                }
            )
        return out

    def _extract_actions(self, model_output: Any) -> list[str]:
        raw_actions = getattr(model_output, "action", None)
        if not isinstance(raw_actions, list):
            return []
        out: list[str] = []
        for item in raw_actions[:5]:
            data: dict[str, Any] = {}
            if hasattr(item, "model_dump"):
                with suppress(Exception):
                    dumped = item.model_dump(exclude_none=True)
                    if isinstance(dumped, dict):
                        data = dumped
            elif isinstance(item, dict):
                data = item
            if not data:
                text = str(item or "").strip()
                if text:
                    out.append(text[:200])
                continue
            for key, value in list(data.items())[:1]:
                name = str(key or "").strip()
                if not name:
                    continue
                if isinstance(value, dict) and value:
                    arg_preview = ", ".join(str(k) for k in list(value.keys())[:3])
                    out.append(f"{name}({arg_preview})")
                else:
                    out.append(name)
        return out

    @staticmethod
    def _latest_step_url(state: _BrowserUseTaskState) -> str | None:
        for step in reversed(state.steps):
            if not isinstance(step, dict):
                continue
            url = str(step.get("url", "") or "").strip()
            if url:
                return url
        return None

    def _resolve_model(self, *, llm: str) -> str:
        candidate = str(llm or "").strip()
        if candidate and candidate.lower() not in {"browser-use-llm", "default"}:
            return candidate
        if self._model_override:
            return self._model_override
        return self._default_model

    @staticmethod
    def _sanitize_domains(allowed_domains: list[str] | None) -> list[str]:
        if not isinstance(allowed_domains, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in allowed_domains:
            value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
        return out

    async def _new_browser_session(
        self,
        *,
        allowed_domains: list[str],
        customer_id: str,
        session_id: str,
    ) -> tuple[Any, dict[str, str]]:
        _, _, browser_session_cls = self._import_browser_use_components()
        if self._browser_use_cloud_enabled():
            cloud_session = await self._new_browser_use_cloud_session(
                customer_id=customer_id,
                session_id=session_id,
            )
            cloud_session_kwargs: dict[str, Any] = {
                "cdp_url": cloud_session["cdp_url"],
                "keep_alive": True,
            }
            if allowed_domains:
                cloud_session_kwargs["allowed_domains"] = allowed_domains
            return browser_session_cls(**cloud_session_kwargs), cloud_session

        session_kwargs: dict[str, Any] = {"headless": self._headless, "keep_alive": True}
        if allowed_domains:
            session_kwargs["allowed_domains"] = allowed_domains
        if self._user_data_dir is not None:
            session_profile_dir = self._profile_dir(customer_id, session_id)
            session_profile_dir.mkdir(parents=True, exist_ok=True)
            session_kwargs["user_data_dir"] = str(session_profile_dir)
        return browser_session_cls(**session_kwargs), {"backend": "local"}

    async def _new_browser_use_cloud_session(
        self, *, customer_id: str, session_id: str
    ) -> dict[str, str]:
        profile_id = self._browser_use_cloud_profile_id(customer_id, session_id)
        if not profile_id:
            profile_id = await self._get_browser_use_cloud_client().create_profile(
                name=self._browser_use_cloud_profile_name(customer_id, session_id)
            )
            self._write_profile_metadata(
                customer_id=customer_id,
                session_id=session_id,
                status="idle",
                backend="browser-use-cloud",
                cloud_profile_id=profile_id,
            )
        session = await self._get_browser_use_cloud_client().create_browser_session(
            profile_id=profile_id,
        )
        self._write_profile_metadata(
            customer_id=customer_id,
            session_id=session_id,
            status="idle",
            backend="browser-use-cloud",
            cloud_profile_id=session.profile_id,
            cloud_browser_session_id=session.id,
            live_url=session.live_url,
            recording_url=session.recording_url,
        )
        return {
            "backend": "browser-use-cloud",
            "cdp_url": session.cdp_url,
            "profile_id": session.profile_id,
            "session_id": session.id,
            "live_url": session.live_url or "",
            "recording_url": session.recording_url or "",
        }

    async def _apply_cloud_agent_session(self, task_id: str, cloud_session: Any) -> None:
        messages = await self._get_browser_use_cloud_client().list_agent_messages(
            cloud_session.id,
            limit=20,
        )
        steps: list[dict[str, Any]] = []
        for idx, message in enumerate(messages, start=1):
            summary = str(message.summary or message.data or "").strip()
            if not summary:
                continue
            steps.append(
                {
                    "number": idx,
                    "url": None,
                    "nextGoal": summary[:500],
                    "actions": [str(message.message_type or message.role or "message")],
                    "screenshotUrl": message.screenshot_url,
                }
            )
        output = self._stringify_cloud_output(getattr(cloud_session, "output", None))
        async with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return
            state.output = output or state.output
            state.is_success = getattr(cloud_session, "is_task_successful", None)
            if steps:
                state.steps = steps
            state.updated_monotonic = time.monotonic()
            session_state = self._sessions.get(
                self._session_key(state.customer_id, str(state.session_id or ""))
            )
            if session_state is not None:
                session_state.live_url = getattr(cloud_session, "live_url", None) or session_state.live_url
                recording_urls = getattr(cloud_session, "recording_urls", None) or []
                if recording_urls:
                    session_state.recording_url = str(recording_urls[-1])
                session_state.updated_monotonic = time.monotonic()

    async def _create_cloud_agent_session(
        self,
        *,
        task: str,
        model: str,
        profile_id: str,
        session_id: str | None,
    ) -> Any:
        client = self._get_browser_use_cloud_client()
        try:
            cloud_session = await client.create_agent_session(
                task=task,
                model=model,
                profile_id=profile_id,
                session_id=session_id,
                keep_alive=True,
            )
            self._validate_cloud_agent_model(cloud_session=cloud_session, requested_model=model)
            return cloud_session
        except Exception:
            if not session_id:
                raise
            cloud_session = await client.create_agent_session(
                task=task,
                model=model,
                profile_id=profile_id,
                session_id=None,
                keep_alive=True,
            )
            self._validate_cloud_agent_model(cloud_session=cloud_session, requested_model=model)
            return cloud_session

    @staticmethod
    def _validate_cloud_agent_model(*, cloud_session: Any, requested_model: str) -> None:
        actual_model = str(getattr(cloud_session, "model", "") or "").strip()
        expected_model = str(requested_model or "").strip()
        if actual_model and expected_model and actual_model != expected_model:
            raise RuntimeError(
                "Browser Use Cloud returned a different model than requested: "
                f"requested {expected_model!r}, got {actual_model!r}"
            )

    async def _finish_cloud_agent_task(self, task_id: str, *, last_status: str) -> None:
        session_to_close: Any | None = None
        async with self._lock:
            state = self._tasks.get(task_id)
            if state is None:
                return
            status = str(last_status or "").strip().lower()
            handoff_prompt = self._extract_owner_handoff_prompt(state.output)
            if state.stop_requested:
                state.status = "stopped"
                state.is_success = False
                state.output = state.output or "Task stopped by user."
            elif handoff_prompt and state.allow_owner_input:
                state.status = _OWNER_WAITING_STATUS
                state.owner_input_prompt = handoff_prompt
                state.owner_input_type = "text"
                state.owner_input_requested_at = _utc_now_iso()
                state.finished_at = None
                state.updated_monotonic = time.monotonic()
                self._write_profile_metadata(
                    customer_id=state.customer_id,
                    session_id=str(state.session_id or ""),
                    status="waiting_for_owner",
                    task_id=state.task_id,
                    backend="browser-use-cloud-agent",
                    last_url=self._latest_step_url(state),
                )
                return
            elif status in {"error", "timed_out"}:
                state.status = "failed"
                state.is_success = False
                state.error = state.error or f"Browser Use Cloud session ended with status {status}"
            elif status == "stopped":
                state.status = "stopped"
                state.is_success = False if state.is_success is None else state.is_success
            else:
                state.status = "finished"
                if state.is_success is None:
                    state.is_success = True
            state.finished_at = _utc_now_iso()
            state.updated_monotonic = time.monotonic()
            self._write_profile_metadata(
                customer_id=state.customer_id,
                session_id=str(state.session_id or ""),
                status=state.status,
                task_id=state.task_id,
                backend="browser-use-cloud-agent",
                last_url=self._latest_step_url(state),
            )
            if state.close_session_when_done:
                session_to_close = self._detach_session_if_unused_locked(
                    state.customer_id,
                    state.session_id,
                )

        if session_to_close is not None:
            await self._close_session(session_to_close)

    def _compose_cloud_agent_task(
        self,
        *,
        task_text: str,
        start_url: str,
        allowed_domains: list[str],
        allow_owner_input: bool,
    ) -> str:
        parts = []
        if start_url:
            parts.append(f"First navigate to this URL: {start_url}.")
        parts.append(task_text)
        if allowed_domains:
            parts.append(
                "Stay within these domains unless the user explicitly asked otherwise: "
                f"{', '.join(allowed_domains)}."
            )
        if allow_owner_input:
            parts.append(
                "Handoff rule: if login, sign-in, account selection, CAPTCHA, MFA, "
                "email/SMS code, authenticator approval, credentials, payment approval, "
                "or another owner-only step blocks progress, do not fail and do not start "
                "a different browser. Stop at that page and make your final output start "
                "with exactly 'OPENTULPA_OWNER_HANDOFF_REQUIRED:' followed by a concise "
                "instruction telling the owner what to do in the live browser. If a "
                "non-auth decision needs OpenTulpa or the owner to choose a strategy, "
                "make your final output start with exactly 'OPENTULPA_DECISION_REQUIRED:' "
                "followed by the concrete decision needed and the visible browser state. "
                "After confirmation, OpenTulpa will continue in this same Browser Use session."
            )
        return "\n\n".join(parts)

    @staticmethod
    def _compose_cloud_agent_recovery_task(
        *,
        original_task: str,
        repeated_signal: str,
        attempt: int,
    ) -> str:
        return (
            f"{original_task}\n\n"
            "OpenTulpa supervisory guidance: the browser run appears stuck repeating "
            f"the same progress signal for attempt {max(1, int(attempt))}: "
            f"{repeated_signal[:500]}. Continue from the current browser state, but "
            "change strategy now. Do not repeat the same click, scroll, wait, or page "
            "analysis loop. If the page is blocking progress, report the concrete "
            "blocker or request owner/decision handoff using the required handoff format."
        )

    @staticmethod
    def _cloud_agent_progress_signature(cloud_session: Any) -> str:
        summary = str(getattr(cloud_session, "last_step_summary", "") or "").strip()
        if summary:
            return summary[:500]
        step_count = getattr(cloud_session, "step_count", None)
        if isinstance(step_count, int) and step_count > 0:
            return f"step_count:{step_count}"
        return ""

    @staticmethod
    def _browser_use_cloud_model(model_name: str) -> str:
        safe_model = str(model_name or "").strip()
        aliases = {
            "google/gemini-3-flash-preview": "gemini-3-flash",
            "gemini-3-flash-preview": "gemini-3-flash",
        }
        return aliases.get(safe_model, safe_model or "gemini-3-flash")

    async def _cloud_agent_session_id(self, customer_id: str, session_id: str) -> str | None:
        session_state = self._sessions.get(self._session_key(customer_id, session_id))
        if session_state is not None and session_state.cloud_browser_session_id:
            return session_state.cloud_browser_session_id
        if self._user_data_dir is None:
            return None
        metadata = self._read_profile_metadata(self._profile_dir(customer_id, session_id))
        value = str(metadata.get("browserUseBrowserSessionId") or "").strip()
        return value or None

    @staticmethod
    def _stringify_cloud_output(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip() or None
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)[:12000]
        except Exception:
            return str(value)[:12000]

    @staticmethod
    def _extract_owner_handoff_prompt(output: str | None) -> str | None:
        text = str(output or "").strip()
        markers = (
            "OPENTULPA_OWNER_HANDOFF_REQUIRED:",
            "OPENTULPA_DECISION_REQUIRED:",
        )
        for marker in markers:
            if text.startswith(marker):
                prompt = text[len(marker):].strip()
                return (
                    prompt
                    or "Open the live browser link, decide how to continue, then reply with direction."
                )
        return None

    @staticmethod
    def _resolve_user_data_dir(value: str | Path | None) -> Path | None:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _safe_profile_name(session_id: str) -> str:
        value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id or "").strip())
        value = value.strip("._-")
        return value[:80] or "default"

    @classmethod
    def _normalize_customer_id(cls, customer_id: str | None) -> str:
        raw = str(customer_id or "").strip()
        return raw or _DEFAULT_CUSTOMER_ID

    @classmethod
    def _normalize_optional_customer_id(cls, customer_id: str | None) -> str | None:
        raw = str(customer_id or "").strip()
        if not raw:
            return None
        return raw

    @staticmethod
    def _session_key(customer_id: str, session_id: str) -> str:
        customer = BrowserUseLocalManager._normalize_customer_id(customer_id)
        session = BrowserUseLocalManager._safe_profile_name(session_id)
        return f"{customer}\0{session}"

    @classmethod
    def _profile_customer_dir_name(cls, customer_id: str) -> str:
        raw = cls._normalize_customer_id(customer_id)
        safe = cls._safe_profile_name(raw)
        if safe == raw:
            return safe
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
        return f"{safe}-{digest}"

    def _profile_dir(self, customer_id: str, session_id: str) -> Path:
        assert self._user_data_dir is not None
        return (
            self._user_data_dir
            / self._profile_customer_dir_name(customer_id)
            / self._safe_profile_name(session_id)
        )

    def _profile_dir_exists(self, customer_id: str, session_id: str) -> bool:
        if self._user_data_dir is None:
            return False
        return self._profile_dir(customer_id, session_id).is_dir()

    def _persisted_profile_dirs(self, customer_id: str) -> list[tuple[str, Path, dict[str, Any]]]:
        if self._user_data_dir is None or not self._user_data_dir.is_dir():
            return []
        customer_dir = self._user_data_dir / self._profile_customer_dir_name(customer_id)
        if not customer_dir.is_dir():
            return []
        out: list[tuple[str, Path, dict[str, Any]]] = []
        for child in customer_dir.iterdir():
            if not child.is_dir():
                continue
            session_id = child.name.strip()
            if not session_id:
                continue
            metadata = self._read_profile_metadata(child)
            out.append((session_id, child, metadata))
        out.sort(key=lambda item: self._profile_last_used_timestamp(item[1], item[2]), reverse=True)
        return out

    @staticmethod
    def _read_profile_metadata(profile_dir: Path) -> dict[str, Any]:
        metadata_path = profile_dir / _PROFILE_METADATA_FILE
        if not metadata_path.exists():
            return {}
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _write_profile_metadata(
        self,
        *,
        customer_id: str,
        session_id: str,
        status: str,
        task_id: str | None = None,
        last_url: str | None = None,
        backend: str | None = None,
        cloud_profile_id: str | None = None,
        cloud_browser_session_id: str | None = None,
        cloud_model: str | None = None,
        live_url: str | None = None,
        recording_url: str | None = None,
        clear_live_session: bool = False,
    ) -> None:
        if self._user_data_dir is None:
            return
        profile_dir = self._profile_dir(customer_id, session_id)
        profile_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._read_profile_metadata(profile_dir)
        now = _utc_now_iso()
        metadata.update(
            {
                "customerId": self._safe_profile_name(customer_id),
                "rawCustomerId": self._normalize_customer_id(customer_id),
                "profileId": self._safe_profile_name(session_id),
                "label": metadata.get("label") or self._safe_profile_name(session_id),
                "createdAt": metadata.get("createdAt") or now,
                "lastUsedAt": now,
                "status": status,
            }
        )
        if task_id:
            metadata["lastTaskId"] = task_id
        if last_url:
            metadata["lastUrl"] = last_url
        if backend:
            metadata["backend"] = backend
        if cloud_profile_id:
            metadata["browserUseProfileId"] = cloud_profile_id
        if cloud_browser_session_id:
            metadata["browserUseBrowserSessionId"] = cloud_browser_session_id
        if cloud_model:
            metadata["browserUseModel"] = cloud_model
        if live_url:
            metadata["liveUrl"] = live_url
        if recording_url:
            metadata["recordingUrl"] = recording_url
        if clear_live_session:
            metadata.pop("browserUseBrowserSessionId", None)
            metadata.pop("liveUrl", None)
        (profile_dir / _PROFILE_METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _profile_last_used_timestamp(profile_dir: Path, metadata: dict[str, Any]) -> float:
        raw = str(metadata.get("lastUsedAt", "") or "").strip()
        if raw:
            with suppress(Exception):
                return datetime.fromisoformat(raw).timestamp()
        return profile_dir.stat().st_mtime

    def _new_controller(self, *, task_id: str, allow_owner_input: bool = True) -> Any | None:
        from browser_use import ActionResult, Controller

        controller: Any = Controller()

        if allow_owner_input:
            @controller.action(
                "Hand off the current live browser session to the OpenTulpa owner and wait for them. "
                "Use this immediately when login, sign-in, CAPTCHA, MFA, credentials, email/SMS code, "
                "authenticator approval, account choice, or another owner-only step blocks progress. "
                "Ask the owner to complete the step in the same live browser session, then reply when done.",
                domains=["*"],
            )
            async def request_owner_input(prompt: str, input_type: str = "text") -> ActionResult:
                try:
                    owner_value = await self.request_owner_input(
                        task_id=task_id,
                        prompt=prompt,
                        input_type=input_type,
                    )
                except TimeoutError:
                    return ActionResult(
                        success=False,
                        error="Owner input timed out after 24 hours.",
                    )
                return ActionResult(
                    extracted_content=(
                        "Owner provided input for the current browser challenge. "
                        f"Input type: {input_type}. Value: {owner_value}"
                    ),
                    include_extracted_content_only_once=True,
                )

        if not self._capsolver_api_key:
            return controller
        from opentulpa.integrations.browser_use_captcha import register_capsolver_action
        from opentulpa.integrations.capsolver import CapSolverClient

        return register_capsolver_action(
            controller,
            CapSolverClient(api_key=self._capsolver_api_key),
        )

    @staticmethod
    def _import_browser_use_components() -> tuple[Any, Any, Any]:
        from browser_use import Agent, ChatOpenAI
        from browser_use.browser import BrowserSession

        return Agent, ChatOpenAI, BrowserSession

    def _state_to_payload(self, state: _BrowserUseTaskState) -> dict[str, Any]:
        session_state = self._sessions.get(
            self._session_key(state.customer_id, str(state.session_id or ""))
        )
        metadata = (
            self._read_profile_metadata(
                self._profile_dir(state.customer_id, str(state.session_id or ""))
            )
            if self._user_data_dir is not None
            else {}
        )
        backend = (
            session_state.backend
            if session_state is not None
            else str(metadata.get("backend") or "local")
        )
        live_url = session_state.live_url if session_state is not None else metadata.get("liveUrl")
        recording_url = (
            session_state.recording_url if session_state is not None else metadata.get("recordingUrl")
        )
        cloud_profile_id = (
            session_state.cloud_profile_id
            if session_state is not None
            else metadata.get("browserUseProfileId")
        )
        cloud_browser_session_id = (
            session_state.cloud_browser_session_id
            if session_state is not None
            else metadata.get("browserUseBrowserSessionId")
        )
        cloud_model = (
            session_state.cloud_model if session_state is not None else metadata.get("browserUseModel")
        )
        return {
            "id": state.task_id,
            "customerId": state.customer_id,
            "sessionId": state.session_id,
            "backend": backend,
            "liveUrl": live_url or None,
            "recordingUrl": recording_url or None,
            "browserUseProfileId": cloud_profile_id or None,
            "browserUseBrowserSessionId": cloud_browser_session_id or None,
            "browserUseModel": cloud_model or None,
            "status": state.status,
            "isSuccess": state.is_success,
            "startedAt": state.started_at,
            "finishedAt": state.finished_at,
            "task": state.task,
            "llm": state.llm,
            "output": state.output,
            "outputFiles": state.output_files,
            "steps": state.steps,
            "error": state.error,
            "ownerInputPrompt": state.owner_input_prompt,
            "ownerInputType": state.owner_input_type,
            "ownerInputRequestedAt": state.owner_input_requested_at,
        }

    async def _close_session(self, session: Any) -> None:
        if session is None:
            return
        if isinstance(session, str):
            with suppress(Exception):
                await self._get_browser_use_cloud_client().stop_agent_session(
                    session,
                    strategy="session",
                )
            return
        cloud_session_id = self._cloud_session_ids_by_browser_session.pop(id(session), "")
        if cloud_session_id:
            with suppress(Exception):
                await self._get_browser_use_cloud_client().stop_browser_session(cloud_session_id)
        if hasattr(session, "stop"):
            with suppress(Exception):
                await session.stop()
                return
        if hasattr(session, "kill"):
            with suppress(Exception):
                await session.kill()

    def _ensure_cleanup_task_locked(self) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="browser_use_local_cleanup",
        )

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(_SESSION_CLEANUP_POLL_SECONDS)
                async with self._lock:
                    self._cleanup_locked()
        except asyncio.CancelledError:
            raise

    def _cleanup_locked(self) -> None:
        now = time.monotonic()
        expired: list[str] = []
        for task_id, state in self._tasks.items():
            if state.status not in _TERMINAL_STATUSES:
                continue
            age = now - float(state.updated_monotonic or state.created_monotonic)
            if age >= self._task_retention_seconds:
                expired.append(task_id)
        for task_id in expired:
            self._tasks.pop(task_id, None)

        for state in self._tasks.values():
            if state.status != _OWNER_WAITING_STATUS:
                continue
            session_state = self._sessions.get(
                self._session_key(state.customer_id, str(state.session_id or ""))
            )
            if session_state is None or not self._session_state_is_cloud(session_state):
                continue
            age = now - float(state.updated_monotonic or state.created_monotonic)
            if age < _CLOUD_SESSION_IDLE_TIMEOUT_SECONDS:
                continue
            state.status = "stopped"
            state.is_success = False
            state.finished_at = _utc_now_iso()
            state.updated_monotonic = now
            state.output = (
                "Browser session stopped after owner login inactivity. "
                "Run browser_use_run again with the same session_id to continue "
                "from the persisted Browser Use profile."
            )
            if state.owner_input_future is not None and not state.owner_input_future.done():
                state.owner_input_future.cancel()
            self._write_profile_metadata(
                customer_id=state.customer_id,
                session_id=str(state.session_id or ""),
                status="idle",
                task_id=state.task_id,
                last_url=self._latest_step_url(state),
            )
            detached = self._detach_session_if_unused_locked(state.customer_id, state.session_id)
            if detached is not None:
                asyncio.create_task(self._close_session(detached))

        expired_sessions: list[str] = []
        for session_key, session_state in self._sessions.items():
            if self._session_has_active_tasks_locked(session_state.customer_id, session_state.session_id):
                continue
            age = now - float(session_state.updated_monotonic or now)
            if age >= _SESSION_IDLE_TIMEOUT_SECONDS:
                expired_sessions.append(session_key)
        for session_key in expired_sessions:
            if session_key not in self._sessions:
                continue
            session_state = self._sessions.pop(session_key)
            self._write_profile_metadata(
                customer_id=session_state.customer_id,
                session_id=session_state.session_id,
                status="idle",
                clear_live_session=True,
            )
            asyncio.create_task(self._close_session(session_state.session))

        self._delete_stale_profiles_locked()

    def _delete_stale_profiles_locked(self) -> None:
        if self._user_data_dir is None or not self._user_data_dir.is_dir():
            return
        cutoff = time.time() - _PROFILE_RETENTION_SECONDS
        live_profile_dirs = {
            self._profile_dir(item.customer_id, item.session_id).resolve()
            for item in self._sessions.values()
        }
        for customer_dir in self._user_data_dir.iterdir():
            if not customer_dir.is_dir():
                continue
            for profile_dir in customer_dir.iterdir():
                if not profile_dir.is_dir():
                    continue
                if profile_dir.resolve() in live_profile_dirs:
                    continue
                metadata = self._read_profile_metadata(profile_dir)
                if self._profile_last_used_timestamp(profile_dir, metadata) >= cutoff:
                    continue
                with suppress(Exception):
                    shutil.rmtree(profile_dir)

    def _detach_session_if_unused_locked(
        self, customer_id: str, session_id: str | None
    ) -> Any | None:
        safe_customer = self._normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return None
        for state in self._tasks.values():
            if state.customer_id != safe_customer:
                continue
            if str(state.session_id or "").strip() != safe_session:
                continue
            if state.status not in _TERMINAL_STATUSES:
                return None
        session_state = self._sessions.pop(self._session_key(safe_customer, safe_session), None)
        self._write_profile_metadata(
            customer_id=safe_customer,
            session_id=safe_session,
            status="idle",
            clear_live_session=True,
        )
        return session_state.session if session_state is not None else None

    def _touch_session_locked(self, customer_id: str, session_id: str | None) -> None:
        safe_customer = self._normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return
        session_state = self._sessions.get(self._session_key(safe_customer, safe_session))
        if session_state is not None:
            session_state.updated_monotonic = time.monotonic()
        self._write_profile_metadata(
            customer_id=safe_customer,
            session_id=safe_session,
            status="running" if self._session_has_active_tasks_locked(safe_customer, safe_session) else "idle",
        )

    def _pick_reusable_session_id_locked(self, customer_id: str) -> str | None:
        safe_customer = self._normalize_customer_id(customer_id)
        reusable: list[tuple[float, str]] = []
        for _, session_state in self._sessions.items():
            if session_state.customer_id != safe_customer:
                continue
            if self._session_has_active_tasks_locked(session_state.customer_id, session_state.session_id):
                continue
            reusable.append((float(session_state.updated_monotonic or 0.0), session_state.session_id))
        if not reusable:
            return None
        reusable.sort(reverse=True)
        return reusable[0][1]

    def _live_session_count_for_customer_locked(self, customer_id: str) -> int:
        safe_customer = self._normalize_customer_id(customer_id)
        return sum(1 for item in self._sessions.values() if item.customer_id == safe_customer)

    def _session_summaries_locked(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        safe_customer = self._normalize_customer_id(customer_id)
        out: list[dict[str, Any]] = []
        for _, session_state in self._sessions.items():
            if session_state.customer_id != safe_customer:
                continue
            active_task = self._active_task_for_session_locked(
                session_state.customer_id, session_state.session_id
            )
            out.append(
                {
                    "session_id": session_state.session_id,
                    "customer_id": session_state.customer_id,
                    "backend": session_state.backend,
                    "reusable": active_task is None,
                    "active_task_id": active_task.task_id if active_task is not None else None,
                    "live_url": session_state.live_url,
                    "recording_url": session_state.recording_url,
                    "cloud_profile_id": session_state.cloud_profile_id,
                    "cloud_browser_session_id": session_state.cloud_browser_session_id,
                    "last_used_monotonic": float(session_state.updated_monotonic or 0.0),
                }
            )
        out.sort(key=lambda item: item["last_used_monotonic"], reverse=True)
        return out

    def _browser_use_cloud_enabled(self) -> bool:
        return bool(self._browser_use_api_key)

    def _get_browser_use_cloud_client(self) -> Any:
        if self._browser_use_cloud_client is None:
            from opentulpa.integrations.browser_use_cloud import BrowserUseCloudClient

            self._browser_use_cloud_client = BrowserUseCloudClient(
                api_key=self._browser_use_api_key,
                proxy_country_code=self._browser_use_cloud_proxy_country_code or None,
                browser_timeout_minutes=self._browser_use_cloud_timeout_minutes,
            )
        return self._browser_use_cloud_client

    def _browser_use_cloud_profile_id(self, customer_id: str, session_id: str) -> str | None:
        if self._user_data_dir is None:
            return None
        metadata = self._read_profile_metadata(self._profile_dir(customer_id, session_id))
        value = str(metadata.get("browserUseProfileId") or "").strip()
        return value or None

    @classmethod
    def _browser_use_cloud_profile_name(cls, customer_id: str, session_id: str) -> str:
        customer = cls._normalize_customer_id(customer_id)
        session = cls._safe_profile_name(session_id)
        return f"opentulpa-{customer}-{session}"[:100]

    def _session_has_active_tasks_locked(self, customer_id: str, session_id: str) -> bool:
        safe_customer = self._normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return False
        for state in self._tasks.values():
            if state.customer_id != safe_customer:
                continue
            if str(state.session_id or "").strip() != safe_session:
                continue
            if state.status not in _TERMINAL_STATUSES:
                return True
        return False

    def _session_is_cloud_locked(self, customer_id: str, session_id: str | None) -> bool:
        safe_customer = self._normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return False
        session_state = self._sessions.get(self._session_key(safe_customer, safe_session))
        return self._session_state_is_cloud(session_state)

    @staticmethod
    def _session_state_is_cloud(session_state: _BrowserUseSessionState | None) -> bool:
        return bool(
            session_state is not None
            and (
                session_state.backend in {"browser-use-cloud", "browser-use-cloud-agent"}
                or session_state.cloud_browser_session_id
            )
        )

    def _active_task_for_session_locked(
        self, customer_id: str, session_id: str
    ) -> _BrowserUseTaskState | None:
        safe_customer = self._normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return None
        active: list[_BrowserUseTaskState] = []
        for state in self._tasks.values():
            if state.customer_id != safe_customer:
                continue
            if str(state.session_id or "").strip() != safe_session:
                continue
            if state.status in _TERMINAL_STATUSES:
                continue
            active.append(state)
        if not active:
            return None
        active.sort(key=lambda item: float(item.updated_monotonic or item.created_monotonic), reverse=True)
        return active[0]
