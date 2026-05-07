"""Local Browser Use task manager with headless-first defaults."""

from __future__ import annotations

import asyncio
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
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_tasks)))
        self._lock = asyncio.Lock()
        self._tasks: dict[str, _BrowserUseTaskState] = {}
        self._sessions: dict[str, _BrowserUseSessionState] = {}
        self._preflight_checked = False
        self._preflight_error: str | None = None
        self._cleanup_task: asyncio.Task[Any] | None = None

    async def preflight(self) -> str | None:
        if self._preflight_checked:
            return self._preflight_error
        self._preflight_checked = True

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
        if not self._openrouter_api_key:
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
        safe_session_id = self._safe_profile_name(explicit_session_id) if explicit_session_id else _DEFAULT_SESSION_ID
        safe_customer_id = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
        session_key = self._session_key(safe_customer_id, safe_session_id)
        safe_max_steps = max(1, min(int(max_steps), 120))
        safe_start_url = str(start_url or "").strip()
        safe_domains = self._sanitize_domains(allowed_domains)

        session_to_close: Any | None = None
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
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
                browser_session = self._new_browser_session(
                    allowed_domains=safe_domains,
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
                )
                session_state = _BrowserUseSessionState(
                    session=browser_session,
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
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

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return None
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return None
            self._touch_session_locked(state.customer_id, state.session_id)
            return self._state_to_payload(state)

    async def list_sessions(self, *, customer_id: str | None = None) -> list[dict[str, Any]]:
        safe_customer_id = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
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
                        "reusable": not active_task_ids,
                        "persisted": self._profile_dir_exists(safe_customer_id, session_state.session_id),
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
                        "reusable": (
                            self._live_session_count_for_customer_locked(safe_customer_id)
                            < _MAX_BROWSER_USE_SESSIONS
                        ),
                        "persisted": True,
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

    async def control_task(self, *, task_id: str, action: str) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        safe_action = str(action or "").strip().lower()
        if not safe_task_id:
            return {"error": "browser_use_task_control requires task_id"}

        session_to_close: Any | None = None
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
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

        if session_to_close is not None:
            await self._close_session(session_to_close)
        return payload

    async def capture_screenshot(
        self,
        *,
        task_id: str,
        full_page: bool = True,
    ) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_screenshot requires task_id"}

        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return {"error": f"browser_use_task_screenshot task not found: {safe_task_id}"}
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
            if state is not None:
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
    ) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        safe_owner_input = str(owner_input or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_owner_input_submit requires task_id"}
        if not safe_owner_input:
            return {"error": "browser_use_owner_input_submit requires owner_input"}

        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return {"error": f"browser_use_owner_input_submit task not found: {safe_task_id}"}
            if state.status != _OWNER_WAITING_STATUS:
                return {
                    "error": (
                        "browser_use_owner_input_submit requires a task waiting for owner input; "
                        f"current status is {state.status}"
                    )
                }
            future = state.owner_input_future
            if future is None or future.done():
                return {"error": "browser_use_owner_input_submit has no pending owner input request"}
            future.set_result(safe_owner_input)
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

            session_to_close: Any | None = None
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

                if state.close_session_when_done:
                    session_to_close = self._detach_session_if_unused_locked(
                        state.customer_id, state.session_id
                    )

            if session_to_close is not None:
                await self._close_session(session_to_close)
        except Exception as exc:
            session_to_close: Any | None = None
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
                    if state.close_session_when_done:
                        session_to_close = self._detach_session_if_unused_locked(
                            state.customer_id, state.session_id
                        )
            if session_to_close is not None:
                await self._close_session(session_to_close)
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

    def _new_browser_session(
        self,
        *,
        allowed_domains: list[str],
        customer_id: str,
        session_id: str,
    ) -> Any:
        _, _, browser_session_cls = self._import_browser_use_components()
        session_kwargs: dict[str, Any] = {"headless": self._headless, "keep_alive": True}
        if allowed_domains:
            session_kwargs["allowed_domains"] = allowed_domains
        if self._user_data_dir is not None:
            session_profile_dir = self._profile_dir(customer_id, session_id)
            session_profile_dir.mkdir(parents=True, exist_ok=True)
            session_kwargs["user_data_dir"] = str(session_profile_dir)
        return browser_session_cls(**session_kwargs)

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

    @staticmethod
    def _session_key(customer_id: str, session_id: str) -> str:
        return f"{BrowserUseLocalManager._safe_profile_name(customer_id)}/{BrowserUseLocalManager._safe_profile_name(session_id)}"

    def _profile_dir(self, customer_id: str, session_id: str) -> Path:
        assert self._user_data_dir is not None
        return (
            self._user_data_dir
            / self._safe_profile_name(customer_id)
            / self._safe_profile_name(session_id)
        )

    def _profile_dir_exists(self, customer_id: str, session_id: str) -> bool:
        if self._user_data_dir is None:
            return False
        return self._profile_dir(customer_id, session_id).is_dir()

    def _persisted_profile_dirs(self, customer_id: str) -> list[tuple[str, Path, dict[str, Any]]]:
        if self._user_data_dir is None or not self._user_data_dir.is_dir():
            return []
        customer_dir = self._user_data_dir / self._safe_profile_name(customer_id)
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

        controller = Controller()

        if allow_owner_input:
            @controller.action(
                "Ask the OpenTulpa owner for input needed to continue the current browser task. "
                "Use when login or verification is blocked by an email code, SMS code, authenticator code, "
                "MFA approval, account choice, or owner-only decision. Keep the current browser page open.",
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
        return {
            "id": state.task_id,
            "customerId": state.customer_id,
            "sessionId": state.session_id,
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

        expired_sessions: list[str] = []
        for session_key, session_state in self._sessions.items():
            if self._session_has_active_tasks_locked(session_state.customer_id, session_state.session_id):
                continue
            age = now - float(session_state.updated_monotonic or now)
            if age >= _SESSION_IDLE_TIMEOUT_SECONDS:
                expired_sessions.append(session_key)
        for session_key in expired_sessions:
            session_state = self._sessions.pop(session_key, None)
            if session_state is not None:
                self._write_profile_metadata(
                    customer_id=session_state.customer_id,
                    session_id=session_state.session_id,
                    status="idle",
                )
                asyncio.create_task(self._close_session(session_state.session))

        self._delete_stale_profiles_locked()

    def _delete_stale_profiles_locked(self) -> None:
        if self._user_data_dir is None or not self._user_data_dir.is_dir():
            return
        cutoff = time.time() - _PROFILE_RETENTION_SECONDS
        live_keys = {
            self._session_key(item.customer_id, item.session_id)
            for item in self._sessions.values()
        }
        for customer_dir in self._user_data_dir.iterdir():
            if not customer_dir.is_dir():
                continue
            customer_id = customer_dir.name
            for profile_dir in customer_dir.iterdir():
                if not profile_dir.is_dir():
                    continue
                session_id = profile_dir.name
                if self._session_key(customer_id, session_id) in live_keys:
                    continue
                metadata = self._read_profile_metadata(profile_dir)
                if self._profile_last_used_timestamp(profile_dir, metadata) >= cutoff:
                    continue
                with suppress(Exception):
                    shutil.rmtree(profile_dir)

    def _detach_session_if_unused_locked(
        self, customer_id: str, session_id: str | None
    ) -> Any | None:
        safe_customer = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
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
        )
        return session_state.session if session_state is not None else None

    def _touch_session_locked(self, customer_id: str, session_id: str | None) -> None:
        safe_customer = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
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

    def _pick_reusable_session_id_locked(self) -> str | None:
        reusable: list[tuple[float, str]] = []
        for _, session_state in self._sessions.items():
            if self._session_has_active_tasks_locked(session_state.customer_id, session_state.session_id):
                continue
            reusable.append((float(session_state.updated_monotonic or 0.0), session_state.session_id))
        if not reusable:
            return None
        reusable.sort(reverse=True)
        return reusable[0][1]

    def _live_session_count_for_customer_locked(self, customer_id: str) -> int:
        safe_customer = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
        return sum(1 for item in self._sessions.values() if item.customer_id == safe_customer)

    def _session_summaries_locked(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        safe_customer = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
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
                    "reusable": active_task is None,
                    "active_task_id": active_task.task_id if active_task is not None else None,
                    "last_used_monotonic": float(session_state.updated_monotonic or 0.0),
                }
            )
        out.sort(key=lambda item: item["last_used_monotonic"], reverse=True)
        return out

    def _session_has_active_tasks_locked(self, customer_id: str, session_id: str) -> bool:
        safe_customer = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
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

    def _active_task_for_session_locked(
        self, customer_id: str, session_id: str
    ) -> _BrowserUseTaskState | None:
        safe_customer = self._safe_profile_name(customer_id or _DEFAULT_CUSTOMER_ID)
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
