"""Local Browser Use task manager with headless-first defaults."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opentulpa.core.ids import new_short_id

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = {"finished", "stopped", "failed"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class _BrowserUseTaskState:
    task_id: str
    session_id: str | None
    task: str
    llm: str
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
    stop_requested: bool = False
    close_session_when_done: bool = False


class BrowserUseLocalManager:
    """Manage local Browser Use runs with in-memory task/session state."""

    def __init__(
        self,
        *,
        openrouter_api_key: str,
        openrouter_base_url: str,
        default_model: str,
        model_override: str | None = None,
        headless: bool = True,
        max_concurrent_tasks: int = 2,
        task_retention_seconds: int = 1800,
    ) -> None:
        self._openrouter_api_key = str(openrouter_api_key or "").strip()
        self._openrouter_base_url = str(openrouter_base_url or "").strip().rstrip("/")
        self._default_model = str(default_model or "").strip()
        self._model_override = str(model_override or "").strip()
        self._headless = bool(headless)
        self._task_retention_seconds = max(60, int(task_retention_seconds))
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_tasks)))
        self._lock = asyncio.Lock()
        self._tasks: dict[str, _BrowserUseTaskState] = {}
        self._sessions: dict[str, Any] = {}
        self._preflight_checked = False
        self._preflight_error: str | None = None

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
    ) -> dict[str, Any]:
        preflight_error = await self.preflight()
        if preflight_error:
            return {"error": preflight_error}
        if not self._openrouter_api_key:
            return {"error": "browser_use_run unavailable: OPENROUTER_API_KEY missing"}

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

        safe_session_id = str(session_id or "").strip() or new_short_id("bses")
        safe_max_steps = max(1, min(int(max_steps), 120))
        safe_start_url = str(start_url or "").strip()
        safe_domains = self._sanitize_domains(allowed_domains)

        async with self._lock:
            self._cleanup_locked()
            browser_session = self._sessions.get(safe_session_id)
            if browser_session is None:
                browser_session = self._new_browser_session(
                    allowed_domains=safe_domains,
                )
                self._sessions[safe_session_id] = browser_session

            task_id = new_short_id("task")
            state = _BrowserUseTaskState(
                task_id=task_id,
                session_id=safe_session_id,
                task=task_text,
                llm=resolved_model,
                status="queued",
                browser_session=browser_session,
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
            return self._state_to_payload(state)

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return None
        async with self._lock:
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return None
            return self._state_to_payload(state)

    async def control_task(self, *, task_id: str, action: str) -> dict[str, Any]:
        safe_task_id = str(task_id or "").strip()
        safe_action = str(action or "").strip().lower()
        if not safe_task_id:
            return {"error": "browser_use_task_control requires task_id"}

        session_to_close: Any | None = None
        async with self._lock:
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
            elif safe_action == "resume":
                if agent is not None and hasattr(agent, "resume"):
                    with suppress(Exception):
                        agent.resume()
                if state.status in {"paused", "queued"}:
                    state.status = "running"
                    state.updated_monotonic = time.monotonic()
            elif safe_action in {"stop", "stop_task_and_session"}:
                state.stop_requested = True
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
                        session_to_close = self._detach_session_if_unused_locked(state.session_id)
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

    async def shutdown(self) -> None:
        async with self._lock:
            task_states = list(self._tasks.values())
            sessions = list(self._sessions.values())
            self._tasks.clear()
            self._sessions.clear()

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
                model_name = state.llm
                task_text = state.task
                browser_session = state.browser_session

            llm = chat_openai_cls(
                model=model_name,
                api_key=self._openrouter_api_key,
                base_url=self._openrouter_base_url,
            )
            composed_task = task_text
            if start_url:
                composed_task = (
                    f"First navigate to this URL: {start_url}. "
                    f"Then complete this task: {task_text}"
                )

            agent = agent_cls(
                task=composed_task,
                llm=llm,
                browser_session=browser_session,
                register_new_step_callback=self._step_callback(task_id),
                directly_open_url=True,
            )
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

                if state.close_session_when_done:
                    session_to_close = self._detach_session_if_unused_locked(state.session_id)

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
                    if state.close_session_when_done:
                        session_to_close = self._detach_session_if_unused_locked(state.session_id)
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

    def _new_browser_session(self, *, allowed_domains: list[str]) -> Any:
        _, _, browser_session_cls = self._import_browser_use_components()
        session_kwargs: dict[str, Any] = {"headless": self._headless}
        if allowed_domains:
            session_kwargs["allowed_domains"] = allowed_domains
        return browser_session_cls(**session_kwargs)

    @staticmethod
    def _import_browser_use_components() -> tuple[Any, Any, Any]:
        from browser_use import Agent, ChatOpenAI
        from browser_use.browser import BrowserSession

        return Agent, ChatOpenAI, BrowserSession

    def _state_to_payload(self, state: _BrowserUseTaskState) -> dict[str, Any]:
        return {
            "id": state.task_id,
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

    def _detach_session_if_unused_locked(self, session_id: str | None) -> Any | None:
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return None
        for state in self._tasks.values():
            if str(state.session_id or "").strip() != safe_session:
                continue
            if state.status not in _TERMINAL_STATUSES:
                return None
        return self._sessions.pop(safe_session, None)
