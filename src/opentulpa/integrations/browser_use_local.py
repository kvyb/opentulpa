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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from opentulpa.core.ids import new_short_id
from opentulpa.integrations.browser_use_session_registry import (
    TERMINAL_STATUSES as _TERMINAL_STATUSES,
)
from opentulpa.integrations.browser_use_session_registry import (
    BrowserUseSessionRegistry,
)
from opentulpa.integrations.browser_use_session_registry import (
    BrowserUseSessionState as _BrowserUseSessionState,
)
from opentulpa.integrations.browser_use_session_registry import (
    BrowserUseTaskState as _BrowserUseTaskState,
)
from opentulpa.integrations.browser_use_session_registry import (
    normalize_customer_id as _normalize_customer_id,
)
from opentulpa.integrations.browser_use_session_registry import (
    normalize_optional_customer_id as _normalize_optional_customer_id,
)
from opentulpa.integrations.browser_use_session_registry import (
    safe_profile_name as _safe_profile_name,
)
from opentulpa.integrations.browser_use_session_registry import (
    session_key as _browser_session_key,
)
from opentulpa.tasks.sandbox import TULPA_STUFF_DIR

logger = logging.getLogger(__name__)

_OWNER_WAITING_STATUS = "waiting_for_owner"
_OWNER_INPUT_TIMEOUT_SECONDS = 24 * 60 * 60
_SESSION_IDLE_TIMEOUT_SECONDS = 3600
_SESSION_CLEANUP_POLL_SECONDS = 60.0
_MAX_BROWSER_USE_SESSIONS = 20
_SESSION_CLOSE_TIMEOUT_SECONDS = 30
_DEFAULT_SESSION_ID = "default"
_PROFILE_RETENTION_SECONDS = 14 * 24 * 60 * 60
_PROFILE_METADATA_FILE = "profile.json"
_DEFAULT_BROWSER_MODEL = "google/gemini-3-flash-preview"
_DISALLOWED_BROWSER_MODEL_ALIASES = {"browser-use-llm", "bu-mini", "default"}
_COLLECT_IMAGE_RESOURCES_SCRIPT = """() => {
  const currentUrl = String(window.location.href || '');
  const absolutize = (rawUrl) => {
    const value = String(rawUrl || '').trim();
    if (!value || value.startsWith('data:') || value.startsWith('blob:')) return '';
    try {
      return new URL(value, document.baseURI || currentUrl).href;
    } catch (_) {
      return '';
    }
  };
  const pushCandidate = (items, candidate) => {
    const url = absolutize(candidate.url);
    if (!url) return;
    items.push({ ...candidate, url });
  };
  const parseSrcset = (srcset) => String(srcset || '')
    .split(',')
    .map((part) => part.trim().split(/\\s+/, 1)[0])
    .filter(Boolean);
  const parseBackgroundUrls = (value) => {
    const urls = [];
    const text = String(value || '');
    const regex = /url\\((['"]?)(.*?)\\1\\)/g;
    let match;
    while ((match = regex.exec(text)) && urls.length < 20) {
      urls.push(match[2]);
    }
    return urls;
  };

  const imageCandidates = [];
  for (const img of Array.from(document.images).slice(0, 80)) {
    pushCandidate(imageCandidates, {
      source: 'img_src',
      url: img.currentSrc || img.src || img.getAttribute('src'),
      alt: img.alt || '',
      title: img.title || '',
      width: img.width || 0,
      height: img.height || 0,
      natural_width: img.naturalWidth || 0,
      natural_height: img.naturalHeight || 0
    });
    for (const srcsetUrl of parseSrcset(img.getAttribute('srcset')).slice(0, 5)) {
      pushCandidate(imageCandidates, {
        source: 'img_srcset',
        url: srcsetUrl,
        alt: img.alt || '',
        title: img.title || '',
        width: img.width || 0,
        height: img.height || 0,
        natural_width: img.naturalWidth || 0,
        natural_height: img.naturalHeight || 0
      });
    }
  }
  for (const source of Array.from(document.querySelectorAll('picture source[srcset]')).slice(0, 40)) {
    for (const srcsetUrl of parseSrcset(source.getAttribute('srcset')).slice(0, 5)) {
      pushCandidate(imageCandidates, { source: 'picture_source_srcset', url: srcsetUrl });
    }
  }
  for (const element of Array.from(document.querySelectorAll('[style]')).slice(0, 120)) {
    for (const backgroundUrl of parseBackgroundUrls(element.style.backgroundImage).slice(0, 3)) {
      pushCandidate(imageCandidates, {
        source: 'css_background',
        url: backgroundUrl,
        title: element.getAttribute('aria-label') || element.getAttribute('title') || ''
      });
    }
  }

  const networkImageResources = performance.getEntriesByType('resource')
    .filter((entry) => {
      const initiator = String(entry.initiatorType || '').toLowerCase();
      const name = String(entry.name || '');
      return initiator === 'img'
        || initiator === 'image'
        || /\\.(avif|gif|jpe?g|png|svg|webp)([?#].*)?$/i.test(name)
        || /\\/images?\\//i.test(name);
    })
    .slice(-80)
    .map((entry) => ({
      source: 'network_resource',
      url: entry.name,
      initiator_type: entry.initiatorType || '',
      transfer_size: Math.max(0, Math.round(entry.transferSize || 0)),
      decoded_body_size: Math.max(0, Math.round(entry.decodedBodySize || 0))
    }));

  return { imageCandidates, networkImageResources };
}"""


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


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
        self._cloud_session_ids_by_browser_session: dict[int, str] = {}
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent_tasks)))
        self._lock = asyncio.Lock()
        self._registry = BrowserUseSessionRegistry()
        self._tasks = self._registry.tasks
        self._sessions = self._registry.sessions
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

        if self._browser_use_cloud_enabled():
            if self._user_data_dir is None:
                self._preflight_error = (
                    "browser_use cloud browser backend unavailable: "
                    "BROWSER_USE_USER_DATA_DIR is required to remember profile ids"
                )
                return self._preflight_error
            try:
                self._get_browser_use_cloud_client()
            except Exception as exc:
                self._preflight_error = (
                    "browser_use cloud browser backend unavailable: "
                    f"Browser Use Cloud SDK import failed ({exc}). Install dependencies with `uv sync`."
                )
                return self._preflight_error
            self._preflight_error = None
            return None

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
                )
                if session_state.cloud_browser_session_id:
                    self._cloud_session_ids_by_browser_session[id(browser_session)] = (
                        session_state.cloud_browser_session_id
                    )
                self._registry.set_session(session_state)
                self._write_profile_metadata(
                    customer_id=safe_customer_id,
                    session_id=safe_session_id,
                    status="idle",
                    backend=session_state.backend,
                    cloud_profile_id=session_state.cloud_profile_id,
                    cloud_browser_session_id=session_state.cloud_browser_session_id,
                    live_url=session_state.live_url,
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
            self._registry.set_task(state)
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
                        "cloud_profile_id": session_state.cloud_profile_id,
                        "cloud_browser_session_id": session_state.cloud_browser_session_id,
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
                        "cloud_profile_id": metadata.get("browserUseProfileId") or None,
                        "cloud_browser_session_id": metadata.get("browserUseBrowserSessionId") or None,
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
        async with self._lock:
            self._ensure_cleanup_task_locked()
            self._cleanup_locked()
            state = self._tasks.get(safe_task_id)
            if state is None:
                return {"error": f"browser_use_task_control task not found: {safe_task_id}"}
            if safe_customer_id is not None and state.customer_id != safe_customer_id:
                return {"error": f"browser_use_task_control task not found: {safe_task_id}"}

            if safe_action == "pause":
                if state.status == "running":
                    state.status = "paused"
                    state.updated_monotonic = time.monotonic()
                    self._touch_session_locked(state.customer_id, state.session_id)
            elif safe_action == "resume":
                if state.status in {"paused", "queued"}:
                    state.status = "running"
                    state.updated_monotonic = time.monotonic()
                    self._touch_session_locked(state.customer_id, state.session_id)
            elif safe_action in {"stop", "stop_task_and_session"}:
                state.stop_requested = True
                if state.owner_input_future is not None and not state.owner_input_future.done():
                    state.owner_input_future.cancel()
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
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                state.status = "running"
                state.started_at = state.started_at or _utc_now_iso()
                state.updated_monotonic = time.monotonic()
                self._touch_session_locked(state.customer_id, state.session_id)
                task_text = state.task
                browser_session = state.browser_session

            assert browser_session is not None
            target_url = self._target_url(start_url=start_url, task_text=task_text)
            if hasattr(browser_session, "start"):
                await browser_session.start()
            if target_url:
                await browser_session.navigate_to(target_url)
            snapshot = await self._browser_snapshot(
                browser_session=browser_session,
                task_id=task_id,
                task_text=task_text,
                target_url=target_url,
            )

            session_to_close: Any | None = None
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is None:
                    return
                state.output = snapshot["output"]
                state.is_success = True
                state.steps = snapshot["steps"]
                state.image_candidates = snapshot["image_candidates"]
                state.network_image_resources = snapshot["network_image_resources"]
                state.error = None
                state.finished_at = _utc_now_iso()
                state.updated_monotonic = time.monotonic()

                if state.stop_requested:
                    state.status = "stopped"
                    state.is_success = False
                    if not state.output:
                        state.output = "Task stopped by user."
                else:
                    state.status = "finished"
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
            session_to_close = None
            async with self._lock:
                state = self._tasks.get(task_id)
                if state is not None:
                    state.status = "failed"
                    state.is_success = False
                    state.error = (str(exc).strip() or exc.__class__.__name__)[:2000]
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

    @classmethod
    def _target_url(cls, *, start_url: str, task_text: str) -> str:
        explicit = str(start_url or "").strip()
        if explicit:
            return explicit
        match = re.search(r"https?://[^\s'\"<>),]+", str(task_text or ""))
        if match:
            return match.group(0).rstrip(".")
        return ""

    async def _browser_snapshot(
        self,
        *,
        browser_session: Any,
        task_id: str,
        task_text: str,
        target_url: str,
    ) -> dict[str, Any]:
        current_url = ""
        title = ""
        state_text = ""
        screenshot_path = ""
        with suppress(Exception):
            current_url = str(await browser_session.get_current_page_url()).strip()
        with suppress(Exception):
            title = str(await browser_session.get_current_page_title()).strip()
        with suppress(Exception):
            state_text = str(await browser_session.get_state_as_text()).strip()
        image_candidates, network_image_resources = await self._collect_image_resources(
            browser_session=browser_session,
            page_url=current_url or target_url,
        )
        screenshot_path = await self._try_capture_task_screenshot(
            browser_session=browser_session,
            task_id=task_id,
        )
        output_parts = [
            "Browser snapshot captured by OpenTulpa.",
            f"Title: {title}" if title else "Title: ",
            f"URL: {current_url}" if current_url else f"URL: {target_url}",
        ]
        if screenshot_path:
            output_parts.append(f"Screenshot: {screenshot_path}")
        if state_text:
            output_parts.append("Visible page state:")
            output_parts.append(state_text[:12000])
        if image_candidates:
            output_parts.append("Image candidates:")
            for candidate in image_candidates[:8]:
                alt = str(candidate.get("alt") or "").strip()
                suffix = f" alt={alt[:80]!r}" if alt else ""
                output_parts.append(f"- {candidate.get('url')}{suffix}")
        if network_image_resources:
            output_parts.append("Network image resources:")
            for resource in network_image_resources[:8]:
                output_parts.append(f"- {resource.get('url')}")
        step = {
            "number": 1,
            "url": current_url or target_url or None,
            "nextGoal": "Use this browser evidence to decide the next OpenTulpa tool call or final answer.",
            "actions": [f"navigate_to({target_url})"] if target_url else ["snapshot_current_page"],
            "screenshotUrl": screenshot_path or None,
        }
        return {
            "output": "\n".join(output_parts).strip(),
            "steps": [step],
            "task": task_text,
            "image_candidates": image_candidates,
            "network_image_resources": network_image_resources,
        }

    async def _collect_image_resources(
        self,
        *,
        browser_session: Any,
        page_url: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not hasattr(browser_session, "get_current_page"):
            return [], []
        page = None
        with suppress(Exception):
            page = await browser_session.get_current_page()
        if page is None or not hasattr(page, "evaluate"):
            return [], []

        raw: Any = None
        with suppress(Exception):
            raw = await page.evaluate(_COLLECT_IMAGE_RESOURCES_SCRIPT)
        if not isinstance(raw, dict):
            return [], []

        image_candidates = self._normalize_image_resource_items(
            raw.get("imageCandidates"),
            page_url=page_url,
            max_items=20,
        )
        network_image_resources = self._normalize_image_resource_items(
            raw.get("networkImageResources"),
            page_url=page_url,
            max_items=20,
        )
        return image_candidates, network_image_resources

    @classmethod
    def _normalize_image_resource_items(
        cls,
        raw_items: Any,
        *,
        page_url: str,
        max_items: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_items, list):
            return []
        assert max_items > 0
        safe_page_url = str(page_url or "").strip()
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_item in raw_items[:100]:
            if len(normalized) >= max_items:
                break
            if not isinstance(raw_item, dict):
                continue
            url = str(raw_item.get("url") or "").strip()
            if not cls._is_http_resource_url(url):
                continue
            key = url.split("#", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, Any] = {
                "url": url,
                "source": str(raw_item.get("source") or "").strip()[:40] or None,
                "page_url": safe_page_url or None,
            }
            for text_key in ("alt", "title", "initiator_type"):
                text_value = str(raw_item.get(text_key) or "").strip()
                if text_value:
                    item[text_key] = text_value[:240]
            for int_key in (
                "width",
                "height",
                "natural_width",
                "natural_height",
                "transfer_size",
                "decoded_body_size",
            ):
                numeric_value = raw_item.get(int_key)
                if isinstance(numeric_value, int | float) and numeric_value >= 0:
                    item[int_key] = int(numeric_value)
            normalized.append(item)
        return normalized

    @staticmethod
    def _is_http_resource_url(url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    async def _try_capture_task_screenshot(self, *, browser_session: Any, task_id: str) -> str:
        if not hasattr(browser_session, "take_screenshot"):
            return ""
        screenshot_dir = (TULPA_STUFF_DIR / "screenshots" / "browser_use").resolve()
        try:
            screenshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return ""
        safe_task_id = self._safe_profile_name(task_id)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        target = (screenshot_dir / f"{safe_task_id}_{timestamp}.png").resolve()
        try:
            raw_bytes = await browser_session.take_screenshot(
                path=str(target),
                full_page=False,
                format="png",
            )
        except Exception:
            return ""
        if not target.exists() and raw_bytes:
            with suppress(Exception):
                target.write_bytes(raw_bytes)
        if not target.exists():
            return ""
        try:
            return str(target.relative_to(TULPA_STUFF_DIR.resolve()))
        except ValueError:
            return str(target)

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
        if self._model_override:
            candidate = self._model_override.strip()
            if candidate.lower() in _DISALLOWED_BROWSER_MODEL_ALIASES:
                return _DEFAULT_BROWSER_MODEL
            return candidate
        for raw in (self._default_model, llm):
            candidate = str(raw or "").strip()
            if not candidate:
                continue
            if candidate.lower() in _DISALLOWED_BROWSER_MODEL_ALIASES:
                continue
            return candidate
        return _DEFAULT_BROWSER_MODEL

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
            cloud_kwargs: dict[str, Any] = {
                "cdp_url": cloud_session["cdp_url"],
                "keep_alive": True,
                "captcha_solver": True,
            }
            if allowed_domains:
                cloud_kwargs["allowed_domains"] = allowed_domains
            return browser_session_cls(**cloud_kwargs), cloud_session

        session_kwargs: dict[str, Any] = {"headless": self._headless, "keep_alive": True}
        if allowed_domains:
            session_kwargs["allowed_domains"] = allowed_domains
        if self._user_data_dir is not None:
            session_profile_dir = self._profile_dir(customer_id, session_id)
            session_profile_dir.mkdir(parents=True, exist_ok=True)
            session_kwargs["user_data_dir"] = str(session_profile_dir)
        return browser_session_cls(**session_kwargs), {"backend": "local"}

    async def _new_browser_use_cloud_session(
        self,
        *,
        customer_id: str,
        session_id: str,
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
        )
        return {
            "backend": "browser-use-cloud",
            "cdp_url": session.cdp_url,
            "profile_id": session.profile_id,
            "session_id": session.id,
            "live_url": session.live_url or "",
        }

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
        return _safe_profile_name(session_id)

    @classmethod
    def _normalize_customer_id(cls, customer_id: str | None) -> str:
        return _normalize_customer_id(customer_id)

    @classmethod
    def _normalize_optional_customer_id(cls, customer_id: str | None) -> str | None:
        return _normalize_optional_customer_id(customer_id)

    @staticmethod
    def _session_key(customer_id: str, session_id: str) -> str:
        return _browser_session_key(customer_id, session_id)

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
        live_url: str | None = None,
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
        if live_url:
            metadata["liveUrl"] = live_url
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

    @staticmethod
    def _import_browser_use_components() -> tuple[Any, Any, Any]:
        from browser_use.browser import BrowserSession

        return None, None, BrowserSession

    def _state_to_payload(self, state: _BrowserUseTaskState) -> dict[str, Any]:
        session_state = self._registry.session_state(
            customer_id=state.customer_id,
            session_id=state.session_id,
        )
        return {
            "id": state.task_id,
            "customerId": state.customer_id,
            "sessionId": state.session_id,
            "backend": session_state.backend if session_state is not None else None,
            "liveUrl": session_state.live_url if session_state is not None else None,
            "browserUseProfileId": (
                session_state.cloud_profile_id if session_state is not None else None
            ),
            "browserUseBrowserSessionId": (
                session_state.cloud_browser_session_id if session_state is not None else None
            ),
            "status": state.status,
            "isSuccess": state.is_success,
            "startedAt": state.started_at,
            "finishedAt": state.finished_at,
            "task": state.task,
            "llm": state.llm,
            "output": state.output,
            "outputFiles": state.output_files,
            "steps": state.steps,
            "imageCandidates": state.image_candidates,
            "networkImageResources": state.network_image_resources,
            "error": state.error,
            "ownerInputPrompt": state.owner_input_prompt,
            "ownerInputType": state.owner_input_type,
            "ownerInputRequestedAt": state.owner_input_requested_at,
        }

    async def _close_session(self, session: Any) -> None:
        if session is None:
            return
        cloud_session_id = self._cloud_session_ids_by_browser_session.pop(id(session), "")
        if hasattr(session, "stop"):
            with suppress(Exception):
                await asyncio.wait_for(
                    session.stop(),
                    timeout=_SESSION_CLOSE_TIMEOUT_SECONDS,
                )
        elif hasattr(session, "kill"):
            with suppress(Exception):
                await asyncio.wait_for(
                    session.kill(),
                    timeout=_SESSION_CLOSE_TIMEOUT_SECONDS,
                )
        if cloud_session_id:
            with suppress(Exception):
                await asyncio.wait_for(
                    self._get_browser_use_cloud_client().stop_browser_session(cloud_session_id),
                    timeout=_SESSION_CLOSE_TIMEOUT_SECONDS,
                )

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
        self._registry.pop_expired_terminal_tasks(
            now=now,
            retention_seconds=self._task_retention_seconds,
        )
        for session_state in self._registry.pop_expired_idle_sessions(
            now=now,
            idle_timeout_seconds=_SESSION_IDLE_TIMEOUT_SECONDS,
        ):
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
        session_state = self._registry.detach_session_if_unused(
            customer_id=safe_customer,
            session_id=safe_session,
        )
        self._write_profile_metadata(
            customer_id=safe_customer,
            session_id=safe_session,
            status="idle",
        )
        return session_state.session if session_state is not None else None

    def _touch_session_locked(self, customer_id: str, session_id: str | None) -> None:
        safe_customer = self._normalize_customer_id(customer_id)
        safe_session = str(session_id or "").strip()
        if not safe_session:
            return
        self._registry.touch_session(customer_id=safe_customer, session_id=safe_session)
        self._write_profile_metadata(
            customer_id=safe_customer,
            session_id=safe_session,
            status="running" if self._session_has_active_tasks_locked(safe_customer, safe_session) else "idle",
        )

    def _pick_reusable_session_id_locked(self, customer_id: str) -> str | None:
        return self._registry.pick_reusable_session_id(customer_id)

    def _live_session_count_for_customer_locked(self, customer_id: str) -> int:
        return self._registry.live_session_count_for_customer(customer_id)

    def _session_summaries_locked(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        return self._registry.session_summaries(customer_id)

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
        customer = cls._safe_profile_name(cls._normalize_customer_id(customer_id))
        session = cls._safe_profile_name(session_id)
        return f"opentulpa-{customer}-{session}"[:100]

    def _session_has_active_tasks_locked(self, customer_id: str, session_id: str) -> bool:
        return self._registry.session_has_active_tasks(
            customer_id=customer_id,
            session_id=session_id,
        )

    def _active_task_for_session_locked(
        self, customer_id: str, session_id: str
    ) -> _BrowserUseTaskState | None:
        return self._registry.active_task_for_session(
            customer_id=customer_id,
            session_id=session_id,
        )
