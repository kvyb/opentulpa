"""Browser Use tool registration."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id

_BROWSER_USE_DEFAULT_WAIT_TIMEOUT_SECONDS = 1800
_BROWSER_USE_MAX_WAIT_TIMEOUT_SECONDS = 1800
_BROWSER_USE_MIN_WAIT_TIMEOUT_SECONDS = 5
_BROWSER_USE_SECONDS_PER_STEP_BUFFER = 50

_BROWSER_USE_TASK_PREFIX = (
    "Use the browser like a careful human operator. Prefer visible page evidence over guesses. "
    "Open source pages before claiming facts, capture the URL/title/date when relevant, and stop "
    "once you have enough evidence for the user's goal. If blocked by login, CAPTCHA, paywall, or "
    "repeated same-state navigation, stop and report the blocker plus live_url. Do not keep browsing "
    "just to be exhaustive. When sending or citing images, prefer returned image_candidates or "
    "network_image_resources URLs from the browser result before constructing any direct image URL.\n\n"
    "Return a concise answer with verified facts, uncertain facts clearly marked, and any blockers."
)


def _get_browser_use_local_manager(runtime: Any) -> tuple[Any | None, str | None]:
    getter = getattr(runtime, "get_browser_use_local_manager", None)
    if not callable(getter):
        return None, "browser_use local backend unavailable: runtime manager not initialized"
    try:
        manager = getter()
    except Exception as exc:
        return None, f"browser_use local backend unavailable: {exc}"
    if manager is None:
        return None, "browser_use local backend unavailable: manager is None"
    return manager, None


def _normalize_allowed_domains(allowed_domains: list[str] | None) -> list[str]:
    if not isinstance(allowed_domains, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in allowed_domains:
        raw = str(item or "").strip().lower()
        if not raw:
            continue
        host = ""
        if "://" in raw:
            host = str(urlparse(raw).hostname or "").strip().lower()
        else:
            host = raw.split("/", 1)[0].split(":", 1)[0].strip().lower()
        host = host.strip(".")
        if not host or "." not in host:
            continue
        if not re.fullmatch(r"[a-z0-9.-]{1,253}", host):
            continue
        if host in seen:
            continue
        seen.add(host)
        out.append(host)
    return out


def _build_browser_use_task(task: str) -> str:
    task_text = str(task or "").strip()
    if not task_text:
        return ""
    return f"{_BROWSER_USE_TASK_PREFIX}\n\nTask:\n{task_text}"


def _compact_browser_use_task_view(
    payload: dict[str, Any],
    *,
    include_steps: bool = False,
    max_steps_preview: int = 3,
    max_output_chars: int = 12000,
) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    steps = data.get("steps", [])
    steps_list = steps if isinstance(steps, list) else []

    output_text = data.get("output")
    output = str(output_text) if output_text is not None else None
    truncated_output = False
    if output and len(output) > max_output_chars:
        output = output[:max_output_chars] + "..."
        truncated_output = True

    output_files_raw = data.get("outputFiles", [])
    output_files: list[dict[str, Any]] = []
    if isinstance(output_files_raw, list):
        for item in output_files_raw[:20]:
            if isinstance(item, dict):
                output_files.append(
                    {
                        "id": item.get("id"),
                        "fileName": item.get("fileName"),
                        "path": item.get("path"),
                    }
                )

    image_candidates = _compact_browser_resource_items(data.get("imageCandidates"), max_items=12)
    network_image_resources = _compact_browser_resource_items(
        data.get("networkImageResources"),
        max_items=12,
    )

    result: dict[str, Any] = {
        "id": data.get("id"),
        "session_id": data.get("sessionId"),
        "status": data.get("status"),
        "is_success": data.get("isSuccess"),
        "started_at": data.get("startedAt"),
        "finished_at": data.get("finishedAt"),
        "task": data.get("task"),
        "llm": data.get("llm"),
        "backend": data.get("backend"),
        "live_url": data.get("liveUrl"),
        "output": output,
        "output_truncated": truncated_output,
        "output_files": output_files,
        "image_candidates": image_candidates,
        "network_image_resources": network_image_resources,
        "steps_count": len(steps_list),
        "owner_input_prompt": data.get("ownerInputPrompt"),
        "owner_input_type": data.get("ownerInputType"),
        "owner_input_requested_at": data.get("ownerInputRequestedAt"),
    }

    if include_steps:
        safe_preview = max(1, min(int(max_steps_preview), 10))
        preview: list[dict[str, Any]] = []
        for step in steps_list[:safe_preview]:
            if not isinstance(step, dict):
                continue
            actions = step.get("actions", [])
            actions_list = [str(a) for a in actions][:5] if isinstance(actions, list) else []
            preview.append(
                {
                    "number": step.get("number"),
                    "url": step.get("url"),
                    "next_goal": str(step.get("nextGoal") or "")[:240],
                    "actions": actions_list,
                    "screenshot_url": step.get("screenshotUrl"),
                }
            )
        result["steps_preview"] = preview
        result["steps_preview_truncated"] = len(steps_list) > safe_preview
    return result


def _compact_browser_resource_items(raw_items: Any, *, max_items: int) -> list[dict[str, Any]]:
    if not isinstance(raw_items, list):
        return []
    assert max_items > 0
    compact: list[dict[str, Any]] = []
    for raw_item in raw_items[:max_items]:
        if not isinstance(raw_item, dict):
            continue
        url = str(raw_item.get("url") or "").strip()
        if not url:
            continue
        item: dict[str, Any] = {"url": url}
        for key in ("source", "alt", "title", "page_url", "initiator_type"):
            value = raw_item.get(key)
            if value:
                item[key] = str(value)[:240]
        for key in (
            "width",
            "height",
            "natural_width",
            "natural_height",
            "transfer_size",
            "decoded_body_size",
        ):
            value = raw_item.get(key)
            if isinstance(value, int | float):
                item[key] = int(value)
        compact.append(item)
    return compact


def register_browser_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def browser_use_session_list() -> Any:
        """
        List known Browser Use sessions so the agent can reuse an idle session_id
        instead of spawning a fresh browser session. Sessions can include persisted
        browser profile state such as cookies; use this before repeat account work.
        """
        customer_id = require_customer_id(runtime)
        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_session_list unavailable"}
        return {"sessions": await manager.list_sessions(customer_id=customer_id)}

    @tool
    async def browser_use_run(
        task: str,
        allowed_domains: list[str] | None = None,
        max_steps: int = 20,
        wait_timeout_seconds: int = _BROWSER_USE_DEFAULT_WAIT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = 4,
        llm: str = "browser-use-llm",
        start_url: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """
        Open a Browser Use-backed browser session, navigate when start_url or a URL
        in task is present, and return OpenTulpa-captured page evidence. Use for
        dynamic pages where a real browser snapshot is needed. Browser sessions are
        kept alive and may use persisted profile state when configured; reuse a prior
        session_id when continuing the same account/site workflow. Do not ask the
        owner to paste credentials into durable memory; use current-turn credentials
        only for the intended browser work. When image candidates are visible or loaded,
        the result can include image_candidates and network_image_resources URLs.
        Do not poll browser_use_task_get after this unless browser_use_run returns
        running or the owner explicitly asks for browser status.
        """
        task_text = _build_browser_use_task(str(task or ""))
        if not task_text:
            return {"error": "browser_use_run requires a non-empty task"}
        customer_id = require_customer_id(runtime)

        safe_max_steps = max(1, min(int(max_steps), 80))
        requested_wait_timeout = max(
            _BROWSER_USE_MIN_WAIT_TIMEOUT_SECONDS,
            min(int(wait_timeout_seconds), _BROWSER_USE_MAX_WAIT_TIMEOUT_SECONDS),
        )
        expected_worker_timeout = min(
            _BROWSER_USE_MAX_WAIT_TIMEOUT_SECONDS,
            safe_max_steps * _BROWSER_USE_SECONDS_PER_STEP_BUFFER,
        )
        safe_wait_timeout = max(requested_wait_timeout, expected_worker_timeout)
        safe_poll_interval = max(2, min(int(poll_interval_seconds), 30))
        safe_domains = _normalize_allowed_domains(allowed_domains)
        safe_llm = str(llm or "").strip() or "browser-use-llm"
        safe_start_url = str(start_url or "").strip()
        safe_session_id = str(session_id or "").strip()

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_run unavailable"}
        turn_mode_getter = getattr(runtime, "get_active_turn_mode", None)
        active_turn_mode = (
            str(turn_mode_getter() or "").strip().lower()
            if callable(turn_mode_getter)
            else "interactive"
        )
        allow_owner_input = active_turn_mode in {"interactive", "workflow_setup", ""}

        created = await manager.start_task(
            task=task_text,
            max_steps=safe_max_steps,
            llm=safe_llm,
            allowed_domains=safe_domains,
            start_url=safe_start_url or None,
            session_id=safe_session_id or None,
            customer_id=customer_id,
            allow_owner_input=allow_owner_input,
        )
        if isinstance(created, dict) and created.get("error"):
            return {
                "error": str(created.get("error")),
                "session_id": created.get("sessionId") or safe_session_id or None,
                "active_task_id": created.get("activeTaskId"),
            }

        task_id = str((created or {}).get("id") or "").strip()
        result_session_id = str((created or {}).get("sessionId") or safe_session_id).strip()
        if not task_id:
            return {
                "error": str((created or {}).get("error") or "browser_use_run create failed: missing task id"),
                "session_id": result_session_id or None,
                "active_task_id": (created or {}).get("activeTaskId"),
            }

        deadline = datetime.now(UTC).timestamp() + safe_wait_timeout
        while True:
            task_data = await manager.get_task(task_id)
            if not isinstance(task_data, dict):
                return {
                    "error": f"browser_use_run poll failed: task not found ({task_id})",
                    "task_id": task_id,
                    "session_id": result_session_id or None,
                }

            status = str(task_data.get("status") or "").strip().lower()
            if status == "waiting_for_owner":
                compact = _compact_browser_use_task_view(task_data)
                compact["task_id"] = task_id
                compact["session_id"] = result_session_id or compact.get("session_id")
                compact["status"] = "waiting_for_owner"
                compact["message"] = (
                    "Browser task is waiting for owner input. Ask the owner for "
                    "owner_input_prompt, then call browser_use_owner_input_submit."
                )
                return compact

            if status in {"finished", "stopped", "failed"}:
                compact = _compact_browser_use_task_view(task_data)
                compact["task_id"] = task_id
                compact["session_id"] = result_session_id or compact.get("session_id")
                compact["status"] = status or str(compact.get("status") or "unknown")
                return compact

            if datetime.now(UTC).timestamp() >= deadline:
                compact = _compact_browser_use_task_view(
                    task_data,
                    include_steps=True,
                    max_steps_preview=3,
                )
                compact["task_id"] = task_id
                compact["session_id"] = result_session_id or compact.get("session_id")
                compact["status"] = status or "started"
                compact["timed_out"] = True
                compact["message"] = (
                    "Browser task is still running after the internal wait window. "
                    "Answer from this status unless the owner asks for a fresh browser status; "
                    "then call browser_use_task_get(task_id)."
                )
                return compact

            await asyncio.sleep(safe_poll_interval)

    @tool
    async def browser_use_task_get(
        task_id: str,
        include_steps: bool = False,
        max_steps_preview: int = 3,
    ) -> Any:
        """
        Get Browser Use task status/details by task_id (compact by default).
        Use when browser_use_run returned running, or when the owner explicitly
        asks what is happening with an existing browser task.
        """
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_get requires task_id"}
        customer_id = require_customer_id(runtime)

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_task_get unavailable"}

        payload = await manager.get_task(safe_task_id, customer_id=customer_id)
        if not isinstance(payload, dict):
            return {"error": f"browser_use_task_get failed: task not found ({safe_task_id})"}
        return _compact_browser_use_task_view(
            payload,
            include_steps=bool(include_steps),
            max_steps_preview=max_steps_preview,
        )

    @tool
    async def browser_use_task_screenshot(
        task_id: str,
        full_page: bool = True,
    ) -> Any:
        """
        Capture a screenshot from an existing Browser Use task/session, save it under
        tulpa_stuff/, and return the local path. Use tulpa_file_send(path) to send it.
        """
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_screenshot requires task_id"}
        customer_id = require_customer_id(runtime)

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_task_screenshot unavailable"}

        payload = await manager.capture_screenshot(
            task_id=safe_task_id,
            full_page=bool(full_page),
            customer_id=customer_id,
        )
        if isinstance(payload, dict) and payload.get("error"):
            return {"error": str(payload.get("error"))}
        return payload if isinstance(payload, dict) else {"error": "browser_use_task_screenshot failed"}

    @tool
    async def browser_use_task_control(task_id: str, action: str = "stop_task_and_session") -> Any:
        """Control Browser Use task execution (stop, pause, resume, or stop_task_and_session)."""
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_control requires task_id"}
        safe_action = str(action or "").strip().lower()
        allowed_actions = {"stop", "pause", "resume", "stop_task_and_session"}
        if safe_action not in allowed_actions:
            return {
                "error": (
                    "browser_use_task_control invalid action. "
                    "Use one of: stop, pause, resume, stop_task_and_session"
                )
            }
        customer_id = require_customer_id(runtime)

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_task_control unavailable"}

        payload = await manager.control_task(
            task_id=safe_task_id,
            action=safe_action,
            customer_id=customer_id,
        )
        if isinstance(payload, dict) and payload.get("error"):
            return {"error": str(payload.get("error"))}
        return _compact_browser_use_task_view(payload if isinstance(payload, dict) else {})

    @tool
    async def browser_use_owner_input_submit(task_id: str, owner_input: str) -> Any:
        """
        Submit owner-provided MFA/email/SMS/authenticator/account-choice input to a
        Browser Use task that is waiting_for_owner. This resumes the same live browser session.
        """
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_owner_input_submit requires task_id"}
        safe_owner_input = str(owner_input or "").strip()
        if not safe_owner_input:
            return {"error": "browser_use_owner_input_submit requires owner_input"}
        customer_id = require_customer_id(runtime)

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_owner_input_submit unavailable"}

        payload = await manager.submit_owner_input(
            task_id=safe_task_id,
            owner_input=safe_owner_input,
            customer_id=customer_id,
        )
        if isinstance(payload, dict) and payload.get("error"):
            return {"error": str(payload.get("error"))}
        return _compact_browser_use_task_view(payload if isinstance(payload, dict) else {})

    return {
        "browser_use_session_list": browser_use_session_list,
        "browser_use_run": browser_use_run,
        "browser_use_task_get": browser_use_task_get,
        "browser_use_task_screenshot": browser_use_task_screenshot,
        "browser_use_task_control": browser_use_task_control,
        "browser_use_owner_input_submit": browser_use_owner_input_submit,
    }
