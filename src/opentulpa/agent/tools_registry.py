"""Tool registration for the OpenTulpa LangGraph runtime."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from opentulpa.agent.file_analysis import summarize_uploaded_blob
from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    extract_html_title as _extract_html_title,
)
from opentulpa.agent.utils import (
    html_to_text as _html_to_text,
)
from opentulpa.agent.utils import (
    looks_like_shell_command as _looks_like_shell_command,
)
from opentulpa.context.customer_profile_models import (
    CustomerScopedClearResponse,
    CustomerScopedOkResponse,
    CustomerScopedRequest,
    DirectiveGetResponse,
    DirectiveSetRequest,
    TimeProfileGetResponse,
    TimeProfileSetRequest,
    TimeProfileSetResponse,
)
from opentulpa.policy.execution_boundary import (
    ExecutionBoundaryContext,
    ExecutionBoundaryGuard,
)


def _require_customer_id(runtime: Any) -> str:
    getter = getattr(runtime, "get_active_customer_id", None)
    customer_id = ""
    if callable(getter):
        customer_id = str(getter() or "").strip()
    if not customer_id:
        customer_id = str(getattr(runtime, "_active_customer_id", "") or "").strip()
    if not customer_id:
        raise RuntimeError("customer_id is missing in runtime context")
    return customer_id


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


def _normalize_cleanup_paths(paths: list[str] | None) -> list[str]:
    if not isinstance(paths, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in paths:
        path = str(item or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _unique_string_list(values: list[str] | None) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


def _normalize_optional_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


_INTAKE_ALLOWED_SINK_TYPES = {"local_csv", "google_sheets_composio", "generic_composio_write"}


def _validate_intake_sink_request(*, sink_type: str, sink_config: dict[str, Any]) -> str | None:
    safe_sink_type = str(sink_type or "").strip().lower()
    safe_config = sink_config if isinstance(sink_config, dict) else {}
    if safe_sink_type not in _INTAKE_ALLOWED_SINK_TYPES:
        if safe_sink_type == "google_sheets":
            return (
                "sink_type=google_sheets is not supported here; use google_sheets_composio and "
                "provide toolkit/field_mapping/static_arguments instead"
            )
        return (
            "sink_type must be one of local_csv|google_sheets_composio|generic_composio_write"
        )
    if safe_sink_type == "local_csv":
        return None
    toolkit = str(safe_config.get("toolkit", "") or "").strip()
    legacy_tool_slug = str(safe_config.get("tool_slug", "") or "").strip()
    if safe_sink_type == "generic_composio_write" and not toolkit and not legacy_tool_slug:
        return (
            "composio sink_config.toolkit is required for generic_composio_write"
        )
    field_mapping = safe_config.get("field_mapping")
    if not isinstance(field_mapping, dict) or not field_mapping:
        return (
            "composio sink_config.field_mapping is required; map sink argument names to workflow fields "
            "before calling intake_workflow_upsert"
        )
    operation_hint = str(safe_config.get("operation_hint", "") or "").strip()
    if safe_sink_type == "generic_composio_write" and not operation_hint and not legacy_tool_slug:
        return (
            "generic_composio_write requires sink_config.operation_hint so the runtime can choose the right tool"
        )
    return None


_WORKING_DIR_PREFIXES: dict[str, str] = {
    "tulpa_stuff": "tulpa_stuff",
    "integrations": "src/opentulpa/integrations",
    "interfaces": "src/opentulpa/interfaces",
    "tools": "src/opentulpa/tools",
    "skills": "src/opentulpa/skills",
    "opentulpa": "src/opentulpa",
}


def _normalize_command_for_working_dir(command: str, working_dir: str) -> str:
    text = str(command or "").strip()
    if not text:
        return ""
    prefix = _WORKING_DIR_PREFIXES.get(str(working_dir or "").strip())
    if not prefix:
        return text
    try:
        parts = shlex.split(text)
    except Exception:
        return text
    if len(parts) <= 1:
        return text

    markers = (f"{prefix}/", f"./{prefix}/")

    def _strip_one(token: str) -> str:
        raw = str(token)
        for marker in markers:
            if raw.startswith(marker):
                return raw[len(marker) :]
        if raw.startswith("--") and "=" in raw:
            key, value = raw.split("=", 1)
            for marker in markers:
                if value.startswith(marker):
                    return f"{key}={value[len(marker):]}"
        return raw

    normalized = [parts[0], *(_strip_one(item) for item in parts[1:])]
    return shlex.join(normalized)


def _normalize_execution_origin(
    *,
    thread_id: str | None,
    execution_origin: str | None,
) -> str:
    return ExecutionBoundaryGuard.normalize_execution_origin(
        thread_id=str(thread_id or "").strip(),
        execution_origin=str(execution_origin or "").strip(),
    )


def _approval_pending_payload(
    *,
    action_name: str,
    command_preview: str,
    decision: dict[str, Any],
) -> dict[str, Any]:
    approval_id = str(decision.get("approval_id", "")).strip()
    if approval_id.lower() in {"none", "null"}:
        approval_id = ""
    summary = str(decision.get("summary", f"execute {action_name}")).strip()
    reason = str(decision.get("reason", "approval_required")).strip()
    if not approval_id:
        return {
            "ok": False,
            "status": "guardrail_unavailable",
            "action_name": action_name,
            "command_preview": command_preview[:300],
            "approval_id": None,
            "delivery_mode": str(decision.get("delivery_mode", "")).strip() or None,
            "summary": summary,
            "reason": reason or "approval_challenge_unavailable",
            "message": (
                "GUARDRAIL_BLOCKED: Approval is required but the approval challenge "
                "could not be created. Please retry."
            ),
            "gate": "require_approval",
            "retryable": True,
        }
    message = (
        "APPROVAL_PENDING: This executable action is waiting for user approval "
        f"(approval_id={approval_id}; summary={summary}; reason={reason})."
    )
    return {
        "ok": False,
        "status": "approval_pending",
        "action_name": action_name,
        "command_preview": command_preview[:300],
        "approval_id": approval_id or None,
        "delivery_mode": str(decision.get("delivery_mode", "")).strip() or None,
        "summary": summary,
        "reason": reason,
        "message": message,
        "gate": "require_approval",
    }


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

    result: dict[str, Any] = {
        "id": data.get("id"),
        "session_id": data.get("sessionId"),
        "status": data.get("status"),
        "is_success": data.get("isSuccess"),
        "started_at": data.get("startedAt"),
        "finished_at": data.get("finishedAt"),
        "task": data.get("task"),
        "llm": data.get("llm"),
        "output": output,
        "output_truncated": truncated_output,
        "output_files": output_files,
        "steps_count": len(steps_list),
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


def _best_crawl4ai_text(result: Any) -> tuple[str, str | None]:
    title: str | None = None
    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        raw_title = metadata.get("title")
        if isinstance(raw_title, str) and raw_title.strip():
            title = raw_title.strip()

    candidates = [
        getattr(result, "fit_markdown", None),
        getattr(result, "markdown", None),
        getattr(result, "extracted_content", None),
        getattr(result, "cleaned_html", None),
        getattr(result, "html", None),
        getattr(result, "text", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = ""
        if isinstance(candidate, str):
            text = candidate
        elif isinstance(candidate, (dict, list)):
            with suppress(Exception):
                text = json.dumps(candidate, ensure_ascii=False)
        else:
            text = str(candidate)
        text = str(text).strip()
        if not text:
            continue
        if "<html" in text.lower() or "</" in text:
            text = _html_to_text(text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            return text, title
    return "", title


async def _crawl4ai_extract(url: str) -> tuple[str, str | None, str | None]:
    """
    Return (text, title, error). If crawl4ai is unavailable/failed, text is empty and error set.
    """
    try:
        from crawl4ai import AsyncWebCrawler
    except Exception as exc:
        return "", None, f"crawl4ai unavailable: {exc}"

    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
    except Exception as exc:
        return "", None, f"crawl4ai crawl failed: {exc}"

    if bool(getattr(result, "success", True)) is False:
        error_message = str(getattr(result, "error_message", "")).strip()
        return "", None, f"crawl4ai crawl failed: {error_message or 'unknown_error'}"

    text, title = _best_crawl4ai_text(result)
    if not text:
        return "", title, "crawl4ai returned no extractable content"
    return text, title, None


def _sanitize_routine_customer_segment(customer_id: str) -> str:
    raw = str(customer_id or "").strip().lower()
    safe = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-")
    return (safe or "customer")[:48]


def _proactive_heartbeat_routine_id(customer_id: str) -> str:
    return f"rtn_proactive_{_sanitize_routine_customer_segment(customer_id)}"


def _directive_disables_proactive_mode(directive: str) -> bool:
    text = str(directive or "").strip().lower()
    if not text:
        return False
    patterns = [
        r"\b(?:disable|turn off|stop|pause|remove)\s+(?:my\s+)?proactive\b",
        r"\bnot\s+proactive\b",
        r"\bmode\s*[:=]?\s*non[- ]?proactive\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _directive_enables_proactive_mode(directive: str) -> bool:
    text = str(directive or "").strip().lower()
    if not text or _directive_disables_proactive_mode(text):
        return False
    patterns = [
        r"\bmode\s*[:=]?\s*proactive\b",
        r"\bproactive\s+mode\b",
        r"\bproactive\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _extract_heartbeat_interval_hours(directive: str, *, default_hours: int) -> int:
    text = str(directive or "").strip().lower()
    interval = max(1, min(int(default_hours), 24))
    if not text:
        return interval
    match = re.search(r"\bevery\s+(\d{1,2})\s*(?:hours?|hrs?|h)\b", text)
    if match:
        with suppress(Exception):
            return max(1, min(int(match.group(1)), 24))
    if re.search(r"\bevery\s+(?:few)\s+hours?\b", text):
        return 3
    if re.search(r"\bevery\s+(?:couple)\s+hours?\b", text):
        return 2
    return interval


def _build_proactive_heartbeat_prompt(interval_hours: int) -> str:
    return (
        "Proactive heartbeat wake. Decide naturally whether to reach out now.\n"
        "Goals: build connection, show care, and be useful without being spammy.\n"
        "Rules:\n"
        "- Use memory/context and recent conversation themes.\n"
        "- If no meaningful outreach is appropriate now, return exactly __NO_NOTIFY__.\n"
        "- If outreach is appropriate, send one concise, natural message.\n"
        "- Prefer varied check-ins/questions/shares over repetitive phrasing.\n"
        "- If sharing content, pick one relevant thing only.\n"
        f"- Heartbeat cadence baseline: every {interval_hours} hour(s).\n"
    )


async def _sync_proactive_heartbeat(
    *,
    runtime: Any,
    customer_id: str,
    directive_text: str,
) -> dict[str, Any]:
    cid = str(customer_id or "").strip()
    if not cid:
        return {"ok": False, "reason": "missing_customer_id"}

    routine_id = _proactive_heartbeat_routine_id(cid)
    wants_proactive = _directive_enables_proactive_mode(directive_text)
    default_hours = int(getattr(runtime, "_proactive_heartbeat_default_hours", 3))
    interval_hours = _extract_heartbeat_interval_hours(
        directive_text,
        default_hours=default_hours,
    )
    routine_name = "Proactive Heartbeat"

    if not wants_proactive:
        response = await runtime._request_with_backoff(
            "DELETE",
            f"/internal/scheduler/routine/{routine_id}",
            params={"customer_id": cid},
            timeout=8.0,
            retries=1,
        )
        if response.status_code != 200:
            return {
                "ok": False,
                "enabled": False,
                "routine_id": routine_id,
                "reason": f"heartbeat_disable_failed_http_{response.status_code}",
            }
        payload = response.json() if response.content else {}
        return {
            "ok": True,
            "enabled": False,
            "routine_id": routine_id,
            "removed": bool(payload.get("ok", False)),
            "interval_hours": interval_hours,
        }

    create = await runtime._request_with_backoff(
        "POST",
        "/internal/scheduler/routine",
        json_body={
            "id": routine_id,
            "name": routine_name,
            "schedule": f"0 */{interval_hours} * * *",
            "is_cron": True,
            "enabled": True,
            "payload": {
                "customer_id": cid,
                "notify_user": True,
                "proactive_heartbeat": True,
                "heartbeat_interval_hours": interval_hours,
                "instruction": _build_proactive_heartbeat_prompt(interval_hours),
            },
        },
        timeout=10.0,
        retries=1,
    )
    if create.status_code != 200:
        return {
            "ok": False,
            "enabled": True,
            "routine_id": routine_id,
            "interval_hours": interval_hours,
            "reason": f"heartbeat_enable_failed_http_{create.status_code}",
        }
    result = create.json() if create.content else {}
    return {
        "ok": True,
        "enabled": True,
        "routine_id": str(result.get("id", routine_id)).strip() or routine_id,
        "name": routine_name,
        "interval_hours": interval_hours,
        "schedule": f"0 */{interval_hours} * * *",
    }


def register_runtime_tools(runtime: Any) -> dict[str, Any]:
    boundary_guard = ExecutionBoundaryGuard(runtime=runtime)

    @tool
    async def memory_search(query: str) -> Any:
        """Search user memory."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/memory/search",
            json_body={"query": query, "user_id": customer_id, "limit": 5},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"memory_search failed: {r.text}"}
        return r.json().get("results", [])

    @tool
    async def memory_add(summary: str) -> Any:
        """Store a user memory summary."""
        customer_id = _require_customer_id(runtime)
        retryable_errors = (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            RuntimeError,
        )
        for attempt in range(2):
            try:
                r = await runtime._request_with_backoff(
                    "POST",
                    "/internal/memory/add",
                    json_body={
                        "messages": [{"role": "user", "content": summary}],
                        "user_id": customer_id,
                    },
                    timeout=30.0,
                    retries=0,
                )
                if r.status_code != 200:
                    return {"error": f"memory_add failed ({r.status_code}): {r.text}"}
                return {"ok": True}
            except retryable_errors as exc:
                if attempt == 0:
                    await asyncio.sleep(1.5)
                    continue
                exc_name = type(exc).__name__
                detail = str(exc) or exc_name
                return {"error": f"memory_add timed out after retries: {detail}"}
        return {"error": "memory_add failed: exhausted retries"}

    @tool
    async def uploaded_file_search(query: str, limit: int = 5) -> Any:
        """Search uploaded files for this user by natural-language query."""
        customer_id = _require_customer_id(runtime)
        safe_limit = max(1, min(int(limit), 20))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/search",
            json_body={
                "query": query,
                "customer_id": customer_id,
                "limit": safe_limit,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_search failed: {r.text}"}
        return r.json().get("results", [])

    @tool
    async def uploaded_file_get(
        file_id: str,
        max_excerpt_chars: int = 16000,
    ) -> Any:
        """Get metadata and text excerpt for one uploaded file."""
        customer_id = _require_customer_id(runtime)
        safe_chars = max(500, min(int(max_excerpt_chars), 60000))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/get",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "max_excerpt_chars": safe_chars,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_get failed: {r.text}"}
        return r.json().get("file", {})

    @tool
    async def uploaded_file_send(
        file_id: str,
        caption: str | None = None,
    ) -> Any:
        """Send a previously uploaded file back to the user's Telegram chat."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/send",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "caption": caption,
            },
            timeout=25.0,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_send failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_file_send(
        path: str,
        caption: str | None = None,
    ) -> Any:
        """Send a local file from tulpa_stuff/ back to the user's Telegram chat."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/send_local",
            json_body={
                "path": path,
                "customer_id": customer_id,
                "caption": caption,
            },
            timeout=25.0,
        )
        if r.status_code != 200:
            return {"error": f"tulpa_file_send failed: {r.text}"}
        return r.json()

    @tool
    async def web_image_send(
        url: str,
        caption: str | None = None,
        max_bytes: int = 10_000_000,
    ) -> Any:
        """
        Download an image from a web URL (validated content-type) and send it to Telegram.
        Use web_search first to find candidate URLs, then call this tool.
        """
        customer_id = _require_customer_id(runtime)
        safe_max_bytes = max(250_000, min(int(max_bytes), 25_000_000))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/send_web_image",
            json_body={
                "url": url,
                "customer_id": customer_id,
                "caption": caption,
                "max_bytes": safe_max_bytes,
            },
            timeout=70.0,
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"web_image_send failed: {r.text}"}
        return r.json()

    @tool
    async def uploaded_file_analyze(
        file_id: str,
        question: str | None = None,
    ) -> Any:
        """Analyze a previously uploaded file again, optionally with a focused question."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/analyze",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "question": question,
            },
            timeout=60.0,
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"uploaded_file_analyze failed: {r.text}"}
        return r.json()

    @tool
    async def skill_list(include_global: bool = True, limit: int = 50) -> Any:
        """List reusable skills available to this user."""
        customer_id = _require_customer_id(runtime)
        safe_limit = max(1, min(int(limit), 200))
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/list",
            json_body={
                "customer_id": customer_id,
                "include_global": bool(include_global),
                "include_disabled": False,
                "limit": safe_limit,
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_list failed: {r.text}"}
        return r.json().get("skills", [])

    @tool
    async def skill_get(
        name: str,
        include_files: bool = True,
        include_global: bool = True,
    ) -> Any:
        """Get one skill by name, using user-scope first then global fallback."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/get",
            json_body={
                "customer_id": customer_id,
                "name": name,
                "include_files": bool(include_files),
                "include_global": bool(include_global),
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_get failed: {r.text}"}
        return r.json().get("skill", {})

    @tool
    async def skill_upsert(
        name: str,
        description: str,
        instructions: str,
        scope: str = "user",
        supporting_files: dict[str, str] | None = None,
    ) -> Any:
        """Create or update a reusable skill for this user (or global when explicitly chosen)."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/upsert",
            json_body={
                "customer_id": customer_id,
                "scope": scope,
                "name": name,
                "description": description,
                "instructions": instructions,
                "supporting_files": supporting_files if isinstance(supporting_files, dict) else None,
                "source": "langgraph_tool",
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_upsert failed: {r.text}"}
        return r.json().get("skill", {})

    @tool
    async def skill_delete(name: str, scope: str = "user") -> Any:
        """Delete a reusable skill by name."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/skills/delete",
            json_body={
                "customer_id": customer_id,
                "scope": scope,
                "name": name,
            },
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"skill_delete failed: {r.text}"}
        return r.json()

    @tool
    async def intake_workflow_upsert(
        name: str,
        intent_description: str,
        required_fields: list[str],
        sink_type: str,
        sink_config: dict[str, Any],
        schedule: str = "*/5 * * * *",
        channel: str = "instagram_dm",
        provider: str = "composio",
        source_config: dict[str, Any] | None | str = None,
        field_guidance: dict[str, Any] | None | str = None,
        assistant_instructions: str = "",
        knowledge_file_ids: list[str] | None = None,
        notify_user: bool = True,
        enabled: bool = True,
        workflow_id: str | None = "",
        thread_id: str = "",
        execution_origin: str | None = None,
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """Create or update a scheduled intake workflow.

        Use this when the user wants OpenTulpa to monitor inbound messages, decide whether
        they match a business workflow, ask follow-up questions, and save the result.

        Important shaping rules:
        - For a brand-new workflow, omit workflow_id or pass an empty string.
        - For updates, pass the existing workflow_id.
        - If the user is refining or editing an existing workflow, prefer intake_workflow_list and
          intake_workflow_get first, then update the matching workflow_id instead of creating a duplicate.
        - required_fields must be a list of plain field names like ["date", "time", "car_type"].
        - field_guidance may be either:
          - a dict keyed by field name, or
          - a short plain-text note; it will be stored as general guidance.
        - source_config is optional.
        - If source_config.conversation_id is omitted, the workflow scans recent conversations
          for the configured source instead of pinning one specific thread.
        - channel/provider pairs supported here:
          - instagram_dm + composio
          - telegram_business_dm + telegram_bot_api
        - For Telegram Business, source_config.business_connection_id is required.
        - assistant_instructions stores durable reply rules, tone, escalation, guardrails, and any other
          important workflow context learned during the conversation that should persist for future inbox turns.
        - knowledge_file_ids is optional. Use it only when the user explicitly wants uploaded files bound to the workflow.
        - The workflow must still work when knowledge_file_ids is empty; in that case rely on the saved instructions
          and other workflow fields instead of pretending files exist.
        - sink_config must contain the concrete configuration needed by the chosen sink_type.
        - Valid sink_type values here are local_csv, google_sheets_composio, or generic_composio_write.
        - Never invent sink_type=google_sheets.
        - For Google Sheets, pass toolkit-level configuration, not a concrete tool slug:
          sink_type=google_sheets_composio
          sink_config={"toolkit": "googlesheets", "field_mapping": {...}, "static_arguments": {...}}
        - OpenTulpa resolves the concrete Composio tool at execution time from the toolkit.
        - If the user only gives a Google Sheet URL, extract the spreadsheet ID and pass it inside
          sink_config.static_arguments.
        - For generic_composio_write, prefer:
          sink_config={"toolkit": "...", "operation_hint": "...", "field_mapping": {...}, "static_arguments": {...}}
        """
        safe_customer = _require_customer_id(runtime)
        safe_name = str(name or "").strip()
        safe_intent = str(intent_description or "").strip()
        safe_schedule = str(schedule or "").strip() or "*/5 * * * *"
        safe_channel = str(channel or "").strip() or "instagram_dm"
        safe_provider = str(provider or "").strip() or "composio"
        safe_sink_type = str(sink_type or "").strip()
        safe_workflow_id = _normalize_optional_id(workflow_id)
        safe_required_fields = _unique_string_list(required_fields)
        safe_knowledge_file_ids = _unique_string_list(knowledge_file_ids)
        safe_sink_config = sink_config if isinstance(sink_config, dict) else {}
        safe_source_config = source_config if isinstance(source_config, dict) else None
        safe_field_guidance = (
            field_guidance
            if isinstance(field_guidance, dict)
            else ({"notes": str(field_guidance).strip()} if str(field_guidance or "").strip() else None)
        )
        safe_assistant_instructions = str(assistant_instructions or "").strip()
        if not safe_name:
            return {"error": "intake_workflow_upsert failed: name is required"}
        if not safe_intent:
            return {"error": "intake_workflow_upsert failed: intent_description is required"}
        if not safe_required_fields:
            return {"error": "intake_workflow_upsert failed: required_fields must contain at least one field"}
        if not safe_sink_type:
            return {"error": "intake_workflow_upsert failed: sink_type is required"}
        if not safe_sink_config:
            return {"error": "intake_workflow_upsert failed: sink_config is required"}
        sink_error = _validate_intake_sink_request(
            sink_type=safe_sink_type,
            sink_config=safe_sink_config,
        )
        if sink_error:
            return {"error": f"intake_workflow_upsert failed: {sink_error}"}

        normalized_origin = _normalize_execution_origin(
            thread_id=thread_id,
            execution_origin=execution_origin,
        )
        guard_payload = guard_context if isinstance(guard_context, dict) else {}
        previous_user = str(guard_payload.get("previous_user_message", "")).strip()
        previous_assistant = str(guard_payload.get("previous_assistant_message", "")).strip()
        approval_action_args = {
            "name": safe_name,
            "intent_description": safe_intent,
            "required_fields": safe_required_fields,
            "sink_type": safe_sink_type,
            "sink_config": safe_sink_config,
            "schedule": safe_schedule,
            "channel": safe_channel,
            "provider": safe_provider,
            "source_config": safe_source_config,
            "field_guidance": safe_field_guidance,
            "assistant_instructions": safe_assistant_instructions,
            "knowledge_file_ids": safe_knowledge_file_ids,
            "notify_user": bool(notify_user),
            "enabled": bool(enabled),
            "workflow_id": safe_workflow_id,
        }
        decision = await boundary_guard.evaluate(
            ExecutionBoundaryContext(
                customer_id=safe_customer,
                thread_id=str(thread_id or "").strip() or f"chat-{safe_customer}",
                action_name="intake_workflow_upsert",
                action_args=approval_action_args,
                execution_origin=normalized_origin,
                preapproved=bool(preapproved),
                action_note=(
                    "Persistent intake workflow creation/update with scheduled external reads and "
                    "potential external writes via configured sink. "
                    f"previous_user_message={previous_user[:800]} "
                    f"previous_assistant_message={previous_assistant[:800]}"
                ),
            )
        )
        gate = str((decision or {}).get("gate", "allow")).strip().lower()
        if gate == "require_approval":
            return _approval_pending_payload(
                action_name="intake_workflow_upsert",
                command_preview=f"{safe_name} -> {safe_sink_type}",
                decision=decision if isinstance(decision, dict) else {},
            )
        if gate == "deny":
            return {
                "ok": False,
                "status": "denied",
                "gate": "deny",
                "reason": str((decision or {}).get("reason", "guardrail_denied")).strip(),
            }

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/upsert",
            json_body={
                "customer_id": safe_customer,
                "workflow_id": safe_workflow_id or None,
                "name": safe_name,
                "channel": safe_channel,
                "provider": safe_provider,
                "source_config": safe_source_config,
                "intent_description": safe_intent,
                "required_fields": safe_required_fields,
                "field_guidance": safe_field_guidance,
                "assistant_instructions": safe_assistant_instructions,
                "knowledge_file_ids": safe_knowledge_file_ids,
                "sink_type": safe_sink_type,
                "sink_config": safe_sink_config,
                "schedule": safe_schedule,
                "notify_user": bool(notify_user),
                "enabled": bool(enabled),
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_upsert failed: {r.text}"}
        return r.json().get("workflow", {})

    @tool
    async def telegram_business_status() -> Any:
        """Check whether Telegram Business is connected for the active user and inspect available business connections."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/telegram/business/status",
            json_body={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"telegram_business_status failed: {r.text}"}
        return r.json()

    @tool
    async def intake_workflow_list(include_disabled: bool = False) -> Any:
        """List intake workflows for the current user."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/list",
            json_body={
                "customer_id": customer_id,
                "include_disabled": bool(include_disabled),
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_list failed: {r.text}"}
        return r.json().get("workflows", [])

    @tool
    async def intake_workflow_get(workflow_id: str) -> Any:
        """Get one intake workflow by id."""
        customer_id = _require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_get failed: workflow_id is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/get",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_get failed: {r.text}"}
        return r.json().get("workflow", {})

    @tool
    async def intake_workflow_delete(workflow_id: str) -> Any:
        """Delete one intake workflow and its scheduled routine."""
        customer_id = _require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_delete failed: workflow_id is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/delete",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_delete failed: {r.text}"}
        return r.json()

    @tool
    async def intake_workflow_run(workflow_id: str, force: bool = False) -> Any:
        """Run one intake workflow immediately for the current user."""
        customer_id = _require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_run failed: workflow_id is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/run",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
                "force": bool(force),
                "event_type": "manual",
            },
            timeout=60.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_run failed: {r.text}"}
        return r.json()

    @tool
    async def composio_status() -> Any:
        """Check whether Composio is configured before trying auth or external tool execution."""
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/composio/status",
            timeout=8.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_status failed: {r.text}"}
        return r.json()

    @tool
    async def composio_authorize_toolkit(toolkit: str, callback_url: str = "") -> Any:
        """Create a Composio auth link for the active user. Share redirect_url with the user so they can finish OAuth."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/authorize",
            json_body={
                "customer_id": customer_id,
                "toolkit": toolkit,
                "callback_url": callback_url,
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_authorize_toolkit failed: {r.text}"}
        payload = r.json()
        redirect_url = str(payload.get("redirect_url", "") or "").strip()
        if redirect_url:
            payload["message"] = (
                f"Open this authorization link to connect {toolkit}: {redirect_url}"
            )
        return payload

    @tool
    async def composio_wait_for_connection(
        connection_id: str,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """Wait for a Composio connection to become active after the user finishes OAuth."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/wait_for_connection",
            json_body={
                "connection_id": connection_id,
                "timeout_seconds": max(1.0, min(float(timeout_seconds), 600.0)),
            },
            timeout=max(10.0, min(float(timeout_seconds) + 5.0, 605.0)),
            retries=0,
        )
        if r.status_code != 200:
            return {"error": f"composio_wait_for_connection failed: {r.text}"}
        return r.json().get("connection", {})

    @tool
    async def composio_toolkits(
        toolkits: list[str] | None = None,
        is_connected: str = "",
        limit: int = 50,
        search: str = "",
    ) -> Any:
        """List Composio toolkit connection state for the active user."""
        customer_id = _require_customer_id(runtime)
        params: dict[str, Any] = {
            "customer_id": customer_id,
            "toolkits": ",".join(toolkits or []),
            "limit": max(1, min(int(limit), 100)),
            "search": str(search or "").strip(),
        }
        if str(is_connected or "").strip():
            params["is_connected"] = str(is_connected).strip()
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/composio/toolkits",
            params=params,
            timeout=15.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_toolkits failed: {r.text}"}
        return r.json().get("items", [])

    @tool
    async def composio_connected_accounts(
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> Any:
        """List Composio connected accounts for the active user."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/composio/connected_accounts",
            params={
                "customer_id": customer_id,
                "toolkits": ",".join(toolkits or []),
                "statuses": ",".join(statuses or []),
                "limit": max(1, min(int(limit), 100)),
            },
            timeout=15.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_connected_accounts failed: {r.text}"}
        return r.json().get("items", [])

    @tool
    async def composio_disable_connected_account(connected_account_id: str) -> Any:
        """Disable a Composio connected account so OpenTulpa stops using it."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/connected_accounts/disable",
            json_body={"connected_account_id": str(connected_account_id or "").strip()},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_disable_connected_account failed: {r.text}"}
        return r.json().get("connected_account", {})

    @tool
    async def composio_delete_connected_account(connected_account_id: str) -> Any:
        """Delete a Composio connected account permanently."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/connected_accounts/delete",
            json_body={"connected_account_id": str(connected_account_id or "").strip()},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_delete_connected_account failed: {r.text}"}
        return r.json().get("connected_account", {})

    @tool
    async def composio_tool_search(
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> Any:
        """Search Composio tools and return candidate tool slugs, descriptions, and input schemas."""
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/composio/tools/search",
            params={
                "query": str(query or "").strip(),
                "toolkits": ",".join(toolkits or []),
                "limit": max(1, min(int(limit), 50)),
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_tool_search failed: {r.text}"}
        return r.json().get("items", [])

    @tool
    async def composio_tool_schema(tool_slug: str) -> Any:
        """Get the input schema for a single Composio tool slug."""
        r = await runtime._request_with_backoff(
            "GET",
            f"/internal/composio/tools/{tool_slug}/schema",
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_tool_schema failed: {r.text}"}
        return r.json().get("tool", {})

    @tool
    async def composio_instagram_reply_precheck(
        recipient_id: str = "",
        conversation_id: str = "",
        connected_account_id: str = "",
        scan_limit: int = 10,
    ) -> Any:
        """Verify the exact Instagram thread for a recipient and capture the latest inbound timestamp before attempting a DM send."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/instagram/reply_precheck",
            json_body={
                "customer_id": customer_id,
                "recipient_id": str(recipient_id or "").strip(),
                "conversation_id": str(conversation_id or "").strip(),
                "connected_account_id": str(connected_account_id or "").strip(),
                "scan_limit": max(1, min(int(scan_limit), 25)),
            },
            timeout=60.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_instagram_reply_precheck failed: {r.text}"}
        return r.json()

    @tool
    async def composio_tool_execute(
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str = "",
        text: str = "",
    ) -> Any:
        """Execute a Composio tool for the active user using explicit JSON arguments from the tool schema. For Instagram sends, verify the exact thread first with composio_instagram_reply_precheck."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/tools/execute",
            json_body={
                "customer_id": customer_id,
                "tool_slug": tool_slug,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "connected_account_id": connected_account_id,
                "text": text,
            },
            timeout=120.0,
        )
        if r.status_code != 200:
            return {"error": f"composio_tool_execute failed: {r.text}"}
        return r.json()

    @tool
    async def directive_get() -> Any:
        """Get the active persistent directive profile for this user."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/get",
            json_body=CustomerScopedRequest(customer_id=customer_id).model_dump(mode="json"),
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_get failed: {r.text}"}
        return DirectiveGetResponse.model_validate(r.json()).model_dump(mode="json")

    @tool
    async def directive_set(directive: str) -> Any:
        """Set or overwrite the user's persistent directive profile."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/set",
            json_body=DirectiveSetRequest(
                customer_id=customer_id,
                directive=directive,
                source="langgraph_tool",
            ).model_dump(mode="json"),
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_set failed: {r.text}"}
        payload = CustomerScopedOkResponse.model_validate(r.json()).model_dump(mode="json")
        heartbeat = await _sync_proactive_heartbeat(
            runtime=runtime,
            customer_id=customer_id,
            directive_text=directive,
        )
        payload["proactive_heartbeat"] = heartbeat
        return payload

    @tool
    async def directive_clear() -> Any:
        """Clear the user's persistent directive profile."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/clear",
            json_body=CustomerScopedRequest(customer_id=customer_id).model_dump(mode="json"),
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_clear failed: {r.text}"}
        payload = CustomerScopedClearResponse.model_validate(r.json()).model_dump(mode="json")
        heartbeat = await _sync_proactive_heartbeat(
            runtime=runtime,
            customer_id=customer_id,
            directive_text="disable proactive mode",
        )
        payload["proactive_heartbeat"] = heartbeat
        return payload

    @tool
    async def time_profile_get() -> Any:
        """Get stored user UTC offset (if known)."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/time_profile/get",
            json_body=CustomerScopedRequest(customer_id=customer_id).model_dump(mode="json"),
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"time_profile_get failed: {r.text}"}
        return TimeProfileGetResponse.model_validate(r.json()).model_dump(mode="json")

    @tool
    async def time_profile_set(utc_offset: str) -> Any:
        """Set user UTC offset in +HH:MM or -HH:MM format."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/time_profile/set",
            json_body=TimeProfileSetRequest(
                customer_id=customer_id,
                utc_offset=utc_offset,
                source="langgraph_tool",
            ).model_dump(mode="json"),
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"time_profile_set failed: {r.text}"}
        return TimeProfileSetResponse.model_validate(r.json()).model_dump(mode="json")

    @tool
    async def web_search(query: str) -> Any:
        """Search the web for current information."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/web_search",
            json_body={"query": query},
            timeout=90.0,
        )
        if r.status_code != 200:
            return {"error": "web_search request failed"}
        return r.json().get("result", "No result.")

    @tool
    async def browser_use_session_list() -> Any:
        """
        List known Browser Use sessions so the agent can reuse an idle session_id
        instead of spawning a fresh browser session.
        """
        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_session_list unavailable"}
        return {"sessions": await manager.list_sessions()}

    @tool
    async def browser_use_run(
        task: str,
        allowed_domains: list[str] | None = None,
        max_steps: int = 20,
        wait_timeout_seconds: int = 600,
        poll_interval_seconds: int = 4,
        llm: str = "browser-use-llm",
        start_url: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """
        Run a local Browser Use task and wait for completion.
        Use for dynamic web tasks that need real browser interactions.
        Reuse a prior session_id when continuing the same browsing workflow.
        """
        task_text = str(task or "").strip()
        if not task_text:
            return {"error": "browser_use_run requires a non-empty task"}
        _require_customer_id(runtime)

        safe_max_steps = max(1, min(int(max_steps), 80))
        safe_wait_timeout = max(30, min(int(wait_timeout_seconds), 1800))
        safe_poll_interval = max(2, min(int(poll_interval_seconds), 30))
        safe_domains = _normalize_allowed_domains(allowed_domains)
        safe_llm = str(llm or "").strip() or "browser-use-llm"
        safe_start_url = str(start_url or "").strip()
        safe_session_id = str(session_id or "").strip()

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_run unavailable"}

        created = await manager.start_task(
            task=task_text,
            max_steps=safe_max_steps,
            llm=safe_llm,
            allowed_domains=safe_domains,
            start_url=safe_start_url or None,
            session_id=safe_session_id or None,
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
            if status in {"finished", "stopped", "failed"}:
                compact = _compact_browser_use_task_view(task_data)
                compact["task_id"] = task_id
                compact["session_id"] = result_session_id or compact.get("session_id")
                compact["status"] = status or str(compact.get("status") or "unknown")
                compact["live_url"] = None
                return compact

            if datetime.now(UTC).timestamp() >= deadline:
                return {
                    "task_id": task_id,
                    "session_id": result_session_id or None,
                    "status": status or "started",
                    "timed_out": True,
                    "message": (
                        "Task is still running. Use browser_use_task_get(task_id) "
                        "to check progress or browser_use_task_control to stop."
                    ),
                }

            await asyncio.sleep(safe_poll_interval)

    @tool
    async def browser_use_task_get(
        task_id: str,
        include_steps: bool = False,
        max_steps_preview: int = 3,
    ) -> Any:
        """Get Browser Use task status/details by task_id (compact by default)."""
        safe_task_id = str(task_id or "").strip()
        if not safe_task_id:
            return {"error": "browser_use_task_get requires task_id"}

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_task_get unavailable"}

        payload = await manager.get_task(safe_task_id)
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

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_task_screenshot unavailable"}

        payload = await manager.capture_screenshot(
            task_id=safe_task_id,
            full_page=bool(full_page),
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

        manager, manager_error = _get_browser_use_local_manager(runtime)
        if manager is None:
            return {"error": manager_error or "browser_use_task_control unavailable"}

        payload = await manager.control_task(task_id=safe_task_id, action=safe_action)
        if isinstance(payload, dict) and payload.get("error"):
            return {"error": str(payload.get("error"))}
        return _compact_browser_use_task_view(payload if isinstance(payload, dict) else {})

    async def _fetch_remote_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
        target: str = "url",
    ) -> Any:
        """Fetch and extract content from URL based on target type (url|file)."""
        raw_url = str(url or "").strip()
        parsed = urlparse(raw_url)
        if parsed.scheme not in {"http", "https"}:
            return {"error": "url must start with http:// or https://"}

        safe_max_chars = max(2000, min(int(max_chars), 120000))
        try:
            async with httpx.AsyncClient(
                timeout=45.0,
                follow_redirects=True,
                headers={"User-Agent": "OpenTulpa/0.1 (+content-fetch)"},
            ) as client:
                resp = await client.get(raw_url)
        except Exception as exc:
            return {"error": f"link fetch failed: {exc}"}

        if resp.status_code >= 400:
            return {"error": f"link fetch failed: HTTP {resp.status_code}"}

        ctype = str(resp.headers.get("content-type", "")).split(";")[0].strip().lower()
        final_url = str(resp.url)
        text_content = ""
        title: str | None = None
        mode = "text"
        is_image = ctype.startswith("image/")
        is_pdf = ctype == "application/pdf" or final_url.lower().endswith(".pdf")
        is_docx = (
            ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            or final_url.lower().endswith(".docx")
        )
        file_like = is_image or is_pdf or is_docx

        safe_target = str(target or "url").strip().lower()
        if safe_target == "url" and file_like:
            return {
                "error": (
                    "URL points to a file-like resource (image/pdf/docx). "
                    "Use fetch_file_content instead."
                ),
                "url": final_url,
                "content_type": ctype or "unknown",
            }
        if safe_target == "file" and not file_like:
            return {
                "error": (
                    "URL does not point to supported file-like content (image/pdf/docx). "
                    "Use fetch_url_content instead."
                ),
                "url": final_url,
                "content_type": ctype or "unknown",
            }

        try:
            if is_image:
                mode = "image_vision"
                if use_vision_for_images:
                    vision = await runtime._model.ainvoke(
                        [
                            SystemMessage(
                                content=(
                                    "Describe the image and extract all readable text. "
                                    "If it is a screenshot/document, summarize key points."
                                )
                            ),
                            HumanMessage(
                                content=[
                                    {"type": "text", "text": "Analyze this image URL."},
                                    {"type": "image_url", "image_url": {"url": final_url}},
                                ]
                            ),
                        ]
                    )
                    text_content = _content_to_text(getattr(vision, "content", "")).strip()
                else:
                    text_content = ""
            elif is_pdf:
                mode = "pdf_llm"
                text_content = await summarize_uploaded_blob(
                    runtime,
                    filename=final_url.rsplit("/", 1)[-1] or "document.pdf",
                    mime_type=ctype or "application/pdf",
                    kind="document",
                    raw_bytes=resp.content,
                    question=(
                        "Extract key information from this PDF and provide a concise but complete "
                        "summary with important facts, entities, dates, and actions."
                    ),
                )
            elif is_docx:
                mode = "docx_llm"
                text_content = await summarize_uploaded_blob(
                    runtime,
                    filename=final_url.rsplit("/", 1)[-1] or "document.docx",
                    mime_type=ctype
                    or "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    kind="document",
                    raw_bytes=resp.content,
                    question=(
                        "Extract key information from this DOCX and provide a concise but complete "
                        "summary with important facts, entities, dates, and actions."
                    ),
                )
            else:
                mode = "web_text"
                raw_text = resp.text
                if "html" in ctype or "<html" in raw_text.lower():
                    crawled_text, crawled_title, crawl_error = await _crawl4ai_extract(final_url)
                    if crawled_text:
                        mode = "web_text_crawl4ai"
                        title = crawled_title or _extract_html_title(raw_text)
                        text_content = crawled_text
                    else:
                        mode = "web_text_fallback"
                        title = _extract_html_title(raw_text)
                        text_content = _html_to_text(raw_text)
                        if crawl_error:
                            text_content = (
                                f"[crawl4ai_fallback_reason] {crawl_error}\n\n{text_content}"
                            )
                else:
                    text_content = raw_text
        except Exception as exc:
            return {
                "error": f"content extraction failed: {exc}",
                "url": final_url,
                "content_type": ctype or "unknown",
            }

        normalized = re.sub(r"\n{3,}", "\n\n", str(text_content or "").strip())
        truncated = len(normalized) > safe_max_chars
        return {
            "url": final_url,
            "content_type": ctype or "unknown",
            "mode": mode,
            "title": title,
            "char_count": len(normalized),
            "truncated": truncated,
            "text": normalized[:safe_max_chars],
        }

    @tool
    async def fetch_url_content(url: str, max_chars: int = 40000) -> Any:
        """Fetch and extract web page/text/JSON content from a URL."""
        return await _fetch_remote_content(
            url=url,
            max_chars=max_chars,
            use_vision_for_images=False,
            target="url",
        )

    @tool
    async def fetch_file_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
    ) -> Any:
        """Fetch and analyze file-like URL content (image/pdf/docx)."""
        return await _fetch_remote_content(
            url=url,
            max_chars=max_chars,
            use_vision_for_images=use_vision_for_images,
            target="file",
        )

    @tool
    async def tulpa_write_file(path: str, content: str) -> Any:
        """Write file in approved paths."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/write_file",
            json_body={"path": path, "content": content},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"write failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_validate_file(path: str) -> Any:
        """Validate generated file syntax/contracts in approved paths."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/validate_file",
            json_body={"path": path},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"validation failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_reload() -> Any:
        """Reload tulpa_stuff routers so newly written connectors become active."""
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/reload",
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"reload failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_run_terminal(
        command: str,
        working_dir: str = "tulpa_stuff",
        timeout_seconds: int = 90,
        thread_id: str = "",
        execution_origin: str | None = None,
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """Run executable shell/script command through execution-boundary guard."""
        safe_working_dir = str(working_dir or "").strip() or "tulpa_stuff"
        safe_command = _normalize_command_for_working_dir(
            command=str(command or "").strip(),
            working_dir=safe_working_dir,
        )
        if not _looks_like_shell_command(safe_command):
            return {
                "error": (
                    "Command rejected: provide a concrete shell command (executable + args), "
                    "not natural language."
                )
            }
        safe_timeout = max(5, min(int(timeout_seconds), 600))
        safe_customer = _require_customer_id(runtime)
        safe_thread = str(thread_id or "").strip()
        normalized_origin = _normalize_execution_origin(
            thread_id=safe_thread,
            execution_origin=execution_origin,
        )

        guard_payload = guard_context if isinstance(guard_context, dict) else {}
        previous_user = str(guard_payload.get("previous_user_message", "")).strip()
        previous_assistant = str(guard_payload.get("previous_assistant_message", "")).strip()
        decision = await boundary_guard.evaluate(
            ExecutionBoundaryContext(
                customer_id=safe_customer,
                thread_id=safe_thread or (f"chat-{safe_customer}" if safe_customer else "interactive"),
                action_name="tulpa_run_terminal",
                action_args={
                    "command": safe_command,
                    "working_dir": safe_working_dir,
                    "timeout_seconds": safe_timeout,
                    "execution_origin": normalized_origin,
                },
                execution_origin=normalized_origin,
                preapproved=bool(preapproved),
                action_note=(
                    "Execution-boundary guard check for terminal/script action. "
                    "Decide based on full command external write side effects. "
                    f"previous_user_message={previous_user[:800]} "
                    f"previous_assistant_message={previous_assistant[:800]}"
                ),
            )
        )
        gate = str((decision or {}).get("gate", "allow")).strip().lower()
        if gate == "require_approval":
            return _approval_pending_payload(
                action_name="tulpa_run_terminal",
                command_preview=safe_command,
                decision=decision if isinstance(decision, dict) else {},
            )
        if gate == "deny":
            return {
                "ok": False,
                "status": "denied",
                "gate": "deny",
                "reason": str((decision or {}).get("reason", "guardrail_denied")).strip(),
            }

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/tulpa/run_terminal",
            json_body={
                "command": safe_command,
                "working_dir": safe_working_dir,
                "timeout_seconds": safe_timeout,
            },
            timeout=max(10.0, float(safe_timeout) + 10.0),
            retries=1,
        )
        if r.status_code != 200:
            return {"error": f"terminal failed: {r.text}"}
        payload = r.json()
        if isinstance(payload, dict):
            payload["execution_origin"] = normalized_origin
        return payload

    @tool
    async def tulpa_read_file(path: str, max_chars: int = 12000) -> Any:
        """Read file in approved paths."""
        safe_max_chars = max(500, min(int(max_chars), 20000))
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/tulpa/read_file",
            params={"path": path, "max_chars": safe_max_chars},
            timeout=15.0,
        )
        if r.status_code != 200:
            return {"error": f"read failed: {r.text}"}
        return r.json()

    @tool
    async def tulpa_catalog() -> Any:
        """Get catalog of tracked files and artifacts."""
        r = await runtime._request_with_backoff("GET", "/internal/tulpa/catalog", timeout=10.0)
        if r.status_code != 200:
            return {"error": f"catalog failed: {r.text}"}
        return r.json().get("catalog", {})

    @tool
    async def task_status(task_id: str) -> Any:
        """Get task status."""
        r = await runtime._request_with_backoff("GET", f"/internal/tasks/{task_id}", timeout=10.0)
        if r.status_code != 200:
            return {"error": f"task_status failed: {r.text}"}
        return r.json().get("task", {})

    @tool
    async def task_events(task_id: str, limit: int = 30, offset: int = 0) -> Any:
        """Get task events."""
        r = await runtime._request_with_backoff(
            "GET",
            f"/internal/tasks/{task_id}/events",
            params={"limit": max(1, min(int(limit), 200)), "offset": max(0, int(offset))},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"task_events failed: {r.text}"}
        return r.json().get("events", [])

    @tool
    async def task_artifacts(task_id: str) -> Any:
        """Get task artifacts."""
        r = await runtime._request_with_backoff(
            "GET", f"/internal/tasks/{task_id}/artifacts", timeout=10.0
        )
        if r.status_code != 200:
            return {"error": f"task_artifacts failed: {r.text}"}
        return r.json().get("artifacts", [])

    @tool
    async def task_relaunch(
        task_id: str, clarification: str | None = None, trigger_reason: str = "user_requested"
    ) -> Any:
        """Relaunch a task."""
        r = await runtime._request_with_backoff(
            "POST",
            f"/internal/tasks/{task_id}/relaunch",
            json_body={"clarification": clarification, "trigger_reason": trigger_reason},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"task_relaunch failed: {r.text}"}
        return r.json().get("task", {})

    @tool
    async def task_cancel(task_id: str) -> Any:
        """Cancel a task."""
        r = await runtime._request_with_backoff(
            "POST", f"/internal/tasks/{task_id}/cancel", timeout=10.0
        )
        if r.status_code != 200:
            return {"error": f"task_cancel failed: {r.text}"}
        return r.json().get("task", {})

    @tool
    async def routine_create(
        name: str,
        schedule: str,
        implementation_command: str,
        instruction: str,
        notify_user: bool = True,
        cleanup_paths: list[str] | None = None,
        thread_id: str = "",
        execution_origin: str | None = None,
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """
        Create a scheduled routine.
        - Recurring: cron (e.g. "0 9 * * *")
        - One-time: local ISO datetime (e.g. "2026-02-18T23:45:00+08:00")
        - instruction: explicit schedule-time scratchpad for each run. Include required scripts,
          files/paths, keys to read from storage, and expected output/action.
        - implementation_command: planned shell/script command used for guardrail evaluation.
        - cleanup_paths: optional repo-relative file paths to remove when deleting this automation.
        """
        safe_name = str(name or "").strip()
        safe_schedule = str(schedule or "").strip()
        safe_instruction = str(instruction or "").strip()
        safe_command = _normalize_command_for_working_dir(
            command=str(implementation_command or "").strip(),
            working_dir="tulpa_stuff",
        )
        safe_customer = _require_customer_id(runtime)
        if not safe_name:
            return {"error": "routine_create failed: name is required"}
        if not safe_schedule:
            return {"error": "routine_create failed: schedule is required"}
        if not safe_instruction:
            return {"error": "routine_create failed: instruction is required"}
        if not safe_command:
            return {
                "error": (
                    "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create requires "
                    "implementation_command (concrete shell/script command)."
                )
            }
        if not _looks_like_shell_command(safe_command):
            return {
                "error": (
                    "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must be a "
                    "concrete shell command (executable + args)."
                )
            }

        normalized_origin = _normalize_execution_origin(
            thread_id=thread_id,
            execution_origin=execution_origin,
        )

        guard_payload = guard_context if isinstance(guard_context, dict) else {}
        previous_user = str(guard_payload.get("previous_user_message", "")).strip()
        previous_assistant = str(guard_payload.get("previous_assistant_message", "")).strip()
        decision = await boundary_guard.evaluate(
            ExecutionBoundaryContext(
                customer_id=safe_customer,
                thread_id=str(thread_id or "").strip() or f"chat-{safe_customer}",
                action_name="routine_create",
                action_args={
                    "name": safe_name,
                    "schedule": safe_schedule,
                    "instruction": safe_instruction[:1200],
                    "notify_user": bool(notify_user),
                    "implementation_command": safe_command,
                },
                execution_origin=normalized_origin,
                preapproved=bool(preapproved),
                action_note=(
                    "Routine creation with planned implementation command. "
                    "Classify external write side effects for future scheduled behavior. "
                    f"previous_user_message={previous_user[:800]} "
                    f"previous_assistant_message={previous_assistant[:800]}"
                ),
            )
        )
        gate = str((decision or {}).get("gate", "allow")).strip().lower()
        if gate == "require_approval":
            return _approval_pending_payload(
                action_name="routine_create",
                command_preview=safe_command,
                decision=decision if isinstance(decision, dict) else {},
            )
        if gate == "deny":
            return {
                "ok": False,
                "status": "denied",
                "gate": "deny",
                "reason": str((decision or {}).get("reason", "guardrail_denied")).strip(),
            }

        auto_notify = bool(notify_user)
        safe_cleanup_paths = _normalize_cleanup_paths(cleanup_paths)

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/scheduler/routine",
            json_body={
                "name": safe_name,
                "schedule": safe_schedule,
                "payload": {
                    "instruction": safe_instruction,
                    "customer_id": safe_customer,
                    "notify_user": auto_notify,
                    "notification_opt_out": not auto_notify,
                    "cleanup_paths": safe_cleanup_paths,
                },
                "is_cron": " " in safe_schedule and len(safe_schedule.split()) >= 5,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_create failed: {r.text}"}
        return r.json()

    @tool
    async def routine_list() -> Any:
        """List routines for the current user."""
        customer_id = _require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "GET",
            "/internal/scheduler/routines",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_list failed: {r.text}"}
        return r.json().get("routines", [])

    @tool
    async def routine_delete(routine_id: str) -> Any:
        """Delete/stop one routine by id for the current user."""
        customer_id = _require_customer_id(runtime)
        rid = str(routine_id or "").strip()
        if not rid:
            return {"error": "routine_delete failed: routine_id is required"}

        r = await runtime._request_with_backoff(
            "DELETE",
            f"/internal/scheduler/routine/{rid}",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"routine_delete failed: {r.text}"}
        payload = r.json() if r.content else {}
        if not bool(payload.get("ok")):
            return {
                "error": "routine_delete failed: routine not found or not accessible",
                "routine_id": rid,
            }

        verify = await runtime._request_with_backoff(
            "GET",
            "/internal/scheduler/routines",
            params={"customer_id": customer_id},
            timeout=10.0,
        )
        if verify.status_code != 200:
            return {
                "ok": True,
                "routine_id": rid,
                "verified_removed": False,
                "warning": "delete succeeded but verification list failed",
            }
        routines = verify.json().get("routines", [])
        still_present = any(str(item.get("id", "")) == rid for item in routines if isinstance(item, dict))
        return {
            "ok": not still_present,
            "routine_id": rid,
            "verified_removed": not still_present,
            "remaining_routines": routines,
        }

    @tool
    async def guardrail_execute_approved_action(approval_id: str) -> Any:
        """Execute a previously approved external-impact action exactly once."""
        customer_id = _require_customer_id(runtime)
        aid = str(approval_id or "").strip()
        if not aid:
            return {"error": "guardrail_execute_approved_action requires approval_id"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/approvals/execute",
            json_body={"approval_id": aid, "customer_id": customer_id},
            timeout=600.0,
            retries=0,
        )
        if r.status_code != 200:
            return {"error": f"guardrail_execute_approved_action failed: {r.text}"}
        return r.json()

    @tool
    async def server_time() -> Any:
        """Get server time."""
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(UTC)
        return {
            "server_time_local_iso": now_local.isoformat(),
            "server_timezone": str(now_local.tzinfo),
            "server_time_utc_iso": now_utc.isoformat(),
            "unix_timestamp": int(now_utc.timestamp()),
        }

    return {
        "memory_search": memory_search,
        "memory_add": memory_add,
        "uploaded_file_search": uploaded_file_search,
        "uploaded_file_get": uploaded_file_get,
        "uploaded_file_send": uploaded_file_send,
        "tulpa_file_send": tulpa_file_send,
        "web_image_send": web_image_send,
        "uploaded_file_analyze": uploaded_file_analyze,
        "skill_list": skill_list,
        "skill_get": skill_get,
        "skill_upsert": skill_upsert,
        "skill_delete": skill_delete,
        "intake_workflow_upsert": intake_workflow_upsert,
        "intake_workflow_list": intake_workflow_list,
        "intake_workflow_get": intake_workflow_get,
        "intake_workflow_delete": intake_workflow_delete,
        "intake_workflow_run": intake_workflow_run,
        "telegram_business_status": telegram_business_status,
        "composio_status": composio_status,
        "composio_authorize_toolkit": composio_authorize_toolkit,
        "composio_wait_for_connection": composio_wait_for_connection,
        "composio_toolkits": composio_toolkits,
        "composio_connected_accounts": composio_connected_accounts,
        "composio_disable_connected_account": composio_disable_connected_account,
        "composio_delete_connected_account": composio_delete_connected_account,
        "composio_tool_search": composio_tool_search,
        "composio_tool_schema": composio_tool_schema,
        "composio_instagram_reply_precheck": composio_instagram_reply_precheck,
        "composio_tool_execute": composio_tool_execute,
        "directive_get": directive_get,
        "directive_set": directive_set,
        "directive_clear": directive_clear,
        "time_profile_get": time_profile_get,
        "time_profile_set": time_profile_set,
        "web_search": web_search,
        "browser_use_session_list": browser_use_session_list,
        "browser_use_run": browser_use_run,
        "browser_use_task_get": browser_use_task_get,
        "browser_use_task_screenshot": browser_use_task_screenshot,
        "browser_use_task_control": browser_use_task_control,
        "fetch_url_content": fetch_url_content,
        "fetch_file_content": fetch_file_content,
        "tulpa_write_file": tulpa_write_file,
        "tulpa_validate_file": tulpa_validate_file,
        "tulpa_reload": tulpa_reload,
        "tulpa_run_terminal": tulpa_run_terminal,
        "tulpa_read_file": tulpa_read_file,
        "tulpa_catalog": tulpa_catalog,
        "task_status": task_status,
        "task_events": task_events,
        "task_artifacts": task_artifacts,
        "task_relaunch": task_relaunch,
        "task_cancel": task_cancel,
        "routine_create": routine_create,
        "routine_list": routine_list,
        "routine_delete": routine_delete,
        "guardrail_execute_approved_action": guardrail_execute_approved_action,
        "server_time": server_time,
    }
