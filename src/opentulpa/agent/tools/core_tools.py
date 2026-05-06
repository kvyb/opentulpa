"""Core/general-purpose tool registration."""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

from opentulpa.agent.file_analysis import summarize_uploaded_blob
from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.tools.common import (
    normalize_command_for_working_dir,
    normalize_execution_origin,
    require_customer_id,
    require_thread_id,
)
from opentulpa.agent.utils import content_to_text as _content_to_text
from opentulpa.agent.utils import extract_html_title as _extract_html_title
from opentulpa.agent.utils import html_to_text as _html_to_text
from opentulpa.agent.utils import looks_like_shell_command as _looks_like_shell_command
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


def _tool_error_payload(tool_name: str, response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        return {"error": f"{tool_name} failed: {response.text}"}
    if not isinstance(payload, dict):
        return {"error": f"{tool_name} failed: {response.text}"}
    payload = dict(payload)
    payload["error"] = f"{tool_name} failed ({response.status_code})"
    return payload


_MISSING_MODULE_RE = re.compile(
    r"(?:ModuleNotFoundError: No module named|ImportError: No module named) ['\"]([^'\"]+)['\"]"
)


def _decorate_python_dependency_failure(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    stderr = str(payload.get("stderr", "") or "").strip()
    match = _MISSING_MODULE_RE.search(stderr)
    if not match:
        return payload
    missing_module = match.group(1).strip()
    safe_payload = dict(payload)
    safe_payload["missing_python_module"] = missing_module
    safe_payload["agent_hint"] = (
        "Missing Python dependency in .opentulpa/agent_venv. "
        "If this package is needed for the task, install it in that venv and retry once. "
        "Otherwise report the dependency blocker clearly."
    )
    return safe_payload


def _with_delivery_instruction(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if payload.get("delivered_to_chat") is not True:
        return payload
    safe_payload = dict(payload)
    safe_payload["delivery_status"] = "delivered_to_telegram_chat"
    safe_payload.setdefault(
        "model_instruction",
        (
            "DELIVERED_TO_CHAT: The file has been sent to Telegram. "
            "Do not call the file-send tool again for this file. "
            "Continue with a short final confirmation only."
        ),
    )
    return safe_payload


def _trim_tool_text(value: Any, *, limit: int = 160) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 15)].rstrip() + " ...[truncated]"


def _compact_file_record_for_tool(raw: Any) -> dict[str, Any]:
    record = raw if isinstance(raw, dict) else {}
    summary = str(record.get("summary", "") or "").strip()
    if " | content_preview=" in summary:
        summary = summary.split(" | content_preview=", 1)[0].strip()
    return {
        "id": str(record.get("id", "") or "").strip(),
        "filename": str(record.get("original_filename", "") or "").strip(),
        "mime_type": str(record.get("mime_type", "") or "").strip(),
        "size_bytes": record.get("size_bytes"),
        "summary": _trim_tool_text(summary, limit=320),
    }


def _compact_row_view(raw: Any, *, value_limit: int = 120, max_values: int = 8) -> dict[str, Any]:
    row = raw if isinstance(raw, dict) else {}
    values = row.get("values") if isinstance(row.get("values"), list) else []
    return {
        key: value
        for key, value in {
            "source_ref": row.get("source_ref"),
            "row": row.get("row"),
            "values": [_trim_tool_text(value, limit=value_limit) for value in values[:max_values]],
        }.items()
        if value not in (None, "", [])
    }


def _compact_uploaded_file_inspection(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    inspection = payload.get("inspection") if isinstance(payload.get("inspection"), dict) else {}
    structure = inspection.get("structure") if isinstance(inspection.get("structure"), dict) else {}
    raw_sheets = structure.get("sheets") if isinstance(structure.get("sheets"), list) else []
    sheet_inventory: list[dict[str, Any]] = []
    relevant_sheets: list[dict[str, Any]] = []
    for raw_sheet in raw_sheets:
        if not isinstance(raw_sheet, dict):
            continue
        matches = raw_sheet.get("matches") if isinstance(raw_sheet.get("matches"), list) else []
        sample_rows = raw_sheet.get("sample_rows") if isinstance(raw_sheet.get("sample_rows"), list) else []
        table_candidates = (
            raw_sheet.get("table_candidates")
            if isinstance(raw_sheet.get("table_candidates"), list)
            else []
        )
        compact_sheet = {
            "index": raw_sheet.get("index"),
            "name": str(raw_sheet.get("name", "") or "").strip(),
            "matched_terms": raw_sheet.get("matched_terms") or [],
            "max_row": raw_sheet.get("max_row"),
            "max_column": raw_sheet.get("max_column"),
            "nonempty_rows": raw_sheet.get("nonempty_rows"),
        }
        sheet_inventory.append(compact_sheet)
        is_relevant = bool(raw_sheet.get("matched_terms") or matches or sample_rows or table_candidates)
        if is_relevant:
            relevant = dict(compact_sheet)
            relevant["sample_rows"] = [_compact_row_view(row) for row in sample_rows[:3]]
            relevant["matches"] = [_compact_row_view(row) for row in matches[:4]]
            relevant["table_candidates"] = [
                {
                    key: value
                    for key, value in {
                        "sheet_name": item.get("sheet_name") if isinstance(item, dict) else None,
                        "row_start": item.get("row_start") if isinstance(item, dict) else None,
                        "row_end": item.get("row_end") if isinstance(item, dict) else None,
                        "sample_rows": [
                            _compact_row_view(row, value_limit=80, max_values=6)
                            for row in (
                                item.get("sample_rows") if isinstance(item, dict) and isinstance(item.get("sample_rows"), list) else []
                            )[:1]
                        ],
                    }.items()
                    if value not in (None, "", [])
                }
                for item in table_candidates[:3]
                if isinstance(item, dict)
            ]
            omitted = str(raw_sheet.get("omitted_detail_reason", "") or "").strip()
            if omitted:
                relevant["omitted_detail_reason"] = omitted
            relevant_sheets.append(relevant)
    return {
        "ok": bool(payload.get("ok", False)),
        "file": _compact_file_record_for_tool(payload.get("file")),
        "inspection": {
            "filename": str(inspection.get("filename", "") or "").strip(),
            "mime_type": str(inspection.get("mime_type", "") or "").strip(),
            "format": str(inspection.get("format", "") or "").strip(),
            "warnings": inspection.get("warnings") or [],
            "structure": {
                "sheet_inventory": sheet_inventory,
                "relevant_sheets": relevant_sheets,
                "selection_format": structure.get("selection_format") or {},
            },
        },
        "model_note": (
            "Inspection is compacted for the model. Use sheet_inventory and relevant_sheets "
            "to understand the workbook before indexing; full source bytes stay in the file vault."
        ),
    }


def _compact_business_knowledge_query(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    return {
        "query": _trim_tool_text(payload.get("query", ""), limit=600),
        "answer_extract": _trim_tool_text(payload.get("answer_extract", ""), limit=3000),
    }


def _compact_business_knowledge_index(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    sources: list[dict[str, Any]] = []
    for source in list(payload.get("sources") or [])[:20]:
        if not isinstance(source, dict):
            continue
        sources.append(
            {
                "file_id": str(source.get("file_id", "") or "").strip(),
                "filename": str(source.get("filename", "") or "").strip(),
                "status": str(source.get("status", "") or "").strip(),
                "source_kind": str(source.get("source_kind", "") or "").strip(),
                "section_count": source.get("section_count"),
                "char_count": source.get("char_count"),
                "warnings": source.get("warnings") or [],
            }
        )
    return {
        "ok": bool(payload.get("ok", False)),
        "scope_type": str(payload.get("scope_type", "") or "").strip(),
        "scope_id": str(payload.get("scope_id", "") or "").strip(),
        "sources": sources,
    }


async def _resolve_business_knowledge_scope(
    runtime: Any,
    *,
    scope_type: str,
    scope_id: str,
) -> tuple[str, str] | dict[str, str]:
    requested_type = str(scope_type or "current_workflow").strip().lower() or "current_workflow"
    requested_id = str(scope_id or "").strip()
    valid_types = {"workflow_setup", "intake_workflow", "customer_business", "user_context"}
    if requested_type in valid_types:
        if requested_type in {"customer_business", "user_context"} and not requested_id:
            return requested_type, require_customer_id(runtime)
        if requested_id:
            return requested_type, requested_id
        return {"error": f"{requested_type} scope requires scope_id"}

    customer_id = require_customer_id(runtime)
    thread_id = require_thread_id(runtime)
    with suppress(Exception):
        setup_response = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/get",
            json_body={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "include_paused": True,
            },
            timeout=5.0,
            retries=0,
        )
        if setup_response.status_code == 200:
            payload = setup_response.json()
            session = payload.get("session") if isinstance(payload, dict) else None
            if isinstance(session, dict):
                session_id = str(session.get("session_id", "") or "").strip()
                if session_id and (not requested_id or requested_id == session_id):
                    return "workflow_setup", session_id

    if requested_id:
        return "intake_workflow", requested_id

    match = re.search(r"(iwf_[A-Za-z0-9]+)", thread_id)
    if match:
        return "intake_workflow", match.group(1)
    return {
        "error": (
            "business knowledge scope could not be inferred; pass scope_type and scope_id "
            "or start/open a workflow setup session"
        )
    }


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
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore[import-untyped]
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


def register_core_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def send_owner_update(message: str, dedupe_key: str = "") -> Any:
        """Send a short interim update to the current owner/support Telegram chat.

        Use only during live owner/support turns when you will continue working
        with tools. This is for long-running interactive or workflow setup work,
        not final answers, inbound lead replies, routine wakes, or background
        event notifications.
        """
        require_customer_id(runtime)
        require_thread_id(runtime)
        safe_message = str(message or "").strip()
        if not safe_message:
            return {"ok": False, "sent": False, "reason": "empty_message"}
        if len(safe_message) > 500:
            safe_message = safe_message[:497].rstrip() + "..."
        emitter = getattr(runtime, "emit_interactive_update", None)
        if not callable(emitter):
            return {"ok": False, "sent": False, "reason": "interactive_update_unavailable"}
        return await emitter(
            text=safe_message,
            dedupe_key=str(dedupe_key or "").strip(),
        )

    @tool
    async def memory_search(query: str) -> Any:
        """Search user memory."""
        customer_id = require_customer_id(runtime)
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
        customer_id = require_customer_id(runtime)
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
        customer_id = require_customer_id(runtime)
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
        customer_id = require_customer_id(runtime)
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
        customer_id = require_customer_id(runtime)
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
        return _with_delivery_instruction(r.json())

    @tool
    async def tulpa_file_send(
        path: str,
        caption: str | None = None,
    ) -> Any:
        """Send a local file from tulpa_stuff/ back to the user's Telegram chat."""
        customer_id = require_customer_id(runtime)
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
        return _with_delivery_instruction(r.json())

    @tool
    async def web_image_send(
        url: str,
        caption: str | None = None,
        max_bytes: int = 10_000_000,
    ) -> Any:
        """Download an image from a web URL and send it to Telegram."""
        customer_id = require_customer_id(runtime)
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
        return _with_delivery_instruction(r.json())

    @tool
    async def uploaded_file_analyze(
        file_id: str,
        question: str | None = None,
    ) -> Any:
        """Analyze a previously uploaded file again, optionally with a focused question.

        Do not use this as a fallback for workflow knowledge files that should be
        indexed and queried with business_knowledge_index/business_knowledge_query.
        """
        customer_id = require_customer_id(runtime)
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
    async def uploaded_file_inspect_structure(
        file_id: str,
        search_terms: list[str] | str | None = None,
    ) -> Any:
        """Inspect an uploaded file's structure before selecting workflow knowledge.

        Use this first for arbitrary spreadsheets or large source files. For XLSX files,
        it opens the workbook, returns sheet names, dimensions, sample rows, table
        candidates, and optional matches for search_terms derived from the user's
        workflow goal. For workflow knowledge, prefer business_knowledge_index and
        business_knowledge_query so the source file stays out of chat context.
        """
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/inspect_structure",
            json_body={
                "file_id": file_id,
                "customer_id": customer_id,
                "search_terms": search_terms,
            },
            timeout=60.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("uploaded_file_inspect_structure", r)
        return _compact_uploaded_file_inspection(r.json())

    @tool
    async def business_knowledge_index(
        file_ids: list[str],
        scope_type: str = "current_workflow",
        scope_id: str = "",
    ) -> Any:
        """Prepare uploaded source files as scoped business knowledge.

        Use this during workflow setup when uploaded files should become durable
        source knowledge for the workflow. If no setup session exists yet, call
        intake_workflow_setup_begin first. This normalizes source files into an
        LLM-readable knowledge pack. Bind original uploaded source file ids to
        draft_patch.knowledge_file_ids; do not create a summarized Markdown pack.
        """
        customer_id = require_customer_id(runtime)
        safe_file_ids = [
            str(item or "").strip()
            for item in list(file_ids or [])
            if str(item or "").strip()
        ][:20]
        if not safe_file_ids:
            return {"error": "business_knowledge_index failed: file_ids is required"}
        resolved = await _resolve_business_knowledge_scope(
            runtime,
            scope_type=scope_type,
            scope_id=scope_id,
        )
        if isinstance(resolved, dict):
            return resolved
        resolved_type, resolved_id = resolved
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/knowledge/index_sources",
            json_body={
                "customer_id": customer_id,
                "scope_type": resolved_type,
                "scope_id": resolved_id,
                "file_ids": safe_file_ids,
            },
            timeout=60.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("business_knowledge_index", r)
        return _compact_business_knowledge_index(r.json())

    @tool
    async def business_knowledge_query(
        query: str,
        scope_type: str = "current_workflow",
        scope_id: str = "",
    ) -> Any:
        """Ask scoped business knowledge for source-backed facts.

        Use this for source-backed business details during workflow setup or intake.
        The oracle reads the full prepared knowledge pack for the resolved scope and
        returns only a compact plain-text answer. During intake, do not call it again
        just to revalidate source-backed facts already present on the active booking.
        """
        customer_id = require_customer_id(runtime)
        safe_query = str(query or "").strip()
        if not safe_query:
            return {"error": "business_knowledge_query failed: query is required"}
        resolved = await _resolve_business_knowledge_scope(
            runtime,
            scope_type=scope_type,
            scope_id=scope_id,
        )
        if isinstance(resolved, dict):
            return resolved
        resolved_type, resolved_id = resolved
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/knowledge/query",
            json_body={
                "customer_id": customer_id,
                "scope_type": resolved_type,
                "scope_id": resolved_id,
                "query": safe_query,
                "max_extract_chars": 3000,
            },
            timeout=60.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("business_knowledge_query", r)
        return _compact_business_knowledge_query(r.json())

    @tool
    async def user_context_add_files(file_ids: list[str]) -> Any:
        """Add uploaded files to the durable interactive user context.

        Use this only when the user's recent instructions clearly ask to add files
        to their reusable context. If intent is unclear, ask what to do with the
        files instead of guessing from filenames or content.
        """
        customer_id = require_customer_id(runtime)
        safe_file_ids = [
            str(item or "").strip()
            for item in list(file_ids or [])
            if str(item or "").strip()
        ][:50]
        if not safe_file_ids:
            return {"error": "user_context_add_files failed: file_ids is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/add_files",
            json_body={"customer_id": customer_id, "file_ids": safe_file_ids},
            timeout=90.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_add_files", r)
        return r.json()

    @tool
    async def user_context_query(query: str, max_extract_chars: int = 3000) -> Any:
        """Query the durable interactive user context for grounded evidence."""
        customer_id = require_customer_id(runtime)
        safe_query = str(query or "").strip()
        if not safe_query:
            return {"error": "user_context_query failed: query is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/query",
            json_body={
                "customer_id": customer_id,
                "query": safe_query,
                "max_extract_chars": max(500, min(int(max_extract_chars), 5000)),
            },
            timeout=70.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_query", r)
        return r.json()

    @tool
    async def user_context_list_sources(include_archived: bool = False) -> Any:
        """List files that are currently registered in the interactive user context."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/list_sources",
            json_body={"customer_id": customer_id, "include_archived": bool(include_archived)},
            timeout=10.0,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_list_sources", r)
        return r.json().get("sources", [])

    @tool
    async def user_context_find_sources(query: str, limit: int = 10) -> Any:
        """Find user-context sources by filename, summary, or extracted text preview."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/find_sources",
            json_body={
                "customer_id": customer_id,
                "query": query,
                "limit": max(1, min(int(limit), 50)),
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_find_sources", r)
        return r.json().get("sources", [])

    @tool
    async def user_context_reindex(file_ids: list[str] | None = None) -> Any:
        """Reindex selected user-context files, or every active source when omitted."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/reindex",
            json_body={"customer_id": customer_id, "file_ids": file_ids or None},
            timeout=90.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_reindex", r)
        return r.json()

    @tool
    async def user_context_archive_sources(file_ids: list[str]) -> Any:
        """Archive selected files so default user-context queries stop using them."""
        customer_id = require_customer_id(runtime)
        safe_file_ids = [
            str(item or "").strip()
            for item in list(file_ids or [])
            if str(item or "").strip()
        ][:50]
        if not safe_file_ids:
            return {"error": "user_context_archive_sources failed: file_ids is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/archive_sources",
            json_body={"customer_id": customer_id, "file_ids": safe_file_ids},
            timeout=20.0,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_archive_sources", r)
        return r.json()

    @tool
    async def user_context_promote_to_intake(workflow_id: str, file_ids: list[str]) -> Any:
        """Index selected user-context files into an existing intake workflow scope."""
        customer_id = require_customer_id(runtime)
        safe_file_ids = [
            str(item or "").strip()
            for item in list(file_ids or [])
            if str(item or "").strip()
        ][:50]
        if not str(workflow_id or "").strip():
            return {"error": "user_context_promote_to_intake failed: workflow_id is required"}
        if not safe_file_ids:
            return {"error": "user_context_promote_to_intake failed: file_ids is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/user_context/promote_to_intake",
            json_body={
                "customer_id": customer_id,
                "workflow_id": str(workflow_id).strip(),
                "file_ids": safe_file_ids,
            },
            timeout=90.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("user_context_promote_to_intake", r)
        return r.json()

    @tool
    async def directive_get() -> Any:
        """Get the active persistent directive profile for this user."""
        customer_id = require_customer_id(runtime)
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
        customer_id = require_customer_id(runtime)
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
        payload["proactive_heartbeat"] = await _sync_proactive_heartbeat(
            runtime=runtime,
            customer_id=customer_id,
            directive_text=directive,
        )
        return payload

    @tool
    async def directive_clear() -> Any:
        """Clear the user's persistent directive profile."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/directive/clear",
            json_body=CustomerScopedRequest(customer_id=customer_id).model_dump(mode="json"),
            timeout=5.0,
        )
        if r.status_code != 200:
            return {"error": f"directive_clear failed: {r.text}"}
        payload = CustomerScopedClearResponse.model_validate(r.json()).model_dump(mode="json")
        payload["proactive_heartbeat"] = await _sync_proactive_heartbeat(
            runtime=runtime,
            customer_id=customer_id,
            directive_text="disable proactive mode",
        )
        return payload

    @tool
    async def time_profile_get() -> Any:
        """Get stored user UTC offset (if known)."""
        customer_id = require_customer_id(runtime)
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
        customer_id = require_customer_id(runtime)
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

    async def _fetch_remote_content(
        url: str,
        max_chars: int = 40000,
        use_vision_for_images: bool = True,
        target: str = "url",
    ) -> Any:
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
    ) -> Any:
        """Run executable shell/script command in the agent venv."""
        safe_working_dir = str(working_dir or "").strip() or "tulpa_stuff"
        safe_command = normalize_command_for_working_dir(
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
        require_customer_id(runtime)
        safe_thread = str(thread_id or "").strip()
        normalized_origin = normalize_execution_origin(
            thread_id=safe_thread,
            execution_origin=execution_origin,
        )

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
            payload = _decorate_python_dependency_failure(payload)
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
        "send_owner_update": send_owner_update,
        "memory_search": memory_search,
        "memory_add": memory_add,
        "uploaded_file_search": uploaded_file_search,
        "uploaded_file_get": uploaded_file_get,
        "uploaded_file_send": uploaded_file_send,
        "tulpa_file_send": tulpa_file_send,
        "web_image_send": web_image_send,
        "uploaded_file_analyze": uploaded_file_analyze,
        "uploaded_file_inspect_structure": uploaded_file_inspect_structure,
        "business_knowledge_index": business_knowledge_index,
        "business_knowledge_query": business_knowledge_query,
        "user_context_add_files": user_context_add_files,
        "user_context_query": user_context_query,
        "user_context_list_sources": user_context_list_sources,
        "user_context_find_sources": user_context_find_sources,
        "user_context_reindex": user_context_reindex,
        "user_context_archive_sources": user_context_archive_sources,
        "user_context_promote_to_intake": user_context_promote_to_intake,
        "directive_get": directive_get,
        "directive_set": directive_set,
        "directive_clear": directive_clear,
        "time_profile_get": time_profile_get,
        "time_profile_set": time_profile_set,
        "web_search": web_search,
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
        "server_time": server_time,
    }
