"""Core/general-purpose tool registration."""

from __future__ import annotations

import json
import re
from contextlib import suppress
from typing import Any

from opentulpa.agent.tools.common import (
    require_customer_id,
    require_thread_id,
)
from opentulpa.agent.utils import html_to_text as _html_to_text


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


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _compact_file_record_for_tool(raw: Any) -> dict[str, Any]:
    record = _dict_or_empty(raw)
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
    row = _dict_or_empty(raw)
    values = _list_or_empty(row.get("values"))
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
    inspection = _dict_or_empty(payload.get("inspection"))
    structure = _dict_or_empty(inspection.get("structure"))
    raw_sheets = _list_or_empty(structure.get("sheets"))
    sheet_inventory: list[dict[str, Any]] = []
    relevant_sheets: list[dict[str, Any]] = []
    for raw_sheet in raw_sheets:
        if not isinstance(raw_sheet, dict):
            continue
        matches = _list_or_empty(raw_sheet.get("matches"))
        sample_rows = _list_or_empty(raw_sheet.get("sample_rows"))
        table_candidates = _list_or_empty(raw_sheet.get("table_candidates"))
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
                                _list_or_empty(item.get("sample_rows")) if isinstance(item, dict) else []
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
    from opentulpa.agent.graph_control_tools import register_graph_control_tools
    from opentulpa.agent.tools.business_knowledge_tools import register_business_knowledge_tools
    from opentulpa.agent.tools.directive_tools import register_directive_tools
    from opentulpa.agent.tools.file_tools import register_file_tools
    from opentulpa.agent.tools.memory_tools import register_memory_tools
    from opentulpa.agent.tools.owner_update_tools import register_owner_update_tools
    from opentulpa.agent.tools.server_time_tools import register_server_time_tools
    from opentulpa.agent.tools.task_tools import register_task_tools
    from opentulpa.agent.tools.time_profile_tools import register_time_profile_tools
    from opentulpa.agent.tools.tulpa_workspace_tools import register_tulpa_workspace_tools
    from opentulpa.agent.tools.user_context_tools import register_user_context_tools
    from opentulpa.agent.tools.web_tools import register_web_tools

    tools: dict[str, Any] = {}
    tools.update(register_owner_update_tools(runtime))
    tools.update(register_graph_control_tools(runtime))
    tools.update(register_memory_tools(runtime))
    tools.update(register_file_tools(runtime))
    tools.update(register_business_knowledge_tools(runtime))
    tools.update(register_user_context_tools(runtime))
    tools.update(register_directive_tools(runtime))
    tools.update(register_time_profile_tools(runtime))
    tools.update(register_web_tools(runtime))
    tools.update(register_tulpa_workspace_tools(runtime))
    tools.update(register_task_tools(runtime))
    tools.update(register_server_time_tools(runtime))
    return tools
