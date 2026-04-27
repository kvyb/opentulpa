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
from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.guardrail_helpers import (
    approval_pending_payload,
    normalize_command_for_working_dir,
    normalize_execution_origin,
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
from opentulpa.policy.execution_boundary import ExecutionBoundaryContext, ExecutionBoundaryGuard


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
    boundary_guard = ExecutionBoundaryGuard(runtime=runtime)

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
        return r.json()

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
        return r.json()

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
        return r.json()

    @tool
    async def uploaded_file_analyze(
        file_id: str,
        question: str | None = None,
    ) -> Any:
        """Analyze a previously uploaded file again, optionally with a focused question."""
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
        workflow goal. Then pass chosen sheet/row selections to
        uploaded_file_prepare_intake_knowledge.
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
        return r.json()

    @tool
    async def uploaded_file_prepare_intake_knowledge(
        file_ids: list[str],
        include_hints: list[str] | str | None = None,
        selected_sections: list[dict[str, Any]] | list[str] | None = None,
        workflow_goal: str = "",
        output_name: str = "intake_workflow_knowledge.md",
    ) -> Any:
        """Compile uploaded source files into a small Markdown knowledge pack for an intake workflow.

        Use this during workflow setup when the user wants uploaded files, spreadsheets,
        price lists, FAQs, or policies bound to the workflow. For arbitrary XLSX files,
        call uploaded_file_inspect_structure first, choose exact sheets/row ranges, and
        pass them as selected_sections. include_hints may be workflow-derived terms from
        the user's stated goal, but do not assume fixed sheet names or source format.
        Bind the returned knowledge_file_id to the workflow's knowledge_file_ids.
        """
        customer_id = require_customer_id(runtime)
        safe_file_ids = [
            str(item or "").strip()
            for item in list(file_ids or [])
            if str(item or "").strip()
        ][:8]
        if not safe_file_ids:
            return {"error": "uploaded_file_prepare_intake_knowledge failed: file_ids is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/files/prepare_intake_knowledge",
            json_body={
                "customer_id": customer_id,
                "file_ids": safe_file_ids,
                "include_hints": include_hints,
                "selected_sections": selected_sections,
                "workflow_goal": workflow_goal,
                "output_name": output_name,
            },
            timeout=60.0,
            retries=1,
        )
        if r.status_code != 200:
            return _tool_error_payload("uploaded_file_prepare_intake_knowledge", r)
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
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """Run executable shell/script command through execution-boundary guard."""
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
        safe_customer = require_customer_id(runtime)
        safe_thread = str(thread_id or "").strip()
        normalized_origin = normalize_execution_origin(
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
            return approval_pending_payload(
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
    async def guardrail_execute_approved_action(approval_id: str) -> Any:
        """Execute a previously approved external-impact action exactly once."""
        customer_id = require_customer_id(runtime)
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
        "uploaded_file_inspect_structure": uploaded_file_inspect_structure,
        "uploaded_file_prepare_intake_knowledge": uploaded_file_prepare_intake_knowledge,
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
        "guardrail_execute_approved_action": guardrail_execute_approved_action,
        "server_time": server_time,
    }
