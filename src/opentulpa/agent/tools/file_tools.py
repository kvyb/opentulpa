"""File tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.core_tools import (
    _compact_uploaded_file_inspection,
    _tool_error_payload,
    _with_delivery_instruction,
)


def register_file_tools(runtime: Any) -> dict[str, Any]:
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
        """Fetch uploaded file metadata and a bounded text excerpt by file_id."""
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

    return {
        "uploaded_file_search": uploaded_file_search,
        "uploaded_file_get": uploaded_file_get,
        "uploaded_file_send": uploaded_file_send,
        "tulpa_file_send": tulpa_file_send,
        "web_image_send": web_image_send,
        "uploaded_file_analyze": uploaded_file_analyze,
        "uploaded_file_inspect_structure": uploaded_file_inspect_structure,
    }
