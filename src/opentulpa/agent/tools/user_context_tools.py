"""User context tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.core_tools import _tool_error_payload


def register_user_context_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def user_context_add_files(file_ids: list[str]) -> Any:
        """Add uploaded files to durable user/chat context knowledge.

        Use this when the user's recent instructions or conversation make it
        clear that files should be remembered for future interactive chat. If
        intent is unclear, ask whether to remember them for chat, use them for a
        workflow/business bot, or answer about them once.
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
        """Query durable user/chat context knowledge for grounded evidence."""
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

    return {
        "user_context_add_files": user_context_add_files,
        "user_context_query": user_context_query,
        "user_context_list_sources": user_context_list_sources,
        "user_context_find_sources": user_context_find_sources,
        "user_context_reindex": user_context_reindex,
        "user_context_archive_sources": user_context_archive_sources,
        "user_context_promote_to_intake": user_context_promote_to_intake,
    }
