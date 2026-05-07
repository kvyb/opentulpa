"""Business knowledge tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.core_tools import (
    _compact_business_knowledge_index,
    _compact_business_knowledge_query,
    _resolve_business_knowledge_scope,
    _tool_error_payload,
)


def register_business_knowledge_tools(runtime: Any) -> dict[str, Any]:
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

    return {
        "business_knowledge_index": business_knowledge_index,
        "business_knowledge_query": business_knowledge_query,
    }
