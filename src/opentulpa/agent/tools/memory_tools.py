"""Memory tools."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id


def register_memory_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def memory_search(query: str) -> Any:
        """Search durable user memory for preferences, facts, projects, and prior context."""
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
        """Store a concise durable user memory summary for future conversations.

        Use this for stable preferences, style instructions, important facts, and reusable context
        the user expects OpenTulpa to remember in normal interactive chat.
        """
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
                        "infer": False,
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

    return {
        "memory_search": memory_search,
        "memory_add": memory_add,
    }
