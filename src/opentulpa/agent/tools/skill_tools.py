"""Reusable skill tool registration."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id


def register_skill_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def skill_list(include_global: bool = True, limit: int = 50) -> Any:
        """List reusable user/global skills available for task-specific instructions."""
        customer_id = require_customer_id(runtime)
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
        """Fetch one reusable skill by name, checking user scope before global fallback."""
        customer_id = require_customer_id(runtime)
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
        """Create or update reusable task instructions as a user or global skill."""
        customer_id = require_customer_id(runtime)
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
        """Delete a reusable user or global skill by name."""
        customer_id = require_customer_id(runtime)
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

    return {
        "skill_list": skill_list,
        "skill_get": skill_get,
        "skill_upsert": skill_upsert,
        "skill_delete": skill_delete,
    }
