"""Directive tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.core_tools import _sync_proactive_heartbeat
from opentulpa.context.customer_profile_models import (
    CustomerScopedClearResponse,
    CustomerScopedOkResponse,
    CustomerScopedRequest,
    DirectiveGetResponse,
    DirectiveSetRequest,
)


def register_directive_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def directive_get() -> Any:
        """Get persistent user directive profile, preferences, and proactive-mode instruction."""
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
        """Set or overwrite persistent user directive profile and proactive behavior."""
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
        """Clear persistent user directive profile and disable proactive behavior."""
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

    return {
        "directive_get": directive_get,
        "directive_set": directive_set,
        "directive_clear": directive_clear,
    }
