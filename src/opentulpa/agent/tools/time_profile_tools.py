"""Time profile tools."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.context.customer_profile_models import (
    CustomerScopedRequest,
    TimeProfileGetResponse,
    TimeProfileSetRequest,
    TimeProfileSetResponse,
)


def register_time_profile_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def time_profile_get() -> Any:
        """Get stored user timezone/UTC offset for scheduling and reminders."""
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
        """Set user timezone/UTC offset in +HH:MM or -HH:MM format."""
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

    return {
        "time_profile_get": time_profile_get,
        "time_profile_set": time_profile_set,
    }
