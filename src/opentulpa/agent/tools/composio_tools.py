"""Composio tool registration."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.internal_http import (
    RETRYABLE_INTERNAL_STATUS_CODES,
    InternalToolHTTPClient,
)


def _toolkit_from_slug(tool_slug: str) -> str:
    prefix = str(tool_slug or "").strip().split("_", 1)[0].lower()
    if prefix == "googlesheets":
        return "googlesheets"
    if prefix == "googledrive":
        return "googledrive"
    if prefix == "instagram":
        return "instagram"
    return prefix


def _composio_failure_payload(
    *,
    operation: str,
    status_code: int,
    response_text: str,
    retryable: bool,
    suggested_next_action: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": f"{operation} failed",
        "status_code": status_code,
        "retryable": retryable,
        "response": response_text,
        "suggested_next_action": suggested_next_action,
    }
    if extra:
        payload.update(extra)
    return payload


def _is_retryable_internal_status(status_code: int) -> bool:
    return int(status_code or 0) in RETRYABLE_INTERNAL_STATUS_CODES


def register_composio_tools(runtime: Any) -> dict[str, Any]:
    http = InternalToolHTTPClient(runtime)

    @tool
    async def composio_status() -> Any:
        """Check whether Composio is configured before trying auth or external tool execution."""
        return await http.request(
            "composio_status",
            "GET",
            "/internal/composio/status",
            timeout=8.0,
        )

    @tool
    async def composio_authorize_toolkit(toolkit: str, callback_url: str = "") -> Any:
        """Create a Composio auth link for the active user. Share redirect_url with the user so they can finish OAuth."""
        customer_id = require_customer_id(runtime)
        payload = await http.request(
            "composio_authorize_toolkit",
            "POST",
            "/internal/composio/authorize",
            json_body={
                "customer_id": customer_id,
                "toolkit": toolkit,
                "callback_url": callback_url,
            },
            timeout=20.0,
        )
        if payload.get("error"):
            return payload
        redirect_url = str(payload.get("redirect_url", "") or "").strip()
        if redirect_url:
            payload["message"] = (
                f"Open this authorization link to connect {toolkit}: {redirect_url}"
            )
        return payload

    @tool
    async def composio_wait_for_connection(
        connection_id: str,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """Wait for a Composio connection to become active after the user finishes OAuth."""
        return await http.request_item(
            "composio_wait_for_connection",
            "POST",
            "/internal/composio/wait_for_connection",
            "connection",
            default={},
            json_body={
                "connection_id": connection_id,
                "timeout_seconds": max(1.0, min(float(timeout_seconds), 600.0)),
            },
            timeout=max(10.0, min(float(timeout_seconds) + 5.0, 605.0)),
            retries=0,
        )

    @tool
    async def composio_toolkits(
        toolkits: list[str] | None = None,
        is_connected: str = "",
        limit: int = 50,
        search: str = "",
    ) -> Any:
        """List Composio toolkit connection state for the active user."""
        customer_id = require_customer_id(runtime)
        params: dict[str, Any] = {
            "customer_id": customer_id,
            "toolkits": ",".join(toolkits or []),
            "limit": max(1, min(int(limit), 100)),
            "search": str(search or "").strip(),
        }
        if str(is_connected or "").strip():
            params["is_connected"] = str(is_connected).strip()
        return await http.request_item(
            "composio_toolkits",
            "GET",
            "/internal/composio/toolkits",
            "items",
            default=[],
            params=params,
            timeout=15.0,
        )

    @tool
    async def composio_connected_accounts(
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> Any:
        """List Composio connected accounts for the active user."""
        customer_id = require_customer_id(runtime)
        return await http.request_item(
            "composio_connected_accounts",
            "GET",
            "/internal/composio/connected_accounts",
            "items",
            default=[],
            params={
                "customer_id": customer_id,
                "toolkits": ",".join(toolkits or []),
                "statuses": ",".join(statuses or []),
                "limit": max(1, min(int(limit), 100)),
            },
            timeout=15.0,
        )

    @tool
    async def composio_disable_connected_account(connected_account_id: str) -> Any:
        """Disable a Composio connected account so OpenTulpa stops using it."""
        return await http.request_item(
            "composio_disable_connected_account",
            "POST",
            "/internal/composio/connected_accounts/disable",
            "connected_account",
            default={},
            json_body={"connected_account_id": str(connected_account_id or "").strip()},
            timeout=20.0,
        )

    @tool
    async def composio_delete_connected_account(connected_account_id: str) -> Any:
        """Delete a Composio connected account permanently."""
        return await http.request_item(
            "composio_delete_connected_account",
            "POST",
            "/internal/composio/connected_accounts/delete",
            "connected_account",
            default={},
            json_body={"connected_account_id": str(connected_account_id or "").strip()},
            timeout=20.0,
        )

    @tool
    async def composio_tool_search(
        query: str = "",
        toolkits: list[str] | None = None,
        limit: int = 20,
    ) -> Any:
        """Search Composio tools and return candidate tool slugs, descriptions, and input schemas."""
        return await http.request_item(
            "composio_tool_search",
            "GET",
            "/internal/composio/tools/search",
            "items",
            default=[],
            params={
                "query": str(query or "").strip(),
                "toolkits": ",".join(toolkits or []),
                "limit": max(1, min(int(limit), 50)),
            },
            timeout=20.0,
        )

    @tool
    async def composio_tool_schema(tool_slug: str) -> Any:
        """Get the input schema for a single Composio tool slug."""
        return await http.request_item(
            "composio_tool_schema",
            "GET",
            f"/internal/composio/tools/{tool_slug}/schema",
            "tool",
            default={},
            timeout=20.0,
        )

    @tool
    async def composio_instagram_reply_precheck(
        recipient_id: str = "",
        conversation_id: str = "",
        connected_account_id: str = "",
        scan_limit: int = 10,
    ) -> Any:
        """Verify the exact Instagram thread for a recipient and capture the latest inbound timestamp before attempting a DM send."""
        customer_id = require_customer_id(runtime)
        return await http.request(
            "composio_instagram_reply_precheck",
            "POST",
            "/internal/composio/instagram/reply_precheck",
            json_body={
                "customer_id": customer_id,
                "recipient_id": str(recipient_id or "").strip(),
                "conversation_id": str(conversation_id or "").strip(),
                "connected_account_id": str(connected_account_id or "").strip(),
                "scan_limit": max(1, min(int(scan_limit), 25)),
            },
            timeout=60.0,
        )

    @tool
    async def composio_tool_execute(
        tool_slug: str,
        arguments: dict[str, Any] | None = None,
        connected_account_id: str = "",
        text: str = "",
    ) -> Any:
        """Execute a Composio tool for the active user using explicit JSON arguments from the tool schema. For Instagram sends, verify the exact thread first with composio_instagram_reply_precheck."""
        customer_id = require_customer_id(runtime)
        safe_tool_slug = str(tool_slug or "").strip()
        schema_response = await runtime._request_with_backoff(
            "GET",
            f"/internal/composio/tools/{safe_tool_slug}/schema",
            timeout=20.0,
        )
        if schema_response.status_code != 200:
            retryable = _is_retryable_internal_status(schema_response.status_code)
            toolkit = _toolkit_from_slug(safe_tool_slug)
            candidates: Any = []
            if toolkit and not retryable:
                search_response = await runtime._request_with_backoff(
                    "GET",
                    "/internal/composio/tools/search",
                    params={"query": safe_tool_slug, "toolkits": toolkit, "limit": 8},
                    timeout=20.0,
                )
                if search_response.status_code == 200:
                    candidates = search_response.json().get("items", [])
            return _composio_failure_payload(
                operation="composio_tool_execute preflight",
                status_code=schema_response.status_code,
                response_text=schema_response.text,
                retryable=retryable,
                suggested_next_action=(
                    "Retry the same call later."
                    if retryable
                    else "The requested Composio tool slug is not available. Choose one of the "
                    "candidate tools from composio_tool_search, then retry with that exact slug."
                ),
                extra={
                    "tool_slug": safe_tool_slug,
                    "candidate_tools": candidates,
                },
            )
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/composio/tools/execute",
            json_body={
                "customer_id": customer_id,
                "tool_slug": safe_tool_slug,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "connected_account_id": connected_account_id,
                "text": text,
            },
            timeout=120.0,
            retries=0,
        )
        if r.status_code != 200:
            retryable = r.status_code in RETRYABLE_INTERNAL_STATUS_CODES
            return _composio_failure_payload(
                operation="composio_tool_execute",
                status_code=r.status_code,
                response_text=r.text,
                retryable=retryable,
                suggested_next_action=(
                    "Retry the same call later."
                    if retryable
                    else "Inspect the error. If it says the account is not connected, ask the user to authorize the toolkit. If arguments are invalid, repair them from composio_tool_schema."
                ),
                extra={"tool_slug": safe_tool_slug},
            )
        return r.json()

    return {
        "composio_status": composio_status,
        "composio_authorize_toolkit": composio_authorize_toolkit,
        "composio_wait_for_connection": composio_wait_for_connection,
        "composio_toolkits": composio_toolkits,
        "composio_connected_accounts": composio_connected_accounts,
        "composio_disable_connected_account": composio_disable_connected_account,
        "composio_delete_connected_account": composio_delete_connected_account,
        "composio_tool_search": composio_tool_search,
        "composio_tool_schema": composio_tool_schema,
        "composio_instagram_reply_precheck": composio_instagram_reply_precheck,
        "composio_tool_execute": composio_tool_execute,
    }
