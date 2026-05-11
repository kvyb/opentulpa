"""Composio route registration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from opentulpa.core.public_urls import build_public_composio_callback_path

logger = logging.getLogger(__name__)
RETRYABLE_COMPOSIO_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _error_response(exc: Exception, *, operation: str) -> JSONResponse:
    status_code = int(getattr(exc, "status_code", 400) or 400)
    body = getattr(exc, "body", None)
    detail = body if isinstance(body, dict) else {"detail": str(exc)}
    logger.warning("Composio route %s failed: %s", operation, detail)
    return JSONResponse(status_code=status_code, content=detail)


def _is_retryable_composio_error(exc: Exception) -> bool:
    status_code = int(getattr(exc, "status_code", 0) or 0)
    if status_code in RETRYABLE_COMPOSIO_STATUS_CODES:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        try:
            nested_status = int(error.get("status", 0) or 0)
        except Exception:
            nested_status = 0
        if nested_status in RETRYABLE_COMPOSIO_STATUS_CODES:
            return True
        code = str(error.get("code", "") or "").lower()
        slug = str(error.get("slug", "") or "").lower()
        message = str(error.get("message", "") or "").lower()
        if any(marker in f"{code} {slug} {message}" for marker in ("timeout", "bad gateway")):
            return True
    return False


async def _execute_composio_tool_with_retry(
    service: Any,
    *,
    customer_id: str,
    tool_slug: str,
    arguments: dict[str, Any],
    connected_account_id: str | None,
    text: str | None,
) -> Any:
    attempts = 3
    for attempt in range(attempts):
        try:
            return service.execute_tool(
                customer_id=customer_id,
                tool_slug=tool_slug,
                arguments=arguments,
                connected_account_id=connected_account_id,
                text=text,
            )
        except Exception as exc:
            if attempt + 1 >= attempts or not _is_retryable_composio_error(exc):
                raise
            delay = 0.4 * (2**attempt)
            logger.warning(
                "Composio tool execute transient failure; retrying tool_slug=%s attempt=%s/%s delay=%.2fs error=%s",
                tool_slug,
                attempt + 1,
                attempts,
                delay,
                f"{type(exc).__name__}: {exc}",
            )
            await asyncio.sleep(delay)
    raise RuntimeError("Composio tool execute retry loop exhausted")


def register_composio_routes(
    app: FastAPI,
    *,
    get_composio: Callable[[], Any],
) -> None:
    """Register internal Composio helper endpoints."""

    @app.get(build_public_composio_callback_path())
    async def composio_callback_landing(request: Request) -> HTMLResponse:
        connection_id = str(
            request.query_params.get("connectedAccountId")
            or request.query_params.get("connection_id")
            or ""
        ).strip()
        toolkit = str(
            request.query_params.get("toolkit")
            or request.query_params.get("toolkit_slug")
            or request.query_params.get("integration")
            or ""
        ).strip()
        title = "Connection complete"
        details: list[str] = ["You can close this tab and return to OpenTulpa."]
        if toolkit:
            details.insert(0, f"Composio finished connecting {toolkit}.")
        if connection_id:
            details.append(f"Connection ID: {connection_id}")
        body = "".join(f"<p>{item}</p>" for item in details)
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<style>body{font-family:system-ui,-apple-system,sans-serif;max-width:640px;"
            "margin:48px auto;padding:0 20px;line-height:1.5;color:#111}"
            "h1{font-size:28px;margin-bottom:16px}p{margin:12px 0}code{background:#f4f4f5;"
            "padding:2px 6px;border-radius:6px}</style></head><body>"
            f"<h1>{title}</h1>{body}</body></html>"
        )
        return HTMLResponse(content=html)

    @app.get("/internal/composio/status")
    async def internal_composio_status() -> dict[str, Any]:
        service = get_composio()
        if hasattr(service, "status"):
            return service.status()
        return {"ok": True, "enabled": bool(getattr(service, "enabled", False))}

    @app.post("/internal/composio/authorize")
    async def internal_composio_authorize(request: Request) -> Any:
        body = await request.json()
        try:
            return get_composio().authorize_toolkit(
                customer_id=str(body.get("customer_id", "")).strip(),
                toolkit=str(body.get("toolkit", "")).strip(),
                callback_url=str(body.get("callback_url", "")).strip() or None,
            )
        except Exception as exc:
            return _error_response(exc, operation="authorize")

    @app.post("/internal/composio/wait_for_connection")
    async def internal_composio_wait_for_connection(request: Request) -> Any:
        body = await request.json()
        try:
            return {
                "ok": True,
                "connection": get_composio().wait_for_connection(
                    connection_id=str(body.get("connection_id", "")).strip(),
                    timeout_seconds=float(body.get("timeout_seconds", 60.0) or 60.0),
                ),
            }
        except Exception as exc:
            return _error_response(exc, operation="wait_for_connection")

    @app.get("/internal/composio/toolkits")
    async def internal_composio_toolkits(
        customer_id: str = "",
        toolkits: str = "",
        is_connected: str = "",
        limit: int = 50,
        search: str = "",
    ) -> Any:
        try:
            connected_flag: bool | None = None
            if str(is_connected or "").strip():
                connected_flag = str(is_connected).strip().lower() in {"1", "true", "yes", "on"}
            return get_composio().list_toolkits(
                customer_id=customer_id,
                toolkits=_parse_csv(toolkits),
                is_connected=connected_flag,
                limit=limit,
                search=search,
            )
        except Exception as exc:
            return _error_response(exc, operation="toolkits")

    @app.get("/internal/composio/connected_accounts")
    async def internal_composio_connected_accounts(
        customer_id: str = "",
        toolkits: str = "",
        statuses: str = "",
        limit: int = 50,
    ) -> Any:
        try:
            return get_composio().list_connected_accounts(
                customer_id=customer_id,
                toolkits=_parse_csv(toolkits),
                statuses=_parse_csv(statuses),
                limit=limit,
            )
        except Exception as exc:
            return _error_response(exc, operation="connected_accounts")

    @app.post("/internal/composio/connected_accounts/disable")
    async def internal_composio_disable_connected_account(request: Request) -> Any:
        body = await request.json()
        try:
            return get_composio().disable_connected_account(
                connected_account_id=str(body.get("connected_account_id", "")).strip(),
            )
        except Exception as exc:
            return _error_response(exc, operation="disable_connected_account")

    @app.post("/internal/composio/connected_accounts/delete")
    async def internal_composio_delete_connected_account(request: Request) -> Any:
        body = await request.json()
        try:
            return get_composio().delete_connected_account(
                connected_account_id=str(body.get("connected_account_id", "")).strip(),
            )
        except Exception as exc:
            return _error_response(exc, operation="delete_connected_account")

    @app.get("/internal/composio/tools/search")
    async def internal_composio_tools_search(
        query: str = "",
        toolkits: str = "",
        limit: int = 20,
    ) -> Any:
        try:
            return get_composio().search_tools(
                query=query,
                toolkits=_parse_csv(toolkits),
                limit=limit,
            )
        except Exception as exc:
            return _error_response(exc, operation="tools_search")

    @app.get("/internal/composio/tools/{tool_slug}/schema")
    async def internal_composio_tool_schema(tool_slug: str) -> Any:
        try:
            return get_composio().get_tool_schema(tool_slug=tool_slug)
        except Exception as exc:
            return _error_response(exc, operation="tool_schema")

    @app.post("/internal/composio/instagram/reply_precheck")
    async def internal_composio_instagram_reply_precheck(request: Request) -> Any:
        body = await request.json()
        try:
            return get_composio().inspect_instagram_reply_target(
                customer_id=str(body.get("customer_id", "")).strip(),
                recipient_id=str(body.get("recipient_id", "")).strip() or None,
                conversation_id=str(body.get("conversation_id", "")).strip() or None,
                connected_account_id=str(body.get("connected_account_id", "")).strip() or None,
                scan_limit=int(body.get("scan_limit", 10) or 10),
            )
        except Exception as exc:
            return _error_response(exc, operation="instagram_reply_precheck")

    @app.post("/internal/composio/tools/execute")
    async def internal_composio_tool_execute(request: Request) -> Any:
        body = await request.json()
        try:
            return await _execute_composio_tool_with_retry(
                get_composio(),
                customer_id=str(body.get("customer_id", "")).strip(),
                tool_slug=str(body.get("tool_slug", "")).strip(),
                arguments=body.get("arguments") if isinstance(body.get("arguments"), dict) else {},
                connected_account_id=str(body.get("connected_account_id", "")).strip() or None,
                text=str(body.get("text", "")).strip() or None,
            )
        except Exception as exc:
            return _error_response(exc, operation="tool_execute")
