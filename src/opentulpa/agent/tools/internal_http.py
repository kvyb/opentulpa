"""Shared internal HTTP helpers for agent tool handlers."""

from __future__ import annotations

from typing import Any

RETRYABLE_INTERNAL_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class InternalToolHTTPClient:
    """Normalizes internal HTTP response parsing for tool handlers."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime

    async def request(
        self,
        tool_name: str,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": timeout}
        if json_body is not None:
            kwargs["json_body"] = json_body
        if params is not None:
            kwargs["params"] = params
        if retries is not None:
            kwargs["retries"] = retries
        response = await self.runtime._request_with_backoff(method, path, **kwargs)
        if int(response.status_code) != 200:
            return failure_payload(tool_name=tool_name, response=response)
        try:
            payload = response.json()
        except Exception:
            return {"error": f"{tool_name} failed: invalid JSON response", "response": response.text}
        if not isinstance(payload, dict):
            return {"error": f"{tool_name} failed: JSON response was not an object", "response": payload}
        return payload

    async def request_item(
        self,
        tool_name: str,
        method: str,
        path: str,
        item_key: str,
        *,
        default: Any,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = 20.0,
        retries: int | None = None,
    ) -> Any:
        payload = await self.request(
            tool_name,
            method,
            path,
            json_body=json_body,
            params=params,
            timeout=timeout,
            retries=retries,
        )
        if isinstance(payload, dict) and payload.get("error"):
            return payload
        return payload.get(item_key, default) if isinstance(payload, dict) else default


def failure_payload(*, tool_name: str, response: Any) -> dict[str, Any]:
    status_code = int(getattr(response, "status_code", 0) or 0)
    return {
        "error": f"{tool_name} failed: {getattr(response, 'text', '')}",
        "status_code": status_code,
        "retryable": status_code in RETRYABLE_INTERNAL_STATUS_CODES,
    }
