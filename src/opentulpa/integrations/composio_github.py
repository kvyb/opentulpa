"""Tenant-owned GitHub REST requests through Composio OAuth."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, Protocol


class ComposioGitHubProvider(Protocol):
    def list_connected_accounts(
        self,
        *,
        customer_id: str,
        toolkits: list[str] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]: ...

    def proxy_tool(
        self,
        *,
        endpoint: str,
        method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
        connected_account_id: str,
        body: object | None = None,
    ) -> dict[str, Any]: ...


class ComposioGitHubProxyError(RuntimeError):
    """Sanitized Composio GitHub proxy failure."""


@dataclass(slots=True)
class ComposioGitHubAPIProxy:
    """Use one tenant-owned active GitHub connection without exposing OAuth."""

    provider: ComposioGitHubProvider
    _connection_cache: dict[str, str] = field(default_factory=dict, init=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    @staticmethod
    def _is_active(value: object) -> bool:
        return str(value or "").strip().casefold().rsplit(".", 1)[-1] == "active"

    def _connection_id(self, tenant_id: str) -> str:
        tenant = str(tenant_id or "").strip()
        if not tenant:
            raise ComposioGitHubProxyError("tenant identity is required")
        if not bool(getattr(self.provider, "enabled", True)):
            raise ComposioGitHubProxyError(
                "Composio is not configured; store COMPOSIO_API_KEY and retry"
            )
        with self._lock:
            cached = self._connection_cache.get(tenant)
        if cached:
            return cached
        try:
            response = self.provider.list_connected_accounts(
                customer_id=tenant,
                toolkits=["github"],
                statuses=["ACTIVE"],
                limit=10,
            )
        except Exception as exc:
            raise ComposioGitHubProxyError("Composio GitHub connection lookup failed") from exc
        items = response.get("items") if isinstance(response, dict) else None
        owned = [
            item
            for item in items or []
            if isinstance(item, dict)
            and str(item.get("user_id") or "") == tenant
            and str(item.get("toolkit_slug") or "").strip().casefold() == "github"
            and self._is_active(item.get("status"))
            and str(item.get("id") or "").strip()
        ]
        if not owned:
            raise ComposioGitHubProxyError(
                "No active tenant-owned Composio GitHub connection is available"
            )
        if len(owned) > 1:
            raise ComposioGitHubProxyError(
                "Multiple active GitHub connections are available; disconnect extras first"
            )
        connection_id = str(owned[0]["id"])
        with self._lock:
            self._connection_cache[tenant] = connection_id
        return connection_id

    def request(
        self,
        *,
        tenant_id: str,
        method: Literal["GET", "POST", "PATCH", "DELETE"],
        endpoint: str,
        body: object | None = None,
    ) -> tuple[int, Any]:
        safe_endpoint = str(endpoint or "").strip()
        if not safe_endpoint.startswith("/repos/") or "://" in safe_endpoint:
            raise ComposioGitHubProxyError("GitHub API endpoint is not allowed")
        connection_id = self._connection_id(tenant_id)
        try:
            response = self.provider.proxy_tool(
                endpoint=f"https://api.github.com{safe_endpoint}",
                method=method,
                connected_account_id=connection_id,
                body=body,
            )
        except Exception as exc:
            with self._lock:
                self._connection_cache.pop(str(tenant_id or "").strip(), None)
            raise ComposioGitHubProxyError("Composio GitHub request failed") from exc
        if not isinstance(response, dict):
            raise ComposioGitHubProxyError("Composio returned an invalid GitHub response")
        status = int(response.get("status") or 0)
        if status < 100 or status > 599:
            raise ComposioGitHubProxyError("Composio returned an invalid GitHub status")
        return status, response.get("data")


__all__ = [
    "ComposioGitHubAPIProxy",
    "ComposioGitHubProvider",
    "ComposioGitHubProxyError",
]
