from __future__ import annotations

from typing import Any

import pytest

from opentulpa.integrations.composio_github import (
    ComposioGitHubAPIProxy,
    ComposioGitHubProxyError,
)


class _Provider:
    def __init__(self, accounts: list[dict[str, Any]]) -> None:
        self.accounts = accounts
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_connected_accounts(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list", kwargs))
        return {"items": self.accounts}

    def proxy_tool(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("proxy", kwargs))
        return {"status": 200, "data": {"login": "owner"}}


def test_github_proxy_uses_one_active_tenant_owned_connection() -> None:
    provider = _Provider(
        [
            {
                "id": "connection-own",
                "user_id": "tenant-1",
                "toolkit_slug": "github",
                "status": "ACTIVE",
            },
            {
                "id": "connection-foreign",
                "user_id": "tenant-2",
                "toolkit_slug": "github",
                "status": "ACTIVE",
            },
        ]
    )
    proxy = ComposioGitHubAPIProxy(provider=provider)

    first = proxy.request(
        tenant_id="tenant-1",
        method="GET",
        endpoint="/repos/acme/project",
    )
    second = proxy.request(
        tenant_id="tenant-1",
        method="GET",
        endpoint="/repos/acme/project/git/ref/heads/main",
    )

    assert first == (200, {"login": "owner"})
    assert second == first
    assert [name for name, _ in provider.calls].count("list") == 1
    proxy_calls = [kwargs for name, kwargs in provider.calls if name == "proxy"]
    assert all(call["connected_account_id"] == "connection-own" for call in proxy_calls)


@pytest.mark.parametrize(
    "accounts,error",
    [
        (
            [
                {
                    "id": "connection-foreign",
                    "user_id": "tenant-2",
                    "toolkit_slug": "github",
                    "status": "ACTIVE",
                }
            ],
            "No active tenant-owned",
        ),
        (
            [
                {
                    "id": "connection-1",
                    "user_id": "tenant-1",
                    "toolkit_slug": "github",
                    "status": "ACTIVE",
                },
                {
                    "id": "connection-2",
                    "user_id": "tenant-1",
                    "toolkit_slug": "github",
                    "status": "ACTIVE",
                },
            ],
            "Multiple active",
        ),
    ],
)
def test_github_proxy_fails_closed_for_ambiguous_or_foreign_accounts(
    accounts: list[dict[str, Any]],
    error: str,
) -> None:
    proxy = ComposioGitHubAPIProxy(provider=_Provider(accounts))

    with pytest.raises(ComposioGitHubProxyError, match=error):
        proxy.request(
            tenant_id="tenant-1",
            method="GET",
            endpoint="/repos/acme/project",
        )


def test_github_proxy_rejects_non_github_endpoints() -> None:
    proxy = ComposioGitHubAPIProxy(provider=_Provider([]))

    with pytest.raises(ComposioGitHubProxyError, match="not allowed"):
        proxy.request(
            tenant_id="tenant-1",
            method="GET",
            endpoint="https://evil.example/repos/acme/project",
        )
