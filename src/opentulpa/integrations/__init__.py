"""Injected external service adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "ComposioService",
    "TenantComposioService",
]

if TYPE_CHECKING:
    from opentulpa.integrations.composio import ComposioService
    from opentulpa.integrations.tenant_composio import TenantComposioService


def __getattr__(name: str) -> Any:
    if name == "ComposioService":
        from opentulpa.integrations.composio import ComposioService

        return ComposioService
    if name == "TenantComposioService":
        from opentulpa.integrations.tenant_composio import TenantComposioService

        return TenantComposioService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
