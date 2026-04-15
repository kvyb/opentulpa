"""Integrations: Browser Use, Composio, web-search, and external service connectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["BrowserUseLocalManager", "ComposioService", "HeadroomService"]

if TYPE_CHECKING:
    from opentulpa.integrations.browser_use_local import BrowserUseLocalManager
    from opentulpa.integrations.composio import ComposioService
    from opentulpa.integrations.headroom import HeadroomService


def __getattr__(name: str) -> Any:
    if name == "BrowserUseLocalManager":
        from opentulpa.integrations.browser_use_local import BrowserUseLocalManager

        return BrowserUseLocalManager
    if name == "ComposioService":
        from opentulpa.integrations.composio import ComposioService

        return ComposioService
    if name == "HeadroomService":
        from opentulpa.integrations.headroom import HeadroomService

        return HeadroomService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
