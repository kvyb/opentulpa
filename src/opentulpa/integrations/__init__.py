"""Integrations: Composio, web-search, and external service connectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["ComposioService"]

if TYPE_CHECKING:
    from opentulpa.integrations.composio import ComposioService


def __getattr__(name: str) -> Any:
    if name != "ComposioService":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from opentulpa.integrations.composio import ComposioService

    return ComposioService
