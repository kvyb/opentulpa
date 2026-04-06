"""Agent runtime package."""

from __future__ import annotations

__all__ = ["OpenTulpaLangGraphRuntime"]


def __getattr__(name: str):
    if name == "OpenTulpaLangGraphRuntime":
        from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime

        return OpenTulpaLangGraphRuntime
    raise AttributeError(f"module 'opentulpa.agent' has no attribute {name!r}")
