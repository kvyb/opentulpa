"""FastAPI app package."""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name != "create_app":
        raise AttributeError(name)
    from opentulpa.api.app import create_app

    return create_app
