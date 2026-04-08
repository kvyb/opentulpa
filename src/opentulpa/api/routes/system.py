"""System/config helper route registration."""

from __future__ import annotations

from fastapi import FastAPI

from opentulpa.core.public_urls import resolve_public_base_url


def register_system_routes(app: FastAPI) -> None:
    """Register small internal helper endpoints for environment-derived system values."""

    @app.get("/internal/system/public_base_url")
    async def internal_public_base_url() -> dict[str, object]:
        public_base_url = resolve_public_base_url()
        return {
            "ok": True,
            "public_base_url": public_base_url,
            "available": bool(public_base_url),
        }
