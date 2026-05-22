"""Internal file-vault route registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request

from opentulpa.api.routes.file_use_cases import TULPA_STUFF_DIR as TULPA_STUFF_DIR
from opentulpa.api.routes.file_use_cases import FileRouteUseCases
from opentulpa.api.routes.file_use_cases import (
    download_image_from_web_url as download_image_from_web_url,
)


def register_file_routes(
    app: FastAPI,
    *,
    get_file_vault: Callable[[], Any],
    get_telegram_chat: Callable[[], Any],
    get_telegram_client: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    telegram_enabled: bool,
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register uploaded-file search/get/send/analyze endpoints."""
    use_cases = FileRouteUseCases(
        get_file_vault=get_file_vault,
        get_telegram_chat=get_telegram_chat,
        get_telegram_client=get_telegram_client,
        get_agent_runtime=get_agent_runtime,
        telegram_enabled=telegram_enabled,
        resolve_customer_id=resolve_customer_id,
        tulpa_stuff_dir=TULPA_STUFF_DIR,
        download_image=download_image_from_web_url,
    )

    @app.post("/internal/files/search")
    async def internal_files_search(request: Request) -> Any:
        return await use_cases.search(await request.json())

    @app.post("/internal/files/get")
    async def internal_files_get(request: Request) -> Any:
        return await use_cases.get(await request.json())

    @app.post("/internal/files/send")
    async def internal_files_send(request: Request) -> Any:
        return await use_cases.send(await request.json())

    @app.post("/internal/files/send_local")
    async def internal_files_send_local(request: Request) -> Any:
        return await use_cases.send_local(await request.json())

    @app.post("/internal/files/send_web_image")
    async def internal_files_send_web_image(request: Request) -> Any:
        return await use_cases.send_web_image(await request.json())

    @app.post("/internal/files/analyze")
    async def internal_files_analyze(request: Request) -> Any:
        return await use_cases.analyze(await request.json())

    @app.post("/internal/files/inspect_structure")
    async def internal_files_inspect_structure(request: Request) -> Any:
        return await use_cases.inspect_structure(await request.json())
