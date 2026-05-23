"""Internal interactive user-context routes.

These endpoints expose the agent-facing user-context workflow for any uploaded
content type. Documents and spreadsheets can be indexed through local extraction;
PDFs, images, audio, and video are first prepared with the runtime multimodal
summarizer so visible text, speech, visual facts, layout details, and other
retrieval facts become text evidence. The routes then delegate to
``UserContextService`` to index, query, list, reindex, archive, or explicitly
promote selected sources into intake workflow knowledge.

No upload intent is inferred here. If a chat turn does not clearly say what to do
with uploaded files, the agent prompt policy is responsible for asking the user.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request

from opentulpa.api.routes.user_context_use_cases import UserContextRouteUseCases


def register_user_context_routes(
    app: FastAPI,
    *,
    get_user_context_service: Callable[[], Any],
    get_file_vault: Callable[[], Any],
    get_agent_runtime: Callable[[], Any],
    resolve_customer_id: Callable[[str], str] | None = None,
) -> None:
    """Register internal user-context endpoints."""
    use_cases = UserContextRouteUseCases(
        get_user_context_service=get_user_context_service,
        get_file_vault=get_file_vault,
        get_agent_runtime=get_agent_runtime,
        resolve_customer_id=resolve_customer_id,
    )

    @app.post("/internal/user_context/add_files")
    async def internal_user_context_add_files(request: Request) -> Any:
        return await use_cases.add_files(await request.json())

    @app.post("/internal/user_context/list_sources")
    async def internal_user_context_list_sources(request: Request) -> Any:
        return await use_cases.list_sources(await request.json())

    @app.post("/internal/user_context/find_sources")
    async def internal_user_context_find_sources(request: Request) -> Any:
        return await use_cases.find_sources(await request.json())

    @app.post("/internal/user_context/query")
    async def internal_user_context_query(request: Request) -> Any:
        return await use_cases.query(await request.json())

    @app.post("/internal/user_context/reindex")
    async def internal_user_context_reindex(request: Request) -> Any:
        return await use_cases.reindex(await request.json())

    @app.post("/internal/user_context/archive_sources")
    async def internal_user_context_archive_sources(request: Request) -> Any:
        return await use_cases.archive_sources(await request.json())

    @app.post("/internal/user_context/promote_to_intake")
    async def internal_user_context_promote_to_intake(request: Request) -> Any:
        return await use_cases.promote_to_intake(await request.json())
