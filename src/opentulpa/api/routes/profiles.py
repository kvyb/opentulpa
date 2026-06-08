"""Directive and time-profile route registration."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from fastapi import FastAPI

from opentulpa.context.customer_profile_models import (
    CustomerScopedClearResponse,
    CustomerScopedOkResponse,
    CustomerScopedRequest,
    DirectiveGetResponse,
    DirectiveSetRequest,
    ProfilesListResponse,
    TelegramBindingRequest,
    TimeProfileGetResponse,
    TimeProfileSetRequest,
    TimeProfileSetResponse,
)
from opentulpa.context.customer_profiles import CustomerProfileService


def _schedule_best_effort_memory_add(
    memory: Any,
    *,
    text: str,
    user_id: str,
    metadata: dict[str, Any],
) -> None:
    if memory is None:
        return

    async def _runner() -> None:
        with suppress(Exception):
            await asyncio.to_thread(
                memory.add_text,
                text,
                user_id=user_id,
                metadata=metadata,
                infer=False,
            )

    with suppress(RuntimeError):
        asyncio.create_task(_runner())


def register_profile_routes(
    app: FastAPI,
    *,
    get_profiles: Callable[[], CustomerProfileService],
    get_memory: Callable[[], Any],
) -> None:
    """Register directive + timezone profile endpoints."""

    @app.get("/profiles", response_model=ProfilesListResponse)
    async def profiles_list() -> ProfilesListResponse:
        profiles = get_profiles()
        return ProfilesListResponse(
            profiles=profiles.list_profiles(),
            bindings=profiles.list_identity_bindings(),
        )

    @app.post("/profiles/bind-telegram", response_model=CustomerScopedOkResponse)
    async def profiles_bind_telegram(body: TelegramBindingRequest) -> CustomerScopedOkResponse:
        profiles = get_profiles()
        binding = profiles.bind_telegram_user_id(
            user_id=body.user_id,
            telegram_user_id=body.telegram_user_id,
        )
        return CustomerScopedOkResponse(customer_id=binding.user_id)

    @app.post("/internal/directive/get", response_model=DirectiveGetResponse)
    async def internal_directive_get(body: CustomerScopedRequest) -> DirectiveGetResponse:
        profiles = get_profiles()
        customer_id = profiles.resolve_customer_id(body.customer_id)
        return DirectiveGetResponse(
            customer_id=customer_id,
            directive=profiles.get_directive(customer_id),
        )

    @app.post("/internal/directive/set", response_model=CustomerScopedOkResponse)
    async def internal_directive_set(body: DirectiveSetRequest) -> CustomerScopedOkResponse:
        profiles = get_profiles()
        customer_id = profiles.resolve_customer_id(body.customer_id)
        profiles.set_directive(customer_id, body.directive, source=body.source)

        # Best-effort memory signal for recall; directive DB remains source of truth.
        memory = get_memory()
        _schedule_best_effort_memory_add(
            memory,
            text=f"Directive updated for this user: {body.directive}",
            user_id=customer_id,
            metadata={"kind": "directive_fact", "source": body.source},
        )

        return CustomerScopedOkResponse(customer_id=customer_id)

    @app.post("/internal/directive/clear", response_model=CustomerScopedClearResponse)
    async def internal_directive_clear(body: CustomerScopedRequest) -> CustomerScopedClearResponse:
        profiles = get_profiles()
        customer_id = profiles.resolve_customer_id(body.customer_id)
        cleared = profiles.clear_directive(customer_id, source="agent")

        # Best-effort memory signal for recall; directive DB remains source of truth.
        memory = get_memory()
        _schedule_best_effort_memory_add(
            memory,
            text="Directive profile cleared for this user. Previous directive no longer applies.",
            user_id=customer_id,
            metadata={"kind": "directive_fact", "source": "agent"},
        )

        return CustomerScopedClearResponse(customer_id=customer_id, cleared=cleared)

    @app.post("/internal/time_profile/get", response_model=TimeProfileGetResponse)
    async def internal_time_profile_get(body: CustomerScopedRequest) -> TimeProfileGetResponse:
        profiles = get_profiles()
        customer_id = profiles.resolve_customer_id(body.customer_id)
        return TimeProfileGetResponse(
            customer_id=customer_id,
            utc_offset=profiles.get_utc_offset(customer_id),
        )

    @app.post("/internal/time_profile/set", response_model=TimeProfileSetResponse)
    async def internal_time_profile_set(body: TimeProfileSetRequest) -> TimeProfileSetResponse:
        profiles = get_profiles()
        customer_id = profiles.resolve_customer_id(body.customer_id)
        updated = profiles.set_utc_offset(customer_id, body.utc_offset, source=body.source)
        normalized = updated.utc_offset or body.utc_offset
        memory = get_memory()
        if normalized:
            _schedule_best_effort_memory_add(
                memory,
                text=f"User timezone is {normalized}.",
                user_id=customer_id,
                metadata={"kind": "life_fact", "source": body.source},
            )
        return TimeProfileSetResponse(customer_id=customer_id, utc_offset=normalized)
