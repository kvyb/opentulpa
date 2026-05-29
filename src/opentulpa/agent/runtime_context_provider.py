"""Runtime-backed provider for prompt context sources."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.composio_context import load_connected_composio_toolkits_context
from opentulpa.agent.turn_runtime_policy import recursion_limit_for_turn


class RuntimeContextSourceProvider:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def register_links_from_text(
        self,
        *,
        customer_id: str,
        text: str,
        source: str,
        limit: int,
    ) -> None:
        self._runtime.register_links_from_text(
            customer_id=customer_id,
            text=text,
            source=source,
            limit=limit,
        )

    def expand_link_aliases(self, *, customer_id: str, text: str) -> str:
        return str(
            self._runtime.expand_link_aliases(customer_id=customer_id, text=text) or ""
        )

    def list_pending_context_events(self, *, customer_id: str, limit: int) -> list[dict[str, Any]]:
        context_events = getattr(self._runtime, "_context_events", None)
        if context_events is None:
            return []
        events = context_events.list_events(customer_id, limit=limit)
        return events if isinstance(events, list) else []

    async def list_available_skills(self, customer_id: str) -> list[Any]:
        skills = await self._runtime._list_available_skills(customer_id)
        return skills if isinstance(skills, list) else []

    async def load_skill_context_by_names(
        self,
        *,
        customer_id: str,
        skill_names: list[str],
    ) -> dict[str, Any]:
        result = await self._runtime._load_skill_context_by_names(
            customer_id=customer_id,
            skill_names=skill_names,
        )
        return result if isinstance(result, dict) else {"skill_names": [], "context": ""}

    def effective_recursion_limit(self, override: int | None) -> int:
        return int(self._runtime._effective_recursion_limit(override))

    def recursion_limit_for_turn(
        self,
        *,
        customer_id: str,
        thread_id: str,
        requested_turn_mode: str,
        requested_limit: int,
        prompt_mode: str,
        user_text: str,
    ) -> int:
        return recursion_limit_for_turn(
            self._runtime,
            customer_id=customer_id,
            thread_id=thread_id,
            requested_turn_mode=requested_turn_mode,
            requested_limit=requested_limit,
            prompt_mode=prompt_mode,
            user_text=user_text,
        )

    def load_thread_rollup_sections(self, thread_id: str) -> dict[str, str]:
        sections = self._runtime._load_thread_rollup_sections(thread_id)
        return sections if isinstance(sections, dict) else {}

    async def load_active_directive(self, customer_id: str) -> Any:
        return await self._runtime._load_active_directive(customer_id)

    async def load_memory_grounding_context(
        self,
        *,
        customer_id: str,
        user_text: str,
        turn_mode: str,
        token_budget: int,
    ) -> str:
        return str(
            await self._runtime._load_memory_grounding_context(
                customer_id=customer_id,
                user_text=user_text,
                turn_mode=turn_mode,
                token_budget=token_budget,
            )
            or ""
        ).strip()

    def build_link_alias_context(self, *, customer_id: str, user_text: str) -> str:
        return str(
            self._runtime._build_link_alias_context(
                customer_id=customer_id,
                user_text=user_text,
            )
            or ""
        ).strip()

    async def load_connected_composio_toolkits_context(self, customer_id: str) -> str:
        cache = getattr(self._runtime, "_composio_prompt_toolkit_cache", None)
        if not isinstance(cache, dict):
            cache = {}
            self._runtime._composio_prompt_toolkit_cache = cache
        return await load_connected_composio_toolkits_context(
            composio=getattr(self._runtime, "composio_service", None),
            cache=cache,
            customer_id=customer_id,
        )
