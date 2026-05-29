"""Context engine facade for prompt working sets and prompt sources."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.context_history import (
    ContextHistoryEngine,
    HistoryWorkingSet,
    trim_text_to_token_budget,
)
from opentulpa.agent.prompt_sources import (
    ContextSourceProvider,
    PromptContextSources,
    PromptSourceLoader,
    has_retrieval_evidence,
    should_include_optional_context,
)


class ContextEngine(ContextHistoryEngine):
    def __init__(
        self,
        *,
        raw_chat_limit: int = 20,
        raw_tool_limit: int = 5,
        stale_summary_token_budget: int = 900,
    ) -> None:
        super().__init__(
            raw_chat_limit=raw_chat_limit,
            raw_tool_limit=raw_tool_limit,
            stale_summary_token_budget=stale_summary_token_budget,
        )
        self._prompt_source_loader = PromptSourceLoader()

    @staticmethod
    def should_include_optional_context(
        *,
        kind: str,
        prompt_mode: str,
        should_retrieve: bool,
    ) -> bool:
        return should_include_optional_context(
            kind=kind,
            prompt_mode=prompt_mode,
            should_retrieve=should_retrieve,
        )

    @staticmethod
    def has_retrieval_evidence(
        *,
        user_text: str,
        prompt_mode: str,
        skill_candidates: list[Any] | None = None,
        thread_rollup_sections: dict[str, str] | None = None,
    ) -> bool:
        return has_retrieval_evidence(
            user_text=user_text,
            prompt_mode=prompt_mode,
            skill_candidates=skill_candidates,
            thread_rollup_sections=thread_rollup_sections,
        )

    async def load_prompt_context_sources(
        self,
        *,
        provider: ContextSourceProvider,
        state: dict[str, Any],
        customer_id: str,
        thread_id: str,
        prompt_mode: str,
        turn_mode: str,
        latest_user: str,
        available_skills: list[Any],
        skill_names: list[str],
        skill_query: str,
    ) -> PromptContextSources:
        return await self._prompt_source_loader.load(
            provider=provider,
            state=state,
            customer_id=customer_id,
            thread_id=thread_id,
            prompt_mode=prompt_mode,
            turn_mode=turn_mode,
            latest_user=latest_user,
            available_skills=available_skills,
            skill_names=skill_names,
            skill_query=skill_query,
        )

__all__ = [
    "ContextEngine",
    "ContextSourceProvider",
    "HistoryWorkingSet",
    "PromptContextSources",
    "trim_text_to_token_budget",
]
