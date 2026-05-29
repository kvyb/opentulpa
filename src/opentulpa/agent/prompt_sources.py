"""Prompt context source loading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class PromptContextSources:
    available_skills: list[Any]
    skill_names: list[str]
    skill_query: str
    should_retrieve: bool
    active_directive: Any
    thread_rollup: str | None
    pending_context_summary: str
    link_alias_context: str
    memory_grounding: str
    connected_composio_toolkits_context: str


class ContextSourceProvider(Protocol):
    async def list_available_skills(self, customer_id: str) -> list[Any]: ...

    def load_thread_rollup_sections(self, thread_id: str) -> dict[str, str]: ...

    async def load_active_directive(self, customer_id: str) -> Any: ...

    async def load_memory_grounding_context(
        self,
        *,
        customer_id: str,
        user_text: str,
        turn_mode: str,
        token_budget: int,
    ) -> str: ...

    def build_link_alias_context(self, *, customer_id: str, user_text: str) -> str: ...

    async def load_connected_composio_toolkits_context(self, customer_id: str) -> str: ...


def should_include_optional_context(
    *,
    kind: str,
    prompt_mode: str,
    should_retrieve: bool,
) -> bool:
    mode = str(prompt_mode or "").strip().lower()
    normalized_kind = str(kind or "").strip().lower()
    if normalized_kind == "pending_context":
        return mode == "execution"
    if normalized_kind == "task_directive":
        return mode != "literal_chat"
    if normalized_kind == "skill_discovery":
        return mode != "literal_chat"
    if normalized_kind in {"thread_rollup", "invoked_skills", "link_aliases"}:
        return mode != "literal_chat" and bool(should_retrieve)
    if normalized_kind == "memory_grounding":
        return mode != "literal_chat"
    return False


def has_retrieval_evidence(
    *,
    user_text: str,
    prompt_mode: str,
    skill_candidates: list[Any] | None = None,
    thread_rollup_sections: dict[str, str] | None = None,
) -> bool:
    mode = str(prompt_mode or "").strip().lower()
    if mode == "literal_chat":
        return False
    text = str(user_text or "").strip().lower()
    if not text:
        return False
    if mode == "execution":
        return True
    tokens = set(re.findall(r"[a-z0-9][a-z0-9._-]{2,}", text))
    if not tokens:
        return False
    if isinstance(skill_candidates, list):
        for item in skill_candidates:
            if not isinstance(item, dict):
                continue
            hay = f"{item.get('name', '')} {item.get('description', '')}".lower()
            if any(tok in hay for tok in tokens):
                return True
    if isinstance(thread_rollup_sections, dict):
        hay = " ".join(
            str(thread_rollup_sections.get(key) or "").lower()
            for key in ("conversation_summary", "open_loops", "durable_facts")
        )
        if any(tok in hay for tok in tokens):
            return True
    return False


class PromptSourceLoader:
    async def load(
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
        safe_available_skills = available_skills
        safe_skill_names = skill_names
        safe_skill_query = skill_query
        if latest_user and latest_user != safe_skill_query:
            safe_available_skills = safe_available_skills or await provider.list_available_skills(
                customer_id
            )
            safe_skill_names = []
            safe_skill_query = latest_user

        rollup_sections = (
            provider.load_thread_rollup_sections(thread_id)
            if should_include_optional_context(
                kind="thread_rollup",
                prompt_mode=prompt_mode,
                should_retrieve=True,
            )
            else {}
        )
        should_retrieve = has_retrieval_evidence(
            user_text=latest_user,
            prompt_mode=prompt_mode,
            skill_candidates=safe_available_skills,
            thread_rollup_sections=rollup_sections,
        )
        active_directive = (
            await provider.load_active_directive(customer_id)
            if should_include_optional_context(
                kind="task_directive",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            )
            else None
        )
        memory_grounding = (
            await provider.load_memory_grounding_context(
                customer_id=customer_id,
                user_text=latest_user,
                turn_mode=turn_mode,
                token_budget=500,
            )
            if should_include_optional_context(
                kind="memory_grounding",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            )
            else ""
        )
        thread_rollup = (
            "\n\n".join(
                part
                for part in (
                    str(rollup_sections.get("open_loops") or "").strip(),
                    str(rollup_sections.get("durable_facts") or "").strip(),
                )
                if part
            ).strip()
            if should_include_optional_context(
                kind="thread_rollup",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            )
            else None
        )
        pending_context_summary = (
            str(state.get("pending_context_summary", "")).strip()
            if should_include_optional_context(
                kind="pending_context",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            )
            else ""
        )
        link_alias_context = (
            provider.build_link_alias_context(
                customer_id=customer_id,
                user_text=latest_user,
            )
            if should_include_optional_context(
                kind="link_aliases",
                prompt_mode=prompt_mode,
                should_retrieve=should_retrieve,
            )
            else ""
        )
        return PromptContextSources(
            available_skills=safe_available_skills,
            skill_names=safe_skill_names,
            skill_query=safe_skill_query,
            should_retrieve=should_retrieve,
            active_directive=active_directive,
            thread_rollup=thread_rollup,
            pending_context_summary=pending_context_summary,
            link_alias_context=link_alias_context,
            memory_grounding=memory_grounding,
            connected_composio_toolkits_context=(
                await provider.load_connected_composio_toolkits_context(customer_id)
            ),
        )
