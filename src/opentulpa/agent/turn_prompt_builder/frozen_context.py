"""Frozen and retrieved prompt-context assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentulpa.agent.context_engine import (
    ContextEngine,
    ContextSourceProvider,
    PromptContextSources,
)
from opentulpa.agent.context_engine import trim_text_to_token_budget as _trim_text_to_token_budget
from opentulpa.agent.models import AgentState
from opentulpa.agent.prompt_sections import PROMPT_DYNAMIC_BOUNDARY
from opentulpa.agent.prompt_sections import build_prompt_mode_message as _build_prompt_mode_message
from opentulpa.agent.turn_policy import (
    build_turn_mode_system_message as _build_turn_mode_system_message,
)
from opentulpa.agent.turn_prompt_builder.entries import (
    append_retrieved_entry,
    make_prompt_context_entry,
)
from opentulpa.agent.utils import content_to_text as _content_to_text


@dataclass(frozen=True, slots=True)
class SkillPromptState:
    skill_query: str
    skill_names: list[str]
    available_skills: list[Any]
    skill_discovery_context: str


@dataclass(frozen=True, slots=True)
class FrozenPromptContextResult:
    context: dict[str, Any]
    skill_state: SkillPromptState


def frozen_prompt_context_matches(
    raw: Any,
    *,
    latest_user: str,
    customer_id: str,
    prompt_mode: str,
    turn_mode: str,
) -> bool:
    if not isinstance(raw, dict):
        return False
    signature = raw.get("signature")
    if not isinstance(signature, dict):
        return False
    return (
        str(signature.get("latest_user", "")).strip() == str(latest_user or "").strip()
        and str(signature.get("customer_id", "")).strip() == str(customer_id or "").strip()
        and str(signature.get("prompt_mode", "")).strip() == str(prompt_mode or "").strip()
        and str(signature.get("turn_mode", "")).strip() == str(turn_mode or "").strip()
    )


async def build_frozen_prompt_context(
    *,
    context_provider: ContextSourceProvider,
    state: AgentState,
    customer_id: str,
    thread_id: str,
    prompt_mode: str,
    turn_mode: str,
    latest_user: str,
    low_budget: int,
    context_engine: ContextEngine,
    available_skills: list[Any],
    skill_names: list[str],
    skill_query: str,
    skill_discovery_context: str,
    invoked_skill_names: list[str],
    invoked_skill_context: str,
) -> FrozenPromptContextResult:
    sources = await context_engine.load_prompt_context_sources(
        provider=context_provider,
        state=dict(state),
        customer_id=customer_id,
        thread_id=thread_id,
        prompt_mode=prompt_mode,
        turn_mode=turn_mode,
        latest_user=latest_user,
        available_skills=available_skills,
        skill_names=skill_names,
        skill_query=skill_query,
    )
    skill_discovery_context = build_relevant_skill_discovery_context(
        available_skills=sources.available_skills,
        selected_names=sources.skill_names,
    )
    late_entries = build_late_prompt_entries(
        sources=sources,
        skill_discovery_context=skill_discovery_context,
        invoked_skill_context=invoked_skill_context,
        invoked_skill_names=invoked_skill_names,
        low_budget=low_budget,
        context_engine=context_engine,
        prompt_mode=prompt_mode,
    )
    context = {
        "signature": {
            "latest_user": latest_user,
            "customer_id": customer_id,
            "prompt_mode": prompt_mode,
            "turn_mode": turn_mode,
        },
        "late_control_content": _build_late_turn_control_text(customer_id=customer_id),
        "current_turn_context_content": _build_current_turn_context_text(
            prompt_mode=prompt_mode,
            turn_mode=turn_mode,
            connected_composio_toolkits_context=sources.connected_composio_toolkits_context,
        ),
        "late_control_sections": ["volatile_injected", "customer_scope", "time_tool_guidance"],
        "current_turn_context_sections": [
            f"prompt_mode:{prompt_mode}",
            f"turn_mode:{turn_mode}",
            *(
                ["connected_composio_toolkits"]
                if sources.connected_composio_toolkits_context
                else []
            ),
        ],
        "late_entries": late_entries,
    }
    return FrozenPromptContextResult(
        context=context,
        skill_state=SkillPromptState(
            skill_query=sources.skill_query,
            skill_names=sources.skill_names,
            available_skills=sources.available_skills,
            skill_discovery_context=skill_discovery_context,
        ),
    )

def build_late_prompt_entries(
    *,
    sources: PromptContextSources,
    skill_discovery_context: str,
    invoked_skill_context: str,
    invoked_skill_names: list[str],
    low_budget: int,
    context_engine: ContextEngine,
    prompt_mode: str,
) -> list[dict[str, str]]:
    late_entries: list[dict[str, str]] = []
    if sources.active_directive:
        append_retrieved_entry(
            late_entries,
            section="task_directive",
            title="Active persistent task/profile directive.",
            body=(
                "Treat this as relevant task context, not conversational topic guidance.\n"
                f"{_trim_text_to_token_budget(sources.active_directive, token_budget=max(120, min(420, int(low_budget * 0.12))))}"
            ),
        )
    if sources.thread_rollup:
        append_retrieved_entry(
            late_entries,
            section="thread_rollup",
            title="Compressed older thread context.",
            body=_trim_text_to_token_budget(
                sources.thread_rollup,
                token_budget=max(300, min(1400, int(low_budget * 0.4))),
            ),
        )
    if sources.pending_context_summary:
        append_retrieved_entry(
            late_entries,
            section="pending_context",
            title="Background system events summary (not user-authored).",
            body=(
                "Use this only to reconcile hidden state and never quote event lines directly.\n"
                f"{_trim_text_to_token_budget(sources.pending_context_summary, token_budget=max(140, min(520, int(low_budget * 0.15))))}"
            ),
        )
    if skill_discovery_context and context_engine.should_include_optional_context(
        kind="skill_discovery",
        prompt_mode=prompt_mode,
        should_retrieve=sources.should_retrieve,
    ):
        append_retrieved_entry(
            late_entries,
            section="skill_discovery",
            title="Relevant skill discovery for this turn.",
            body=_trim_text_to_token_budget(
                skill_discovery_context,
                token_budget=max(160, min(620, int(low_budget * 0.18))),
            ),
        )
    if invoked_skill_context and context_engine.should_include_optional_context(
        kind="invoked_skills",
        prompt_mode=prompt_mode,
        should_retrieve=sources.should_retrieve,
    ):
        append_retrieved_entry(
            late_entries,
            section="invoked_skills",
            title=(
                "Previously invoked skill instructions still relevant in this session "
                f"(skills: {', '.join(invoked_skill_names) if invoked_skill_names else 'unknown'})."
            ),
            body=_trim_text_to_token_budget(
                invoked_skill_context,
                token_budget=max(400, min(1800, int(low_budget * 0.45))),
            ),
        )
    if sources.link_alias_context and context_engine.should_include_optional_context(
        kind="link_aliases",
        prompt_mode=prompt_mode,
        should_retrieve=sources.should_retrieve,
    ):
        entry = make_prompt_context_entry(
            section="link_aliases",
            content=_trim_text_to_token_budget(
                sources.link_alias_context,
                token_budget=max(120, min(320, int(low_budget * 0.08))),
            ),
        )
        if entry is not None:
            late_entries.append(entry)
    if sources.memory_grounding:
        append_retrieved_entry(
            late_entries,
            section="memory_grounding",
            title="Relevant long-term memory grounding (dynamic retrieval).",
            body=(
                "Use this to ground historical facts, preferences, directives, projects, technical details, and recalled files. "
                "Treat it as retrieved memory, not as a user-authored message in this turn.\n"
                f"{_trim_text_to_token_budget(sources.memory_grounding, token_budget=500)}"
            ),
        )
    return late_entries


def _build_late_turn_control_text(*, customer_id: str) -> str:
    parts: list[str] = [
        PROMPT_DYNAMIC_BOUNDARY,
        (
            f"customer_id={customer_id}. "
            "Customer scope for customer-scoped tools is resolved automatically from runtime state."
        ),
        (
            'Time context is available through tool_group_exec(group="memory", command="server_time", args_json={}). '
            "Call it before interpreting relative dates, times, reminders, schedules, deadlines, or timezone-sensitive wording. "
            "Do not infer current time from stale conversation history."
        ),
    ]
    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())


def _build_current_turn_context_text(
    *,
    prompt_mode: str,
    turn_mode: str,
    connected_composio_toolkits_context: str = "",
) -> str:
    parts: list[str] = [
        "OPENTULPA_CURRENT_TURN_CONTEXT",
        _content_to_text(_build_prompt_mode_message(prompt_mode).content),  # type: ignore[arg-type]
        _content_to_text(_build_turn_mode_system_message(turn_mode).content),
        connected_composio_toolkits_context,
    ]
    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())


def build_relevant_skill_discovery_context(
    *,
    available_skills: Any,
    selected_names: list[str] | None,
) -> str:
    if not isinstance(available_skills, list):
        return ""
    wanted = {str(name).strip() for name in selected_names or [] if str(name).strip()}
    include_all = not wanted
    lines = [
        "Available skills registry:",
        "This is a compact glossary only. If a skill is needed, call skill_get(name) before relying on its instructions.",
    ]
    seen: set[str] = set()
    for item in available_skills:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name in seen:
            continue
        if not include_all and name not in wanted:
            continue
        seen.add(name)
        description_words = " ".join(str(item.get("description", "")).split()).strip().split()
        description = " ".join(description_words[:20]).strip()
        scope = str(item.get("scope", "")).strip() or "user"
        lines.append(
            f"- {name} ({scope}): {description[:220]}" if description else f"- {name} ({scope})"
        )
    if len(lines) <= 2:
        return ""
    return "\n".join(lines)
