"""Prompt assembly for one runtime graph agent turn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from opentulpa.agent.context_engine import ContextEngine, ContextSourceProvider
from opentulpa.agent.lc_messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from opentulpa.agent.models import AgentState
from opentulpa.agent.prompt_cache_policy import CACHE_STICKY_ROUTING_ANCHOR
from opentulpa.agent.prompt_policy import (
    build_current_web_search_backend_prompt_message,
)
from opentulpa.agent.prompt_policy import (
    build_system_prompt_message as _build_system_prompt_message,
)
from opentulpa.agent.turn_budget import budget_status_context as _budget_status_context
from opentulpa.agent.turn_context import build_dynamic_turn_context as _build_dynamic_turn_context
from opentulpa.agent.turn_prompt_builder.cache_metadata import (
    build_prompt_cache_metadata,
)
from opentulpa.agent.turn_prompt_builder.entries import (
    normalize_prompt_context_entries,
    select_optional_prompt_entries,
)
from opentulpa.agent.turn_prompt_builder.entries import (
    prompt_overhead_tokens as _prompt_overhead_tokens,
)
from opentulpa.agent.turn_prompt_builder.frozen_context import (
    build_frozen_prompt_context,
    frozen_prompt_context_matches,
)
from opentulpa.agent.turn_prompt_builder.history_projection import build_history_projection
from opentulpa.agent.utils import latest_user_text as _latest_user_text

LOOP_LIMIT_REPAIR_INSTRUCTION = (
    "LOOP_LIMIT_APPROACHING: This turn is near its graph step limit. Do not call more tools. "
    "Write a concise user-facing status update now with what is done, the current blocker, "
    "or the next exact step."
)


@dataclass(frozen=True, slots=True)
class TurnPrompt:
    model_messages: list[Any]
    prompt_context_update: dict[str, Any]
    live_user_steering: list[str]
    skill_state_update: dict[str, Any]
    stable_prefix_count: int
    cacheable_prefix_count: int
    call_context: dict[str, Any]
    prompt_ready_log_fields: dict[str, Any]


async def build_turn_prompt(
    *,
    runtime: Any,
    state: AgentState,
    customer_id: str,
    thread_id: str,
    turn_mode: str,
    prompt_mode: str,
    base_prompt_context_update: dict[str, Any],
    live_user_steering: list[str],
    context_engine: ContextEngine,
    context_provider: ContextSourceProvider,
    loop_limit_near: Callable[[AgentState], bool],
) -> TurnPrompt:
    messages = state.get("messages", [])
    latest_user = _latest_user_text(messages)
    prompt_context_update = dict(base_prompt_context_update)

    cached_query = str(state.get("active_skill_query", "")).strip()
    cached_names = state.get("active_skill_names", []) or []
    cached_available = state.get("active_available_skills", []) or []
    cached_discovery = str(state.get("active_skill_discovery_context", "")).strip()
    cached_invoked_names = state.get("active_invoked_skill_names", []) or []
    cached_invoked_context = str(state.get("active_invoked_skill_context", "")).strip()
    legacy_cached_context = str(state.get("active_skill_context", "")).strip()
    skill_query = cached_query
    skill_names = cached_names if isinstance(cached_names, list) else []
    skill_discovery_context = cached_discovery
    invoked_skill_names = (
        [str(n).strip() for n in cached_invoked_names if str(n).strip()]
        if isinstance(cached_invoked_names, list)
        else []
    )
    invoked_skill_context = cached_invoked_context or legacy_cached_context
    available_skills = cached_available if isinstance(cached_available, list) else []

    prompt_budget = max(4000, int(getattr(runtime, "_context_token_limit", 20000)))
    low_budget = max(1500, int(getattr(runtime, "_context_short_term_low_tokens", 3500)))
    optional_context_budget = max(1000, min(3600, int(low_budget * 0.7)))
    frozen_prompt_context_raw = state.get("frozen_prompt_context")
    if frozen_prompt_context_matches(
        frozen_prompt_context_raw,
        latest_user=latest_user,
        customer_id=customer_id,
        prompt_mode=prompt_mode,
        turn_mode=turn_mode,
    ):
        frozen_prompt_context = dict(frozen_prompt_context_raw or {})
    else:
        frozen_result = await build_frozen_prompt_context(
            context_provider=context_provider,
            state=state,
            customer_id=customer_id,
            thread_id=thread_id,
            prompt_mode=prompt_mode,
            turn_mode=turn_mode,
            latest_user=latest_user,
            low_budget=low_budget,
            context_engine=context_engine,
            available_skills=available_skills,
            skill_names=skill_names,
            skill_query=skill_query,
            skill_discovery_context=skill_discovery_context,
            invoked_skill_names=invoked_skill_names,
            invoked_skill_context=invoked_skill_context,
        )
        frozen_prompt_context = frozen_result.context
        prompt_context_update["frozen_prompt_context"] = frozen_prompt_context
        skill_state = frozen_result.skill_state
        available_skills = skill_state.available_skills
        skill_names = skill_state.skill_names
        skill_query = skill_state.skill_query
        skill_discovery_context = skill_state.skill_discovery_context

    stable_prompt_messages: list[AnyMessage] = [_build_system_prompt_message()]
    stable_prompt_sections = ["stable_core_policy"]
    late_entries = normalize_prompt_context_entries(frozen_prompt_context.get("late_entries"))
    late_control_content = str(frozen_prompt_context.get("late_control_content", "")).strip()
    late_control_sections = [
        str(section).strip()
        for section in (frozen_prompt_context.get("late_control_sections") or [])
        if str(section).strip()
    ]
    current_turn_context_content = str(
        frozen_prompt_context.get("current_turn_context_content", "")
    ).strip()
    current_turn_context_sections = [
        str(section).strip()
        for section in (frozen_prompt_context.get("current_turn_context_sections") or [])
        if str(section).strip()
    ]

    stable_turn_context_messages: list[AnyMessage] = []
    stable_turn_context_sections: list[str] = []
    stable_turn_context_messages.append(build_current_web_search_backend_prompt_message())
    stable_turn_context_sections.append("web_search_backend")

    prefix_messages: list[AnyMessage] = [
        *stable_prompt_messages,
        *stable_turn_context_messages,
        HumanMessage(content=CACHE_STICKY_ROUTING_ANCHOR),
    ]
    prefix_sections = [
        *stable_prompt_sections,
        *stable_turn_context_sections,
        "cache_sticky_routing_anchor",
    ]

    late_control_message = SystemMessage(content=late_control_content) if late_control_content else None
    selected_frozen_late_entries, used_optional_tokens = select_optional_prompt_entries(
        late_entries,
        initial_used_tokens=0,
        optional_context_budget=optional_context_budget,
    )
    prompt_messages_base: list[AnyMessage] = [
        *prefix_messages,
        *([late_control_message] if late_control_message is not None else []),
    ]
    max_overhead_tokens = max(1400, int(prompt_budget * 0.72))
    prompt_messages: list[AnyMessage] = [
        *prompt_messages_base,
        *(message for _, message in selected_frozen_late_entries),
    ]
    prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
    while selected_frozen_late_entries and prompt_overhead_tokens > max_overhead_tokens:
        selected_frozen_late_entries.pop()
        prompt_messages = [
            *prompt_messages_base,
            *(message for _, message in selected_frozen_late_entries),
        ]
        prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)

    history_projection = build_history_projection(
        runtime=runtime,
        state=state,
        messages=messages,
        context_engine=context_engine,
        prompt_context_update=prompt_context_update,
        prompt_messages_base=prompt_messages_base,
        selected_frozen_late_entries=selected_frozen_late_entries,
        used_optional_tokens=used_optional_tokens,
        optional_context_budget=optional_context_budget,
        max_overhead_tokens=max_overhead_tokens,
        prompt_budget=prompt_budget,
    )
    prompt_context_update = history_projection.prompt_context_update
    older_history_messages = history_projection.older_history_messages
    latest_turn_messages = history_projection.latest_turn_messages
    selected_summary_entries = history_projection.selected_summary_entries
    history_budget = history_projection.history_budget
    prompt_overhead_tokens = history_projection.prompt_overhead_tokens
    frozen_late_messages: list[AnyMessage] = [
        *([late_control_message] if late_control_message is not None else []),
        *(message for _, message in selected_frozen_late_entries),
        *(message for _, message in selected_summary_entries),
    ]
    dynamic_context_state_raw = dict(state)
    dynamic_context_state_raw.update(prompt_context_update)
    dynamic_context_state = cast(AgentState, dynamic_context_state_raw)
    dynamic_turn_context = _build_dynamic_turn_context(
        runtime=runtime,
        state=dynamic_context_state,
        customer_id=customer_id,
        thread_id=thread_id,
        turn_mode=turn_mode,
        current_turn_context_content=current_turn_context_content,
        current_turn_context_sections=current_turn_context_sections,
        tool_outcomes=state.get("tool_outcomes"),
        budget_context=_budget_status_context(prompt_context_update.get("turn_budget")),
        loop_limit_repair_instruction=LOOP_LIMIT_REPAIR_INSTRUCTION,
        loop_limit_near=loop_limit_near(dynamic_context_state),
        live_user_steering=live_user_steering,
        workflow_setup_repair_instruction=str(
            state.get("workflow_setup_repair_instruction", "") or ""
        ).strip(),
    )
    dynamic_late_messages = dynamic_turn_context.messages
    dynamic_late_sections = dynamic_turn_context.sections
    prompt_section_names = [
        *prefix_sections,
        *late_control_sections,
        *(section for section, _ in selected_frozen_late_entries),
        *(section for section, _ in selected_summary_entries),
        *dynamic_late_sections,
    ]
    optional_context_messages = (
        len(selected_frozen_late_entries)
        + len(selected_summary_entries)
    )
    cache_metadata = build_prompt_cache_metadata(
        runtime=runtime,
        state=dict(state),
        customer_id=customer_id,
        thread_id=thread_id,
        turn_mode=turn_mode,
        prompt_mode=prompt_mode,
        prefix_messages=prefix_messages,
        older_history_messages=older_history_messages,
        frozen_late_messages=frozen_late_messages,
        latest_turn_messages=latest_turn_messages,
        dynamic_late_messages=dynamic_late_messages,
        prompt_section_names=prompt_section_names,
        prompt_overhead_tokens=prompt_overhead_tokens,
        history_budget=history_budget,
        optional_context_messages=optional_context_messages,
        context_engine=context_engine,
    )
    skill_state_update: dict[str, Any] = {}
    if skill_query:
        skill_state_update = {
            "active_skill_query": skill_query,
            "active_skill_names": skill_names,
            "active_available_skills": available_skills,
            "active_skill_discovery_context": skill_discovery_context,
            "active_invoked_skill_context": invoked_skill_context,
            "active_invoked_skill_names": invoked_skill_names,
            "active_skill_context": invoked_skill_context,
        }
    return TurnPrompt(
        model_messages=cache_metadata.cache_plan.model_messages,
        prompt_context_update=prompt_context_update,
        live_user_steering=live_user_steering,
        skill_state_update=skill_state_update,
        stable_prefix_count=cache_metadata.stable_prefix_count,
        cacheable_prefix_count=cache_metadata.cache_plan.cacheable_prefix_count,
        call_context=cache_metadata.call_context,
        prompt_ready_log_fields=cache_metadata.prompt_ready_log_fields,
    )
