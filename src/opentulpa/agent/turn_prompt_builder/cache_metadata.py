"""Prompt-cache planning and prompt metadata assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentulpa.agent.context_engine import ContextEngine
from opentulpa.agent.lc_messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from opentulpa.agent.prompt_cache_policy import PromptCachePlan
from opentulpa.agent.prompt_cache_policy import build_prompt_cache_plan as _build_prompt_cache_plan
from opentulpa.agent.prompt_cache_policy import message_tokens as _message_tokens


@dataclass(frozen=True, slots=True)
class PromptCacheMetadata:
    cache_plan: PromptCachePlan
    prompt_ready_log_fields: dict[str, Any]
    call_context: dict[str, Any]
    stable_prefix_count: int


def build_prompt_cache_metadata(
    *,
    runtime: Any,
    state: dict[str, Any],
    customer_id: str,
    thread_id: str,
    turn_mode: str,
    prompt_mode: str,
    prefix_messages: list[AnyMessage],
    older_history_messages: list[AnyMessage],
    frozen_late_messages: list[AnyMessage],
    latest_turn_messages: list[AnyMessage],
    dynamic_late_messages: list[AnyMessage],
    prompt_section_names: list[str],
    prompt_overhead_tokens: int,
    history_budget: int,
    optional_context_messages: int,
    context_engine: ContextEngine,
) -> PromptCacheMetadata:
    cache_profile = _prompt_cache_profile(runtime)
    stable_prefix_count = len(prefix_messages)
    actual_history_messages = [*older_history_messages, *latest_turn_messages]
    raw_chat_history_count = sum(
        1 for msg in actual_history_messages if isinstance(msg, (HumanMessage, AIMessage))
    )
    raw_tool_history_count = sum(1 for msg in actual_history_messages if isinstance(msg, ToolMessage))
    protected_history_count = len(context_engine._protected_suffix_indices(actual_history_messages))
    stable_prefix_tokens = _message_tokens(prefix_messages)
    frozen_late_tokens = _message_tokens(frozen_late_messages)
    dynamic_late_tokens = _message_tokens(dynamic_late_messages)
    older_history_tokens = _message_tokens(older_history_messages)
    latest_turn_tokens = _message_tokens(latest_turn_messages)
    cache_plan = _build_prompt_cache_plan(
        prefix_messages=prefix_messages,
        older_history_messages=older_history_messages,
        frozen_late_messages=frozen_late_messages,
        latest_turn_messages=latest_turn_messages,
        dynamic_late_messages=dynamic_late_messages,
        cache_profile=cache_profile,
    )
    shared_fields = {
        "stable_prefix_count": stable_prefix_count,
        "stable_prefix_tokens": stable_prefix_tokens,
        "requested_cacheable_prefix_count": cache_plan.requested_cacheable_prefix_count,
        "cacheable_prefix_count": cache_plan.cacheable_prefix_count,
        "cacheable_prefix_tokens": cache_plan.cacheable_prefix_tokens,
        "cacheable_prefix_mode": cache_plan.cacheable_prefix_mode,
        "cache_breakpoint_index": cache_plan.cache_breakpoint_index,
        "frozen_late_tokens": frozen_late_tokens,
        "dynamic_late_tokens": dynamic_late_tokens,
        "cacheable_history_tokens": cache_plan.cacheable_history_tokens,
        "frontier_history_tokens": cache_plan.frontier_history_tokens,
        "older_history_tokens": older_history_tokens,
        "latest_turn_tokens": latest_turn_tokens,
        "prompt_overhead_tokens": prompt_overhead_tokens,
        "history_message_count": len(actual_history_messages),
        "raw_chat_history_count": raw_chat_history_count,
        "raw_tool_history_count": raw_tool_history_count,
        "protected_history_count": protected_history_count,
        "optional_context_messages": optional_context_messages,
    }
    prompt_ready_log_fields = {
        "prompt_message_count": len(cache_plan.model_messages),
        "history_budget": history_budget,
        "prompt_sections": ",".join(prompt_section_names),
        "prompt_cache_strategy": str(cache_profile.get("strategy", "")),
        "prompt_cache_enabled": bool(cache_profile.get("enabled", False)),
        "prompt_cache_breakpoints": bool(cache_profile.get("supports_breakpoints", False)),
        "prompt_cache_top_level": bool(cache_profile.get("supports_top_level", False)),
        "turn_mode": turn_mode,
        **shared_fields,
    }
    call_context = {
        "call_site": "graph_agent",
        "trace_id": state.get("agent_trace_id"),
        "thread_id": thread_id,
        "customer_id": customer_id,
        "turn_mode": turn_mode,
        "prompt_mode": prompt_mode,
        "_langfuse_graph_callback_covers_call": bool(state.get("langfuse_graph_callback_attached")),
        "prompt_sections": prompt_section_names,
        **shared_fields,
    }
    return PromptCacheMetadata(
        cache_plan=cache_plan,
        prompt_ready_log_fields=prompt_ready_log_fields,
        call_context=call_context,
        stable_prefix_count=stable_prefix_count,
    )


def _prompt_cache_profile(runtime: Any) -> dict[str, Any]:
    cache_profile_fn = getattr(runtime, "prompt_cache_profile", None)
    if not callable(cache_profile_fn):
        return {}
    try:
        return dict(cache_profile_fn())
    except Exception:
        return {}
