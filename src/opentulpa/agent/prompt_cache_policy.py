"""Prompt-cache policy helpers for runtime graph prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from opentulpa.agent.model_pool import prompt_cache_breakpoint_message_index
from opentulpa.agent.utils import approx_tokens as _approx_tokens
from opentulpa.agent.utils import content_to_text as _content_to_text

CACHE_STICKY_ROUTING_ANCHOR = (
    "OpenTulpa cache anchor v1. Real conversation messages follow; do not answer this marker."
)


@dataclass(frozen=True)
class PromptCachePlan:
    model_messages: list[Any]
    requested_cacheable_prefix_count: int
    cacheable_prefix_count: int
    cacheable_prefix_mode: str
    cache_breakpoint_index: int | None
    cacheable_prefix_tokens: int
    cacheable_history_messages: list[Any]
    frontier_history_messages: list[Any]
    cacheable_history_tokens: int
    frontier_history_tokens: int


def message_tokens(messages: Sequence[Any]) -> int:
    return sum(max(0, _approx_tokens(_content_to_text(getattr(msg, "content", "")))) for msg in messages)


def build_prompt_cache_plan(
    *,
    prefix_messages: Sequence[Any],
    older_history_messages: Sequence[Any],
    frozen_late_messages: Sequence[Any],
    latest_turn_messages: Sequence[Any],
    dynamic_late_messages: Sequence[Any],
    cache_profile: dict[str, Any],
) -> PromptCachePlan:
    stable_prefix_count = len(prefix_messages)
    prompt_cache_strategy = str(cache_profile.get("strategy", ""))
    cacheable_history_messages: list[Any] = []
    frontier_history_messages: list[Any] = list(latest_turn_messages)
    if prompt_cache_strategy in {"implicit_stable_prefix", "explicit_stable_prefix"}:
        requested_cacheable_prefix_count = stable_prefix_count
        cacheable_prefix_mode = "stable_prefix_only"
        model_messages: list[Any] = [
            *prefix_messages,
            *older_history_messages,
            *frozen_late_messages,
            *latest_turn_messages,
            *dynamic_late_messages,
        ]
    else:
        requested_cacheable_prefix_count = stable_prefix_count + len(older_history_messages)
        cacheable_prefix_mode = "full_older_history"
        model_messages = [
            *prefix_messages,
            *older_history_messages,
            *frozen_late_messages,
            *latest_turn_messages,
            *dynamic_late_messages,
        ]
    cacheable_prefix_count = requested_cacheable_prefix_count
    cache_breakpoint_index: int | None = None
    if bool(cache_profile.get("supports_breakpoints", False)):
        cache_breakpoint_index = prompt_cache_breakpoint_message_index(
            model_messages,
            effective_prefix_count=requested_cacheable_prefix_count,
        )
        if cache_breakpoint_index is not None:
            cacheable_prefix_count = cache_breakpoint_index + 1
    return PromptCachePlan(
        model_messages=model_messages,
        requested_cacheable_prefix_count=requested_cacheable_prefix_count,
        cacheable_prefix_count=cacheable_prefix_count,
        cacheable_prefix_mode=cacheable_prefix_mode,
        cache_breakpoint_index=cache_breakpoint_index,
        cacheable_prefix_tokens=message_tokens(model_messages[:cacheable_prefix_count]),
        cacheable_history_messages=cacheable_history_messages,
        frontier_history_messages=frontier_history_messages,
        cacheable_history_tokens=message_tokens(cacheable_history_messages),
        frontier_history_tokens=message_tokens(frontier_history_messages),
    )
