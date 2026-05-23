"""Prompt-cache policy helpers for runtime graph prompts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from opentulpa.agent.context_engineer import ContextEngineer
from opentulpa.agent.lc_messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from opentulpa.agent.model_pool import prompt_cache_breakpoint_message_index
from opentulpa.agent.utils import approx_tokens as _approx_tokens
from opentulpa.agent.utils import content_to_text as _content_to_text

CACHE_STICKY_ROUTING_ANCHOR = (
    "OpenTulpa cache anchor v1. Real conversation messages follow; do not answer this marker."
)
QWEN_CACHE_VOLATILE_HISTORY_TAIL_TOKENS = 300
QWEN_CACHE_INITIAL_COMMIT_TOKENS = 1024
QWEN_CACHE_HISTORY_COMMIT_TOKENS = 4000


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


def tail_sized_history_message_count(
    token_counts: list[int],
    *,
    target_tail_tokens: int = QWEN_CACHE_VOLATILE_HISTORY_TAIL_TOKENS,
) -> int:
    safe_counts = [max(0, int(token_count)) for token_count in token_counts]
    if not safe_counts:
        return 0
    included = 0
    remaining_tokens = sum(safe_counts)
    safe_target = max(0, int(target_tail_tokens))
    for token_count in safe_counts:
        if remaining_tokens <= safe_target:
            break
        included += 1
        remaining_tokens -= token_count
    return included


def _committed_tail_sized_history_message_count(
    token_counts: list[int],
    *,
    target_tail_tokens: int = QWEN_CACHE_VOLATILE_HISTORY_TAIL_TOKENS,
    initial_commit_tokens: int = QWEN_CACHE_INITIAL_COMMIT_TOKENS,
    commit_token_step: int = QWEN_CACHE_HISTORY_COMMIT_TOKENS,
) -> int:
    tail_sized_count = tail_sized_history_message_count(
        token_counts,
        target_tail_tokens=target_tail_tokens,
    )
    if tail_sized_count <= 0:
        return 0
    safe_counts = [max(0, int(token_count)) for token_count in token_counts[:tail_sized_count]]
    total_tokens = sum(safe_counts)
    safe_initial = max(1, int(initial_commit_tokens))
    if total_tokens < safe_initial:
        return 0
    safe_step = max(1, int(commit_token_step))
    committed_tokens = safe_initial + ((total_tokens - safe_initial) // safe_step) * safe_step
    included = 0
    included_tokens = 0
    for token_count in safe_counts:
        next_tokens = included_tokens + token_count
        if next_tokens > committed_tokens and included > 0:
            break
        included += 1
        included_tokens = next_tokens
    return included


def _latest_turn_frontier_start_index(
    latest_turn_messages: Sequence[Any],
) -> int:
    if not latest_turn_messages:
        return 0
    last_index = len(latest_turn_messages) - 1
    last = latest_turn_messages[last_index]
    if isinstance(last, HumanMessage):
        return last_index
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return last_index
    if isinstance(last, ToolMessage):
        tool_call_id = str(getattr(last, "tool_call_id", "") or "").strip()
        previous_index = last_index - 1
        if previous_index >= 0 and isinstance(latest_turn_messages[previous_index], AIMessage):
            call_ids = set(ContextEngineer._tool_call_ids(latest_turn_messages[previous_index]))
            if not tool_call_id or tool_call_id in call_ids:
                return previous_index
        return last_index
    return len(latest_turn_messages)


def qwen_cache_safe_history_count(
    messages: Sequence[Any],
    requested_count: int,
) -> int:
    safe_count = max(0, min(int(requested_count), len(messages)))
    while safe_count > 0:
        last = messages[safe_count - 1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            safe_count -= 1
            continue
        if isinstance(last, ToolMessage):
            tool_call_id = str(getattr(last, "tool_call_id", "") or "").strip()
            for previous in reversed(messages[: safe_count - 1]):
                if not isinstance(previous, AIMessage):
                    continue
                call_ids = set(ContextEngineer._tool_call_ids(previous))
                if not call_ids:
                    continue
                if not tool_call_id or tool_call_id in call_ids:
                    return safe_count
                break
            safe_count -= 1
            continue
        if _content_to_text(getattr(last, "content", "")).strip():
            return safe_count
        safe_count -= 1
    return 0


def split_qwen_cacheable_history(
    *,
    older_history_messages: Sequence[Any],
    latest_turn_messages: Sequence[Any],
    target_tail_tokens: int = QWEN_CACHE_VOLATILE_HISTORY_TAIL_TOKENS,
) -> tuple[list[Any], list[Any], str]:
    frontier_start = _latest_turn_frontier_start_index(latest_turn_messages)
    stable_candidates = [*older_history_messages, *latest_turn_messages[:frontier_start]]
    stable_count = _committed_tail_sized_history_message_count(
        [message_tokens([message]) for message in stable_candidates],
        target_tail_tokens=target_tail_tokens,
    )
    stable_count = qwen_cache_safe_history_count(stable_candidates, stable_count)
    if stable_count <= 0:
        return [], [*stable_candidates, *latest_turn_messages[frontier_start:]], "stable_prefix_tail_only"
    cacheable_history = [
        _qwen_cacheable_history_message(stable_candidates[:stable_count])
    ]
    frontier_history = [*stable_candidates[stable_count:], *latest_turn_messages[frontier_start:]]
    return cacheable_history, frontier_history, "committed_tail_sized_history"


def _qwen_message_role(message: AnyMessage) -> str:
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        return "system"
    return type(message).__name__


def _qwen_cacheable_history_item(message: AnyMessage) -> dict[str, Any]:
    item: dict[str, Any] = {
        "role": _qwen_message_role(message),
        "content": _content_to_text(getattr(message, "content", "")).strip(),
    }
    if isinstance(message, HumanMessage):
        text = item["content"]
        if text.startswith("INTERNAL_ONBOARDING_SEED."):
            item["role"] = "system"
            item["note"] = "internal_onboarding_seed"
    if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
        item["tool_calls"] = getattr(message, "tool_calls", []) or []
    if isinstance(message, ToolMessage):
        item["tool_call_id"] = str(getattr(message, "tool_call_id", "") or "").strip()
    return item


def _qwen_cacheable_history_message(messages: Sequence[Any]) -> HumanMessage:
    payload = [_qwen_cacheable_history_item(message) for message in messages]
    return HumanMessage(
        content=(
            "QWEN_CACHEABLE_COMMITTED_HISTORY\n"
            "Stable completed prior conversation/tool history. Use as context; live frontier follows later.\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
        )
    )


def _qwen_volatile_history_message(messages: Sequence[Any]) -> HumanMessage:
    payload = [_qwen_cacheable_history_item(message) for message in messages]
    return HumanMessage(
        content=(
            "QWEN_VOLATILE_RECENT_HISTORY\n"
            "Recent uncommitted conversation/tool history. Use as context; current live frontier follows.\n"
            f"{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
        )
    )


def compact_qwen_frontier_history(messages: Sequence[Any]) -> list[Any]:
    frontier_start = _latest_turn_frontier_start_index(messages)
    if frontier_start <= 0:
        return list(messages)
    return [
        _qwen_volatile_history_message(messages[:frontier_start]),
        *messages[frontier_start:],
    ]


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
    cacheable_history_messages: list[Any] = list(older_history_messages)
    frontier_history_messages: list[Any] = list(latest_turn_messages)
    if prompt_cache_strategy == "explicit_committed_breakpoint":
        cacheable_history_messages, frontier_history_messages, cacheable_prefix_mode = (
            split_qwen_cacheable_history(
                older_history_messages=older_history_messages,
                latest_turn_messages=latest_turn_messages,
            )
        )
        frontier_history_messages = compact_qwen_frontier_history(frontier_history_messages)
        requested_cacheable_prefix_count = stable_prefix_count + len(cacheable_history_messages)
        model_messages: list[Any] = [
            *prefix_messages,
            *cacheable_history_messages,
            *frozen_late_messages,
            *frontier_history_messages,
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
