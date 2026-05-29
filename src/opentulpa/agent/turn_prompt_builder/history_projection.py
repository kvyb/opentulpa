"""History projection for one runtime prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opentulpa.agent.context_engine import ContextEngine
from opentulpa.agent.lc_messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_message_protocol import (
    collapse_completed_tool_call_segments_for_model as _collapse_completed_tool_call_segments_for_model,
)
from opentulpa.agent.tool_message_protocol import (
    enforce_tool_message_protocol as _enforce_tool_message_protocol,
)
from opentulpa.agent.tool_message_protocol import (
    sanitize_history_messages_for_model as _sanitize_history_messages_for_model,
)
from opentulpa.agent.turn_prompt_builder.entries import (
    make_retrieved_context_entry,
    select_optional_prompt_entries,
)
from opentulpa.agent.turn_prompt_builder.entries import (
    prompt_overhead_tokens as _prompt_overhead_tokens,
)


@dataclass(frozen=True, slots=True)
class HistoryProjection:
    older_history_messages: list[AnyMessage]
    latest_turn_messages: list[AnyMessage]
    selected_summary_entries: list[tuple[str, SystemMessage]]
    stale_summary_text: str
    history_budget: int
    prompt_overhead_tokens: int
    prompt_context_update: dict[str, Any]


def build_history_projection(
    *,
    runtime: Any,
    state: AgentState,
    messages: list[AnyMessage],
    context_engine: ContextEngine,
    prompt_context_update: dict[str, Any],
    prompt_messages_base: list[AnyMessage],
    selected_frozen_late_entries: list[tuple[str, SystemMessage]],
    used_optional_tokens: int,
    optional_context_budget: int,
    max_overhead_tokens: int,
    prompt_budget: int,
) -> HistoryProjection:
    prompt_messages: list[AnyMessage] = [
        *prompt_messages_base,
        *(message for _, message in selected_frozen_late_entries),
    ]
    prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
    history_budget = max(800, prompt_budget - prompt_overhead_tokens)
    sanitized_history = _enforce_tool_message_protocol(_sanitize_history_messages_for_model(messages))
    frozen_history_projection_raw = state.get("frozen_history_projection")
    turn_history_messages = _enforce_tool_message_protocol(_latest_turn_messages(sanitized_history))
    turn_start_index = max(0, len(sanitized_history) - len(turn_history_messages))
    older_history_messages: list[AnyMessage] = []
    stale_summary_text = ""
    history_working_set = context_engine.build_history_working_set(
        sanitized_history,
        token_budget=history_budget,
    )

    if _valid_frozen_history_projection(frozen_history_projection_raw, len(sanitized_history)):
        assert isinstance(frozen_history_projection_raw, dict)
        turn_start_index = int(frozen_history_projection_raw.get("turn_start_index", 0))
        older_history_messages = _enforce_tool_message_protocol(
            _sanitize_history_messages_for_model(
                _normalize_frozen_history_messages(
                    frozen_history_projection_raw.get("older_history_messages")
                )
            )
        )
        stale_summary_text = str(frozen_history_projection_raw.get("stale_summary_text", "")).strip()
    else:
        projected_summary_entries = _select_summary_entries(
            history_working_set.summary_text,
            used_optional_tokens=used_optional_tokens,
            optional_context_budget=optional_context_budget,
        )
        prompt_messages, prompt_overhead_tokens = _fit_prompt_overhead(
            prompt_messages_base=prompt_messages_base,
            selected_frozen_late_entries=selected_frozen_late_entries,
            selected_summary_entries=projected_summary_entries,
            max_overhead_tokens=max_overhead_tokens,
        )
        if projected_summary_entries:
            history_budget = max(800, prompt_budget - prompt_overhead_tokens)
            history_working_set = context_engine.build_history_working_set(
                sanitized_history,
                token_budget=history_budget,
            )
        bounded_messages = _enforce_tool_message_protocol(history_working_set.raw_messages)
        if not _model_uses_current_turn_raw_history_only(runtime):
            bounded_latest_turn = _latest_turn_messages(bounded_messages)
            bounded_latest_turn_count = len(bounded_latest_turn)
            older_history_messages = (
                bounded_messages[:-bounded_latest_turn_count]
                if 0 < bounded_latest_turn_count < len(bounded_messages)
                else []
            )
        stale_summary_text = history_working_set.summary_text
        prompt_context_update["frozen_history_projection"] = {
            "turn_start_index": turn_start_index,
            "older_history_messages": older_history_messages,
            "stale_summary_text": stale_summary_text,
        }

    if _model_uses_current_turn_raw_history_only(runtime):
        older_history_messages = []
    latest_turn_messages = _enforce_tool_message_protocol(sanitized_history[turn_start_index:])
    if _model_uses_current_turn_raw_history_only(runtime):
        latest_turn_messages = _enforce_tool_message_protocol(
            _collapse_completed_tool_call_segments_for_model(latest_turn_messages)
        )
        prompt_context_update["frozen_history_projection"] = {
            "turn_start_index": turn_start_index,
            "older_history_messages": [],
            "stale_summary_text": stale_summary_text,
        }

    selected_summary_entries = _select_summary_entries(
        stale_summary_text,
        used_optional_tokens=used_optional_tokens,
        optional_context_budget=optional_context_budget,
    )
    _, prompt_overhead_tokens = _fit_prompt_overhead(
        prompt_messages_base=prompt_messages_base,
        selected_frozen_late_entries=selected_frozen_late_entries,
        selected_summary_entries=selected_summary_entries,
        max_overhead_tokens=max_overhead_tokens,
    )
    return HistoryProjection(
        older_history_messages=older_history_messages,
        latest_turn_messages=latest_turn_messages,
        selected_summary_entries=selected_summary_entries,
        stale_summary_text=stale_summary_text,
        history_budget=history_budget,
        prompt_overhead_tokens=prompt_overhead_tokens,
        prompt_context_update=prompt_context_update,
    )


def _select_summary_entries(
    summary_text: str,
    *,
    used_optional_tokens: int,
    optional_context_budget: int,
) -> list[tuple[str, SystemMessage]]:
    summary_entry = (
        make_retrieved_context_entry(
            section="stale_history_summary",
            title="Compressed older in-thread context.",
            body=summary_text,
        )
        if summary_text
        else None
    )
    if summary_entry is None:
        return []
    selected, _ = select_optional_prompt_entries(
        [summary_entry],
        initial_used_tokens=used_optional_tokens,
        optional_context_budget=optional_context_budget,
    )
    return selected


def _fit_prompt_overhead(
    *,
    prompt_messages_base: list[AnyMessage],
    selected_frozen_late_entries: list[tuple[str, SystemMessage]],
    selected_summary_entries: list[tuple[str, SystemMessage]],
    max_overhead_tokens: int,
) -> tuple[list[AnyMessage], int]:
    prompt_messages = [
        *prompt_messages_base,
        *(message for _, message in selected_frozen_late_entries),
        *(message for _, message in selected_summary_entries),
    ]
    prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
    while selected_summary_entries and prompt_overhead_tokens > max_overhead_tokens:
        selected_summary_entries.pop()
        prompt_messages = [
            *prompt_messages_base,
            *(message for _, message in selected_frozen_late_entries),
            *(message for _, message in selected_summary_entries),
        ]
        prompt_overhead_tokens = _prompt_overhead_tokens(prompt_messages)
    return prompt_messages, prompt_overhead_tokens


def _normalize_frozen_history_messages(raw: Any) -> list[AnyMessage]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, (HumanMessage, AIMessage, ToolMessage))]


def _valid_frozen_history_projection(raw: Any, message_count: int) -> bool:
    return (
        isinstance(raw, dict)
        and int(raw.get("turn_start_index", -1)) >= 0
        and int(raw.get("turn_start_index", -1)) <= message_count
    )


def _latest_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    if not messages:
        return []
    start = 0
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            start = idx
            break
    return messages[start:]


def _model_uses_current_turn_raw_history_only(runtime: Any) -> bool:
    return "deepseek" in str(getattr(runtime, "model_name", "") or "").strip().lower()
