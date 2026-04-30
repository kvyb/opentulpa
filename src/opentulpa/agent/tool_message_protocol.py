"""Helpers for robust model-visible tool message history."""

from __future__ import annotations

from opentulpa.agent.lc_messages import AIMessage, AnyMessage, SystemMessage, ToolMessage


def sanitize_history_messages_for_model(messages: list[AnyMessage]) -> list[AnyMessage]:
    sanitized: list[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            additional_kwargs = getattr(msg, "additional_kwargs", {}) or {}
            if isinstance(additional_kwargs, dict) and (
                additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call")
            ):
                clean_kwargs = dict(additional_kwargs)
                clean_kwargs.pop("tool_calls", None)
                clean_kwargs.pop("function_call", None)
                sanitized.append(msg.model_copy(update={"additional_kwargs": clean_kwargs}))
                continue
        sanitized.append(msg)
    return sanitized


def enforce_tool_message_protocol(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Ensure model-visible history does not contain orphaned tool-call turns.
    If an AI tool-call turn lacks contiguous matching ToolMessage responses,
    drop that incomplete tool segment to avoid provider INVALID_ARGUMENT errors.
    """
    if not messages:
        return []
    out: list[AnyMessage] = []
    i = 0
    total = len(messages)
    while i < total:
        msg = messages[i]
        if isinstance(msg, ToolMessage):
            i += 1
            continue
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            out.append(msg)
            i += 1
            continue

        call_ids = [str((call or {}).get("id", "")).strip() for call in (msg.tool_calls or [])]
        call_ids = [cid for cid in call_ids if cid]
        j = i + 1
        contiguous_tools: list[ToolMessage] = []
        while j < total and isinstance(messages[j], ToolMessage):
            contiguous_tools.append(messages[j])  # type: ignore[arg-type]
            j += 1
        seen_ids = {
            str(getattr(tool_msg, "tool_call_id", "") or "").strip() for tool_msg in contiguous_tools
        }
        if call_ids and all(cid in seen_ids for cid in call_ids):
            out.append(msg)
            out.extend(contiguous_tools)
        i = j
    return out
