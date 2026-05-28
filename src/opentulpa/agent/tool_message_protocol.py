"""Helpers for robust model-visible tool message history."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.lc_messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from opentulpa.agent.tool_parser import compact_tool_call_record, compact_tool_payload
from opentulpa.agent.utils import content_to_text


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


def collapse_completed_tool_call_segments_for_model(messages: list[AnyMessage]) -> list[AnyMessage]:
    """
    Convert completed historical tool-call protocol segments into plain context.

    Some providers reject replayed assistant/tool protocol history even when the
    original graph state is valid. Keep raw graph state elsewhere; use this only
    for provider-bound prompt projection.
    """
    out: list[AnyMessage] = []
    i = 0
    total = len(messages)
    while i < total:
        msg = messages[i]
        if not isinstance(msg, AIMessage) or not getattr(msg, "tool_calls", None):
            out.append(msg)
            i += 1
            continue

        call_records = _tool_call_records(msg)
        call_ids = [record["id"] for record in call_records if record["id"]]
        j = i + 1
        tool_messages: list[ToolMessage] = []
        while j < total and isinstance(messages[j], ToolMessage):
            tool_messages.append(messages[j])  # type: ignore[arg-type]
            j += 1
        seen_ids = {
            str(getattr(tool_message, "tool_call_id", "") or "").strip()
            for tool_message in tool_messages
        }
        if call_ids and all(call_id in seen_ids for call_id in call_ids):
            out.append(_completed_tool_segment_message(msg, call_records, tool_messages))
            i = j
            continue

        out.append(msg)
        i += 1
    return out


def _tool_call_records(message: AIMessage) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for call in getattr(message, "tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id", "") or "").strip()
        if not call_id:
            continue
        records.append(
            {
                "id": call_id,
                "name": str(call.get("name", "") or "").strip() or "tool",
                "args": call.get("args"),
            }
        )
    return records


def _completed_tool_segment_message(
    assistant: AIMessage,
    call_records: list[dict[str, Any]],
    tool_messages: list[ToolMessage],
) -> SystemMessage:
    by_id = {
        str(getattr(tool_message, "tool_call_id", "") or "").strip(): tool_message
        for tool_message in tool_messages
    }
    lines = ["VERIFIED_TOOL_RESULTS", "Completed tool calls from prior model steps."]
    assistant_text = content_to_text(getattr(assistant, "content", "")).strip()
    if assistant_text:
        lines.append(f"Assistant note: {assistant_text[:600]}")
    for record in call_records:
        tool_message = by_id.get(str(record["id"]))
        result = content_to_text(getattr(tool_message, "content", "") if tool_message else "")
        lines.append(
            compact_tool_call_record(
                tool_name=str(record["name"]),
                args=compact_tool_payload(record.get("args"), value_char_limit=240),
                result=result,
                args_value_char_limit=None,
                result_value_char_limit=1200,
            )
        )
    return SystemMessage(content="\n".join(line for line in lines if line.strip()))
