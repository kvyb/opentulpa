"""Helpers for robust model-visible tool message history."""

from __future__ import annotations

import json
from typing import Any

from opentulpa.agent.lc_messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    safe_json as _safe_json,
)


def tool_message_control_payload(message: ToolMessage) -> dict[str, Any]:
    raw_extra = getattr(message, "additional_kwargs", {}) or {}
    if isinstance(raw_extra, dict):
        maybe_control = raw_extra.get("opentulpa_control", {})
        if isinstance(maybe_control, dict):
            return maybe_control
    raw_text = _content_to_text(getattr(message, "content", "")).strip()
    if not raw_text or not raw_text.startswith("{"):
        return {}
    try:
        parsed = json.loads(raw_text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_message_is_approval_pending(message: ToolMessage) -> bool:
    payload = tool_message_control_payload(message)
    status = str(payload.get("status", "")).strip().lower()
    return status == "approval_pending"


def compact_approval_pending_tool_message(message: ToolMessage) -> ToolMessage | None:
    if not tool_message_is_approval_pending(message):
        return None
    payload = tool_message_control_payload(message)
    approval_id = str(payload.get("approval_id", "")).strip()
    compact_payload: dict[str, Any] = {"status": "approval_pending"}
    if approval_id:
        compact_payload["approval_id"] = approval_id
    tool_call_id = str(getattr(message, "tool_call_id", "") or "").strip()
    if not tool_call_id:
        return None
    return ToolMessage(
        content=_safe_json(compact_payload),
        tool_call_id=tool_call_id,
        additional_kwargs={"opentulpa_control": compact_payload},
    )


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
        if isinstance(msg, ToolMessage):
            compact = compact_approval_pending_tool_message(msg)
            if compact is not None:
                sanitized.append(compact)
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
