"""Runtime-owned guardrails for consecutive duplicate tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from opentulpa.agent.tool_budget import coerce_tool_group_calls


@dataclass(frozen=True, slots=True)
class ToolActionSignature:
    key: str
    label: str


@dataclass(frozen=True, slots=True)
class DuplicateToolCall:
    tool_call_id: str
    error: str
    signature: str


def duplicate_tool_error(label: str) -> str:
    safe_label = str(label or "tool action").strip() or "tool action"
    return (
        f"DUPLICATE_TOOL_CALL_BLOCKED: {safe_label} already just succeeded. "
        "Do not call the same tool with the same arguments twice in a row. Use the "
        "previous tool result, choose a different next action, or write the final "
        "user-facing answer/blocker now."
    )


def tool_action_signature(call_name: str, args: Any) -> ToolActionSignature | None:
    name = str(call_name or "").strip()
    if not name:
        return None
    if name == "tool_group_exec":
        return _tool_group_exec_signature(args)
    normalized_args = _canonicalize(args)
    return ToolActionSignature(
        key=_signature_key(name, normalized_args),
        label=f"{name}({_short_json(normalized_args)})",
    )


def find_duplicate_tool_calls(
    *,
    requested_calls: list[Any],
    prior_tool_outcomes: Any,
    trace_id: str,
) -> list[DuplicateToolCall]:
    last_success_signature = _last_success_signature(prior_tool_outcomes, trace_id=trace_id)
    seen_in_request: set[str] = set()
    duplicates: list[DuplicateToolCall] = []
    for call in requested_calls:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id", "") or "").strip()
        signature = tool_action_signature(
            str(call.get("name", "") or "").strip(),
            call.get("args", {}) or {},
        )
        if signature is None:
            continue
        if signature.key == last_success_signature or signature.key in seen_in_request:
            duplicates.append(
                DuplicateToolCall(
                    tool_call_id=call_id,
                    error=duplicate_tool_error(signature.label),
                    signature=signature.key,
                )
            )
            continue
        seen_in_request.add(signature.key)
    return duplicates


def _tool_group_exec_signature(args: Any) -> ToolActionSignature | None:
    if not isinstance(args, dict):
        return None
    batch = coerce_tool_group_calls(args.get("calls"))
    if batch:
        normalized_batch: list[dict[str, Any]] = []
        for item in batch:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command", "") or "").strip()
            normalized_batch.append(
                {
                    "group": str(item.get("group", "") or "").strip().lower(),
                    "command": command,
                    "args_json": _canonicalize(_parse_args_json(item.get("args_json"))),
                }
            )
        if not normalized_batch:
            return None
        return ToolActionSignature(
            key=_signature_key("tool_group_exec.batch", normalized_batch),
            label=f"tool_group_exec batch({_short_json(normalized_batch)})",
        )
    command = str(args.get("command", "") or "").strip()
    normalized = {
        "group": str(args.get("group", "") or "").strip().lower(),
        "command": command,
        "args_json": _canonicalize(_parse_args_json(args.get("args_json"))),
    }
    return ToolActionSignature(
        key=_signature_key("tool_group_exec", normalized),
        label=f'tool_group_exec(command="{command}", args_json={_short_json(normalized["args_json"])})',
    )


def _last_success_signature(prior_tool_outcomes: Any, *, trace_id: str) -> str:
    if not isinstance(prior_tool_outcomes, list):
        return ""
    active_trace_id = str(trace_id or "").strip()
    for outcome in reversed(prior_tool_outcomes):
        if not isinstance(outcome, dict):
            continue
        if str(outcome.get("status", "") or "").strip() != "ok":
            continue
        signature = str(outcome.get("tool_signature", "") or "").strip()
        if not signature:
            continue
        outcome_trace_id = str(outcome.get("trace_id", "") or "").strip()
        if active_trace_id and outcome_trace_id and outcome_trace_id != active_trace_id:
            continue
        if active_trace_id and not outcome_trace_id:
            continue
        return signature
    return ""


def _parse_args_json(value: Any) -> Any:
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    if value is None:
        return {}
    return value


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item is not None
        }
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _signature_key(name: str, normalized_args: Any) -> str:
    return json.dumps(
        {"tool": name, "args": normalized_args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _short_json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= 220 else text[:217] + "..."
