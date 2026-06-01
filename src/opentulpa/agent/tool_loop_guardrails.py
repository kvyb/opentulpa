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
        f"DUPLICATE_TOOL_CALL_PREVIOUS_SUCCESS: {safe_label} already just succeeded. "
        "The action was completed by the previous tool result; this repeat was not run. "
        "Do not repair arguments or retry this same call. Use the previous result, choose "
        "the next different action, or write the final user-facing answer/blocker now."
    )


def tool_action_signature(call_name: str, args: Any) -> ToolActionSignature | None:
    signatures = tool_action_signatures(call_name, args)
    if not signatures:
        return None
    if len(signatures) == 1:
        return signatures[0]
    signature_keys = [signature.key for signature in signatures]
    return ToolActionSignature(
        key=_signature_key("tool_group_exec.batch", signature_keys),
        label=f"tool_group_exec batch({_short_json([item.label for item in signatures])})",
    )


def tool_action_signatures(call_name: str, args: Any) -> list[ToolActionSignature]:
    name = str(call_name or "").strip()
    if not name:
        return []
    if name == "tool_group_exec":
        return _tool_group_exec_signatures(args)
    normalized_args = _canonicalize(args)
    return [
        ToolActionSignature(
            key=_signature_key(name, normalized_args),
            label=f"{name}({_short_json(normalized_args)})",
        )
    ]


def find_duplicate_tool_calls(
    *,
    requested_calls: list[Any],
    prior_tool_outcomes: Any,
    trace_id: str,
) -> list[DuplicateToolCall]:
    last_success_signatures = _last_success_signatures(prior_tool_outcomes, trace_id=trace_id)
    seen_in_request: set[str] = set()
    duplicates: list[DuplicateToolCall] = []
    for call in requested_calls:
        if not isinstance(call, dict):
            continue
        call_id = str(call.get("id", "") or "").strip()
        signatures = tool_action_signatures(
            str(call.get("name", "") or "").strip(),
            call.get("args", {}) or {},
        )
        if not signatures:
            continue
        duplicate_signature = next(
            (
                signature
                for signature in signatures
                if signature.key in last_success_signatures or signature.key in seen_in_request
            ),
            None,
        )
        if duplicate_signature is not None:
            duplicates.append(
                DuplicateToolCall(
                    tool_call_id=call_id,
                    error=duplicate_tool_error(duplicate_signature.label),
                    signature=duplicate_signature.key,
                )
            )
            continue
        seen_in_request.update(signature.key for signature in signatures)
    return duplicates


def _tool_group_exec_signatures(args: Any) -> list[ToolActionSignature]:
    if not isinstance(args, dict):
        return []
    batch = coerce_tool_group_calls(args.get("calls"))
    if batch:
        signatures: list[ToolActionSignature] = []
        for item in batch:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command", "") or "").strip()
            normalized = _normalized_tool_group_action(
                group=item.get("group"),
                command=command,
                args_json=item.get("args_json"),
            )
            signatures.append(
                ToolActionSignature(
                    key=_signature_key("tool_group_exec", normalized),
                    label=(
                        f'tool_group_exec(command="{command}", '
                        f'args_json={_short_json(normalized["args_json"])})'
                    ),
                )
            )
        return signatures
    command = str(args.get("command", "") or "").strip()
    normalized = _normalized_tool_group_action(
        group=args.get("group"),
        command=command,
        args_json=args.get("args_json"),
    )
    return [
        ToolActionSignature(
            key=_signature_key("tool_group_exec", normalized),
            label=f'tool_group_exec(command="{command}", args_json={_short_json(normalized["args_json"])})',
        )
    ]


def _normalized_tool_group_action(*, group: Any, command: str, args_json: Any) -> dict[str, Any]:
    return {
        "group": str(group or "").strip().lower(),
        "command": str(command or "").strip(),
        "args_json": _canonicalize(_parse_args_json(args_json)),
    }


def _last_success_signatures(prior_tool_outcomes: Any, *, trace_id: str) -> set[str]:
    if not isinstance(prior_tool_outcomes, list):
        return set()
    active_trace_id = str(trace_id or "").strip()
    for outcome in reversed(prior_tool_outcomes):
        if not isinstance(outcome, dict):
            continue
        if str(outcome.get("status", "") or "").strip() != "ok":
            continue
        signatures = _outcome_signatures(outcome)
        if not signatures:
            continue
        outcome_trace_id = str(outcome.get("trace_id", "") or "").strip()
        if active_trace_id and outcome_trace_id and outcome_trace_id != active_trace_id:
            continue
        if active_trace_id and not outcome_trace_id:
            continue
        return signatures
    return set()


def _outcome_signatures(outcome: dict[str, Any]) -> set[str]:
    raw_signatures = outcome.get("tool_signatures")
    signatures: set[str] = set()
    if isinstance(raw_signatures, list):
        signatures.update(str(item).strip() for item in raw_signatures if str(item).strip())
    signature = str(outcome.get("tool_signature", "") or "").strip()
    if signature:
        signatures.add(signature)
    return signatures


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
