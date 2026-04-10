#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _list_records(records: list[dict[str, Any]], *, limit: int) -> str:
    lines: list[str] = []
    for record in records[-max(1, int(limit)) :]:
        lines.append(
            " | ".join(
                [
                    str(record.get("ts", "")),
                    str(record.get("trace_id", "")) or "-",
                    str(record.get("call_site", "")) or "-",
                    str(record.get("model_name", "")) or "-",
                    f"prompt={record.get('native_tokens_prompt') or '-'}",
                    f"completion={record.get('native_tokens_completion') or '-'}",
                ]
            )
        )
    return "\n".join(lines).strip()


def _decompose_record(record: dict[str, Any]) -> str:
    lines = [
        f"ts: {record.get('ts', '')}",
        f"trace_id: {record.get('trace_id', '')}",
        f"call_site: {record.get('call_site', '')}",
        f"model_name: {record.get('model_name', '')}",
        f"thread_id: {record.get('thread_id', '')}",
        f"customer_id: {record.get('customer_id', '')}",
        f"turn_mode: {record.get('turn_mode', '')}",
        f"prompt_mode: {record.get('prompt_mode', '')}",
        f"stable_prefix_count: {record.get('stable_prefix_count', 0)}",
        f"prompt_sections: {', '.join(record.get('prompt_sections') or [])}",
        f"prompt_overhead_tokens: {record.get('prompt_overhead_tokens', '')}",
        f"history_message_count: {record.get('history_message_count', '')}",
        f"raw_chat_history_count: {record.get('raw_chat_history_count', '')}",
        f"raw_tool_history_count: {record.get('raw_tool_history_count', '')}",
        f"optional_context_messages: {record.get('optional_context_messages', '')}",
        f"native_tokens_prompt: {record.get('native_tokens_prompt', '')}",
        f"native_tokens_completion: {record.get('native_tokens_completion', '')}",
        "",
        "Prompt messages:",
    ]
    for idx, message in enumerate(record.get("prompt_messages") or []):
        if not isinstance(message, dict):
            continue
        lines.append(
            f"{idx:02d}. role={message.get('role', '')} "
            f"type={message.get('type', '')} "
            f"approx_tokens={message.get('approx_tokens', '')}"
        )
        text = str(message.get("text", "") or "")
        if text:
            lines.append(text)
        else:
            lines.append(json.dumps(message.get("content"), ensure_ascii=False))
        lines.append("")
    lines.extend(
        [
            "Response text:",
            str(record.get("response_text", "") or ""),
        ]
    )
    response_tool_calls = record.get("response_tool_calls")
    if response_tool_calls:
        lines.extend(
            [
                "",
                "Response tool calls:",
                json.dumps(response_tool_calls, ensure_ascii=False, indent=2),
            ]
        )
    error = str(record.get("error", "") or "").strip()
    if error:
        lines.extend(["", f"Error: {error}"])
    return "\n".join(lines).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local OpenTulpa LLM call traces.")
    parser.add_argument(
        "--path",
        default=".opentulpa/logs/llm_call_traces.jsonl",
        help="Path to the local LLM call trace JSONL file.",
    )
    parser.add_argument("--limit", type=int, default=10, help="How many recent records to list.")
    parser.add_argument("--trace-id", default="", help="Trace id to inspect in full.")
    parser.add_argument("--latest", action="store_true", help="Show the latest record in full.")
    args = parser.parse_args()

    path = Path(args.path).resolve()
    records = _load_records(path)
    if not records:
        print(f"No records found at {path}")
        return 1
    if args.trace_id:
        for record in reversed(records):
            if str(record.get("trace_id", "")).strip() == str(args.trace_id).strip():
                print(_decompose_record(record))
                return 0
        print(f"Trace id not found: {args.trace_id}")
        return 1
    if args.latest:
        print(_decompose_record(records[-1]))
        return 0
    print(_list_records(records, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
