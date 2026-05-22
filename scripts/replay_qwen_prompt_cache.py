from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from opentulpa.agent.graph_builder import (
    CACHE_STICKY_ROUTING_ANCHOR,
    _compact_qwen_frontier_history,
    _split_qwen_cacheable_history,
)
from opentulpa.agent.lc_messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from opentulpa.agent.model_pool import infer_stable_system_prefix_count
from opentulpa.agent.prompt_sections import PROMPT_DYNAMIC_BOUNDARY
from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _observations_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("body"), dict):
        data = payload["body"].get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _fetch_langfuse_observations(trace_id: str, *, limit: int) -> list[dict[str, Any]]:
    assert trace_id.strip(), "trace_id must not be empty"
    assert limit > 0, "limit must be positive"
    cmd = [
        "npx",
        "--yes",
        "langfuse-cli",
        "api",
        "observations",
        "list",
        "--trace-id",
        trace_id,
        "--type",
        "GENERATION",
        "--fields",
        "basic,time,io,model,usage",
        "--limit",
        str(limit),
        "--json",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
    return _observations_from_payload(json.loads(result.stdout))


def _parse_langfuse_messages(observation: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw_input = observation.get("input")
    if not isinstance(raw_input, str) or not raw_input.strip():
        return None
    try:
        parsed = json.loads(raw_input)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    messages = [item for item in parsed if isinstance(item, dict) and isinstance(item.get("role"), str)]
    return messages or None


def _tool_from_trailing_schema(message: dict[str, Any]) -> dict[str, Any] | None:
    if message.get("role") != "tool" or message.get("tool_call_id"):
        return None
    content = message.get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            return None
    if not isinstance(content, dict):
        return None
    if content.get("type") == "function" and isinstance(content.get("function"), dict):
        return content
    return None


def _split_messages_and_tools(raw_messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    messages = list(raw_messages)
    tools: list[dict[str, Any]] = []
    while messages:
        tool = _tool_from_trailing_schema(messages[-1])
        if tool is None:
            break
        tools.append(tool)
        messages.pop()
    tools.reverse()
    return messages, tools


def _lc_tool_calls(tool_calls: Any) -> list[dict[str, Any]]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or function.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        args = call.get("args")
        if args is None:
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    args = {"arguments": arguments}
            else:
                args = {}
        normalized.append(
            {
                "id": str(call.get("id") or f"call_replay_{index}"),
                "name": name,
                "args": args if isinstance(args, dict) else {"value": args},
            }
        )
    return normalized


def _to_lc_messages(raw_messages: list[dict[str, Any]]) -> list[Any]:
    converted: list[Any] = []
    for raw in raw_messages:
        role = str(raw.get("role") or "")
        content = _strip_cache_control(raw.get("content") or "")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content, tool_calls=_lc_tool_calls(raw.get("tool_calls"))))
        elif role == "tool":
            tool_call_id = str(raw.get("tool_call_id") or "call_replay_missing")
            converted.append(ToolMessage(content=content, tool_call_id=tool_call_id))
    return converted


def _message_content_for_openai(content: Any) -> Any:
    if isinstance(content, list):
        return content
    if content is None:
        return ""
    return content


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_cache_control(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _strip_cache_control(item)
            for key, item in value.items()
            if key != "cache_control"
        }
    return value


def _to_openai_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            converted.append({"role": "system", "content": _message_content_for_openai(message.content)})
        elif isinstance(message, HumanMessage):
            converted.append({"role": "user", "content": _message_content_for_openai(message.content)})
        elif isinstance(message, AIMessage):
            item: dict[str, Any] = {
                "role": "assistant",
                "content": _message_content_for_openai(message.content),
            }
            tool_calls = []
            for call in getattr(message, "tool_calls", None) or []:
                args = call.get("args") if isinstance(call, dict) else {}
                tool_calls.append(
                    {
                        "id": str(call.get("id") or f"call_replay_{len(tool_calls)}"),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or ""),
                            "arguments": json.dumps(args if isinstance(args, dict) else {"value": args}),
                        },
                    }
                )
            if tool_calls:
                item["tool_calls"] = tool_calls
            converted.append(item)
        elif isinstance(message, ToolMessage):
            converted.append(
                {
                    "role": "tool",
                    "content": _message_content_for_openai(message.content),
                    "tool_call_id": str(getattr(message, "tool_call_id", "") or "call_replay_missing"),
                }
            )
    return converted


def _cache_breakpoint_index(messages: list[dict[str, Any]]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        content = messages[index].get("content")
        if isinstance(content, list) and any(
            isinstance(item, dict) and isinstance(item.get("cache_control"), dict) for item in content
        ):
            return index
    return None


def _is_dynamic_system_message(message: Any) -> bool:
    if not isinstance(message, SystemMessage):
        return False
    text = str(getattr(message, "content", "") or "")
    return (
        PROMPT_DYNAMIC_BOUNDARY in text
        or text.startswith("WORKFLOW_SETUP_CONTROL_CARD")
        or text.startswith("Compressed older")
        or text.startswith("Active persistent")
        or text.startswith("Relevant ")
        or text.startswith("Known long-link")
    )


def _usage_value(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int | float):
            return int(value)
    prompt_details = usage.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        for key in keys:
            value = prompt_details.get(key)
            if isinstance(value, int | float):
                return int(value)
    return 0


def _usage_cost(usage: dict[str, Any]) -> float:
    value = usage.get("cost")
    if isinstance(value, int | float):
        return float(value)
    details = usage.get("cost_details")
    if isinstance(details, dict):
        for key in ("total", "upstream_inference_cost"):
            value = details.get(key)
            if isinstance(value, int | float):
                return float(value)
    return 0.0


def _openrouter_call(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    retry_attempts: int,
    retry_delay_seconds: float,
    retry_backoff_multiplier: float,
    retry_max_delay_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    assert api_key.strip(), "api key must not be empty"
    assert messages, "messages must not be empty"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "usage": {"include": True},
    }
    if tools:
        payload["tools"] = tools
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "OpenTulpa qwen cache replay",
        "HTTP-Referer": "https://opentulpa.com",
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        next_delay = max(0.0, retry_delay_seconds)
        for attempt in range(1, max(1, retry_attempts) + 1):
            response = client.post(f"{base_url.rstrip('/')}/chat/completions", headers=headers, json=payload)
            if response.status_code < 400:
                return response.json()
            print(f"provider_error attempt={attempt} status={response.status_code} {response.text[:500]}")
            if response.status_code != 429 or attempt >= max(1, retry_attempts):
                response.raise_for_status()
            time.sleep(next_delay)
            next_delay = min(
                max(0.0, retry_max_delay_seconds),
                max(0.0, next_delay * max(1.0, retry_backoff_multiplier)),
            )
    raise RuntimeError("OpenRouter call ended without a response")


def _select_observations(observations: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for observation in observations:
        raw_messages = _parse_langfuse_messages(observation)
        if raw_messages is None:
            continue
        messages, _ = _split_messages_and_tools(raw_messages)
        if len(messages) < 2:
            continue
        first = messages[0]
        if first.get("role") != "system" or "You are OpenTulpa" not in str(first.get("content") or ""):
            continue
        candidates.append(observation)
    candidates.sort(key=lambda item: str(item.get("startTime") or item.get("createdAt") or ""))
    if len(candidates) <= limit:
        return candidates
    best_start = 0
    best_score = -1
    for start in range(0, len(candidates) - limit + 1):
        window = candidates[start : start + limit]
        score = min(int(item.get("inputUsage") or item.get("totalUsage") or 0) for item in window)
        if score > best_score:
            best_score = score
            best_start = start
    return candidates[best_start : best_start + limit]


def _build_replay_payload(
    runtime: OpenTulpaLangGraphRuntime,
    observation: dict[str, Any],
    *,
    model: str,
    target_tail_tokens: int,
    cache_salt: str,
) -> dict[str, Any]:
    raw_messages = _parse_langfuse_messages(observation)
    assert raw_messages is not None, "selected observation must have parseable messages"
    raw_messages, tools = _split_messages_and_tools(raw_messages)
    lc_messages = _to_lc_messages(raw_messages)
    stable_prefix_count = infer_stable_system_prefix_count(lc_messages)
    if (
        stable_prefix_count < len(lc_messages)
        and isinstance(lc_messages[stable_prefix_count], HumanMessage)
        and str(lc_messages[stable_prefix_count].content or "").startswith(
            CACHE_STICKY_ROUTING_ANCHOR
        )
    ):
        stable_prefix_count += 1
    prefix_messages = lc_messages[:stable_prefix_count]
    if not (
        prefix_messages
        and isinstance(prefix_messages[-1], HumanMessage)
        and str(prefix_messages[-1].content or "").startswith(CACHE_STICKY_ROUTING_ANCHOR)
    ):
        prefix_messages = [*prefix_messages, HumanMessage(content=CACHE_STICKY_ROUTING_ANCHOR)]
    salt = str(cache_salt or "").strip()
    if salt and prefix_messages and isinstance(prefix_messages[0], SystemMessage):
        prefix_messages = [
            SystemMessage(
                content=(
                    f"{str(prefix_messages[0].content or '').rstrip()}\n\n"
                    f"BENCHMARK_CACHE_SALT: {salt}"
                )
            ),
            *prefix_messages[1:],
        ]
    if salt and prefix_messages and isinstance(prefix_messages[-1], HumanMessage):
        prefix_messages = [
            *prefix_messages[:-1],
            HumanMessage(content=f"{CACHE_STICKY_ROUTING_ANCHOR}\nBenchmark cache salt: {salt}"),
        ]
    replay_tail = lc_messages[stable_prefix_count:]
    frozen_late_messages: list[Any] = []
    dynamic_late_messages = [message for message in replay_tail if _is_dynamic_system_message(message)]
    history_messages = [
        message for message in replay_tail if not _is_dynamic_system_message(message)
    ]
    cacheable_history, frontier_history, cache_policy = _split_qwen_cacheable_history(
        older_history_messages=[],
        latest_turn_messages=history_messages,
        target_tail_tokens=target_tail_tokens,
    )
    frontier_history = _compact_qwen_frontier_history(frontier_history)
    cacheable_prefix_count = len(prefix_messages) + len(cacheable_history)
    reordered_messages = [
        *prefix_messages,
        *cacheable_history,
        *frozen_late_messages,
        *frontier_history,
        *dynamic_late_messages,
    ]
    prepared = runtime.prepare_messages_for_prompt_cache(
        reordered_messages,
        model_name=model,
        stable_prefix_count=len(prefix_messages),
        cacheable_prefix_count=cacheable_prefix_count,
    )
    openai_messages = _to_openai_messages(prepared)
    breakpoint_index = _cache_breakpoint_index(openai_messages)
    assert breakpoint_index is not None, "cache breakpoint must be present"
    assert breakpoint_index < len(openai_messages) - 1, "cache breakpoint must stay before live frontier"
    return {
        "messages": openai_messages,
        "tools": tools,
        "stable_prefix_count": len(prefix_messages),
        "cacheable_prefix_count": cacheable_prefix_count,
        "cache_breakpoint_index": breakpoint_index,
        "cache_policy": cache_policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Langfuse OpenTulpa generations against qwen prompt caching.")
    parser.add_argument("--observations-file", type=Path)
    parser.add_argument("--trace-id")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--model", default="qwen/qwen3.7-max")
    parser.add_argument("--target-tail-tokens", type=int, default=300)
    parser.add_argument("--cache-salt", default="")
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--delay-seconds", type=float, default=0.0)
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--retry-delay-seconds", type=float, default=60.0)
    parser.add_argument("--retry-backoff-multiplier", type=float, default=1.0)
    parser.add_argument("--retry-max-delay-seconds", type=float, default=300.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--out", type=Path, default=Path(".tmp/qwen-cache-replay/latest.json"))
    args = parser.parse_args()

    if not args.observations_file and not args.trace_id:
        raise SystemExit("Provide --observations-file or --trace-id")
    if args.limit <= 0 or args.passes <= 0:
        raise SystemExit("--limit and --passes must be positive")

    if args.observations_file:
        observations = _observations_from_payload(_load_json(args.observations_file))
    else:
        observations = _fetch_langfuse_observations(str(args.trace_id), limit=1000)
    selected = _select_observations(observations, limit=args.limit)
    if not selected:
        raise SystemExit("No replayable OpenTulpa generation observations found")

    api_key = os.environ.get("OPENAI_COMPATIBLE_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or ""
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    runtime = OpenTulpaLangGraphRuntime(
        app_url="http://127.0.0.1:8000",
        openrouter_api_key=api_key or "missing",
        openrouter_base_url=base_url,
        model_name=args.model,
        checkpoint_db_path=".opentulpa/qwen-cache-replay.sqlite",
        prompt_caching_enabled=True,
    )

    started_at = datetime.now(tz=UTC).isoformat()
    results: list[dict[str, Any]] = []
    print("pass observation model prompt cached write cost breakpoint cacheable messages tools")
    for pass_index in range(args.passes):
        for observation in selected:
            if results and args.delay_seconds > 0:
                time.sleep(float(args.delay_seconds))
            payload = _build_replay_payload(
                runtime,
                observation,
                model=args.model,
                target_tail_tokens=int(args.target_tail_tokens),
                cache_salt=str(args.cache_salt or ""),
            )
            started = time.perf_counter()
            try:
                response = _openrouter_call(
                    api_key=api_key,
                    base_url=base_url,
                    model=args.model,
                    messages=payload["messages"],
                    tools=payload["tools"],
                    max_tokens=args.max_tokens,
                    retry_attempts=int(args.retry_attempts),
                    retry_delay_seconds=float(args.retry_delay_seconds),
                    retry_backoff_multiplier=float(args.retry_backoff_multiplier),
                    retry_max_delay_seconds=float(args.retry_max_delay_seconds),
                    timeout_seconds=args.timeout_seconds,
                )
            except httpx.HTTPStatusError as exc:
                if not args.continue_on_error:
                    raise
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                status_code = exc.response.status_code if exc.response is not None else 0
                error_text = exc.response.text[:500] if exc.response is not None else str(exc)
                result = {
                    "pass": pass_index + 1,
                    "observation_id": observation.get("id"),
                    "trace_id": observation.get("traceId"),
                    "source_model": observation.get("model"),
                    "target_model": args.model,
                    "input_usage": observation.get("inputUsage"),
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                    "cost_usd": 0.0,
                    "elapsed_ms": elapsed_ms,
                    "error_status": status_code,
                    "error": error_text,
                    **payload,
                }
                result.pop("messages")
                result.pop("tools")
                results.append(result)
                print(
                    f"{result['pass']} {result['observation_id']} {observation.get('model')} "
                    f"ERROR status={status_code} {elapsed_ms}ms"
                )
                continue
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
            result = {
                "pass": pass_index + 1,
                "observation_id": observation.get("id"),
                "trace_id": observation.get("traceId"),
                "source_model": observation.get("model"),
                "target_model": args.model,
                "input_usage": observation.get("inputUsage"),
                "prompt_tokens": _usage_value(usage, "prompt_tokens", "input_tokens"),
                "completion_tokens": _usage_value(usage, "completion_tokens", "output_tokens"),
                "cached_tokens": _usage_value(usage, "cached_tokens", "input_cache_read"),
                "cache_write_tokens": _usage_value(
                    usage,
                    "cache_write_tokens",
                    "input_cache_write",
                    "input_cache_creation",
                ),
                "cost_usd": _usage_cost(usage),
                "elapsed_ms": elapsed_ms,
                **payload,
            }
            result.pop("messages")
            result.pop("tools")
            results.append(result)
            print(
                f"{result['pass']} {result['observation_id']} {observation.get('model')} "
                f"{result['prompt_tokens']} {result['cached_tokens']} {result['cache_write_tokens']} "
                f"{result['cost_usd']:.8f} {result['cache_breakpoint_index']} "
                f"{result['cacheable_prefix_count']} {len(payload['messages'])} {len(payload['tools'])}"
            )

    summary = {
        "started_at": started_at,
        "model": args.model,
        "cache_policy": "tail_sized",
        "target_tail_tokens": args.target_tail_tokens,
        "source": str(args.observations_file) if args.observations_file else {"trace_id": args.trace_id},
        "selected_observation_ids": [item.get("id") for item in selected],
        "results": results,
        "totals": {
            "cached_tokens": sum(int(item["cached_tokens"]) for item in results),
            "cache_write_tokens": sum(int(item["cache_write_tokens"]) for item in results),
            "cost_usd": sum(float(item["cost_usd"]) for item in results),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
