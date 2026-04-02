"""Thread-context compaction and rollup persistence helpers."""

from __future__ import annotations

from contextlib import suppress
import re
from typing import Any

from opentulpa.agent.lc_messages import HumanMessage, SystemMessage
from opentulpa.agent.utils import (
    approx_tokens as _approx_tokens,
)
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    message_to_text as _message_to_text,
)

_SECRET_PATTERNS = (
    r"\b(?:api[_-]?key|api[_-]?hash|client[_-]?secret|access[_-]?token|refresh[_-]?token|stringsession|session[_-]?token)\b\s*[:=]\s*[^\s,;]+",
    r"\bmlsn\.[A-Za-z0-9._-]+",
    r"\bGOCSPX-[A-Za-z0-9._-]+",
)


def _rollup_token_budget(runtime: Any) -> int:
    return max(500, int(getattr(runtime, "_context_rollup_tokens", 2200)))


def _short_term_high_token_budget(runtime: Any) -> int:
    configured = int(
        getattr(
            runtime,
            "_context_short_term_high_tokens",
            getattr(runtime, "_context_token_limit", 12000),
        )
    )
    return max(2000, configured)


def _short_term_low_token_budget(runtime: Any) -> int:
    configured = int(
        getattr(
            runtime,
            "_context_short_term_low_tokens",
            getattr(runtime, "_context_recent_tokens", 3500),
        )
    )
    high = _short_term_high_token_budget(runtime)
    return max(1000, min(configured, max(1000, high - 500)))


def _compaction_source_budget(runtime: Any) -> int:
    return max(
        _rollup_token_budget(runtime),
        int(getattr(runtime, "_context_compaction_source_tokens", 100000)),
    )


def _trim_text_to_token_budget(text: str, token_budget: int) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    budget = max(1, int(token_budget))
    if _approx_tokens(raw) <= budget:
        return raw

    max_chars = max(800, budget * 4)
    if len(raw) <= max_chars:
        return raw

    # Keep both earliest and latest sections to preserve stable preferences and newest updates.
    reserve = max(20, max_chars // 2 - 8)
    compact = f"{raw[:reserve]}\n...\n{raw[-reserve:]}"
    while _approx_tokens(compact) > budget and reserve > 64:
        reserve = max(64, int(reserve * 0.85))
        compact = f"{raw[:reserve]}\n...\n{raw[-reserve:]}"
    if _approx_tokens(compact) <= budget:
        return compact.strip()
    return raw[:max_chars].strip()


def _sanitize_rollup_text(text: str) -> str:
    cleaned = str(text or "")
    for pattern in _SECRET_PATTERNS:
        cleaned = re.sub(pattern, "[redacted secret reference]", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def split_rollup_sections(text: str) -> dict[str, str]:
    raw = _sanitize_rollup_text(text)
    if not raw:
        return {
            "conversation_summary": "",
            "open_loops": "",
            "durable_facts": "",
            "sensitive_refs": "",
            "style_notes": "",
        }
    lowered = raw.lower()
    return {
        "conversation_summary": raw[:4000].strip(),
        "open_loops": (
            raw[:1500].strip()
            if any(token in lowered for token in ("unresolved", "open loop", "next step", "follow-up"))
            else ""
        ),
        "durable_facts": raw[:2500].strip(),
        "sensitive_refs": "",
        "style_notes": (
            raw[:500].strip()
            if any(token in lowered for token in ("tone", "style", "concise", "formal", "friendly", "warm"))
            else ""
        ),
    }


def _select_split_index(message_tokens: list[int], *, tokens_to_compact: int) -> int:
    if not message_tokens or len(message_tokens) <= 1 or tokens_to_compact <= 0:
        return 0
    consumed = 0
    split_idx = 0
    for idx, tok in enumerate(message_tokens):
        consumed += max(0, int(tok))
        split_idx = idx + 1
        if consumed >= tokens_to_compact and split_idx < len(message_tokens):
            break
    if split_idx >= len(message_tokens):
        split_idx = len(message_tokens) - 1
    return max(0, split_idx)


def split_text_chunks(text: str, *, approx_tokens_per_chunk: int = 25000) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    max_chars = max(12000, approx_tokens_per_chunk * 4)
    if len(raw) <= max_chars:
        return [raw]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0
    for para in raw.split("\n\n"):
        piece = para.strip()
        if not piece:
            continue
        piece_len = len(piece) + 2
        if current and current_chars + piece_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_chars = piece_len
        else:
            current.append(piece)
            current_chars += piece_len
    if current:
        chunks.append("\n\n".join(current))
    if not chunks:
        chunks = [raw[i : i + max_chars] for i in range(0, len(raw), max_chars)]
    return chunks


async def compress_rollup(runtime: Any, existing_rollup: str, additional_text: str) -> str:
    rollup_budget = _rollup_token_budget(runtime)
    running = _trim_text_to_token_budget(_sanitize_rollup_text(str(existing_rollup or "").strip()), rollup_budget)
    chunk_budget = min(25000, _compaction_source_budget(runtime))
    chunks = split_text_chunks(_sanitize_rollup_text(additional_text), approx_tokens_per_chunk=chunk_budget)
    if not chunks:
        return running
    existing_chars = max(4000, rollup_budget * 4)
    chunk_chars = max(20000, _compaction_source_budget(runtime) * 4)
    for chunk in chunks:
        response = await runtime._model.ainvoke(
            [
                SystemMessage(
                    content=(
                        "You compress long-running assistant conversations into durable context.\n"
                        "Return plain text only. Preserve:\n"
                        "- user preferences/directives\n"
                        "- active goals and constraints\n"
                        "- important decisions and why\n"
                        "- unresolved tasks / follow-ups\n"
                        "- key facts with dates, IDs, links, and paths\n"
                        "Be concise and structured with short headings."
                    )
                ),
                HumanMessage(
                    content=(
                        "Existing compressed context (may be empty):\n"
                        f"{running[:existing_chars]}\n\n"
                        "Older conversation segment to fold in:\n"
                        f"{str(chunk or '')[:chunk_chars]}"
                    )
                ),
            ]
        )
        running = _sanitize_rollup_text(_content_to_text(getattr(response, "content", "")).strip() or running)
        running = _trim_text_to_token_budget(running, rollup_budget)
    return _trim_text_to_token_budget(_sanitize_rollup_text(running), rollup_budget)


async def persist_rollup_memory(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
    rollup: str,
) -> None:
    cid = str(customer_id or "").strip()
    if not cid:
        return
    with suppress(Exception):
        await runtime._request_with_backoff(
            "POST",
            "/internal/memory/add",
            json_body={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Thread context rollup updated for {thread_id}: "
                            f"{_sanitize_rollup_text(str(rollup or ''))[:12000]}"
                        ),
                    }
                ],
                "user_id": cid,
                "metadata": {
                    "kind": "thread_context_rollup",
                    "thread_id": str(thread_id or ""),
                },
            },
            timeout=10.0,
            retries=1,
        )


async def maybe_compact_thread_context(
    runtime: Any,
    *,
    thread_id: str,
    customer_id: str,
) -> None:
    tid = str(thread_id or "").strip()
    if not tid:
        return
    if runtime._graph is None:
        return
    if runtime._checkpointer is None or not hasattr(runtime._checkpointer, "adelete_thread"):
        return

    config = {"configurable": {"thread_id": tid}, "recursion_limit": runtime.recursion_limit}
    short_term_high_budget = _short_term_high_token_budget(runtime)
    short_term_low_budget = _short_term_low_token_budget(runtime)
    source_budget = _compaction_source_budget(runtime)
    for _ in range(8):
        try:
            snapshot = await runtime._graph.aget_state(config=config)
            values = getattr(snapshot, "values", {}) or {}
            state_messages = values.get("messages", [])
            if not isinstance(state_messages, list) or not state_messages:
                return
            message_texts = [_message_to_text(m) for m in state_messages]
            message_tokens = [_approx_tokens(t) for t in message_texts]
            total_tokens = sum(message_tokens)
            # Hysteresis window: compact only at/above high watermark.
            if total_tokens < short_term_high_budget:
                return

            # Compact enough oldest context to move back near low watermark.
            overflow_tokens = total_tokens - short_term_low_budget
            target_tokens = min(source_budget, overflow_tokens)
            split_idx = _select_split_index(message_tokens, tokens_to_compact=target_tokens)
            if split_idx <= 0:
                return

            oldest_segment = "\n\n".join(message_texts[:split_idx]).strip()
            if not oldest_segment:
                return

            existing_rollup = runtime._load_thread_rollup(tid) or ""
            updated_rollup = await compress_rollup(runtime, existing_rollup, oldest_segment)
            if not updated_rollup:
                return

            runtime._save_thread_rollup(tid, updated_rollup)
            await persist_rollup_memory(
                runtime,
                customer_id=customer_id,
                thread_id=tid,
                rollup=updated_rollup,
            )

            remaining_messages = state_messages[split_idx:]
            await runtime._checkpointer.adelete_thread(tid)
            if remaining_messages:
                await runtime._graph.aupdate_state(
                    config=config,
                    values={"messages": remaining_messages},
                )
        except Exception:
            return
