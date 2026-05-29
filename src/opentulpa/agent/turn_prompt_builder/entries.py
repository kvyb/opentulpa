"""Prompt entry selection helpers."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.lc_messages import AnyMessage, SystemMessage
from opentulpa.agent.prompt_sections import (
    build_retrieved_context_message as _build_retrieved_context_message,
)
from opentulpa.agent.utils import approx_tokens as _approx_tokens
from opentulpa.agent.utils import content_to_text as _content_to_text


def append_retrieved_entry(
    entries: list[dict[str, str]],
    *,
    section: str,
    title: str,
    body: str,
) -> None:
    entry = make_retrieved_context_entry(section=section, title=title, body=body)
    if entry is not None:
        entries.append(entry)


def make_prompt_context_entry(*, section: str, content: str) -> dict[str, str] | None:
    safe_section = str(section or "").strip()
    safe_content = str(content or "").strip()
    if not safe_section or not safe_content:
        return None
    return {"section": safe_section, "content": safe_content}


def make_retrieved_context_entry(
    *,
    section: str,
    title: str,
    body: str,
) -> dict[str, str] | None:
    message = _build_retrieved_context_message(title=title, body=body)
    if message is None:
        return None
    return make_prompt_context_entry(
        section=section,
        content=_content_to_text(getattr(message, "content", "")).strip(),
    )


def normalize_prompt_context_entries(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = make_prompt_context_entry(
            section=str(item.get("section", "")).strip(),
            content=str(item.get("content", "")).strip(),
        )
        if entry is not None:
            normalized.append(entry)
    return normalized


def prompt_overhead_tokens(messages: list[AnyMessage]) -> int:
    return sum(_approx_tokens(_content_to_text(getattr(msg, "content", ""))) for msg in messages)


def select_optional_prompt_entries(
    entries: list[dict[str, str]],
    *,
    initial_used_tokens: int,
    optional_context_budget: int,
) -> tuple[list[tuple[str, SystemMessage]], int]:
    kept: list[tuple[str, SystemMessage]] = []
    used_tokens = max(0, int(initial_used_tokens))
    for entry in entries:
        content = str(entry.get("content", "")).strip()
        section = str(entry.get("section", "")).strip()
        if not content or not section:
            continue
        msg_tokens = _approx_tokens(content)
        if (used_tokens > 0 or kept) and used_tokens + msg_tokens > optional_context_budget:
            continue
        kept.append((section, SystemMessage(content=content)))
        used_tokens += msg_tokens
    return kept, used_tokens
