"""Small reply-normalization helpers for intake workflows."""

from __future__ import annotations

from typing import Any


def looks_like_cyrillic(*values: Any) -> bool:
    text = " ".join(str(value or "") for value in values)
    return any("\u0400" <= char <= "\u04ff" for char in text)


def build_missing_field_follow_up_reply(
    *,
    missing_field: str,
    workflow: dict[str, Any],
    conversation_summary: dict[str, Any],
) -> str:
    safe_field = str(missing_field or "").strip()
    if not safe_field:
        return ""
    label = safe_field.replace("_", " ").strip()
    context_text = " ".join(
        [
            str(workflow.get("assistant_instructions", "") or ""),
            str(workflow.get("intent_description", "") or ""),
            str(conversation_summary.get("latest_inbound_message_text_preview", "") or ""),
        ]
    )
    if looks_like_cyrillic(context_text):
        return f"Какое значение указать для поля «{label}»?"
    return f"What {label} should I use for the booking?"
