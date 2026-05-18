"""Prompt-mode classification helpers."""

from __future__ import annotations

import re
from typing import Literal

PromptMode = Literal["literal_chat", "task_chat", "execution", "workflow_setup"]

_LITERAL_PATTERNS = [
    r"^\s*what does\b",
    r"^\s*what is\b",
    r"^\s*what's\b",
    r"^\s*meaning of\b",
    r"^\s*define\b",
    r"^\s*translate\b",
    r"^\s*how do you say\b",
]

_EXECUTION_HINTS = (
    "open ",
    "search ",
    "look up",
    "check ",
    "find ",
    "send ",
    "post ",
    "schedule ",
    "create ",
    "update ",
    "edit ",
    "fix ",
    "run ",
    "browse ",
)

def classify_prompt_mode(user_text: str, *, turn_mode: str) -> PromptMode:
    normalized_turn_mode = str(turn_mode or "").strip().lower()
    if normalized_turn_mode == "workflow_setup":
        return "workflow_setup"
    if normalized_turn_mode in {"routine_wake", "event_notification"}:
        return "execution"

    text = str(user_text or "").strip()
    lowered = text.lower()
    if not lowered:
        return "literal_chat"

    if len(lowered) <= 120 and any(re.search(pattern, lowered) for pattern in _LITERAL_PATTERNS):
        return "literal_chat"

    if (
        len(lowered) <= 160
        and "?" in lowered
        and not any(hint in lowered for hint in _EXECUTION_HINTS)
    ):
        return "literal_chat"

    if any(hint in lowered for hint in _EXECUTION_HINTS):
        return "execution"

    return "task_chat"
