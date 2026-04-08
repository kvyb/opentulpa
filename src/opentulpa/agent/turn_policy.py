"""Turn-mode policy helpers for runtime planning."""

from __future__ import annotations

from typing import Literal

from opentulpa.agent.lc_messages import SystemMessage

TurnMode = Literal["interactive", "routine_wake", "approval_recovery", "event_notification"]


def normalize_turn_mode(value: str | None) -> TurnMode:
    normalized = str(value or "").strip().lower()
    if normalized == "routine_wake":
        return "routine_wake"
    if normalized == "approval_recovery":
        return "approval_recovery"
    if normalized == "event_notification":
        return "event_notification"
    return "interactive"


def build_turn_mode_system_message(turn_mode: str | None) -> SystemMessage:
    normalized = normalize_turn_mode(turn_mode)
    if normalized == "routine_wake":
        return SystemMessage(
            content=(
                "Turn mode: routine_wake.\n"
                "This is a scheduled routine execution, not an interactive user turn.\n"
                "Execute autonomously using tools and skills as needed.\n"
                "Do not stop to ask clarifying questions unless the instruction is materially blocked or missing a required dependency.\n"
                "Focus on doing the work, then return a concise outcome summary."
            )
        )
    if normalized == "approval_recovery":
        return SystemMessage(
            content=(
                "Turn mode: approval_recovery.\n"
                "A previously approved action is being executed, repaired, or summarized.\n"
                "You may use tools needed to finish or repair the already-approved task.\n"
                "Treat this as continuation of the approved execution, not a fresh interactive approval request.\n"
                "Do not ask the user to repeat the same approval flow unless a genuinely unrelated new action is required."
            )
        )
    if normalized == "event_notification":
        return SystemMessage(
            content=(
                "Turn mode: event_notification.\n"
                "This is a background event/status notification, not a fresh user request.\n"
                "Prefer a concise status update over exploratory tool use.\n"
                "Do not create new routines or launch side-effecting plans unless the event explicitly requires it."
            )
        )
    return SystemMessage(
        content=(
            "Turn mode: interactive.\n"
            "This is a live user-guided turn.\n"
            "If the user intent is ambiguous about acting now vs drafting/planning, ask one concise clarifying question before taking side-effecting action."
        )
    )


def execution_origin_for_turn_mode(turn_mode: str | None, *, thread_id: str | None = None) -> str:
    normalized = normalize_turn_mode(turn_mode)
    if normalized == "routine_wake":
        return "scheduled"
    if normalized == "approval_recovery":
        return "scheduled"
    if normalized == "event_notification":
        return "interactive"
    safe_thread_id = str(thread_id or "").strip().lower()
    if safe_thread_id.startswith(("wake_", "wake-", "routine_", "routine-")):
        return "scheduled"
    return "interactive"
