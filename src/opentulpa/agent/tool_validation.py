"""Tool-call validation helpers for the runtime graph."""

from __future__ import annotations

import logging
import shlex
from datetime import datetime
from typing import Any

from opentulpa.agent.lc_messages import ToolMessage
from opentulpa.agent.turn_policy import normalize_turn_mode as _normalize_turn_mode
from opentulpa.agent.utils import (
    content_to_text as _content_to_text,
)
from opentulpa.agent.utils import (
    extract_relative_delay_minutes as _extract_relative_delay_minutes,
)
from opentulpa.agent.utils import (
    is_cron_like_schedule as _is_cron_like_schedule,
)
from opentulpa.agent.utils import (
    looks_like_shell_command as _looks_like_shell_command,
)

logger = logging.getLogger(__name__)

_WORKING_DIR_PREFIXES: dict[str, str] = {
    "tulpa_stuff": "tulpa_stuff",
    "integrations": "src/opentulpa/integrations",
    "interfaces": "src/opentulpa/interfaces",
    "tools": "src/opentulpa/tools",
    "skills": "src/opentulpa/skills",
    "opentulpa": "src/opentulpa",
}


def _is_iso_datetime_schedule(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text)
    except Exception:
        return False
    return True


def _has_redundant_working_dir_prefix(command: str, working_dir: str) -> bool:
    prefix = _WORKING_DIR_PREFIXES.get(str(working_dir or "").strip())
    text = str(command or "").strip()
    if not prefix or not text:
        return False
    try:
        parts = shlex.split(text)
    except Exception:
        return False
    if len(parts) <= 1:
        return False
    markers = (f"{prefix}/", f"./{prefix}/")
    for token in parts[1:]:
        raw = str(token)
        candidates = [raw]
        if raw.startswith("--") and "=" in raw:
            _, value = raw.split("=", 1)
            candidates.append(value)
        for candidate in candidates:
            if any(candidate.startswith(marker) for marker in markers):
                return True
    return False


def _has_duplicate_allowed_root_prefix(path: str) -> str | None:
    text = str(path or "").strip()
    if not text:
        return None
    for prefix in _WORKING_DIR_PREFIXES.values():
        normalized = str(prefix or "").strip("/")
        if normalized and text.startswith(f"{normalized}/{normalized}/"):
            return normalized
    return None


def _routine_create_turn_mode_error(*, turn_mode: str) -> str | None:
    normalized_turn_mode = _normalize_turn_mode(turn_mode)
    if normalized_turn_mode == "routine_wake":
        return None
    if normalized_turn_mode == "event_notification":
        return (
            "TURN_MODE_MISMATCH: this is a background event-notification turn, not a fresh "
            "user scheduling request. Do not call routine_create here unless the event "
            "explicitly instructs schedule management."
        )
    return None


def _validate_model_tool_call(
    *,
    call_name: str,
    args: Any,
    latest_user_text: str,
    turn_mode: str,
    required_args: dict[str, tuple[str, ...]],
    forbidden_tool_args: dict[str, set[str]],
) -> str | None:
    if not isinstance(args, dict):
        return f"TOOL_VALIDATION_ERROR: arguments for {call_name} must be an object"

    blocked_args = sorted(arg for arg in args if arg in forbidden_tool_args.get(call_name, set()))
    if blocked_args:
        return (
            f"TOOL_VALIDATION_ERROR: {call_name} must not include argument(s): "
            f"{', '.join(blocked_args)}. These are runtime-managed."
        )

    missing = [arg for arg in required_args.get(call_name, ()) if not args.get(arg)]
    if missing:
        if call_name == "routine_create" and "implementation_command" in missing:
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create needs "
                "implementation_command (a concrete shell/script command like "
                "`python3 scripts/digest.py`) describing what will run "
                "on each scheduled execution (the command runs with working_dir=tulpa_stuff "
                "by default, so no tulpa_stuff/ prefix needed). Repair the call and retry."
            )
        return (
            f"TOOL_VALIDATION_ERROR: missing required argument(s) for "
            f"{call_name}: {', '.join(missing)}"
        )

    if call_name == "tulpa_run_terminal":
        command = str(args.get("command", "")).strip()
        if not _looks_like_shell_command(command):
            return (
                "TOOL_VALIDATION_ERROR: command must be a concrete shell command "
                "with executable + args."
            )
        working_dir = str(args.get("working_dir", "tulpa_stuff") or "").strip() or "tulpa_stuff"
        if _has_redundant_working_dir_prefix(command, working_dir):
            return (
                "TOOL_VALIDATION_ERROR: command includes a redundant working-dir path prefix. "
                "When working_dir is set, use paths relative to that directory "
                "(example: use `python3 tg_login.py`, not `python3 tulpa_stuff/tg_login.py`)."
            )

    if call_name == "send_owner_update" and _normalize_turn_mode(turn_mode) in {
        "routine_wake",
        "event_notification",
    }:
        normalized_turn_mode = _normalize_turn_mode(turn_mode)
        return (
            "TOOL_VALIDATION_ERROR: send_owner_update is only for live owner/support turns. "
            f"For {normalized_turn_mode}, put the user-visible notification, proposal, or blocker "
            "summary in the final assistant response so the owning orchestrator can deliver it."
        )

    if call_name == "browser_use_owner_input_submit" and _normalize_turn_mode(turn_mode) not in {
        "interactive",
        "workflow_setup",
    }:
        normalized_turn_mode = _normalize_turn_mode(turn_mode)
        return (
            "TOOL_VALIDATION_ERROR: browser_use_owner_input_submit is only for live "
            "owner/support chat turns. For "
            f"{normalized_turn_mode}, do not submit owner authentication input."
        )

    if call_name in {"tulpa_read_file", "tulpa_write_file", "tulpa_validate_file", "tulpa_file_send"}:
        path_arg = str(args.get("path", "")).strip()
        duplicate_prefix = _has_duplicate_allowed_root_prefix(path_arg)
        if duplicate_prefix:
            return (
                "TOOL_VALIDATION_ERROR: path includes a duplicated allowed-root prefix. "
                f"Use `{duplicate_prefix}/...`, not `{duplicate_prefix}/{duplicate_prefix}/...`."
            )

    if call_name == "routine_create":
        schedule = str(args.get("schedule", "")).strip()
        implementation_command = str(args.get("implementation_command", "")).strip()
        turn_mode_error = _routine_create_turn_mode_error(turn_mode=turn_mode)
        if turn_mode_error:
            return turn_mode_error
        if not (_is_cron_like_schedule(schedule) or _is_iso_datetime_schedule(schedule)):
            return (
                "TOOL_VALIDATION_ERROR: routine_create schedule must be either cron "
                "(five-part expression) or local ISO datetime."
            )
        if not implementation_command:
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_REQUIRED: routine_create must include "
                "a non-empty implementation_command (shell/script command) so scheduled "
                "runs execute a concrete implementation."
            )
        if not _looks_like_shell_command(implementation_command):
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command must "
                "be a concrete shell command (executable + args), not natural language."
            )
        if _has_redundant_working_dir_prefix(implementation_command, "tulpa_stuff"):
            return (
                "ROUTINE_IMPLEMENTATION_COMMAND_INVALID: implementation_command should be relative "
                "to working_dir=tulpa_stuff (example: `python3 tg_login.py`, "
                "not `python3 tulpa_stuff/tg_login.py`)."
            )
        delay_minutes = _extract_relative_delay_minutes(latest_user_text)
        if delay_minutes is not None and _is_cron_like_schedule(schedule):
            return (
                "TOOL_VALIDATION_ERROR: for one-time relative reminders, "
                "use a local ISO datetime schedule (not cron)."
            )

    return None


async def _routine_create_intent_validation_error(
    runtime: Any,
    *,
    args: Any,
    latest_user_text: str,
    prior_assistant_text: str,
    turn_mode: str,
) -> str | None:
    """Use the runtime classifier to decide whether routine_create is user-authorized."""
    turn_mode_error = _routine_create_turn_mode_error(turn_mode=turn_mode)
    if turn_mode_error:
        return turn_mode_error
    if _normalize_turn_mode(turn_mode) == "routine_wake":
        return None

    classifier = getattr(runtime, "classify_routine_create_intent", None)
    if not callable(classifier):
        logger.warning("routine_create intent classifier unavailable; allowing structural validation result")
        return None
    try:
        decision = await classifier(
            latest_user_text=latest_user_text,
            prior_assistant_text=prior_assistant_text,
            routine_args=args if isinstance(args, dict) else {},
            turn_mode=_normalize_turn_mode(turn_mode),
        )
    except Exception as exc:
        logger.warning("routine_create intent classifier failed; allowing structural validation result: %s", exc)
        return None

    if not isinstance(decision, dict):
        return None
    if not bool(decision.get("ok", True)):
        logger.warning(
            "routine_create intent classifier returned non-ok; allowing structural validation result: %s",
            str(decision.get("error", "unknown"))[:200],
        )
        return None
    if bool(decision.get("allow_create", False)):
        return None

    reason = str(decision.get("reason", "")).strip()[:300] or "classifier did not find user authorization"
    return (
        "ACTION_CLARIFICATION_REQUIRED: routine_create was not clearly authorized by the "
        f"current conversation. Ask one concise clarifying question. Reason={reason}"
    )


def _summarize_tool_validation_errors(messages: list[ToolMessage]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for message in messages:
        text = _content_to_text(getattr(message, "content", "")).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " | ".join(parts[:3])


def _build_tool_validation_repair_message(messages: list[ToolMessage]) -> str:
    summary = _summarize_tool_validation_errors(messages)
    if not summary:
        return (
            "VALIDATION_REPAIR_REQUIRED: Your previous tool call was blocked. Do not claim success. "
            "Repair the tool call or clearly state that the action was not completed yet."
        )
    is_routine_create_error = "routine_create" in summary or "ROUTINE_" in summary
    needs_clarification = any(
        marker in summary
        for marker in ("ACTION_CLARIFICATION_REQUIRED", "CHAT_MODE_LOCKED", "TURN_MODE_MISMATCH")
    )
    if is_routine_create_error and needs_clarification:
        return (
            "VALIDATION_REPAIR_REQUIRED: The scheduled action was not created. Do not say it was scheduled. "
            "Ask one concise clarifying question or continue in chat if automation is not explicit. "
            f"Reason={summary}"
        )
    if needs_clarification:
        return (
            "VALIDATION_REPAIR_REQUIRED: Your previous tool call was blocked. Do not claim success. "
            "Ask one concise clarifying question or continue in chat if the requested action is not explicit. "
            f"Reason={summary}"
        )
    if is_routine_create_error:
        return (
            "VALIDATION_REPAIR_REQUIRED: The scheduled action was not created yet. Do not claim success. "
            "Repair the tool call arguments and retry only if you can satisfy the validation error exactly. "
            f"Reason={summary}"
        )
    return (
        "VALIDATION_REPAIR_REQUIRED: The requested tool action was not completed yet. Do not claim success. "
        "Repair the tool call arguments and retry only if you can satisfy the validation error exactly. "
        f"Reason={summary}"
    )
