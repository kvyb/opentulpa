"""Turn-mode policy helpers for runtime planning."""

from __future__ import annotations

from typing import Literal

from opentulpa.agent.lc_messages import SystemMessage

TurnMode = Literal["interactive", "workflow_setup", "routine_wake", "approval_recovery", "event_notification"]


def normalize_turn_mode(value: str | None) -> TurnMode:
    normalized = str(value or "").strip().lower()
    if normalized == "workflow_setup":
        return "workflow_setup"
    if normalized == "routine_wake":
        return "routine_wake"
    if normalized == "approval_recovery":
        return "approval_recovery"
    if normalized == "event_notification":
        return "event_notification"
    return "interactive"


def build_turn_mode_system_message(turn_mode: str | None) -> SystemMessage:
    normalized = normalize_turn_mode(turn_mode)
    if normalized == "workflow_setup":
        return SystemMessage(
            content=(
                "Turn mode: workflow_setup.\n"
                "You are collaborating on an intake workflow draft, not executing a normal chat task.\n"
                "Maintain the workflow setup draft and scratchpad through the dedicated setup tools.\n"
                "A workflow setup turn that changes, validates, proposes, confirms, or saves the workflow must call the dedicated intake_workflow_setup_* tools before replying.\n"
                "When the owner provides new workflow facts, sink details, source files, field requirements, or behavior rules, call intake_workflow_setup_update to persist them in the draft; do not merely acknowledge them in prose.\n"
                "When the owner provides Google Sheets details, store them under sink_type=google_sheets_composio and sink_config.static_arguments, including spreadsheetId and sheetName when provided.\n"
                "After updating a draft that now appears complete, call intake_workflow_setup_preflight. If ready, call intake_workflow_setup_mark_proposed before showing the proposal.\n"
                "When the owner explicitly confirms a shown proposal, call intake_workflow_setup_confirm_current and intake_workflow_setup_commit before saying it is saved. If a proposal was shown in the conversation but last_proposed_draft_hash is missing, call preflight, mark_proposed, confirm_current, and commit in order.\n"
                "If any setup tool returns an error or focused follow-up, report that specific blocker instead of repeating an older proposal.\n"
                "Ask one high-value setup question at a time.\n"
                "If uploaded files are part of the workflow, track original source_file_ids and prepare them with business_knowledge_index.\n"
                "Before proposing a file-grounded workflow, query the business knowledge for representative facts and run setup preflight so unsupported or weak files are caught.\n"
                "If file inspection, knowledge prep, or workflow compilation will take multiple tool calls, call send_owner_update as the first tool call before continuing.\n"
                "For telegram_business_dm workflows, do not ask for polling, scanning, or schedule intervals; inbound Telegram Business messages trigger the workflow directly.\n"
                "Synthesize a concise intent_description from the user's stated goal instead of asking for it as a form field when the goal is already clear.\n"
                "Keep the setup schema machine-readable: required_fields are stable ASCII snake_case ids, while localized labels, owner wording, and extraction hints belong in field_guidance, assistant_instructions, or sink field mappings.\n"
                "field_guidance keys must match required_fields ids; do not create a separate localized field id when a stable id can represent the same meaning.\n"
                "Once the draft has channel, purpose, required fields, sink, and behavior rules, propose it with explicit assumptions and wait for confirmation instead of asking optional questions.\n"
                "Do not persist the workflow until the user has seen a proposal and explicitly confirmed it.\n"
                "Do not dump the full draft unless the user asks for it.\n"
                "If the user wants to stop setup for now, pause or cancel the setup session and hand back to normal chat.\n"
                "If editing, modify the draft loaded from the existing workflow; do not treat the live workflow as already changed."
            )
        )
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
            "For long-running work with multiple tool calls, call send_owner_update as the first tool call before continuing.\n"
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
