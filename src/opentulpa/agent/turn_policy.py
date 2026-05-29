"""Turn-mode policy helpers for runtime planning."""

from __future__ import annotations

from typing import Literal

from opentulpa.agent.lc_messages import SystemMessage

TurnMode = Literal["interactive", "workflow_setup", "routine_wake", "event_notification"]


def normalize_turn_mode(value: str | None) -> TurnMode:
    normalized = str(value or "").strip().lower()
    if normalized == "workflow_setup":
        return "workflow_setup"
    if normalized == "routine_wake":
        return "routine_wake"
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
                "Maintain the workflow setup draft and scratchpad through the intake tool group. Execute known setup commands directly with tool_group_exec(group=\"intake\", command=\"...\", args_json={...}); do not spend tool calls describing intake commands already named here.\n"
                "A workflow setup turn that changes, validates, proposes, confirms, or saves the workflow must execute the relevant intake_workflow_setup_* command before replying.\n"
                "When the owner provides new workflow facts, sink details, source files, field requirements, or behavior rules, execute intake_workflow_setup_update to persist them in the draft; do not merely acknowledge them in prose.\n"
                "When the owner provides Google Sheets details, store them under sink_type=google_sheets_composio and sink_config.static_arguments, including spreadsheetId and sheetName when provided.\n"
                "When the owner provides a local CSV path, store it under sink_type=local_csv and sink_config.file_path; do not ask for more sink details when the path is already present.\n"
                "After updating a draft that now appears complete, call intake_workflow_setup_propose_current before showing the proposal.\n"
                "When the owner explicitly confirms a shown proposal, call intake_workflow_setup_finalize_confirmation before saying it is saved. If the same message adds small final behavior rules, pass them in that finalize tool call instead of doing a separate update/preflight loop.\n"
                "If any setup tool returns an error or focused follow-up, report that specific blocker instead of repeating an older proposal.\n"
                "Ask one high-value setup question at a time.\n"
                "If uploaded files are part of the workflow, track original source_file_ids and prepare them with business_knowledge_index.\n"
                "Before proposing a file-grounded workflow, query the business knowledge for representative facts and run setup preflight so unsupported or weak files are caught.\n"
                "If file inspection, knowledge prep, or workflow compilation will take multiple tool calls, attach one concise visible progress sentence to the tool-call step or call send_owner_update as the first tool call before continuing.\n"
                "For telegram_business_dm workflows, do not ask for polling, scanning, or schedule intervals; inbound Telegram Business messages trigger the workflow directly.\n"
                "Synthesize a concise intent_description from the user's stated goal instead of asking for it as a form field when the goal is already clear.\n"
                "Do not add an intent pre-filter by default; set source_config.intent_match_required=true only if the owner explicitly asks to handle only messages matching the stated intent.\n"
                "Keep the setup schema machine-readable: required_fields are stable ASCII snake_case ids, while localized labels, owner wording, and extraction hints belong in field_guidance, assistant_instructions, or sink field mappings.\n"
                "Store compact owner-stated business facts such as prices, service menu highlights, hours, discounts, locations, and policies in draft.business_facts so future intake turns can rely on them without bound files. Do not paste uploaded files, spreadsheets, large tables, or extracted document text into business_facts; bind files through knowledge_file_ids instead.\n"
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
                "Return the user-visible routine notification or blocker summary as the final answer.\n"
                "Focus on doing the work, then return a concise outcome summary."
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
            "For long-running work with multiple tool calls, attach one concise visible progress sentence to the tool-call step or call send_owner_update as the first tool call before continuing.\n"
            "For longer-horizon research, discovery, comparison, analysis, report/list creation, lead/prospect finding, or other complex multi-step work, use turn_plan as a private current-turn checklist; make the plan realistic for this turn's runtime with a clear goal and stop condition, update it as steps complete, and return a useful final or partial answer before reply timeout.\n"
            "Apply retrieved user preferences, directive facts, and style facts to the current reply unless the latest user message overrides them.\n"
            "If the user gives a durable preference for normal chat style, such as asking OpenTulpa to write naturally or stop making Telegram answers look like Markdown documents, store a concise preference with tool_group_exec(group=\"memory\", command=\"memory_add\", args_json={\"summary\": \"User prefers ...\"}) before or while following it.\n"
            "For natural Telegram chat, avoid headings, horizontal rules, bold/italic marker style, and report-like templates unless the user explicitly asks for a structured document. Lists are fine when they fit the answer naturally.\n"
            "If the user intent is ambiguous about acting now vs drafting/planning, ask one concise clarifying question before taking side-effecting action."
        )
    )


def execution_origin_for_turn_mode(turn_mode: str | None, *, thread_id: str | None = None) -> str:
    normalized = normalize_turn_mode(turn_mode)
    if normalized == "routine_wake":
        return "scheduled"
    if normalized == "event_notification":
        return "interactive"
    safe_thread_id = str(thread_id or "").strip().lower()
    if safe_thread_id.startswith(("wake_", "wake-", "routine_", "routine-")):
        return "scheduled"
    return "interactive"
