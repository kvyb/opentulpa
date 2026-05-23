"""Workflow setup control context for the agent prompt."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _truthy_config_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "required", "strict"}


def _draft_hash(draft: dict[str, Any]) -> str:
    payload = json.dumps(_safe_dict(draft), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _nonempty_list(value: Any) -> list[str]:
    return [str(item or "").strip() for item in _safe_list(value) if str(item or "").strip()]


def _sink_status(*, sink_type: str, sink_config: dict[str, Any]) -> tuple[str, list[str]]:
    if not sink_type:
        return "missing", ["sink_type"]
    if sink_type == "local_csv":
        if str(sink_config.get("file_path", "") or "").strip():
            return "ready", []
        return "missing_details", ["sink_config.file_path"]
    if sink_type == "google_sheets_composio":
        missing: list[str] = []
        static_arguments = _safe_dict(sink_config.get("static_arguments"))
        if not str(static_arguments.get("spreadsheetId", "") or "").strip():
            missing.append("sink_config.static_arguments.spreadsheetId")
        toolkit = str(sink_config.get("toolkit", "") or "").strip()
        if toolkit != "googlesheets":
            missing.append("sink_config.toolkit=googlesheets")
        if missing:
            return "missing_details", missing
        return "ready", []
    if sink_type == "generic_composio_write":
        missing = []
        if not str(sink_config.get("toolkit", "") or "").strip():
            missing.append("sink_config.toolkit")
        if not str(sink_config.get("operation_hint", "") or "").strip():
            missing.append("sink_config.operation_hint")
        if missing:
            return "missing_details", missing
        return "ready", []
    return "unknown_sink_type", ["sink_type"]


def _proposal_status(*, current_hash: str, proposed_hash: str) -> str:
    if proposed_hash and proposed_hash == current_hash:
        return "proposed_current"
    if proposed_hash:
        return "stale_proposal_draft_changed"
    return "not_proposed"


def _confirmation_status(*, current_hash: str, confirmed_hash: str) -> str:
    if confirmed_hash and confirmed_hash == current_hash:
        return "confirmed_current"
    if confirmed_hash:
        return "confirmed_stale_draft_changed"
    return "not_confirmed"


def build_workflow_setup_control_context(session: dict[str, Any] | None) -> str:
    """Render an authoritative workflow-setup control card for the model."""

    if not session:
        return (
            "WORKFLOW_SETUP_CONTROL_CARD\n"
            "Source: current setup database. Trust this card over stale summaries if they conflict.\n\n"
            "STATE:\n"
            "- session_status: none\n\n"
            "SUGGESTED_NEXT_ACTION:\n"
            "If the owner is creating or editing an intake workflow, call intake_workflow_setup_begin. "
            "Do not call business_knowledge_index until setup_begin returns a session. Otherwise answer normally.\n\n"
            "STOP_RULE:\n"
            "After the needed setup action or one focused blocker question, stop tool-calling and reply concisely."
        )

    status = str(session.get("status", "") or "").strip().lower() or "unknown"
    mode = str(session.get("mode", "") or "").strip().lower() or "unknown"
    draft = _safe_dict(session.get("draft_upsert"))
    scratchpad = _safe_dict(session.get("scratchpad"))
    current_hash = _draft_hash(draft)
    proposed_hash = str(session.get("last_proposed_draft_hash", "") or "").strip()
    confirmed_hash = str(session.get("confirmed_draft_hash", "") or "").strip()
    last_preflight = _safe_dict(scratchpad.get("last_preflight"))
    preflight_status = str(last_preflight.get("status", "") or "").strip() or "not_run"
    preflight_ok = bool(last_preflight.get("ok", False))
    follow_up_questions = _nonempty_list(last_preflight.get("follow_up_questions"))
    errors = _nonempty_list(last_preflight.get("errors"))
    warnings = _nonempty_list(last_preflight.get("warnings"))
    preflight_next_action = str(last_preflight.get("next_action", "") or "").strip()

    name = str(draft.get("name", "") or "").strip()
    channel = str(draft.get("channel", "") or "").strip()
    intent = str(draft.get("intent_description", "") or "").strip()
    source_config = _safe_dict(draft.get("source_config"))
    intent_match_required = _truthy_config_flag(source_config.get("intent_match_required"))
    required_fields = _nonempty_list(draft.get("required_fields"))
    knowledge_file_ids = _nonempty_list(draft.get("knowledge_file_ids"))
    source_file_ids = _nonempty_list(scratchpad.get("source_file_ids"))
    sink_type = str(draft.get("sink_type", "") or "").strip().lower()
    sink_config = _safe_dict(draft.get("sink_config"))
    sink_status, sink_missing = _sink_status(sink_type=sink_type, sink_config=sink_config)
    knowledge_preflight = _safe_dict(scratchpad.get("knowledge_last_preflight"))
    knowledge_index = _safe_dict(scratchpad.get("knowledge_last_index"))
    if bool(knowledge_preflight.get("ok", False)):
        knowledge_status = str(knowledge_preflight.get("status", "") or "").strip() or "ready"
    elif bool(knowledge_index.get("ok", False)):
        knowledge_status = "indexed_needs_preflight"
    elif knowledge_file_ids or source_file_ids:
        knowledge_status = "source_files_bound_needs_index_or_preflight"
    else:
        knowledge_status = "not_bound"

    missing_core: list[str] = []
    if not name:
        missing_core.append("name")
    if not channel:
        missing_core.append("channel")
    if not intent:
        missing_core.append("intent_description")
    if not required_fields:
        missing_core.append("required_fields")
    missing_core.extend(sink_missing)

    proposal = _proposal_status(current_hash=current_hash, proposed_hash=proposed_hash)
    confirmation = _confirmation_status(current_hash=current_hash, confirmed_hash=confirmed_hash)
    if status == "completed":
        draft_status = "completed"
    elif preflight_ok and preflight_status == "ready":
        draft_status = "preflight_ready"
    elif preflight_status != "not_run":
        draft_status = "needs_clarification"
    elif missing_core:
        draft_status = "incomplete"
    else:
        draft_status = "needs_preflight"

    if status == "completed":
        workflow_id = str(session.get("created_or_updated_workflow_id", "") or "").strip()
        suggested = (
            f"No setup tool is needed. Report that the workflow is active"
            f"{f' with workflow_id={workflow_id}' if workflow_id else ''}, then stop."
        )
    elif status == "paused":
        suggested = (
            "If the owner wants to continue, call intake_workflow_setup_begin with the current mode to resume. "
            "If they do not, answer the setup question without changing the draft."
        )
    elif confirmation == "confirmed_current":
        suggested = "Call intake_workflow_setup_commit, then report the created workflow id and stop."
    elif proposal == "proposed_current":
        suggested = (
            "If the latest owner message explicitly confirms the proposal, call "
            "intake_workflow_setup_finalize_confirmation, then report the created workflow id and stop. "
            "If the owner requests changes, call intake_workflow_setup_update, then preflight again."
        )
    elif preflight_ok and preflight_status == "ready":
        suggested = (
            "Call intake_workflow_setup_propose_current. Then send a concise proposal summary and ask for confirmation, then stop. "
            "If the latest owner message is already an explicit confirmation of a proposal visible in this conversation, "
            "call intake_workflow_setup_finalize_confirmation instead."
        )
    elif preflight_status != "not_run" and follow_up_questions:
        suggested = (
            "If the latest owner message answers the preflight blocker, call intake_workflow_setup_update and rerun preflight. "
            f"Otherwise ask this blocker only: {follow_up_questions[0]}"
        )
    elif missing_core:
        suggested = (
            "Call intake_workflow_setup_update for any new owner-provided facts in the latest message. "
            "If facts are still missing after that, ask one focused question for the missing critical input."
        )
    elif knowledge_status == "source_files_bound_needs_index_or_preflight":
        suggested = (
            "Prepare the bound source files with business_knowledge_index, verify representative facts with "
            "business_knowledge_query, then call intake_workflow_setup_propose_current."
        )
    else:
        suggested = "Call intake_workflow_setup_propose_current when the draft is complete; otherwise update the draft or ask the focused missing-field question."

    state_lines = [
        "WORKFLOW_SETUP_CONTROL_CARD",
        "Source: current setup database after any tools already run in this turn. Trust this card over stale summaries or old tool outputs if they conflict.",
        "Tool access: execute named intake_workflow_setup_* commands with tool_group_exec(group=\"intake\", command=\"...\", args_json={...}) when they are not directly bound.",
        "",
        "STATE:",
        f"- session_status: {status}",
        f"- mode: {mode}",
        f"- draft_status: {draft_status}",
        f"- has_name: {_yes_no(bool(name))}",
        f"- channel: {channel or 'missing'}",
        f"- has_intent_description: {_yes_no(bool(intent))}",
        f"- intent_match_required: {_yes_no(intent_match_required)}",
        f"- required_field_count: {len(required_fields)}",
        f"- sink_type: {sink_type or 'missing'}",
        f"- sink_status: {sink_status}",
        f"- knowledge_status: {knowledge_status}",
        f"- knowledge_file_count: {len(knowledge_file_ids or source_file_ids)}",
        f"- last_preflight_status: {preflight_status}",
        f"- last_preflight_ok: {_yes_no(preflight_ok)}",
        f"- last_preflight_next_action: {preflight_next_action or 'none'}",
        f"- proposal_status: {proposal}",
        f"- confirmation_status: {confirmation}",
        f"- missing_core_inputs: {', '.join(missing_core) if missing_core else 'none'}",
    ]
    if errors:
        state_lines.append(f"- latest_preflight_errors: {' | '.join(errors[:3])}")
    if warnings:
        state_lines.append(f"- latest_preflight_warnings: {' | '.join(warnings[:3])}")
    if follow_up_questions:
        state_lines.append(f"- latest_preflight_follow_up: {follow_up_questions[0]}")
    state_lines.extend(
        [
            "",
            "SUGGESTED_NEXT_ACTION:",
            suggested,
            "",
            "TOOL_CALL_DISCIPLINE:",
            "Use setup_get, file inspection, or business knowledge tools only when the control card says the needed state is missing or stale.",
            "After a ready preflight, do not re-query knowledge just to feel safer; mark the proposal or proceed with confirmation.",
            "After commit succeeds, do not keep setting up; report the workflow id and stop.",
            "",
            "STOP_RULE:",
            "After the suggested terminal step for this turn is done (proposal shown, commit reported, or blocker asked), stop tool-calling and reply concisely.",
        ]
    )
    return "\n".join(state_lines)
