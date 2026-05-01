"""Shared builder for durable intake workflow skills."""

from __future__ import annotations

import json
from typing import Any

from opentulpa.skills.service import build_skill_markdown


def workflow_skill_name(workflow_id: str) -> str:
    return f"intake-workflow-{str(workflow_id or '').strip()}"


def _safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        out.append(text)
    return out


def _truthy_config_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "on", "required", "strict"}


def _clean_mapping(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for raw_key, raw_field in value.items():
        key = str(raw_key or "").strip()
        field = str(raw_field or "").strip()
        if key and field:
            out[key] = field
    return out


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def channel_label(channel: str) -> str:
    safe_channel = str(channel or "").strip().lower()
    if safe_channel == "telegram_business_dm":
        return "Telegram Business DMs"
    return "Instagram DMs"


def reply_channel_label(channel: str) -> str:
    safe_channel = str(channel or "").strip().lower()
    if safe_channel == "telegram_business_dm":
        return "Telegram Business DM"
    return "Instagram DM"


def workflow_skill_sink_summary(
    *,
    sink_type: str,
    sink_config: dict[str, Any],
) -> str:
    safe_sink_type = str(sink_type or "").strip()
    safe_sink_config = _safe_dict(sink_config)
    if safe_sink_type == "local_csv":
        file_path = str(safe_sink_config.get("file_path", "") or "").strip()
        if file_path:
            return f"Write completed bookings to local CSV at `{file_path}`."
        return "Write completed bookings to the configured local CSV file."
    toolkit = str(safe_sink_config.get("toolkit", "") or "").strip()
    operation_hint = str(safe_sink_config.get("operation_hint", "") or "").strip()
    field_mapping = _clean_mapping(safe_sink_config.get("field_mapping"))
    static_arguments = _safe_dict(safe_sink_config.get("static_arguments"))
    parts: list[str] = []
    if toolkit:
        parts.append(f"toolkit={toolkit}")
    if operation_hint:
        parts.append(f"operation={operation_hint}")
    if field_mapping:
        parts.append(
            "mapped_fields=" + ", ".join(
                [str(key or "").strip() for key in field_mapping if str(key or "").strip()]
            )
        )
    if static_arguments:
        parts.append(
            "static_argument_keys=" + ", ".join(
                [str(key or "").strip() for key in static_arguments if str(key or "").strip()]
            )
        )
    detail = "; ".join(parts)
    if detail:
        return f"Write completed bookings through the configured sink ({detail})."
    return "Write completed bookings through the configured sink."


def build_intake_workflow_skill(
    workflow: dict[str, Any],
) -> dict[str, Any]:
    safe_workflow = workflow if isinstance(workflow, dict) else {}
    workflow_id = str(safe_workflow.get("workflow_id", "") or "").strip()
    name = workflow_skill_name(workflow_id)
    channel = str(safe_workflow.get("channel", "") or "").strip()
    channel_text = channel_label(channel)
    reply_channel_text = reply_channel_label(channel)
    description = (
        f"Operate the {safe_workflow['name']} {channel_text} intake workflow for this user."
    )
    knowledge_file_ids = _unique_string_list(safe_workflow.get("knowledge_file_ids"))
    field_guidance = _safe_dict(safe_workflow.get("field_guidance"))
    business_facts = _safe_dict(safe_workflow.get("business_facts"))
    source_config = _safe_dict(safe_workflow.get("source_config"))
    intent_match_required = _truthy_config_flag(source_config.get("intent_match_required"))
    sink_config = _safe_dict(safe_workflow.get("sink_config"))
    required_fields = [
        str(item or "").strip()
        for item in list(safe_workflow.get("required_fields") or [])
        if str(item or "").strip()
    ]
    sink_summary = workflow_skill_sink_summary(
        sink_type=str(safe_workflow.get("sink_type", "") or "").strip(),
        sink_config=sink_config,
    )
    source_lines = [
        f"- Channel: {channel_text}",
        f"- Provider: {str(safe_workflow.get('provider', '') or '').strip()}",
    ]
    if source_config:
        for key, value in source_config.items():
            safe_key = str(key or "").strip()
            safe_value = str(value or "").strip()
            if safe_key and safe_value:
                source_lines.append(f"- {safe_key}: {safe_value}")
    edit_rule_lines = [
        "- If the user later changes how this workflow should behave, update this durable workflow rather than creating a near-duplicate.",
    ]
    if channel == "telegram_business_dm":
        edit_rule_lines = [
            "- For Telegram Business, this workflow is the single durable intake policy for the connected business account.",
            "- Telegram Business workflows cannot be edited in place.",
            "- If the user wants to change this Telegram Business workflow, first fetch the current workflow for context, then delete it, then create a replacement workflow.",
            "- When recreating it, the backend can reuse the single connected Telegram Business account automatically; only specify a different business_connection_id when the user explicitly wants another connected business account.",
        ]
    if intent_match_required:
        matching_rule = (
            f"- Strictly match only conversations that fit this intent: {safe_workflow['intent_description']}\n"
            "- If the request is not actually about this workflow, ignore it rather than forcing a match.\n\n"
        )
    else:
        matching_rule = (
            "- Do not use the workflow intent as a front-door filter; ordinary openers and ambiguous early-stage messages from this source are part of the workflow.\n"
            "- Reply usefully to move the customer toward the workflow before deciding to ignore.\n\n"
        )

    instructions = (
        "## Purpose\n"
        f"Support the durable intake workflow `{safe_workflow['name']}`.\n\n"
        "## Workflow Goal\n"
        f"- Primary business intent: {safe_workflow['intent_description']}\n"
        f"- Success means moving the customer toward a complete booking in {reply_channel_text}, "
        "capturing the required fields accurately, and saving only when the request is ready.\n\n"
        "## Operating Context\n"
        + "\n".join(source_lines)
        + "\n"
        + f"- Sink target: {sink_summary}\n\n"
        "## Matching Rule\n"
        + matching_rule
        + "## Required Fields\n"
        f"- Collect these fields before save: {', '.join(required_fields)}\n\n"
        "## Execution Strategy\n"
        "- Understand whether the customer is starting a new request, continuing an active booking, or editing a recent completed booking.\n"
        "- Ask only the minimum high-leverage follow-up question needed to unblock the next step.\n"
        "- Use durable workflow instructions, field guidance, and bound knowledge files before improvising.\n"
        "- Save only when the required fields are sufficiently clear and internally consistent.\n"
        "## Behavioral Rules\n"
        f"- Ask concise follow-up questions in the {reply_channel_text} when fields are missing.\n"
        "- When all required fields are present, save through the configured sink.\n"
        "- Treat the same DM thread as one active booking until completion.\n"
        "- If the last completed booking is still inside the edit window, follow-up changes may edit it.\n"
        "- Otherwise, a clearly new request should create a new booking.\n"
        "- Telegram notifications should stay concise and only summarize booking success or failures.\n"
    )
    if field_guidance:
        guidance_lines = []
        for key, value in field_guidance.items():
            safe_key = str(key or "").strip()
            safe_value = str(value or "").strip()
            if safe_key and safe_value:
                guidance_lines.append(f"- {safe_key}: {safe_value}")
        if guidance_lines:
            instructions += (
                "\n## Field Guidance\n"
                + "\n".join(guidance_lines)
                + "\n"
            )
    if business_facts:
        instructions += (
            "\n## Owner-Provided Business Facts\n"
            "- These are compact facts explicitly provided by the owner during setup.\n"
            "- Treat them as authoritative workflow configuration unless bound knowledge answers contradict them.\n"
            f"- Facts JSON: {_json_dumps(business_facts)}\n"
        )
    assistant_instructions = str(safe_workflow.get("assistant_instructions", "") or "").strip()
    if assistant_instructions:
        instructions += (
            "\n## Durable Business Rules\n"
            f"{assistant_instructions}\n"
        )
    else:
        instructions += (
            "\n## Durable Business Rules\n"
            "- Follow the stored workflow goal, field requirements, and sink behavior even when no extra business rules were provided.\n"
        )
    if knowledge_file_ids:
        instructions += (
            "\n## Knowledge Files\n"
            "- Use workflow-bound business knowledge answers before answering source-specific customer questions.\n"
            f"- Bound file ids: {', '.join(knowledge_file_ids)}\n"
        )
    else:
        instructions += (
            "\n## Knowledge Files\n"
            "- No workflow-bound knowledge files are required. Rely on the stored workflow instructions and current conversation.\n"
        )
    instructions += (
        "\n## Save Behavior\n"
        f"- {sink_summary}\n"
        "- Preserve field meaning when mapping to the sink; do not invent values just to complete the write.\n"
        "- If the customer is still missing required details, keep the booking active instead of forcing a save.\n"
        "\n## Safety\n"
        "- Do not promise unavailable options unless the workflow or external tools confirm them.\n"
        "- Do not create duplicate bookings for the same active request.\n"
        "- When uncertain, ask a concise clarifying question rather than guessing.\n"
        "- The user's business goals and instructions in this workflow take priority over generic style defaults.\n"
        "\n## Edit Rule\n"
        + "\n".join(edit_rule_lines)
        + "\n"
    )
    skill_markdown = build_skill_markdown(
        name=name,
        description=description,
        instructions=instructions,
    )
    supporting_files = {
        "workflow.json": _json_dumps(
            {
                "workflow_id": workflow_id,
                "name": safe_workflow["name"],
                "channel": safe_workflow["channel"],
                "provider": safe_workflow["provider"],
                "source_config": source_config,
                "intent_description": safe_workflow["intent_description"],
                "required_fields": safe_workflow["required_fields"],
                "field_guidance": field_guidance,
                "assistant_instructions": safe_workflow.get("assistant_instructions", ""),
                "business_facts": business_facts,
                "knowledge_file_ids": knowledge_file_ids,
                "sink_type": safe_workflow["sink_type"],
                "sink_config": safe_workflow.get("sink_config", {}),
            }
        )
        + "\n"
    }
    return {
        "name": name,
        "description": description,
        "skill_markdown": skill_markdown,
        "supporting_files": supporting_files,
    }
