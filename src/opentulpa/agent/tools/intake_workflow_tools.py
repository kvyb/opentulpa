"""Intake workflow CRUD and run tool registration."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id
from opentulpa.agent.tools.internal_http import InternalToolHTTPClient


def _unique_string_list(values: list[str] | None) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
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


def _normalize_optional_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null"}:
        return ""
    return text


_INTAKE_ALLOWED_SINK_TYPES = {"local_csv", "google_sheets_composio", "generic_composio_write"}


def _validate_intake_sink_request(*, sink_type: str, sink_config: dict[str, Any]) -> str | None:
    safe_sink_type = str(sink_type or "").strip().lower()
    safe_config = sink_config if isinstance(sink_config, dict) else {}
    if safe_sink_type not in _INTAKE_ALLOWED_SINK_TYPES:
        if safe_sink_type == "google_sheets":
            return (
                "sink_type=google_sheets is not supported here; use google_sheets_composio and "
                "provide toolkit/field_mapping/static_arguments instead"
            )
        return (
            "sink_type must be one of local_csv|google_sheets_composio|generic_composio_write"
        )
    if safe_sink_type == "local_csv":
        return None
    toolkit = str(safe_config.get("toolkit", "") or "").strip()
    legacy_tool_slug = str(safe_config.get("tool_slug", "") or "").strip()
    if safe_sink_type == "generic_composio_write" and not toolkit and not legacy_tool_slug:
        return "composio sink_config.toolkit is required for generic_composio_write"
    field_mapping = safe_config.get("field_mapping")
    if not isinstance(field_mapping, dict) or not field_mapping:
        return (
            "composio sink_config.field_mapping is required; map sink argument names to workflow fields "
            "before calling intake_workflow_upsert"
        )
    operation_hint = str(safe_config.get("operation_hint", "") or "").strip()
    if safe_sink_type == "generic_composio_write" and not operation_hint and not legacy_tool_slug:
        return (
            "generic_composio_write requires sink_config.operation_hint so the runtime can choose the right tool"
        )
    return None


def _active_thread_id(runtime: Any, explicit_thread_id: str | None) -> str:
    safe_thread = str(explicit_thread_id or "").strip()
    if safe_thread:
        return safe_thread
    getter = getattr(runtime, "get_active_thread_id", None)
    if callable(getter):
        safe_thread = str(getter() or "").strip()
    if safe_thread:
        return safe_thread
    return str(getattr(runtime, "_active_thread_id", "") or "").strip()


def register_intake_workflow_tools(runtime: Any) -> dict[str, Any]:
    http = InternalToolHTTPClient(runtime)

    @tool
    async def intake_workflow_upsert(
        name: str,
        intent_description: str,
        required_fields: list[str],
        sink_type: str,
        sink_config: dict[str, Any],
        schedule: str = "*/2 * * * *",
        channel: str = "instagram_dm",
        provider: str = "composio",
        source_config: dict[str, Any] | None | str = None,
        field_guidance: dict[str, Any] | None | str = None,
        assistant_instructions: str = "",
        business_facts: dict[str, Any] | None | str = None,
        knowledge_file_ids: list[str] | None = None,
        notify_user: bool = True,
        enabled: bool = True,
        reply_mode: str = "",
        workflow_id: str | None = "",
        thread_id: str = "",
        execution_origin: str | None = None,
        preapproved: bool = False,
        guard_context: dict[str, Any] | None = None,
    ) -> Any:
        """Create or update an intake workflow.

        Use this when the user wants OpenTulpa to monitor inbound messages, decide whether
        they match a business workflow, ask follow-up questions, and save the result.

        Important shaping rules:
        - Use direct intake_workflow_upsert when the owner explicitly asks to create or update an intake workflow
          and has provided the required workflow fields.
        - Prefer intake_workflow_list and intake_workflow_get before editing an existing workflow.
        - In setup mode, call intake_workflow_upsert only when the draft already contains the exact workflow fields to save.
        - In workflow setup mode, call intake_workflow_setup_propose_current before showing the final proposal.
        - For a brand-new workflow, omit workflow_id or pass an empty string.
        - For updates, pass the existing workflow_id.
        - If the user is refining or editing an existing workflow, prefer intake_workflow_list and
          intake_workflow_get first so you have the full current workflow before changing anything.
        - Telegram Business workflows cannot be edited in place.
        - If the user wants to change an existing Telegram Business workflow, do this sequence:
          1. intake_workflow_list or intake_workflow_get to fetch the current workflow for context
          2. intake_workflow_delete for that workflow_id
          3. intake_workflow_upsert with the replacement workflow
        - When recreating a Telegram Business workflow, you do not need to manually carry
          source_config.business_connection_id if this user has exactly one connected Telegram
          Business account; the backend resolves it automatically.
        - If the user has multiple connected Telegram Business accounts, specify
          source_config.business_connection_id explicitly.
        - Do not try to patch or overwrite an existing Telegram Business workflow by reusing its workflow_id.
        - required_fields must be stable machine-readable field ids, not display labels.
          Use concise ASCII snake_case ids like ["date", "time", "vehicle_type", "service_name"].
          If the user describes fields in another language, translate the meaning into stable ids and
          keep the user's wording in field_guidance or assistant_instructions.
        - field_guidance may be either:
          - a dict keyed by field name, or
          - a short plain-text note; it will be stored as general guidance.
        - field_guidance dict keys must match required_fields ids. Do not create parallel localized keys.
        - If a sink needs human-readable or localized column names, put those labels in
          sink_config.field_mapping; do not change required_fields ids.
        - source_config is optional.
        - reply_mode is always auto.
        - If source_config.conversation_id is omitted, the workflow scans recent conversations
          for the configured source instead of pinning one specific thread.
        - By default, do not filter inbound messages by intent before the workflow can reply.
          Only set source_config.intent_match_required=true when the user explicitly asks for
          this workflow to handle only messages matching a specific intent.
        - channel/provider pairs supported here:
          - instagram_dm + composio
          - telegram_business_dm + telegram_bot_api
        - For Telegram Business, source_config.business_connection_id may be omitted only when the
          user has exactly one connected Telegram Business account; otherwise it must be provided.
        - assistant_instructions should store the durable business brief for the workflow:
          the user's goals, reply style, qualification rules, booking policy, escalation boundaries,
          important constraints, and any other operating instructions learned during the conversation that
          should persist for future inbox turns.
        - business_facts should store compact explicit facts the owner states in chat and that intake may
          rely on without a bound source file: prices, service menu highlights, hours, discounts, addresses,
          policies, package names, and other concrete facts. Do not copy uploaded files, spreadsheets, long
          catalogues, or extracted document text into business_facts; keep those in knowledge_file_ids.
        - knowledge_file_ids is optional. Use it only when the user explicitly wants uploaded source files bound to the workflow.
        - For spreadsheets or broad source docs, call business_knowledge_index on the original uploaded file ids, query the business knowledge with business_knowledge_query for representative business facts, then bind those same source file ids here.
        - The workflow must still work when knowledge_file_ids is empty; in that case rely on the saved instructions
          and other workflow fields instead of pretending files exist.
        - sink_config must contain the concrete configuration needed by the chosen sink_type.
        - Valid sink_type values here are local_csv, google_sheets_composio, or generic_composio_write.
        - Never invent sink_type=google_sheets.
        - For local_csv, use sink_config={"file_path": "tulpa_stuff/bookings.csv"}.
        - Do not use sink_config.filename for local_csv workflows.
        - For Google Sheets, pass toolkit-level configuration, not a concrete tool slug:
          sink_type=google_sheets_composio
          sink_config={"toolkit": "googlesheets", "field_mapping": {...}, "static_arguments": {...}}
        - OpenTulpa resolves the concrete Composio tool at execution time from the toolkit.
        - If the user only gives a Google Sheet URL, extract the spreadsheet ID and pass it inside
          sink_config.static_arguments.
        - For Google Sheets, include sink_config.static_arguments.sheetName when the target tab is known.
          If it is unknown, do not guess names like Sheet1 or Лист1: pass the spreadsheetId and let the
          backend auto-fill sheetName only when Composio can prove the spreadsheet has exactly one tab.
          If the spreadsheet has multiple tabs, ask the user which tab to use.
        - For generic_composio_write, prefer:
          sink_config={"toolkit": "...", "operation_hint": "...", "field_mapping": {...}, "static_arguments": {...}}
        """
        _ = execution_origin, preapproved, guard_context
        safe_customer = require_customer_id(runtime)
        safe_name = str(name or "").strip()
        safe_intent = str(intent_description or "").strip()
        safe_channel = str(channel or "").strip() or "instagram_dm"
        safe_provider = str(provider or "").strip() or "composio"
        safe_schedule = "" if safe_channel == "telegram_business_dm" else (str(schedule or "").strip() or "*/2 * * * *")
        _active_thread_id(runtime, thread_id)
        safe_reply_mode = "auto"
        safe_sink_type = str(sink_type or "").strip()
        safe_workflow_id = _normalize_optional_id(workflow_id)
        safe_required_fields = _unique_string_list(required_fields)
        safe_knowledge_file_ids = _unique_string_list(knowledge_file_ids)
        safe_sink_config = sink_config if isinstance(sink_config, dict) else {}
        safe_source_config = source_config if isinstance(source_config, dict) else None
        safe_field_guidance = (
            field_guidance
            if isinstance(field_guidance, dict)
            else ({"notes": str(field_guidance).strip()} if str(field_guidance or "").strip() else None)
        )
        safe_assistant_instructions = str(assistant_instructions or "").strip()
        safe_business_facts = (
            business_facts
            if isinstance(business_facts, dict)
            else ({"notes": str(business_facts).strip()} if str(business_facts or "").strip() else None)
        )
        if not safe_name:
            return {"error": "intake_workflow_upsert failed: name is required"}
        if not safe_intent:
            return {"error": "intake_workflow_upsert failed: intent_description is required"}
        if not safe_required_fields:
            return {"error": "intake_workflow_upsert failed: required_fields must contain at least one field"}
        if not safe_sink_type:
            return {"error": "intake_workflow_upsert failed: sink_type is required"}
        if not safe_sink_config:
            return {"error": "intake_workflow_upsert failed: sink_config is required"}
        sink_error = _validate_intake_sink_request(
            sink_type=safe_sink_type,
            sink_config=safe_sink_config,
        )
        if sink_error:
            return {"error": f"intake_workflow_upsert failed: {sink_error}"}

        return await http.request_item(
            "intake_workflow_upsert",
            "POST",
            "/internal/intake/workflows/upsert",
            "workflow",
            default={},
            json_body={
                "customer_id": safe_customer,
                "workflow_id": safe_workflow_id or None,
                "name": safe_name,
                "channel": safe_channel,
                "provider": safe_provider,
                "source_config": safe_source_config,
                "intent_description": safe_intent,
                "required_fields": safe_required_fields,
                "field_guidance": safe_field_guidance,
                "assistant_instructions": safe_assistant_instructions,
                "business_facts": safe_business_facts,
                "knowledge_file_ids": safe_knowledge_file_ids,
                "sink_type": safe_sink_type,
                "sink_config": safe_sink_config,
                "schedule": safe_schedule,
                "notify_user": bool(notify_user),
                "enabled": bool(enabled),
                "reply_mode": safe_reply_mode,
            },
            timeout=20.0,
        )

    @tool
    async def telegram_business_status() -> Any:
        """Check whether Telegram Business is connected for the active user and inspect available business connections."""
        customer_id = require_customer_id(runtime)
        return await http.request(
            "telegram_business_status",
            "POST",
            "/internal/telegram/business/status",
            json_body={"customer_id": customer_id},
            timeout=10.0,
        )

    @tool
    async def intake_workflow_list(include_disabled: bool = False) -> Any:
        """List saved intake workflows for inbound DM automation and booking flows."""
        customer_id = require_customer_id(runtime)
        return await http.request_item(
            "intake_workflow_list",
            "POST",
            "/internal/intake/workflows/list",
            "workflows",
            default=[],
            json_body={
                "customer_id": customer_id,
                "include_disabled": bool(include_disabled),
            },
            timeout=10.0,
        )

    @tool
    async def intake_workflow_get(workflow_id: str) -> Any:
        """Fetch one saved intake workflow configuration by workflow_id."""
        customer_id = require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_get failed: workflow_id is required"}
        return await http.request_item(
            "intake_workflow_get",
            "POST",
            "/internal/intake/workflows/get",
            "workflow",
            default={},
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
            },
            timeout=10.0,
        )

    @tool
    async def intake_workflow_delete(workflow_id: str) -> Any:
        """Delete one intake workflow automation and its scheduled routine, if any."""
        customer_id = require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_delete failed: workflow_id is required"}
        result = await http.request(
            "intake_workflow_delete",
            "POST",
            "/internal/intake/workflows/delete",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
            },
            timeout=10.0,
        )
        if isinstance(result, dict) and bool(result.get("deleted", False)):
            result["final_response_hint"] = "Deleted the intake workflow. It is gone now."
        return result

    @tool
    async def intake_workflow_run(workflow_id: str, force: bool = False) -> Any:
        """Manually run one intake workflow now for testing or forced processing."""
        customer_id = require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_run failed: workflow_id is required"}
        return await http.request(
            "intake_workflow_run",
            "POST",
            "/internal/intake/workflows/run",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
                "force": bool(force),
                "event_type": "manual",
            },
            timeout=60.0,
        )

    return {
        "intake_workflow_upsert": intake_workflow_upsert,
        "intake_workflow_list": intake_workflow_list,
        "intake_workflow_get": intake_workflow_get,
        "intake_workflow_delete": intake_workflow_delete,
        "intake_workflow_run": intake_workflow_run,
        "telegram_business_status": telegram_business_status,
    }
