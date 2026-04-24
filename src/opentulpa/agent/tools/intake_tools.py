"""Intake workflow tool registration."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id, require_thread_id


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


def register_intake_tools(runtime: Any) -> dict[str, Any]:
    @tool
    async def intake_workflow_upsert(
        name: str,
        intent_description: str,
        required_fields: list[str],
        sink_type: str,
        sink_config: dict[str, Any],
        schedule: str = "*/5 * * * *",
        channel: str = "instagram_dm",
        provider: str = "composio",
        source_config: dict[str, Any] | None | str = None,
        field_guidance: dict[str, Any] | None | str = None,
        assistant_instructions: str = "",
        knowledge_file_ids: list[str] | None = None,
        notify_user: bool = True,
        enabled: bool = True,
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
        - Prefer the intake workflow setup wizard for interactive workflow authoring and edits.
        - In normal chat, use intake_workflow_setup_begin plus the setup tools to build the draft with the user.
        - Reserve direct intake_workflow_upsert for the final persist step after the wizard draft is complete and explicitly confirmed.
        - Outside workflow setup mode, do not use intake_workflow_upsert to author a brand-new workflow from scratch.
        - In setup mode, call intake_workflow_upsert only when the draft already contains the exact workflow fields to save.
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
        - required_fields must be a list of plain field names like ["date", "time", "car_type"].
        - field_guidance may be either:
          - a dict keyed by field name, or
          - a short plain-text note; it will be stored as general guidance.
        - source_config is optional.
        - If source_config.conversation_id is omitted, the workflow scans recent conversations
          for the configured source instead of pinning one specific thread.
        - channel/provider pairs supported here:
          - instagram_dm + composio
          - telegram_business_dm + telegram_bot_api
        - For Telegram Business, source_config.business_connection_id may be omitted only when the
          user has exactly one connected Telegram Business account; otherwise it must be provided.
        - assistant_instructions should store the durable business brief for the workflow:
          the user's goals, reply style, qualification rules, booking policy, escalation boundaries,
          important constraints, and any other operating instructions learned during the conversation that
          should persist for future inbox turns.
        - knowledge_file_ids is optional. Use it only when the user explicitly wants uploaded files bound to the workflow.
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
        - For generic_composio_write, prefer:
          sink_config={"toolkit": "...", "operation_hint": "...", "field_mapping": {...}, "static_arguments": {...}}
        """
        _ = thread_id, execution_origin, preapproved, guard_context
        safe_customer = require_customer_id(runtime)
        safe_name = str(name or "").strip()
        safe_intent = str(intent_description or "").strip()
        safe_channel = str(channel or "").strip() or "instagram_dm"
        safe_provider = str(provider or "").strip() or "composio"
        safe_schedule = "" if safe_channel == "telegram_business_dm" else (str(schedule or "").strip() or "*/5 * * * *")
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

        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/upsert",
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
                "knowledge_file_ids": safe_knowledge_file_ids,
                "sink_type": safe_sink_type,
                "sink_config": safe_sink_config,
                "schedule": safe_schedule,
                "notify_user": bool(notify_user),
                "enabled": bool(enabled),
            },
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_upsert failed: {r.text}"}
        return r.json().get("workflow", {})

    @tool
    async def telegram_business_status() -> Any:
        """Check whether Telegram Business is connected for the active user and inspect available business connections."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/telegram/business/status",
            json_body={"customer_id": customer_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"telegram_business_status failed: {r.text}"}
        return r.json()

    @tool
    async def intake_workflow_list(include_disabled: bool = False) -> Any:
        """List intake workflows for the current user."""
        customer_id = require_customer_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/list",
            json_body={
                "customer_id": customer_id,
                "include_disabled": bool(include_disabled),
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_list failed: {r.text}"}
        return r.json().get("workflows", [])

    @tool
    async def intake_workflow_get(workflow_id: str) -> Any:
        """Get one intake workflow by id."""
        customer_id = require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_get failed: workflow_id is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/get",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_get failed: {r.text}"}
        return r.json().get("workflow", {})

    @tool
    async def intake_workflow_delete(workflow_id: str) -> Any:
        """Delete one intake workflow and its scheduled routine."""
        customer_id = require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_delete failed: workflow_id is required"}
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/workflows/delete",
            json_body={
                "customer_id": customer_id,
                "workflow_id": safe_workflow_id,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_delete failed: {r.text}"}
        return r.json()

    @tool
    async def intake_workflow_setup_begin(mode: str, workflow_id: str = "") -> Any:
        """Begin or resume a workflow setup wizard for the current thread.

        Use this when the user wants to create a new intake workflow or edit an existing one.
        - mode=create starts a new draft workflow setup session for this thread.
        - mode=edit loads the existing workflow into the wizard draft and requires workflow_id.
        - Once the wizard is active, stay in setup mode and use the setup tools until commit, pause, or cancel.
        """
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        safe_mode = str(mode or "").strip().lower()
        safe_workflow_id = str(workflow_id or "").strip()
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/begin",
            json_body={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "mode": safe_mode,
                "workflow_id": safe_workflow_id or None,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_begin failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_get(include_paused: bool = True) -> Any:
        """Get the current workflow setup session for this thread."""
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/get",
            json_body={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "include_paused": bool(include_paused),
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_get failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_update(
        draft_patch: dict[str, Any] | None = None,
        scratchpad_patch: dict[str, Any] | None = None,
    ) -> Any:
        """Patch the workflow setup draft and scratchpad for the current thread.

        Use this inside workflow setup mode to record newly learned workflow fields and internal setup notes.
        For local_csv workflows, use draft_patch.sink_config={"file_path": "..."}.
        When replacing field-specific guidance or sink_config.field_mapping, send the full current object.
        """
        if not isinstance(draft_patch, dict) and not isinstance(scratchpad_patch, dict):
            return {"error": "intake_workflow_setup_update failed: draft_patch or scratchpad_patch is required"}
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/update",
            json_body={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "draft_patch": draft_patch if isinstance(draft_patch, dict) else None,
                "scratchpad_patch": scratchpad_patch if isinstance(scratchpad_patch, dict) else None,
            },
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_update failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_mark_proposed() -> Any:
        """Mark the current workflow setup draft as the proposal shown to the user."""
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/mark_proposed",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_mark_proposed failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_confirm_current() -> Any:
        """Confirm the current proposed workflow draft for the active setup session."""
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/confirm_current",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_confirm_current failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_commit() -> Any:
        """Persist the confirmed workflow setup draft and activate the workflow."""
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/commit",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=20.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_commit failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_pause() -> Any:
        """Pause the active workflow setup session for the current thread."""
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/pause",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_pause failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_setup_cancel() -> Any:
        """Cancel the workflow setup session for the current thread."""
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/cancel",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=10.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_cancel failed: {r.text}"}
        return r.json().get("session", {})

    @tool
    async def intake_workflow_run(workflow_id: str, force: bool = False) -> Any:
        """Run one intake workflow immediately for the current user."""
        customer_id = require_customer_id(runtime)
        safe_workflow_id = str(workflow_id or "").strip()
        if not safe_workflow_id:
            return {"error": "intake_workflow_run failed: workflow_id is required"}
        r = await runtime._request_with_backoff(
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
        if r.status_code != 200:
            return {"error": f"intake_workflow_run failed: {r.text}"}
        return r.json()

    return {
        "intake_workflow_upsert": intake_workflow_upsert,
        "intake_workflow_list": intake_workflow_list,
        "intake_workflow_get": intake_workflow_get,
        "intake_workflow_delete": intake_workflow_delete,
        "intake_workflow_setup_begin": intake_workflow_setup_begin,
        "intake_workflow_setup_get": intake_workflow_setup_get,
        "intake_workflow_setup_update": intake_workflow_setup_update,
        "intake_workflow_setup_mark_proposed": intake_workflow_setup_mark_proposed,
        "intake_workflow_setup_confirm_current": intake_workflow_setup_confirm_current,
        "intake_workflow_setup_commit": intake_workflow_setup_commit,
        "intake_workflow_setup_pause": intake_workflow_setup_pause,
        "intake_workflow_setup_cancel": intake_workflow_setup_cancel,
        "intake_workflow_run": intake_workflow_run,
        "telegram_business_status": telegram_business_status,
    }
