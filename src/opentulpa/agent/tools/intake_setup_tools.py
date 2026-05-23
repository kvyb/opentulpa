"""Intake workflow setup wizard tool registration."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from opentulpa.agent.tools.common import require_customer_id, require_thread_id


def _proposal_final_response_hint(session: dict[str, Any]) -> str:
    if not str(session.get("last_proposed_draft_hash", "") or "").strip():
        return ""
    preflight = session.get("preflight")
    if not isinstance(preflight, dict) or str(preflight.get("status", "") or "") != "ready":
        return ""
    draft = session.get("draft_upsert")
    if not isinstance(draft, dict):
        return ""
    raw_fields = draft.get("required_fields")
    fields = ", ".join(
        str(item or "").strip()
        for item in (raw_fields if isinstance(raw_fields, list) else [])
        if str(item or "").strip()
    )
    sink = str(draft.get("sink_type", "") or "").strip() or "unspecified"
    channel = str(draft.get("channel", "") or "").strip() or "unspecified"
    name = str(draft.get("name", "") or "").strip() or "Untitled workflow"
    return (
        "Workflow proposal is ready.\n"
        f"- Name: {name}\n"
        f"- Channel: {channel}\n"
        f"- Required fields: {fields or 'none'}\n"
        f"- Sink: {sink}\n"
        "Confirm/save to activate it, or tell me what to change."
    )


def register_intake_setup_tools(runtime: Any) -> dict[str, Any]:
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
        Put workflow fields directly inside draft_patch; do not nest them under draft, draft_upsert,
        workflow, or workflow_upsert.
        If the owner changes source/channel later, latest explicit owner request wins: update
        draft_patch.channel and draft_patch.provider in the same call before proposing.
        For local_csv workflows, use draft_patch.sink_config={"file_path": "..."}.
        For google_sheets_composio workflows, put spreadsheet targets in
        draft_patch.sink_config.static_arguments, for example spreadsheetId and sheetName;
        put output column labels in draft_patch.sink_config.field_mapping keyed by required field id.
        draft_patch.required_fields must be stable ASCII snake_case ids, not display labels.
        Put localized wording, owner terminology, and extraction hints in draft_patch.field_guidance
        or draft_patch.assistant_instructions. Store compact owner-stated facts like prices,
        service menu highlights, hours, discounts, addresses, and policies in draft_patch.business_facts.
        Do not paste uploaded file contents, large tables, or extracted document text into business_facts;
        bind files through draft_patch.knowledge_file_ids instead.
        field_guidance keys must match required_fields ids.
        Do not set draft_patch.source_config.intent_match_required by default; set it to true only
        when the owner explicitly wants the workflow to ignore messages that do not match the stated intent.
        When replacing field-specific guidance or sink_config.field_mapping, send the full current object.
        If uploaded files are being used as workflow knowledge, patch scratchpad_patch.source_file_ids
        with original uploaded file ids, prepare them with business_knowledge_index, and set
        draft_patch.knowledge_file_ids to those same source file ids.
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
    async def intake_workflow_setup_preflight() -> Any:
        """Run workflow setup preflight validation on the current draft before proposing it.

        Use this after the draft has the intended channel, source, sink, required_fields, and knowledge files.
        Prefer intake_workflow_setup_propose_current when the next step is to show a proposal; use this
        tool only when you need validation details without marking the draft proposed.
        It is non-destructive: it may normalize safe sink details like a single discovered Google Sheets tab,
        and it returns a dry-run preview of what the sink write would look like without writing rows.
        If status is not ready, ask the returned focused follow-up question instead of proposing the workflow.
        """
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/preflight",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=30.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_preflight failed: {r.text}"}
        return r.json().get("preflight", {})

    @tool
    async def intake_workflow_setup_propose_current() -> Any:
        """Preflight the current workflow setup draft and mark it proposed when ready.

        Use this after the draft has the intended channel, source, sink, required_fields,
        and knowledge files, when you are ready to show the owner a proposal. If status is
        ready, this tool marks the exact current draft as proposed in the same backend step.
        Then summarize the returned draft/preflight and ask the owner to confirm before saving.
        If status is not ready, ask the returned focused follow-up question instead.
        """
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/propose_current",
            json_body={"customer_id": customer_id, "thread_id": thread_id},
            timeout=30.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_propose_current failed: {r.text}"}
        session = r.json().get("session", {})
        if isinstance(session, dict):
            hint = _proposal_final_response_hint(session)
            if hint:
                session["final_response_hint"] = hint
        return session

    @tool
    async def intake_workflow_setup_mark_proposed() -> Any:
        """Mark the current workflow setup draft as the proposal shown to the user.

        Prefer intake_workflow_setup_propose_current. Call this only when you already ran
        preflight in a previous step and still need to mark that exact current draft.
        Without this marker, owner confirmation cannot be committed safely.
        """
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
        """Confirm the current proposed workflow draft for the active setup session.

        Use this only after the owner explicitly confirms the proposal. If this reports
        that the workflow draft has not been proposed yet, call preflight and
        intake_workflow_setup_mark_proposed before retrying.
        """
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
        """Persist the confirmed workflow setup draft and activate the workflow.

        This must follow intake_workflow_setup_confirm_current in the same confirmation flow.
        Do not tell the owner the workflow is saved until this tool succeeds.
        """
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
    async def intake_workflow_setup_finalize_confirmation(
        draft_patch: dict[str, Any] | None = None,
        scratchpad_patch: dict[str, Any] | None = None,
    ) -> Any:
        """Finalize an explicitly confirmed workflow setup in one backend operation.

        Use this when the owner confirms a shown workflow proposal. If the same owner
        message also adds small final behavior rules, pass them as draft_patch or
        scratchpad_patch; this tool will persist them, preflight, mark the current
        draft as proposed, confirm it, and commit it. Do not call the separate
        preflight/mark_proposed/confirm_current/commit tools after this succeeds.
        Put workflow fields directly inside draft_patch; do not nest them under draft,
        draft_upsert, workflow, or workflow_upsert.
        """
        customer_id = require_customer_id(runtime)
        thread_id = require_thread_id(runtime)
        r = await runtime._request_with_backoff(
            "POST",
            "/internal/intake/setup/finalize_confirmation",
            json_body={
                "customer_id": customer_id,
                "thread_id": thread_id,
                "draft_patch": draft_patch if isinstance(draft_patch, dict) else None,
                "scratchpad_patch": scratchpad_patch if isinstance(scratchpad_patch, dict) else None,
            },
            timeout=45.0,
        )
        if r.status_code != 200:
            return {"error": f"intake_workflow_setup_finalize_confirmation failed: {r.text}"}
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

    return {
        "intake_workflow_setup_begin": intake_workflow_setup_begin,
        "intake_workflow_setup_get": intake_workflow_setup_get,
        "intake_workflow_setup_update": intake_workflow_setup_update,
        "intake_workflow_setup_preflight": intake_workflow_setup_preflight,
        "intake_workflow_setup_propose_current": intake_workflow_setup_propose_current,
        "intake_workflow_setup_mark_proposed": intake_workflow_setup_mark_proposed,
        "intake_workflow_setup_confirm_current": intake_workflow_setup_confirm_current,
        "intake_workflow_setup_commit": intake_workflow_setup_commit,
        "intake_workflow_setup_finalize_confirmation": intake_workflow_setup_finalize_confirmation,
        "intake_workflow_setup_pause": intake_workflow_setup_pause,
        "intake_workflow_setup_cancel": intake_workflow_setup_cancel,
    }
