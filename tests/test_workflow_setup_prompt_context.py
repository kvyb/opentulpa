from __future__ import annotations

from opentulpa.agent.workflow_setup_prompt_context import build_workflow_setup_control_context


def _session_with_google_sheets_sink(*, sheet_name: str = "") -> dict[str, object]:
    static_arguments: dict[str, str] = {"spreadsheetId": "sheet_123"}
    if sheet_name:
        static_arguments["sheetName"] = sheet_name
    return {
        "status": "active",
        "mode": "create",
        "draft_upsert": {
            "name": "AutoSpa",
            "channel": "telegram_business_dm",
            "intent_description": "Book car wash leads",
            "required_fields": ["service_name", "date", "time", "phone"],
            "sink_type": "google_sheets_composio",
            "sink_config": {
                "toolkit": "googlesheets",
                "static_arguments": static_arguments,
            },
        },
        "scratchpad": {},
    }


def test_control_card_allows_preflight_without_google_sheets_sheet_name() -> None:
    card = build_workflow_setup_control_context(_session_with_google_sheets_sink())

    assert "draft_status: needs_preflight" in card
    assert "missing_core_inputs: none" in card
    assert "Call intake_workflow_setup_preflight" in card


def test_control_card_asks_follow_up_when_preflight_needs_google_sheets_tab_clarification() -> None:
    session = _session_with_google_sheets_sink()
    session["scratchpad"] = {
        "last_preflight": {
            "ok": False,
            "status": "needs_clarification",
            "follow_up_questions": ["Which tab should I write into?"],
        }
    }

    card = build_workflow_setup_control_context(session)

    assert "draft_status: needs_clarification" in card
    assert "latest_preflight_follow_up: Which tab should I write into?" in card
    assert "ask this blocker only: Which tab should I write into?" in card
