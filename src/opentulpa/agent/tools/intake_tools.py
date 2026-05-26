"""Intake workflow tool registration."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.tools.intake_setup_tools import register_intake_setup_tools
from opentulpa.agent.tools.intake_workflow_tools import register_intake_workflow_tools


def register_intake_tools(runtime: Any) -> dict[str, Any]:
    workflow_tools = register_intake_workflow_tools(runtime)
    setup_tools = register_intake_setup_tools(runtime)
    return {
        "intake_workflow_upsert": workflow_tools["intake_workflow_upsert"],
        "intake_workflow_list": workflow_tools["intake_workflow_list"],
        "intake_workflow_get": workflow_tools["intake_workflow_get"],
        "intake_workflow_delete": workflow_tools["intake_workflow_delete"],
        "intake_workflow_setup_begin": setup_tools["intake_workflow_setup_begin"],
        "intake_workflow_setup_get": setup_tools["intake_workflow_setup_get"],
        "intake_workflow_setup_update": setup_tools["intake_workflow_setup_update"],
        "intake_workflow_setup_preflight": setup_tools["intake_workflow_setup_preflight"],
        "intake_workflow_setup_propose_current": setup_tools["intake_workflow_setup_propose_current"],
        "intake_workflow_setup_mark_proposed": setup_tools["intake_workflow_setup_mark_proposed"],
        "intake_workflow_setup_confirm_current": setup_tools["intake_workflow_setup_confirm_current"],
        "intake_workflow_setup_commit": setup_tools["intake_workflow_setup_commit"],
        "intake_workflow_setup_finalize_confirmation": setup_tools["intake_workflow_setup_finalize_confirmation"],
        "intake_workflow_setup_pause": setup_tools["intake_workflow_setup_pause"],
        "intake_workflow_setup_cancel": setup_tools["intake_workflow_setup_cancel"],
        "intake_workflow_run": workflow_tools["intake_workflow_run"],
        "telegram_business_status": workflow_tools["telegram_business_status"],
    }
