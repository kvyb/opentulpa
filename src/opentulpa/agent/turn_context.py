"""Dynamic per-turn prompt context for the runtime graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from opentulpa.agent.lc_messages import AnyMessage, SystemMessage
from opentulpa.agent.models import AgentState
from opentulpa.agent.tool_outcome_context import build_tool_outcome_context
from opentulpa.agent.turn_plan import (
    build_turn_plan_prompt_context,
    turn_plan_enabled_for_turn_mode,
)
from opentulpa.agent.workflow_setup_prompt_context import build_workflow_setup_control_context

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DynamicTurnContext:
    messages: list[AnyMessage]
    sections: list[str]


def build_dynamic_turn_context(
    *,
    runtime: Any,
    state: AgentState,
    customer_id: str,
    thread_id: str,
    turn_mode: str,
    current_turn_context_content: str,
    current_turn_context_sections: list[str],
    tool_outcomes: Any,
    budget_context: str,
    loop_limit_repair_instruction: str,
    loop_limit_near: bool,
    live_user_steering: list[str],
    workflow_setup_repair_instruction: str,
) -> DynamicTurnContext:
    messages: list[AnyMessage] = []
    sections: list[str] = []

    turn_plan_context = (
        build_turn_plan_prompt_context(state)
        if turn_plan_enabled_for_turn_mode(turn_mode)
        else ""
    )
    if turn_plan_context:
        messages.append(SystemMessage(content=turn_plan_context))
        sections.append("turn_plan")

    if current_turn_context_content:
        messages.append(SystemMessage(content=current_turn_context_content))
        sections.extend(current_turn_context_sections)

    tool_outcome_context = build_tool_outcome_context(tool_outcomes)
    if tool_outcome_context:
        messages.append(SystemMessage(content=tool_outcome_context))
        sections.append("current_turn_tool_results")

    if loop_limit_near:
        messages.append(SystemMessage(content=loop_limit_repair_instruction))
        sections.append("loop_limit_repair")

    if budget_context:
        messages.append(SystemMessage(content=budget_context))
        sections.append("turn_budget")

    if live_user_steering:
        steering_lines = [
            f"- User steers with message: {fragment}"
            for fragment in live_user_steering
            if str(fragment).strip()
        ]
        if steering_lines:
            messages.append(SystemMessage(content="\n".join(steering_lines)))
            sections.append("live_user_steering")

    if turn_mode == "workflow_setup":
        workflow_setup_context = build_workflow_setup_prompt_context(
            runtime,
            customer_id=customer_id,
            thread_id=thread_id,
        ).strip()
        if workflow_setup_context:
            messages.append(SystemMessage(content=workflow_setup_context))
            sections.append("workflow_setup_control_card")

    if workflow_setup_repair_instruction:
        messages.append(SystemMessage(content=workflow_setup_repair_instruction))
        sections.append("workflow_setup_repair")

    return DynamicTurnContext(messages=messages, sections=sections)


def build_workflow_setup_prompt_context(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
) -> str:
    service = getattr(runtime, "workflow_setup_service", None)
    if service is None or not hasattr(service, "get_thread_session"):
        return ""
    try:
        session = service.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=True,
        )
    except Exception:
        logger.exception(
            "Failed to build workflow setup prompt context (customer_id=%s, thread_id=%s)",
            customer_id,
            thread_id,
        )
        return ""
    return build_workflow_setup_control_context(session)
