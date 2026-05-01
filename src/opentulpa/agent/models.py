"""Agent runtime state models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict

from opentulpa.agent.lc_messages import AnyMessage


class ToolOutcome(TypedDict, total=False):
    tool_name: str
    tool_call_id: str
    status: Literal["ok", "error"]
    result_text: str
    error: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    agent_trace_id: str
    customer_id: str
    thread_id: str
    compact_wake: bool
    turn_mode: Literal["interactive", "workflow_setup", "routine_wake", "event_notification"]
    prompt_mode: Literal["literal_chat", "task_chat", "execution", "workflow_setup"]
    turn_status: Literal["running", "completed", "failed"]
    final_response_text: str
    pending_context_summary: str
    active_skill_query: str
    active_skill_names: list[str]
    active_available_skills: list[dict[str, Any]]
    active_skill_discovery_context: str
    active_invoked_skill_context: str
    active_invoked_skill_names: list[str]
    active_skill_context: str
    tool_outcomes: list[ToolOutcome]
    tool_validation_passed: bool
    tool_error_count: int
    last_tool_error: str
    workflow_setup_no_progress_retry_count: int
    workflow_setup_repair_instruction: str
    frozen_prompt_context: dict[str, Any] | None
    frozen_history_projection: dict[str, Any] | None
    remaining_steps: RemainingSteps
    loop_limit_status_update_sent: bool
