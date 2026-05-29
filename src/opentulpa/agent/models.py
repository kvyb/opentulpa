"""Agent runtime state models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from langgraph.managed.is_last_step import RemainingSteps
from typing_extensions import TypedDict

from opentulpa.agent.lc_messages import AnyMessage
from opentulpa.agent.tool_outcome_context import add_tool_outcomes


class ToolOutcome(TypedDict, total=False):
    round_id: int
    tool_name: str
    tool_call_id: str
    status: Literal["ok", "error"]
    result_text: str
    error: str
    final_response_hint: str
    tool_signature: str
    trace_id: str


TurnPlanStatus = Literal["pending", "in_progress", "completed", "cancelled"]


class TurnPlanItem(TypedDict):
    id: str
    content: str
    status: TurnPlanStatus


class TurnBudget(TypedDict, total=False):
    max_model_calls: int
    used_model_calls: int
    used_tool_rounds: int
    max_search_calls: int
    used_search_calls: int
    finalizer_used: bool
    exhausted_reason: str


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
    tool_outcomes: Annotated[list[ToolOutcome], add_tool_outcomes]
    turn_plan: list[TurnPlanItem]
    turn_budget: TurnBudget
    turn_finalization_reason: str
    tool_validation_passed: bool
    tool_error_count: int
    last_tool_error: str
    workflow_setup_no_progress_retry_count: int
    workflow_setup_repair_instruction: str
    frozen_prompt_context: dict[str, Any] | None
    frozen_history_projection: dict[str, Any] | None
    live_user_steering: list[str]
    stream_model_calls: bool
    remaining_steps: RemainingSteps
