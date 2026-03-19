"""Agent runtime state models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from opentulpa.agent.lc_messages import AnyMessage


class ToolOutcome(TypedDict, total=False):
    tool_name: str
    tool_call_id: str
    status: Literal["ok", "error", "approval_pending"]
    approval_id: str
    result_text: str
    error: str


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    agent_trace_id: str
    customer_id: str
    thread_id: str
    turn_mode: Literal["interactive", "routine_wake", "approval_recovery", "event_notification"]
    turn_status: Literal["running", "approval_pending", "completed", "failed"]
    final_response_text: str
    pending_context_summary: str
    active_skill_query: str
    active_skill_context: str
    active_skill_names: list[str]
    tool_outcomes: list[ToolOutcome]
    tool_validation_passed: bool
    tool_error_count: int
    last_tool_error: str
    approval_handoff: bool
    claim_check_verdict: dict[str, Any]
    claim_check_needs_retry: bool
    claim_check_retry_count: int
