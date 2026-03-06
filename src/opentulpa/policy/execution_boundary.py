"""Execution-boundary guard orchestration for executable actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_SCHEDULED_ORIGINS = {"scheduled", "schedule", "routine", "wake", "background"}
_SCHEDULED_THREAD_PREFIXES = ("wake_", "wake-", "routine_", "routine-")


@dataclass(slots=True)
class ExecutionBoundaryContext:
    customer_id: str
    thread_id: str
    action_name: str
    action_args: dict[str, Any]
    execution_origin: str | None = None
    preapproved: bool = False
    action_note: str | None = None


class ExecutionBoundaryGuard:
    """
    Centralized execution-boundary guard.

    Guardrails are evaluated only for interactive executable actions.
    Scheduled/wake-origin executions are pre-authorized at schedule creation time.
    """

    def __init__(self, *, runtime: Any) -> None:
        self._runtime = runtime

    @staticmethod
    def _normalize_thread_id(thread_id: str | None) -> str:
        return str(thread_id or "").strip()

    @classmethod
    def normalize_execution_origin(cls, *, thread_id: str | None, execution_origin: str | None) -> str:
        raw_origin = str(execution_origin or "").strip().lower()
        if raw_origin in _SCHEDULED_ORIGINS:
            return "scheduled"
        if raw_origin in {"interactive", "manual", "chat"}:
            return "interactive"
        safe_thread = cls._normalize_thread_id(thread_id).lower()
        if any(safe_thread.startswith(prefix) for prefix in _SCHEDULED_THREAD_PREFIXES):
            return "scheduled"
        return "interactive"

    async def evaluate(self, context: ExecutionBoundaryContext) -> dict[str, Any]:
        if bool(context.preapproved):
            return {
                "gate": "allow",
                "reason": "approval_execution_preapproved",
                "summary": f"execute {context.action_name}",
            }

        normalized_origin = self.normalize_execution_origin(
            thread_id=context.thread_id,
            execution_origin=context.execution_origin,
        )
        if normalized_origin == "scheduled":
            return {
                "gate": "allow",
                "reason": "scheduled_execution_skip_guardrail",
                "summary": f"execute {context.action_name}",
            }

        if not hasattr(self._runtime, "evaluate_tool_guardrail"):
            return {
                "gate": "allow",
                "reason": "guardrail_runtime_unavailable_allow",
                "summary": f"execute {context.action_name}",
            }

        try:
            return await self._runtime.evaluate_tool_guardrail(
                customer_id=str(context.customer_id or "").strip(),
                thread_id=self._normalize_thread_id(context.thread_id) or "interactive",
                action_name=str(context.action_name or "").strip(),
                action_args=context.action_args if isinstance(context.action_args, dict) else {},
                action_note=str(context.action_note or "").strip()[:2000] or None,
            )
        except Exception as exc:
            return {
                "gate": "require_approval",
                "reason": f"guardrail_boundary_error:{exc}",
                "summary": f"execute {context.action_name}",
            }
