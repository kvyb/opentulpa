"""Tool execution policy for the runtime graph tools node."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from opentulpa.agent.lc_messages import AnyMessage
from opentulpa.agent.turn_policy import execution_origin_for_turn_mode, normalize_turn_mode
from opentulpa.agent.utils import (
    extract_relative_delay_minutes,
    is_cron_like_schedule,
    latest_user_text,
)


@dataclass(frozen=True)
class ToolExecutionPolicy:
    customer_id: str
    thread_id: str
    turn_mode: str
    execution_origin: str
    allowed_tool_names: set[str]

    @classmethod
    def from_runtime_state(cls, *, runtime: Any, state: Any) -> ToolExecutionPolicy:
        customer_id = str(state.get("customer_id", "") or "")
        thread_id = str(state.get("thread_id", "") or "").strip()
        turn_mode = normalize_turn_mode(state.get("turn_mode"))
        allowed_tool_names: set[str] = set()
        tools_for_turn_mode = getattr(runtime, "tools_for_turn_mode", None)
        if callable(tools_for_turn_mode):
            try:
                allowed_tools = list(tools_for_turn_mode(turn_mode))
                allowed_tool_names = _allowed_tool_names(runtime=runtime, tools=allowed_tools)
            except Exception:
                allowed_tool_names = set()
        return cls(
            customer_id=customer_id,
            thread_id=thread_id,
            turn_mode=turn_mode,
            execution_origin=execution_origin_for_turn_mode(turn_mode, thread_id=thread_id),
            allowed_tool_names=allowed_tool_names,
        )

    def validate_call(self, *, call_name: str, customer_scoped_tools: set[str]) -> None:
        assert call_name
        if self.allowed_tool_names and call_name not in self.allowed_tool_names:
            raise ValueError(
                f"{call_name} is not bound in this turn. Use tool_group_exec "
                "with the matching group and command instead."
            )
        if call_name in customer_scoped_tools and not self.customer_id.strip():
            raise ValueError(f"{call_name} requires customer scope, but customer_id is missing")

    def prepare_args(
        self,
        *,
        call_name: str,
        args: dict[str, Any],
        messages: list[AnyMessage],
    ) -> dict[str, Any]:
        prepared = dict(args)
        if call_name in {"tulpa_run_terminal", "routine_create"}:
            prepared = {
                **prepared,
                "thread_id": self.thread_id,
                "execution_origin": self.execution_origin,
            }
        if call_name == "routine_create":
            latest_user = latest_user_text(messages)
            delay_minutes = extract_relative_delay_minutes(latest_user)
            if delay_minutes is not None and is_cron_like_schedule(
                str(prepared.get("schedule", ""))
            ):
                run_at_local = datetime.now().astimezone() + timedelta(
                    minutes=max(1, delay_minutes)
                )
                prepared["schedule"] = run_at_local.isoformat()
        return prepared


def _allowed_tool_names(*, runtime: Any, tools: list[Any]) -> set[str]:
    names = {
        str(getattr(tool, "name", "") or "").strip()
        for tool in tools
        if str(getattr(tool, "name", "") or "").strip()
    }
    runtime_tools = getattr(runtime, "_tools", None)
    if not isinstance(runtime_tools, dict):
        return names
    for tool in tools:
        for key, candidate in runtime_tools.items():
            if candidate is tool:
                safe_key = str(key or "").strip()
                if safe_key:
                    names.add(safe_key)
    return names
