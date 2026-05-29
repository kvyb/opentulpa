"""Prepare graph input and invocation config for one user turn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from opentulpa.agent.lc_messages import HumanMessage
from opentulpa.agent.prompt_classifier import classify_prompt_mode as _classify_prompt_mode
from opentulpa.agent.turn_budget import initial_turn_budget
from opentulpa.agent.turn_policy import normalize_turn_mode as _normalize_turn_mode


@dataclass(slots=True)
class PreparedTurnContext:
    through_id: int | None
    config: dict[str, Any]
    graph_input: dict[str, Any]


class TurnContextProvider(Protocol):
    def register_links_from_text(
        self,
        *,
        customer_id: str,
        text: str,
        source: str,
        limit: int,
    ) -> None: ...

    def expand_link_aliases(self, *, customer_id: str, text: str) -> str: ...

    def list_pending_context_events(self, *, customer_id: str, limit: int) -> list[dict[str, Any]]: ...

    async def list_available_skills(self, customer_id: str) -> list[Any]: ...

    async def load_skill_context_by_names(
        self,
        *,
        customer_id: str,
        skill_names: list[str],
    ) -> dict[str, Any]: ...

    def effective_recursion_limit(self, override: int | None) -> int: ...

    def recursion_limit_for_turn(
        self,
        *,
        customer_id: str,
        thread_id: str,
        requested_turn_mode: str,
        requested_limit: int,
        prompt_mode: str,
        user_text: str,
    ) -> int: ...


def summarize_pending_payload(payload: Any, *, payload_limit: int = 240) -> str:
    if isinstance(payload, dict):
        allowed_keys = (
            "status",
            "action_name",
            "execution_ok",
            "execution_status",
            "execution_summary",
            "execution_error",
            "notification_status",
            "notification_error",
            "retryable",
            "event_label",
            "routine_id",
            "routine_name",
            "task_id",
            "reason",
        )
        parts: list[str] = []
        for key in allowed_keys:
            value = payload.get(key)
            if value in (None, ""):
                continue
            text = " ".join(str(value).split())
            if len(text) > 90:
                text = text[:90] + "..."
            parts.append(f"{key}={text}")
        if not parts:
            keys = sorted(str(key) for key in payload)
            if keys:
                shown = ", ".join(keys[:6])
                more = f" (+{len(keys) - 6})" if len(keys) > 6 else ""
                return f"payload_keys={shown}{more}"
            return ""
        summary = "; ".join(parts)
    else:
        summary = " ".join(str(payload).split())
    if len(summary) > payload_limit:
        summary = summary[:payload_limit] + "..."
    return summary


def format_pending_context(events: list[dict[str, Any]], *, payload_limit: int = 240) -> str:
    lines: list[str] = []
    for idx, event in enumerate(events, start=1):
        source = str(event.get("source", "event"))
        event_type = str(event.get("event_type", "update"))
        payload_text = summarize_pending_payload(
            event.get("payload", {}),
            payload_limit=payload_limit,
        )
        lines.append(
            f"{idx}. [{source}/{event_type}] {payload_text}"
            if payload_text
            else f"{idx}. [{source}/{event_type}]"
        )
    return "\n".join(lines)


def build_pending_context_summary(
    *,
    events: list[dict[str, Any]],
    customer_id: str,
    include_pending_context: bool,
) -> tuple[str, int | None]:
    del customer_id
    if not include_pending_context:
        return "", None
    if not events:
        return "", None
    through_id = int(events[-1]["id"])
    return format_pending_context(events), through_id


async def pre_resolve_skill_state(
    context_provider: TurnContextProvider,
    *,
    customer_id: str,
    user_text: str,
    prompt_mode: str,
    forced_skill_names: list[str] | None = None,
) -> dict[str, Any]:
    query = str(user_text or "").strip()
    available_skills = await context_provider.list_available_skills(customer_id)
    forced_names = [
        str(item or "").strip()
        for item in (forced_skill_names or [])
        if str(item or "").strip()
    ]
    forced_skill_context = (
        await context_provider.load_skill_context_by_names(
            customer_id=customer_id,
            skill_names=forced_names,
        )
        if forced_names
        else {"skill_names": [], "context": ""}
    )
    invoked_context = str(forced_skill_context.get("context", "")).strip()
    invoked_names = list(forced_skill_context.get("skill_names", []) or [])
    if not query or forced_names:
        return _skill_state(
            prompt_mode=prompt_mode,
            query=query,
            skill_names=forced_names,
            available_skills=available_skills,
            invoked_context=invoked_context,
            invoked_names=invoked_names,
        )
    return _skill_state(
        prompt_mode=prompt_mode,
        query=query,
        skill_names=[],
        available_skills=available_skills,
        invoked_context="",
        invoked_names=[],
    )


def _skill_state(
    *,
    prompt_mode: str,
    query: str,
    skill_names: list[str],
    available_skills: list[Any],
    invoked_context: str,
    invoked_names: list[Any],
) -> dict[str, Any]:
    return {
        "prompt_mode": prompt_mode,
        "active_skill_query": query,
        "active_skill_names": skill_names,
        "active_available_skills": available_skills,
        "active_skill_discovery_context": "",
        "active_invoked_skill_context": invoked_context,
        "active_invoked_skill_names": invoked_names,
        "active_skill_context": invoked_context,
    }


def build_graph_input(
    *,
    user_text: str,
    customer_id: str,
    thread_id: str,
    turn_mode: str,
    prompt_mode: str,
    pending_context_summary: str,
    trace_id: str,
    skill_state: dict[str, Any],
    turn_budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    safe_turn_budget = turn_budget or initial_turn_budget(
        turn_mode=turn_mode,
        graph_recursion_limit=30,
    )
    return {
        "messages": [HumanMessage(content=user_text)],
        "customer_id": customer_id,
        "thread_id": thread_id,
        "turn_mode": _normalize_turn_mode(turn_mode),
        "prompt_mode": prompt_mode,
        "turn_status": "running",
        "final_response_text": "",
        "pending_context_summary": pending_context_summary,
        "agent_trace_id": trace_id,
        "langfuse_graph_callback_attached": False,
        "tool_error_count": 0,
        "turn_plan": [],
        "turn_budget": safe_turn_budget,
        "turn_finalization_reason": "",
        "workflow_setup_no_progress_retry_count": 0,
        "workflow_setup_repair_instruction": "",
        "live_user_steering": [],
        "stream_model_calls": False,
        **skill_state,
    }


async def prepare_turn_context(
    context_provider: TurnContextProvider,
    *,
    thread_id: str,
    customer_id: str,
    text: str,
    turn_mode: str,
    include_pending_context: bool,
    trace_id: str,
    recursion_limit_override: int | None,
    forced_skill_names: list[str] | None,
    prompt_mode_override: str | None,
    build_langfuse_callbacks: Callable[..., list[Any]],
    tool_schema_trace_fields: Callable[[str], dict[str, Any]],
    langchain_callback_metadata: Callable[[dict[str, Any]], dict[str, str]],
) -> PreparedTurnContext | None:
    user_text = str(text or "")
    context_provider.register_links_from_text(
        customer_id=customer_id,
        text=user_text,
        source="user_turn",
        limit=30,
    )
    user_text = context_provider.expand_link_aliases(customer_id=customer_id, text=user_text)
    pending_context_summary, through_id = build_pending_context_summary(
        events=context_provider.list_pending_context_events(customer_id=customer_id, limit=20),
        customer_id=customer_id,
        include_pending_context=include_pending_context,
    )
    prompt_mode = str(prompt_mode_override or "").strip().lower() or _classify_prompt_mode(
        user_text,
        turn_mode=turn_mode,
    )
    skill_state = await pre_resolve_skill_state(
        context_provider,
        customer_id=customer_id,
        user_text=user_text,
        prompt_mode=prompt_mode,
        forced_skill_names=forced_skill_names,
    )
    requested_limit = context_provider.effective_recursion_limit(recursion_limit_override)
    graph_recursion_limit = context_provider.recursion_limit_for_turn(
        customer_id=customer_id,
        thread_id=thread_id,
        requested_turn_mode=turn_mode,
        requested_limit=requested_limit,
        prompt_mode=prompt_mode,
        user_text=user_text,
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": graph_recursion_limit,
    }
    callbacks = build_langfuse_callbacks(
        customer_id=customer_id,
        trace_id=trace_id,
        thread_id=thread_id,
        turn_mode=turn_mode,
        prompt_mode=prompt_mode,
    )
    graph_langfuse_callback_attached = bool(callbacks)
    if callbacks:
        config["callbacks"] = callbacks
        config_metadata = {
            "langfuse_user_id": str(customer_id or "").strip(),
            "langfuse_session_id": str(thread_id or "").strip(),
            "langfuse_tags": [
                item
                for item in (str(turn_mode or "").strip(), str(prompt_mode or "").strip())
                if item
            ],
            "opentulpa_trace_id": str(trace_id or "").strip(),
            "thread_id": str(thread_id or "").strip(),
            "turn_mode": str(turn_mode or "").strip(),
            "prompt_mode": str(prompt_mode or "").strip(),
        }
        config_metadata.update(tool_schema_trace_fields(turn_mode))
        config["metadata"] = langchain_callback_metadata(config_metadata)
        config["tags"] = list(config_metadata["langfuse_tags"])
    graph_input = build_graph_input(
        user_text=user_text,
        customer_id=customer_id,
        thread_id=thread_id,
        turn_mode=turn_mode,
        prompt_mode=prompt_mode,
        pending_context_summary=pending_context_summary,
        trace_id=trace_id,
        skill_state=skill_state,
        turn_budget=dict(
            initial_turn_budget(
                turn_mode=turn_mode,
                graph_recursion_limit=graph_recursion_limit,
            )
        ),
    )
    graph_input["langfuse_graph_callback_attached"] = graph_langfuse_callback_attached
    return PreparedTurnContext(
        through_id=through_id,
        config=config,
        graph_input=graph_input,
    )
