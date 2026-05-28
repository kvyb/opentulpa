"""Runtime turn-mode policy shared by graph and transports."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.turn_policy import normalize_turn_mode

WORKFLOW_SETUP_RECURSION_LIMIT = 128
_WORKFLOW_SETUP_BUDGET_TERMS = (
    "workflow",
    "work flow",
    "intake",
    "воркфлоу",
    "workflow setup",
    "telegram business",
)


def active_workflow_setup_session(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
) -> dict[str, Any] | None:
    service = getattr(runtime, "workflow_setup_service", None)
    if service is None or not hasattr(service, "get_thread_session"):
        return None
    try:
        session = service.get_thread_session(
            customer_id=customer_id,
            thread_id=thread_id,
            include_paused=False,
        )
    except Exception:
        return None
    if not isinstance(session, dict):
        return None
    status = str(session.get("status", "") or "").strip().lower()
    return session if status == "active" else None


def effective_turn_mode(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
    requested_turn_mode: str,
) -> str:
    mode = normalize_turn_mode(requested_turn_mode)
    if mode == "workflow_setup":
        return mode
    if mode == "interactive" and active_workflow_setup_session(
        runtime,
        customer_id=customer_id,
        thread_id=thread_id,
    ):
        return "workflow_setup"
    return mode


def recursion_limit_for_turn(
    runtime: Any,
    *,
    customer_id: str,
    thread_id: str,
    requested_turn_mode: str,
    requested_limit: int,
    prompt_mode: str = "",
    user_text: str = "",
) -> int:
    base = max(5, int(requested_limit))
    effective_mode = effective_turn_mode(
        runtime,
        customer_id=customer_id,
        thread_id=thread_id,
        requested_turn_mode=requested_turn_mode,
    )
    if effective_mode != "workflow_setup" and not _looks_like_workflow_setup_budget_turn(
        prompt_mode=prompt_mode,
        user_text=user_text,
    ):
        return base
    return max(base, WORKFLOW_SETUP_RECURSION_LIMIT)


def _looks_like_workflow_setup_budget_turn(*, prompt_mode: str, user_text: str) -> bool:
    if str(prompt_mode or "").strip().lower() == "workflow_setup":
        return True
    lowered = str(user_text or "").casefold()
    return any(term in lowered for term in _WORKFLOW_SETUP_BUDGET_TERMS)
