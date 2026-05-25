"""Runtime turn-mode policy shared by graph and transports."""

from __future__ import annotations

from typing import Any

from opentulpa.agent.turn_policy import normalize_turn_mode

WORKFLOW_SETUP_RECURSION_LIMIT = 128


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
) -> int:
    base = max(5, min(int(requested_limit), 200))
    if effective_turn_mode(
        runtime,
        customer_id=customer_id,
        thread_id=thread_id,
        requested_turn_mode=requested_turn_mode,
    ) != "workflow_setup":
        return base
    return max(base, min(WORKFLOW_SETUP_RECURSION_LIMIT, 200))
