from __future__ import annotations

from typing import Any

from opentulpa.agent.turn_runtime_policy import (
    effective_turn_mode,
    recursion_limit_for_turn,
)


class _WorkflowSetupService:
    def __init__(self, session: dict[str, Any] | None) -> None:
        self.session = session

    def get_thread_session(
        self,
        *,
        customer_id: str,
        thread_id: str,
        include_paused: bool = False,
    ) -> dict[str, Any] | None:
        del customer_id, thread_id, include_paused
        return self.session


class _Runtime:
    def __init__(self, session: dict[str, Any] | None) -> None:
        self.workflow_setup_service = _WorkflowSetupService(session)


def test_effective_turn_mode_promotes_interactive_with_active_workflow_setup() -> None:
    runtime = _Runtime({"status": "active"})

    assert (
        effective_turn_mode(
            runtime,
            customer_id="customer",
            thread_id="thread",
            requested_turn_mode="interactive",
        )
        == "workflow_setup"
    )


def test_recursion_limit_for_turn_boosts_workflow_setup_only() -> None:
    runtime = _Runtime({"status": "active"})

    assert (
        recursion_limit_for_turn(
            runtime,
            customer_id="customer",
            thread_id="thread",
            requested_turn_mode="interactive",
            requested_limit=80,
        )
        > 80
    )
    assert (
        recursion_limit_for_turn(
            _Runtime(None),
            customer_id="customer",
            thread_id="thread",
            requested_turn_mode="interactive",
            requested_limit=80,
        )
        == 80
    )


def test_recursion_limit_for_turn_preserves_high_configured_limit() -> None:
    assert (
        recursion_limit_for_turn(
            _Runtime(None),
            customer_id="customer",
            thread_id="thread",
            requested_turn_mode="interactive",
            requested_limit=256,
        )
        == 256
    )


def test_recursion_limit_for_turn_boosts_initial_workflow_setup_request() -> None:
    assert (
        recursion_limit_for_turn(
            _Runtime(None),
            customer_id="customer",
            thread_id="thread",
            requested_turn_mode="interactive",
            requested_limit=30,
            prompt_mode="execution",
            user_text="Хочу создать workflow для Telegram Business входящих сообщений.",
        )
        == 128
    )
    assert (
        recursion_limit_for_turn(
            _Runtime({"status": "active"}),
            customer_id="customer",
            thread_id="thread",
            requested_turn_mode="interactive",
            requested_limit=256,
        )
        == 256
    )
