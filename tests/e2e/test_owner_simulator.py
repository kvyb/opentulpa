from __future__ import annotations

import json
from typing import Any

from harness.owner_simulator import OwnerProfile, OwnerSimulator


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


def _profile() -> OwnerProfile:
    return OwnerProfile(
        objective="Create a Telegram Business DM workflow.",
        initial_message="I need an intake workflow.",
        known_facts={
            "workflow_name": "E2E Car Wash",
            "required_fields": "car_model, date, time",
            "sink": "local CSV tulpa_stuff/e2e.csv",
        },
        max_turns=5,
    )


def test_owner_simulator_returns_done_only_after_workflow_created() -> None:
    simulator = OwnerSimulator(api_key="test-key", base_url="https://example.com")

    plan = simulator.plan_next_turn(
        profile=_profile(),
        transcript=[
            {"role": "owner", "text": "I need an intake workflow."},
            {"role": "assistant", "text": "Saved and activated."},
        ],
        workflow_state={"workflow_count": 1},
    )

    assert plan.done is True
    assert plan.reason == "workflow_created"
    assert plan.message == ""


def test_owner_simulator_ignores_premature_done_when_workflow_missing(monkeypatch: Any) -> None:
    def _fake_post(*args: Any, **kwargs: Any) -> _DummyResponse:
        _ = args, kwargs
        return _DummyResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "done": True,
                                    "message": "",
                                    "reason": "assistant_seems_finished",
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("harness.owner_simulator.httpx.post", _fake_post)

    simulator = OwnerSimulator(api_key="test-key", base_url="https://example.com")
    plan = simulator.plan_next_turn(
        profile=_profile(),
        transcript=[
            {"role": "owner", "text": "I need an intake workflow."},
            {"role": "assistant", "text": "What fields should I collect?"},
        ],
        workflow_state={"workflow_count": 0},
    )

    assert plan.done is False
    assert plan.message
    assert "premature_done_without_workflow" in plan.reason
