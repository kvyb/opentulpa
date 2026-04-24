from __future__ import annotations

import json
from typing import Any

from harness.lead_simulator import LeadProfile, LeadSimulator


class _DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return dict(self._payload)


def _profile() -> LeadProfile:
    return LeadProfile(
        objective="Book a car wash.",
        initial_message="Hi, I want to book a wash.",
        known_facts={
            "car_model": "Toyota RAV4",
            "car_type": "SUV",
            "wash_type": "full wash",
            "date": "tomorrow",
            "time": "10:00",
        },
        max_turns=6,
    )


def test_lead_simulator_returns_done_only_for_completed_booking() -> None:
    simulator = LeadSimulator(api_key="test-key", base_url="https://example.com")

    plan = simulator.plan_next_turn(
        profile=_profile(),
        transcript=[
            {"role": "lead", "text": "Hi, I want to book a wash."},
            {"role": "assistant", "text": "Your booking is confirmed for tomorrow at 10:00."},
        ],
        booking_state={"status": "completed", "sink_write_status": "succeeded"},
    )

    assert plan.done is True
    assert plan.reason == "booking_state_completed"
    assert plan.message == ""


def test_lead_simulator_ignores_premature_done_when_booking_is_still_active(
    monkeypatch: Any,
) -> None:
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
                                    "reason": "assistant_already_confirmed_booking",
                                    "shared_fact_keys": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("harness.lead_simulator.httpx.post", _fake_post)

    simulator = LeadSimulator(api_key="test-key", base_url="https://example.com")
    plan = simulator.plan_next_turn(
        profile=_profile(),
        transcript=[
            {"role": "lead", "text": "Hi, I want to book a full wash for tomorrow. How much is it?"},
            {
                "role": "assistant",
                "text": (
                    "I don't have the exact pricing in front of me right now, "
                    "but I'll get that confirmed for you. What car model do you have?"
                ),
            },
        ],
        booking_state={"status": "active", "sink_write_status": "pending"},
    )

    assert plan.done is False
    assert plan.message
    lowered_message = plan.message.lower()
    assert "toyota" in lowered_message or "rav4" in lowered_message
    assert "premature_done_without_completed_booking" in plan.reason
