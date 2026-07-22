from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from apscheduler.triggers.cron import CronTrigger  # type: ignore[import-untyped]
from pydantic import ValidationError

from opentulpa.schedules.models import AgentJob, At, Cron, Reminder, ScheduleWrite


def test_cron_requires_valid_expression_and_iana_timezone() -> None:
    trigger = Cron(timezone="Europe/Moscow", expression="0 9 * * 1-5")

    assert trigger.misfire_policy == "skip"
    assert trigger.misfire_grace_seconds == 1
    assert trigger.coalesce is True

    with pytest.raises(ValidationError, match="valid IANA timezone"):
        Cron(timezone="+03:00", expression="0 9 * * *")
    with pytest.raises(ValidationError, match="five-field cron"):
        Cron(timezone="UTC", expression="not a cron")


def test_at_requires_an_absolute_instant() -> None:
    trigger = At(timezone="UTC", run_at=datetime(2030, 1, 1, tzinfo=UTC))
    assert trigger.kind == "at"

    with pytest.raises(ValidationError, match="UTC offset"):
        At(timezone="UTC", run_at=datetime(2030, 1, 1))


def test_schedule_write_uses_discriminated_trigger_and_action() -> None:
    reminder = ScheduleWrite.model_validate(
        {
            "name": "Stand up",
            "trigger": {
                "kind": "cron",
                "timezone": "America/New_York",
                "expression": "0 9 * * 1-5",
            },
            "action": {"kind": "reminder", "message": "Time to stand up"},
        }
    )
    agent_job = ScheduleWrite.model_validate(
        {
            "name": "Briefing",
            "trigger": {
                "kind": "at",
                "timezone": "UTC",
                "run_at": "2030-01-01T09:00:00+00:00",
            },
            "action": {"kind": "agent_job", "instruction": "Prepare the briefing"},
        }
    )

    assert isinstance(reminder.trigger, Cron)
    assert isinstance(reminder.action, Reminder)
    assert isinstance(agent_job.trigger, At)
    assert isinstance(agent_job.action, AgentJob)


def test_cron_preserves_local_time_across_spring_dst_transition() -> None:
    timezone = ZoneInfo("America/New_York")
    trigger = CronTrigger.from_crontab("0 9 * * *", timezone=timezone)
    before_transition = datetime(2026, 3, 7, 10, tzinfo=timezone)

    first = trigger.get_next_fire_time(None, before_transition)
    assert first is not None
    second = trigger.get_next_fire_time(first, first)
    assert second is not None

    assert (first.month, first.day, first.hour, first.utcoffset()) == (
        3,
        8,
        9,
        timedelta(hours=-4),
    )
    assert (second.month, second.day, second.hour) == (3, 9, 9)


def test_cron_distinguishes_both_fall_dst_occurrences() -> None:
    timezone = ZoneInfo("America/New_York")
    trigger = CronTrigger.from_crontab("30 1 * * *", timezone=timezone)
    before_transition = datetime(2026, 10, 31, 3, tzinfo=timezone)

    first = trigger.get_next_fire_time(None, before_transition)
    assert first is not None
    second = trigger.get_next_fire_time(first, first)
    assert second is not None

    assert (first.month, first.day, first.hour, first.minute, first.fold) == (11, 1, 1, 30, 0)
    assert (second.month, second.day, second.hour, second.minute, second.fold) == (
        11,
        1,
        1,
        30,
        1,
    )
    assert second.astimezone(UTC).timestamp() - first.astimezone(UTC).timestamp() == 3600
