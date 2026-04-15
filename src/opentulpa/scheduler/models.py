"""Models for routines and one-off scheduled items."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class Routine:
    """Recurring or one-off scheduled item that can wake the agent."""

    id: str
    name: str
    # Cron expression or next run time; for one-off, run once then disable
    schedule: str  # e.g. "0 9 * * *" (9am daily) or "2025-02-18 10:00:00"
    payload: dict[str, Any]  # e.g. {"instruction": "You must ...", "action": "wake"}
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    is_cron: bool = True  # False = one-off at schedule time
