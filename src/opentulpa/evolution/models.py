"""Durable event sent from the stable source controller to the serving runtime."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class EvolutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    id: str = Field(
        default_factory=lambda: f"evolution_event_{uuid4().hex}",
        min_length=1,
        max_length=100,
    )
    event_key: str = Field(min_length=1, max_length=300)
    event_type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,99}$")
    release_id: str = Field(min_length=1, max_length=100)
    origin: dict[str, JsonValue] = Field(default_factory=dict)
    payload: dict[str, JsonValue]
    status: str = Field(default="pending", pattern=r"^(?:pending|delivered)$")
    attempt_count: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    delivered_at: datetime | None = None

    @field_validator("created_at", "delivered_at")
    @classmethod
    def _utc_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evolution event timestamp must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _delivery_is_consistent(self) -> EvolutionEvent:
        if (self.status == "delivered") != (self.delivered_at is not None):
            raise ValueError("delivered evolution events require delivered_at")
        return self


__all__ = ["EvolutionEvent"]
