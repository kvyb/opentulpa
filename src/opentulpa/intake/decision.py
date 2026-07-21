"""Typed output contract for the restricted intake Deep Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _IntakeDecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class BookingPatch(_IntakeDecisionModel):
    """Proposed booking changes; the deterministic applier owns all persistence."""

    booking_id: str | None = Field(default=None, max_length=200)
    fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list, max_length=50)
    status: Literal["open", "completed", "cancelled"] | None = None


class IntakeDecision(_IntakeDecisionModel):
    action: Literal["ignore", "reply", "request_fields", "propose_booking", "escalate"]
    reply_text: str | None = Field(default=None, max_length=10_000)
    booking_patch: BookingPatch | None = None
    evidence_source_ids: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_action_payload(self) -> IntakeDecision:
        if self.action == "ignore" and (self.reply_text or self.booking_patch is not None):
            raise ValueError("ignore decisions cannot include a reply or booking patch")
        if self.action in {"reply", "request_fields"} and not self.reply_text:
            raise ValueError(f"{self.action} decisions require reply_text")
        if self.action in {"request_fields", "propose_booking"} and self.booking_patch is None:
            raise ValueError(f"{self.action} decisions require booking_patch")
        return self


__all__ = ["BookingPatch", "IntakeDecision"]
