"""Audit identity carried across the trusted source boundary."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

_AUDIT_LIMITS = {
    "tenant_id": 200,
    "actor_id": 200,
    "thread_id": 8_192,
    "channel": 64,
    "run_kind": 64,
    "correlation_id": 8_192,
    "origin": 4_000,
    "authority": 100,
    "reason": 4_000,
}


class EvolutionAuditContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: str | None = Field(default=None, min_length=1, max_length=200)
    actor_id: str | None = Field(default=None, min_length=1, max_length=200)
    thread_id: str | None = Field(default=None, min_length=1, max_length=8_192)
    channel: str | None = Field(default=None, min_length=1, max_length=64)
    run_kind: str | None = Field(default=None, min_length=1, max_length=64)
    correlation_id: str | None = Field(default=None, min_length=1, max_length=8_192)
    origin: str | None = Field(default=None, min_length=1, max_length=4_000)
    authority: str | None = Field(default=None, min_length=1, max_length=100)
    reason: str | None = Field(default=None, min_length=1, max_length=4_000)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> Self:
        if value is None:
            return cls()
        return cls.model_validate(
            {
                key: cleaned[:limit]
                for key, limit in _AUDIT_LIMITS.items()
                if (cleaned := str(value.get(key, "") or "").strip())
            }
        )

    def as_metadata(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], self.model_dump(mode="json", exclude_none=True))


__all__ = ["EvolutionAuditContext"]
