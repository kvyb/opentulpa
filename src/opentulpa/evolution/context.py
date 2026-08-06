"""Typed audit and source-session context carried across evolution boundaries."""

from __future__ import annotations

import hashlib
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
    """Sanitized owner and conversation identity for one evolution operation."""

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
        sanitized = {
            key: cleaned[:limit]
            for key, limit in _AUDIT_LIMITS.items()
            if (cleaned := str(value.get(key, "") or "").strip())
        }
        return cls.model_validate(sanitized)

    def as_metadata(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], self.model_dump(mode="json", exclude_none=True))

    def required_source_identity(self) -> tuple[str, str]:
        if self.tenant_id is None:
            raise ValueError("source session context is incomplete")
        session_key = hashlib.sha256(self.tenant_id.encode("utf-8")).hexdigest()
        return session_key, self.tenant_id


class SourceSessionContext(BaseModel):
    """Backward-compatible typed view of flat candidate session metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    active: bool = False
    session_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tenant_id: str | None = Field(default=None, min_length=1, max_length=200)
    requested_by: EvolutionAuditContext | None = None

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Self:
        requested_by = metadata.get("requested_by")
        return cls(
            active=metadata.get("source_session") is True,
            session_key=(
                str(metadata["source_session_key"])
                if metadata.get("source_session_key") is not None
                else None
            ),
            tenant_id=(
                str(metadata["source_tenant_id"])
                if metadata.get("source_tenant_id") is not None
                else None
            ),
            requested_by=(
                EvolutionAuditContext.from_mapping(requested_by)
                if isinstance(requested_by, Mapping)
                else None
            ),
        )

    @classmethod
    def create(cls, audit: EvolutionAuditContext) -> Self:
        session_key, tenant_id = audit.required_source_identity()
        return cls(
            active=True,
            session_key=session_key,
            tenant_id=tenant_id,
            requested_by=audit,
        )

    def matches(self, *, session_key: str, tenant_id: str) -> bool:
        return self.active and (
            self.session_key == session_key
            or self.tenant_id == tenant_id
            or (self.requested_by is not None and self.requested_by.tenant_id == tenant_id)
        )

    def apply_to_metadata(self, metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        rendered = dict(metadata)
        for key in ("source_session", "source_session_key", "source_tenant_id", "requested_by"):
            rendered.pop(key, None)
        if not self.active:
            return rendered
        rendered["source_session"] = True
        if self.session_key is not None:
            rendered["source_session_key"] = self.session_key
        if self.tenant_id is not None:
            rendered["source_tenant_id"] = self.tenant_id
        if self.requested_by is not None:
            rendered["requested_by"] = self.requested_by.as_metadata()
        return rendered


__all__ = ["EvolutionAuditContext", "SourceSessionContext"]
