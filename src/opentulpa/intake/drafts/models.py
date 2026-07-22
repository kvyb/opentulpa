"""Typed, revisioned intake workflow drafts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)


class _IntakeDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class IntakeWorkflowProposal(_IntakeDraftModel):
    """Exact active workflow configuration produced by draft preparation."""

    workflow_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    channel: Literal["instagram_dm", "telegram_business_dm"] = "instagram_dm"
    provider: Literal["composio", "telegram_bot_api"] = "composio"
    source_config: dict[str, JsonValue] = Field(default_factory=dict)
    intent_description: str = Field(min_length=1, max_length=4000)
    required_fields: list[str] = Field(min_length=1, max_length=100)
    field_guidance: dict[str, JsonValue] = Field(default_factory=dict)
    assistant_instructions: str = Field(default="", max_length=20_000)
    business_facts: dict[str, JsonValue] = Field(default_factory=dict)
    knowledge_file_ids: list[str] = Field(default_factory=list, max_length=200)
    sink_type: Literal[
        "google_sheets_composio",
        "local_csv",
        "generic_composio_write",
    ]
    sink_config: dict[str, JsonValue] = Field(default_factory=dict)
    schedule: str = Field(default="*/2 * * * *", max_length=100)
    notify_user: bool = True
    enabled: bool = True
    reply_mode: Literal["auto"] = "auto"

    @field_validator("required_fields", "knowledge_file_ids", mode="before")
    @classmethod
    def _normalize_string_list(cls, value: Any, info: ValidationInfo) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{info.field_name} must be a list")
        result: list[str] = []
        seen: set[str] = set()
        for raw in value:
            item = str(raw or "").strip()
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    @model_validator(mode="after")
    def _validate_channel_provider(self) -> IntakeWorkflowProposal:
        if self.channel == "instagram_dm" and self.provider != "composio":
            raise ValueError("instagram_dm workflows require provider=composio")
        if self.channel == "telegram_business_dm":
            if self.provider != "telegram_bot_api":
                raise ValueError("telegram_business_dm workflows require provider=telegram_bot_api")
            if not str(self.source_config.get("business_connection_id") or "").strip():
                raise ValueError(
                    "telegram_business_dm workflows require source_config.business_connection_id"
                )
        return self


IntakeDraftStatus = Literal["editing", "prepared", "activating", "activated"]


class IntakeDraft(_IntakeDraftModel):
    id: str = Field(min_length=1, max_length=100)
    tenant_id: str = Field(min_length=1, max_length=200)
    workflow_id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=1)
    status: IntakeDraftStatus
    payload: dict[str, JsonValue]
    proposal: IntakeWorkflowProposal | None = None
    proposal_hash: str | None = None
    prepared_revision: int | None = Field(default=None, ge=1)
    created_by_actor_id: str = Field(min_length=1, max_length=200)
    updated_by_actor_id: str = Field(min_length=1, max_length=200)
    created_at: datetime
    updated_at: datetime
    prepared_at: datetime | None = None
    activated_at: datetime | None = None

    @field_validator("created_at", "updated_at", "prepared_at", "activated_at")
    @classmethod
    def _require_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("draft timestamps must include a UTC offset")
        return value


class PreparedIntakeDraft(_IntakeDraftModel):
    draft: IntakeDraft
    proposal: IntakeWorkflowProposal
    proposal_hash: str
    confirmation_token: str = Field(min_length=32)


class ActivatedIntakeDraft(_IntakeDraftModel):
    draft: IntakeDraft
    workflow: dict[str, JsonValue]


__all__ = [
    "ActivatedIntakeDraft",
    "IntakeDraft",
    "IntakeDraftStatus",
    "IntakeWorkflowProposal",
    "PreparedIntakeDraft",
]
