"""Typed inference contracts shared by persistence, runtime, and V2 clients."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

InferenceProvider = Literal["api", "codex"]


class _InferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class InferenceSelection(_InferenceModel):
    provider: InferenceProvider
    model: str = Field(min_length=1, max_length=300)
    reasoning_effort: str | None = Field(default=None, max_length=50)
    service_tier: str | None = Field(default=None, max_length=50)
    fallback_to_api: bool = False

    @field_validator("model", "reasoning_effort", "service_tier")
    @classmethod
    def validate_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned or any(ord(character) < 32 for character in cleaned):
            raise ValueError("inference identifier is invalid")
        return cleaned


class ResolvedInferencePlan(_InferenceModel):
    primary: InferenceSelection
    preference_revision: int = Field(ge=0)
    digest: str = Field(min_length=64, max_length=64)

    @classmethod
    def resolve(
        cls,
        selection: InferenceSelection,
        *,
        preference_revision: int,
    ) -> ResolvedInferencePlan:
        payload = {
            "primary": selection.model_dump(mode="json"),
            "preference_revision": preference_revision,
        }
        canonical = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return cls(
            primary=selection,
            preference_revision=preference_revision,
            digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )


class InferenceServiceTier(_InferenceModel):
    id: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class InferenceModel(_InferenceModel):
    provider: InferenceProvider
    id: str = Field(min_length=1, max_length=300)
    reasoning_efforts: tuple[str, ...] = ()
    default_reasoning_effort: str | None = None
    service_tiers: tuple[InferenceServiceTier, ...] = ()
    default_service_tier: str | None = None


__all__ = [
    "InferenceModel",
    "InferenceProvider",
    "InferenceSelection",
    "InferenceServiceTier",
    "ResolvedInferencePlan",
]
