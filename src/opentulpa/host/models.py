"""Typed contracts for the stable host configuration."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class HostConfigInput(BaseModel):
    """Revisioned configuration accepted by setup and settings APIs."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    expected_revision: int | None = Field(default=None, ge=1)
    api_key: SecretStr | None = None
    base_url: str = Field(default="https://openrouter.ai/api/v1", max_length=2_000)
    model: str = Field(default="moonshotai/kimi-k3", min_length=1, max_length=300)
    telegram_bot_token: SecretStr | None = None
    telegram_user_id: int | None = Field(default=None, ge=1)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        normalized = value.rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        return normalized


class HostConfig(BaseModel):
    """Decrypted configuration available only inside the stable host."""

    model_config = ConfigDict(frozen=True)

    revision: int = Field(ge=1)
    status: Literal["staged", "active", "inactive", "failed"]
    api_key: SecretStr
    base_url: str
    model: str
    telegram_bot_token: SecretStr | None = None
    telegram_user_id: int | None = None
    internal_runtime_token: SecretStr
    telegram_pairing_code: SecretStr | None = None
    created_at: datetime
    error: str | None = None


class HostConfigView(BaseModel):
    """Secret-free host configuration returned to owner clients."""

    model_config = ConfigDict(frozen=True)

    revision: int = Field(ge=1)
    status: Literal["staged", "active", "inactive", "failed"]
    base_url: str
    model: str
    api_key_configured: bool
    telegram_configured: bool
    telegram_user_id: int | None
    telegram_pairing_required: bool
    created_at: datetime
    error: str | None = None


__all__ = ["HostConfig", "HostConfigInput", "HostConfigView"]
