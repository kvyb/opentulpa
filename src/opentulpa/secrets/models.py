"""Public-safe secret handles and trusted ephemeral grant values."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, field_validator

from opentulpa.specs.protocol import ProtocolId, ProtocolSlug

SecretScope = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$",
    ),
]


class SecretState(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    REVOKED = "revoked"


class _SecretModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class SecretHandle(_SecretModel):
    """Model-safe reference that never contains plaintext or ciphertext."""

    tenant_id: ProtocolId
    id: ProtocolSlug
    revision: int = Field(ge=1)
    name: ProtocolSlug
    state: SecretState
    scopes: tuple[SecretScope, ...] = Field(min_length=1, max_length=100)
    created_at: datetime
    created_by: ProtocolId

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must include a UTC offset")
        return value.astimezone(UTC)


class SecretGrantReceipt(_SecretModel):
    """Auditable grant metadata safe to show without the bearer token."""

    id: ProtocolSlug
    tenant_id: ProtocolId
    secret_id: ProtocolSlug
    secret_revision: int = Field(ge=1)
    capability_id: ProtocolSlug
    scopes: tuple[SecretScope, ...] = Field(min_length=1, max_length=100)
    expires_at: datetime

    @field_validator("expires_at")
    @classmethod
    def validate_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("expires_at must include a UTC offset")
        return value.astimezone(UTC)


class IssuedSecretGrant(_SecretModel):
    """One-time bearer token returned only to the trusted capability host."""

    receipt: SecretGrantReceipt
    token: SecretStr


class SecretMaterial(_SecretModel):
    """Decrypted value available only after a scoped one-time grant is redeemed."""

    grant_id: ProtocolSlug
    secret_id: ProtocolSlug
    name: ProtocolSlug
    scope: SecretScope
    value: SecretStr


__all__ = [
    "IssuedSecretGrant",
    "SecretGrantReceipt",
    "SecretHandle",
    "SecretMaterial",
    "SecretScope",
    "SecretState",
]
