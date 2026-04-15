"""Pydantic models for customer profile storage and internal API payloads."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _ProfileModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CustomerScopedRequest(_ProfileModel):
    customer_id: str = Field(min_length=1)


class CustomerScopedMutationRequest(CustomerScopedRequest):
    source: str = Field(default="agent", min_length=1)


class DirectiveSetRequest(CustomerScopedMutationRequest):
    directive: str = Field(min_length=1)


class TimeProfileSetRequest(CustomerScopedMutationRequest):
    utc_offset: str = Field(pattern=r"^[+-]\d{2}:\d{2}$")


class CustomerProfileRecord(_ProfileModel):
    customer_id: str
    directive_text: str | None = None
    utc_offset: str | None = None
    locale: str | None = None
    source: str
    updated_at: str


class DirectiveGetResponse(CustomerScopedRequest):
    directive: str | None = None


class TimeProfileGetResponse(CustomerScopedRequest):
    utc_offset: str | None = None


class CustomerScopedOkResponse(_ProfileModel):
    ok: bool = True
    customer_id: str


class CustomerScopedClearResponse(CustomerScopedOkResponse):
    cleared: bool


class TimeProfileSetResponse(CustomerScopedOkResponse):
    utc_offset: str


class LegacyProfileImportSummary(_ProfileModel):
    directives: int = Field(default=0, ge=0)
    utc_offsets: int = Field(default=0, ge=0)
