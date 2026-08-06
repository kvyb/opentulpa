"""Typed source evidence carried by durable evolution candidates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from opentulpa.evolution.generation import (
    UPSTREAM_LINEAGE_METADATA_KEY,
    UpstreamLineage,
)

VERIFIED_UPSTREAM_MERGE_COMMIT_KEY = "opentulpa.evolution.upstream_merge_commit"

_FIELD_KEYS = {
    "changed_paths": "changed_paths",
    "diff_sha256": "diff_sha256",
    "promotion_eligible": "promotion_eligible",
    "accepted_upstream_commit": "accepted_upstream_commit",
    "upstream_lineage": UPSTREAM_LINEAGE_METADATA_KEY,
    "verified_upstream_merge_commit": VERIFIED_UPSTREAM_MERGE_COMMIT_KEY,
}


class CandidateSourceProvenance(BaseModel):
    """Integrity-bearing source metadata without owning unrelated candidate metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    changed_paths: tuple[str, ...] | None = None
    diff_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    promotion_eligible: bool | None = None
    accepted_upstream_commit: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
    )
    upstream_lineage: UpstreamLineage | None = None
    verified_upstream_merge_commit: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$",
    )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Self:
        """Parse the existing flat persisted keys without consuming unknown metadata."""

        values: dict[str, object] = {}
        for field, key in _FIELD_KEYS.items():
            if key not in metadata:
                continue
            value = metadata[key]
            if field == "changed_paths" and isinstance(value, list):
                value = tuple(value)
            values[field] = value
        return cls.model_validate(values)

    def apply_to_metadata(self, metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        """Replace owned flat keys while preserving all unrelated persisted metadata."""

        rendered = dict(metadata)
        for key in _FIELD_KEYS.values():
            rendered.pop(key, None)
        values = self.model_dump(mode="json", exclude_none=True)
        for field, value in values.items():
            rendered[_FIELD_KEYS[field]] = cast(JsonValue, value)
        return rendered


__all__ = ["CandidateSourceProvenance", "VERIFIED_UPSTREAM_MERGE_COMMIT_KEY"]
