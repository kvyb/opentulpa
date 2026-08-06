"""Typed operational metadata attached to append-only evaluation reports."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class EvaluationMetadata(BaseModel):
    """Known evaluation bindings while preserving extension metadata unchanged."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_release_operation_id: str | None = Field(default=None, min_length=1, max_length=100)

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> Self:
        value = metadata.get("source_release_operation_id")
        return cls(
            source_release_operation_id=str(value) if value is not None else None,
        )

    def apply_to_metadata(self, metadata: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        rendered = dict(metadata)
        rendered.pop("source_release_operation_id", None)
        if self.source_release_operation_id is not None:
            rendered["source_release_operation_id"] = cast(
                JsonValue,
                self.source_release_operation_id,
            )
        return rendered


__all__ = ["EvaluationMetadata"]
