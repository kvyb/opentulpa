"""Deterministic JSON serialization helpers for evolution metadata."""

from __future__ import annotations

import json

from pydantic import BaseModel


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a model or JSON value as deterministic ASCII JSON."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


__all__ = ["canonical_json_bytes"]
