"""Trusted live-source release builder contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class ReleaseBuildError(RuntimeError):
    """Sanitized trusted-builder failure safe for evaluation evidence."""


@dataclass(frozen=True, slots=True)
class ReleaseBuildRequest:
    """Evaluated source inputs; digest fields are raw lowercase SHA-256 hex.

    The caller must derive ``evaluation_input_sha256`` from deterministic,
    pre-artifact evaluation evidence before invoking the release builder.
    """

    candidate_id: str
    workspace: Path
    base_commit: str
    source_commit: str
    dependency_lock_hash: str | None
    evaluator_version: str
    evaluator_fingerprint: str
    evaluation_input_sha256: str | None = None


class OciReleaseArtifact(BaseModel):
    """Verified live-source release artifact plus metadata bound to its commit."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_kind: Literal["live_repo"] = "live_repo"
    image_reference: str = Field(min_length=1, max_length=300)
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ReleaseBuilder(Protocol):
    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact: ...


__all__ = [
    "OciReleaseArtifact",
    "ReleaseBuilder",
    "ReleaseBuildError",
    "ReleaseBuildRequest",
]
