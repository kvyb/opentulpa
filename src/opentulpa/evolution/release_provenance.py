"""Typed immutable artifact provenance shared across release boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

_DEPENDENCY_BASE_FIELDS = (
    "dependency_base_id",
    "dependency_inventory_sha256",
    "dependency_resolver_fingerprint",
    "dependency_site_sha256",
    "dependency_wheelhouse_sha256",
)


def live_repo_artifact_digest(source_commit: str) -> str:
    """Deterministic digest for a release whose artifact is the Git commit itself."""

    return f"sha256:{hashlib.sha256(f'live-repo:{source_commit}'.encode('ascii')).hexdigest()}"


class ReleaseArtifactProvenance(BaseModel):
    """The immutable artifact identity carried by evolution and bootstrap releases."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_kind: Literal["live_repo"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)
    image_reference: str = Field(min_length=1, max_length=300)
    dependency_lock_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dependency_base_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dependency_inventory_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    dependency_resolver_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    dependency_site_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    dependency_wheelhouse_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evaluation_input_digest: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    evaluator_fingerprint: str | None = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def _coherent_artifact(self) -> Self:
        if any(not value or "\x00" in value or len(value) > 4_096 for value in self.entrypoint):
            raise ValueError("release entrypoint is invalid")
        dependency_values = tuple(getattr(self, field) for field in _DEPENDENCY_BASE_FIELDS)
        if any(value is not None for value in dependency_values) and (
            self.dependency_lock_hash is None or any(value is None for value in dependency_values)
        ):
            raise ValueError("dependency base provenance is incomplete")
        expected_digest = live_repo_artifact_digest(self.source_commit)
        if (
            self.image_reference != f"git-commit:{self.source_commit}"
            or self.artifact_digest != self.manifest_digest
            or self.artifact_digest != expected_digest
        ):
            raise ValueError("live repo provenance is inconsistent")
        return self

    @classmethod
    def from_values(
        cls,
        *,
        source_commit: str,
        artifact_digest: str,
        manifest_digest: str,
        entrypoint: tuple[str, ...],
        metadata: Mapping[str, object],
    ) -> Self:
        """Parse flat persisted release fields into the canonical typed contract."""

        image_reference = str(metadata.get("image_reference") or "")
        values = {
            "artifact_kind": metadata.get("artifact_kind"),
            "source_commit": source_commit,
            "artifact_digest": artifact_digest,
            "manifest_digest": manifest_digest,
            "entrypoint": entrypoint,
            "image_reference": image_reference,
            "dependency_lock_hash": metadata.get("dependency_lock_hash"),
            "dependency_base_id": metadata.get("dependency_base_id"),
            "dependency_inventory_sha256": metadata.get("dependency_inventory_sha256"),
            "dependency_resolver_fingerprint": metadata.get(
                "dependency_resolver_fingerprint"
            ),
            "dependency_site_sha256": metadata.get("dependency_site_sha256"),
            "dependency_wheelhouse_sha256": metadata.get("dependency_wheelhouse_sha256"),
            "evaluation_input_digest": metadata.get("evaluation_input_digest"),
            "evaluator_fingerprint": metadata.get("evaluator_fingerprint"),
        }
        return cls.model_validate(values)

    def release_metadata(self) -> dict[str, JsonValue]:
        """Render the canonical flat metadata persisted by release models."""

        values = self.model_dump(
            mode="json",
            exclude={"source_commit", "artifact_digest", "manifest_digest", "entrypoint"},
            exclude_none=True,
        )
        return cast(dict[str, JsonValue], values)


__all__ = ["ReleaseArtifactProvenance", "live_repo_artifact_digest"]
