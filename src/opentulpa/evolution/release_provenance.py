"""Typed immutable artifact provenance shared across release boundaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from opentulpa.evolution.generation import StateContract

_DEPENDENCY_BASE_FIELDS = (
    "dependency_base_id",
    "dependency_inventory_sha256",
    "dependency_resolver_fingerprint",
    "dependency_site_sha256",
    "dependency_wheelhouse_sha256",
)


class ReleaseArtifactProvenance(BaseModel):
    """The immutable artifact identity carried by evolution and bootstrap releases."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    artifact_kind: Literal["oci_image", "python_generation"]
    source_commit: str = Field(pattern=r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)
    image_reference: str = Field(min_length=1, max_length=300)
    generation_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
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
    state_contract: StateContract | None = None
    state_contract_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    install_profile: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
    )
    controller_protocol: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _coherent_artifact(self) -> Self:
        if any(not value or "\x00" in value or len(value) > 4_096 for value in self.entrypoint):
            raise ValueError("release entrypoint is invalid")
        dependency_values = tuple(getattr(self, field) for field in _DEPENDENCY_BASE_FIELDS)
        if any(value is not None for value in dependency_values) and (
            self.dependency_lock_hash is None or any(value is None for value in dependency_values)
        ):
            raise ValueError("dependency base provenance is incomplete")
        if self.state_contract is not None and (
            self.state_contract_sha256 is None
            or self.state_contract.sha256() != self.state_contract_sha256
        ):
            raise ValueError("state contract provenance is inconsistent")
        if self.artifact_kind == "python_generation":
            required = (
                self.generation_id,
                self.evaluator_fingerprint,
                self.state_contract_sha256,
                self.install_profile,
                self.controller_protocol,
            )
            if any(value is None for value in required):
                raise ValueError("Python generation provenance is incomplete")
            if (
                self.image_reference != f"python-generation:{self.generation_id}"
                or self.artifact_digest != self.manifest_digest
            ):
                raise ValueError("Python generation provenance is inconsistent")
        elif self.generation_id is not None:
            raise ValueError("OCI image provenance cannot carry a generation identity")
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

        state_digests = {
            str(value)
            for key in ("state_contract_sha256", "state_contract_digest")
            if (value := metadata.get(key)) is not None and value != ""
        }
        if len(state_digests) > 1:
            raise ValueError("state contract digest aliases disagree")
        image_reference = str(metadata.get("image_reference") or "")
        generation_id = metadata.get("generation_id")
        if generation_id is None and image_reference.startswith("python-generation:"):
            generation_id = image_reference.removeprefix("python-generation:")
        values = {
            "artifact_kind": metadata.get("artifact_kind"),
            "source_commit": source_commit,
            "artifact_digest": artifact_digest,
            "manifest_digest": manifest_digest,
            "entrypoint": entrypoint,
            "image_reference": image_reference,
            "generation_id": generation_id,
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
            "state_contract": metadata.get("state_contract"),
            "state_contract_sha256": next(iter(state_digests), None),
            "install_profile": metadata.get("install_profile"),
            "controller_protocol": metadata.get("controller_protocol"),
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


__all__ = ["ReleaseArtifactProvenance"]
