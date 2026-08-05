"""Deterministic contracts for immutable wheel and virtualenv generations."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_COMMIT_PATTERN = r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_FINGERPRINT_PATTERN = r"^sha256:[0-9a-f]{64}$"
_TOKEN_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
_CANONICAL_EXTRA_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

UPSTREAM_LINEAGE_METADATA_KEY = "opentulpa.evolution.upstream_lineage"


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


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _validate_relative_path(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("path must be non-empty and contain no NUL")
    if value.startswith("/") or value.startswith("\\") or re.match(r"^[A-Za-z]:", value):
        raise ValueError("path must be relative")
    if "\\" in value:
        raise ValueError("path must use POSIX separators")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path must have non-empty canonical components and no '..'")
    return value


def _is_beneath(path: str, parent: str) -> bool:
    path_parts = PurePosixPath(path).parts
    parent_parts = PurePosixPath(parent).parts
    return len(path_parts) > len(parent_parts) and path_parts[: len(parent_parts)] == parent_parts


class _GenerationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class UpstreamLineage(_GenerationModel):
    """Rollback-safe source lineage stored under ``UPSTREAM_LINEAGE_METADATA_KEY``."""

    upstream_commit: str | None = Field(default=None, pattern=_COMMIT_PATTERN)
    merge_base_commit: str | None = Field(default=None, pattern=_COMMIT_PATTERN)

    @model_validator(mode="after")
    def _coupled_commits(self) -> Self:
        if (self.upstream_commit is None) != (self.merge_base_commit is None):
            raise ValueError("upstream_commit and merge_base_commit must be recorded together")
        if (
            self.upstream_commit is not None
            and self.merge_base_commit is not None
            and len(self.upstream_commit) != len(self.merge_base_commit)
        ):
            raise ValueError("upstream lineage commits must use the same object format")
        return self


class StateContract(_GenerationModel):
    """Versions shared by a runtime generation and its stable controller."""

    runtime_protocol: int = Field(ge=1)
    controller_min: int = Field(ge=1)
    controller_max: int = Field(ge=1)
    product_state_schema: int = Field(ge=1)
    workspace_api: int = Field(ge=1)

    @model_validator(mode="after")
    def _compatible_protocol_range(self) -> Self:
        if self.controller_min > self.controller_max:
            raise ValueError("controller_min cannot exceed controller_max")
        if not self.controller_min <= self.runtime_protocol <= self.controller_max:
            raise ValueError("runtime_protocol must be within the controller protocol range")
        return self

    def sha256(self) -> str:
        """Return the digest of the canonical state contract."""

        return _sha256(self)


class GenerationIdentity(_GenerationModel):
    """All deterministic inputs that identify one runnable generation."""

    source_commit: str = Field(pattern=_COMMIT_PATTERN)
    source_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    wheel_sha256: str = Field(pattern=_SHA256_PATTERN)
    uv_lock_sha256: str = Field(pattern=_SHA256_PATTERN)
    evaluator_fingerprint: str = Field(pattern=_FINGERPRINT_PATTERN)
    evaluation_input_sha256: str = Field(pattern=_SHA256_PATTERN)
    python_runtime_sha256: str = Field(pattern=_SHA256_PATTERN)
    cpython_version: str = Field(
        pattern=r"^[1-9][0-9]*\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$"
    )
    cpython_cache_tag: str = Field(pattern=r"^cpython-[1-9][0-9]*$")
    cpython_abi_tag: str = Field(pattern=r"^cp[1-9][0-9]*$")
    os_name: Literal["posix"]
    platform: str = Field(min_length=1, max_length=200, pattern=_TOKEN_PATTERN)
    machine: str = Field(min_length=1, max_length=200, pattern=_TOKEN_PATTERN)
    build_recipe_version: str = Field(min_length=1, max_length=100, pattern=_TOKEN_PATTERN)
    runtime_protocol: int = Field(ge=1)
    controller_min: int = Field(ge=1)
    controller_max: int = Field(ge=1)
    state_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    install_profile: str = Field(min_length=1, max_length=100, pattern=_TOKEN_PATTERN)
    extras: tuple[str, ...] = Field(default=(), max_length=100)
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=100)
    generation_id: str = Field(default="", pattern=_SHA256_PATTERN, validate_default=False)

    @field_validator("extras")
    @classmethod
    def _canonical_extras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for extra in value:
            normalized = re.sub(r"[-_.]+", "-", extra).lower()
            if extra != normalized or not _CANONICAL_EXTRA_PATTERN.fullmatch(extra):
                raise ValueError("extras must use canonical Python package-extra names")
        if len(set(value)) != len(value):
            raise ValueError("extras must be unique")
        if value != tuple(sorted(value)):
            raise ValueError("extras must be sorted")
        return value

    @field_validator("entrypoint")
    @classmethod
    def _relative_entrypoint(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_relative_path(value[0])
        if any(not item or "\x00" in item or len(item) > 4_096 for item in value):
            raise ValueError("entrypoint values must be non-empty, bounded, and contain no NUL")
        return value

    @model_validator(mode="after")
    def _coherent_runtime(self) -> Self:
        major, minor, _ = self.cpython_version.split(".")
        version_tag = f"{major}{minor}"
        if self.cpython_cache_tag != f"cpython-{version_tag}":
            raise ValueError("CPython cache tag does not match cpython_version")
        if self.cpython_abi_tag != f"cp{version_tag}":
            raise ValueError("CPython ABI tag does not match cpython_version")
        if self.controller_min > self.controller_max:
            raise ValueError("controller_min cannot exceed controller_max")
        if not self.controller_min <= self.runtime_protocol <= self.controller_max:
            raise ValueError("runtime_protocol must be within the controller protocol range")
        return self

    @model_validator(mode="after")
    def _derive_or_verify_generation_id(self) -> Self:
        expected = _sha256(self.model_dump(mode="json", exclude={"generation_id"}))
        if not self.generation_id:
            object.__setattr__(self, "generation_id", expected)
        elif self.generation_id != expected:
            raise ValueError("generation_id does not match generation identity contents")
        return self


class GenerationDescriptor(_GenerationModel):
    """Relative artifact layout and sizes for one installed generation."""

    wheel_path: str = Field(min_length=1, max_length=4_096)
    wheel_size_bytes: int = Field(ge=1)
    uv_lock_path: str = Field(min_length=1, max_length=4_096)
    uv_lock_size_bytes: int = Field(ge=1)
    venv_path: str = Field(min_length=1, max_length=4_096)

    @field_validator("wheel_path", "uv_lock_path", "venv_path")
    @classmethod
    def _relative_paths(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def _valid_path_hierarchy(self) -> Self:
        if len({self.wheel_path, self.uv_lock_path, self.venv_path}) != 3:
            raise ValueError("generation artifact paths must be distinct")
        paths = (self.wheel_path, self.uv_lock_path, self.venv_path)
        for file_path in (self.wheel_path, self.uv_lock_path):
            if any(_is_beneath(path, file_path) for path in paths if path != file_path):
                raise ValueError("generation file paths cannot be ancestors of other paths")
        return self


class GenerationManifest(_GenerationModel):
    """Format-versioned identity and artifact metadata for one generation."""

    manifest_format: Literal[2] = 2
    identity: GenerationIdentity
    state_contract: StateContract
    descriptor: GenerationDescriptor
    runtime_tree_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _consistent_contract_and_layout(self) -> Self:
        expected = {
            "state_contract_sha256": self.state_contract.sha256(),
            "runtime_protocol": self.state_contract.runtime_protocol,
            "controller_min": self.state_contract.controller_min,
            "controller_max": self.state_contract.controller_max,
        }
        mismatches = [
            field for field, value in expected.items() if getattr(self.identity, field) != value
        ]
        if mismatches:
            raise ValueError(
                "generation identity contradicts state contract: " + ", ".join(mismatches)
            )

        executable = self.identity.entrypoint[0]
        executable_path = PurePosixPath(executable)
        venv_path = PurePosixPath(self.descriptor.venv_path)
        if executable_path.parent != venv_path / "bin":
            raise ValueError("entrypoint executable must be in the declared virtualenv")
        for file_path in (self.descriptor.wheel_path, self.descriptor.uv_lock_path):
            if (
                executable == file_path
                or _is_beneath(executable, file_path)
                or _is_beneath(file_path, executable)
            ):
                raise ValueError("generation files and the entrypoint cannot contain one another")
        return self


__all__ = [
    "GenerationDescriptor",
    "GenerationIdentity",
    "GenerationManifest",
    "StateContract",
    "UPSTREAM_LINEAGE_METADATA_KEY",
    "UpstreamLineage",
    "canonical_json_bytes",
]
