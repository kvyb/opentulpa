"""Trusted OCI and fixed-recipe Python release builders."""

from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import platform as platform_module
import re
import shutil
import stat
import sys
import sysconfig
import time
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from opentulpa.bootstrap.oci_host import (
    LocalOciCommandRunner,
    OciCommandRunner,
)
from opentulpa.evolution.dependency_resolver import DependencyResolver, ResolvedDependencyBase
from opentulpa.evolution.generation import (
    GenerationDescriptor,
    GenerationIdentity,
    GenerationManifest,
    StateContract,
    canonical_json_bytes,
    generation_manifest_sha256,
)
from opentulpa.evolution.generation_store import (
    GenerationStore,
    GenerationStoreError,
)
from opentulpa.evolution.git_security import (
    GitSecurityError,
    RepositoryMutationLockError,
    _repository_config_is_safe,
    discover_git_directories,
    repository_mutation_lock,
    run_hardened_git,
)
from opentulpa.evolution.process import BoundedProcessResult, run_bounded_process
from opentulpa.evolution.workspace import (
    candidate_content_contains_secret,
    candidate_path_is_promotable,
    candidate_path_is_runtime_overlay,
)

_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_LOCK_HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAG_PREFIX_RE = re.compile(r"[a-z0-9][a-z0-9._/-]{0,127}\Z")
_SENSITIVE_BASENAMES = frozenset(
    {
        ".env",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_SENSITIVE_SUFFIXES = frozenset({".key", ".pem", ".p12", ".pfx"})
_PUBLIC_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
_MANIFEST_LABEL = "org.opentulpa.release.manifest-digest"
_SOURCE_LABEL = "org.opentulpa.release.source-commit"
_PROTOCOL_LABEL = "org.opentulpa.release.protocol-version"
_SOURCE_LAYOUT_LABEL = "org.opentulpa.release.source-layout"
_SOURCE_LAYOUT_VERSION = "full-source-v1"
_RECIPE_VERSION = "full-source-v1"
_WHEEL_RECIPE_VERSION = "fixed-wheel-v2"
_DOCKERIGNORE_STAGING = ".opentulpa-candidate-dockerignore"
_GENERATION_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DEFAULT_PACKAGING_METADATA_PATHS = (
    "MANIFEST.in",
    "hatch.toml",
    "hatch_build.py",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
    "uv.toml",
)
_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED = "trusted_runtime_base_rebuild_required"


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
    """Verified release artifact plus the manifest bound to its source."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_kind: Literal["oci_image", "python_generation"] = "oci_image"
    image_reference: str = Field(min_length=1, max_length=300)
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)


class ReleaseBuilder(Protocol):
    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact: ...


class BoundedCommandRunner(Protocol):
    """Injectable synchronous runner used by the Python generation builder."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> BoundedProcessResult: ...


@dataclass(frozen=True, slots=True)
class OciReleaseBuildPolicy:
    base_image_digest: str
    base_dependency_lock_hash: str
    container_cli: str = "docker"
    git_cli: str = "git"
    state_root: Path = Path(".opentulpa/release-builds")
    image_tag_prefix: str = "opentulpa-release"
    entrypoint: tuple[str, ...] = (
        "/opt/opentulpa-install/controller/generations/image/bin/python",
        "-P",
        "-m",
        "opentulpa",
    )
    timeout_seconds: int = 1_800
    max_output_bytes: int = 1_000_000
    max_context_bytes: int = 512 * 1024 * 1024
    max_context_entries: int = 100_000

    def __post_init__(self) -> None:
        engine = Path(self.container_cli).name
        if engine not in {"docker", "podman"} or "\x00" in self.container_cli:
            raise ValueError("container_cli must be a Docker or Podman executable")
        if Path(self.git_cli).name != "git" or "\x00" in self.git_cli:
            raise ValueError("git_cli must be a Git executable")
        if not _DIGEST_RE.fullmatch(self.base_image_digest):
            raise ValueError("base_image_digest must be an immutable local OCI image ID")
        if not _LOCK_HASH_RE.fullmatch(self.base_dependency_lock_hash):
            raise ValueError("base_dependency_lock_hash must be a SHA-256 lockfile hash")
        if not _TAG_PREFIX_RE.fullmatch(self.image_tag_prefix):
            raise ValueError("image_tag_prefix is invalid")
        if not self.entrypoint or any(
            not item or "\x00" in item or len(item) > 4_096 for item in self.entrypoint
        ):
            raise ValueError("entrypoint must contain safe exec arguments")
        if not 60 <= self.timeout_seconds <= 86_400:
            raise ValueError("release build timeout must be between 60 and 86400 seconds")
        if self.max_output_bytes < 1_024:
            raise ValueError("release build output limit is too small")
        if self.max_context_bytes < 1_024 * 1_024:
            raise ValueError("release build context limit is too small")
        if self.max_context_entries < 100:
            raise ValueError("release build entry limit is too small")


@dataclass(frozen=True, slots=True)
class WheelReleaseBuildPolicy:
    """Trusted fixed-recipe inputs, including external CPython/system-library policy.

    ``generations_root`` must be a dedicated runtime-generations directory outside
    private bootstrap/controller state so an unprivileged child can traverse it.
    """

    generations_root: Path
    base_dependency_lock_hash: str
    state_contract: StateContract
    trusted_metadata_hashes: Mapping[str, str]
    trusted_wheelhouse: Path
    external_python_runtime_policy_sha256: str
    build_root: Path | None = None
    python_executable: str = sys.executable
    uv_cli: str = "uv"
    git_cli: str = "git"
    entrypoint: tuple[str, ...] = ("venv/bin/python", "-I", "-m", "opentulpa")
    extras: tuple[str, ...] = ()
    install_profile: str = "runtime"
    import_name: str = "opentulpa"
    resource_paths: tuple[str, ...] = ("resources/release_contract.json",)
    packaging_metadata_paths: tuple[str, ...] = _DEFAULT_PACKAGING_METADATA_PATHS
    package_roots: tuple[str, ...] = ("src/opentulpa",)
    trusted_bridge_assets: tuple[tuple[str, str, str], ...] = ()
    timeout_seconds: int = 1_800
    max_output_bytes: int = 1_000_000
    max_source_bytes: int = 512 * 1024 * 1024
    max_source_entries: int = 100_000
    max_wheel_bytes: int = 256 * 1024 * 1024
    build_recipe_version: str = _WHEEL_RECIPE_VERSION
    dependency_base_id: str | None = None

    def __post_init__(self) -> None:
        if not _LOCK_HASH_RE.fullmatch(self.base_dependency_lock_hash):
            raise ValueError("base_dependency_lock_hash must be a SHA-256 lockfile hash")
        if not isinstance(self.state_contract, StateContract):
            raise ValueError("state_contract must be a trusted StateContract")
        if not _LOCK_HASH_RE.fullmatch(self.external_python_runtime_policy_sha256):
            raise ValueError("external Python runtime policy digest is invalid")
        if Path(self.git_cli).name != "git" or "\x00" in self.git_cli:
            raise ValueError("git_cli must be a Git executable")
        if Path(self.uv_cli).name != "uv" or "\x00" in self.uv_cli:
            raise ValueError("uv_cli must be a uv executable")
        if not self.python_executable or "\x00" in self.python_executable:
            raise ValueError("python_executable is invalid")
        if not 1 <= self.timeout_seconds <= 86_400:
            raise ValueError("generation build timeout must be between 1 and 86400 seconds")
        if self.max_output_bytes < 1_024:
            raise ValueError("generation build output limit is too small")
        if self.max_source_bytes < 1024 * 1024:
            raise ValueError("generation source limit is too small")
        if self.max_source_entries < 100:
            raise ValueError("generation source entry limit is too small")
        if self.max_wheel_bytes < 1024:
            raise ValueError("generation wheel limit is too small")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", self.import_name):
            raise ValueError("generation smoke import name is invalid")
        if not self.entrypoint or any(
            not item or "\x00" in item or len(item) > 4_096 for item in self.entrypoint
        ):
            raise ValueError("entrypoint must contain safe exec arguments")
        executable = PurePosixPath(self.entrypoint[0])
        if executable.is_absolute() or executable.parent != PurePosixPath("venv/bin"):
            raise ValueError("entrypoint executable must be inside venv/bin")
        if self.entrypoint[:4] != ("venv/bin/python", "-I", "-m", self.import_name):
            raise ValueError("generation entrypoint must invoke the package with isolated Python")
        if not self.build_recipe_version or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.build_recipe_version
        ):
            raise ValueError("build_recipe_version is invalid")
        if self.dependency_base_id is not None and not _LOCK_HASH_RE.fullmatch(
            self.dependency_base_id
        ):
            raise ValueError("dependency_base_id is invalid")
        if not self.install_profile or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.install_profile
        ):
            raise ValueError("install_profile is invalid")
        if self.extras != tuple(sorted(set(self.extras))) or any(
            extra != re.sub(r"[-_.]+", "-", extra).lower()
            or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", extra) is None
            for extra in self.extras
        ):
            raise ValueError("extras must be sorted, unique, canonical package names")
        self._validate_relative_paths(self.resource_paths, label="resource")
        self._validate_relative_paths(self.packaging_metadata_paths, label="packaging metadata")
        self._validate_relative_paths(self.package_roots, label="package root")
        if "pyproject.toml" not in self.packaging_metadata_paths:
            raise ValueError("packaging metadata must include pyproject.toml")
        if len(set(self.packaging_metadata_paths)) != len(self.packaging_metadata_paths):
            raise ValueError("packaging metadata paths must be unique")
        metadata_hashes = dict(self.trusted_metadata_hashes)
        if "pyproject.toml" not in metadata_hashes:
            raise ValueError("trusted metadata hashes must pin pyproject.toml")
        self._validate_relative_paths(tuple(metadata_hashes), label="trusted metadata")
        if any(not _LOCK_HASH_RE.fullmatch(digest) for digest in metadata_hashes.values()):
            raise ValueError("trusted metadata hash is invalid")
        object.__setattr__(self, "trusted_metadata_hashes", MappingProxyType(metadata_hashes))
        bridge_sources: set[str] = set()
        bridge_destinations: set[str] = set()
        for source, destination, digest in self.trusted_bridge_assets:
            self._validate_relative_paths((source, destination), label="trusted bridge asset")
            if (
                source in bridge_sources
                or destination in bridge_destinations
                or not _LOCK_HASH_RE.fullmatch(digest)
                or any(
                    part.endswith(".dist-info") for part in PurePosixPath(destination).parts
                )
            ):
                raise ValueError("trusted bridge asset declaration is invalid")
            bridge_sources.add(source)
            bridge_destinations.add(destination)

    @staticmethod
    def _validate_relative_paths(paths: tuple[str, ...], *, label: str) -> None:
        if not paths:
            raise ValueError(f"{label} paths cannot be empty")
        for raw_path in paths:
            path = PurePosixPath(raw_path)
            if (
                "\x00" in raw_path
                or "\\" in raw_path
                or re.match(r"^[A-Za-z]:", raw_path)
                or path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
            ):
                raise ValueError(f"{label} path is unsafe")


# The explicit trusted name is retained as a descriptive alias for callers.
TrustedWheelBuildPolicy = WheelReleaseBuildPolicy


def _export_exact_git_blobs(
    *,
    runner: BoundedCommandRunner,
    git_cli: str,
    workspace: Path,
    source_commit: str,
    destination_root: Path,
    command_cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    max_bytes: int,
    max_entries: int,
    validate_path: Callable[[PurePosixPath], None],
    secret_scan_paths: frozenset[str],
    destination_path: Callable[[PurePosixPath], PurePosixPath] | None = None,
) -> str:
    """Materialize regular blobs from one commit without archive/worktree filters."""

    listing = runner(
        (
            git_cli,
            "-C",
            str(workspace),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            source_commit,
        ),
        cwd=command_cwd,
        env=environment,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_bytes,
    )
    if listing.returncode != 0 or listing.timed_out or listing.truncated:
        raise ReleaseBuildError("candidate commit tree could not be inspected safely")
    try:
        destination_root.mkdir(parents=True, exist_ok=False, mode=0o700)
    except OSError as exc:
        raise ReleaseBuildError("candidate build context could not be created") from exc
    digest = hashlib.sha256()
    total_bytes = 0
    entries = 0
    materialized_paths: set[str] = set()
    for raw_entry in listing.output.split(b"\0"):
        if not raw_entry:
            continue
        entries += 1
        if entries > max_entries:
            raise ReleaseBuildError("candidate build context has too many entries")
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, object_id = header.split(b" ", 2)
            path_text = raw_path.decode("utf-8")
            object_id_text = object_id.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReleaseBuildError("candidate commit tree entry was invalid") from exc
        if object_type != b"blob" or mode not in {b"100644", b"100755"}:
            raise ReleaseBuildError("candidate build context contains a link or special file")
        relative = PurePosixPath(path_text)
        validate_path(relative)
        remaining = max_bytes - total_bytes
        blob = runner(
            (git_cli, "-C", str(workspace), "cat-file", "blob", object_id_text),
            cwd=command_cwd,
            env=environment,
            timeout_seconds=timeout_seconds,
            max_output_bytes=remaining + 1,
        )
        if (
            blob.returncode != 0
            or blob.timed_out
            or blob.truncated
            or len(blob.output) > remaining
        ):
            raise ReleaseBuildError("candidate build context exceeds its byte limit")
        if relative.as_posix() in secret_scan_paths and candidate_content_contains_secret(
            relative.as_posix(), blob.output
        ):
            raise ReleaseBuildError("candidate build context contains credential material")
        total_bytes += len(blob.output)
        digest.update(mode + b"\0" + raw_path + b"\0" + object_id + b"\0")
        digest.update(blob.output)
        digest.update(b"\0")
        materialized = destination_path(relative) if destination_path is not None else relative
        normalized_materialized = materialized.as_posix().casefold()
        if normalized_materialized in materialized_paths:
            raise ReleaseBuildError("candidate build context contains colliding paths")
        materialized_paths.add(normalized_materialized)
        destination = destination_root.joinpath(*materialized.parts)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(destination, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(blob.output)
            destination.chmod(0o755 if mode == b"100755" else 0o644)
        except OSError as exc:
            raise ReleaseBuildError("candidate build context could not be materialized") from exc
    return digest.hexdigest()


class TrustedOciReleaseBuilder:
    """Overlay an evaluated source tree onto a trusted immutable dependency image."""

    def __init__(
        self,
        *,
        policy: OciReleaseBuildPolicy,
        runner: OciCommandRunner | None = None,
    ) -> None:
        self._policy = policy
        self._state_root = self._policy.state_root.expanduser().resolve()
        self._state_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._runner = runner or LocalOciCommandRunner(cwd=self._state_root)

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        workspace = request.workspace.expanduser().resolve(strict=True)
        self._validate_request(request, workspace)
        await asyncio.to_thread(self._verify_exact_commit, request, workspace)
        changed_paths = await asyncio.to_thread(self._verify_promotable_diff, request, workspace)

        build_id = uuid4().hex
        context_root = self._state_root / "contexts" / build_id
        iid_path = self._state_root / "image-ids" / f"{build_id}.txt"
        recipe_path = self._state_root / "recipes" / f"{build_id}.Dockerfile"
        for parent in (
            context_root.parent,
            iid_path.parent,
            recipe_path.parent,
        ):
            parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            tree_digest = await asyncio.to_thread(
                self._export_commit,
                request,
                workspace,
                context_root,
                frozenset(changed_paths),
            )
            manifest_digest = self._manifest_digest(request, tree_digest=tree_digest)
            tag = f"{self._policy.image_tag_prefix}:{manifest_digest.removeprefix('sha256:')[:32]}"
            await self._require_rootless_engine()
            await self._verify_base_image()
            await asyncio.to_thread(self._write_trusted_recipe, recipe_path)
            await self._build_image(
                context_root=context_root,
                iid_path=iid_path,
                recipe_path=recipe_path,
                tag=tag,
                source_commit=request.source_commit,
                manifest_digest=manifest_digest,
            )
            artifact_digest = self._read_image_id(iid_path)
            await self._verify_image(
                image_reference=artifact_digest,
                artifact_digest=artifact_digest,
                source_commit=request.source_commit,
                manifest_digest=manifest_digest,
            )
            return OciReleaseArtifact(
                artifact_digest=artifact_digest,
                manifest_digest=manifest_digest,
                image_reference=tag,
                entrypoint=self._policy.entrypoint,
            )
        finally:
            shutil.rmtree(context_root, ignore_errors=True)
            iid_path.unlink(missing_ok=True)
            recipe_path.unlink(missing_ok=True)

    def _validate_request(self, request: ReleaseBuildRequest, workspace: Path) -> None:
        if not request.candidate_id or len(request.candidate_id) > 100:
            raise ReleaseBuildError("candidate build identity is invalid")
        if not workspace.is_dir() or workspace.is_symlink():
            raise ReleaseBuildError("candidate build workspace is invalid")
        if not _COMMIT_RE.fullmatch(request.base_commit) or not _COMMIT_RE.fullmatch(
            request.source_commit
        ):
            raise ReleaseBuildError("candidate source commit is invalid")
        if not _DIGEST_RE.fullmatch(request.evaluator_fingerprint):
            raise ReleaseBuildError("candidate evaluator identity is invalid")
        if not request.evaluator_version.strip():
            raise ReleaseBuildError("candidate evaluator version is invalid")
        if request.dependency_lock_hash != self._policy.base_dependency_lock_hash:
            raise ReleaseBuildError(
                "candidate dependency lock changed; a trusted runtime base rebuild is required"
            )

    def _verify_exact_commit(self, request: ReleaseBuildRequest, workspace: Path) -> None:
        environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"}
        head = run_bounded_process(
            (self._policy.git_cli, "-C", str(workspace), "rev-parse", "--verify", "HEAD^{commit}"),
            cwd=self._state_root,
            env=environment,
            timeout_seconds=30,
            max_output_bytes=1_024,
        )
        resolved = head.output.decode("ascii", errors="ignore").strip().lower()
        if head.returncode != 0 or head.truncated or resolved != request.source_commit:
            raise ReleaseBuildError("candidate workspace no longer matches its evaluated commit")
        status_result = run_bounded_process(
            (
                self._policy.git_cli,
                "-C",
                str(workspace),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            cwd=self._state_root,
            env=environment,
            timeout_seconds=30,
            max_output_bytes=64 * 1_024,
        )
        if status_result.returncode != 0 or status_result.truncated or status_result.output.strip():
            raise ReleaseBuildError("candidate workspace changed after evaluation")

    def _verify_promotable_diff(
        self,
        request: ReleaseBuildRequest,
        workspace: Path,
    ) -> tuple[str, ...]:
        environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"}
        ancestor = run_bounded_process(
            (
                self._policy.git_cli,
                "-C",
                str(workspace),
                "merge-base",
                "--is-ancestor",
                request.base_commit,
                request.source_commit,
            ),
            cwd=self._state_root,
            env=environment,
            timeout_seconds=30,
            max_output_bytes=1_024,
        )
        if ancestor.returncode != 0 or ancestor.truncated:
            raise ReleaseBuildError("candidate base commit is not an ancestor of evaluated source")
        changed = run_bounded_process(
            (
                self._policy.git_cli,
                "-C",
                str(workspace),
                "diff",
                "--name-only",
                "--no-ext-diff",
                "--diff-filter=ACDMRTUXB",
                "-z",
                request.base_commit,
                request.source_commit,
                "--",
            ),
            cwd=self._state_root,
            env=environment,
            timeout_seconds=30,
            max_output_bytes=256 * 1_024,
        )
        if changed.returncode != 0 or changed.truncated:
            raise ReleaseBuildError("candidate source change set could not be verified")
        paths = tuple(
            path for path in changed.output.decode("utf-8", errors="replace").split("\0") if path
        )
        if any(not candidate_path_is_promotable(path) for path in paths):
            raise ReleaseBuildError(
                "candidate changes are contribution-only and cannot enter a production release"
            )
        return paths

    def _export_commit(
        self,
        request: ReleaseBuildRequest,
        workspace: Path,
        context_root: Path,
        secret_scan_paths: frozenset[str],
    ) -> str:
        environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"}
        digest = _export_exact_git_blobs(
            runner=run_bounded_process,
            git_cli=self._policy.git_cli,
            workspace=workspace,
            source_commit=request.source_commit,
            destination_root=context_root,
            command_cwd=self._state_root,
            environment=environment,
            timeout_seconds=60,
            max_bytes=self._policy.max_context_bytes,
            max_entries=self._policy.max_context_entries,
            validate_path=self._validate_source_path,
            secret_scan_paths=secret_scan_paths,
            destination_path=lambda relative: (
                PurePosixPath(_DOCKERIGNORE_STAGING)
                if relative.as_posix() == ".dockerignore"
                else relative
            ),
        )
        # Candidate Docker ignore rules are restored inside the image, but cannot
        # alter which exact Git blobs enter the trusted build context.
        (context_root / ".dockerignore").write_bytes(b"")
        return digest

    async def _verify_base_image(self) -> None:
        result = await self._runner.run(
            (
                self._policy.container_cli,
                "image",
                "inspect",
                self._policy.base_image_digest,
                "--format",
                "{{json .}}",
            ),
            timeout_seconds=30,
            max_output_bytes=256 * 1_024,
        )
        if result.returncode != 0 or result.truncated:
            raise ReleaseBuildError("trusted runtime base image is unavailable")
        try:
            document: Any = json.loads(result.output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseBuildError("trusted runtime base image inspection was invalid") from exc
        if (
            not isinstance(document, dict)
            or str(document.get("Id") or "").lower() != self._policy.base_image_digest
        ):
            raise ReleaseBuildError("trusted runtime base image identity changed")

    def _write_trusted_recipe(self, recipe_path: Path) -> None:
        recipe = (
            f"FROM {self._policy.base_image_digest}\n"
            "USER root\n"
            "RUN find /app -mindepth 1 -maxdepth 1 -exec rm -rf -- '{}' +\n"
            "COPY --chown=65532:65532 . /app/\n"
            f"RUN if [ -f /app/{_DOCKERIGNORE_STAGING} ]; then "
            f"mv /app/{_DOCKERIGNORE_STAGING} /app/.dockerignore; "
            "else rm -f /app/.dockerignore; fi\n"
            "WORKDIR /app\n"
            "USER 65532:65532\n"
        )
        recipe_path.write_text(recipe, encoding="ascii")
        recipe_path.chmod(0o600)

    def _validate_source_path(self, relative: PurePosixPath) -> None:
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", "..", ".git"} for part in relative.parts)
        ):
            raise ReleaseBuildError("candidate build context contains an unsafe path")
        if self._is_sensitive(relative):
            raise ReleaseBuildError("candidate build context contains a forbidden secret path")
        if relative.parts[0] == _DOCKERIGNORE_STAGING or (
            relative.parts[0] == ".dockerignore" and relative.as_posix() != ".dockerignore"
        ):
            raise ReleaseBuildError("candidate build context contains a reserved path")
        if not candidate_path_is_runtime_overlay(relative.as_posix()):
            raise ReleaseBuildError(
                "candidate changes are contribution-only and cannot enter a production release"
            )

    @staticmethod
    def _is_sensitive(path: PurePosixPath) -> bool:
        for component in path.parts:
            lowered = component.casefold()
            if lowered in _PUBLIC_ENV_TEMPLATES:
                continue
            if (
                lowered in _SENSITIVE_BASENAMES
                or lowered.startswith(".env.")
                or Path(lowered).suffix in _SENSITIVE_SUFFIXES
            ):
                return True
        return False

    async def _require_rootless_engine(self) -> None:
        engine = Path(self._policy.container_cli).name
        if engine == "podman":
            argv = (
                self._policy.container_cli,
                "info",
                "--format",
                "{{.Host.Security.Rootless}}",
            )
            expected = "true"
        else:
            argv = (
                self._policy.container_cli,
                "info",
                "--format",
                "{{json .SecurityOptions}}",
            )
            expected = "name=rootless"
        result = await self._runner.run(
            argv,
            timeout_seconds=30,
            max_output_bytes=16 * 1_024,
        )
        output = result.output.decode("utf-8", errors="ignore").casefold()
        if result.returncode != 0 or result.truncated or expected not in output:
            raise ReleaseBuildError("a rootless OCI builder is required")

    async def _build_image(
        self,
        *,
        context_root: Path,
        iid_path: Path,
        recipe_path: Path,
        tag: str,
        source_commit: str,
        manifest_digest: str,
    ) -> None:
        pull_policy = (
            "--pull=never" if Path(self._policy.container_cli).name == "podman" else "--pull=false"
        )
        argv = (
            self._policy.container_cli,
            "build",
            pull_policy,
            "--network=none",
            "--no-cache",
            "--iidfile",
            str(iid_path),
            "--tag",
            tag,
            "--label",
            f"{_MANIFEST_LABEL}={manifest_digest}",
            "--label",
            f"{_SOURCE_LABEL}={source_commit}",
            "--label",
            f"{_PROTOCOL_LABEL}=1",
            "--label",
            f"{_SOURCE_LAYOUT_LABEL}={_SOURCE_LAYOUT_VERSION}",
            "--file",
            str(recipe_path),
            str(context_root),
        )
        result = await self._runner.run(
            argv,
            timeout_seconds=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
        )
        if result.returncode != 0 or result.timed_out or result.truncated:
            raise ReleaseBuildError("candidate OCI image build failed")

    @staticmethod
    def _read_image_id(iid_path: Path) -> str:
        try:
            image_id = iid_path.read_text(encoding="ascii").strip().lower()
        except (OSError, UnicodeError) as exc:
            raise ReleaseBuildError("candidate OCI builder did not return an image ID") from exc
        if not _DIGEST_RE.fullmatch(image_id):
            raise ReleaseBuildError("candidate OCI builder returned an invalid image ID")
        return image_id

    async def _verify_image(
        self,
        *,
        image_reference: str,
        artifact_digest: str,
        source_commit: str,
        manifest_digest: str,
    ) -> None:
        result = await self._runner.run(
            (
                self._policy.container_cli,
                "image",
                "inspect",
                image_reference,
                "--format",
                "{{json .}}",
            ),
            timeout_seconds=30,
            max_output_bytes=256 * 1_024,
        )
        if result.returncode != 0 or result.truncated:
            raise ReleaseBuildError("candidate OCI image could not be inspected")
        try:
            document: Any = json.loads(result.output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseBuildError("candidate OCI image inspection was invalid") from exc
        if (
            not isinstance(document, dict)
            or str(document.get("Id") or "").lower() != artifact_digest
        ):
            raise ReleaseBuildError("candidate OCI image ID changed after build")
        config = document.get("Config")
        labels = config.get("Labels") if isinstance(config, dict) else None
        labels = labels if isinstance(labels, dict) else {}
        required = {
            _MANIFEST_LABEL: manifest_digest,
            _SOURCE_LABEL: source_commit,
            _PROTOCOL_LABEL: "1",
            _SOURCE_LAYOUT_LABEL: _SOURCE_LAYOUT_VERSION,
        }
        if any(str(labels.get(key) or "") != value for key, value in required.items()):
            raise ReleaseBuildError("candidate OCI image labels failed verification")

    def _manifest_digest(self, request: ReleaseBuildRequest, *, tree_digest: str) -> str:
        payload = {
            "base_commit": request.base_commit,
            "candidate_id": request.candidate_id,
            "dependency_lock_hash": request.dependency_lock_hash,
            "entrypoint": list(self._policy.entrypoint),
            "evaluator_fingerprint": request.evaluator_fingerprint,
            "evaluator_version": request.evaluator_version,
            "recipe_version": _RECIPE_VERSION,
            "runtime_base_image": self._policy.base_image_digest,
            "protocol_version": 1,
            "source_commit": request.source_commit,
            "source_tree_sha256": tree_digest,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class DependencyAwareWheelReleaseBuilder:
    """Select a sealed resolver base without mutating the startup trust policy."""

    def __init__(
        self,
        *,
        base_builder: ReleaseBuilder,
        base_policy: WheelReleaseBuildPolicy,
        resolver: DependencyResolver,
        builder_factory: Callable[[WheelReleaseBuildPolicy], ReleaseBuilder],
    ) -> None:
        self._base_builder = base_builder
        self._base_policy = base_policy
        self._resolver = resolver
        self._builder_factory = builder_factory
        self._builders: dict[str, ReleaseBuilder] = {}

    @property
    def state_contract(self) -> StateContract:
        return self._base_policy.state_contract

    @property
    def install_profile(self) -> str:
        return self._base_policy.install_profile

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        lock_hash = request.dependency_lock_hash
        if lock_hash == self._base_policy.base_dependency_lock_hash:
            return await self._base_builder.build(request)
        if lock_hash is None:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
        resolved = self._resolver.base_for_lock(lock_hash)
        if resolved is None:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
        builder = self._builders.get(resolved.id)
        if builder is None:
            builder = self._builder_factory(self._resolved_policy(resolved))
            self._builders[resolved.id] = builder
        return await builder.build(request)

    def _resolved_policy(self, resolved: ResolvedDependencyBase) -> WheelReleaseBuildPolicy:
        metadata_hashes = dict(self._base_policy.trusted_metadata_hashes)
        metadata_hashes["pyproject.toml"] = resolved.pyproject_sha256
        return replace(
            self._base_policy,
            base_dependency_lock_hash=resolved.lock_sha256,
            trusted_metadata_hashes=metadata_hashes,
            trusted_wheelhouse=resolved.wheelhouse,
            build_recipe_version=f"resolved-wheel-v1-{resolved.id}",
            dependency_base_id=resolved.id,
        )


class TrustedWheelReleaseBuilder:
    """Build a final-path virtualenv from trusted metadata and exact candidate blobs."""

    def __init__(
        self,
        *,
        policy: WheelReleaseBuildPolicy,
        process_runner: BoundedCommandRunner = run_bounded_process,
    ) -> None:
        self._policy = policy
        self._store = GenerationStore(policy.generations_root)
        self._generations_root = self._store.root
        raw_build_root = (
            policy.build_root.expanduser()
            if policy.build_root is not None
            else self._generations_root.parent / ".generation-builds"
        )
        self._build_root = self._secure_controller_directory(
            raw_build_root,
            create=True,
            label="generation build root",
        )
        if self._build_root == self._generations_root or self._is_relative_to(
            self._build_root, self._generations_root
        ):
            raise ValueError("build_root must be outside generations_root")
        try:
            self._wheelhouse = self._secure_controller_directory(
                policy.trusted_wheelhouse,
                create=False,
                label="trusted dependency wheelhouse",
            )
        except ValueError as exc:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED) from exc
        self._validate_wheelhouse()
        self._runner = process_runner

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        return await asyncio.to_thread(self._build, request)

    def _build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        raw_workspace = request.workspace.expanduser()
        if raw_workspace.is_symlink():
            raise ReleaseBuildError("candidate build workspace is invalid")
        try:
            workspace = raw_workspace.resolve(strict=True)
        except OSError as exc:
            raise ReleaseBuildError("candidate build workspace is invalid") from exc
        self._validate_request(request, workspace)
        staging_root = self._build_root / uuid4().hex
        source_root = staging_root / "source"
        wheel_root = staging_root / "wheels"
        staging_root.mkdir(mode=0o700)
        try:
            environment = self._environment(staging_root)
            git_directory, common_directory = self._git_directories(workspace)
            try:
                with repository_mutation_lock(common_directory):
                    self._assert_safe_git_configuration(
                        workspace,
                        git_directory,
                        common_directory,
                    )
                    changed_paths = self._verify_exact_candidate(request, workspace)
                    self._verify_trusted_metadata(request, workspace, changed_paths)
                    tree_digest = self._export_trusted_commit(
                        workspace,
                        request.source_commit,
                        source_root,
                        secret_scan_paths=frozenset(changed_paths),
                    )
            except RepositoryMutationLockError as exc:
                raise ReleaseBuildError("candidate repository lock failed") from exc
            lock_bytes = self._read_exported_lock(source_root)
            wheel_path = self._build_wheel(source_root, wheel_root, environment)
            wheel_size, wheel_sha256 = self._hash_wheel(wheel_path)
            runtime = self._runtime_metadata(environment)
            manifest = self._generation_manifest(
                request=request,
                tree_digest=tree_digest,
                wheel_path=wheel_path,
                wheel_size=wheel_size,
                wheel_sha256=wheel_sha256,
                lock_bytes=lock_bytes,
                runtime=runtime,
            )
            with self._store.locked():
                existing_path = self._generations_root / manifest.identity.generation_id
                if os.path.lexists(existing_path):
                    try:
                        existing = self._store.open_for_builder_reuse(
                            manifest.identity.generation_id,
                            expected_state_contract_digest=manifest.state_contract.sha256(),
                            expected_evaluator_fingerprint=request.evaluator_fingerprint,
                            expected_install_profile=self._policy.install_profile,
                            controller_protocol=self._policy.state_contract.runtime_protocol,
                        )
                    except GenerationStoreError:
                        self._store.quarantine(manifest.identity.generation_id)
                    else:
                        if (
                            existing.manifest.identity == manifest.identity
                            and existing.manifest.descriptor == manifest.descriptor
                            and existing.manifest.state_contract == manifest.state_contract
                        ):
                            return self._artifact(existing)
                        self._store.quarantine(manifest.identity.generation_id)
                generation_path = self._reserve_generation(manifest.identity.generation_id)
                try:
                    final_manifest = self._install_generation(
                        generation_path=generation_path,
                        source_root=source_root,
                        staging_root=staging_root,
                        wheel_path=wheel_path,
                        lock_bytes=lock_bytes,
                        manifest=manifest,
                        runtime=runtime,
                    )
                    expected_manifest_digest = generation_manifest_sha256(final_manifest)
                    installed = self._store.open(
                        final_manifest.identity.generation_id,
                        expected_manifest_digest=expected_manifest_digest,
                        expected_state_contract_digest=final_manifest.state_contract.sha256(),
                        expected_evaluator_fingerprint=request.evaluator_fingerprint,
                        expected_install_profile=self._policy.install_profile,
                        controller_protocol=self._policy.state_contract.runtime_protocol,
                    )
                except (GenerationStoreError, ReleaseBuildError, OSError, UnicodeError, ValueError):
                    if os.path.lexists(generation_path):
                        self._store.quarantine(manifest.identity.generation_id)
                    raise
                return self._artifact(installed)
        except ReleaseBuildError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise ReleaseBuildError("trusted Python generation build failed") from exc
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)

    def _validate_request(self, request: ReleaseBuildRequest, workspace: Path) -> None:
        if not request.candidate_id or len(request.candidate_id) > 100 or "\x00" in request.candidate_id:
            raise ReleaseBuildError("candidate build identity is invalid")
        if not workspace.is_dir() or workspace.is_symlink():
            raise ReleaseBuildError("candidate build workspace is invalid")
        if not _GENERATION_COMMIT_RE.fullmatch(
            request.base_commit
        ) or not _GENERATION_COMMIT_RE.fullmatch(request.source_commit):
            raise ReleaseBuildError("candidate source commit is invalid")
        if not _DIGEST_RE.fullmatch(request.evaluator_fingerprint):
            raise ReleaseBuildError("candidate evaluator identity is invalid")
        if not request.evaluator_version.strip():
            raise ReleaseBuildError("candidate evaluator version is invalid")
        if request.evaluation_input_sha256 is None or not _LOCK_HASH_RE.fullmatch(
            request.evaluation_input_sha256
        ):
            raise ReleaseBuildError("candidate evaluation input digest is invalid")
        if request.dependency_lock_hash != self._policy.base_dependency_lock_hash:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)

    def _verify_exact_candidate(
        self,
        request: ReleaseBuildRequest,
        workspace: Path,
    ) -> tuple[str, ...]:
        head = self._run_trusted_git(workspace, "rev-parse", "--verify", "HEAD^{commit}")
        if head.output.decode("ascii", errors="ignore").strip().lower() != request.source_commit:
            raise ReleaseBuildError("candidate workspace no longer matches its evaluated commit")
        status_result = self._run_trusted_git(
            workspace,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        if status_result.output.strip():
            raise ReleaseBuildError("candidate workspace changed after evaluation")
        ancestor = self._run_trusted_git(
            workspace,
            "merge-base",
            "--is-ancestor",
            request.base_commit,
            request.source_commit,
            allowed_returncodes=frozenset({0, 1}),
        )
        if ancestor.returncode != 0:
            raise ReleaseBuildError("candidate base commit is not an ancestor of evaluated source")
        changed = self._run_trusted_git(
            workspace,
            "diff",
            "--name-only",
            "--no-ext-diff",
            "--diff-filter=ACDMRTUXB",
            "-z",
            request.base_commit,
            request.source_commit,
            "--",
        )
        try:
            paths = tuple(path for path in changed.output.decode("utf-8").split("\0") if path)
        except UnicodeDecodeError as exc:
            raise ReleaseBuildError("candidate source change set was invalid") from exc
        if any(not candidate_path_is_promotable(path) for path in paths):
            raise ReleaseBuildError(
                "candidate changes are contribution-only and cannot enter a production release"
            )
        return paths

    def _verify_trusted_metadata(
        self,
        request: ReleaseBuildRequest,
        workspace: Path,
        changed_paths: tuple[str, ...],
    ) -> None:
        metadata_paths = set(self._policy.packaging_metadata_paths)
        dependency_upgrade = self._policy.dependency_base_id is not None
        if "uv.lock" in changed_paths and not dependency_upgrade:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
        changed_metadata = {path for path in changed_paths if path in metadata_paths}
        if changed_metadata and (
            not dependency_upgrade or changed_metadata != {"pyproject.toml"}
        ):
            raise ReleaseBuildError(
                "candidate packaging or build metadata changed; trusted release input is required"
            )
        base_lock = self._git_blob(workspace, request.base_commit, "uv.lock", required=True)
        source_lock = self._git_blob(workspace, request.source_commit, "uv.lock", required=True)
        assert base_lock is not None and source_lock is not None
        expected_lock_hash = self._policy.base_dependency_lock_hash
        if (
            (not dependency_upgrade and base_lock != source_lock)
            or hashlib.sha256(source_lock).hexdigest() != expected_lock_hash
        ):
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
        metadata_to_verify = set(self._policy.packaging_metadata_paths) | set(
            self._policy.trusted_metadata_hashes
        )
        for metadata_path in sorted(metadata_to_verify):
            base_blob = self._git_blob(
                workspace,
                request.base_commit,
                metadata_path,
                required=metadata_path in self._policy.trusted_metadata_hashes,
            )
            source_blob = self._git_blob(
                workspace,
                request.source_commit,
                metadata_path,
                required=metadata_path in self._policy.trusted_metadata_hashes,
            )
            if base_blob != source_blob and not (
                dependency_upgrade and metadata_path == "pyproject.toml"
            ):
                raise ReleaseBuildError(
                    "candidate packaging or build metadata changed; trusted release input is required"
                )
            if source_blob is not None:
                expected = self._policy.trusted_metadata_hashes.get(metadata_path)
                if expected is None or hashlib.sha256(source_blob).hexdigest() != expected:
                    raise ReleaseBuildError("trusted packaging metadata pin does not match")
        for source, _, expected_hash in self._policy.trusted_bridge_assets:
            blob = self._git_blob(workspace, request.source_commit, source, required=True)
            assert blob is not None
            if hashlib.sha256(blob).hexdigest() != expected_hash:
                raise ReleaseBuildError("trusted bridge asset pin does not match")

    def _git_blob(
        self,
        workspace: Path,
        commit: str,
        relative_path: str,
        *,
        required: bool,
    ) -> bytes | None:
        listing = self._run_trusted_git(
            workspace,
            "ls-tree",
            "-z",
            "--full-tree",
            commit,
            "--",
            relative_path,
        )
        entries = tuple(item for item in listing.output.split(b"\0") if item)
        if not entries:
            if required:
                raise ReleaseBuildError("trusted release metadata is missing")
            return None
        if len(entries) != 1:
            raise ReleaseBuildError("trusted release metadata is ambiguous")
        try:
            header, raw_path = entries[0].split(b"\t", 1)
            mode, object_type, object_id = header.split(b" ", 2)
            decoded_path = raw_path.decode("utf-8")
            object_id_text = object_id.decode("ascii")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReleaseBuildError("trusted release metadata is invalid") from exc
        if (
            decoded_path != relative_path
            or object_type != b"blob"
            or mode not in {b"100644", b"100755"}
        ):
            raise ReleaseBuildError("trusted release metadata is not a regular file")
        blob = self._run_trusted_git(
            workspace,
            "cat-file",
            "blob",
            object_id_text,
            max_output_bytes=self._policy.max_source_bytes,
        )
        return blob.output

    @staticmethod
    def _git_directories(workspace: Path) -> tuple[Path, Path]:
        try:
            return discover_git_directories(workspace)
        except GitSecurityError as exc:
            raise ReleaseBuildError("candidate Git directories are invalid") from exc

    def _assert_safe_git_configuration(
        self,
        workspace: Path,
        git_directory: Path,
        common_directory: Path,
    ) -> None:
        unsafe_state = (
            common_directory / "shallow",
            common_directory / "info" / "grafts",
            common_directory / "objects" / "info" / "alternates",
            common_directory / "objects" / "info" / "http-alternates",
            git_directory / "shallow",
            git_directory / "info" / "grafts",
            common_directory / "config.worktree",
            git_directory / "config.worktree",
        )
        config_path = common_directory / "config"
        try:
            config_metadata = config_path.lstat()
        except OSError as exc:
            raise ReleaseBuildError("candidate repository Git configuration is unsafe") from exc
        if any(os.path.lexists(path) for path in unsafe_state) or not stat.S_ISREG(
            config_metadata.st_mode
        ):
            raise ReleaseBuildError("candidate repository Git configuration is unsafe")
        result = self._run_trusted_git(
            workspace,
            "config",
            "--file",
            str(config_path),
            "--no-includes",
            "--null",
            "--list",
        )
        try:
            records = tuple(record for record in result.output.split(b"\0") if record)
            entries: list[tuple[str, str]] = []
            for record in records:
                raw_name, separator, raw_value = record.partition(b"\n")
                if not separator:
                    raise ValueError("Git config record has no value separator")
                entries.append(
                    (
                        raw_name.decode("utf-8", errors="strict").casefold(),
                        raw_value.decode("utf-8", errors="strict"),
                    )
                )
        except (UnicodeError, ValueError) as exc:
            raise ReleaseBuildError("candidate repository Git configuration is unsafe") from exc
        if not _repository_config_is_safe(tuple(entries)):
            raise ReleaseBuildError("candidate repository Git configuration is unsafe")

    def _run_trusted_git(
        self,
        workspace: Path,
        *arguments: str,
        allowed_returncodes: frozenset[int] = frozenset({0}),
        max_output_bytes: int | None = None,
    ) -> BoundedProcessResult:
        try:
            result = run_hardened_git(
                workspace,
                arguments,
                timeout_seconds=min(self._policy.timeout_seconds, 300),
                max_output_bytes=max_output_bytes or self._policy.max_output_bytes,
                git_executable=self._policy.git_cli,
            )
        except (OSError, ValueError) as exc:
            raise ReleaseBuildError("trusted Git operation failed") from exc
        if (
            result.returncode not in allowed_returncodes
            or result.truncated
            or result.timed_out
        ):
            raise ReleaseBuildError("trusted Git operation failed")
        return result

    def _export_trusted_commit(
        self,
        workspace: Path,
        source_commit: str,
        destination_root: Path,
        *,
        secret_scan_paths: frozenset[str],
    ) -> str:
        listing = self._run_trusted_git(
            workspace,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            source_commit,
            max_output_bytes=self._policy.max_source_bytes,
        )
        destination_root.mkdir(mode=0o700)
        digest = hashlib.sha256()
        total_bytes = 0
        entries = 0
        materialized_paths: set[str] = set()
        for raw_entry in listing.output.split(b"\0"):
            if not raw_entry:
                continue
            entries += 1
            if entries > self._policy.max_source_entries:
                raise ReleaseBuildError("candidate source tree has too many entries")
            try:
                header, raw_path = raw_entry.split(b"\t", 1)
                mode, object_type, object_id = header.split(b" ", 2)
                path_text = raw_path.decode("utf-8")
                object_id_text = object_id.decode("ascii")
            except (UnicodeDecodeError, ValueError) as exc:
                raise ReleaseBuildError("candidate source tree entry is invalid") from exc
            if object_type != b"blob" or mode not in {b"100644", b"100755"}:
                raise ReleaseBuildError("candidate source contains a link or special file")
            relative = PurePosixPath(path_text)
            self._validate_source_path(relative)
            normalized = relative.as_posix().casefold()
            if normalized in materialized_paths:
                raise ReleaseBuildError("candidate source contains colliding paths")
            materialized_paths.add(normalized)
            remaining = self._policy.max_source_bytes - total_bytes
            blob = self._run_trusted_git(
                workspace,
                "cat-file",
                "blob",
                object_id_text,
                max_output_bytes=remaining + 1,
            )
            if len(blob.output) > remaining:
                raise ReleaseBuildError("candidate source exceeds its byte limit")
            if relative.as_posix() in secret_scan_paths and candidate_content_contains_secret(
                relative.as_posix(), blob.output
            ):
                raise ReleaseBuildError("candidate source contains credential material")
            total_bytes += len(blob.output)
            digest.update(mode + b"\0" + raw_path + b"\0" + object_id + b"\0")
            digest.update(blob.output)
            digest.update(b"\0")
            destination = destination_root.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            self._write_file(
                destination,
                blob.output,
                mode=0o755 if mode == b"100755" else 0o644,
            )
        return digest.hexdigest()

    def _read_exported_lock(self, source_root: Path) -> bytes:
        lock_path = source_root / "uv.lock"
        self._require_regular_file(lock_path, label="exported dependency lock")
        lock_bytes = lock_path.read_bytes()
        if hashlib.sha256(lock_bytes).hexdigest() != self._policy.base_dependency_lock_hash:
            raise ReleaseBuildError("exported dependency lock failed verification")
        return lock_bytes

    def _build_wheel(
        self,
        source_root: Path,
        wheel_root: Path,
        environment: Mapping[str, str],
    ) -> Path:
        del environment
        wheel_root.mkdir(mode=0o700)
        pyproject_path = source_root / "pyproject.toml"
        self._require_regular_file(pyproject_path, label="trusted pyproject metadata")
        try:
            document = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
            raise ReleaseBuildError("trusted pyproject metadata is invalid") from exc
        project = document.get("project")
        build_system = document.get("build-system")
        if not isinstance(project, dict) or not isinstance(build_system, dict):
            raise ReleaseBuildError("trusted pyproject metadata is incomplete")
        if build_system.get("backend-path"):
            raise ReleaseBuildError("PEP 517 backend-path is forbidden for autonomous releases")
        tool = document.get("tool")
        tool = tool if isinstance(tool, dict) else {}
        hatch = tool.get("hatch")
        hatch = hatch if isinstance(hatch, dict) else {}
        hatch_build = hatch.get("build")
        hatch_build = hatch_build if isinstance(hatch_build, dict) else {}
        if hatch_build.get("hooks"):
            raise ReleaseBuildError("candidate build hooks are forbidden for autonomous releases")
        targets = hatch_build.get("targets")
        targets = targets if isinstance(targets, dict) else {}
        wheel_configuration = targets.get("wheel")
        wheel_configuration = (
            wheel_configuration if isinstance(wheel_configuration, dict) else {}
        )
        if wheel_configuration.get("hooks"):
            raise ReleaseBuildError("candidate build hooks are forbidden for autonomous releases")
        configured_packages = wheel_configuration.get("packages")
        if configured_packages is not None and (
            not isinstance(configured_packages, list)
            or tuple(configured_packages) != self._policy.package_roots
        ):
            raise ReleaseBuildError("trusted package layout does not match builder policy")
        configured_force_include = wheel_configuration.get("force-include", {})
        expected_force_include = {
            source: destination for source, destination, _ in self._policy.trusted_bridge_assets
        }
        if configured_force_include != expected_force_include:
            raise ReleaseBuildError("trusted bridge asset layout does not match builder policy")

        name = project.get("name")
        version = project.get("version")
        requires_python = project.get("requires-python")
        scripts = project.get("scripts", {})
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
            or not isinstance(version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", version)
            or (requires_python is not None and not isinstance(requires_python, str))
            or not isinstance(scripts, dict)
        ):
            raise ReleaseBuildError("trusted project metadata is invalid")
        for command, target in scripts.items():
            if (
                not isinstance(command, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", command)
                or not isinstance(target, str)
                or not re.fullmatch(
                    r"[A-Za-z_][A-Za-z0-9_.]*(?::[A-Za-z_][A-Za-z0-9_.]*)?", target
                )
            ):
                raise ReleaseBuildError("trusted project script metadata is invalid")

        entries: dict[str, tuple[bytes, int]] = {}
        for package_root in self._policy.package_roots:
            source_package = source_root.joinpath(*PurePosixPath(package_root).parts)
            self._collect_package_entries(source_root, source_package, package_root, entries)
        for source, destination, expected_hash in self._policy.trusted_bridge_assets:
            source_path = source_root.joinpath(*PurePosixPath(source).parts)
            self._require_regular_file(source_path, label="trusted bridge asset")
            payload = source_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected_hash:
                raise ReleaseBuildError("trusted bridge asset failed hash verification")
            self._add_wheel_entry(entries, destination, payload, 0o644)

        package_destination = PurePosixPath(*self._policy.import_name.split("."))
        for resource in self._policy.resource_paths:
            expected_resource = (package_destination / resource).as_posix()
            if expected_resource not in entries:
                raise ReleaseBuildError("required package resource is absent from assembled wheel")

        normalized_name = re.sub(r"[-_.]+", "_", name)
        normalized_version = re.sub(r"[-]+", "_", version)
        dist_info = f"{normalized_name}-{normalized_version}.dist-info"
        metadata = self._wheel_metadata(project, name=name, version=version)
        wheel_metadata = (
            "Wheel-Version: 1.0\n"
            f"Generator: opentulpa-{self._policy.build_recipe_version}\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode()
        entry_points = "[console_scripts]\n" + "".join(
            f"{command} = {scripts[command]}\n" for command in sorted(scripts)
        )
        self._add_wheel_entry(entries, f"{dist_info}/METADATA", metadata, 0o644)
        self._add_wheel_entry(entries, f"{dist_info}/WHEEL", wheel_metadata, 0o644)
        self._add_wheel_entry(
            entries,
            f"{dist_info}/entry_points.txt",
            entry_points.encode("utf-8"),
            0o644,
        )
        wheel_path = wheel_root / f"{normalized_name}-{normalized_version}-py3-none-any.whl"
        self._write_deterministic_wheel(wheel_path, entries, dist_info=dist_info)
        self._verify_structural_wheel(wheel_path, dist_info=dist_info)
        return wheel_path

    def _collect_package_entries(
        self,
        source_root: Path,
        source_package: Path,
        package_root: str,
        entries: dict[str, tuple[bytes, int]],
    ) -> None:
        self._require_directory(source_package, label="allowlisted package root")
        root_parts = PurePosixPath(package_root).parts
        destination_root = PurePosixPath(*(root_parts[1:] if root_parts[0] == "src" else root_parts))
        for path in sorted(source_package.rglob("*")):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise ReleaseBuildError("allowlisted package contains a link or special file")
            if stat.S_ISDIR(metadata.st_mode):
                continue
            relative = path.relative_to(source_package)
            destination = (destination_root / PurePosixPath(relative.as_posix())).as_posix()
            if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
                raise ReleaseBuildError("allowlisted package contains generated Python files")
            mode = 0o755 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o644
            self._add_wheel_entry(entries, destination, path.read_bytes(), mode)

    @staticmethod
    def _add_wheel_entry(
        entries: dict[str, tuple[bytes, int]],
        raw_path: str,
        payload: bytes,
        mode: int,
    ) -> None:
        path = PurePosixPath(raw_path)
        if (
            "\x00" in raw_path
            or "\\" in raw_path
            or re.match(r"^[A-Za-z]:", raw_path)
            or any(existing.casefold() == raw_path.casefold() for existing in entries)
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or raw_path in entries
        ):
            raise ReleaseBuildError("assembled wheel contains an unsafe or duplicate path")
        entries[raw_path] = (payload, mode)

    @staticmethod
    def _wheel_metadata(project: Mapping[str, Any], *, name: str, version: str) -> bytes:
        lines = ["Metadata-Version: 2.3", f"Name: {name}", f"Version: {version}"]
        requires_python = project.get("requires-python")
        if isinstance(requires_python, str):
            lines.append(f"Requires-Python: {requires_python}")
        dependencies = project.get("dependencies", [])
        if not isinstance(dependencies, list) or any(
            not isinstance(dependency, str) for dependency in dependencies
        ):
            raise ReleaseBuildError("trusted project dependencies are invalid")
        lines.extend(f"Requires-Dist: {dependency}" for dependency in dependencies)
        optional = project.get("optional-dependencies", {})
        if not isinstance(optional, dict):
            raise ReleaseBuildError("trusted optional dependencies are invalid")
        for extra in sorted(optional):
            requirements = optional[extra]
            if not isinstance(extra, str) or not isinstance(requirements, list) or any(
                not isinstance(requirement, str) for requirement in requirements
            ):
                raise ReleaseBuildError("trusted optional dependencies are invalid")
            lines.append(f"Provides-Extra: {extra}")
            for requirement in requirements:
                if ";" in requirement:
                    distribution, marker = requirement.split(";", 1)
                    rendered = f"{distribution.strip()}; ({marker.strip()}) and extra == '{extra}'"
                else:
                    rendered = f"{requirement}; extra == '{extra}'"
                lines.append(f"Requires-Dist: {rendered}")
        return ("\n".join(lines) + "\n").encode("utf-8")

    @staticmethod
    def _write_deterministic_wheel(
        wheel_path: Path,
        entries: Mapping[str, tuple[bytes, int]],
        *,
        dist_info: str,
    ) -> None:
        record_path = f"{dist_info}/RECORD"
        rows: list[tuple[str, str, str]] = []
        with zipfile.ZipFile(
            wheel_path,
            mode="x",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(entries):
                payload, mode = entries[path]
                info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(
                    info,
                    payload,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
                encoded_hash = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
                rows.append((path, f"sha256={encoded_hash.decode('ascii')}", str(len(payload))))
            output = io.StringIO(newline="")
            writer = csv.writer(output, lineterminator="\n")
            writer.writerows((*rows, (record_path, "", "")))
            record = output.getvalue().encode("utf-8")
            info = zipfile.ZipInfo(record_path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(
                info,
                record,
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

    @staticmethod
    def _verify_structural_wheel(wheel_path: Path, *, dist_info: str) -> None:
        try:
            with zipfile.ZipFile(wheel_path) as archive:
                names = archive.namelist()
                required = {
                    f"{dist_info}/METADATA",
                    f"{dist_info}/WHEEL",
                    f"{dist_info}/entry_points.txt",
                    f"{dist_info}/RECORD",
                }
                if len(names) != len(set(names)) or not required.issubset(names):
                    raise ReleaseBuildError("assembled wheel structure is invalid")
                archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
                if any(
                    name.startswith("/")
                    or ".." in PurePosixPath(name).parts
                    or stat.S_ISLNK((item.external_attr >> 16) & 0xFFFF)
                    for item, name in ((item, item.filename) for item in archive.infolist())
                ):
                    raise ReleaseBuildError("assembled wheel contains an unsafe path")
        except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
            raise ReleaseBuildError("assembled wheel failed structural verification") from exc

    def _hash_wheel(self, wheel_path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        with wheel_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                if size > self._policy.max_wheel_bytes:
                    raise ReleaseBuildError("candidate wheel exceeds its byte limit")
                digest.update(chunk)
        if size < 1:
            raise ReleaseBuildError("candidate wheel is empty")
        return size, digest.hexdigest()

    def _runtime_metadata(self, environment: Mapping[str, str]) -> dict[str, str]:
        script = (
            "import json,os,platform,sys,sysconfig;"
            "print(json.dumps({'cpython_version':platform.python_version(),"
            "'cpython_cache_tag':sys.implementation.cache_tag or '',"
            "'cpython_abi_tag':f'cp{sys.version_info.major}{sys.version_info.minor}',"
            "'os_name':os.name,'platform':sysconfig.get_platform(),"
            "'machine':platform.machine(),'implementation':sys.implementation.name,"
            "'libpython':os.path.join(sysconfig.get_config_var('LIBDIR') or '',"
            "sysconfig.get_config_var('LDLIBRARY') or '')},"
            "sort_keys=True,separators=(',',':')))"
        )
        result = self._run_checked(
            (self._policy.python_executable, "-I", "-c", script),
            cwd=self._build_root,
            env=environment,
            timeout_seconds=30,
            max_output_bytes=16 * 1_024,
            failure="trusted Python runtime could not be identified",
        )
        try:
            document: Any = json.loads(result.output.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReleaseBuildError("trusted Python runtime identity was invalid") from exc
        fields = (
            "cpython_version",
            "cpython_cache_tag",
            "cpython_abi_tag",
            "os_name",
            "platform",
            "machine",
            "libpython",
        )
        if not isinstance(document, dict) or document.get("implementation") != "cpython" or any(
            not isinstance(document.get(field), str) for field in fields
        ):
            raise ReleaseBuildError("trusted Python runtime identity was invalid")
        host = {
            "cpython_version": platform_module.python_version(),
            "cpython_cache_tag": str(sys.implementation.cache_tag or ""),
            "cpython_abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
            "os_name": os.name,
            "platform": sysconfig.get_platform(),
            "machine": platform_module.machine(),
        }
        if any(document[field] != value for field, value in host.items()):
            raise ReleaseBuildError("trusted Python runtime is incompatible with this host")
        runtime = {field: str(document[field]) for field in fields}
        runtime["python_runtime_sha256"] = self._python_runtime_sha256()
        return runtime

    def _python_runtime_sha256(self) -> str:
        try:
            interpreter = Path(self._policy.python_executable).expanduser().resolve(strict=True)
            metadata = interpreter.lstat()
        except OSError as exc:
            raise ReleaseBuildError("trusted Python interpreter is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ReleaseBuildError("trusted Python interpreter is not a regular file")
        digest = hashlib.sha256()
        try:
            with interpreter.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError as exc:
            raise ReleaseBuildError("trusted Python interpreter could not be hashed") from exc
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "external_python_runtime_policy_sha256": (
                        self._policy.external_python_runtime_policy_sha256
                    ),
                    "interpreter_sha256": digest.hexdigest(),
                }
            )
        ).hexdigest()

    def _generation_manifest(
        self,
        *,
        request: ReleaseBuildRequest,
        tree_digest: str,
        wheel_path: Path,
        wheel_size: int,
        wheel_sha256: str,
        lock_bytes: bytes,
        runtime: Mapping[str, str],
    ) -> GenerationManifest:
        contract = self._policy.state_contract
        identity = GenerationIdentity(
            source_commit=request.source_commit,
            source_tree_sha256=tree_digest,
            wheel_sha256=wheel_sha256,
            uv_lock_sha256=hashlib.sha256(lock_bytes).hexdigest(),
            evaluator_fingerprint=request.evaluator_fingerprint,
            evaluation_input_sha256=str(request.evaluation_input_sha256),
            python_runtime_sha256=runtime["python_runtime_sha256"],
            cpython_version=runtime["cpython_version"],
            cpython_cache_tag=runtime["cpython_cache_tag"],
            cpython_abi_tag=runtime["cpython_abi_tag"],
            os_name="posix",
            platform=runtime["platform"],
            machine=runtime["machine"],
            build_recipe_version=self._policy.build_recipe_version,
            runtime_protocol=contract.runtime_protocol,
            controller_min=contract.controller_min,
            controller_max=contract.controller_max,
            state_contract_sha256=contract.sha256(),
            install_profile=self._policy.install_profile,
            extras=self._policy.extras,
            entrypoint=self._policy.entrypoint,
        )
        return GenerationManifest(
            identity=identity,
            state_contract=contract,
            descriptor=GenerationDescriptor(
                wheel_path=f"artifacts/{wheel_path.name}",
                wheel_size_bytes=wheel_size,
                uv_lock_path="artifacts/uv.lock",
                uv_lock_size_bytes=len(lock_bytes),
                venv_path="venv",
            ),
            runtime_tree_sha256="0" * 64,
        )

    def _reserve_generation(self, generation_id: str) -> Path:
        generation_path = self._generations_root / generation_id
        try:
            generation_path.mkdir(mode=0o700, exist_ok=False)
            building = canonical_json_bytes(
                {
                    "nonce": uuid4().hex,
                    "pid": os.getpid(),
                    "started_at": time.time(),
                }
            )
            self._write_file(generation_path / "BUILDING", building, mode=0o600)
            self._fsync_directory(self._generations_root)
        except FileExistsError as exc:
            raise ReleaseBuildError("Python generation identity is already reserved") from exc
        except OSError as exc:
            raise ReleaseBuildError("Python generation path could not be reserved") from exc
        return generation_path

    def _install_generation(
        self,
        *,
        generation_path: Path,
        source_root: Path,
        staging_root: Path,
        wheel_path: Path,
        lock_bytes: bytes,
        manifest: GenerationManifest,
        runtime: Mapping[str, str],
    ) -> GenerationManifest:
        artifacts_path = generation_path / "artifacts"
        artifacts_path.mkdir(mode=0o700)
        final_wheel = generation_path / manifest.descriptor.wheel_path
        self._copy_file(wheel_path, final_wheel)
        self._write_file(generation_path / manifest.descriptor.uv_lock_path, lock_bytes, mode=0o600)
        self._fsync_directory(artifacts_path)

        environment = self._environment(staging_root)
        venv_path = generation_path / manifest.descriptor.venv_path
        self._run_checked(
            (
                self._policy.python_executable,
                "-I",
                "-m",
                "venv",
                "--copies",
                "--without-pip",
                str(venv_path),
            ),
            cwd=generation_path,
            env=environment,
            timeout_seconds=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            failure="final-path virtualenv creation failed",
        )
        interpreter = venv_path / "bin/python"
        self._require_regular_file(interpreter, label="final generation interpreter")
        self._copy_runtime_library(runtime["libpython"], venv_path)

        requirements_path = staging_root / "trusted-requirements.txt"
        self._validate_wheelhouse()
        export_command = [
            self._policy.uv_cli,
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-header",
            "--output-file",
            str(requirements_path),
        ]
        for extra in self._policy.extras:
            export_command.extend(("--extra", extra))
        self._run_checked(
            tuple(export_command),
            cwd=source_root,
            env=environment,
            timeout_seconds=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            failure=_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED,
        )
        self._validate_exported_requirements(requirements_path)
        self._run_checked(
            (
                self._policy.uv_cli,
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--offline",
                "--no-index",
                "--find-links",
                str(self._wheelhouse),
                "--link-mode=copy",
                "--require-hashes",
                "--only-binary=:all:",
                "--strict",
                "--requirements",
                str(requirements_path),
            ),
            cwd=generation_path,
            env=environment,
            timeout_seconds=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            failure=_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED,
        )
        self._run_checked(
            (
                self._policy.uv_cli,
                "pip",
                "install",
                "--python",
                str(interpreter),
                "--offline",
                "--no-index",
                "--link-mode=copy",
                "--no-deps",
                "--strict",
                str(final_wheel),
            ),
            cwd=generation_path,
            env=environment,
            timeout_seconds=self._policy.timeout_seconds,
            max_output_bytes=self._policy.max_output_bytes,
            failure="candidate wheel installation failed",
        )
        self._verify_installed_structure(generation_path, interpreter)
        self._remove_expected_venv_links(venv_path)
        return self._store.publish_staged(manifest.identity.generation_id, manifest)

    def _copy_runtime_library(self, raw_library: str, venv_path: Path) -> None:
        if not raw_library:
            return
        source = Path(raw_library).expanduser().resolve(strict=False)
        if not source.exists():
            return
        self._require_regular_file(source, label="trusted Python runtime library")
        destination = venv_path / "lib" / source.name
        if destination.exists():
            self._require_regular_file(destination, label="virtualenv Python runtime library")
            return
        self._copy_file(source, destination)
        self._fsync_directory(destination.parent)

    def _verify_installed_structure(self, generation_path: Path, interpreter: Path) -> None:
        self._require_regular_file(interpreter, label="final generation interpreter")
        entrypoint = generation_path / self._policy.entrypoint[0]
        self._require_regular_file(entrypoint, label="final generation entrypoint")
        if not os.access(interpreter, os.X_OK) or not os.access(entrypoint, os.X_OK):
            raise ReleaseBuildError("installed generation executables are not executable")

    def _validate_exported_requirements(self, requirements_path: Path) -> None:
        self._require_regular_file(requirements_path, label="trusted exported requirements")
        try:
            requirements = requirements_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED) from exc
        lowered = requirements.casefold()
        if any(
            marker in lowered
            for marker in ("http://", "https://", "file:", "git+", " @ ", "../", "./")
        ) or any(line.lstrip().startswith(("-e ", "--editable ")) for line in requirements.splitlines()):
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
        logical_requirements = [
            line
            for line in requirements.splitlines()
            if line and not line.startswith((" ", "#", "--hash=")) and line != "\\"
        ]
        if logical_requirements and "--hash=sha256:" not in requirements:
            raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)

    @staticmethod
    def _remove_expected_venv_links(venv_path: Path) -> None:
        lib64 = venv_path / "lib64"
        if lib64.is_symlink() and os.readlink(lib64) == "lib":
            lib64.unlink()

    @staticmethod
    def _artifact(installed: Any) -> OciReleaseArtifact:
        return OciReleaseArtifact(
            artifact_kind="python_generation",
            artifact_digest=installed.manifest_digest,
            manifest_digest=installed.manifest_digest,
            image_reference=f"python-generation:{installed.generation_id}",
            entrypoint=installed.manifest.identity.entrypoint,
        )

    @staticmethod
    def _secure_controller_directory(
        raw_path: Path,
        *,
        create: bool,
        label: str,
    ) -> Path:
        path = raw_path.expanduser().absolute()
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            if os.path.lexists(current) and current.is_symlink():
                raise ValueError(f"{label} has a symbolic-link ancestor")
        if create:
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ValueError(f"{label} is unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) & 0o077
        ):
            raise ValueError(f"{label} is not a private regular directory")
        return path

    def _validate_wheelhouse(self) -> None:
        for path in self._wheelhouse.rglob("*"):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
            if stat.S_ISDIR(metadata.st_mode):
                if stat.S_IMODE(metadata.st_mode) & 0o022 or metadata.st_uid != os.geteuid():
                    raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)
                continue
            if (
                not stat.S_ISREG(metadata.st_mode)
                or path.suffix != ".whl"
                or stat.S_IMODE(metadata.st_mode) & 0o022
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                raise ReleaseBuildError(_TRUSTED_RUNTIME_BASE_REBUILD_REQUIRED)

    def _validate_source_path(self, relative: PurePosixPath) -> None:
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", "..", ".git"} for part in relative.parts)
        ):
            raise ReleaseBuildError("candidate build context contains an unsafe path")
        if TrustedOciReleaseBuilder._is_sensitive(relative):
            raise ReleaseBuildError("candidate build context contains a forbidden secret path")
        if not candidate_path_is_runtime_overlay(relative.as_posix()):
            raise ReleaseBuildError(
                "candidate changes are contribution-only and cannot enter a production release"
            )

    def _run_checked(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
        failure: str,
    ) -> BoundedProcessResult:
        result = self._run_raw(
            argv,
            cwd=cwd,
            env=env,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if result.returncode != 0 or result.timed_out or result.truncated:
            raise ReleaseBuildError(failure)
        return result

    def _run_raw(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        timeout_seconds: float,
        max_output_bytes: int,
    ) -> BoundedProcessResult:
        try:
            return self._runner(
                argv,
                cwd=cwd,
                env=env,
                timeout_seconds=timeout_seconds,
                max_output_bytes=max_output_bytes,
            )
        except OSError as exc:
            raise ReleaseBuildError("trusted generation command could not be executed") from exc

    @staticmethod
    def _environment(root: Path) -> dict[str, str]:
        home = root / ".home"
        temporary = root / ".tmp"
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        (home / ".config").mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary.mkdir(parents=True, exist_ok=True, mode=0o700)
        return {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "TMPDIR": str(temporary),
            "LANG": "C",
            "LC_ALL": "C",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "PIP_CONFIG_FILE": os.devnull,
            "UV_NO_CONFIG": "1",
            "UV_OFFLINE": "1",
        }

    @staticmethod
    def _write_file(path: Path, payload: bytes, *, mode: int) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, mode)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _copy_file(source: Path, destination: Path) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(destination, flags, 0o600)
        try:
            with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
        except BaseException:
            destination.unlink(missing_ok=True)
            raise

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _require_directory(path: Path, *, label: str) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ReleaseBuildError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ReleaseBuildError(f"{label} is invalid")

    @staticmethod
    def _require_regular_file(path: Path, *, label: str) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise ReleaseBuildError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ReleaseBuildError(f"{label} is not a regular file")

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
        except ValueError:
            return False
        return True


__all__ = [
    "BoundedCommandRunner",
    "DependencyAwareWheelReleaseBuilder",
    "OciReleaseArtifact",
    "OciReleaseBuildPolicy",
    "ReleaseBuildError",
    "ReleaseBuildRequest",
    "ReleaseBuilder",
    "TrustedWheelBuildPolicy",
    "TrustedOciReleaseBuilder",
    "TrustedWheelReleaseBuilder",
    "WheelReleaseBuildPolicy",
]
