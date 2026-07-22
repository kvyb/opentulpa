"""Trusted, content-addressed OCI builder for evaluated source candidates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from opentulpa.bootstrap.oci_host import (
    LocalOciCommandRunner,
    OciCommandRunner,
)
from opentulpa.evolution.process import run_bounded_process
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
_DOCKERIGNORE_STAGING = ".opentulpa-candidate-dockerignore"


class ReleaseBuildError(RuntimeError):
    """Sanitized trusted-builder failure safe for evaluation evidence."""


@dataclass(frozen=True, slots=True)
class ReleaseBuildRequest:
    candidate_id: str
    workspace: Path
    base_commit: str
    source_commit: str
    dependency_lock_hash: str | None
    evaluator_version: str
    evaluator_fingerprint: str


class OciReleaseArtifact(BaseModel):
    """Verified release artifact plus the manifest bound to its source."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    artifact_kind: Literal["oci_image", "source_overlay"] = "oci_image"
    image_reference: str = Field(min_length=1, max_length=300)
    entrypoint: tuple[str, ...] = Field(min_length=1, max_length=64)


class ReleaseBuilder(Protocol):
    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact: ...


@dataclass(frozen=True, slots=True)
class OciReleaseBuildPolicy:
    base_image_digest: str
    base_dependency_lock_hash: str
    container_cli: str = "docker"
    git_cli: str = "git"
    state_root: Path = Path(".opentulpa/release-builds")
    image_tag_prefix: str = "opentulpa-release"
    entrypoint: tuple[str, ...] = ("./start.sh", "run", "server")
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
class SourceOverlayBuildPolicy:
    """Trusted bounds for a source overlay using the host's pinned environment."""

    base_dependency_lock_hash: str
    git_cli: str = "git"
    entrypoint: tuple[str, ...] = ("python", "-m", "opentulpa")
    max_tree_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        if not _LOCK_HASH_RE.fullmatch(self.base_dependency_lock_hash):
            raise ValueError("base_dependency_lock_hash must be a SHA-256 lockfile hash")
        if Path(self.git_cli).name != "git" or "\x00" in self.git_cli:
            raise ValueError("git_cli must be a Git executable")
        if not self.entrypoint or any(not item or "\x00" in item for item in self.entrypoint):
            raise ValueError("entrypoint must contain safe exec arguments")
        if self.max_tree_bytes < 1024 * 1024:
            raise ValueError("source overlay tree limit is too small")


class TrustedSourceOverlayBuilder:
    """Bind an evaluated commit to the immutable dependencies in the host image."""

    def __init__(self, *, policy: SourceOverlayBuildPolicy) -> None:
        self._policy = policy

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        return await asyncio.to_thread(self._build, request)

    def _build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        workspace = request.workspace.expanduser().resolve(strict=True)
        if not workspace.is_dir() or workspace.is_symlink():
            raise ReleaseBuildError("candidate build workspace is invalid")
        if not _COMMIT_RE.fullmatch(request.base_commit) or not _COMMIT_RE.fullmatch(
            request.source_commit
        ):
            raise ReleaseBuildError("candidate source commit is invalid")
        if request.dependency_lock_hash != self._policy.base_dependency_lock_hash:
            raise ReleaseBuildError(
                "candidate dependency lock changed; a trusted host rebuild is required"
            )
        environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"}
        head = self._git(workspace, "rev-parse", "--verify", "HEAD^{commit}")
        if head.decode("ascii", errors="ignore").strip().lower() != request.source_commit:
            raise ReleaseBuildError("candidate workspace no longer matches its evaluated commit")
        if self._git(
            workspace,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ).strip():
            raise ReleaseBuildError("candidate workspace changed after evaluation")
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
            cwd=workspace,
            env=environment,
            timeout_seconds=30,
            max_output_bytes=1_024,
        )
        if ancestor.returncode != 0 or ancestor.truncated:
            raise ReleaseBuildError("candidate base commit is not an ancestor of evaluated source")
        changed = self._git(
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
        paths = tuple(
            path for path in changed.decode("utf-8", errors="replace").split("\0") if path
        )
        if any(not candidate_path_is_promotable(path) for path in paths):
            raise ReleaseBuildError(
                "candidate changes are contribution-only and cannot enter a production release"
            )
        listing = self._git(
            workspace,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            request.source_commit,
            max_output_bytes=self._policy.max_tree_bytes,
        )
        tree_digest = hashlib.sha256(listing).hexdigest()
        manifest = {
            "artifact_kind": "source_overlay",
            "base_commit": request.base_commit,
            "candidate_id": request.candidate_id,
            "dependency_lock_hash": request.dependency_lock_hash,
            "entrypoint": list(self._policy.entrypoint),
            "evaluator_fingerprint": request.evaluator_fingerprint,
            "evaluator_version": request.evaluator_version,
            "protocol_version": 1,
            "source_commit": request.source_commit,
            "source_tree_sha256": tree_digest,
        }
        encoded = json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return OciReleaseArtifact(
            artifact_kind="source_overlay",
            artifact_digest=f"sha256:{tree_digest}",
            manifest_digest=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            image_reference=f"source-overlay:{request.source_commit}",
            entrypoint=self._policy.entrypoint,
        )

    def _git(
        self,
        workspace: Path,
        *arguments: str,
        max_output_bytes: int = 256 * 1024,
    ) -> bytes:
        result = run_bounded_process(
            (self._policy.git_cli, "-C", str(workspace), *arguments),
            cwd=workspace,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
            timeout_seconds=60,
            max_output_bytes=max_output_bytes,
        )
        if result.returncode != 0 or result.truncated:
            raise ReleaseBuildError("candidate source could not be verified")
        return result.output


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
        await asyncio.to_thread(self._verify_promotable_diff, request, workspace)

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
    ) -> None:
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

    def _export_commit(
        self,
        request: ReleaseBuildRequest,
        workspace: Path,
        context_root: Path,
    ) -> str:
        environment = {"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"}
        listing = run_bounded_process(
            (
                self._policy.git_cli,
                "-C",
                str(workspace),
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                request.source_commit,
            ),
            cwd=self._state_root,
            env=environment,
            timeout_seconds=60,
            max_output_bytes=self._policy.max_context_bytes,
        )
        if listing.returncode != 0 or listing.truncated:
            raise ReleaseBuildError("candidate commit tree could not be inspected safely")
        context_root.mkdir(parents=True, exist_ok=False, mode=0o700)
        digest = hashlib.sha256()
        total_bytes = 0
        entries = 0
        for raw_entry in listing.output.split(b"\0"):
            if not raw_entry:
                continue
            entries += 1
            if entries > self._policy.max_context_entries:
                raise ReleaseBuildError("candidate build context has too many entries")
            try:
                header, raw_path = raw_entry.split(b"\t", 1)
                mode, object_type, object_id = header.split(b" ", 2)
                path_text = raw_path.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise ReleaseBuildError("candidate commit tree entry was invalid") from exc
            if object_type != b"blob" or mode not in {b"100644", b"100755"}:
                raise ReleaseBuildError("candidate build context contains a link or special file")
            relative = PurePosixPath(path_text)
            self._validate_source_path(relative)
            remaining = self._policy.max_context_bytes - total_bytes
            blob = run_bounded_process(
                (
                    self._policy.git_cli,
                    "-C",
                    str(workspace),
                    "cat-file",
                    "blob",
                    object_id.decode("ascii"),
                ),
                cwd=self._state_root,
                env=environment,
                timeout_seconds=60,
                max_output_bytes=remaining + 1,
            )
            if blob.returncode != 0 or blob.truncated or len(blob.output) > remaining:
                raise ReleaseBuildError("candidate build context exceeds its byte limit")
            if candidate_content_contains_secret(relative.as_posix(), blob.output):
                raise ReleaseBuildError("candidate build context contains credential material")
            total_bytes += len(blob.output)
            digest.update(mode + b"\0" + raw_path + b"\0" + object_id + b"\0")
            digest.update(blob.output)
            digest.update(b"\0")
            destination_relative = (
                PurePosixPath(_DOCKERIGNORE_STAGING)
                if relative.as_posix() == ".dockerignore"
                else relative
            )
            destination = context_root.joinpath(*destination_relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            try:
                destination.write_bytes(blob.output)
            except OSError as exc:
                raise ReleaseBuildError(
                    "candidate build context could not be materialized"
                ) from exc
            destination.chmod(0o755 if mode == b"100755" else 0o644)
        # Candidate Docker ignore rules are restored inside the image, but cannot
        # alter which exact Git blobs enter the trusted build context.
        (context_root / ".dockerignore").write_bytes(b"")
        return digest.hexdigest()

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
            "RUN find /app -mindepth 1 -maxdepth 1 ! -name .venv "
            "-exec rm -rf -- '{}' +\n"
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


__all__ = [
    "OciReleaseArtifact",
    "OciReleaseBuildPolicy",
    "ReleaseBuildError",
    "ReleaseBuildRequest",
    "ReleaseBuilder",
    "SourceOverlayBuildPolicy",
    "TrustedOciReleaseBuilder",
    "TrustedSourceOverlayBuilder",
]
