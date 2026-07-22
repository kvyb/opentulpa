"""Railway-native source evolution owned by the stable host process."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import stat
from pathlib import Path

from pydantic import JsonValue

from opentulpa.bootstrap.evolution_runtime import InitialReleaseProvider
from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.evolution.activation import (
    ReleaseActivationResult,
    ReleaseActivationStatus,
)
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvaluationCheck,
    EvaluationReport,
    Release,
)
from opentulpa.evolution.process import run_bounded_process
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.host.runtime import RuntimeSupervisor, RuntimeUnavailableError


def seed_source_repository(*, seed_root: Path, repository: Path) -> Path:
    """Import the source bundled with the image into persistent local Git history."""

    source = seed_root.expanduser().resolve(strict=True)
    if source.is_symlink() or not (source / "uv.lock").is_file():
        raise RuntimeError("bundled evolution source is unavailable")
    _validate_seed_tree(source)
    destination = repository.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination.exists():
        destination.mkdir(mode=0o700)
        _copy_seed(source, destination)
        _git(destination, "init")
        _git(destination, "config", "user.name", "OpenTulpa Host")
        _git(destination, "config", "user.email", "host@opentulpa.local")
        _git(destination, "checkout", "-B", "upstream")
        _git(destination, "add", "--all")
        _git(destination, "commit", "--no-gpg-sign", "--no-verify", "-m", "Bundled source")
        return destination
    if destination.is_symlink() or not (destination / ".git").is_dir():
        raise RuntimeError("persistent evolution source repository is invalid")
    if _git(destination, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise RuntimeError("persistent evolution source repository is unexpectedly dirty")
    _replace_working_tree(source, destination)
    _git(destination, "add", "--all")
    if _git(destination, "diff", "--cached", "--quiet", check=False).returncode != 0:
        _git(
            destination,
            "commit",
            "--no-gpg-sign",
            "--no-verify",
            "-m",
            "Import bundled source update",
        )
    return destination


class SourceOverlayReleaseActivator:
    """Materialize a verified commit and health-check it through the stable host."""

    def __init__(
        self,
        *,
        repository: Path,
        releases_root: Path,
        runtime: RuntimeSupervisor,
    ) -> None:
        self._repository = repository.expanduser().resolve(strict=True)
        self._releases_root = releases_root.expanduser().resolve()
        self._releases_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._runtime = runtime
        self._lock = asyncio.Lock()

    async def materialize(self, release: Release | ReleaseRecord) -> Path:
        return await asyncio.to_thread(self._materialize, release)

    async def activate(
        self,
        release: ReleaseRecord,
        *,
        activation_id: str,
        origin: ReleaseOrigin | None,
        reason: str,
        rollback: bool,
    ) -> ReleaseActivationResult:
        del origin, reason, rollback
        async with self._lock:
            try:
                source_root = await self.materialize(release)
                if self._runtime.project_root != source_root or self._runtime.status != "ready":
                    await self._runtime.replace_source(source_root)
            except RuntimeUnavailableError:
                return ReleaseActivationResult(
                    activation_id=activation_id,
                    status=ReleaseActivationStatus.ROLLED_BACK,
                    failure_code="release_unhealthy",
                    failure_message="The candidate failed its health check; the previous release was restored.",
                )
            except Exception:
                return ReleaseActivationResult(
                    activation_id=activation_id,
                    status=ReleaseActivationStatus.FAILED,
                    failure_code="release_activation_failed",
                    failure_message="The verified source release could not be activated.",
                )
        return ReleaseActivationResult(
            activation_id=activation_id,
            status=ReleaseActivationStatus.ACTIVE,
        )

    def _materialize(self, release: Release | ReleaseRecord) -> Path:
        if str(release.metadata.get("artifact_kind") or "") != "source_overlay":
            raise RuntimeError("release is not a source overlay")
        expected_reference = f"source-overlay:{release.source_commit}"
        if str(release.metadata.get("image_reference") or "") != expected_reference:
            raise RuntimeError("source overlay reference is invalid")
        listing = _git(
            self._repository,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            release.source_commit,
            max_output_bytes=512 * 1024 * 1024,
        ).output
        digest = f"sha256:{hashlib.sha256(listing).hexdigest()}"
        if digest != release.artifact_digest:
            raise RuntimeError("source overlay digest failed verification")
        _git(self._repository, "worktree", "prune", "--expire", "now")
        target = self._releases_root / release.source_commit
        if target.exists():
            head = _git(target, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
            if (
                head == release.source_commit
                and not _git(
                    target, "status", "--porcelain=v1", "--untracked-files=all"
                ).stdout.strip()
            ):
                return target.resolve(strict=True)
            _git(self._repository, "worktree", "remove", "--force", str(target))
        _git(
            self._repository,
            "worktree",
            "add",
            "--detach",
            str(target),
            release.source_commit,
        )
        return target.resolve(strict=True)


class HostEvolutionRuntime:
    """Prepare active source before the child starts, then own promotion dispatch."""

    def __init__(
        self,
        *,
        runtime: RuntimeSupervisor,
        archive: EvolutionArchive,
        evolution: EvolutionSupervisor,
        activator: SourceOverlayReleaseActivator,
        initial_release: InitialReleaseProvider,
    ) -> None:
        self._runtime = runtime
        self._archive = archive
        self._evolution = evolution
        self._activator = activator
        self._initial_release = initial_release
        self._prepared = False
        self._started = False

    @property
    def service(self) -> EvolutionSupervisor:
        return self._evolution

    async def prepare(self) -> None:
        if self._prepared:
            return
        await self._archive.start()
        current = await self._archive.get_current_release()
        bundled = await self._initial_release.build()
        if current is None or (
            bool(current.metadata.get("bootstrap_initial"))
            and current.source_commit != bundled.source_commit
        ):
            await self._seed_initial_lineage(bundled)
            current = await self._archive.get_current_release()
        if current is None:
            raise RuntimeError("host evolution has no active source release")
        source_root = await self._activator.materialize(current)
        self._runtime.set_project_root(source_root)
        self._prepared = True

    async def start(self) -> None:
        if self._started:
            return
        if not self._prepared:
            await self.prepare()
        await self._evolution.start()
        self._started = True

    async def shutdown(self) -> None:
        if self._started:
            self._started = False
            await self._evolution.shutdown()
        else:
            await self._archive.shutdown()

    async def _seed_initial_lineage(self, release: ReleaseRecord) -> None:
        previous = await self._archive.get_current_release()
        empty_diff = hashlib.sha256(b"").hexdigest()
        fingerprint = str(release.metadata.get("evaluator_fingerprint") or release.manifest_digest)
        evaluator_version = str(
            release.metadata.get("evaluator_version") or "host-source-install-v1"
        )
        candidate = await self._archive.get_candidate(release.candidate_id)
        if candidate is None:
            candidate = await self._archive.create_candidate(
                Candidate(
                    id=release.candidate_id,
                    base_commit=release.source_commit,
                    requested_improvement="Trusted source bundled with the stable host",
                    source_commit=release.source_commit,
                    dependency_lock_hash=(
                        str(release.metadata.get("dependency_lock_hash") or "") or None
                    ),
                    artifact_digest=release.artifact_digest,
                    evaluator_fingerprint=fingerprint,
                    metadata={
                        "artifact_kind": "source_overlay",
                        "manifest_digest": release.manifest_digest,
                        "image_reference": str(release.metadata.get("image_reference") or ""),
                        "release_entrypoint": list(release.entrypoint),
                        "changed_paths": [],
                        "diff_sha256": empty_diff,
                        "bootstrap_initial": True,
                    },
                )
            )
        if (
            candidate.source_commit != release.source_commit
            or candidate.artifact_digest != release.artifact_digest
        ):
            raise RuntimeError("initial host source conflicts with persisted lineage")
        report = candidate.evaluation_report
        if report is None:
            report = EvaluationReport(
                candidate_id=candidate.id,
                source_commit=release.source_commit,
                artifact_digest=release.artifact_digest,
                evaluator_fingerprint=fingerprint,
                evaluator_version=evaluator_version,
                passed=True,
                checks=(
                    EvaluationCheck(
                        name="host.initial_source",
                        passed=True,
                        summary="Bundled source is bound to the host dependency lock.",
                    ),
                ),
                summary="Trusted bundled source installed successfully.",
            )
            candidate = await self._archive.append_evaluation(
                report,
                expected_revision=candidate.revision,
            )
        if candidate.status is CandidateStatus.BUILDING:
            candidate = await self._archive.transition_status(
                candidate.id,
                expected_status=CandidateStatus.BUILDING,
                new_status=CandidateStatus.READY,
                expected_revision=candidate.revision,
            )
        changed_paths: list[JsonValue] = []
        initial = Release(
            id=release.id,
            candidate_id=candidate.id,
            source_commit=release.source_commit,
            artifact_digest=release.artifact_digest,
            previous_release_id=previous.id if previous is not None else None,
            reason="Trusted host source installation",
            metadata={
                "artifact_kind": "source_overlay",
                "manifest_digest": release.manifest_digest,
                "image_reference": str(release.metadata.get("image_reference") or ""),
                "release_entrypoint": list(release.entrypoint),
                "base_commit": candidate.base_commit,
                "changed_paths": changed_paths,
                "diff_sha256": empty_diff,
                "evaluation_report_id": report.id,
                "evaluation_summary": report.summary,
                "evaluator_fingerprint": report.evaluator_fingerprint,
                "evaluator_version": report.evaluator_version,
                "activation_state": "active",
                "bootstrap_initial": True,
            },
        )
        if candidate.status is CandidateStatus.READY:
            await self._archive.promote_candidate(
                initial,
                expected_revision=candidate.revision,
            )
        elif candidate.status is CandidateStatus.PROMOTED:
            await self._archive.record_promotion(initial)
        else:
            raise RuntimeError("initial host source is not promotable")


def _validate_seed_tree(root: Path) -> None:
    entries = 0
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        for name in [*directory_names, *file_names]:
            entries += 1
            if entries > 100_000:
                raise RuntimeError("bundled source has too many entries")
            path = Path(directory) / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise RuntimeError("bundled source contains an unsafe entry")
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
                if total > 512 * 1024 * 1024:
                    raise RuntimeError("bundled source exceeds its size limit")


def _copy_seed(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True, copy_function=shutil.copy2)


def _replace_working_tree(source: Path, destination: Path) -> None:
    for entry in os.scandir(destination):
        if entry.name == ".git":
            continue
        path = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(path)
        else:
            path.unlink()
    _copy_seed(source, destination)


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    max_output_bytes: int = 10 * 1024 * 1024,
):  # type: ignore[no-untyped-def]
    result = run_bounded_process(
        ("git", "-C", str(repository), *arguments),
        cwd=repository,
        env={
            "HOME": "/tmp",
            "PATH": os.environ.get("PATH", os.defpath),
            "GIT_AUTHOR_NAME": "OpenTulpa Host",
            "GIT_AUTHOR_EMAIL": "host@opentulpa.local",
            "GIT_COMMITTER_NAME": "OpenTulpa Host",
            "GIT_COMMITTER_EMAIL": "host@opentulpa.local",
        },
        timeout_seconds=120,
        max_output_bytes=max_output_bytes,
    )
    if check and (result.returncode != 0 or result.truncated):
        raise RuntimeError("persistent source Git operation failed")
    return _GitResult(result.returncode, result.output)


class _GitResult:
    def __init__(self, returncode: int, output: bytes) -> None:
        self.returncode = returncode
        self.output = output
        self.stdout = output.decode("utf-8", errors="replace")

    def strip(self) -> bytes:
        return self.output.strip()


__all__ = [
    "HostEvolutionRuntime",
    "SourceOverlayReleaseActivator",
    "seed_source_repository",
]
