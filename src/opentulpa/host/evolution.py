"""Railway-native source evolution owned by the stable host process."""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import shutil
import stat
import tarfile
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from pydantic import JsonValue

from opentulpa.bootstrap.evolution_runtime import InitialReleaseProvider
from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.evolution.activation import (
    ReleaseActivationResult,
    ReleaseActivationStatus,
)
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.generation import StateContract
from opentulpa.evolution.generation_store import GenerationStore, InstalledGeneration
from opentulpa.evolution.git_security import (
    discover_git_directories,
    repository_mutation_lock,
    run_hardened_git,
)
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvaluationCheck,
    EvaluationReport,
    EvolutionEvent,
    Release,
)
from opentulpa.evolution.release_builder import (
    ReleaseBuilder,
    ReleaseBuildRequest,
)
from opentulpa.evolution.release_provenance import ReleaseArtifactProvenance
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.host.runtime import (
    RuntimeGenerationSpec,
    RuntimeSupervisor,
    RuntimeUnavailableError,
)


class RuntimeEvolutionEventSink:
    """Deliver durable evolution events through the exact serving child identity."""

    def __init__(self, runtime: RuntimeSupervisor) -> None:
        self._runtime = runtime

    async def deliver(self, event: EvolutionEvent) -> None:
        await self._runtime.deliver_evolution_event(event)


def seed_source_repository(*, seed_root: Path, repository: Path) -> Path:
    """Import the source bundled with the image into persistent local Git history."""

    raw_source = seed_root.expanduser()
    if raw_source.is_symlink():
        raise RuntimeError("bundled evolution source is unavailable")
    source = raw_source.resolve(strict=True)
    if not (source / "uv.lock").is_file():
        raise RuntimeError("bundled evolution source is unavailable")
    _validate_seed_tree(source)
    destination = repository.expanduser().absolute()
    if destination.is_symlink():
        raise RuntimeError("persistent evolution source repository is invalid")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not destination.exists():
        staging = destination.parent / f".{destination.name}.initialize-{uuid4().hex}"
        staging.mkdir(mode=0o700)
        try:
            _copy_seed(source, staging)
            _git(staging, "init")
            _git(staging, "config", "user.name", "OpenTulpa Host")
            _git(staging, "config", "user.email", "host@opentulpa.local")
            _git(staging, "checkout", "-B", "upstream")
            _git(staging, "add", "--all")
            _git(staging, "commit", "--no-gpg-sign", "--no-verify", "-m", "Bundled source")
            _git(staging, "checkout", "--detach")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return destination
    if destination.is_symlink() or not (destination / ".git").is_dir():
        raise RuntimeError("persistent evolution source repository is invalid")
    _, common_directory = discover_git_directories(destination)
    with repository_mutation_lock(common_directory):
        _cleanup_seed_imports(destination)
        if _git(destination, "status", "--porcelain=v1", "--untracked-files=all").strip():
            raise RuntimeError("persistent evolution source repository is unexpectedly dirty")
        previous = _git(
            destination,
            "rev-parse",
            "--verify",
            "refs/heads/upstream^{commit}",
        ).stdout.strip()
        symbolic = _git(destination, "symbolic-ref", "-q", "HEAD", check=False)
        if symbolic.returncode == 0:
            if symbolic.stdout.strip() != "refs/heads/upstream":
                raise RuntimeError("persistent evolution source HEAD is unexpected")
            if _git(destination, "rev-parse", "HEAD").stdout.strip() != previous:
                raise RuntimeError("persistent evolution source HEAD is inconsistent")
            _git(destination, "checkout", "--detach", previous)

        imports_root = destination.parent / "seed-imports"
        imports_root.mkdir(mode=0o700, exist_ok=True)
        target = imports_root / uuid4().hex
        _git(destination, "worktree", "add", "--detach", str(target), previous)
        try:
            _replace_working_tree(source, target)
            _git(target, "add", "--all")
            if _git(target, "diff", "--cached", "--quiet", check=False).returncode == 0:
                return destination
            _git(
                target,
                "commit",
                "--no-gpg-sign",
                "--no-verify",
                "-m",
                "Import bundled source update",
            )
            updated = _git(target, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
            _git(destination, "update-ref", "refs/heads/upstream", updated, previous)
        finally:
            _git(destination, "worktree", "remove", "--force", str(target), check=False)
            if target.exists():
                shutil.rmtree(target)
    return destination


class TrustedGenerationReleaseProvider:
    """Build the bundled upstream commit with the fixed Python generation recipe."""

    def __init__(
        self,
        *,
        source_repository: Path,
        worktrees_root: Path,
        builder: ReleaseBuilder,
        evaluator_version: str,
        evaluator_fingerprint: str | Callable[[], str],
        state_contract: StateContract,
        install_profile: str,
    ) -> None:
        self._repository = source_repository.expanduser().resolve(strict=True)
        self._worktrees_root = worktrees_root.expanduser().resolve()
        self._worktrees_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._builder = builder
        self._evaluator_version = evaluator_version
        self._evaluator_fingerprint = evaluator_fingerprint
        self._state_contract = state_contract
        self._install_profile = install_profile

    async def build(self) -> ReleaseRecord:
        workspace, source_commit, lock_hash = await asyncio.to_thread(self._prepare_workspace)
        evaluator_fingerprint = self._current_evaluator_fingerprint()
        evaluation_input = hashlib.sha256(
            (
                f"{source_commit}:{lock_hash}:{self._evaluator_version}:"
                f"{evaluator_fingerprint}"
            ).encode()
        ).hexdigest()
        try:
            artifact = await self._builder.build(
                ReleaseBuildRequest(
                    candidate_id=f"bootstrap_generation_{source_commit[:16]}",
                    workspace=workspace,
                    base_commit=source_commit,
                    source_commit=source_commit,
                    dependency_lock_hash=lock_hash,
                    evaluator_version=self._evaluator_version,
                    evaluator_fingerprint=evaluator_fingerprint,
                    evaluation_input_sha256=evaluation_input,
                )
            )
        finally:
            await asyncio.to_thread(self._remove_workspace, workspace)
        suffix = artifact.artifact_digest.removeprefix("sha256:")[:16]
        candidate_id = f"bootstrap_generation_{source_commit[:12]}_{suffix}"
        generation_id = artifact.image_reference.removeprefix("python-generation:")
        return ReleaseRecord(
            id=f"release_initial_{suffix}",
            candidate_id=candidate_id,
            source_commit=source_commit,
            artifact_digest=artifact.artifact_digest,
            manifest_digest=artifact.manifest_digest,
            entrypoint=artifact.entrypoint,
            metadata={
                "artifact_kind": artifact.artifact_kind,
                "image_reference": artifact.image_reference,
                "generation_id": generation_id,
                "initial": True,
                "bootstrap_initial": True,
                "dependency_lock_hash": lock_hash,
                "evaluator_version": self._evaluator_version,
                "evaluator_fingerprint": evaluator_fingerprint,
                "state_contract_sha256": self._state_contract.sha256(),
                "controller_protocol": self._state_contract.runtime_protocol,
                "install_profile": self._install_profile,
                "base_commit": source_commit,
                "changed_paths": [],
                "diff_sha256": hashlib.sha256(b"").hexdigest(),
            },
        )

    def _current_evaluator_fingerprint(self) -> str:
        value = (
            self._evaluator_fingerprint()
            if callable(self._evaluator_fingerprint)
            else self._evaluator_fingerprint
        )
        safe_value = str(value or "").strip()
        if not safe_value:
            raise RuntimeError("initial release evaluator fingerprint is unavailable")
        return safe_value

    def source_commit(self) -> str:
        """Return the exact bundled upstream commit without rebuilding a generation."""

        _, common_directory = discover_git_directories(self._repository)
        with repository_mutation_lock(common_directory):
            value = _git(
                self._repository,
                "rev-parse",
                "--verify",
                "refs/heads/upstream^{commit}",
            ).stdout.strip()
        return str(value)

    def _prepare_workspace(self) -> tuple[Path, str, str]:
        _, common_directory = discover_git_directories(self._repository)
        with repository_mutation_lock(common_directory):
            source_commit = _git(
                self._repository,
                "rev-parse",
                "--verify",
                "refs/heads/upstream^{commit}",
            ).stdout.strip()
            workspace = self._worktrees_root / f"initial-{uuid4().hex}"
            _git(
                self._repository,
                "worktree",
                "add",
                "--detach",
                str(workspace),
                source_commit,
            )
        lockfile = workspace / "uv.lock"
        if lockfile.is_symlink() or not lockfile.is_file():
            self._remove_workspace(workspace)
            raise RuntimeError("bundled generation lockfile is unavailable")
        lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
        return workspace, source_commit, lock_hash

    def _remove_workspace(self, workspace: Path) -> None:
        _, common_directory = discover_git_directories(self._repository)
        with repository_mutation_lock(common_directory):
            _git(
                self._repository,
                "worktree",
                "remove",
                "--force",
                str(workspace),
                check=False,
            )
            if workspace.exists():
                shutil.rmtree(workspace)


class HostReleaseActivator:
    """Verify immutable generations and health-check them through the host."""

    def __init__(
        self,
        *,
        runtime: RuntimeSupervisor,
        generation_store: GenerationStore | None = None,
        state_contract: StateContract | None = None,
        evaluator_fingerprint: str | None = None,
        evaluator_fingerprint_resolver: Callable[[Release | ReleaseRecord], str] | None = None,
        install_profile: str = "runtime",
        controller_protocol: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._generation_store = generation_store
        self._state_contract = state_contract
        self._evaluator_fingerprint = str(evaluator_fingerprint or "")
        self._evaluator_fingerprint_resolver = evaluator_fingerprint_resolver
        self._install_profile = install_profile
        self._controller_protocol = controller_protocol
        self._lock = asyncio.Lock()

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
                spec = self.generation_spec(release)
                await self.verify_generation(
                    spec,
                    expected_source_commit=release.source_commit,
                )
                if self._runtime.generation != spec or self._runtime.status != "ready":
                    await self._runtime.replace_generation(spec)
            except RuntimeUnavailableError:
                if (
                    self._runtime.status == "ready"
                    and getattr(self._runtime, "endpoint", None) is not None
                ):
                    return ReleaseActivationResult(
                        activation_id=activation_id,
                        status=ReleaseActivationStatus.ROLLED_BACK,
                        failure_code="release_unhealthy",
                        failure_message="The candidate failed its health check; the previous release was restored.",
                    )
                if self._runtime.status == "recovery_required":
                    failure_code = "release_containment_failed"
                    failure_message = "The candidate could not be contained; runtime recovery is required."
                elif self._runtime.status == "failed":
                    failure_code = "release_rollback_failed"
                    failure_message = "The candidate failed and the previous release could not be restored."
                else:
                    failure_code = "release_runtime_unavailable"
                    failure_message = "The candidate failed and no previous release is serving."
                return ReleaseActivationResult(
                    activation_id=activation_id,
                    status=ReleaseActivationStatus.FAILED,
                    failure_code=failure_code,
                    failure_message=failure_message,
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

    def generation_spec(self, release: Release | ReleaseRecord) -> RuntimeGenerationSpec:
        self.artifact_kind(release)
        provenance = _release_artifact_provenance(release)
        if provenance.artifact_kind != "python_generation" or provenance.generation_id is None:
            raise RuntimeError("release is not a Python generation")
        store = self._generation_store
        contract = self._state_contract
        protocol = self._controller_protocol
        if store is None or contract is None or protocol is None or not self._evaluator_fingerprint:
            raise RuntimeError("Python generation activation is not configured")
        expected_evaluator_fingerprint = self._evaluator_fingerprint
        if self._evaluator_fingerprint_resolver is not None:
            expected_evaluator_fingerprint = self._evaluator_fingerprint_resolver(release)
        if not expected_evaluator_fingerprint:
            raise RuntimeError("Python generation evaluator provenance is unavailable")
        expected_values = {
            "evaluator_fingerprint": (
                provenance.evaluator_fingerprint,
                expected_evaluator_fingerprint,
            ),
            "state_contract_sha256": (
                provenance.state_contract_sha256,
                contract.sha256(),
            ),
            "install_profile": (provenance.install_profile, self._install_profile),
            "controller_protocol": (provenance.controller_protocol, protocol),
        }
        for key, (recorded, expected) in expected_values.items():
            if recorded != expected:
                raise RuntimeError(f"Python generation {key} provenance is inconsistent")
        return RuntimeGenerationSpec(
            generation_id=provenance.generation_id,
            expected_manifest_digest=provenance.manifest_digest,
            expected_state_contract_digest=contract.sha256(),
            expected_evaluator_fingerprint=expected_evaluator_fingerprint,
            expected_install_profile=self._install_profile,
            controller_protocol=protocol,
        )

    async def verify_generation(
        self,
        spec: RuntimeGenerationSpec,
        *,
        expected_source_commit: str,
    ) -> InstalledGeneration:
        store = self._generation_store
        if store is None:
            raise RuntimeError("Python generation storage is unavailable")
        installed = await asyncio.to_thread(
            store.open,
            spec.generation_id,
            expected_manifest_digest=spec.expected_manifest_digest,
            expected_state_contract_digest=spec.expected_state_contract_digest,
            expected_evaluator_fingerprint=spec.expected_evaluator_fingerprint,
            expected_install_profile=spec.expected_install_profile,
            controller_protocol=spec.controller_protocol,
        )
        if installed.manifest.identity.source_commit != expected_source_commit:
            raise RuntimeError("Python generation source provenance is inconsistent")
        return installed

    @staticmethod
    def artifact_kind(release: Release | ReleaseRecord) -> str:
        kind = str(release.metadata.get("artifact_kind") or "")
        if kind != "python_generation":
            raise RuntimeError("host releases must be immutable Python generations")
        return kind


class HostEvolutionRuntime:
    """Prepare active source before the child starts, then own promotion dispatch."""

    def __init__(
        self,
        *,
        runtime: RuntimeSupervisor,
        archive: EvolutionArchive,
        evolution: EvolutionSupervisor,
        activator: HostReleaseActivator,
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
        if current is None or bool(current.metadata.get("bootstrap_initial")):
            bundled = await self._initial_release.build()
            if current is None or not _same_release_provenance(current, bundled):
                await self._seed_initial_lineage(bundled)
                current = await self._archive.get_current_release()
        if current is None:
            raise RuntimeError("host evolution has no active source release")
        generation = self._activator.generation_spec(current)
        await self._activator.verify_generation(
            generation,
            expected_source_commit=current.source_commit,
        )
        self._runtime.set_generation(generation)
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
                        **release.metadata,
                        "artifact_kind": str(release.metadata.get("artifact_kind") or ""),
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
                **release.metadata,
                "artifact_kind": str(release.metadata.get("artifact_kind") or ""),
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


def _same_release_provenance(left: Release | ReleaseRecord, right: ReleaseRecord) -> bool:
    try:
        return _release_artifact_provenance(left) == _release_artifact_provenance(right)
    except (ValueError, TypeError):
        return False


def _release_artifact_provenance(
    release: Release | ReleaseRecord,
) -> ReleaseArtifactProvenance:
    manifest_digest = str(
        getattr(release, "manifest_digest", "")
        or release.metadata.get("manifest_digest")
        or ""
    )
    recorded_entrypoint = getattr(release, "entrypoint", ())
    if recorded_entrypoint:
        entrypoint = tuple(recorded_entrypoint)
    else:
        raw_entrypoint = release.metadata.get("release_entrypoint")
        if not isinstance(raw_entrypoint, list) or not all(
            isinstance(value, str) for value in raw_entrypoint
        ):
            raise ValueError("release entrypoint provenance is invalid")
        entrypoint = tuple(raw_entrypoint)
    return ReleaseArtifactProvenance.from_values(
        source_commit=release.source_commit,
        artifact_digest=release.artifact_digest,
        manifest_digest=manifest_digest,
        entrypoint=entrypoint,
        metadata=release.metadata,
    )


def _validate_seed_tree(root: Path) -> None:
    root_metadata = root.lstat()
    if (
        stat.S_ISLNK(root_metadata.st_mode)
        or not stat.S_ISDIR(root_metadata.st_mode)
        or root_metadata.st_uid != os.geteuid()
        or stat.S_IMODE(root_metadata.st_mode) & 0o022
    ):
        raise RuntimeError("bundled source root ownership or permissions are unsafe")
    entries = 0
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path == root and ".git" in directory_names:
            directory_names.remove(".git")
        if directory_path == root and ".git" in file_names:
            file_names.remove(".git")
        for name in [*directory_names, *file_names]:
            if name == ".git":
                raise RuntimeError("bundled source contains Git administration data")
            entries += 1
            if entries > 100_000:
                raise RuntimeError("bundled source has too many entries")
            path = Path(directory) / name
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise RuntimeError("bundled source contains an unsafe entry")
            if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o022:
                raise RuntimeError("bundled source ownership or permissions are unsafe")
            if stat.S_ISREG(metadata.st_mode):
                if metadata.st_nlink != 1:
                    raise RuntimeError("bundled source contains a hard-linked file")
                total += metadata.st_size
                if total > 512 * 1024 * 1024:
                    raise RuntimeError("bundled source exceeds its size limit")


def _copy_seed(source: Path, destination: Path) -> None:
    if os.path.lexists(source / ".git"):
        _copy_git_seed(source, destination)
        return
    _validate_seed_tree(source)
    before = _seed_tree_digest(source)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        copy_function=shutil.copy2,
        ignore=shutil.ignore_patterns(".git"),
    )
    after = _seed_tree_digest(source)
    copied = _seed_tree_digest(destination)
    if before != after or before != copied:
        raise RuntimeError("bundled source changed while it was imported")


def _copy_git_seed(source: Path, destination: Path) -> None:
    _, common_directory = discover_git_directories(source)
    with repository_mutation_lock(common_directory):
        _validate_seed_tree(source)
        if _git(source, "status", "--porcelain=v1", "--untracked-files=all", "-z").output:
            raise RuntimeError("bundled Git source must be exactly clean")
        commit = _git(source, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
        archive = _git(
            source,
            "archive",
            "--format=tar",
            commit,
            max_output_bytes=512 * 1024 * 1024,
        ).output
        _extract_seed_archive(archive, destination)
        if (
            _git(source, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip() != commit
            or _git(source, "status", "--porcelain=v1", "--untracked-files=all", "-z").output
        ):
            raise RuntimeError("bundled Git source changed while it was imported")


def _extract_seed_archive(archive: bytes, destination: Path) -> None:
    entries = 0
    total_bytes = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as source:
            for member in source:
                entries += 1
                if entries > 100_000:
                    raise RuntimeError("bundled Git source has too many entries")
                relative = _safe_seed_archive_path(member.name)
                path = destination.joinpath(*relative.parts)
                if member.isdir():
                    if path.exists():
                        if path.is_symlink() or not path.is_dir():
                            raise RuntimeError("bundled Git source directory conflicts")
                    else:
                        path.mkdir(parents=True, mode=0o700)
                    continue
                if not member.isreg() or member.size < 0:
                    raise RuntimeError("bundled Git source contains a link or special file")
                total_bytes += member.size
                if total_bytes > 512 * 1024 * 1024:
                    raise RuntimeError("bundled Git source exceeds its size limit")
                path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                stream = source.extractfile(member)
                if stream is None:
                    raise RuntimeError("bundled Git source file is unavailable")
                payload = stream.read(member.size + 1)
                if len(payload) != member.size:
                    raise RuntimeError("bundled Git source file size is inconsistent")
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(payload)
                path.chmod(0o700 if member.mode & 0o111 else 0o600)
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError("bundled Git source archive could not be imported") from exc


def _safe_seed_archive_path(value: str) -> Path:
    if not value or "\x00" in value or "\\" in value:
        raise RuntimeError("bundled Git source archive path is invalid")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError("bundled Git source archive path is invalid")
    return path


def _seed_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    entries = 0
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        if directory_path == root and ".git" in directory_names:
            directory_names.remove(".git")
        if directory_path == root and ".git" in file_names:
            file_names.remove(".git")
        directory_names.sort()
        file_names.sort()
        for name in [*directory_names, *file_names]:
            path = directory_path / name
            metadata = path.lstat()
            relative = path.relative_to(root).as_posix().encode()
            entries += 1
            if entries > 100_000:
                raise RuntimeError("bundled source has too many entries")
            if stat.S_ISDIR(metadata.st_mode):
                digest.update(b"d\0" + relative + b"\0")
                continue
            if (
                stat.S_ISLNK(metadata.st_mode)
                or not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
            ):
                raise RuntimeError("bundled source contains an unsafe entry")
            total_bytes += metadata.st_size
            if total_bytes > 512 * 1024 * 1024:
                raise RuntimeError("bundled source exceeds its size limit")
            digest.update(b"f\0" + relative + b"\0")
            digest.update(f"{stat.S_IMODE(metadata.st_mode):04o}".encode() + b"\0")
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")
    return digest.hexdigest()


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


def _cleanup_seed_imports(repository: Path) -> None:
    _git(repository, "worktree", "prune", "--expire", "now")
    imports_root = repository.parent / "seed-imports"
    if not imports_root.exists():
        return
    if imports_root.is_symlink() or not imports_root.is_dir():
        raise RuntimeError("persistent source import workspace is invalid")
    entries = tuple(os.scandir(imports_root))
    if len(entries) > 32:
        raise RuntimeError("persistent source has too many stale import worktrees")
    for entry in entries:
        path = Path(entry.path)
        metadata = entry.stat(follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("persistent source import workspace is invalid")
        _git(repository, "worktree", "remove", "--force", str(path), check=False)
        if path.exists():
            shutil.rmtree(path)


def _git(
    repository: Path,
    *arguments: str,
    check: bool = True,
    max_output_bytes: int = 10 * 1024 * 1024,
):  # type: ignore[no-untyped-def]
    result = run_hardened_git(
        repository,
        tuple(arguments),
        env={
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
    "HostReleaseActivator",
    "HostEvolutionRuntime",
    "RuntimeEvolutionEventSink",
    "TrustedGenerationReleaseProvider",
    "seed_source_repository",
]
