"""Railway-native source evolution owned by the stable host process."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue, ValidationError

from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.evolution.activation import (
    ReleaseActivationResult,
    ReleaseActivationStatus,
)
from opentulpa.evolution.archive import EvolutionArchive
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
    OciReleaseArtifact,
    ReleaseBuildError,
    ReleaseBuildRequest,
)
from opentulpa.evolution.release_provenance import (
    ReleaseArtifactProvenance,
    live_repo_artifact_digest,
)
from opentulpa.evolution.supervisor import EvolutionSupervisor
from opentulpa.host.runtime import (
    RuntimeLiveSourceSpec,
    RuntimeSupervisor,
    RuntimeUnavailableError,
)
from opentulpa.host.runtime_environment import (
    LiveSourceRuntimeEnvironmentStore,
    RuntimeEnvFileManager,
    RuntimeEnvironmentError,
)

_RUNTIME_ENVIRONMENT_INVARIANT_KEYS = (
    "runtime_dependency_lock_hash",
    "runtime_pyproject_sha256",
    "runtime_install_profile",
)


class RuntimeEvolutionEventSink:
    """Deliver durable evolution events through the exact serving child identity."""

    def __init__(self, runtime: RuntimeSupervisor) -> None:
        self._runtime = runtime

    async def deliver(self, event: EvolutionEvent) -> None:
        await self._runtime.deliver_evolution_event(event)


class InitialReleaseProvider(Protocol):
    async def build(self) -> ReleaseRecord: ...

    def source_commit(self) -> str: ...


class HostEvolutionControlService:
    """Host-only internal API facade for source evolution plus runtime environment writes."""

    def __init__(
        self,
        *,
        evolution: EvolutionSupervisor,
        runtime_env_file_manager: RuntimeEnvFileManager | None = None,
    ) -> None:
        self._evolution = evolution
        self._runtime_env_file_manager = runtime_env_file_manager

    def __getattr__(self, name: str) -> object:
        return getattr(self._evolution, name)

    async def source_set_runtime_env(
        self,
        *,
        name: str,
        value: str,
        idempotency_key: str,
        audit_context: dict[str, str] | None = None,
    ) -> dict[str, JsonValue]:
        manager = self._runtime_env_file_manager
        if manager is None:
            return {
                "status": "failed",
                "name": str(name or "")[:128],
                "changed": False,
                "restarted": False,
                "rollback_restored": True,
                "failure_stage": "env_write",
                "error": {
                    "code": "runtime_env_update_unavailable",
                    "message": "Runtime .env updates are unavailable in this deployment.",
                    "retryable": False,
                },
                "value": "[redacted]",
            }
        return await manager.set(
            name=name,
            value=value,
            idempotency_key=idempotency_key,
            audit_context=audit_context,
        )


class TrustedLiveRepoReleaseBuilder:
    """Treat an evaluated Git commit as the release artifact."""

    entrypoint: tuple[str, ...] = ("python", "-P", "-m", "opentulpa")

    def __init__(
        self,
        *,
        runtime_environment_store: LiveSourceRuntimeEnvironmentStore | None = None,
    ) -> None:
        self._runtime_environment_store = runtime_environment_store

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        return await asyncio.to_thread(self._build, request)

    def _build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        workspace = request.workspace.expanduser().resolve(strict=True)
        try:
            resolved = _git(
                workspace,
                "rev-parse",
                "--verify",
                f"{request.source_commit}^{{commit}}",
            ).stdout.strip()
            if resolved != request.source_commit:
                raise ReleaseBuildError("candidate source commit is not exact")
            if _git(
                workspace,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ).output:
                raise ReleaseBuildError("candidate source checkout is dirty")
        except ReleaseBuildError:
            raise
        except Exception as exc:
            raise ReleaseBuildError("candidate source commit could not be verified") from exc
        digest = live_repo_artifact_digest(request.source_commit)
        metadata: dict[str, JsonValue] = {}
        if self._runtime_environment_store is not None:
            try:
                runtime_environment = self._runtime_environment_store.prepare(
                    request.source_commit,
                    workspace=workspace,
                )
            except RuntimeEnvironmentError as exc:
                raise ReleaseBuildError(f"{exc.code}: {exc.public_message}") from exc
            metadata.update(runtime_environment.release_metadata())
        return OciReleaseArtifact(
            artifact_kind="live_repo",
            artifact_digest=digest,
            manifest_digest=digest,
            image_reference=f"git-commit:{request.source_commit}",
            entrypoint=self.entrypoint,
            metadata=metadata,
        )


class TrustedLiveRepoReleaseProvider:
    """Seed the initial release from the checked-out trusted source commit."""

    def __init__(
        self,
        *,
        source_repository: Path,
        evaluator_version: str,
        evaluator_fingerprint: str | Callable[[], str],
        runtime_environment_store: LiveSourceRuntimeEnvironmentStore | None = None,
    ) -> None:
        self._repository = source_repository.expanduser().resolve(strict=True)
        self._evaluator_version = evaluator_version
        self._evaluator_fingerprint = evaluator_fingerprint
        self._runtime_environment_store = runtime_environment_store

    async def build(self) -> ReleaseRecord:
        source_commit, lock_hash = await asyncio.to_thread(self._source_identity)
        evaluator_fingerprint = self._current_evaluator_fingerprint()
        evaluation_input = hashlib.sha256(
            (
                f"{source_commit}:{lock_hash}:{self._evaluator_version}:"
                f"{evaluator_fingerprint}"
            ).encode()
        ).hexdigest()
        digest = live_repo_artifact_digest(source_commit)
        suffix = digest.removeprefix("sha256:")[:16]
        runtime_metadata: dict[str, JsonValue] = {}
        if self._runtime_environment_store is not None:
            try:
                runtime_environment = await asyncio.to_thread(
                    self._runtime_environment_store.prepare,
                    source_commit,
                )
            except RuntimeEnvironmentError as exc:
                raise RuntimeError(exc.public_message) from exc
            runtime_metadata.update(runtime_environment.release_metadata())
        return ReleaseRecord(
            id=f"release_initial_{suffix}",
            candidate_id=f"bootstrap_live_repo_{source_commit[:16]}",
            source_commit=source_commit,
            artifact_digest=digest,
            manifest_digest=digest,
            entrypoint=TrustedLiveRepoReleaseBuilder.entrypoint,
            metadata={
                "artifact_kind": "live_repo",
                "image_reference": f"git-commit:{source_commit}",
                **runtime_metadata,
                "initial": True,
                "bootstrap_initial": True,
                "dependency_lock_hash": lock_hash,
                "evaluator_version": self._evaluator_version,
                "evaluator_fingerprint": evaluator_fingerprint,
                "evaluation_input_digest": f"sha256:{evaluation_input}",
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
        return self._source_identity()[0]

    def _source_identity(self) -> tuple[str, str]:
        repository = prepare_live_source_repository(self._repository)
        _, common_directory = discover_git_directories(repository)
        with repository_mutation_lock(common_directory):
            source_commit = _git(repository, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
            lockfile = repository / "uv.lock"
            if lockfile.is_symlink() or not lockfile.is_file():
                raise RuntimeError("live source lockfile is unavailable")
            lock_hash = hashlib.sha256(lockfile.read_bytes()).hexdigest()
        return source_commit, lock_hash


class HostReleaseActivator:
    """Verify live source releases and health-check them through the host."""

    def __init__(
        self,
        *,
        runtime: RuntimeSupervisor,
        runtime_environment_store: LiveSourceRuntimeEnvironmentStore | None = None,
    ) -> None:
        self._runtime = runtime
        self._runtime_environment_store = runtime_environment_store
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
                live_spec = await self._prepared_live_source_spec(release)
                await self.verify_live_source(live_spec)
                if self._runtime.live_source != live_spec or self._runtime.status != "ready":
                    await self._runtime.replace_live_source(live_spec)
            except RuntimeEnvironmentError as exc:
                return ReleaseActivationResult(
                    activation_id=activation_id,
                    status=ReleaseActivationStatus.FAILED,
                    failure_code=exc.code,
                    failure_message=exc.public_message,
                )
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

    async def prepare_runtime(self, release: Release | ReleaseRecord) -> None:
        spec = await self._prepared_live_source_spec(release)
        await self.verify_live_source(spec)
        self._runtime.set_live_source(spec)

    def live_source_spec(self, release: Release | ReleaseRecord) -> RuntimeLiveSourceSpec:
        provenance = _release_artifact_provenance(release)
        if provenance.artifact_kind != "live_repo":
            raise RuntimeError("release is not a live repo commit")
        return RuntimeLiveSourceSpec.from_release_metadata(
            release.metadata,
            source_commit=provenance.source_commit,
        )

    async def _prepared_live_source_spec(
        self,
        release: Release | ReleaseRecord,
    ) -> RuntimeLiveSourceSpec:
        spec = self.live_source_spec(release)
        store = self._runtime_environment_store
        if store is None:
            return spec
        environment = await asyncio.to_thread(store.prepare, spec.source_commit)
        metadata = environment.release_metadata()
        if spec.has_runtime_environment:
            recorded = spec.model_dump(mode="json")
            if any(recorded.get(key) != metadata.get(key) for key in _RUNTIME_ENVIRONMENT_INVARIANT_KEYS):
                raise RuntimeEnvironmentError(
                    "runtime_environment_provenance_mismatch",
                    "Runtime dependency environment provenance is inconsistent.",
                    stage="dependency_install",
                )
        return RuntimeLiveSourceSpec.model_validate(
            {**spec.model_dump(mode="json"), **metadata}
        )

    async def verify_live_source(self, spec: RuntimeLiveSourceSpec) -> RuntimeLiveSourceSpec:
        return await self._runtime.verify_live_source(spec)

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
        control_service: HostEvolutionControlService | None = None,
    ) -> None:
        self._runtime = runtime
        self._archive = archive
        self._evolution = evolution
        self._activator = activator
        self._initial_release = initial_release
        self._control_service = control_service or HostEvolutionControlService(evolution=evolution)
        self._prepared = False
        self._started = False

    @property
    def service(self) -> HostEvolutionControlService:
        return self._control_service

    async def prepare(self) -> None:
        if self._prepared:
            return
        await self._archive.start()
        current = await self._archive.get_current_release()
        if current is None or _requires_live_repo_initial_seed(current):
            bundled = await self._initial_release.build()
            if current is None or not _same_release_provenance(current, bundled):
                await self._seed_initial_lineage(bundled)
                current = await self._archive.get_current_release()
        if current is None:
            raise RuntimeError("host evolution has no active source release")
        await self._activator.prepare_runtime(current)
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
    except (ValidationError, ValueError, TypeError):
        return False


def _requires_live_repo_initial_seed(release: Release | ReleaseRecord) -> bool:
    if bool(release.metadata.get("bootstrap_initial")):
        return True
    try:
        _release_artifact_provenance(release)
    except (ValidationError, ValueError, TypeError):
        return True
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


def prepare_live_source_repository(repository: Path) -> Path:
    """Validate the mounted source checkout and seed local lineage refs if needed."""

    source = repository.expanduser().resolve(strict=True)
    if source.is_symlink() or not (source / "uv.lock").is_file():
        raise RuntimeError("live source repository is unavailable")
    if not (source / "src" / "opentulpa" / "__init__.py").is_file():
        raise RuntimeError("live source repository is not an OpenTulpa checkout")
    try:
        _, common_directory = discover_git_directories(source)
    except Exception as exc:
        raise RuntimeError("live source Git metadata is unavailable") from exc
    with repository_mutation_lock(common_directory):
        if _status_without_untracked_runtime_env(
            _git(source, "status", "--porcelain=v1", "--untracked-files=all", "-z").output
        ):
            raise RuntimeError("live source repository is unexpectedly dirty")
        head = _git(source, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
        if _git(source, "cat-file", "-e", f"{head}:.env", check=False).returncode == 0:
            raise RuntimeError("live source repository must not commit .env")
        upstream = _git(
            source,
            "rev-parse",
            "--verify",
            "refs/heads/upstream^{commit}",
            check=False,
        )
        if upstream.returncode != 0:
            _git(source, "branch", "upstream", head)
    return source


def _status_without_untracked_runtime_env(status: bytes) -> bytes:
    dirty: list[bytes] = []
    parts = status.split(b"\0")
    index = 0
    while index < len(parts):
        entry = parts[index]
        index += 1
        if not entry:
            continue
        code = entry[:2]
        path = entry[3:] if len(entry) > 3 else b""
        if code == b"??" and path == b".env":
            continue
        dirty.append(entry)
        if (code[:1] in {b"R", b"C"} or code[1:2] in {b"R", b"C"}) and index < len(parts):
            dirty.append(parts[index])
            index += 1
    return b"\0".join(dirty)


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
    "HostEvolutionControlService",
    "HostEvolutionRuntime",
    "RuntimeEvolutionEventSink",
    "TrustedLiveRepoReleaseBuilder",
    "TrustedLiveRepoReleaseProvider",
    "prepare_live_source_repository",
]
