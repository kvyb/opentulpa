"""Stable ownership of source evolution and the initial mutable release."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import JsonValue

from opentulpa.bootstrap.models import OutboxEvent, ReleaseOrigin, ReleaseRecord
from opentulpa.bootstrap.supervisor import BootstrapSupervisor
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.candidate_provenance import CandidateSourceProvenance
from opentulpa.evolution.lineage import GitLineage, GitLineageError
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvaluationCheck,
    EvaluationReport,
    EvolutionEvent,
    Release,
)
from opentulpa.evolution.process import run_bounded_process
from opentulpa.evolution.release_builder import (
    ReleaseBuilder,
    ReleaseBuildRequest,
    TrustedWheelReleaseBuilder,
)
from opentulpa.evolution.release_provenance import ReleaseArtifactProvenance
from opentulpa.evolution.supervisor import EvolutionSupervisor

_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_GENERATION_REFERENCE_RE = re.compile(r"python-generation:([0-9a-f]{64})\Z")
class InitialReleaseProvider(Protocol):
    async def build(self) -> ReleaseRecord: ...


class TrustedSourceReleaseProvider:
    """Build the canonical checkout through its configured immutable recipe."""

    def __init__(
        self,
        *,
        source_repository: Path,
        builder: ReleaseBuilder,
        evaluator_version: str,
        evaluator_fingerprint: str,
        git_cli: str = "git",
        lineage: GitLineage | None = None,
    ) -> None:
        self._source = source_repository.expanduser().resolve()
        self._builder = builder
        self._evaluator_version = str(evaluator_version or "").strip()
        self._evaluator_fingerprint = str(evaluator_fingerprint or "").strip()
        self._git_cli = git_cli
        self._lineage = lineage

    async def build(self) -> ReleaseRecord:
        source_commit = self._source_commit()
        lock_hash = self._lock_hash()
        empty_diff_sha256 = hashlib.sha256(b"").hexdigest()
        evaluation_input_sha256 = self._evaluation_input_sha256(
            source_commit=source_commit,
            dependency_lock_hash=lock_hash,
        )
        artifact = await self._builder.build(
            ReleaseBuildRequest(
                candidate_id=f"bootstrap_{source_commit[:16]}",
                workspace=self._source,
                base_commit=source_commit,
                source_commit=source_commit,
                dependency_lock_hash=lock_hash,
                evaluator_version=self._evaluator_version,
                evaluator_fingerprint=self._evaluator_fingerprint,
                evaluation_input_sha256=evaluation_input_sha256,
            )
        )
        accepted_upstream = await self._accepted_upstream(source_commit)
        if artifact.artifact_kind not in {"oci_image", "python_generation", "live_repo"}:
            raise RuntimeError("trusted builder returned a mutable release artifact")
        generation = _GENERATION_REFERENCE_RE.fullmatch(artifact.image_reference)
        if artifact.artifact_kind == "python_generation" and generation is None:
            raise RuntimeError("trusted builder returned an invalid Python generation identity")
        artifact_metadata: dict[str, JsonValue] = {
            "artifact_kind": artifact.artifact_kind,
            "image_reference": artifact.image_reference,
            **artifact.metadata,
            **(
                {"generation_id": generation.group(1)}
                if artifact.artifact_kind == "python_generation" and generation is not None
                else {}
            ),
        }
        source_metadata = CandidateSourceProvenance(
            changed_paths=(),
            diff_sha256=empty_diff_sha256,
            accepted_upstream_commit=accepted_upstream,
        ).apply_to_metadata({})
        if isinstance(self._builder, TrustedWheelReleaseBuilder):
            policy = self._builder._policy
            artifact_metadata.update(
                {
                    "state_contract": policy.state_contract.model_dump(mode="json"),
                    "state_contract_sha256": policy.state_contract.sha256(),
                    "install_profile": policy.install_profile,
                    "controller_protocol": policy.state_contract.runtime_protocol,
                }
            )
        suffix = artifact.artifact_digest.removeprefix("sha256:")[:20]
        return ReleaseRecord(
            id=f"release_initial_{suffix}",
            candidate_id=f"bootstrap_{source_commit[:16]}",
            source_commit=source_commit,
            artifact_digest=artifact.artifact_digest,
            manifest_digest=artifact.manifest_digest,
            entrypoint=artifact.entrypoint,
            metadata={
                **artifact_metadata,
                "initial": True,
                "dependency_lock_hash": lock_hash,
                "evaluation_input_digest": f"sha256:{evaluation_input_sha256}",
                "evaluator_version": self._evaluator_version,
                "evaluator_fingerprint": self._evaluator_fingerprint,
                "base_commit": source_commit,
                **source_metadata,
            },
        )

    def _evaluation_input_sha256(
        self,
        *,
        source_commit: str,
        dependency_lock_hash: str | None,
    ) -> str:
        payload = (
            f"{source_commit}:{dependency_lock_hash or 'none'}:"
            f"{self._evaluator_version}:{self._evaluator_fingerprint}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def _accepted_upstream(self, source_commit: str) -> str | None:
        lineage = self._lineage
        if lineage is None:
            return None
        try:
            upstream = await asyncio.to_thread(lineage.resolve_ref, lineage.upstream_ref)
            accepted = await asyncio.to_thread(lineage.merge_base, source_commit, upstream)
            in_source, in_upstream = await asyncio.gather(
                asyncio.to_thread(lineage.is_ancestor, accepted, source_commit),
                asyncio.to_thread(lineage.is_ancestor, accepted, upstream),
            )
        except GitLineageError as exc:
            raise RuntimeError("initial release source lineage is invalid") from exc
        if not in_source or not in_upstream:
            raise RuntimeError("initial release accepted upstream is invalid")
        return accepted

    def _source_commit(self) -> str:
        result = run_bounded_process(
            (
                self._git_cli,
                "-C",
                str(self._source),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ),
            cwd=self._source,
            env={"PATH": os.environ.get("PATH", os.defpath), "HOME": "/tmp"},
            timeout_seconds=30,
            max_output_bytes=1_024,
        )
        value = result.output.decode("ascii", errors="ignore").strip().lower()
        if result.returncode != 0 or result.truncated or _COMMIT_RE.fullmatch(value) is None:
            raise RuntimeError("canonical source commit is unavailable")
        return value

    def _lock_hash(self) -> str | None:
        lockfile = self._source / "uv.lock"
        if lockfile.is_symlink() or not lockfile.is_file():
            return None
        digest = hashlib.sha256()
        with lockfile.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


class BootstrapEvolutionEventSink:
    """Project evolution completion into the bootstrap's durable owner outbox."""

    def __init__(self, supervisor: BootstrapSupervisor) -> None:
        self._supervisor = supervisor

    async def deliver(self, event: EvolutionEvent) -> None:
        self._supervisor.store.append_outbox(
            OutboxEvent(
                event_key=f"evolution:{event.event_key}",
                event_type=event.event_type,
                origin=self._origin(event.origin),
                payload={"candidate_id": event.candidate_id, **event.payload},
            )
        )
        await self._supervisor.flush_outbox()

    @staticmethod
    def _origin(value: Mapping[str, object]) -> ReleaseOrigin | None:
        required = ("tenant_id", "actor_id", "thread_id", "channel", "correlation_id")
        cleaned = {key: str(value.get(key) or "").strip() for key in required}
        if any(not cleaned[key] for key in required):
            return None
        run_id = str(value.get("run_id") or "").strip() or None
        return ReleaseOrigin(**cleaned, run_id=run_id)


class ManagedEvolutionRuntime:
    """Start source evolution only after the immutable bootstrap owns a healthy release."""

    def __init__(
        self,
        *,
        bootstrap: BootstrapSupervisor,
        archive: EvolutionArchive,
        evolution: EvolutionSupervisor,
        initial_release: InitialReleaseProvider | None,
    ) -> None:
        self._bootstrap = bootstrap
        self._archive = archive
        self._evolution = evolution
        self._initial_release = initial_release
        self._started = False

    @property
    def service(self) -> EvolutionSupervisor:
        return self._evolution

    async def start(self) -> None:
        if self._started:
            return
        await self._archive.start()
        state = self._bootstrap.store.get_state()
        releases = self._bootstrap.store.list_releases(limit=2)
        if state.serving_release_id is None and not releases:
            if self._initial_release is None:
                raise RuntimeError("an initial trusted release is required for an empty bootstrap")
            release = await self._initial_release.build()
            await self._bootstrap.install_initial(release)
            state = self._bootstrap.store.get_state()
        if state.serving_release_id is None:
            raise RuntimeError("bootstrap has no healthy serving release")
        active = self._bootstrap.store.get_release(state.serving_release_id)
        if active is None:
            raise RuntimeError("bootstrap serving release metadata is unavailable")
        if await self._archive.get_current_release() is None:
            await self._seed_initial_lineage(active)
        await self._evolution.start()
        self._started = True

    async def shutdown(self) -> None:
        if not self._started:
            await self._archive.shutdown()
            return
        self._started = False
        await self._evolution.shutdown()

    async def _seed_initial_lineage(self, release: ReleaseRecord) -> None:
        candidate = await self._archive.get_candidate(release.candidate_id)
        empty_diff_sha256 = hashlib.sha256(b"").hexdigest()
        release_provenance = ReleaseArtifactProvenance.from_values(
            source_commit=release.source_commit,
            artifact_digest=release.artifact_digest,
            manifest_digest=release.manifest_digest,
            entrypoint=release.entrypoint,
            metadata=release.metadata,
        ).release_metadata()
        accepted_upstream = release.metadata.get("accepted_upstream_commit")
        raw_release_paths = release.metadata.get("changed_paths")
        source_provenance = CandidateSourceProvenance(
            changed_paths=(
                tuple(str(path) for path in raw_release_paths)
                if isinstance(raw_release_paths, list)
                and all(isinstance(path, str) for path in raw_release_paths)
                else ()
            ),
            diff_sha256=str(release.metadata.get("diff_sha256") or empty_diff_sha256),
            accepted_upstream_commit=(
                str(accepted_upstream) if accepted_upstream is not None else None
            ),
        )
        source_metadata = source_provenance.apply_to_metadata({})
        if source_provenance.accepted_upstream_commit is not None:
            release_provenance["accepted_upstream_commit"] = (
                source_provenance.accepted_upstream_commit
            )
        fingerprint = str(release.metadata.get("evaluator_fingerprint") or release.manifest_digest)
        evaluator_version = str(
            release.metadata.get("evaluator_version") or "bootstrap-trusted-install-v1"
        )
        expected_lock_hash = str(release.metadata.get("dependency_lock_hash") or "") or None
        if candidate is None:
            candidate = await self._archive.create_candidate(
                Candidate(
                    id=release.candidate_id,
                    base_commit=release.source_commit,
                    requested_improvement="Trusted initial release installed by bootstrap",
                    source_commit=release.source_commit,
                    dependency_lock_hash=expected_lock_hash,
                    artifact_digest=release.artifact_digest,
                    evaluator_fingerprint=fingerprint,
                    metadata={
                        "manifest_digest": release.manifest_digest,
                        "release_entrypoint": list(release.entrypoint),
                        **source_metadata,
                        **release_provenance,
                        "bootstrap_initial": True,
                    },
                )
            )
        if (
            candidate.base_commit != release.source_commit
            or candidate.source_commit != release.source_commit
            or candidate.dependency_lock_hash != expected_lock_hash
            or candidate.artifact_digest != release.artifact_digest
            or candidate.evaluator_fingerprint != fingerprint
        ):
            raise RuntimeError("initial evolution lineage conflicts with the serving release")
        existing_report = candidate.evaluation_report
        if existing_report is not None and (
            existing_report.candidate_id != candidate.id
            or existing_report.source_commit != release.source_commit
            or existing_report.artifact_digest != release.artifact_digest
            or existing_report.evaluator_fingerprint != fingerprint
            or existing_report.evaluator_version != evaluator_version
            or not existing_report.passed
            or not any(
                check.name == "bootstrap.initial_release"
                and check.required
                and check.passed
                for check in existing_report.checks
            )
        ):
            raise RuntimeError("initial evolution evaluation conflicts with the serving release")
        for key, expected in (
            ("manifest_digest", release.manifest_digest),
            ("release_entrypoint", list(release.entrypoint)),
            *tuple(source_metadata.items()),
            *tuple(release_provenance.items()),
        ):
            existing = candidate.metadata.get(key)
            if existing is not None and existing != expected:
                raise RuntimeError("initial evolution evidence conflicts with the serving release")
        bound_metadata = {
            **candidate.metadata,
            "manifest_digest": release.manifest_digest,
            "release_entrypoint": list(release.entrypoint),
            **source_metadata,
            **release_provenance,
            "bootstrap_initial": True,
        }
        if bound_metadata != candidate.metadata:
            candidate = await self._archive.update_candidate(
                candidate.model_copy(update={"metadata": bound_metadata}),
                expected_revision=candidate.revision,
            )
        if existing_report is None:
            report = EvaluationReport(
                candidate_id=candidate.id,
                source_commit=release.source_commit,
                artifact_digest=release.artifact_digest,
                evaluator_fingerprint=fingerprint,
                evaluator_version=evaluator_version,
                passed=True,
                checks=(
                    EvaluationCheck(
                        name="bootstrap.initial_release",
                        passed=True,
                        summary="Installed and health-checked by the immutable bootstrap.",
                    ),
                ),
                summary="Trusted initial release installed successfully.",
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
        evaluation_report = candidate.evaluation_report
        if evaluation_report is None:
            raise RuntimeError("initial evolution evaluation evidence is unavailable")
        candidate_source = CandidateSourceProvenance.from_metadata(candidate.metadata)
        initial = Release(
            id=release.id,
            candidate_id=candidate.id,
            source_commit=release.source_commit,
            artifact_digest=release.artifact_digest,
            reason="Trusted initial installation",
            metadata={
                "artifact_kind": str(release.metadata.get("artifact_kind") or "oci_image"),
                "manifest_digest": release.manifest_digest,
                "release_entrypoint": list(release.entrypoint),
                "base_commit": candidate.base_commit,
                "changed_paths": list(candidate_source.changed_paths or ()),
                "diff_sha256": candidate_source.diff_sha256 or empty_diff_sha256,
                "evaluation_report_id": evaluation_report.id,
                "evaluation_summary": evaluation_report.summary,
                "evaluator_fingerprint": evaluation_report.evaluator_fingerprint,
                "evaluator_version": evaluation_report.evaluator_version,
                **release_provenance,
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
            raise RuntimeError("initial evolution candidate is not promotable")


__all__ = [
    "BootstrapEvolutionEventSink",
    "InitialReleaseProvider",
    "ManagedEvolutionRuntime",
    "TrustedSourceReleaseProvider",
]
