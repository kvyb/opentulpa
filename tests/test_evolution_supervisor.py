from __future__ import annotations

import asyncio
import hashlib
import shutil
import sqlite3
import stat
import subprocess
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.evolution.activation import (
    ReleaseActivationResult,
    ReleaseActivationStatus,
)
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.dependency_resolver import ResolvedDependencyBase
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    EvaluationCommand,
    EvaluationCommandResult,
    LocalEvaluationRunner,
)
from opentulpa.evolution.generation import (
    UPSTREAM_LINEAGE_METADATA_KEY,
    UpstreamLineage,
)
from opentulpa.evolution.lineage import (
    ACCEPTED_UPSTREAM_REF,
    INSTANCE_REF,
    GitLineage,
    GitLineageError,
    GitLineageSnapshot,
    NativeMerge,
    UpstreamSync,
)
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
    EvaluationCheck,
    EvaluationReport,
    PromotionAttempt,
    PromotionAttemptStatus,
    Release,
    SourceReleaseOperationStatus,
)
from opentulpa.evolution.release import AtomicReleasePointer
from opentulpa.evolution.release_builder import (
    OciReleaseArtifact,
    ReleaseBuildError,
    ReleaseBuildRequest,
)
from opentulpa.evolution.supervisor import (
    EvolutionSupervisor,
    EvolutionSupervisorError,
    InMemoryEvolutionEventSink,
)
from opentulpa.evolution.workspace import (
    CandidateCommit,
    CandidateWorkspace,
    ContributionArtifact,
    GitCandidateWorkspace,
    ReviewArtifact,
)

_TEST_SUPERVISORS: list[EvolutionSupervisor] = []


@pytest_asyncio.fixture(autouse=True)
async def _shutdown_evolution_supervisors() -> AsyncIterator[None]:
    first_new_supervisor = len(_TEST_SUPERVISORS)
    try:
        yield
    finally:
        supervisors = _TEST_SUPERVISORS[first_new_supervisor:]
        for supervisor in reversed(supervisors):
            if supervisor.started:
                await supervisor.shutdown()
        del _TEST_SUPERVISORS[first_new_supervisor:]


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / "site_app.py").write_text(
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n"
        "@app.get('/health')\n"
        "def health():\n"
        "    return {'status': 'ok'}\n",
        encoding="utf-8",
    )
    (root / "capabilities").mkdir()
    (root / "capabilities" / "web.toml").write_text(
        'name = "web"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    _git(root, "add", ".")
    _git(root, "commit", "-m", "seed web capability")
    return root


class _FakeReleaseBuilder:
    def __init__(
        self,
        *,
        fail: bool = False,
        artifact_kind: str = "oci_image",
    ) -> None:
        self.fail = fail
        self.artifact_kind = artifact_kind
        self.requests: list[ReleaseBuildRequest] = []

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        self.requests.append(request)
        if self.fail:
            raise ReleaseBuildError("Candidate OCI image build failed.")
        image = hashlib.sha256(f"image:{request.source_commit}".encode()).hexdigest()
        manifest = hashlib.sha256(f"manifest:{request.source_commit}".encode()).hexdigest()
        return OciReleaseArtifact(
            artifact_kind=self.artifact_kind,  # type: ignore[arg-type]
            artifact_digest=(
                f"sha256:{manifest}"
                if self.artifact_kind == "python_generation"
                else f"sha256:{image}"
            ),
            manifest_digest=f"sha256:{manifest}",
            image_reference=(
                f"python-generation:{manifest}"
                if self.artifact_kind == "python_generation"
                else f"opentulpa-release:{manifest[:32]}"
            ),
            entrypoint=("python", "-m", "site_app"),
        )


class _FakeReleaseActivator:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failure: tuple[ReleaseActivationStatus, str, str] | None = None
        self.before_activate: Any = None

    async def activate(
        self,
        release: ReleaseRecord,
        *,
        activation_id: str,
        origin: ReleaseOrigin | None,
        reason: str,
        rollback: bool,
    ) -> ReleaseActivationResult:
        self.calls.append(
            {
                "release": release,
                "activation_id": activation_id,
                "origin": origin,
                "reason": reason,
                "rollback": rollback,
            }
        )
        if self.before_activate is not None:
            await self.before_activate(release, rollback)
        if self.failure is not None:
            status, code, message = self.failure
            self.failure = None
            return ReleaseActivationResult(
                activation_id=activation_id,
                status=status,
                failure_code=code,
                failure_message=message,
            )
        return ReleaseActivationResult(
            activation_id=activation_id,
            status=ReleaseActivationStatus.ACTIVE,
        )


class _FakeDependencyResolver:
    def __init__(self, root: Path) -> None:
        base_id = "d" * 64
        base = root / base_id
        base.mkdir(parents=True)
        lock = base / "uv.lock"
        lock.write_text("version = 1\nresolved = true\n", encoding="utf-8")
        lock.chmod(0o444)
        base.chmod(0o555)
        self.workspaces: list[Path] = []
        self.base = ResolvedDependencyBase(
            id=base_id,
            root=base,
            lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
            requirements_sha256="e" * 64,
            wheelhouse_sha256="f" * 64,
            inventory_sha256="a" * 64,
            pyproject_sha256="c" * 64,
            site_sha256="1" * 64,
            resolver_fingerprint="sha256:" + "b" * 64,
        )

    async def resolve(self, workspace: Path) -> ResolvedDependencyBase:
        self.workspaces.append(workspace)
        return self.base

    def base_for_lock(self, lock_sha256: str) -> ResolvedDependencyBase | None:
        return self.base if lock_sha256 == self.base.lock_sha256 else None


class _FailingEventSink:
    async def deliver(self, event: Any) -> None:
        del event
        raise RuntimeError("delivery unavailable")


@dataclass(frozen=True, slots=True)
class _ShellResponse:
    output: str
    exit_code: int
    truncated: bool = False


class _WritableSourceBackend:
    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> _ShellResponse:
        completed = await asyncio.to_thread(
            subprocess.run,
            ["/bin/sh", "-lc", command],
            cwd=self._workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return _ShellResponse(
            output=completed.stdout + completed.stderr,
            exit_code=completed.returncode,
        )


class _EvolutionSupervisorEvaluationRunner:
    """Deterministic evaluator for supervisor orchestration tests."""

    def __init__(self, *, outcomes: list[bool] | None = None) -> None:
        self._outcomes = list(outcomes or [])
        self.calls: list[tuple[Path, EvaluationCommand]] = []

    @property
    def fingerprint(self) -> str:
        return "evolution-supervisor-in-process-v1"

    async def run(
        self,
        *,
        workspace: Path,
        command: EvaluationCommand,
    ) -> EvaluationCommandResult:
        self.calls.append((workspace, command))
        if self._outcomes:
            passed = self._outcomes.pop(0)
        else:
            source = (workspace / "site_app.py").read_text(encoding="utf-8")
            passed = "@app.get('/status')" in source or "@app.get('/capabilities')" in source
        return EvaluationCommandResult(
            name=command.name,
            stage=command.stage,
            passed=passed,
            exit_code=0 if passed else 9,
            duration_seconds=0.0,
            output="website status check passed" if passed else "website status check failed",
        )


class _EvolutionSupervisorWorkspaceFake:
    """Small persistent workspace port for non-Git supervisor orchestration."""

    def __init__(self, *, source_repository: Path, state_root: Path) -> None:
        self._source_repository = source_repository
        self._worktrees_root = state_root / "worktrees"
        self._snapshots_root = state_root / "snapshots"
        self._artifacts_root = state_root / "artifacts"
        self._parents_root = state_root / "parents"
        self._worktrees_root.mkdir(parents=True, exist_ok=True)
        self._snapshots_root.mkdir(parents=True, exist_ok=True)
        self._artifacts_root.mkdir(parents=True, exist_ok=True)
        self._parents_root.mkdir(parents=True, exist_ok=True)
        seed = self._files(source_repository)
        self._seed_commit = _git(source_repository, "rev-parse", "HEAD")
        self._store_snapshot(self._seed_commit, seed)

    def create(self, *, candidate_id: str, base_ref: str = "HEAD") -> CandidateWorkspace:
        base_commit = self._seed_commit if base_ref == "HEAD" else base_ref
        files = self._snapshot(base_commit)
        path = self._worktrees_root / candidate_id
        path.mkdir()
        self._write_files(path, files)
        (path / ".git").write_text(base_commit, encoding="ascii")
        return CandidateWorkspace(candidate_id, path, base_commit)

    def adopt(self, workspace: CandidateWorkspace) -> CandidateWorkspace:
        expected = self._worktrees_root / workspace.candidate_id
        if workspace.path != expected or not workspace.path.is_dir():
            raise RuntimeError("candidate workspace is unavailable")
        self._snapshot(workspace.base_commit)
        head = self.head(workspace)
        current = head
        while current != workspace.base_commit:
            parent = self._parents_root / current
            if not parent.is_file():
                raise RuntimeError("candidate workspace lineage is unavailable")
            current = parent.read_text(encoding="ascii")
        return workspace

    def head(self, workspace: CandidateWorkspace) -> str:
        return (workspace.path / ".git").read_text(encoding="ascii")

    def status(self, workspace: CandidateWorkspace) -> tuple[str, ...]:
        head_files = self._snapshot(self.head(workspace))
        current_files = self._files(workspace.path)
        return tuple(
            f"{'??' if path not in head_files else ' M'} {path}"
            for path in sorted(head_files.keys() | current_files.keys())
            if head_files.get(path) != current_files.get(path)
        )

    def full_diff(self, workspace: CandidateWorkspace) -> str:
        return self._evidence(
            self._snapshot(workspace.base_commit),
            self._files(workspace.path),
        )

    def commit(self, workspace: CandidateWorkspace, *, message: str) -> CandidateCommit:
        del message
        current_files = self._files(workspace.path)
        base_files = self._snapshot(workspace.base_commit)
        changed_paths = self._changed_paths(base_files, current_files)
        if not changed_paths:
            raise RuntimeError("candidate did not change source files")
        parent = self.head(workspace)
        source_commit = self._commit_id(parent, current_files)
        self._store_snapshot(source_commit, current_files)
        (self._parents_root / source_commit).write_text(parent, encoding="ascii")
        (workspace.path / ".git").write_text(source_commit, encoding="ascii")
        evidence = self._evidence(base_files, current_files)
        return CandidateCommit(
            candidate_id=workspace.candidate_id,
            base_commit=workspace.base_commit,
            source_commit=source_commit,
            diff_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
            changed_paths=changed_paths,
            promotion_eligible=True,
        )

    def recover_commit(self, workspace: CandidateWorkspace) -> CandidateCommit:
        head = self.head(workspace)
        current_files = self._files(workspace.path)
        if current_files != self._snapshot(head):
            raise RuntimeError("candidate worktree is not clean")
        base_files = self._snapshot(workspace.base_commit)
        evidence = self._evidence(base_files, current_files)
        return CandidateCommit(
            candidate_id=workspace.candidate_id,
            base_commit=workspace.base_commit,
            source_commit=head,
            diff_sha256=hashlib.sha256(evidence.encode()).hexdigest(),
            changed_paths=self._changed_paths(base_files, current_files),
            promotion_eligible=True,
        )

    def remove(self, workspace: CandidateWorkspace) -> None:
        shutil.rmtree(workspace.path)

    def review_artifact(
        self,
        *,
        candidate_id: str,
        base_commit: str,
        head_commit: str,
    ) -> ReviewArtifact:
        payload = self._evidence(self._snapshot(base_commit), self._snapshot(head_commit))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        path = self._artifacts_root / f"review-{candidate_id}-{digest[:16]}.patch"
        path.write_text(payload, encoding="utf-8")
        return ReviewArtifact(candidate_id, base_commit, head_commit, path, digest)

    def contribution_metadata(
        self,
        *,
        candidate_id: str,
        base_commit: str,
        head_commit: str,
    ) -> ContributionArtifact:
        payload = self._evidence(self._snapshot(base_commit), self._snapshot(head_commit))
        digest = hashlib.sha256(payload.encode()).hexdigest()
        path = self._artifacts_root / f"{candidate_id}-{digest[:16]}.patch"
        path.write_text(payload, encoding="utf-8")
        return ContributionArtifact(
            candidate_id,
            base_commit,
            head_commit,
            f"opentulpa/candidate-{candidate_id}",
            path,
            digest,
        )

    def parent(self, commit: str) -> str:
        return (self._parents_root / commit).read_text(encoding="ascii")

    def read_text(self, commit: str, path: str) -> str:
        return self._snapshot(commit)[path].decode("utf-8")

    @staticmethod
    def _changed_paths(
        before: dict[str, bytes],
        after: dict[str, bytes],
    ) -> tuple[str, ...]:
        return tuple(
            path
            for path in sorted(before.keys() | after.keys())
            if before.get(path) != after.get(path)
        )

    @classmethod
    def _evidence(cls, before: dict[str, bytes], after: dict[str, bytes]) -> str:
        records = []
        for path in cls._changed_paths(before, after):
            old = before.get(path, b"").decode("utf-8", errors="replace")
            new = after.get(path, b"").decode("utf-8", errors="replace")
            records.append(f"--- a/{path}\n+++ b/{path}\n-{old}\n+{new}\n")
        return "".join(records)

    @staticmethod
    def _commit_id(parent: str, files: dict[str, bytes]) -> str:
        digest = hashlib.sha256(parent.encode())
        for path, content in sorted(files.items()):
            digest.update(path.encode())
            digest.update(b"\0")
            digest.update(content)
        return digest.hexdigest()[:40]

    def _snapshot(self, commit: str) -> dict[str, bytes]:
        root = self._snapshots_root / commit
        if not root.is_dir():
            raise RuntimeError("candidate base snapshot is unavailable")
        return self._files(root)

    def _store_snapshot(self, commit: str, files: dict[str, bytes]) -> None:
        root = self._snapshots_root / commit
        if root.exists():
            return
        root.mkdir()
        self._write_files(root, files)

    @staticmethod
    def _files(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and path.name != ".git" and ".git" not in path.parts
        }

    @staticmethod
    def _write_files(root: Path, files: dict[str, bytes]) -> None:
        for relative, content in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


class _ManualPromotionEvolutionSupervisor(EvolutionSupervisor):
    """Disable the dispatcher so tests drive each durable transition exactly once."""

    async def start(self) -> None:
        activator = self._release_activator
        self._release_activator = None
        try:
            await super().start()
        except Exception:
            await self._archive.shutdown()
            raise
        finally:
            self._release_activator = activator


class _EvolutionSupervisorLineageFake:
    upstream_ref = "refs/heads/upstream"
    instance_ref = INSTANCE_REF
    accepted_upstream_ref = ACCEPTED_UPSTREAM_REF

    def __init__(
        self,
        *,
        instance_commit: str,
        upstream_commit: str | None = None,
        accepted_upstream_commit: str | None = None,
        initialized: bool = False,
    ) -> None:
        self._instance = instance_commit if initialized else None
        self._upstream = upstream_commit or instance_commit
        self._accepted = accepted_upstream_commit if initialized else None
        self._merge_base = accepted_upstream_commit or instance_commit
        self._native_merges: dict[str, NativeMerge] = {}
        self._verified_merges: dict[str, tuple[str, str]] = {}

    def resolve_ref(self, ref: str) -> str:
        values = {
            self.instance_ref: self._instance,
            self.upstream_ref: self._upstream,
            self.accepted_upstream_ref: self._accepted,
        }
        value = values.get(ref)
        if value is None:
            raise GitLineageError("lineage ref is unavailable")
        return value

    def initialize(
        self,
        instance_commit: str,
        accepted_upstream_commit: str | None = None,
    ) -> GitLineageSnapshot:
        self._instance = instance_commit
        self._accepted = accepted_upstream_commit or self._upstream
        self._merge_base = self._accepted
        return self.snapshot()

    def project(
        self,
        instance_commit: str,
        accepted_upstream_commit: str,
        *,
        expected_instance_commit: str,
        expected_accepted_upstream_commit: str,
    ) -> GitLineageSnapshot:
        if (
            self._instance != expected_instance_commit
            or self._accepted != expected_accepted_upstream_commit
        ):
            raise GitLineageError("lineage projection changed")
        self._instance = instance_commit
        self._accepted = accepted_upstream_commit
        self._merge_base = accepted_upstream_commit
        return self.snapshot()

    def snapshot(self) -> GitLineageSnapshot:
        if self._instance is None or self._accepted is None:
            raise GitLineageError("lineage is not initialized")
        return GitLineageSnapshot(
            instance_commit=self._instance,
            upstream_commit=self._upstream,
            accepted_upstream_commit=self._accepted,
            merge_base_commit=self._merge_base,
        )

    def is_ancestor(self, ancestor_commit: str, descendant_commit: str) -> bool:
        if ancestor_commit == descendant_commit:
            return True
        if ancestor_commit in self._verified_merges.get(descendant_commit, ()):
            return True
        if ancestor_commit in {self._accepted, self._merge_base}:
            return True
        return ancestor_commit == self._upstream and self._upstream == self._accepted

    def merge_base(self, instance_commit: str, upstream_commit: str) -> str:
        del instance_commit, upstream_commit
        return self._merge_base

    def prepare_merge(
        self,
        workspace: CandidateWorkspace,
        lineage: UpstreamLineage | None = None,
    ) -> NativeMerge:
        del lineage
        merge = NativeMerge(
            instance_commit=workspace.base_commit,
            upstream_commit=self._upstream,
            merge_base_commit=self._accepted or workspace.base_commit,
            conflicted_paths=(),
        )
        self._native_merges[workspace.candidate_id] = merge
        (workspace.path / "upstream-merge.txt").write_text("upstream\n", encoding="utf-8")
        return merge

    def inspect_native_merge(self, workspace: CandidateWorkspace) -> NativeMerge | None:
        merge = self._native_merges.get(workspace.candidate_id)
        if merge is None:
            return None
        head = (workspace.path / ".git").read_text(encoding="ascii")
        return merge if head == workspace.base_commit else None

    def conflicted_paths(self, workspace: CandidateWorkspace) -> tuple[str, ...]:
        del workspace
        return ()

    def verify_merged_tip(
        self,
        tip_commit: str,
        *,
        instance_commit: str,
        upstream_commit: str,
        expected_merge_commit: str | None = None,
    ) -> str:
        if expected_merge_commit is not None:
            self._verified_merges[tip_commit] = (instance_commit, upstream_commit)
            return expected_merge_commit
        self._verified_merges[tip_commit] = (instance_commit, upstream_commit)
        return tip_commit

    def advance_upstream(self) -> str:
        self._upstream = hashlib.sha256(f"upstream:{self._upstream}".encode()).hexdigest()[:40]
        return self._upstream

    def sync_upstream(self, repository_url: str, remote_ref: str) -> UpstreamSync:
        assert repository_url == "https://github.com/kvyb/opentulpa"
        assert remote_ref == "refs/heads/main"
        previous = self._upstream
        return UpstreamSync(
            previous_commit=previous,
            upstream_commit=self.advance_upstream(),
        )

    def diverge_instance(self) -> str:
        self._instance = hashlib.sha256(f"instance:{self._instance}".encode()).hexdigest()[:40]
        return self._instance

    def merge_parents(self, commit: str) -> tuple[str, str]:
        return self._verified_merges[commit]


def _supervisor(
    tmp_path: Path,
    source: Path,
    *,
    builder: _FakeReleaseBuilder | None = None,
    activator: _FakeReleaseActivator | None = None,
    event_sink: Any = None,
    lineage: Any = None,
    evaluation_runner: _EvolutionSupervisorEvaluationRunner | None = None,
    automatic_promotions: bool = False,
    real_git: bool = False,
    source_mutation_enabled: bool = True,
    dependency_resolver: Any = None,
    dependency_evaluator_factory: Any = None,
) -> EvolutionSupervisor:
    supervisor_type = (
        EvolutionSupervisor if automatic_promotions else _ManualPromotionEvolutionSupervisor
    )
    workspaces = (
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "contributions",
        )
        if real_git or isinstance(lineage, GitLineage)
        else _EvolutionSupervisorWorkspaceFake(
            source_repository=source,
            state_root=tmp_path / "fake-workspace",
        )
    )
    supervisor = supervisor_type(
        archive=EvolutionArchive(tmp_path / "evolution.db"),
        workspaces=workspaces,  # type: ignore[arg-type]
        candidate_backend_factory=_WritableSourceBackend,
        evaluator=CandidateEvaluator(
            runner=evaluation_runner or _EvolutionSupervisorEvaluationRunner(),
            commands=(
                EvaluationCommand(
                    name="website.status",
                    stage="public",
                    argv=(
                        sys.executable,
                        "-c",
                        (
                            "from fastapi.testclient import TestClient; "
                            "from site_app import app; c=TestClient(app); "
                            "assert c.get('/health').status_code == 200; "
                            "assert c.get('/status').json()['status'] == 'ready' "
                            "or c.get('/capabilities').status_code == 200"
                        ),
                    ),
                ),
            ),
        ),
        release_pointer=AtomicReleasePointer(tmp_path / "release" / "current.json"),
        release_builder=builder or _FakeReleaseBuilder(),
        release_activator=activator or _FakeReleaseActivator(),
        event_sink=event_sink,
        lineage=lineage,
        source_mutation_enabled=source_mutation_enabled,
        source_mutation_unavailable_reason=(
            None if source_mutation_enabled else "isolated root Linux requirements are unavailable"
        ),
        dependency_resolver=dependency_resolver,
        dependency_evaluator_factory=dependency_evaluator_factory,
    )
    _TEST_SUPERVISORS.append(supervisor)
    return supervisor


async def _terminal_attempt(
    supervisor: EvolutionSupervisor,
    attempt: PromotionAttempt,
) -> PromotionAttempt:
    assert await supervisor.process_queued_promotions() == 1
    current = await supervisor.get_promotion_attempt(attempt.id)
    assert current is not None
    assert current.status in {PromotionAttemptStatus.ACTIVE, PromotionAttemptStatus.FAILED}
    return current


async def _rollback(supervisor: EvolutionSupervisor) -> Release:
    attempt = await supervisor.queue_rollback()
    assert attempt.status is PromotionAttemptStatus.QUEUED
    completed = await _terminal_attempt(supervisor, attempt)
    assert completed.status is PromotionAttemptStatus.ACTIVE
    current = await supervisor._archive.get_current_release()
    assert current is not None and current.id == completed.release.id
    return current


def _source_audit() -> dict[str, str]:
    return {
        "tenant_id": "owner",
        "actor_id": "owner-1",
        "thread_id": "thread-source",
        "channel": "web",
        "run_kind": "owner",
        "correlation_id": "run-source",
    }


@pytest.mark.asyncio
async def test_integrity_only_supervisor_reports_source_mutation_unavailable(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(
        tmp_path,
        source,
        source_mutation_enabled=False,
    )
    await supervisor.start()

    status = await supervisor.source_status(audit_context=_source_audit())

    assert supervisor.source_mutation_enabled is False
    assert status["available"] is False
    assert status["source_mutation_enabled"] is False
    assert "isolated root Linux" in status["reason"]
    with pytest.raises(EvolutionSupervisorError, match="isolated root Linux"):
        await supervisor.source_shell(command="true", audit_context=_source_audit())
    with pytest.raises(EvolutionSupervisorError, match="isolated root Linux"):
        await supervisor.queue_rollback()


def _release_binding(status: dict[str, Any]) -> dict[str, str]:
    return {
        "expected_candidate_id": str(status["candidate_id"]),
        "expected_diff_sha256": str(status["diff_sha256"]),
    }


def _rollback_binding(status: dict[str, Any]) -> dict[str, str]:
    assert status["current_release_id"]
    assert status["rollback_target_release_id"]
    return {
        "expected_current_release_id": str(status["current_release_id"]),
        "expected_target_release_id": str(status["rollback_target_release_id"]),
    }


def _promotion_attempt_count(db_path: Path, attempt_id: str) -> int:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM evolution_promotion_attempts WHERE id = ?",
            (attempt_id,),
        ).fetchone()
    assert row is not None
    return int(row[0])


def _route_command(route: str) -> str:
    if route == "status":
        body = (
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
        )
    elif route == "capabilities":
        body = (
            "@app.get('/capabilities')\ndef capabilities():\n    return {'capabilities': ['web']}\n"
        )
    else:
        raise ValueError("unsupported test route")
    return f"cat >> site_app.py <<'PY'\n\n{body}PY"


def _lineage(source: Path, tmp_path: Path) -> GitLineage:
    (tmp_path / "worktrees").mkdir(exist_ok=True)
    return GitLineage(source, worktrees_root=tmp_path / "worktrees")


async def _seed_active_release(
    supervisor: EvolutionSupervisor,
    *,
    source_commit: str,
    accepted_upstream_commit: str | None,
    release_id: str = "release_seed",
    candidate_id: str = "candidate_seed",
    artifact_kind: str = "oci_image",
) -> Release:
    await supervisor._archive.start()
    artifact_digest = f"sha256:{hashlib.sha256(candidate_id.encode()).hexdigest()}"
    manifest_digest = f"sha256:{hashlib.sha256(release_id.encode()).hexdigest()}"
    fingerprint = f"sha256:{'e' * 64}"
    metadata: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "manifest_digest": manifest_digest,
        "release_entrypoint": ["python", "-m", "site_app"],
        "changed_paths": [],
        "diff_sha256": hashlib.sha256(b"").hexdigest(),
    }
    release_metadata: dict[str, Any] = {
        **metadata,
        "base_commit": source_commit,
        "evaluation_report_id": f"evaluation_{candidate_id}",
        "evaluation_summary": "seeded",
        "evaluator_fingerprint": fingerprint,
        "evaluator_version": "seed-v1",
        "activation_state": "active",
    }
    if accepted_upstream_commit is not None:
        metadata["accepted_upstream_commit"] = accepted_upstream_commit
        release_metadata["accepted_upstream_commit"] = accepted_upstream_commit
    candidate = await supervisor._archive.create_candidate(
        Candidate(
            id=candidate_id,
            base_commit=source_commit,
            requested_improvement="Seed active release",
            source_commit=source_commit,
            artifact_digest=artifact_digest,
            evaluator_fingerprint=fingerprint,
            metadata=metadata,
        )
    )
    report = EvaluationReport(
        id=f"evaluation_{candidate_id}",
        candidate_id=candidate.id,
        source_commit=source_commit,
        artifact_digest=artifact_digest,
        evaluator_fingerprint=fingerprint,
        evaluator_version="seed-v1",
        passed=True,
        checks=(EvaluationCheck(name="seed", passed=True),),
        summary="seeded",
    )
    candidate = await supervisor._archive.append_evaluation(
        report,
        expected_revision=candidate.revision,
    )
    candidate = await supervisor._archive.transition_status(
        candidate.id,
        expected_status=CandidateStatus.BUILDING,
        new_status=CandidateStatus.READY,
        expected_revision=candidate.revision,
    )
    _, release = await supervisor._archive.promote_candidate(
        Release(
            id=release_id,
            candidate_id=candidate.id,
            source_commit=source_commit,
            artifact_digest=artifact_digest,
            reason="Seed active release",
            metadata=release_metadata,
        ),
        expected_revision=candidate.revision,
    )
    return release


async def _release_route(
    supervisor: EvolutionSupervisor,
    *,
    route: str,
    idempotency_key: str,
    audit: dict[str, str] | None = None,
) -> tuple[Candidate, PromotionAttempt]:
    context = audit or _source_audit()
    edited = await supervisor.source_shell(
        command=_route_command(route),
        audit_context=context,
    )
    released = await supervisor.source_release(
        idempotency_key=idempotency_key,
        **_release_binding(edited),
        message=f"Add {route} route",
        audit_context=context,
    )
    candidate = await supervisor.get_candidate(str(edited["candidate"]["id"]))
    assert candidate is not None
    return candidate, PromotionAttempt.model_validate(released["promotion"])


@pytest.mark.asyncio
async def test_interactive_source_session_survives_restart_and_releases(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    first = _supervisor(tmp_path, source, activator=activator)
    audit = _source_audit()
    await first.start()
    empty = await first.source_status(audit_context=audit)
    assert empty["available"] is True
    assert empty["active"] is False
    assert empty["session_active"] is False
    assert empty["candidate_id"] is None
    shell = await first.source_shell(
        command=(
            "set -e\n"
            "test ! -e .git\n"
            "cat >> site_app.py <<'PY'\n\n"
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
            "PY\n"
            f"{sys.executable} -c \"compile(open('site_app.py').read(), 'site_app.py', 'exec')\"\n"
            "echo source-edited"
        ),
        audit_context=audit,
    )
    candidate_id = str(shell["candidate"]["id"])
    assert shell["exit_code"] == 0
    assert shell["output"] == "source-edited\n"
    assert shell["dirty"] is True
    assert "diff" not in shell
    assert shell["diff_sha256"]
    await first.shutdown()

    resumed = _supervisor(tmp_path, source, activator=activator)
    await resumed.start()
    try:
        status = await resumed.source_status(audit_context=audit)
        assert status["available"] is True
        assert status["active"] is True
        assert status["session_active"] is True
        assert status["candidate"]["id"] == candidate_id
        assert "@app.get('/status')" in status["diff"]

        released = await resumed.source_release(
            idempotency_key="release-restart",
            **_release_binding(status),
            message="Add interactive status route",
            audit_context=audit,
        )

        assert released["active"] is False
        assert released["candidate"]["id"] == candidate_id
        assert released["candidate"]["status"] == CandidateStatus.READY.value
        assert released["candidate"]["evaluation"]["passed"] is True
        final_status = await resumed.source_status(audit_context=audit)
        assert final_status["available"] is True
        assert final_status["active"] is False
        assert final_status["session_active"] is False
        attempt = PromotionAttempt.model_validate(released["promotion"])
        completed = await _terminal_attempt(resumed, attempt)
        assert completed.status is PromotionAttemptStatus.ACTIVE
        assert activator.calls[-1]["release"].candidate_id == candidate_id
    finally:
        await resumed.shutdown()


@pytest.mark.asyncio
async def test_source_dependency_resolution_binds_proposal_and_installs_trusted_lock(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    resolver = _FakeDependencyResolver(tmp_path / "dependency-bases")
    dynamic_runners: list[_EvolutionSupervisorEvaluationRunner] = []

    def dependency_evaluator(_: ResolvedDependencyBase) -> CandidateEvaluator:
        runner = _EvolutionSupervisorEvaluationRunner()
        dynamic_runners.append(runner)
        return CandidateEvaluator(
            runner=runner,
            commands=(
                EvaluationCommand(
                    name="resolved.status",
                    argv=(sys.executable, "-c", "pass"),
                ),
            ),
        )

    supervisor = _supervisor(
        tmp_path,
        source,
        dependency_resolver=resolver,
        dependency_evaluator_factory=dependency_evaluator,
    )
    await supervisor.start()
    audit = _source_audit()
    edited = await supervisor.source_shell(
        command=(
            "printf '%s\\n' "
            "\"[project]\" \"name = 'opentulpa'\" \"version = '0.1.0'\" "
            "\"dependencies = ['demo>=1']\" > pyproject.toml\n"
            "cat >> site_app.py <<'PY'\n\n"
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'status': 'ready'}\n"
            "PY"
        ),
        audit_context=audit,
    )

    resolved = await supervisor.source_resolve_dependencies(
        expected_candidate_id=str(edited["candidate_id"]),
        expected_diff_sha256=str(edited["diff_sha256"]),
        audit_context=audit,
    )

    workspace = resolver.workspaces[0]
    assert (workspace / "uv.lock").read_bytes() == resolver.base.lock_path.read_bytes()
    assert stat.S_IMODE((workspace / "uv.lock").stat().st_mode) == 0o600
    assert resolved["dependency_base_id"] == resolver.base.id
    assert resolved["dependency_lock_hash"] == resolver.base.lock_sha256
    assert resolved["dependency_wheelhouse_sha256"] == resolver.base.wheelhouse_sha256
    assert resolved["dirty"] is True
    assert set(resolved["changed_files"]) == {"pyproject.toml", "site_app.py", "uv.lock"}
    with pytest.raises(EvolutionSupervisorError, match="proposal changed"):
        await supervisor.source_resolve_dependencies(
            expected_candidate_id=str(edited["candidate_id"]),
            expected_diff_sha256=str(edited["diff_sha256"]),
            audit_context=audit,
        )
    await supervisor.source_release(
        idempotency_key="resolved-dependency-release",
        expected_candidate_id=str(resolved["candidate_id"]),
        expected_diff_sha256=str(resolved["diff_sha256"]),
        message="Use resolved dependency",
        audit_context=audit,
    )
    candidate = await supervisor.get_candidate(str(resolved["candidate_id"]))
    assert candidate is not None
    assert candidate.status is CandidateStatus.READY
    assert candidate.dependency_lock_hash == resolver.base.lock_sha256
    assert candidate.metadata["dependency_base_id"] == resolver.base.id
    assert candidate.evaluator_fingerprint == CandidateEvaluator(
        runner=dynamic_runners[0],
        commands=(
            EvaluationCommand(
                name="resolved.status",
                argv=(sys.executable, "-c", "pass"),
            ),
        ),
    ).fingerprint
    assert dynamic_runners[0].calls


@pytest.mark.asyncio
async def test_source_release_announces_cutover_to_original_thread_before_activation(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    sink = InMemoryEvolutionEventSink()
    activator = _FakeReleaseActivator()
    audit = _source_audit()
    supervisor = _supervisor(
        tmp_path,
        source,
        activator=activator,
        event_sink=sink,
    )

    async def require_switch_announcement(_release: ReleaseRecord, _rollback: bool) -> None:
        assert sink.events[-1].event_type == "build.switching"
        assert sink.events[-1].origin["thread_id"] == audit["thread_id"]
        assert sink.events[-1].origin["correlation_id"] == audit["correlation_id"]

    activator.before_activate = require_switch_announcement
    await supervisor.start()

    candidate, attempt = await _release_route(
        supervisor,
        route="status",
        idempotency_key="visible-cutover",
        audit=audit,
    )

    assert [event.event_type for event in sink.events] == [
        "build.preparing",
        "candidate.ready",
    ]
    completed = await _terminal_attempt(supervisor, attempt)
    assert completed.status is PromotionAttemptStatus.ACTIVE
    assert [event.event_type for event in sink.events] == [
        "build.preparing",
        "candidate.ready",
        "build.switching",
        "promotion.active",
    ]
    assert all(event.candidate_id == candidate.id for event in sink.events)
    assert all(event.origin["thread_id"] == audit["thread_id"] for event in sink.events)


@pytest.mark.asyncio
async def test_source_release_keeps_failed_session_editable_and_retries_same_commit(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder(fail=True)
    supervisor = _supervisor(
        tmp_path,
        source,
        builder=builder,
        real_git=True,
        evaluation_runner=LocalEvaluationRunner(),  # type: ignore[arg-type]
    )
    audit = _source_audit()
    await supervisor.start()
    try:
        first_shell = await supervisor.source_shell(
            command="printf 'experiment notes\\n' > experiment.txt",
            audit_context=audit,
        )
        candidate_id = str(first_shell["candidate"]["id"])
        evaluation_failure = await supervisor.source_release(
            idempotency_key="release-evaluation-failure",
            **_release_binding(first_shell),
            message="First experiment",
            audit_context=audit,
        )

        assert evaluation_failure["active"] is True
        assert evaluation_failure["promotion"] is None
        assert evaluation_failure["candidate"]["id"] == candidate_id
        assert evaluation_failure["candidate"]["evaluation"]["passed"] is False
        assert "experiment notes" in evaluation_failure["diff"]
        assert builder.requests == []

        fixed = await supervisor.source_shell(
            command=(
                "cat >> site_app.py <<'PY'\n\n"
                "@app.get('/status')\n"
                "def status():\n"
                "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
                "PY"
            ),
            audit_context=audit,
        )
        assert fixed["candidate"]["id"] == candidate_id
        build_failure = await supervisor.source_release(
            idempotency_key="release-build-failure",
            **_release_binding(fixed),
            message="Fix public check",
            audit_context=audit,
        )

        assert build_failure["active"] is True
        assert build_failure["promotion"] is None
        assert build_failure["candidate"]["id"] == candidate_id
        assert build_failure["candidate"]["evaluation"]["checks"][-1]["name"] == (
            "build:release.artifact"
        )
        assert build_failure["candidate"]["evaluation"]["passed"] is False
        assert len(builder.requests) == 1

        builder.fail = False
        released = await supervisor.source_release(
            idempotency_key="release-success",
            **_release_binding(fixed),
            message="Retry exact candidate",
            audit_context=audit,
        )

        assert released["candidate"]["id"] == candidate_id
        assert released["candidate"]["status"] == CandidateStatus.READY.value
        assert released["promotion"] is not None
        assert len(builder.requests) == 2
        assert builder.requests[0].source_commit == builder.requests[1].source_commit
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_source_session_is_shared_across_owner_threads(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
    first_audit = _source_audit()
    second_audit = {**first_audit, "thread_id": "thread-telegram", "channel": "telegram"}
    await supervisor.start()
    try:
        first, second = await asyncio.gather(
            supervisor.source_shell(
                command="printf 'web\n' > web-note.txt",
                audit_context=first_audit,
            ),
            supervisor.source_shell(
                command="printf 'telegram\n' > telegram-note.txt",
                audit_context=second_audit,
            ),
        )

        assert first["candidate"]["id"] == second["candidate"]["id"]
        status = await supervisor.source_status(audit_context=second_audit)
        assert set(status["changed_files"]) == {"telegram-note.txt", "web-note.txt"}
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_source_release_rejects_edits_after_owner_approval_snapshot(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
    web = _source_audit()
    telegram = {**web, "thread_id": "thread-telegram", "channel": "telegram"}
    await supervisor.start()
    try:
        await supervisor.source_shell(
            command=_route_command("status"),
            audit_context=web,
        )
        approved = await supervisor.source_status(audit_context=web)
        changed = await supervisor.source_shell(
            command="printf 'changed after approval\n' > late-change.txt",
            audit_context=telegram,
        )

        with pytest.raises(EvolutionSupervisorError, match="source changed"):
            await supervisor.source_release(
                idempotency_key="stale-owner-approval",
                **_release_binding(approved),
                message="Do not release later edits",
                audit_context=web,
            )

        assert changed["candidate_id"] == approved["candidate_id"]
        assert changed["diff_sha256"] != approved["diff_sha256"]
        assert (
            await supervisor._archive.get_source_release_operation(
                tenant_id="owner",
                idempotency_key="stale-owner-approval",
            )
            is None
        )
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_source_release_replays_exact_result_across_restart(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder()
    first = _supervisor(tmp_path, source, builder=builder)
    audit = _source_audit()
    await first.start()
    edited = await first.source_shell(
        command=(
            "cat >> site_app.py <<'PY'\n\n"
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
            "PY"
        ),
        audit_context=audit,
    )
    released = await first.source_release(
        idempotency_key="durable-release",
        **_release_binding(edited),
        message="Add status route",
        audit_context=audit,
    )
    replayed = await first.source_release(
        idempotency_key="durable-release",
        **_release_binding(edited),
        message="Add status route",
        audit_context={**audit, "thread_id": "another-interface"},
    )
    assert replayed == released
    assert len(builder.requests) == 1
    await first.shutdown()

    restarted = _supervisor(tmp_path, source, builder=builder)
    await restarted.start()
    try:
        replayed_after_restart = await restarted.source_release(
            idempotency_key="durable-release",
            **_release_binding(edited),
            message="Add status route",
            audit_context=audit,
        )
        assert replayed_after_restart == released
        assert len(builder.requests) == 1
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_source_release_recovers_commit_before_archive_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder()
    first = _supervisor(tmp_path, source, builder=builder)
    audit = _source_audit()
    await first.start()
    edited = await first.source_shell(
        command=(
            "cat >> site_app.py <<'PY'\n\n"
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
            "PY"
        ),
        audit_context=audit,
    )
    candidate_id = str(edited["candidate"]["id"])
    original_update = first._archive.update_candidate
    failed = False

    async def fail_after_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated crash after commit")
        return await original_update(*args, **kwargs)

    monkeypatch.setattr(first._archive, "update_candidate", fail_after_commit)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await first.source_release(
            idempotency_key="commit-crash",
            **_release_binding(edited),
            message="Recover committed source",
            audit_context=audit,
        )
    await first.shutdown()

    restarted = _supervisor(tmp_path, source, builder=builder)
    await restarted.start()
    try:
        released = await restarted.source_release(
            idempotency_key="commit-crash",
            **_release_binding(edited),
            message="Recover committed source",
            audit_context=audit,
        )
        assert released["candidate"]["id"] == candidate_id
        assert released["candidate"]["status"] == CandidateStatus.READY.value
        assert released["promotion"] is not None
        assert len(builder.requests) == 1
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_source_release_recovers_second_commit_after_prior_failed_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder()
    first = _supervisor(tmp_path, source, builder=builder)
    audit = _source_audit()
    await first.start()
    initial = await first.source_shell(
        command="printf 'first experiment\n' > experiment.txt",
        audit_context=audit,
    )
    failed = await first.source_release(
        idempotency_key="first-evaluation-failure",
        **_release_binding(initial),
        message="Record failed experiment",
        audit_context=audit,
    )
    assert failed["promotion"] is None
    prior = await first.get_candidate(str(initial["candidate_id"]))
    assert prior is not None and prior.source_commit is not None
    prior_commit = prior.source_commit

    fixed = await first.source_shell(
        command=_route_command("status"),
        audit_context=audit,
    )
    original_update = first._archive.update_candidate
    crashed = False

    async def crash_after_second_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal crashed
        candidate = args[0]
        if not crashed and candidate.source_commit != prior_commit:
            crashed = True
            raise RuntimeError("simulated crash after second commit")
        return await original_update(*args, **kwargs)

    monkeypatch.setattr(first._archive, "update_candidate", crash_after_second_commit)
    with pytest.raises(RuntimeError, match="second commit"):
        await first.source_release(
            idempotency_key="second-commit-crash",
            **_release_binding(fixed),
            message="Recover second committed source",
            audit_context=audit,
        )
    await first.shutdown()

    restarted = _supervisor(tmp_path, source, builder=builder)
    await restarted.start()
    try:
        released = await restarted.source_release(
            idempotency_key="second-commit-crash",
            **_release_binding(fixed),
            message="Recover second committed source",
            audit_context=audit,
        )
        recovered = await restarted.get_candidate(str(initial["candidate_id"]))
        assert recovered is not None
        assert recovered.status in {CandidateStatus.READY, CandidateStatus.PROMOTED}
        assert recovered.source_commit != prior_commit
        assert released["promotion"] is not None
        assert len(builder.requests) == 1
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_source_release_closes_unrecoverable_operation_after_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    first = _supervisor(tmp_path, source)
    audit = _source_audit()
    await first.start()
    edited = await first.source_shell(
        command="printf 'unfinished\n' > unfinished.txt",
        audit_context=audit,
    )

    async def crash_before_commit(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated source process loss")

    monkeypatch.setattr(first, "_source_commit", crash_before_commit)
    with pytest.raises(RuntimeError, match="source process loss"):
        await first.source_release(
            idempotency_key="unrecoverable-release",
            **_release_binding(edited),
            message="Interrupted source release",
            audit_context=audit,
        )
    candidate = await first.get_candidate(str(edited["candidate_id"]))
    assert candidate is not None and candidate.worktree_path is not None
    worktree = Path(candidate.worktree_path)
    await first.shutdown()
    shutil.rmtree(worktree)

    restarted = _supervisor(tmp_path, source)
    await restarted.start()
    try:
        operation = await restarted._archive.get_source_release_operation(
            tenant_id="owner",
            idempotency_key="unrecoverable-release",
        )
        assert operation is not None
        assert operation.status is SourceReleaseOperationStatus.COMPLETED
        assert operation.result is not None
        assert operation.result["error"] == {
            "code": "source_release_unrecoverable",
            "message": "Source release could not be recovered; start a new source session.",
            "retryable": False,
        }
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_source_release_recovers_ready_candidate_before_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder()
    first = _supervisor(tmp_path, source, builder=builder)
    audit = _source_audit()
    await first.start()
    edited = await first.source_shell(
        command=(
            "cat >> site_app.py <<'PY'\n\n"
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
            "PY"
        ),
        audit_context=audit,
    )

    async def fail_before_promotion(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated crash before promotion")

    monkeypatch.setattr(first, "queue_promotion", fail_before_promotion)
    with pytest.raises(RuntimeError, match="simulated crash"):
        await first.source_release(
            idempotency_key="ready-crash",
            **_release_binding(edited),
            message="Recover ready source",
            audit_context=audit,
        )
    await first.shutdown()

    restarted = _supervisor(tmp_path, source, builder=builder)
    await restarted.start()
    try:
        released = await restarted.source_release(
            idempotency_key="ready-crash",
            **_release_binding(edited),
            message="Recover ready source",
            audit_context=audit,
        )
        assert released["candidate"]["status"] == CandidateStatus.READY.value
        assert released["promotion"] is not None
        assert len(builder.requests) == 1
    finally:
        await restarted.shutdown()

    attempt_id = str(released["promotion"]["id"])
    replay = _supervisor(tmp_path, source, builder=builder)
    await replay.start()
    try:
        replayed = await replay.source_release(
            idempotency_key="ready-crash",
            **_release_binding(edited),
            message="Recover ready source",
            audit_context=audit,
        )
        assert replayed == released
    finally:
        await replay.shutdown()
    assert _promotion_attempt_count(tmp_path / "evolution.db", attempt_id) == 1


@pytest.mark.asyncio
async def test_source_release_reuses_promotion_after_response_loss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder()
    first = _supervisor(tmp_path, source, builder=builder)
    audit = _source_audit()
    await first.start()
    edited = await first.source_shell(
        command=(
            "cat >> site_app.py <<'PY'\n\n"
            "@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
            "PY"
        ),
        audit_context=audit,
    )
    original_complete = first._archive.complete_source_release_operation
    failed = False

    async def lose_response(*args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if not failed:
            failed = True
            raise RuntimeError("simulated response loss")
        return await original_complete(*args, **kwargs)

    monkeypatch.setattr(
        first._archive,
        "complete_source_release_operation",
        lose_response,
    )
    with pytest.raises(RuntimeError, match="response loss"):
        await first.source_release(
            idempotency_key="response-loss",
            **_release_binding(edited),
            message="Recover promotion response",
            audit_context=audit,
        )
    operation = await first._archive.get_source_release_operation(
        tenant_id="owner",
        idempotency_key="response-loss",
    )
    assert operation is not None
    expected_attempt_id = (
        "promotion_" + hashlib.sha256(f"{operation.id}:promotion".encode()).hexdigest()[:48]
    )
    original_attempt = await first.get_promotion_attempt(expected_attempt_id)
    assert original_attempt is not None
    await first.shutdown()

    restarted = _supervisor(tmp_path, source, builder=builder)
    await restarted.start()
    try:
        released = await restarted.source_release(
            idempotency_key="response-loss",
            **_release_binding(edited),
            message="Recover promotion response",
            audit_context=audit,
        )
        assert released["promotion"]["id"] == expected_attempt_id
        assert len(builder.requests) == 1
        attempts = [
            attempt
            for attempt in await restarted._archive.list_incomplete_promotion_attempts()
            if attempt.candidate_id == original_attempt.candidate_id
        ]
        assert len(attempts) <= 1
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_source_release_rejects_a_session_based_on_an_inactive_release(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
    audit = _source_audit()
    await supervisor.start()
    try:
        stale = await supervisor.source_shell(
            command="printf 'pending\n' > pending.txt",
            audit_context=audit,
        )
        other_audit = {**audit, "tenant_id": "other-owner", "thread_id": "other-thread"}
        _, other_attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="other-release",
            audit=other_audit,
        )
        assert (await _terminal_attempt(supervisor, other_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )

        with pytest.raises(EvolutionSupervisorError, match="inactive release"):
            await supervisor.source_release(
                idempotency_key="stale-release",
                **_release_binding(stale),
                message="Do not overwrite a newer release",
                audit_context=audit,
            )
        current = await supervisor.get_candidate(str(stale["candidate"]["id"]))
        assert current is not None
        assert current.status is CandidateStatus.BUILDING
        assert (
            await supervisor._archive.get_source_release_operation(
                tenant_id="owner",
                idempotency_key="stale-release",
            )
            is None
        )
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_self_improvement_builds_archives_promotes_and_contributes_website(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
    await supervisor.start()
    try:
        candidate, attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="website-contribution",
        )

        assert candidate.source_commit
        assert candidate.artifact_digest
        assert candidate.metadata["artifact_kind"] == "oci_image"
        assert candidate.metadata["manifest_digest"]
        assert candidate.evaluation_report is not None
        assert candidate.evaluation_report.passed is True
        assert candidate.worktree_path is None
        assert "/status" not in (source / "site_app.py").read_text(encoding="utf-8")
        workspaces = supervisor._workspaces
        assert isinstance(workspaces, _EvolutionSupervisorWorkspaceFake)
        evolved_source = workspaces.read_text(candidate.source_commit, "site_app.py")
        assert "@app.get('/status')" in evolved_source
        review_patch = await supervisor.review_patch(candidate.id)
        assert "@app.get('/status')" in review_patch.read_text(encoding="utf-8")

        assert (await _terminal_attempt(supervisor, attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        release = await supervisor._archive.get_current_release()
        promoted = await supervisor.get_candidate(candidate.id)
        assert promoted is not None
        assert promoted.status is CandidateStatus.PROMOTED
        assert release is not None
        assert release.candidate_id == candidate.id
        assert release.metadata["activation_state"] == "active"

        contributed = await supervisor.prepare_contribution(candidate.id)
        assert contributed.contribution is not None
        assert contributed.contribution.sanitized is True
        assert contributed.contribution.metadata["sanitation_scanner"]
        assert contributed.contribution.metadata["requires_owner_review"] is True
        patch_name = str(contributed.contribution.metadata["patch_filename"])
        assert (tmp_path / "fake-workspace" / "artifacts" / patch_name).is_file()
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_second_generation_can_roll_back_to_prior_release(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    supervisor = _supervisor(tmp_path, source, activator=activator)
    await supervisor.start()
    try:
        first, first_attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="first-generation",
        )
        assert (await _terminal_attempt(supervisor, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        first_release = await supervisor._archive.get_current_release()
        assert first_release is not None
        second, second_attempt = await _release_route(
            supervisor,
            route="capabilities",
            idempotency_key="second-generation",
        )
        assert (await _terminal_attempt(supervisor, second_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        second_release = await supervisor._archive.get_current_release()
        assert second_release is not None

        rollback = await _rollback(supervisor)

        assert rollback.candidate_id == first.id
        assert rollback.metadata["rollback_of"] == second_release.id
        assert rollback.metadata["rollback_target"] == first_release.id
        assert activator.calls[-1]["rollback"] is True
        assert activator.calls[-1]["release"].id == rollback.id
        assert activator.calls[-1]["release"].artifact_digest == first_release.artifact_digest
        rolled_back = await supervisor.get_candidate(second.id)
        assert rolled_back is not None
        assert rolled_back.status is CandidateStatus.ROLLED_BACK
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_source_rollback_replays_after_response_loss_without_rolling_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    first = _supervisor(tmp_path, source, activator=activator)
    audit = _source_audit()
    await first.start()
    try:
        _, first_attempt = await _release_route(
            first,
            route="status",
            idempotency_key="rollback-replay-first",
        )
        assert (await _terminal_attempt(first, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        _, second_attempt = await _release_route(
            first,
            route="capabilities",
            idempotency_key="rollback-replay-second",
        )
        assert (await _terminal_attempt(first, second_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        approved = await first.source_status(audit_context=audit)
        binding = _rollback_binding(approved)

        original_queue = first.queue_rollback
        lost = False

        async def lose_first_response(**kwargs: Any) -> PromotionAttempt:
            nonlocal lost
            attempt = await original_queue(**kwargs)
            if not lost:
                lost = True
                raise RuntimeError("simulated rollback response loss")
            return attempt

        monkeypatch.setattr(first, "queue_rollback", lose_first_response)
        with pytest.raises(RuntimeError, match="response loss"):
            await first.source_rollback(
                idempotency_key="rollback-response-loss",
                **binding,
                reason="Undo the capabilities release",
                audit_context=audit,
            )
        digest = hashlib.sha256(b"owner\x00rollback-response-loss").hexdigest()
        expected_attempt_id = f"rollback_{digest[:48]}"
        queued = await first.get_promotion_attempt(expected_attempt_id)
        assert queued is not None
        assert queued.release.id == f"release_rollback_{digest[:48]}"
    finally:
        await first.shutdown()

    restarted = _supervisor(tmp_path, source, activator=activator)
    await restarted.start()
    try:
        replayed = await restarted.source_rollback(
            idempotency_key="rollback-response-loss",
            **binding,
            reason="Undo the capabilities release",
            audit_context=audit,
        )
        assert replayed.id == expected_attempt_id
        completed = await _terminal_attempt(restarted, replayed)
        assert completed.status is PromotionAttemptStatus.ACTIVE
        current = await restarted._archive.get_current_release()
        assert current is not None
        assert current.id == completed.release.id
        assert current.metadata["rollback_target"] == binding["expected_target_release_id"]

        with pytest.raises(EvolutionSupervisorError, match="another request"):
            await restarted.source_rollback(
                idempotency_key="rollback-response-loss",
                **binding,
                reason="A different rollback request",
                audit_context=audit,
            )
        with pytest.raises(EvolutionSupervisorError, match="source changed"):
            await restarted.source_rollback(
                idempotency_key="rollback-new-key-with-stale-binding",
                **binding,
                reason="Do not roll forward on retry",
                audit_context=audit,
            )
        other_tenant = {**audit, "tenant_id": "other-tenant"}
        with pytest.raises(EvolutionSupervisorError, match="source changed"):
            await restarted.source_rollback(
                idempotency_key="rollback-response-loss",
                **binding,
                reason="Undo the capabilities release",
                audit_context=other_tenant,
            )
    finally:
        await restarted.shutdown()

    final_restart = _supervisor(tmp_path, source, activator=activator)
    await final_restart.start()
    try:
        replayed_after_restart = await final_restart.source_rollback(
            idempotency_key="rollback-response-loss",
            **binding,
            reason="Undo the capabilities release",
            audit_context=audit,
        )
        assert replayed_after_restart.id == expected_attempt_id
        assert replayed_after_restart.status is PromotionAttemptStatus.ACTIVE
    finally:
        await final_restart.shutdown()
    assert (
        _promotion_attempt_count(
            tmp_path / "evolution.db",
            expected_attempt_id,
        )
        == 1
    )


@pytest.mark.asyncio
async def test_source_rollback_rejects_a_stale_approved_release_pair(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
    audit = _source_audit()
    await supervisor.start()
    try:
        _, first_attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="rollback-stale-first",
        )
        assert (await _terminal_attempt(supervisor, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        _, second_attempt = await _release_route(
            supervisor,
            route="capabilities",
            idempotency_key="rollback-stale-second",
        )
        assert (await _terminal_attempt(supervisor, second_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        approved = await supervisor.source_status(audit_context=audit)
        binding = _rollback_binding(approved)

        await _rollback(supervisor)
        current_before_rejected_request = await supervisor._archive.get_current_release()
        assert current_before_rejected_request is not None
        with pytest.raises(EvolutionSupervisorError, match="source changed"):
            await supervisor.source_rollback(
                idempotency_key="rollback-stale-owner-approval",
                **binding,
                reason="Stale owner approval",
                audit_context=audit,
            )
        current_after_rejected_request = await supervisor._archive.get_current_release()
        assert current_after_rejected_request == current_before_rejected_request
        digest = hashlib.sha256(b"owner\x00rollback-stale-owner-approval").hexdigest()
        assert (
            _promotion_attempt_count(
                tmp_path / "evolution.db",
                f"rollback_{digest[:48]}",
            )
            == 0
        )
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_source_release_rejects_a_base_change_during_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
    audit = _source_audit()
    await supervisor.start()
    try:
        _, first_attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="base-change-first",
        )
        assert (await _terminal_attempt(supervisor, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        edited = await supervisor.source_shell(
            command=_route_command("capabilities"),
            audit_context=audit,
        )
        original_current_release = supervisor._archive.get_current_release
        base_release = await original_current_release()
        assert base_release is not None
        intervening_release = base_release.model_copy(update={"id": "release_intervening"})
        original_evaluate = supervisor._evaluator.evaluate
        evaluation_finished = False

        async def evaluate_then_change_base(workspace: Path) -> Any:
            nonlocal evaluation_finished
            results = await original_evaluate(workspace)
            evaluation_finished = True
            return results

        async def current_release_with_intervening_change() -> Release | None:
            if evaluation_finished:
                return intervening_release
            return await original_current_release()

        monkeypatch.setattr(supervisor._evaluator, "evaluate", evaluate_then_change_base)
        monkeypatch.setattr(
            supervisor._archive,
            "get_current_release",
            current_release_with_intervening_change,
        )
        with pytest.raises(EvolutionSupervisorError, match="inactive release"):
            await supervisor.source_release(
                idempotency_key="base-change-during-evaluation",
                **_release_binding(edited),
                message="Reject stale base after evaluation",
                audit_context=audit,
            )

        operation = await supervisor._archive.get_source_release_operation(
            tenant_id="owner",
            idempotency_key="base-change-during-evaluation",
        )
        assert operation is not None
        attempt_digest = hashlib.sha256(f"{operation.id}:promotion".encode()).hexdigest()
        assert (
            _promotion_attempt_count(
                tmp_path / "evolution.db",
                f"promotion_{attempt_digest[:48]}",
            )
            == 0
        )
        assert await original_current_release() == base_release
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_failed_evaluation_cannot_be_promoted(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    runner = _EvolutionSupervisorEvaluationRunner(outcomes=[False])
    supervisor = _supervisor(
        tmp_path,
        source,
        evaluation_runner=runner,
    )
    await supervisor.start()
    try:
        edited = await supervisor.source_shell(
            command=_route_command("status"),
            audit_context=_source_audit(),
        )
        result = await supervisor.source_release(
            idempotency_key="forced-evaluation-failure",
            **_release_binding(edited),
            message="Exercise evaluator failure",
            audit_context=_source_audit(),
        )
        candidate = await supervisor.get_candidate(str(edited["candidate"]["id"]))

        assert candidate is not None
        assert candidate.status is CandidateStatus.BUILDING
        assert candidate.evaluation_report is not None
        assert candidate.evaluation_report.passed is False
        assert len(runner.calls) == 1
        assert result["promotion"] is None
        with pytest.raises(EvolutionSupervisorError, match="not ready"):
            await supervisor.queue_promotion(candidate.id)
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_activation_failure_is_persisted_without_false_promotion(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    activator.failure = (
        ReleaseActivationStatus.FAILED,
        "staging_unhealthy",
        "The staged release was unhealthy.",
    )
    supervisor = _supervisor(tmp_path, source, activator=activator)
    await supervisor.start()
    try:
        candidate, queued = await _release_route(
            supervisor,
            route="status",
            idempotency_key="activation-failure",
        )
        assert queued.status is PromotionAttemptStatus.QUEUED
        attempt = await _terminal_attempt(supervisor, queued)

        retained = await supervisor.get_candidate(candidate.id)
        assert retained is not None
        assert retained.status is CandidateStatus.READY
        failure = retained.metadata["last_activation_failure"]
        assert isinstance(failure, dict)
        assert str(failure["attempt_id"]) == attempt.id
        assert attempt.status is PromotionAttemptStatus.FAILED
        assert attempt.failure_code == "staging_unhealthy"
        assert await supervisor._archive.get_current_release() is None
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_promotion_request_returns_before_background_activation(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    entered = asyncio.Event()
    release_activation = asyncio.Event()

    async def block_activation(release: ReleaseRecord, rollback: bool) -> None:
        del release, rollback
        entered.set()
        await asyncio.wait_for(release_activation.wait(), timeout=1)

    activator.before_activate = block_activation
    supervisor = _supervisor(
        tmp_path,
        source,
        activator=activator,
        automatic_promotions=True,
    )
    terminal = asyncio.Event()
    publish = supervisor._publish_promotion_event

    async def record_terminal(*args: Any, **kwargs: Any) -> None:
        await publish(*args, **kwargs)
        terminal.set()

    supervisor._publish_promotion_event = record_terminal  # type: ignore[method-assign]
    await supervisor.start()
    try:
        _, queued = await _release_route(
            supervisor,
            route="status",
            idempotency_key="background-activation",
        )

        assert queued.status is PromotionAttemptStatus.QUEUED
        await asyncio.wait_for(entered.wait(), timeout=1)
        pending = await supervisor.get_promotion_attempt(queued.id)
        assert pending is not None and pending.status is PromotionAttemptStatus.ACTIVATING
        release_activation.set()
        await asyncio.wait_for(terminal.wait(), timeout=1)
        completed = await supervisor.get_promotion_attempt(queued.id)
        assert completed is not None
        assert completed.status is PromotionAttemptStatus.ACTIVE
    finally:
        release_activation.set()
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_rollback_activation_failure_keeps_current_release(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    supervisor = _supervisor(tmp_path, source, activator=activator)
    await supervisor.start()
    try:
        _, first_attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="rollback-first-generation",
        )
        assert (await _terminal_attempt(supervisor, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        second, second_attempt = await _release_route(
            supervisor,
            route="capabilities",
            idempotency_key="rollback-second-generation",
        )
        assert (await _terminal_attempt(supervisor, second_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        second_release = await supervisor._archive.get_current_release()
        assert second_release is not None
        activator.failure = (
            ReleaseActivationStatus.ROLLED_BACK,
            "probation_unhealthy",
            "The candidate failed probation and was rolled back.",
        )

        queued = await supervisor.queue_rollback()
        attempt = await _terminal_attempt(supervisor, queued)
        assert attempt.status is PromotionAttemptStatus.FAILED
        assert attempt.failure_code == "probation_unhealthy"

        current = await supervisor._archive.get_current_release()
        retained = await supervisor.get_candidate(second.id)
        assert current is not None and current.id == second_release.id
        assert retained is not None and retained.status is CandidateStatus.PROMOTED
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_candidate_completion_event_retains_full_origin_and_survives_restart(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    first = _supervisor(tmp_path, source, event_sink=_FailingEventSink())
    await first.start()
    try:
        audit = {
            "tenant_id": "tenant_1",
            "actor_id": "owner_1",
            "thread_id": "thread_1",
            "channel": "telegram",
            "run_kind": "owner",
            "correlation_id": "correlation_1",
            "origin": (
                '{"interface":"telegram","source_id":"bot_1",'
                '"conversation_id":"chat_1","message_id":"message_1"}'
            ),
        }
        candidate, _ = await _release_route(
            first,
            route="status",
            idempotency_key="event-origin",
            audit=audit,
        )
        assert candidate.status in {CandidateStatus.READY, CandidateStatus.PROMOTED}
        assert (await first._archive.pending_events())[0].attempt_count >= 1
    finally:
        await first.shutdown()

    sink = InMemoryEvolutionEventSink()
    restarted = _supervisor(tmp_path, source, event_sink=sink)
    await restarted.start()
    try:
        assert [event.event_type for event in sink.events] == [
            "build.preparing",
            "candidate.ready",
        ]
        event = next(event for event in sink.events if event.event_type == "candidate.ready")
        assert event.event_type == "candidate.ready"
        assert event.origin == {
            "tenant_id": "tenant_1",
            "actor_id": "owner_1",
            "thread_id": "thread_1",
            "channel": "telegram",
            "run_kind": "owner",
            "correlation_id": "correlation_1",
            "origin": (
                '{"interface":"telegram","source_id":"bot_1",'
                '"conversation_id":"chat_1","message_id":"message_1"}'
            ),
        }
        assert all(item.origin == event.origin for item in sink.events)
        assert await restarted._archive.pending_events() == []
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_restart_commits_bootstrap_active_attempt_after_archive_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    first = _supervisor(tmp_path, source, activator=activator)
    await first.start()
    archive = first._archive
    promote_candidate = archive.promote_candidate
    interrupted_archive = asyncio.Event()

    async def interrupted(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        interrupted_archive.set()
        raise OSError("simulated archive interruption")

    monkeypatch.setattr(archive, "promote_candidate", interrupted)
    try:
        candidate, attempt = await _release_route(
            first,
            route="status",
            idempotency_key="archive-interruption",
        )
        assert await first.process_queued_promotions() == 1
        await asyncio.wait_for(interrupted_archive.wait(), timeout=1)
        attempts = await archive.list_incomplete_promotion_attempts()
        retained = await first.get_candidate(candidate.id)
        assert retained is not None and retained.status is CandidateStatus.READY
        assert len(attempts) == 1
        assert attempts[0].id == attempt.id
        assert attempts[0].status is PromotionAttemptStatus.ACTIVATING
    finally:
        await first.shutdown()
    monkeypatch.setattr(archive, "promote_candidate", promote_candidate)

    restarted = _supervisor(tmp_path, source, activator=activator)
    await restarted.start()
    try:
        completed = await _terminal_attempt(restarted, attempt)
        assert completed.status is PromotionAttemptStatus.ACTIVE
        recovered = await restarted.get_candidate(candidate.id)
        current = await restarted._archive.get_current_release()
        assert recovered is not None and recovered.status is CandidateStatus.PROMOTED
        assert current is not None and current.candidate_id == candidate.id
        assert await restarted._archive.list_incomplete_promotion_attempts() == []
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_lineage_initializes_from_archive_and_derives_old_release_acceptance(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    source_commit = _git(source, "rev-parse", "HEAD")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=source_commit)
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=source_commit,
        accepted_upstream_commit=None,
    )

    await supervisor.start()
    try:
        status = await supervisor.source_status(audit_context=_source_audit())
        assert lineage.resolve_ref(INSTANCE_REF) == source_commit
        assert lineage.resolve_ref(ACCEPTED_UPSTREAM_REF) == source_commit
        assert status["active_commit"] == source_commit
        assert status["instance_commit"] == source_commit
        assert status["upstream_commit"] == source_commit
        assert status["accepted_upstream_commit"] == source_commit
        assert status["merge_base_commit"] == source_commit
        assert status["upstream_pending"] is False
        assert status["conflict_paths"] == []
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_source_sync_upstream_binds_active_release_and_opens_reconciliation(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    active_commit = _git(source, "rev-parse", "HEAD")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=active_commit)
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    release = await _seed_active_release(
        supervisor,
        source_commit=active_commit,
        accepted_upstream_commit=active_commit,
    )
    await supervisor.start()

    synced = await supervisor.source_sync_upstream(
        expected_active_release_id=release.id,
        audit_context=_source_audit(),
    )

    assert synced["synced"] is True
    assert synced["previous_upstream_commit"] == active_commit
    assert synced["upstream_commit"] != active_commit
    assert synced["session_active"] is True
    assert synced["candidate_id"]
    assert synced["upstream_pending"] is True
    with pytest.raises(EvolutionSupervisorError, match="finish or release"):
        await supervisor.source_sync_upstream(
            expected_active_release_id=release.id,
            audit_context=_source_audit(),
        )
    with pytest.raises(EvolutionSupervisorError, match="active release changed"):
        await supervisor.source_sync_upstream(
            expected_active_release_id="release-stale",
            audit_context={**_source_audit(), "tenant_id": "other-owner"},
        )


@pytest.mark.asyncio
async def test_local_release_advances_instance_and_preserves_accepted_upstream(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        candidate, attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="lineage-local",
        )
        assert (await _terminal_attempt(supervisor, attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        assert candidate.source_commit is not None
        assert lineage.resolve_ref(INSTANCE_REF) == candidate.source_commit
        assert lineage.resolve_ref(ACCEPTED_UPSTREAM_REF) == base
        release = await supervisor._archive.get_current_release()
        assert release is not None
        assert release.metadata["accepted_upstream_commit"] == base
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_pending_upstream_is_committed_with_exact_two_parent_order(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    _git(source, "branch", "upstream", base)
    (source / "instance.txt").write_text("instance\n", encoding="utf-8")
    _git(source, "add", "instance.txt")
    _git(source, "commit", "-m", "instance change")
    instance = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "upstream")
    with (source / "site_app.py").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
        )
    _git(source, "add", "site_app.py")
    _git(source, "commit", "-m", "upstream status route")
    upstream = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "main")
    lineage = _lineage(source, tmp_path)
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=instance,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        merged = await supervisor.source_shell(command="true", audit_context=_source_audit())
        candidate = await supervisor.get_candidate(str(merged["candidate_id"]))
        assert candidate is not None
        assert UpstreamLineage.model_validate(
            candidate.metadata[UPSTREAM_LINEAGE_METADATA_KEY]
        ) == UpstreamLineage(upstream_commit=upstream, merge_base_commit=base)
        released = await supervisor.source_release(
            idempotency_key="upstream-clean-merge",
            **_release_binding(merged),
            audit_context=_source_audit(),
        )
        candidate = await supervisor.get_candidate(str(merged["candidate_id"]))
        assert candidate is not None and candidate.source_commit is not None
        assert _git(source, "show", "-s", "--format=%P", candidate.source_commit).split() == [
            instance,
            upstream,
        ]
        attempt = PromotionAttempt.model_validate(released["promotion"])
        assert (await _terminal_attempt(supervisor, attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        assert _git(source, "rev-parse", ACCEPTED_UPSTREAM_REF) == upstream
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_upstream_conflict_persists_until_native_index_is_resolved(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    (source / "conflict.txt").write_text("base\n", encoding="utf-8")
    _git(source, "add", "conflict.txt")
    _git(source, "commit", "-m", "conflict base")
    base = _git(source, "rev-parse", "HEAD")
    _git(source, "branch", "upstream", base)
    (source / "conflict.txt").write_text("instance\n", encoding="utf-8")
    _git(source, "commit", "-am", "instance conflict")
    instance = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "upstream")
    (source / "conflict.txt").write_text("upstream\n", encoding="utf-8")
    with (source / "site_app.py").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n@app.get('/status')\n"
            "def status():\n"
            "    return {'runtime': 'opentulpa', 'status': 'ready'}\n"
        )
    _git(source, "add", "conflict.txt", "site_app.py")
    _git(source, "commit", "-m", "upstream conflict")
    _git(source, "switch", "main")
    lineage = _lineage(source, tmp_path)
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=instance,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        conflicted = await supervisor.source_shell(command="true", audit_context=_source_audit())
        assert conflicted["conflict_paths"] == ["conflict.txt"]
        resolved = await supervisor.source_shell(
            command="printf 'resolved\n' > conflict.txt",
            audit_context=_source_audit(),
        )
        assert resolved["conflict_paths"] == []
        released = await supervisor.source_release(
            idempotency_key="upstream-conflict-resolution",
            **_release_binding(resolved),
            audit_context=_source_audit(),
        )
        attempt = PromotionAttempt.model_validate(released["promotion"])
        assert (await _terminal_attempt(supervisor, attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_lineage_rejects_stale_upstream_before_build(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    builder = _FakeReleaseBuilder()
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    supervisor = _supervisor(tmp_path, source, builder=builder, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        edited = await supervisor.source_shell(
            command=_route_command("status"),
            audit_context=_source_audit(),
        )
        lineage.advance_upstream()
        with pytest.raises(EvolutionSupervisorError, match="lineage changed"):
            await supervisor.source_release(
                idempotency_key="stale-upstream",
                **_release_binding(edited),
                audit_context=_source_audit(),
            )
        assert builder.requests == []
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_activation_failure_leaves_archive_and_lineage_at_previous_release(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    activator = _FakeReleaseActivator()
    activator.failure = (
        ReleaseActivationStatus.FAILED,
        "unhealthy",
        "Candidate was unhealthy.",
    )
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    supervisor = _supervisor(tmp_path, source, activator=activator, lineage=lineage)
    initial = await _seed_active_release(
        supervisor,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        _, attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="lineage-activation-failure",
        )
        assert (await _terminal_attempt(supervisor, attempt)).status is (
            PromotionAttemptStatus.FAILED
        )
        current = await supervisor._archive.get_current_release()
        assert current is not None and current.id == initial.id
        assert lineage.resolve_ref(INSTANCE_REF) == base
        assert lineage.resolve_ref(ACCEPTED_UPSTREAM_REF) == base
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_rollback_restores_instance_and_accepted_refs(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        first, first_attempt = await _release_route(
            supervisor,
            route="status",
            idempotency_key="lineage-rollback-first",
        )
        assert (await _terminal_attempt(supervisor, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        first_release = await supervisor._archive.get_current_release()
        assert first_release is not None and first.source_commit is not None
        _, second_attempt = await _release_route(
            supervisor,
            route="capabilities",
            idempotency_key="lineage-rollback-second",
        )
        assert (await _terminal_attempt(supervisor, second_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        await _rollback(supervisor)
        assert lineage.resolve_ref(INSTANCE_REF) == first.source_commit
        assert lineage.resolve_ref(ACCEPTED_UPSTREAM_REF) == base
        current = await supervisor._archive.get_current_release()
        assert current is not None
        assert current.metadata["rollback_target"] == first_release.id
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_restart_repairs_archive_new_lineage_old_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    _git(source, "branch", "upstream", base)
    lineage = _lineage(source, tmp_path)
    first = _supervisor(tmp_path, source, lineage=lineage)
    initial = await _seed_active_release(
        first,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await first.start()
    interrupted_projection = asyncio.Event()

    async def interrupt_projection(release: Release) -> None:
        del release
        interrupted_projection.set()
        raise OSError("simulated projection interruption")

    monkeypatch.setattr(first, "_project_release", interrupt_projection)
    try:
        candidate, attempt = await _release_route(
            first,
            route="status",
            idempotency_key="projection-interruption",
        )
        assert await first.process_queued_promotions() == 1
        await asyncio.wait_for(interrupted_projection.wait(), timeout=1)
        current = await first._archive.get_current_release()
        pending = await first.get_promotion_attempt(attempt.id)
        assert current is not None and current.candidate_id == candidate.id
        assert pending is not None and pending.status is PromotionAttemptStatus.ACTIVATING
        assert _git(source, "rev-parse", INSTANCE_REF) == initial.source_commit
    finally:
        await first.shutdown()

    restarted = _supervisor(tmp_path, source, lineage=_lineage(source, tmp_path))
    await restarted.start()
    try:
        assert candidate.source_commit is not None
        assert _git(source, "rev-parse", INSTANCE_REF) == candidate.source_commit
        assert (await _terminal_attempt(restarted, attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_start_fails_closed_on_unexplained_lineage_divergence(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    first = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        first,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await first.start()
    try:
        lineage.diverge_instance()
    finally:
        await first.shutdown()

    restarted = _supervisor(tmp_path, source, lineage=lineage)
    with pytest.raises(EvolutionSupervisorError, match="diverged"):
        await restarted.start()
    await restarted._archive.shutdown()


@pytest.mark.asyncio
async def test_python_generation_metadata_and_evaluation_digest_are_wired(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(source, "add", "uv.lock")
    _git(source, "commit", "-m", "add dependency lock")
    base = _git(source, "rev-parse", "HEAD")
    builder = _FakeReleaseBuilder(artifact_kind="python_generation")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    supervisor = _supervisor(
        tmp_path,
        source,
        builder=builder,
        lineage=lineage,
    )
    await _seed_active_release(
        supervisor,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        edited = await supervisor.source_shell(
            command=_route_command("status"),
            audit_context=_source_audit(),
        )
        candidate = await supervisor.get_candidate(str(edited["candidate_id"]))
        assert candidate is not None
        candidate = await supervisor._archive.update_candidate(
            candidate.model_copy(
                update={
                    "metadata": {
                        **candidate.metadata,
                        "state_contract_sha256": "f" * 64,
                        "install_profile": "runtime",
                        "controller_protocol": 1,
                    }
                }
            ),
            expected_revision=candidate.revision,
        )
        released = await supervisor.source_release(
            idempotency_key="python-generation",
            **_release_binding(edited),
            audit_context=_source_audit(),
        )
        candidate = await supervisor.get_candidate(candidate.id)
        assert candidate is not None and candidate.source_commit is not None
        assert len(builder.requests) == 1
        request = builder.requests[0]
        expected_input = hashlib.sha256(
            (
                f"{candidate.source_commit}:{candidate.dependency_lock_hash}:"
                f"opentulpa-evaluator-v1:{candidate.evaluator_fingerprint}"
            ).encode()
        ).hexdigest()
        assert request.evaluation_input_sha256 == expected_input
        assert candidate.metadata["artifact_kind"] == "python_generation"
        assert candidate.metadata["generation_id"]
        assert candidate.metadata["accepted_upstream_commit"] == base
        promotion = PromotionAttempt.model_validate(released["promotion"])
        assert promotion.release.metadata["generation_id"] == candidate.metadata["generation_id"]
        assert promotion.release.metadata["manifest_digest"] == candidate.metadata["manifest_digest"]
        assert promotion.release.metadata["dependency_lock_hash"] == candidate.dependency_lock_hash
        assert promotion.release.metadata["state_contract_sha256"] == "f" * 64
        assert promotion.release.metadata["install_profile"] == "runtime"
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_restart_adopts_legacy_local_session_without_lineage_metadata(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    lineage = _EvolutionSupervisorLineageFake(instance_commit=base)
    first = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        first,
        source_commit=base,
        accepted_upstream_commit=base,
    )
    await first.start()
    try:
        edited = await first.source_shell(
            command="printf 'legacy session\n' > legacy.txt",
            audit_context=_source_audit(),
        )
        candidate = await first.get_candidate(str(edited["candidate_id"]))
        assert candidate is not None and candidate.worktree_path is not None
        metadata = dict(candidate.metadata)
        metadata.pop(UPSTREAM_LINEAGE_METADATA_KEY)
        metadata.pop("accepted_upstream_commit")
        await first._archive.update_candidate(
            candidate.model_copy(update={"metadata": metadata}),
            expected_revision=candidate.revision,
        )
        worktree = Path(candidate.worktree_path)
    finally:
        await first.shutdown()

    restarted = _supervisor(tmp_path, source, lineage=lineage)
    await restarted.start()
    try:
        status = await restarted.source_status(audit_context=_source_audit())
        repaired = await restarted.get_candidate(candidate.id)
        assert status["candidate_id"] == candidate.id
        assert "legacy.txt" in status["changed_files"]
        assert worktree.is_dir()
        assert repaired is not None
        assert UpstreamLineage.model_validate(
            repaired.metadata[UPSTREAM_LINEAGE_METADATA_KEY]
        ) == UpstreamLineage(upstream_commit=base, merge_base_commit=base)
        assert repaired.metadata["accepted_upstream_commit"] == base
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_restart_adopts_legacy_session_from_native_merge_head(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    _git(source, "branch", "upstream", base)
    (source / "instance.txt").write_text("instance\n", encoding="utf-8")
    _git(source, "add", "instance.txt")
    _git(source, "commit", "-m", "instance")
    instance = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "upstream")
    (source / "upstream.txt").write_text("upstream\n", encoding="utf-8")
    _git(source, "add", "upstream.txt")
    _git(source, "commit", "-m", "upstream")
    upstream = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "main")
    lineage = _EvolutionSupervisorLineageFake(
        instance_commit=instance,
        upstream_commit=upstream,
        accepted_upstream_commit=base,
    )
    first = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        first,
        source_commit=instance,
        accepted_upstream_commit=base,
    )
    await first.start()
    try:
        opened = await first.source_shell(command="true", audit_context=_source_audit())
        candidate = await first.get_candidate(str(opened["candidate_id"]))
        assert candidate is not None
        metadata = dict(candidate.metadata)
        metadata.pop(UPSTREAM_LINEAGE_METADATA_KEY)
        metadata.pop("accepted_upstream_commit")
        await first._archive.update_candidate(
            candidate.model_copy(update={"metadata": metadata}),
            expected_revision=candidate.revision,
        )
    finally:
        await first.shutdown()

    restarted = _supervisor(tmp_path, source, lineage=lineage)
    await restarted.start()
    try:
        repaired = await restarted.get_candidate(candidate.id)
        status = await restarted.source_status(audit_context=_source_audit())
        assert repaired is not None
        assert UpstreamLineage.model_validate(
            repaired.metadata[UPSTREAM_LINEAGE_METADATA_KEY]
        ) == UpstreamLineage(upstream_commit=upstream, merge_base_commit=base)
        assert repaired.metadata["accepted_upstream_commit"] == upstream
        assert status["candidate_id"] == candidate.id
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_failed_evaluated_merge_accepts_a_verified_fixup_commit(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    base = _git(source, "rev-parse", "HEAD")
    _git(source, "branch", "upstream", base)
    (source / "instance.txt").write_text("instance\n", encoding="utf-8")
    _git(source, "add", "instance.txt")
    _git(source, "commit", "-m", "instance")
    instance = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "upstream")
    (source / "upstream.txt").write_text("upstream\n", encoding="utf-8")
    _git(source, "add", "upstream.txt")
    _git(source, "commit", "-m", "upstream")
    upstream = _git(source, "rev-parse", "HEAD")
    _git(source, "switch", "main")
    lineage = _EvolutionSupervisorLineageFake(
        instance_commit=instance,
        upstream_commit=upstream,
        accepted_upstream_commit=base,
    )
    supervisor = _supervisor(tmp_path, source, lineage=lineage)
    await _seed_active_release(
        supervisor,
        source_commit=instance,
        accepted_upstream_commit=base,
    )
    await supervisor.start()
    try:
        merged = await supervisor.source_shell(command="true", audit_context=_source_audit())
        failed = await supervisor.source_release(
            idempotency_key="merge-evaluation-failed",
            **_release_binding(merged),
            audit_context=_source_audit(),
        )
        assert failed["promotion"] is None
        after_merge = await supervisor.get_candidate(str(merged["candidate_id"]))
        assert after_merge is not None and after_merge.source_commit is not None
        merge_commit = after_merge.source_commit
        assert lineage.merge_parents(merge_commit) == (
            instance,
            upstream,
        )

        fixed = await supervisor.source_shell(
            command=_route_command("status"),
            audit_context=_source_audit(),
        )
        released = await supervisor.source_release(
            idempotency_key="merge-evaluation-fixup",
            **_release_binding(fixed),
            audit_context=_source_audit(),
        )
        final = await supervisor.get_candidate(str(merged["candidate_id"]))
        assert final is not None and final.source_commit is not None
        workspaces = supervisor._workspaces
        assert isinstance(workspaces, _EvolutionSupervisorWorkspaceFake)
        assert workspaces.parent(final.source_commit) == merge_commit
        assert final.metadata["opentulpa.evolution.upstream_merge_commit"] == merge_commit
        attempt = PromotionAttempt.model_validate(released["promotion"])
        assert (await _terminal_attempt(supervisor, attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_restart_reconstructs_active_promotion_event_after_enqueue_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    first = _supervisor(tmp_path, source)
    await first.start()
    crashed = asyncio.Event()

    async def crash_before_event(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        crashed.set()
        raise RuntimeError("simulated terminal event crash")

    monkeypatch.setattr(first, "_publish_promotion_event", crash_before_event)
    try:
        _, queued = await _release_route(
            first,
            route="status",
            idempotency_key="active-event-crash",
        )
        with pytest.raises(RuntimeError, match="terminal event crash"):
            await first.process_queued_promotions()
        await asyncio.wait_for(crashed.wait(), timeout=1)
        completed = await first.get_promotion_attempt(queued.id)
        assert completed is not None and completed.status is PromotionAttemptStatus.ACTIVE
    finally:
        await first.shutdown()

    sink = InMemoryEvolutionEventSink()
    restarted = _supervisor(tmp_path, source, event_sink=sink)
    await restarted.start()
    try:
        matching = [event for event in sink.events if event.event_type == "promotion.active"]
        assert len(matching) == 1
        assert matching[0].payload["attempt_id"] == queued.id
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_restart_reconstructs_failed_promotion_event_after_enqueue_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    activator = _FakeReleaseActivator()
    activator.failure = (
        ReleaseActivationStatus.FAILED,
        "staging_unhealthy",
        "The staged release was unhealthy.",
    )
    first = _supervisor(tmp_path, source, activator=activator)
    await first.start()
    crashed = asyncio.Event()

    async def crash_before_event(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        crashed.set()
        raise RuntimeError("simulated terminal event crash")

    monkeypatch.setattr(first, "_publish_promotion_event", crash_before_event)
    try:
        _, queued = await _release_route(
            first,
            route="status",
            idempotency_key="failed-event-crash",
        )
        with pytest.raises(RuntimeError, match="terminal event crash"):
            await first.process_queued_promotions()
        await asyncio.wait_for(crashed.wait(), timeout=1)
        completed = await first.get_promotion_attempt(queued.id)
        assert completed is not None and completed.status is PromotionAttemptStatus.FAILED
    finally:
        await first.shutdown()

    sink = InMemoryEvolutionEventSink()
    restarted = _supervisor(tmp_path, source, activator=activator, event_sink=sink)
    await restarted.start()
    try:
        matching = [event for event in sink.events if event.event_type == "promotion.failed"]
        assert len(matching) == 1
        assert matching[0].payload["attempt_id"] == queued.id
    finally:
        await restarted.shutdown()


@pytest.mark.asyncio
async def test_restart_reconstructs_rollback_event_after_enqueue_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source_repository(tmp_path)
    first = _supervisor(tmp_path, source)
    await first.start()
    crashed = asyncio.Event()
    try:
        _, first_attempt = await _release_route(
            first,
            route="status",
            idempotency_key="rollback-event-first",
        )
        assert (await _terminal_attempt(first, first_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )
        _, second_attempt = await _release_route(
            first,
            route="capabilities",
            idempotency_key="rollback-event-second",
        )
        assert (await _terminal_attempt(first, second_attempt)).status is (
            PromotionAttemptStatus.ACTIVE
        )

        async def crash_before_event(*args: Any, **kwargs: Any) -> None:
            del args, kwargs
            crashed.set()
            raise RuntimeError("simulated terminal event crash")

        monkeypatch.setattr(first, "_publish_promotion_event", crash_before_event)
        queued = await first.queue_rollback()
        with pytest.raises(RuntimeError, match="terminal event crash"):
            await first.process_queued_promotions()
        await asyncio.wait_for(crashed.wait(), timeout=1)
        completed = await first.get_promotion_attempt(queued.id)
        assert completed is not None and completed.status is PromotionAttemptStatus.ACTIVE
    finally:
        await first.shutdown()

    sink = InMemoryEvolutionEventSink()
    restarted = _supervisor(tmp_path, source, event_sink=sink)
    await restarted.start()
    try:
        matching = [event for event in sink.events if event.event_type == "rollback.active"]
        assert len(matching) == 1
        assert matching[0].payload["attempt_id"] == queued.id
    finally:
        await restarted.shutdown()
