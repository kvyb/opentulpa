from __future__ import annotations

import asyncio
import hashlib
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from opentulpa.bootstrap.models import ReleaseOrigin, ReleaseRecord
from opentulpa.evolution.activation import (
    ReleaseActivationResult,
    ReleaseActivationStatus,
)
from opentulpa.evolution.archive import EvolutionArchive
from opentulpa.evolution.evaluator import (
    CandidateEvaluator,
    EvaluationCommand,
    LocalEvaluationRunner,
)
from opentulpa.evolution.models import (
    Candidate,
    CandidateStatus,
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
from opentulpa.evolution.workspace import GitCandidateWorkspace


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
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.requests: list[ReleaseBuildRequest] = []

    async def build(self, request: ReleaseBuildRequest) -> OciReleaseArtifact:
        self.requests.append(request)
        if self.fail:
            raise ReleaseBuildError("Candidate OCI image build failed.")
        image = hashlib.sha256(f"image:{request.source_commit}".encode()).hexdigest()
        manifest = hashlib.sha256(f"manifest:{request.source_commit}".encode()).hexdigest()
        return OciReleaseArtifact(
            artifact_digest=f"sha256:{image}",
            manifest_digest=f"sha256:{manifest}",
            image_reference=f"opentulpa-release:{manifest[:32]}",
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


def _supervisor(
    tmp_path: Path,
    source: Path,
    *,
    builder: _FakeReleaseBuilder | None = None,
    activator: _FakeReleaseActivator | None = None,
    event_sink: Any = None,
) -> EvolutionSupervisor:
    return EvolutionSupervisor(
        archive=EvolutionArchive(tmp_path / "evolution.db"),
        workspaces=GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "contributions",
        ),
        candidate_backend_factory=_WritableSourceBackend,
        evaluator=CandidateEvaluator(
            runner=LocalEvaluationRunner(),
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
    )


async def _terminal_attempt(
    supervisor: EvolutionSupervisor,
    attempt: PromotionAttempt,
) -> PromotionAttempt:
    for _ in range(300):
        current = await supervisor.get_promotion_attempt(attempt.id)
        assert current is not None
        if current.status in {PromotionAttemptStatus.ACTIVE, PromotionAttemptStatus.FAILED}:
            return current
        await asyncio.sleep(0.01)
    raise AssertionError("promotion attempt did not reach a terminal state")


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
async def test_source_release_keeps_failed_session_editable_and_retries_same_commit(
    tmp_path: Path,
) -> None:
    source = _source_repository(tmp_path)
    builder = _FakeReleaseBuilder(fail=True)
    supervisor = _supervisor(tmp_path, source, builder=builder)
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
        evolved_source = _git(source, "show", f"{candidate.source_commit}:site_app.py")
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
        assert (tmp_path / "contributions" / patch_name).is_file()
    finally:
        await supervisor.shutdown()


@pytest.mark.asyncio
async def test_second_generation_can_roll_back_to_prior_release(tmp_path: Path) -> None:
    source = _source_repository(tmp_path)
    supervisor = _supervisor(tmp_path, source)
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
    supervisor = EvolutionSupervisor(
        archive=EvolutionArchive(tmp_path / "evolution.db"),
        workspaces=GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "contributions",
        ),
        candidate_backend_factory=_WritableSourceBackend,
        evaluator=CandidateEvaluator(
            runner=LocalEvaluationRunner(),
            commands=(
                EvaluationCommand(
                    name="forced.failure",
                    argv=(sys.executable, "-c", "raise SystemExit(9)"),
                ),
            ),
        ),
        release_pointer=AtomicReleasePointer(tmp_path / "release.json"),
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
        await release_activation.wait()

    activator.before_activate = block_activation
    supervisor = _supervisor(tmp_path, source, activator=activator)
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
        completed = await _terminal_attempt(supervisor, queued)
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
        assert len(sink.events) == 1
        event = sink.events[0]
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

    async def interrupted(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("simulated archive interruption")

    monkeypatch.setattr(archive, "promote_candidate", interrupted)
    candidate, attempt = await _release_route(
        first,
        route="status",
        idempotency_key="archive-interruption",
    )
    for _ in range(300):
        attempts = await archive.list_incomplete_promotion_attempts()
        if attempts and attempts[0].status is PromotionAttemptStatus.ACTIVATING:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError("promotion attempt was not activated")
    retained = await first.get_candidate(candidate.id)
    assert retained is not None and retained.status is CandidateStatus.READY
    assert len(attempts) == 1
    assert attempts[0].id == attempt.id
    assert attempts[0].status is PromotionAttemptStatus.ACTIVATING
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
