from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from opentulpa.evolution.process import BoundedProcessResult
from opentulpa.host.evolution import (
    HostEvolutionControlService,
    SourceEvolutionError,
    _ActivationJournal,
    _TrustedSourceWorkspace,
)
from opentulpa.host.reviewer import ReleaseReviewDecision
from opentulpa.host.runtime import RuntimeLiveSourceSpec
from opentulpa.host.runtime_environment import (
    LiveSourceRuntimeEnvironment,
    LiveSourceRuntimeEnvironmentStore,
)
from opentulpa.inference.models import InferenceSelection, ResolvedInferencePlan


def _seed(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "seed"
    package = root / "src" / "opentulpa"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "opentulpa"\nversion = "0"\n', encoding="utf-8"
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (root / "README.md").write_text("before\n", encoding="utf-8")
    _git(root, "init")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.test")
    _git(root, "add", "--all")
    _git(root, "commit", "-m", "initial")
    return root, _git(root, "rev-parse", "HEAD")


def test_trusted_workspace_persists_direct_edits_and_imports_exact_commit(
    tmp_path: Path,
) -> None:
    source, bundled = _seed(tmp_path)
    path = tmp_path / "control" / "source"
    workspace = _TrustedSourceWorkspace(
        source_repository=source,
        path=path,
        max_output_bytes=100_000,
    )

    assert workspace.prepare() == bundled
    assert workspace.read("README.md", offset=1, limit=10)["content"] == "before\n"
    workspace.edit("README.md", old_text="before", new_text="after", replace_all=False)
    assert workspace.bash("git status --short", timeout_seconds=10)["exit_code"] == 0
    commit, changed = workspace.commit("Improve source")

    assert changed is True
    assert commit != bundled
    assert workspace.changes() == ()
    assert _git(source, "cat-file", "-e", commit, check=False) == ""

    workspace.import_into_live_repository(commit)
    assert _git(source, "rev-parse", f"{commit}^{{commit}}") == commit
    assert _git(source, "show", f"{commit}:README.md") == "after"

    reopened = _TrustedSourceWorkspace(
        source_repository=source,
        path=path,
        max_output_bytes=100_000,
    )
    assert reopened.prepare() == bundled
    assert reopened.head() == commit


@pytest.mark.parametrize("credential_path", [".env", "config/credentials.json", "secrets.yaml"])
def test_trusted_workspace_refuses_to_activate_credential_files(
    tmp_path: Path,
    credential_path: str,
) -> None:
    source, _ = _seed(tmp_path)
    workspace = _TrustedSourceWorkspace(
        source_repository=source,
        path=tmp_path / "control" / "source",
        max_output_bytes=100_000,
    )
    workspace.prepare()
    workspace.write(credential_path, "TOKEN=secret\n")

    with pytest.raises(SourceEvolutionError, match="credential"):
        workspace.commit("Do not commit this")

    assert credential_path in "\n".join(workspace.changes())


def test_trusted_workspace_validates_committed_files_and_allows_env_example(
    tmp_path: Path,
) -> None:
    source, _ = _seed(tmp_path)
    workspace = _TrustedSourceWorkspace(
        source_repository=source,
        path=tmp_path / "control" / "source",
        max_output_bytes=100_000,
    )
    workspace.prepare()
    workspace.write(".env.example", "TOKEN=placeholder\n")
    _, changed = workspace.commit("Document environment")
    assert changed is True

    workspace.write("credentials.json", '{"token":"secret"}\n')
    assert workspace.bash(
        "git add credentials.json && git commit -m credential",
        timeout_seconds=10,
    )["exit_code"] == 0

    with pytest.raises(SourceEvolutionError, match="credential"):
        workspace.commit("Activate committed source")


def test_runtime_environment_records_final_interpreter_path(tmp_path: Path) -> None:
    source, commit = _seed(tmp_path)
    uv = tmp_path / "uv"
    python = tmp_path / "python"
    for executable in (uv, python):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o700)

    def run_uv(*args: Any, **kwargs: Any) -> BoundedProcessResult:
        commands.append(tuple(args[0]))
        target = Path(kwargs["env"]["UV_PROJECT_ENVIRONMENT"])
        interpreter = target / "bin" / "python"
        interpreter.parent.mkdir()
        interpreter.write_text("#!/bin/sh\n", encoding="utf-8")
        interpreter.chmod(0o700)
        return BoundedProcessResult(returncode=0, output=b"", truncated=False, timed_out=False)

    envs = tmp_path / "runtime-envs"
    commands: list[tuple[str, ...]] = []
    store = LiveSourceRuntimeEnvironmentStore(
        source_repository=source,
        envs_root=envs,
        worktrees_root=tmp_path / "runtime-worktrees",
        uv_cli=str(uv),
        python_executable=str(python),
        extras=("bundled",),
        runner=run_uv,
    )

    environment = store.prepare(commit)
    expected = envs / environment.id / "bin" / "python"
    metadata = json.loads((expected.parents[1] / "runtime-env.json").read_text(encoding="utf-8"))

    assert environment.python_interpreter == expected
    assert expected.is_file()
    assert metadata["python_interpreter"] == str(expected)
    assert commands[0][2:7] == (
        "--frozen",
        "--no-dev",
        "--no-install-project",
        "--extra",
        "bundled",
    )
    assert store.prepare(commit).python_interpreter == expected


def test_activation_journal_is_idempotent_and_persists_release_state(tmp_path: Path) -> None:
    journal_path = tmp_path / "activations.db"
    journal = _ActivationJournal(journal_path)
    initial = "a" * 40
    target = "b" * 40
    state = journal.initialize(initial)
    release = journal.release_for_commit(target)

    operation, replayed = journal.begin(
        tenant_id="owner",
        idempotency_key="activate-1",
        request_hash="c" * 64,
        kind="activate",
        target_release_id=release["id"],
        previous_release_id=state["active_release_id"],
        reason="test",
        audit={"tenant_id": "owner"},
    )
    repeated, replayed_again = journal.begin(
        tenant_id="owner",
        idempotency_key="activate-1",
        request_hash="c" * 64,
        kind="activate",
        target_release_id=release["id"],
        previous_release_id=state["active_release_id"],
        reason="test",
        audit={"tenant_id": "owner"},
    )

    assert replayed is False
    assert replayed_again is True
    assert repeated["id"] == operation["id"]
    journal.complete_success(
        operation["id"],
        result={"status": "active", "source_commit": target},
    )

    reopened = _ActivationJournal(journal_path)
    persisted = reopened.state()
    assert reopened.release(persisted["active_release_id"])["source_commit"] == target  # type: ignore[index]
    assert reopened.release(persisted["previous_release_id"])["source_commit"] == initial  # type: ignore[index]


class _EnvironmentStore:
    def prepare(self, source_commit: str) -> LiveSourceRuntimeEnvironment:
        return LiveSourceRuntimeEnvironment(
            id="e" * 64,
            source_commit=source_commit,
            python_interpreter=Path(sys.executable),
            dependency_lock_hash="f" * 64,
            pyproject_sha256="1" * 64,
            install_profile="runtime-no-dev-extras-no-install-project-v1",
        )


class _Runtime:
    def __init__(self) -> None:
        self.status = "stopped"
        self.live_source: RuntimeLiveSourceSpec | None = None
        self.replacements: list[tuple[RuntimeLiveSourceSpec, RuntimeLiveSourceSpec | None]] = []
        self.events: list[Any] = []
        self.fail_next = False
        self.fail_replace_at: int | None = None
        self.stop_calls = 0

    def configure_source_recovery(self, source: Any, reconciler: Any) -> None:
        del source, reconciler

    def set_live_source(self, spec: RuntimeLiveSourceSpec) -> None:
        self.live_source = spec
        self.status = "ready"

    async def replace_live_source(
        self,
        spec: RuntimeLiveSourceSpec,
        *,
        rollback: RuntimeLiveSourceSpec | None = None,
    ) -> None:
        self.replacements.append((spec, rollback))
        if self.fail_next or len(self.replacements) == self.fail_replace_at:
            self.fail_next = False
            self.live_source = rollback
            self.status = "ready"
            raise RuntimeError("activation failed")
        self.live_source = spec
        self.status = "ready"

    async def stop(self) -> None:
        self.stop_calls += 1
        self.status = "stopped"

    async def deliver_evolution_event(self, event: Any) -> None:
        self.events.append(event)


class _RejectingReviewer:
    async def review(self, **kwargs: Any) -> ReleaseReviewDecision:
        assert Path(kwargs["candidate_root"]).is_dir()
        assert Path(kwargs["reviewer_root"]).is_dir()
        assert kwargs["review_instructions"] == "Verify VALUE in deployment."
        assert kwargs["inference_plan"].primary.model == "owner-model"
        return ReleaseReviewDecision(
            approved=False,
            summary="The deployed value is wrong.",
            findings=["Expected VALUE=2 but observed VALUE=1."],
            repair_handoff="Fix VALUE in src/opentulpa/__init__.py and rerun its test.",
        )


@pytest.mark.asyncio
async def test_reviewer_rejection_rolls_back_and_notifies_owner_handoff(tmp_path: Path) -> None:
    source, bundled = _seed(tmp_path)
    runtime = _Runtime()
    service = HostEvolutionControlService(
        runtime=runtime,  # type: ignore[arg-type]
        workspace=_TrustedSourceWorkspace(
            source_repository=source,
            path=tmp_path / "control" / "source",
            max_output_bytes=100_000,
        ),
        journal=_ActivationJournal(tmp_path / "control" / "activations.db"),
        runtime_environment_store=_EnvironmentStore(),  # type: ignore[arg-type]
        reviewer=_RejectingReviewer(),  # type: ignore[arg-type]
    )

    async def checks() -> list[Any]:
        return [{"name": "focused", "passed": True}]

    service._run_checks = checks  # type: ignore[method-assign]
    await service.prepare()
    await service.start()
    await service.source_edit(
        path="src/opentulpa/__init__.py",
        old_text="VALUE = 1",
        new_text="VALUE = 2",
    )
    inference_plan = ResolvedInferencePlan.resolve(
        InferenceSelection(provider="api", model="owner-model", reasoning_effort="xhigh"),
        preference_revision=4,
    )
    queued = await service.source_activate(
        idempotency_key="review-reject",
        review_instructions="Verify VALUE in deployment.",
        inference_plan=inference_plan,
        audit_context={"tenant_id": "owner", "thread_id": "thread-1"},
    )
    await service._tasks[str(queued["activation_id"])]

    status = await service.source_status()
    assert status["active_source_commit"] == bundled
    assert status["activation"]["status"] == "rolled_back"  # type: ignore[index]
    assert "Repair handoff:" in status["activation"]["error"]  # type: ignore[index]
    assert runtime.events[-1].event_type == "promotion.failed"
    assert "src/opentulpa/__init__.py" in runtime.events[-1].payload["error"]
    assert status["workspace_head"] != bundled
    await service.shutdown()


@pytest.mark.asyncio
async def test_source_service_activates_rolls_back_and_replays(tmp_path: Path) -> None:
    source, bundled = _seed(tmp_path)
    runtime = _Runtime()
    service = HostEvolutionControlService(
        runtime=runtime,  # type: ignore[arg-type]
        workspace=_TrustedSourceWorkspace(
            source_repository=source,
            path=tmp_path / "control" / "source",
            max_output_bytes=100_000,
        ),
        journal=_ActivationJournal(tmp_path / "control" / "activations.db"),
        runtime_environment_store=_EnvironmentStore(),  # type: ignore[arg-type]
    )

    async def checks() -> list[Any]:
        return [{"name": "focused", "passed": True}]

    service._run_checks = checks  # type: ignore[method-assign]
    await service.prepare()
    await service.start()
    await service.source_edit(
        path="README.md",
        old_text="before",
        new_text="after",
        audit_context={"tenant_id": "owner"},
    )
    queued = await service.source_activate(
        idempotency_key="activate-1",
        message="Improve source",
        audit_context={"tenant_id": "owner"},
    )
    await service._tasks[str(queued["activation_id"])]

    active = await service.source_status()
    assert active["active_source_commit"] != bundled
    assert runtime.live_source is not None
    assert runtime.live_source.source_commit == active["active_source_commit"]
    replayed = await service.source_activate(
        idempotency_key="activate-1",
        message="Improve source",
        audit_context={"tenant_id": "owner"},
    )
    assert replayed["replayed"] is True
    assert replayed["status"] == "active"

    rollback = await service.source_rollback(
        idempotency_key="rollback-1",
        expected_active_release_id=str(active["active_release_id"]),
        audit_context={"tenant_id": "owner"},
    )
    await service._tasks[str(rollback["activation_id"])]
    restored = await service.source_status()
    assert restored["active_source_commit"] == bundled
    assert restored["workspace_head"] == active["active_source_commit"]

    runtime.fail_next = True
    await service.source_edit(
        path="README.md",
        old_text="after",
        new_text="fixed",
        audit_context={"tenant_id": "owner"},
    )
    failed = await service.source_activate(
        idempotency_key="activate-2",
        message="Try another source",
        audit_context={"tenant_id": "owner"},
    )
    await service._tasks[str(failed["activation_id"])]
    failure = await service.source_status()
    assert failure["active_source_commit"] == bundled
    assert failure["activation"]["status"] == "rolled_back"  # type: ignore[index]
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_after_commit", [False, True])
async def test_source_activation_restores_runtime_when_journal_finalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_after_commit: bool,
) -> None:
    source, bundled = _seed(tmp_path)
    runtime = _Runtime()
    journal = _ActivationJournal(tmp_path / "control" / "activations.db")
    service = HostEvolutionControlService(
        runtime=runtime,  # type: ignore[arg-type]
        workspace=_TrustedSourceWorkspace(
            source_repository=source,
            path=tmp_path / "control" / "source",
            max_output_bytes=100_000,
        ),
        journal=journal,
        runtime_environment_store=_EnvironmentStore(),  # type: ignore[arg-type]
    )

    async def checks() -> list[Any]:
        return [{"name": "focused", "passed": True}]

    service._run_checks = checks  # type: ignore[method-assign]
    await service.prepare()
    await service.start()
    previous_spec = runtime.live_source
    original_complete_success = journal.complete_success

    def fail_finalization(*args: Any, **kwargs: Any) -> Any:
        if failure_after_commit:
            original_complete_success(*args, **kwargs)
        raise OSError("journal unavailable")

    await service.source_edit(
        path="README.md",
        old_text="before",
        new_text="after",
        audit_context={"tenant_id": "owner"},
    )
    monkeypatch.setattr(
        journal,
        "complete_success",
        fail_finalization,
    )

    queued = await service.source_activate(
        idempotency_key="activate-finalization-failure",
        audit_context={"tenant_id": "owner"},
    )
    await service._tasks[str(queued["activation_id"])]

    status = await service.source_status()
    assert runtime.live_source is not None
    if failure_after_commit:
        assert status["active_source_commit"] != bundled
        assert status["activation"]["status"] == "active"  # type: ignore[index]
        assert runtime.live_source.source_commit == status["active_source_commit"]
        assert len(runtime.replacements) == 1
    else:
        assert status["active_source_commit"] == bundled
        assert status["activation"]["status"] == "rolled_back"  # type: ignore[index]
        assert runtime.live_source.source_commit == bundled
        assert len(runtime.replacements) == 2
        assert runtime.replacements[1] == (previous_spec, runtime.replacements[0][0])
    await service.shutdown()


@pytest.mark.asyncio
async def test_source_activation_leaves_pending_when_rollback_cannot_be_proven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, bundled = _seed(tmp_path)
    runtime = _Runtime()
    journal = _ActivationJournal(tmp_path / "control" / "activations.db")
    service = HostEvolutionControlService(
        runtime=runtime,  # type: ignore[arg-type]
        workspace=_TrustedSourceWorkspace(
            source_repository=source,
            path=tmp_path / "control" / "source",
            max_output_bytes=100_000,
        ),
        journal=journal,
        runtime_environment_store=_EnvironmentStore(),  # type: ignore[arg-type]
    )

    async def checks() -> list[Any]:
        return [{"name": "focused", "passed": True}]

    service._run_checks = checks  # type: ignore[method-assign]
    await service.prepare()
    await service.start()

    def fail_finalization(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise OSError("journal unavailable")

    await service.source_edit(
        path="README.md",
        old_text="before",
        new_text="after",
        audit_context={"tenant_id": "owner"},
    )
    runtime.fail_replace_at = 2
    monkeypatch.setattr(journal, "complete_success", fail_finalization)

    queued = await service.source_activate(
        idempotency_key="activate-unproven-rollback",
        audit_context={"tenant_id": "owner"},
    )
    with pytest.raises(SourceEvolutionError, match="remains pending"):
        await service._tasks[str(queued["activation_id"])]

    status = await service.source_status()
    assert status["active_source_commit"] == bundled
    assert status["runtime_status"] == "stopped"
    assert status["activation"]["status"] == "preparing"  # type: ignore[index]
    assert len(runtime.replacements) == 2
    assert runtime.stop_calls == 1
    await service.shutdown()


@pytest.mark.asyncio
async def test_activation_compile_check_does_not_import_editable_source(tmp_path: Path) -> None:
    source, _ = _seed(tmp_path)
    marker = tmp_path / "candidate-imported"
    workspace = _TrustedSourceWorkspace(
        source_repository=source,
        path=tmp_path / "control" / "source",
        max_output_bytes=100_000,
    )
    workspace.prepare()
    workspace.write(
        "src/opentulpa/__init__.py",
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('unsafe')\n",
    )
    service = HostEvolutionControlService(
        runtime=_Runtime(),  # type: ignore[arg-type]
        workspace=workspace,
        journal=_ActivationJournal(tmp_path / "control" / "activations.db"),
        runtime_environment_store=_EnvironmentStore(),  # type: ignore[arg-type]
    )

    assert await service._run_checks() == [
        {
            "name": "python.compile",
            "passed": True,
            "exit_code": 0,
            "output": "",
            "output_truncated": False,
        }
    ]
    assert not marker.exists()


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
