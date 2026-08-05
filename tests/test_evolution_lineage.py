from __future__ import annotations

import multiprocessing
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from opentulpa.evolution.generation import UpstreamLineage
from opentulpa.evolution.lineage import (
    ACCEPTED_UPSTREAM_REF,
    INSTANCE_REF,
    UPSTREAM_REF,
    ConflictStage,
    GitLineage,
    GitLineageError,
)
from opentulpa.evolution.workspace import (
    CandidateWorkspace,
    GitCandidateError,
    GitCandidateWorkspace,
    RepositoryMutationLockError,
    repository_mutation_lock,
)

_REPOSITORY_TEMPLATES: dict[bool, tuple[Path, str, str, str]] = {}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(root: Path, path: str, content: str, message: str) -> str:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(root, "add", path)
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _create_repository(root: Path, *, conflict: bool) -> tuple[str, str, str]:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    base = _commit(root, "app.py", "VALUE = 'base'\n", "base")
    _git(root, "branch", "upstream", base)
    if conflict:
        instance = _commit(root, "app.py", "VALUE = 'instance'\n", "instance")
    else:
        instance = _commit(root, "instance.py", "INSTANCE = True\n", "instance")
    _git(root, "switch", "upstream")
    if conflict:
        upstream = _commit(root, "app.py", "VALUE = 'upstream'\n", "upstream")
    else:
        upstream = _commit(root, "upstream.py", "UPSTREAM = True\n", "upstream")
    _git(root, "switch", "main")
    return base, instance, upstream


@pytest.fixture(scope="module", autouse=True)
def _repository_templates(tmp_path_factory: pytest.TempPathFactory) -> None:
    global _REPOSITORY_TEMPLATES

    templates: dict[bool, tuple[Path, str, str, str]] = {}
    for conflict in (False, True):
        root = tmp_path_factory.mktemp(f"lineage-repository-template-{conflict}") / "source"
        base, instance, upstream = _create_repository(root, conflict=conflict)
        templates[conflict] = (root, base, instance, upstream)
    _REPOSITORY_TEMPLATES = templates


def _repository(tmp_path: Path, *, conflict: bool = False) -> tuple[Path, str, str, str]:
    template, base, instance, upstream = _REPOSITORY_TEMPLATES[conflict]
    root = tmp_path / "source"
    shutil.copytree(template, root)
    return root, base, instance, upstream


def _manager(root: Path, tmp_path: Path) -> GitCandidateWorkspace:
    return GitCandidateWorkspace(
        source_repository=root,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )


def _lineage(root: Path, tmp_path: Path) -> GitLineage:
    worktrees_root = tmp_path / "worktrees"
    worktrees_root.mkdir(exist_ok=True)
    return GitLineage(root, worktrees_root=worktrees_root)


def _hold_repository_lock(
    common_directory: str,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with repository_mutation_lock(common_directory):
        ready.set()
        release.wait(timeout=10)


def _acquire_repository_lock(
    common_directory: str,
    acquired: multiprocessing.synchronize.Event,
) -> None:
    with repository_mutation_lock(common_directory, timeout_seconds=5):
        acquired.set()


def test_clean_merge_has_exact_parent_order_and_keeps_canonical_checkout(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    canonical_head = _git(root, "rev-parse", "HEAD")
    canonical_branch = _git(root, "branch", "--show-current")
    canonical_app = (root / "app.py").read_text(encoding="utf-8")
    lineage = _lineage(root, tmp_path)
    initialized = lineage.initialize(instance, base)
    manager = _manager(root, tmp_path)
    workspace = manager.create(candidate_id="merge-clean", base_ref=instance)

    merge = lineage.prepare_merge(workspace, initialized.upstream_lineage)
    assert merge.conflicted_paths == ()
    assert merge.merge_base_commit == base
    assert _git(workspace.path, "rev-parse", "MERGE_HEAD") == upstream

    committed = manager.commit(workspace, message="Merge exact upstream")
    assert lineage.verify_final_merge(workspace, merge) == committed.source_commit
    assert _git(root, "show", "-s", "--format=%P", committed.source_commit).split() == [
        instance,
        upstream,
    ]
    projected = lineage.project(
        committed.source_commit,
        upstream,
        expected_instance_commit=instance,
        expected_accepted_upstream_commit=base,
    )
    assert projected.instance_commit == committed.source_commit
    assert projected.accepted_upstream_commit == upstream
    assert _git(root, "rev-parse", "HEAD") == canonical_head
    assert _git(root, "branch", "--show-current") == canonical_branch
    assert (root / "app.py").read_text(encoding="utf-8") == canonical_app


def test_conflict_persists_in_native_index_and_can_be_resolved(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path, conflict=True)
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)
    manager = _manager(root, tmp_path)
    workspace = manager.create(candidate_id="merge-conflict", base_ref=instance)

    merge = lineage.prepare_merge(workspace, snapshot.upstream_lineage)
    assert merge.conflicted_paths == ("app.py",)
    assert _git(workspace.path, "rev-parse", "MERGE_HEAD") == upstream
    stages = {
        int(line.split()[2]) for line in _git(workspace.path, "ls-files", "--unmerged").splitlines()
    }
    assert stages == {
        ConflictStage.BASE,
        ConflictStage.INSTANCE,
        ConflictStage.UPSTREAM,
    }
    assert _git(workspace.path, "show", ":1:app.py") == "VALUE = 'base'"
    assert _git(workspace.path, "show", ":2:app.py") == "VALUE = 'instance'"
    assert _git(workspace.path, "show", ":3:app.py") == "VALUE = 'upstream'"

    with pytest.raises(GitCandidateError, match="unresolved merge entries"):
        manager.commit(workspace, message="Do not stage unresolved markers")

    recovered = _lineage(root, tmp_path).merge_state(workspace, merge.upstream_lineage)
    assert recovered == merge
    assert lineage.stage_resolved_conflicts(workspace, merge.conflicted_paths) == ("app.py",)
    (workspace.path / "app.py").write_text("VALUE = 'resolved'\n", encoding="utf-8")
    assert lineage.stage_resolved_conflicts(workspace, merge.conflicted_paths) == ()
    committed = manager.commit(workspace, message="Resolve native conflict")
    assert lineage.verify_final_merge(workspace, merge) == committed.source_commit
    assert (workspace.path / "app.py").read_text(encoding="utf-8") == "VALUE = 'resolved'\n"


def test_verified_merge_remains_valid_below_a_fixup_commit(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)
    manager = _manager(root, tmp_path)
    workspace = manager.create(candidate_id="merge-fixup", base_ref=instance)
    merge = lineage.prepare_merge(workspace, snapshot.upstream_lineage)
    merged = manager.commit(workspace, message="Merge exact upstream")
    (workspace.path / "fixup.py").write_text("FIXED = True\n", encoding="utf-8")
    fixed = manager.commit(workspace, message="Fix evaluation failure")

    assert lineage.verify_merged_tip(
        fixed.source_commit,
        instance_commit=instance,
        upstream_commit=upstream,
        expected_merge_commit=merged.source_commit,
    ) == merged.source_commit
    assert lineage.verify_final_merge(workspace, merge) == fixed.source_commit


def test_wrong_merge_parent_order_is_rejected(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    lineage.initialize(instance, base)
    tree = _git(root, "rev-parse", f"{instance}^{{tree}}")
    reversed_merge = _git(
        root,
        "commit-tree",
        tree,
        "-p",
        upstream,
        "-p",
        instance,
        "-m",
        "wrong order",
    )

    with pytest.raises(GitLineageError, match="parent order"):
        lineage.verify_merge_commit(
            reversed_merge,
            instance_commit=instance,
            upstream_commit=upstream,
        )


def test_atomic_ref_projection_rejects_one_stale_expected_ref(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    lineage.initialize(instance, base)
    tree = _git(root, "rev-parse", f"{instance}^{{tree}}")
    merged = _git(
        root,
        "commit-tree",
        tree,
        "-p",
        instance,
        "-p",
        upstream,
        "-m",
        "merged",
    )
    lineage.project(
        merged,
        upstream,
        expected_instance_commit=instance,
        expected_accepted_upstream_commit=base,
    )
    before = (_git(root, "rev-parse", INSTANCE_REF), _git(root, "rev-parse", ACCEPTED_UPSTREAM_REF))

    with pytest.raises(GitLineageError, match="operation failed"):
        lineage.project(
            merged,
            upstream,
            expected_instance_commit=merged,
            expected_accepted_upstream_commit=base,
        )

    assert (_git(root, "rev-parse", INSTANCE_REF), _git(root, "rev-parse", ACCEPTED_UPSTREAM_REF)) == before


def test_stale_candidate_is_rejected_before_merge_mutation(tmp_path: Path) -> None:
    root, base, instance, _ = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)
    manager = _manager(root, tmp_path)
    workspace = manager.create(candidate_id="merge-stale", base_ref=instance)
    tree = _git(root, "rev-parse", f"{instance}^{{tree}}")
    next_instance = _git(root, "commit-tree", tree, "-p", instance, "-m", "next instance")
    lineage.project(
        next_instance,
        base,
        expected_instance_commit=instance,
        expected_accepted_upstream_commit=base,
    )

    with pytest.raises(GitLineageError, match="stale instance"):
        lineage.prepare_merge(workspace, snapshot.upstream_lineage)

    assert _git(workspace.path, "rev-parse", "--verify", "HEAD") == instance
    merge_head_path = Path(_git(workspace.path, "rev-parse", "--git-path", "MERGE_HEAD"))
    assert merge_head_path.exists() is False


def test_lineage_returns_typed_exact_snapshot(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)

    assert lineage.resolve_ref(UPSTREAM_REF) == upstream
    assert lineage.is_ancestor(base, instance) is True
    assert lineage.is_ancestor(instance, upstream) is False
    assert lineage.snapshot_upstream_lineage() == UpstreamLineage(
        upstream_commit=upstream,
        merge_base_commit=base,
    )
    assert snapshot.upstream_lineage == lineage.snapshot_upstream_lineage()


def test_upstream_sync_fetches_https_and_advances_ref_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    lineage.initialize(instance, base)
    tree = _git(root, "rev-parse", f"{upstream}^{{tree}}")
    fetched = _git(root, "commit-tree", tree, "-p", upstream, "-m", "remote update")
    calls: list[tuple[str, str]] = []

    def fetch(repository_url: str, remote_ref: str, target_ref: str) -> None:
        calls.append((repository_url, remote_ref))
        _git(root, "update-ref", target_ref, fetched)

    monkeypatch.setattr(lineage, "_fetch_https_upstream", fetch)

    synced = lineage.sync_upstream(
        "https://github.com/kvyb/opentulpa.git",
        "refs/heads/main",
    )

    assert synced.previous_commit == upstream
    assert synced.upstream_commit == fetched
    assert synced.changed is True
    assert calls == [("https://github.com/kvyb/opentulpa.git", "refs/heads/main")]
    assert _git(root, "rev-parse", UPSTREAM_REF) == fetched
    with pytest.raises(subprocess.CalledProcessError):
        _git(root, "rev-parse", "refs/opentulpa/upstream/fetched")


def test_upstream_sync_rejects_remote_rewrite_and_credential_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    lineage.initialize(instance, base)
    tree = _git(root, "rev-parse", f"{base}^{{tree}}")
    rewritten = _git(root, "commit-tree", tree, "-p", base, "-m", "rewritten remote")

    def fetch(_repository_url: str, _remote_ref: str, target_ref: str) -> None:
        _git(root, "update-ref", target_ref, rewritten)

    monkeypatch.setattr(lineage, "_fetch_https_upstream", fetch)

    with pytest.raises(GitLineageError, match="rewrote"):
        lineage.sync_upstream(
            "https://github.com/kvyb/opentulpa.git",
            "refs/heads/main",
        )
    assert _git(root, "rev-parse", UPSTREAM_REF) == upstream
    with pytest.raises(ValueError, match="unauthenticated HTTPS"):
        lineage.sync_upstream(
            "https://token@github.com/kvyb/opentulpa.git",
            "refs/heads/main",
        )


def test_lineage_ignores_replacement_refs_and_rejects_grafts(tmp_path: Path) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    tree = _git(root, "rev-parse", f"{instance}^{{tree}}")
    replacement = _git(
        root,
        "commit-tree",
        tree,
        "-p",
        upstream,
        "-m",
        "replacement",
    )
    _git(root, "replace", instance, replacement)
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)
    assert snapshot.merge_base_commit == base

    grafts = root / ".git" / "info" / "grafts"
    grafts.write_text(f"{instance} {upstream}\n", encoding="ascii")
    with pytest.raises(GitLineageError, match="configuration is unsafe"):
        _lineage(root, tmp_path)


def test_repository_mutation_lock_blocks_workspace_in_another_process(tmp_path: Path) -> None:
    root, _, instance, _ = _repository(tmp_path)
    manager = _manager(root, tmp_path)
    common = (root / _git(root, "rev-parse", "--git-common-dir")).resolve()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_repository_lock,
        args=(str(common), ready, release),
    )
    finished = threading.Event()
    errors: list[BaseException] = []

    def create_workspace() -> None:
        try:
            manager.create(candidate_id="locked-candidate", base_ref=instance)
        except BaseException as exc:
            errors.append(exc)
        else:
            finished.set()

    worker = threading.Thread(target=create_workspace)
    holder.start()
    try:
        assert ready.wait(timeout=5)
        with (
            pytest.raises(RepositoryMutationLockError, match="timed out"),
            repository_mutation_lock(common, timeout_seconds=0.02),
        ):
            pass
        worker.start()
        assert finished.wait(timeout=0.02) is False
        release.set()
        assert finished.wait(timeout=5)
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(timeout=5)
        holder.join(timeout=5)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)
    assert not errors
    assert holder.exitcode == 0


def test_repository_mutation_lock_is_not_reentrant_after_fork(tmp_path: Path) -> None:
    root, _, _, _ = _repository(tmp_path)
    common = (root / _git(root, "rev-parse", "--git-common-dir")).resolve()
    context = multiprocessing.get_context("fork")
    acquired = context.Event()

    child = context.Process(target=_acquire_repository_lock, args=(str(common), acquired))
    try:
        with repository_mutation_lock(common):
            child.start()
            assert acquired.wait(timeout=0.02) is False
        assert acquired.wait(timeout=5)
    finally:
        if child.pid is not None:
            child.join(timeout=5)
            if child.is_alive():
                child.terminate()
                child.join(timeout=5)
    assert child.exitcode == 0


def test_lineage_rejects_canonical_unregistered_and_mismatched_worktrees(
    tmp_path: Path,
) -> None:
    root, base, instance, _ = _repository(tmp_path)
    manager = _manager(root, tmp_path)
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)
    canonical = CandidateWorkspace("canonical", root, instance)
    with pytest.raises(GitLineageError, match="managed detached worktree"):
        lineage.prepare_merge(canonical, snapshot.upstream_lineage)

    rogue_path = manager.worktrees_root / "rogue"
    _git(root, "worktree", "add", "--detach", str(rogue_path), instance)
    rogue = CandidateWorkspace("rogue", rogue_path, instance)
    with pytest.raises(GitLineageError, match="managed detached worktree"):
        lineage.prepare_merge(rogue, snapshot.upstream_lineage)
    adopted = manager.adopt(rogue)
    assert lineage.prepare_merge(adopted, snapshot.upstream_lineage).upstream_lineage == (
        snapshot.upstream_lineage
    )

    managed = manager.create(candidate_id="managed", base_ref=instance)
    mismatched = CandidateWorkspace("another-id", managed.path, instance)
    with pytest.raises(GitLineageError, match="managed detached worktree"):
        lineage.prepare_merge(mismatched, snapshot.upstream_lineage)
    wrong_base = CandidateWorkspace("managed", managed.path, base)
    with pytest.raises(GitLineageError, match="managed detached worktree"):
        lineage.prepare_merge(wrong_base, snapshot.upstream_lineage)


def test_projection_verifies_observed_upstream_in_ref_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base, instance, upstream = _repository(tmp_path)
    lineage = _lineage(root, tmp_path)
    lineage.initialize(instance, base)
    tree = _git(root, "rev-parse", f"{instance}^{{tree}}")
    merged = _git(
        root,
        "commit-tree",
        tree,
        "-p",
        instance,
        "-p",
        upstream,
        "-m",
        "merged",
    )
    forced = _git(root, "commit-tree", tree, "-p", base, "-m", "forced upstream")
    before = (_git(root, "rev-parse", INSTANCE_REF), _git(root, "rev-parse", ACCEPTED_UPSTREAM_REF))
    run_git = lineage._run_git
    raced = False

    def race_upstream(cwd: Path, *arguments: str, **kwargs: object):  # type: ignore[no-untyped-def]
        nonlocal raced
        if not raced and arguments[:2] == ("update-ref", "--no-deref"):
            raced = True
            _git(root, "update-ref", UPSTREAM_REF, forced)
        return run_git(cwd, *arguments, **kwargs)

    monkeypatch.setattr(lineage, "_run_git", race_upstream)
    with pytest.raises(GitLineageError, match="operation failed"):
        lineage.project(
            merged,
            upstream,
            expected_instance_commit=instance,
            expected_accepted_upstream_commit=base,
        )
    assert raced is True
    assert (_git(root, "rev-parse", INSTANCE_REF), _git(root, "rev-parse", ACCEPTED_UPSTREAM_REF)) == before


def test_lineage_disables_hooks_and_ignores_host_merge_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, base, instance, _ = _repository(tmp_path, conflict=True)
    marker = tmp_path / "host-command-ran"
    command = tmp_path / "host-command"
    command.write_text(f"#!/bin/sh\nprintf pwned > '{marker}'\nexit 1\n", encoding="utf-8")
    command.chmod(0o755)
    hooks = root / ".git" / "host-hooks"
    hooks.mkdir()
    for name in ("pre-merge-commit", "post-merge"):
        hook = hooks / name
        hook.write_text(command.read_text(encoding="utf-8"), encoding="utf-8")
        hook.chmod(0o755)
    included = tmp_path / "included.gitconfig"
    included.write_text(
        f"[merge \"host\"]\n\tdriver = {command} %O %A %B %L %P\n"
        f"[core]\n\thooksPath = {hooks}\n",
        encoding="utf-8",
    )
    hostile_global = tmp_path / "global.gitconfig"
    hostile_global.write_text(
        f"[include]\n\tpath = {included}\n[core]\n\tfsmonitor = {command}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile_global))
    attributes = root / ".git" / "info" / "attributes"
    attributes.write_text("app.py merge=host\n", encoding="utf-8")
    lineage = _lineage(root, tmp_path)
    snapshot = lineage.initialize(instance, base)
    manager = _manager(root, tmp_path)
    workspace = manager.create(candidate_id="merge-hostile", base_ref=instance)

    merge = lineage.prepare_merge(workspace, snapshot.upstream_lineage)

    assert merge.conflicted_paths == ("app.py",)
    assert marker.exists() is False
