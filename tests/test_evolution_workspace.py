from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from opentulpa.evolution.workspace import (
    GitCandidateError,
    GitCandidateWorkspace,
    candidate_path_is_promotable,
    candidate_path_is_runtime_overlay,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "seed")
    return root


def test_candidate_commit_is_isolated_and_retained_by_private_ref(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    original = (source / "app.py").read_text(encoding="utf-8")
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_1")
    (workspace.path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")

    commit = manager.commit(workspace, message="Improve value")

    assert commit.base_commit == workspace.base_commit
    assert commit.source_commit != commit.base_commit
    assert len(commit.diff_sha256) == 64
    assert commit.changed_paths == ("app.py",)
    assert commit.promotion_eligible is True
    assert (source / "app.py").read_text(encoding="utf-8") == original
    assert (
        _git(source, "rev-parse", "refs/opentulpa/candidates/candidate_1")
        == commit.source_commit
    )

    review = manager.review_artifact(
        candidate_id="candidate_1",
        base_commit=commit.base_commit,
        head_commit=commit.source_commit,
    )
    assert review.patch_sha256 == commit.diff_sha256
    assert review.patch_path.is_file()

    artifact = manager.contribution_metadata(
        candidate_id="candidate_1",
        base_commit=commit.base_commit,
        head_commit=commit.source_commit,
    )
    assert artifact.patch_path.read_text(encoding="utf-8").startswith("From ")
    assert artifact.patch_sha256

    manager.remove(workspace)
    assert not workspace.path.exists()
    assert (
        _git(source, "rev-parse", "refs/opentulpa/candidates/candidate_1")
        == commit.source_commit
    )


def test_candidate_recovers_clean_commit_after_persistence_crash(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_recovery")
    (workspace.path / "app.py").write_text("VALUE = 9\n", encoding="utf-8")
    _git(workspace.path, "add", "app.py")
    _git(
        workspace.path,
        "-c",
        "user.name=Recovery Test",
        "-c",
        "user.email=recovery@example.com",
        "commit",
        "-m",
        "committed before crash",
    )

    recovered = manager.recover_commit(workspace)

    assert recovered.base_commit == workspace.base_commit
    assert recovered.source_commit == manager.head(workspace)
    assert recovered.changed_paths == ("app.py",)
    assert recovered.promotion_eligible is True
    assert (
        _git(source, "rev-parse", "refs/opentulpa/candidates/candidate_recovery")
        == recovered.source_commit
    )
    manager.remove(workspace)


def test_candidate_recovery_rejects_dirty_worktree(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_dirty_recovery")
    (workspace.path / "app.py").write_text("VALUE = 9\n", encoding="utf-8")

    with pytest.raises(GitCandidateError, match="not clean"):
        manager.recover_commit(workspace)

    manager.remove(workspace)


def test_candidate_rejects_unsafe_ids_and_sensitive_files(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )

    with pytest.raises(ValueError, match="candidate_id"):
        manager.create(candidate_id="../escape")

    workspace = manager.create(candidate_id="candidate_safe")
    (workspace.path / ".env").write_text("TOKEN=private\n", encoding="utf-8")

    with pytest.raises(GitCandidateError, match="sensitive"):
        manager.commit(workspace, message="Unsafe")

    assert (source / ".env").exists() is False
    manager.remove(workspace)


@pytest.mark.parametrize(
    "sensitive_path",
    (".env.production", ".npmrc", "private.p12", "nested/credentials.json"),
)
def test_candidate_classification_rejects_secret_paths(sensitive_path: str) -> None:
    assert candidate_path_is_promotable(sensitive_path) is False
    assert candidate_path_is_runtime_overlay(sensitive_path) is False


def test_candidate_diff_includes_untracked_files(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_diff")
    (workspace.path / "new.py").write_text("NEW = True\n", encoding="utf-8")

    assert "new.py" in manager.diff(workspace)

    manager.remove(workspace)


def test_candidate_can_commit_public_environment_template(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_env_template")
    (workspace.path / ".env.example").write_text("PUBLIC_SETTING=\n", encoding="utf-8")

    commit = manager.commit(workspace, message="Document public setting")

    assert commit.source_commit != commit.base_commit
    assert commit.promotion_eligible is True
    manager.remove(workspace)


@pytest.mark.parametrize(
    "content",
    (
        'BOT_TOKEN = "123456789:K7mQ2vN9xR4pL8cD6sW3uY5tH1jF0aBz"\n',
        "Authorization: Bearer K7mQ2vN9xR4pL8cD6sW3uY5tH1jF0aBz\n",
        'CLIENT_SECRET = "K7mQ2vN9xR4pL8cD6sW3uY5tH1jF0aBz"\n',
        "-----BEGIN PRIVATE KEY-----\nK7mQ2vN9xR4pL8cD6sW3uY5tH1jF0aBz\n"
        "-----END PRIVATE KEY-----\n",
    ),
)
def test_candidate_rejects_real_credential_content_before_commit(
    tmp_path: Path,
    content: str,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_credential")
    previous_head = manager.head(workspace)
    (workspace.path / "settings.py").write_text(content, encoding="utf-8")

    with pytest.raises(GitCandidateError, match="credential material"):
        manager.commit(workspace, message="Do not commit the token")

    assert manager.head(workspace) == previous_head
    manager.remove(workspace)


def test_candidate_allows_explicit_test_credential_fixture(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_fixture")
    fixture = workspace.path / "tests" / "fixtures" / "provider_token.txt"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        "# opentulpa: allow-test-credential\n"
        "123456789:K7mQ2vN9xR4pL8cD6sW3uY5tH1jF0aBz\n",
        encoding="utf-8",
    )

    commit = manager.commit(workspace, message="Add explicit credential fixture")

    assert commit.promotion_eligible is True
    manager.remove(workspace)


def test_candidate_commit_disables_repository_git_hooks(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    _git(source, "config", "core.hooksPath", ".githooks")
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_hooks")
    marker = tmp_path / "host-hook-ran"
    hooks = workspace.path / ".githooks"
    hooks.mkdir()
    hook = hooks / "post-commit"
    hook.write_text(f"#!/bin/sh\nprintf pwned > {marker}\n", encoding="utf-8")
    hook.chmod(0o755)
    (workspace.path / "app.py").write_text("VALUE = 3\n", encoding="utf-8")

    manager.commit(workspace, message="Do not run hooks")

    assert marker.exists() is False
    manager.remove(workspace)


@pytest.mark.parametrize(
    "protected_path",
    (
        "src/opentulpa/bootstrap/gateway.py",
        "src/opentulpa/api/app.py",
        "src/opentulpa/deep_agent/service.py",
        "src/opentulpa/application/product_tools.py",
        "src/opentulpa/capabilities/service.py",
        "src/opentulpa/integrations/content_fetch.py",
        "src/opentulpa/secrets/vault.py",
        "src/opentulpa/specs/dispatcher.py",
        "src/opentulpa/tooling/contract.py",
        "pyproject.toml",
        "start.sh",
    ),
)
def test_candidate_can_commit_full_source_change_for_release(
    tmp_path: Path,
    protected_path: str,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_kernel")
    path = workspace.path / protected_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# candidate change\n", encoding="utf-8")

    commit = manager.commit(workspace, message="Propose kernel change")

    assert commit.changed_paths == (protected_path,)
    assert commit.promotion_eligible is True
    manager.remove(workspace)


@pytest.mark.parametrize(
    ("extension_path", "promotable", "runtime_overlay"),
    (
        ("src/opentulpa/capabilities/bundled.py", True, True),
        ("src/opentulpa/capability_workers/generated.py", True, True),
        ("src/opentulpa/capability_workers/__init__.py", True, True),
        ("src/opentulpa/capability_workers/__main__.py", True, True),
        ("src/opentulpa/capability_workers/sitecustomize.py", True, True),
        ("src/opentulpa/capability_workers/.venv/evil.py", False, False),
        ("src/opentulpa/capability_manifests/telegram.json", True, True),
        ("src/opentulpa/capability_manifests/nested/unsafe.json", True, True),
        ("src/opentulpa/client/tui.py", True, True),
        ("src/opentulpa/client/api.py", True, True),
        ("src/opentulpa/client/unsafe.py", True, True),
        ("src/opentulpa/deep_agent/prompts.py", True, True),
        ("docs/evolution.md", True, True),
        ("tests/test_generated.py", True, True),
    ),
)
def test_candidate_classifies_explicit_extension_surfaces(
    tmp_path: Path,
    extension_path: str,
    promotable: bool,
    runtime_overlay: bool,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_extension")
    path = workspace.path / extension_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# generated extension\n", encoding="utf-8")

    commit = manager.commit(workspace, message="Add extension")

    assert commit.source_commit != commit.base_commit
    assert commit.promotion_eligible is promotable
    assert candidate_path_is_promotable(extension_path) is promotable
    assert candidate_path_is_runtime_overlay(extension_path) is runtime_overlay
    manager.remove(workspace)
