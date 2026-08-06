# opentulpa: allow-test-credential
from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from opentulpa.evolution.workspace import (
    CandidateWorkspace,
    GitCandidateError,
    GitCandidateWorkspace,
    candidate_content_contains_secret,
    candidate_path_is_promotable,
    candidate_path_is_runtime_overlay,
)

_REPOSITORY_TEMPLATE: Path | None = None


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _create_repository(root: Path) -> None:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    (root / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", "app.py")
    _git(root, "commit", "-m", "seed")


@pytest.fixture(scope="module", autouse=True)
def _repository_template(tmp_path_factory: pytest.TempPathFactory) -> None:
    global _REPOSITORY_TEMPLATE

    root = tmp_path_factory.mktemp("workspace-repository-template") / "source"
    _create_repository(root)
    _REPOSITORY_TEMPLATE = root


def _repository(tmp_path: Path) -> Path:
    assert _REPOSITORY_TEMPLATE is not None
    root = tmp_path / "source"
    shutil.copytree(_REPOSITORY_TEMPLATE, root)
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
    assert stat.S_IMODE(review.patch_path.stat().st_mode) == 0o600

    artifact = manager.contribution_metadata(
        candidate_id="candidate_1",
        base_commit=commit.base_commit,
        head_commit=commit.source_commit,
    )
    assert artifact.patch_path.read_text(encoding="utf-8").startswith("From ")
    assert artifact.patch_sha256
    assert stat.S_IMODE(artifact.patch_path.stat().st_mode) == 0o600
    assert not tuple((tmp_path / "artifacts").glob("*.tmp"))

    manager.remove(workspace)
    assert not workspace.path.exists()
    assert (
        _git(source, "rev-parse", "refs/opentulpa/candidates/candidate_1")
        == commit.source_commit
    )


def test_new_candidate_is_an_independent_remote_free_repository(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-full-repository")

    assert (workspace.path / ".git").is_dir()
    assert _git(workspace.path, "rev-parse", "--git-common-dir") == ".git"
    assert _git(workspace.path, "remote") == ""
    assert _git(workspace.path, "config", "--local", "user.name") == "OpenTulpa Candidate"
    assert (
        _git(workspace.path, "config", "--local", "user.email")
        == "candidate@opentulpa.local"
    )
    config = (workspace.path / ".git" / "config").read_text(encoding="utf-8")
    assert "credential." not in config
    assert "remote." not in config
    assert "core.hookspath" not in config
    source_objects = (source / ".git" / "objects").stat()
    candidate_objects = (workspace.path / ".git" / "objects").stat()
    assert (source_objects.st_dev, source_objects.st_ino) != (
        candidate_objects.st_dev,
        candidate_objects.st_ino,
    )

    assert _git(workspace.path, "status", "--porcelain") == ""
    assert _git(workspace.path, "diff") == ""
    assert _git(workspace.path, "log", "-1", "--format=%s") == "seed"
    assert _git(workspace.path, "blame", "app.py")
    _git(workspace.path, "branch", "review")
    _git(workspace.path, "merge", "--no-edit", "review")
    _git(workspace.path, "rebase", "HEAD")
    (workspace.path / "topic.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(workspace.path, "add", "topic.py")
    _git(
        workspace.path,
        "-c",
        "user.name=Candidate",
        "-c",
        "user.email=candidate@opentulpa.local",
        "commit",
        "-m",
        "topic",
    )
    topic = _git(workspace.path, "rev-parse", "HEAD")
    _git(workspace.path, "reset", "--hard", "HEAD^")
    _git(workspace.path, "cherry-pick", topic)
    _git(workspace.path, "bisect", "start")
    _git(workspace.path, "bisect", "reset")
    manager.remove(workspace)


def test_exact_import_transfers_objects_and_cleans_temporary_ref(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-exact-import")
    (workspace.path / "app.py").write_text("VALUE = 7\n", encoding="utf-8")
    committed = manager.commit(workspace, message="Exact object import")

    assert _git(source, "cat-file", "-e", committed.source_commit) == ""
    assert manager.import_exact_commit(workspace, source_commit=committed.source_commit) == (
        committed.source_commit
    )
    assert _git(source, "rev-parse", committed.source_commit) == committed.source_commit
    with pytest.raises(subprocess.CalledProcessError):
        _git(source, "rev-parse", f"refs/opentulpa/imports/{workspace.candidate_id}")


def test_full_candidate_rejects_post_creation_remote_alternate_and_hardlink(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )

    remote_workspace = manager.create(candidate_id="candidate-remote-mutation")
    _git(remote_workspace.path, "remote", "add", "origin", "https://example.com/repo.git")
    with pytest.raises(GitCandidateError, match="configuration|remotes"):
        manager.head(remote_workspace)

    alternate_workspace = manager.create(candidate_id="candidate-alternate-mutation")
    alternate = alternate_workspace.path / ".git" / "objects" / "info" / "alternates"
    alternate.write_text(str(source / ".git" / "objects") + "\n", encoding="utf-8")
    with pytest.raises(GitCandidateError, match="object store|configuration"):
        manager.head(alternate_workspace)

    hardlink_workspace = manager.create(candidate_id="candidate-hardlink-mutation")
    source_object = next(
        path
        for path in (source / ".git" / "objects").glob("[0-9a-f][0-9a-f]/*")
        if path.is_file()
    )
    candidate_object = hardlink_workspace.path / ".git" / "objects" / source_object.relative_to(
        source / ".git" / "objects"
    )
    candidate_object.unlink()
    os.link(source_object, candidate_object)
    with pytest.raises(GitCandidateError, match="object store"):
        manager.head(hardlink_workspace)


def test_full_candidate_preserves_sha256_object_format(tmp_path: Path) -> None:
    source = tmp_path / "source-sha256"
    source.mkdir()
    initialized = subprocess.run(
        ["git", "-C", str(source), "init", "--object-format=sha256", "-b", "main"],
        capture_output=True,
        check=False,
    )
    if initialized.returncode != 0:
        pytest.skip("installed Git does not support SHA-256 repositories")
    _git(source, "config", "user.name", "Test")
    _git(source, "config", "user.email", "test@example.com")
    (source / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "app.py")
    _git(source, "commit", "-m", "seed")
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )

    workspace = manager.create(candidate_id="candidate-sha256")

    assert _git(workspace.path, "rev-parse", "--show-object-format") == "sha256"
    assert manager.head(workspace) == _git(source, "rev-parse", "HEAD")


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
    assert manager.recover_commit(workspace) == recovered
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
        'CLIENT_SECRET = "exampleexampleexampleexample1234"\n',
        'API_KEY = "testtesttesttesttesttesttest"\n',
        'TOKEN = "sk-testtesttesttesttesttesttest"\n',
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


def test_candidate_immutable_diff_digest_handles_non_utf8_blob_bytes(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-byte-diff")
    (workspace.path / "app.py").write_bytes(b"VALUE = '\xff'\n")

    assert manager.diff(workspace).startswith("opentulpa-immutable-review-v1\n")
    approved_evidence = manager.full_diff(workspace).encode("ascii")

    committed = manager.commit(workspace, message="Preserve raw diff bytes")
    review = manager.review_artifact(
        candidate_id=workspace.candidate_id,
        base_commit=committed.base_commit,
        head_commit=committed.source_commit,
    )
    raw_review = review.patch_path.read_bytes()
    assert hashlib.sha256(approved_evidence).hexdigest() == committed.diff_sha256
    assert hashlib.sha256(raw_review).hexdigest() == committed.diff_sha256
    assert review.patch_sha256 == committed.diff_sha256


def test_candidate_review_evidence_ignores_worktree_attributes(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-attributes")
    (workspace.path / ".gitattributes").write_text("app.py -diff\n", encoding="utf-8")
    (workspace.path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    approved = manager.full_diff(workspace).encode("ascii")

    committed = manager.commit(workspace, message="Attribute-independent evidence")
    (source / ".gitattributes").write_text("app.py binary\n", encoding="utf-8")
    review = manager.review_artifact(
        candidate_id=workspace.candidate_id,
        base_commit=committed.base_commit,
        head_commit=committed.source_commit,
    )

    assert hashlib.sha256(approved).hexdigest() == committed.diff_sha256
    assert review.patch_sha256 == committed.diff_sha256
    assert review.patch_path.read_bytes() == approved


def test_candidate_rejects_marker_labeled_test_credential_fixture(tmp_path: Path) -> None:
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

    with pytest.raises(GitCandidateError, match="credential material"):
        manager.commit(workspace, message="Add explicit credential fixture")

    manager.remove(workspace)


def test_secret_ingress_source_is_not_mistaken_for_credential_material() -> None:
    path = Path(__file__).resolve().parents[1] / "src" / "opentulpa" / "secrets" / "ingress.py"

    assert candidate_content_contains_secret(
        "src/opentulpa/secrets/ingress.py",
        path.read_bytes(),
    ) is False


def test_candidate_rejects_repository_git_hooks(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    _git(source, "config", "core.hooksPath", ".githooks")
    marker = tmp_path / "host-hook-ran"
    hooks = source / ".githooks"
    hooks.mkdir()
    hook = hooks / "post-commit"
    hook.write_text(f"#!/bin/sh\nprintf pwned > {marker}\n", encoding="utf-8")
    hook.chmod(0o755)

    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "artifacts",
        )

    assert marker.exists() is False


def test_candidate_revalidates_changed_repository_config_without_timing(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-config-cache")
    assert manager.head(workspace) == workspace.base_commit

    _git(source, "config", "core.hooksPath", ".host-hooks")

    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        manager.head(workspace)


def test_candidate_config_cache_binds_file_contents_not_only_metadata(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-config-content-cache")
    config = source / ".git" / "config"
    original = config.read_bytes()
    metadata = config.stat()
    assert b"filemode = true" in original
    hostile = original.replace(b"filemode = true", b"filemode = nope")
    assert len(hostile) == len(original)

    config.write_bytes(hostile)
    os.utime(config, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))

    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        manager.head(workspace)


def test_candidate_ignores_host_config_includes_filters_and_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _repository(tmp_path)
    (source / ".gitattributes").write_text("*.py filter=host diff=host\n", encoding="utf-8")
    _git(source, "add", ".gitattributes")
    _git(source, "commit", "-m", "attributes")
    marker = tmp_path / "host-command-ran"
    command = tmp_path / "host-command"
    command.write_text(f"#!/bin/sh\nprintf pwned > '{marker}'\nexit 1\n", encoding="utf-8")
    command.chmod(0o755)
    included = tmp_path / "included.gitconfig"
    included.write_text(
        f"[filter \"host\"]\n\tclean = {command}\n\tsmudge = {command}\n\trequired = true\n"
        f"[diff]\n\texternal = {command}\n[credential]\n\thelper = !{command}\n",
        encoding="utf-8",
    )
    hostile_global = tmp_path / "global.gitconfig"
    hostile_global.write_text(
        f"[include]\n\tpath = {included}\n[core]\n\tfsmonitor = {command}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(hostile_global))
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_host_config")
    (workspace.path / "app.py").write_text("VALUE = 5\n", encoding="utf-8")

    assert "app.py" in manager.diff(workspace)
    manager.commit(workspace, message="Ignore host Git execution config")

    assert marker.exists() is False
    manager.remove(workspace)


def test_candidate_rejects_repository_config_includes_without_exposing_path(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path)
    private_include = tmp_path / "private-host-path.gitconfig"
    private_include.write_text("[filter \"host\"]\n\trequired = true\n", encoding="utf-8")
    _git(source, "config", "include.path", str(private_include))

    with pytest.raises(GitCandidateError, match="configuration is unsafe") as error:
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "artifacts",
        )

    assert str(private_include) not in str(error.value)


@pytest.mark.parametrize(
    ("key", "value"),
    (
        ("core.worktree", "../host-worktree"),
        ("extensions.worktreeConfig", "true"),
        ("core.fsmonitor", "true"),
        ("diff.host.textconv", "/bin/true"),
        ("filter.host.clean", "/bin/true"),
        ("merge.host.driver", "/bin/true %O %A %B"),
        ("credential.helper", "store"),
        ("branch.main.rebase", "true"),
        ("remote.origin.uploadpack", "/bin/true"),
    ),
)
def test_candidate_rejects_non_allowlisted_repository_config(
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    source = _repository(tmp_path)
    _git(source, "config", key, value)

    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "artifacts",
        )


def test_candidate_rejects_worktree_config_and_shallow_repository(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    (source / ".git" / "config.worktree").write_text(
        "[core]\n\tfsmonitor = true\n",
        encoding="utf-8",
    )
    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees-a",
            artifacts_root=tmp_path / "artifacts-a",
        )

    (source / ".git" / "config.worktree").unlink()
    (source / ".git" / "shallow").write_text(_git(source, "rev-parse", "HEAD") + "\n")
    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees-b",
            artifacts_root=tmp_path / "artifacts-b",
        )


def test_candidate_accepts_normal_clone_remote_and_branch_metadata(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    clone = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", str(source), str(clone)],
        check=True,
        capture_output=True,
    )

    manager = GitCandidateWorkspace(
        source_repository=clone,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-clone")

    assert manager.head(workspace) == _git(clone, "rev-parse", "HEAD")


@pytest.mark.parametrize(
    "url",
    (
        "https://owner:secret@example.invalid/repository.git",
        "file:///private/repository.git",
        "ext::sh -c command",
        "http://example.invalid/repository.git",
    ),
)
def test_candidate_rejects_unsafe_or_credentialed_remote_urls(
    tmp_path: Path,
    url: str,
) -> None:
    source = _repository(tmp_path)
    _git(source, "config", "remote.origin.url", url)

    with pytest.raises(GitCandidateError, match="configuration is unsafe"):
        GitCandidateWorkspace(
            source_repository=source,
            worktrees_root=tmp_path / "worktrees",
            artifacts_root=tmp_path / "artifacts",
        )

def test_candidate_rejects_invalid_base_before_running_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = CandidateWorkspace(
        candidate_id="candidate-invalid-base",
        path=tmp_path / "worktrees" / "candidate-invalid-base",
        base_commit="--upload-pack=/host/program",
    )
    called = False

    def fail_git(*args: object, **kwargs: object) -> str:
        nonlocal called
        called = True
        raise AssertionError("Git must not run")

    monkeypatch.setattr(manager, "_run_git", fail_git)
    with pytest.raises(ValueError, match="Git commit is invalid"):
        manager.head(workspace)
    assert called is False


def test_candidate_can_adopt_dirty_and_committed_legacy_worktrees(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    base = _git(source, "rev-parse", "HEAD")
    dirty_path = manager.worktrees_root / "candidate-dirty-adopted"
    _git(source, "worktree", "add", "--detach", str(dirty_path), base)
    (dirty_path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty_workspace = CandidateWorkspace("candidate-dirty-adopted", dirty_path, base)

    adopted_dirty = manager.adopt(dirty_workspace)

    assert adopted_dirty.path == dirty_path
    assert manager.status(adopted_dirty)
    assert manager.adopt(dirty_workspace) == adopted_dirty

    committed_path = manager.worktrees_root / "candidate-committed-adopted"
    _git(source, "worktree", "add", "--detach", str(committed_path), base)
    (committed_path / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    _git(committed_path, "add", "app.py")
    _git(
        committed_path,
        "-c",
        "user.name=Legacy",
        "-c",
        "user.email=legacy@example.com",
        "commit",
        "-m",
        "legacy commit",
    )
    committed_workspace = CandidateWorkspace("candidate-committed-adopted", committed_path, base)
    committed_head = _git(committed_path, "rev-parse", "HEAD")
    _git(
        source,
        "update-ref",
        "refs/opentulpa/candidates/candidate-committed-adopted",
        committed_head,
    )
    adopted_committed = manager.adopt(committed_workspace)

    assert manager.head(adopted_committed) == committed_head
    assert manager.status(adopted_committed) == ()


def test_candidate_adoption_rejects_canonical_and_unrelated_worktrees(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    base = _git(source, "rev-parse", "HEAD")
    with pytest.raises(GitCandidateError, match="metadata is unsafe"):
        manager.adopt(CandidateWorkspace("source", source, base))

    unrelated = tmp_path / "unrelated"
    _git(source, "worktree", "add", "--detach", str(unrelated), base)
    with pytest.raises(GitCandidateError, match="metadata is unsafe"):
        manager.adopt(CandidateWorkspace("unrelated", unrelated, base))


def test_candidate_commit_rejects_secret_in_adopted_descendant_history(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    base = _git(source, "rev-parse", "HEAD")
    legacy_path = manager.worktrees_root / "candidate-adopted-secret"
    _git(source, "worktree", "add", "--detach", str(legacy_path), base)
    (legacy_path / "settings.py").write_text(
        'CLIENT_SECRET = "exampleexampleexampleexample1234"\n',
        encoding="utf-8",
    )
    _git(legacy_path, "add", "settings.py")
    _git(
        legacy_path,
        "-c",
        "user.name=Legacy",
        "-c",
        "user.email=legacy@example.com",
        "commit",
        "-m",
        "legacy secret",
    )
    workspace = manager.adopt(
        CandidateWorkspace("candidate-adopted-secret", legacy_path, base)
    )
    (legacy_path / "settings.py").write_text("SETTING = 'sanitized'\n", encoding="utf-8")

    previous_head = manager.head(workspace)
    with pytest.raises(GitCandidateError, match="credential material"):
        manager.commit(workspace, message="Benign follow-up must not hide old secret")
    assert manager.head(workspace) == previous_head


def test_candidate_id_and_ref_replacement_are_rejected(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-reused")
    (workspace.path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    committed = manager.commit(workspace, message="Retain candidate")
    manager.remove(workspace)
    with pytest.raises(GitCandidateError, match="already been retained"):
        manager.create(candidate_id="candidate-reused")

    workspace = manager.create(candidate_id="candidate-replaced")
    _git(
        source,
        "update-ref",
        "refs/opentulpa/candidates/candidate-replaced",
        committed.base_commit,
    )
    (workspace.path / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(GitCandidateError, match="already retains another commit"):
        manager.commit(workspace, message="Do not replace retained ref")


def test_same_candidate_session_can_advance_retained_ref_with_exact_cas(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-repeat")
    (workspace.path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = manager.commit(workspace, message="First candidate commit")
    (workspace.path / "app.py").write_text("VALUE = 3\n", encoding="utf-8")

    second = manager.commit(workspace, message="Second candidate commit")

    assert second.source_commit != first.source_commit
    assert second.changed_paths == ("app.py",)
    assert _git(source, "rev-parse", "refs/opentulpa/candidates/candidate-repeat") == (
        second.source_commit
    )
    review = manager.review_artifact(
        candidate_id=workspace.candidate_id,
        base_commit=workspace.base_commit,
        head_commit=second.source_commit,
    )
    assert review.patch_sha256 == second.diff_sha256


def test_contribution_ref_is_cas_retained_and_idempotent(tmp_path: Path) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    first_workspace = manager.create(candidate_id="candidate-contribution-a")
    (first_workspace.path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = manager.commit(first_workspace, message="First contribution")
    first_artifact = manager.contribution_metadata(
        candidate_id="shared-contribution",
        base_commit=first.base_commit,
        head_commit=first.source_commit,
    )
    assert manager.contribution_metadata(
        candidate_id="shared-contribution",
        base_commit=first.base_commit,
        head_commit=first.source_commit,
    ) == first_artifact

    second_workspace = manager.create(candidate_id="candidate-contribution-b")
    (second_workspace.path / "app.py").write_text("VALUE = 3\n", encoding="utf-8")
    second = manager.commit(second_workspace, message="Second contribution")
    with pytest.raises(GitCandidateError, match="already retains another commit"):
        manager.contribution_metadata(
            candidate_id="shared-contribution",
            base_commit=second.base_commit,
            head_commit=second.source_commit,
        )


def test_candidate_detects_staged_index_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-index-race")
    (workspace.path / "app.py").write_text("VALUE = 2\n", encoding="utf-8")
    validate = manager._validate_index_content

    def race_index(root: Path, paths: tuple[str, ...]) -> None:
        validate(root, paths)
        (root / "app.py").write_text("VALUE = 99\n", encoding="utf-8")
        _git(root, "add", "app.py")

    monkeypatch.setattr(manager, "_validate_index_content", race_index)
    with pytest.raises(GitCandidateError, match="index changed"):
        manager.commit(workspace, message="Detect staged race")


def test_candidate_rejects_non_utf8_index_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate-non-utf8")
    repository = os.fsencode(workspace.path)
    blob = subprocess.run(
        [b"git", b"-C", repository, b"hash-object", b"-w", b"--stdin"],
        input=b"VALUE = 2\n",
        check=True,
        capture_output=True,
    ).stdout.strip()
    subprocess.run(
        [
            b"git",
            b"-C",
            repository,
            b"update-index",
            b"--add",
            b"--cacheinfo",
            b"100644," + blob + b",invalid-\xff.py",
        ],
        check=True,
        capture_output=True,
    )
    run_git = manager._run_git

    def preserve_index(root: Path, *arguments: str, **kwargs: object):  # type: ignore[no-untyped-def]
        if arguments == ("add", "--all"):
            return ""
        return run_git(root, *arguments, **kwargs)

    monkeypatch.setattr(manager, "_run_git", preserve_index)

    with pytest.raises(GitCandidateError, match="non-UTF-8 path"):
        manager.commit(workspace, message="Reject invalid path")


def test_candidate_can_commit_full_source_changes_for_release(tmp_path: Path) -> None:
    protected_paths = (
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
    )
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_kernel")
    for protected_path in protected_paths:
        path = workspace.path / protected_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# candidate change\n", encoding="utf-8")

    commit = manager.commit(workspace, message="Propose kernel change")

    assert commit.changed_paths == tuple(sorted(protected_paths))
    assert commit.promotion_eligible is True
    manager.remove(workspace)


def test_candidate_classifies_and_commits_explicit_extension_surfaces(tmp_path: Path) -> None:
    extension_surfaces = (
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
    )
    source = _repository(tmp_path)
    manager = GitCandidateWorkspace(
        source_repository=source,
        worktrees_root=tmp_path / "worktrees",
        artifacts_root=tmp_path / "artifacts",
    )
    workspace = manager.create(candidate_id="candidate_extension")
    for extension_path, promotable, runtime_overlay in extension_surfaces:
        path = workspace.path / extension_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# generated extension\n", encoding="utf-8")
        assert candidate_path_is_promotable(extension_path) is promotable
        assert candidate_path_is_runtime_overlay(extension_path) is runtime_overlay

    commit = manager.commit(workspace, message="Add extension")

    assert commit.source_commit != commit.base_commit
    assert commit.changed_paths == tuple(sorted(path for path, _, _ in extension_surfaces))
    assert commit.promotion_eligible is False
    manager.remove(workspace)
