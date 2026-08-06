"""Git-native candidate worktrees that never mutate the running checkout."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal, cast

from opentulpa.evolution.git_security import (
    GitSecurityError,
    RepositoryMutationLockError,
    repository_mutation_lock,
)
from opentulpa.evolution.git_security import (
    candidate_worktree_directories as _security_candidate_worktree_directories,
)
from opentulpa.evolution.git_security import (
    discover_git_directories as _security_discover_git_directories,
)
from opentulpa.evolution.git_security import (
    read_git_admin_file as _security_read_git_admin_file,
)
from opentulpa.evolution.git_security import (
    repository_git_configuration_is_unsafe as _repository_git_configuration_is_unsafe,
)
from opentulpa.evolution.git_security import (
    run_trusted_git_process as _run_trusted_git_process,
)

_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
_IMMUTABLE_REVIEW_HEADER = b"opentulpa-immutable-review-v1\n"
_SENSITIVE_NAMES = frozenset(
    {
        ".env",
        ".npmrc",
        ".pypirc",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_PUBLIC_ENV_TEMPLATES = frozenset({".env.example", ".env.sample", ".env.template"})
_RESERVED_SOURCE_COMPONENTS = frozenset({".git", ".venv"})
_PRIVATE_KEY_BEGIN_RE = re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_PRIVATE_KEY_END_RE = re.compile(rb"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_FULL_REPOSITORY_MARKER = "opentulpa-full-repository-v1"
_BEARER_RE = re.compile(rb"(?i)\bbearer[ \t]+([A-Za-z0-9._~+/=-]{20,})")
_PROVIDER_TOKEN_RES = (
    re.compile(rb"\bsk-(?:proj-|live-|lf-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\b[0-9]{6,12}:[A-Za-z0-9_-]{30,}\b"),
)
_ASSIGNED_CREDENTIAL_RE = re.compile(
    rb"\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|"
    rb"bot[_-]?token|client[_-]?secret|password|secret)\b[ \t]*[\"']?[ \t]*"
    rb"[:=][ \t]*(?:\"([A-Za-z0-9_./+=:@-]{24,})\"|"
    rb"'([A-Za-z0-9_./+=:@-]{24,})'|([A-Za-z0-9_./+=:@-]{24,})[ \t]*$)",
    re.I | re.M,
)
class GitCandidateError(RuntimeError):
    """A candidate repository operation failed without exposing command output."""


def _discover_git_directories(repository: Path) -> tuple[Path, Path]:
    try:
        return _security_discover_git_directories(repository)
    except GitSecurityError:
        raise GitCandidateError("Git repository metadata is unsafe") from None


def _candidate_worktree_directories(
    path: Path,
    *,
    candidate_id: str,
    base_commit: str | None,
    worktrees_root: Path,
    common_directory: Path,
    require_registration: bool = True,
) -> tuple[Path, Path]:
    try:
        return _security_candidate_worktree_directories(
            path,
            candidate_id=candidate_id,
            base_commit=base_commit,
            worktrees_root=worktrees_root,
            common_directory=common_directory,
            require_registration=require_registration,
        )
    except GitSecurityError:
        raise GitCandidateError("candidate worktree metadata is unsafe") from None


def _read_git_admin_file(path: Path) -> str:
    try:
        return _security_read_git_admin_file(path)
    except GitSecurityError:
        raise GitCandidateError("Git repository metadata is unsafe") from None


@dataclass(frozen=True, slots=True)
class CandidateWorkspace:
    candidate_id: str
    path: Path
    base_commit: str
    repository_kind: Literal["full_repository", "linked_worktree"] = "linked_worktree"


def candidate_repository_directory(
    path: Path,
    *,
    candidate_id: str,
    base_commit: str,
    worktrees_root: Path,
) -> Path:
    """Validate a newly-created, independent candidate repository."""
    try:
        root = path.expanduser().resolve(strict=True)
        expected = (worktrees_root / candidate_id).resolve(strict=False)
        if root != expected or not root.is_dir():
            raise GitCandidateError("candidate repository identity is invalid")
        git_directory, common_directory = _discover_git_directories(root)
        if git_directory != common_directory:
            raise GitCandidateError("candidate repository is not independent")
        marker = git_directory / "opentulpa-candidate"
        if _read_git_admin_file(marker) != f"{_FULL_REPOSITORY_MARKER} {candidate_id} {base_commit}":
            raise GitCandidateError("candidate repository discriminator is invalid")
        _require_independent_object_store(git_directory / "objects")
        return root
    except GitCandidateError:
        raise
    except OSError:
        raise GitCandidateError("candidate repository is unavailable") from None


def _require_independent_object_store(objects: Path) -> None:
    try:
        if objects.is_symlink() or not objects.is_dir():
            raise OSError("invalid object store")
        for directory, directories, files in os.walk(objects, followlinks=False):
            root = Path(directory)
            if any((root / name).is_symlink() for name in directories):
                raise OSError("linked object directory")
            for name in files:
                path = root / name
                metadata = path.lstat()
                if not path.is_file() or path.is_symlink() or metadata.st_nlink != 1:
                    raise OSError("shared object file")
    except OSError:
        raise GitCandidateError("candidate repository object store is not independent") from None


@dataclass(frozen=True, slots=True)
class CandidateCommit:
    candidate_id: str
    base_commit: str
    source_commit: str
    diff_sha256: str
    changed_paths: tuple[str, ...]
    promotion_eligible: bool


@dataclass(frozen=True, slots=True)
class ContributionArtifact:
    candidate_id: str
    base_commit: str
    head_commit: str
    branch_name: str
    patch_path: Path
    patch_sha256: str
    tree_oid: str


@dataclass(frozen=True, slots=True)
class ReviewArtifact:
    candidate_id: str
    base_commit: str
    head_commit: str
    patch_path: Path
    patch_sha256: str


class GitCandidateWorkspace:
    """Create detached candidates and retain their commits under private Git refs."""

    def __init__(
        self,
        *,
        source_repository: str | Path,
        worktrees_root: str | Path,
        artifacts_root: str | Path,
        timeout_seconds: int = 120,
        max_git_output_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        if timeout_seconds < 1 or timeout_seconds > 3_600:
            raise ValueError("Git timeout must be between 1 and 3600 seconds")
        if max_git_output_bytes < 1_024:
            raise ValueError("Git output limit is too small")
        self._source_repository = self._regular_directory(source_repository, create=False)
        self._worktrees_root = self._regular_directory(worktrees_root, create=True)
        self._artifacts_root = self._regular_directory(artifacts_root, create=True)
        if self._is_relative_to(self._source_repository, self._worktrees_root):
            raise ValueError("source repository cannot be inside the worktree root")
        self._timeout_seconds = timeout_seconds
        self._max_git_output_bytes = max_git_output_bytes
        self._lock = threading.RLock()
        self._operation_state = threading.local()
        self._git_directory, self._git_common_directory = _discover_git_directories(
            self._source_repository
        )
        with self._mutation():
            object_format = self._run_git(
                self._source_repository,
                "rev-parse",
                "--show-object-format",
            ).strip()
        if object_format not in {"sha1", "sha256"}:
            raise GitCandidateError("repository object format is unsupported")
        self._oid_length = 40 if object_format == "sha1" else 64

    @property
    def source_repository(self) -> Path:
        return self._source_repository

    @property
    def worktrees_root(self) -> Path:
        return self._worktrees_root

    def create(self, *, candidate_id: str, base_ref: str = "HEAD") -> CandidateWorkspace:
        safe_id = self._candidate_id(candidate_id)
        safe_ref = self._git_ref(base_ref)
        with self._mutation():
            candidate_ref = f"refs/opentulpa/candidates/{safe_id}"
            if self._private_ref(candidate_ref) is not None:
                raise GitCandidateError("candidate_id has already been retained")
            base_commit = self._resolve_commit(safe_ref)
            path = self._candidate_path(safe_id)
            if os.path.lexists(path):
                raise GitCandidateError("candidate worktree already exists")
            self._create_full_repository(path, base_commit, safe_id)
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                raise GitCandidateError("candidate worktree registration failed") from None
            if not self._is_relative_to(resolved, self._worktrees_root):
                self._remove_path(safe_id, resolved)
                raise GitCandidateError("candidate worktree escaped its configured root")
            candidate_repository_directory(
                resolved,
                candidate_id=safe_id,
                base_commit=base_commit,
                worktrees_root=self._worktrees_root,
            )
            workspace = CandidateWorkspace(
                candidate_id=safe_id,
                path=resolved,
                base_commit=base_commit,
                repository_kind="full_repository",
            )
            self._validate_workspace(workspace)
            return workspace

    def adopt(self, workspace: CandidateWorkspace) -> CandidateWorkspace:
        """Register one exact pre-registration candidate, including dirty sessions."""

        safe_id = self._candidate_id(workspace.candidate_id)
        safe_base = self._commit(workspace.base_commit)
        with self._mutation():
            root, git_directory = _candidate_worktree_directories(
                workspace.path,
                candidate_id=safe_id,
                base_commit=safe_base,
                worktrees_root=self._worktrees_root,
                common_directory=self._git_common_directory,
                require_registration=False,
            )
            self._assert_safe_git_configuration(root)
            symbolic = self._run_git(
                root,
                "symbolic-ref",
                "--quiet",
                "HEAD",
                allowed_returncodes=frozenset({0, 1}),
            ).strip()
            if symbolic:
                raise GitCandidateError("candidate worktree must remain detached")
            resolved_base = self._run_git(
                root,
                "rev-parse",
                "--verify",
                f"{safe_base}^{{commit}}",
            ).strip()
            if resolved_base != safe_base:
                raise GitCandidateError("candidate adoption base is invalid")
            head = self._head(root)
            try:
                self._run_git(root, "merge-base", "--is-ancestor", safe_base, head)
            except GitCandidateError:
                raise GitCandidateError("candidate adoption lineage is invalid") from None
            retained = self._private_ref(f"refs/opentulpa/candidates/{safe_id}")
            if retained is not None and retained != head:
                raise GitCandidateError("candidate adoption conflicts with its retained ref")
            if retained is None and head != safe_base:
                self._retain_ref(f"refs/opentulpa/candidates/{safe_id}", head)
            registration = git_directory / "opentulpa-candidate"
            if os.path.lexists(registration):
                if _read_git_admin_file(registration) != f"{safe_id} {safe_base}":
                    raise GitCandidateError("candidate worktree registration is invalid")
            else:
                self._register_candidate_worktree(git_directory, safe_id, safe_base)
            return CandidateWorkspace(safe_id, root, safe_base, "linked_worktree")

    def head(self, workspace: CandidateWorkspace) -> str:
        with self._mutation():
            root = self._validate_workspace(workspace)
            return self._head(root)

    def status(self, workspace: CandidateWorkspace) -> tuple[str, ...]:
        with self._mutation():
            root = self._validate_workspace(workspace)
            raw = self._run_git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z")
            return tuple(item for item in raw.split("\0") if item)

    def diff(self, workspace: CandidateWorkspace) -> str:
        with self._mutation():
            root = self._validate_workspace(workspace)
            self._reject_unmerged_index(root)
            self._run_git(root, "add", "--all")
            tree = self._write_tree(root)
            evidence, paths = self._immutable_review(root, self._head(root), tree)
            self._validate_changed_paths(paths)
            return evidence.decode("ascii")

    def full_diff(self, workspace: CandidateWorkspace) -> str:
        """Return one canonical base-to-working-tree diff across intermediate commits."""

        with self._mutation():
            root = self._validate_workspace(workspace)
            self._reject_unmerged_index(root)
            self._run_git(root, "add", "--all")
            tree = self._write_tree(root)
            evidence, paths = self._immutable_review(root, workspace.base_commit, tree)
            self._validate_changed_paths(paths)
            return evidence.decode("ascii")

    def commit(
        self,
        workspace: CandidateWorkspace,
        *,
        message: str,
    ) -> CandidateCommit:
        safe_message = " ".join(str(message or "").split())[:500]
        if not safe_message:
            raise ValueError("candidate commit message is required")
        with self._mutation():
            root = self._validate_workspace(workspace)
            base_commit = self._commit(workspace.base_commit)
            previous_head = self._head(root)
            self._validate_commit_lineage(root, base_commit, previous_head)
            self._reject_unmerged_index(root)
            self._run_git(root, "add", "--all")
            self._reject_unmerged_index(root)
            changed_paths = self._staged_changed_paths(root)
            if not changed_paths:
                raise GitCandidateError("candidate did not change source files")
            self._validate_changed_paths(changed_paths)
            expected_tree = self._write_tree(root)
            self._validate_index_content(root, changed_paths)
            expected_evidence, final_paths = self._immutable_review(
                root,
                base_commit,
                expected_tree,
            )
            if not final_paths:
                raise GitCandidateError("candidate did not change source files from its base")
            self._validate_changed_paths(final_paths)
            self._validate_tree_content(root, expected_tree, final_paths)
            self._require_unchanged_index(root, changed_paths, expected_tree)
            env = {
                "GIT_AUTHOR_NAME": "OpenTulpa Candidate",
                "GIT_AUTHOR_EMAIL": "candidate@opentulpa.local",
                "GIT_COMMITTER_NAME": "OpenTulpa Supervisor",
                "GIT_COMMITTER_EMAIL": "supervisor@opentulpa.local",
            }
            self._run_git(
                root,
                "commit",
                "--no-gpg-sign",
                "--no-verify",
                "-m",
                safe_message,
                env=env,
            )
            source_commit = self._head(root)
            committed_tree = self._run_git(
                root,
                "show",
                "--no-patch",
                "--format=%T",
                source_commit,
            ).strip()
            if committed_tree != expected_tree:
                raise GitCandidateError("candidate commit tree changed after validation")
            self._validate_commit_lineage(root, base_commit, source_commit)
            evidence, committed_paths = self._immutable_review(root, base_commit, source_commit)
            if evidence != expected_evidence or committed_paths != final_paths:
                raise GitCandidateError("candidate immutable review evidence changed")
            diff_sha256 = hashlib.sha256(evidence).hexdigest()
            self._import_git_objects(root, source_commit, base_commit)
            self._advance_candidate_ref(
                workspace,
                source_commit,
                previous_head=previous_head,
            )
            return CandidateCommit(
                candidate_id=workspace.candidate_id,
                base_commit=base_commit,
                source_commit=source_commit,
                diff_sha256=diff_sha256,
                changed_paths=final_paths,
                promotion_eligible=all(
                    candidate_path_is_promotable(path) for path in final_paths
                ),
            )

    def recover_commit(self, workspace: CandidateWorkspace) -> CandidateCommit:
        """Rebind a clean descendant commit after a crash before archive persistence."""

        with self._mutation():
            root = self._validate_workspace(workspace)
            if self.status(workspace):
                raise GitCandidateError("candidate worktree is not clean")
            source_commit = self.head(workspace)
            base_commit = self._commit(workspace.base_commit)
            if source_commit == base_commit:
                raise GitCandidateError("candidate has no committed source changes")
            self._validate_commit_lineage(root, base_commit, source_commit)
            evidence, changed_paths = self._immutable_review(root, base_commit, source_commit)
            if not changed_paths:
                raise GitCandidateError("candidate has no committed source changes")
            self._validate_changed_paths(changed_paths)
            self._validate_tree_content(root, source_commit, changed_paths)
            diff_sha256 = hashlib.sha256(evidence).hexdigest()
            self._import_git_objects(root, source_commit, base_commit)
            self._advance_candidate_ref(
                workspace,
                source_commit,
                previous_head=None,
            )
            return CandidateCommit(
                candidate_id=workspace.candidate_id,
                base_commit=base_commit,
                source_commit=source_commit,
                diff_sha256=diff_sha256,
                changed_paths=changed_paths,
                promotion_eligible=all(
                    candidate_path_is_promotable(path) for path in changed_paths
                ),
            )

    def contribution_metadata(
        self,
        *,
        candidate_id: str,
        base_commit: str,
        head_commit: str,
    ) -> ContributionArtifact:
        safe_id = self._candidate_id(candidate_id)
        safe_base = self._commit(base_commit)
        safe_head = self._commit(head_commit)
        with self._mutation():
            self._run_git(
                self._source_repository,
                "merge-base",
                "--is-ancestor",
                safe_base,
                safe_head,
            )
            self._validate_commit_lineage(self._source_repository, safe_base, safe_head)
            self._validate_changed_paths(
                self._changed_paths(self._source_repository, safe_base, safe_head)
            )
            patch = self._run_git_bytes(
                self._source_repository,
                "format-patch",
                "--stdout",
                "--no-signature",
                f"{safe_base}..{safe_head}",
            )
            if not patch.strip():
                raise GitCandidateError("candidate contribution patch is empty")
            digest = hashlib.sha256(patch).hexdigest()
            tree_oid = self._run_git(
                self._source_repository,
                "show",
                "--no-patch",
                "--format=%T",
                safe_head,
            ).strip()
            patch_path = self._artifacts_root / f"{safe_id}-{digest[:16]}.patch"
            self._publish_artifact(patch_path, patch)
            branch_name = f"opentulpa/candidate-{safe_id}"
            self._retain_ref(f"refs/opentulpa/contributions/{safe_id}", safe_head)
            return ContributionArtifact(
                candidate_id=safe_id,
                base_commit=safe_base,
                head_commit=safe_head,
                branch_name=branch_name,
                patch_path=patch_path,
                patch_sha256=digest,
                tree_oid=tree_oid,
            )

    def import_exact_commit(
        self,
        workspace: CandidateWorkspace,
        *,
        source_commit: str,
    ) -> str:
        """Import the candidate's verified Git objects into the canonical ODB.

        This transfers Git objects through pack/index-pack, never by applying a
        patch. A temporary canonical ref protects the imported graph while all
        ancestry, tree, mode, and blob checks are repeated in the destination.
        """
        with self._mutation():
            root = self._validate_workspace(workspace)
            exact = self._commit(source_commit)
            base = self._commit(workspace.base_commit)
            self._validate_commit_lineage(root, base, exact)
            self._import_git_objects(root, exact, base)
            temporary_ref = f"refs/opentulpa/imports/{workspace.candidate_id}"
            self._retain_ref(temporary_ref, exact)
            try:
                self._validate_commit_lineage(self._source_repository, base, exact)
                if self._head(root) != exact:
                    raise GitCandidateError("candidate exact import source changed")
                imported_tree = self._run_git(
                    self._source_repository,
                    "show",
                    "--no-patch",
                    "--format=%T",
                    exact,
                ).strip()
                candidate_tree = self._run_git(root, "show", "--no-patch", "--format=%T", exact).strip()
                if imported_tree != candidate_tree:
                    raise GitCandidateError("candidate exact import tree changed")
            finally:
                self._run_git(
                    self._source_repository,
                    "update-ref",
                    "-d",
                    temporary_ref,
                    exact,
                    allowed_returncodes=frozenset({0, 1}),
                )
            return exact

    def _import_git_objects(self, root: Path, exact: str, base: str) -> None:
        pack = self._run_git_bytes(
            root,
            "pack-objects",
            "--stdout",
            "--thin",
            "--revs",
            input_bytes=f"{exact}\n^{base}\n".encode("ascii"),
        )
        if not pack:
            raise GitCandidateError("candidate object import was empty")
        self._run_git_bytes(
            self._source_repository,
            "index-pack",
            "--stdin",
            "--fix-thin",
            "--keep=opentulpa-exact-import",
            input_bytes=pack,
        )

    def review_artifact(
        self,
        *,
        candidate_id: str,
        base_commit: str,
        head_commit: str,
    ) -> ReviewArtifact:
        """Materialize the exact bounded candidate diff for owner review."""

        safe_id = self._candidate_id(candidate_id)
        safe_base = self._commit(base_commit)
        safe_head = self._commit(head_commit)
        with self._mutation():
            self._validate_commit_lineage(self._source_repository, safe_base, safe_head)
            patch, changed_paths = self._immutable_review(
                self._source_repository,
                safe_base,
                safe_head,
            )
            self._validate_changed_paths(changed_paths)
            if not changed_paths:
                raise GitCandidateError("candidate review patch is empty")
            digest = hashlib.sha256(patch).hexdigest()
            patch_path = self._artifacts_root / f"review-{safe_id}-{digest[:16]}.patch"
            self._publish_artifact(patch_path, patch)
            return ReviewArtifact(
                candidate_id=safe_id,
                base_commit=safe_base,
                head_commit=safe_head,
                patch_path=patch_path,
                patch_sha256=digest,
            )

    def remove(self, workspace: CandidateWorkspace) -> None:
        with self._mutation():
            root = self._validate_workspace(workspace)
            if self._is_full_repository(root, workspace):
                shutil.rmtree(root)
            else:
                self._remove_path(workspace.candidate_id, root)

    def _create_full_repository(self, path: Path, base_commit: str, candidate_id: str) -> None:
        """Copy Git objects, then initialize a clean repository without remotes."""
        try:
            path.mkdir()
            init_arguments = ["init", "--quiet", "-b", "candidate"]
            if self._oid_length == 64:
                init_arguments.insert(2, "--object-format=sha256")
            self._run_git(path, *init_arguments)
            self._run_git(path, "config", "--local", "user.name", "OpenTulpa Candidate")
            self._run_git(
                path,
                "config",
                "--local",
                "user.email",
                "candidate@opentulpa.local",
            )
            candidate_git, _ = _discover_git_directories(path)
            shutil.copytree(
                self._git_common_directory / "objects",
                candidate_git / "objects",
                dirs_exist_ok=True,
            )
            if (candidate_git / "objects" / "info" / "alternates").exists():
                raise GitCandidateError("candidate repository object store is not independent")
            marker = candidate_git / "opentulpa-candidate"
            marker.write_text(
                f"{_FULL_REPOSITORY_MARKER} {candidate_id} {base_commit}\n",
                encoding="ascii",
            )
            self._run_git(path, "checkout", "--detach", base_commit)
            self._run_git(path, "remote", "remove", "origin", allowed_returncodes=frozenset({0, 2}))
        except (OSError, GitCandidateError):
            with suppress(OSError):
                shutil.rmtree(path)
            raise GitCandidateError("candidate repository creation failed") from None

    def _remove_path(self, candidate_id: str, root: Path) -> None:
        try:
            expected = self._candidate_path(candidate_id).resolve(strict=False)
            resolved = root.resolve(strict=False)
        except OSError:
            raise GitCandidateError("candidate worktree path is unsafe") from None
        if resolved != expected:
            raise GitCandidateError("refusing to remove an unknown worktree")
        self._run_git(
            self._source_repository,
            "worktree",
            "remove",
            "--force",
            str(root),
        )
        self._run_git(self._source_repository, "worktree", "prune")

    def _publish_artifact(self, path: Path, content: bytes) -> None:
        temporary = self._artifacts_root / f".{path.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        directory_descriptor = -1
        try:
            descriptor = os.open(temporary, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_descriptor = os.open(
                self._artifacts_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            os.fsync(directory_descriptor)
        except OSError:
            with suppress(OSError):
                if descriptor >= 0:
                    os.close(descriptor)
                if os.path.lexists(temporary):
                    temporary.unlink()
            raise GitCandidateError("candidate artifact publication failed") from None
        finally:
            if directory_descriptor >= 0:
                os.close(directory_descriptor)

    @staticmethod
    def _register_candidate_worktree(
        git_directory: Path,
        candidate_id: str,
        base_commit: str,
    ) -> None:
        registration = git_directory / "opentulpa-candidate"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(registration, flags, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(f"{candidate_id} {base_commit}\n".encode("ascii"))
                stream.flush()
                os.fsync(stream.fileno())
            directory_descriptor = os.open(
                git_directory,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC,
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            if descriptor >= 0:
                with suppress(OSError):
                    os.close(descriptor)
            raise GitCandidateError("candidate worktree registration failed") from None

    def _resolve_commit(self, ref: str) -> str:
        value = self._run_git(
            self._source_repository,
            "rev-parse",
            "--verify",
            f"{ref}^{{commit}}",
        ).strip()
        return self._commit(value)

    def _private_ref(self, ref: str) -> str | None:
        value = self._run_git(
            self._source_repository,
            "for-each-ref",
            "--format=%(refname)%00%(objectname)%00%(objecttype)%00%(symref)",
            ref,
        ).rstrip("\n")
        if not value:
            return None
        fields = value.split("\0")
        if len(fields) != 4 or fields[0] != ref:
            raise GitCandidateError("managed Git ref is ambiguous")
        if fields[3]:
            raise GitCandidateError("managed Git ref is symbolic")
        if fields[2] != "commit":
            raise GitCandidateError("managed Git ref is invalid")
        try:
            commit = self._commit(fields[1])
        except ValueError:
            raise GitCandidateError("managed Git ref is invalid") from None
        return commit

    def _retain_ref(self, ref: str, commit: str) -> None:
        exact = self._commit(commit)
        current = self._private_ref(ref)
        if current == exact:
            return
        if current is not None:
            raise GitCandidateError("managed Git ref already retains another commit")
        self._cas_ref(ref, exact, expected_old=None)

    def _advance_candidate_ref(
        self,
        workspace: CandidateWorkspace,
        commit: str,
        *,
        previous_head: str | None,
    ) -> None:
        exact = self._commit(commit)
        base = self._commit(workspace.base_commit)
        ref = f"refs/opentulpa/candidates/{self._candidate_id(workspace.candidate_id)}"
        current = self._private_ref(ref)
        if current == exact:
            return
        if current is None:
            if previous_head is not None and self._commit(previous_head) != base:
                raise GitCandidateError("managed Git ref no longer retains the prior commit")
            self._cas_ref(ref, exact, expected_old=None)
            return
        if previous_head is not None:
            expected = self._commit(previous_head)
            if expected == base or current != expected:
                raise GitCandidateError("managed Git ref already retains another commit")
        else:
            if current == base:
                raise GitCandidateError("managed Git ref already retains another commit")
            try:
                self._run_git(
                    self._source_repository,
                    "merge-base",
                    "--is-ancestor",
                    current,
                    exact,
                )
            except GitCandidateError:
                raise GitCandidateError("managed Git ref already retains another commit") from None
            expected = current
        self._cas_ref(ref, exact, expected_old=expected)

    def _cas_ref(self, ref: str, commit: str, *, expected_old: str | None) -> None:
        zero = "0" * self._oid_length
        self._run_git(
            self._source_repository,
            "update-ref",
            "--no-deref",
            ref,
            commit,
            expected_old or zero,
            allowed_returncodes=frozenset({0, 1, 128}),
        )
        if self._private_ref(ref) != commit:
            raise GitCandidateError("managed Git ref compare-and-swap failed")

    def _head(self, root: Path) -> str:
        value = self._run_git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
        try:
            return self._commit(value)
        except ValueError:
            raise GitCandidateError("candidate HEAD is invalid") from None

    def _validate_workspace(self, workspace: CandidateWorkspace) -> Path:
        base_commit = self._commit(workspace.base_commit)
        safe_id = self._candidate_id(workspace.candidate_id)
        if workspace.repository_kind == "full_repository":
            root = candidate_repository_directory(
                workspace.path,
                candidate_id=safe_id,
                base_commit=base_commit,
                worktrees_root=self._worktrees_root,
            )
        else:
            root, _ = _candidate_worktree_directories(
                workspace.path,
                candidate_id=safe_id,
                base_commit=base_commit,
                worktrees_root=self._worktrees_root,
                common_directory=self._git_common_directory,
            )
        self._assert_safe_git_configuration(root)
        symbolic = self._run_git(
            root,
            "symbolic-ref",
            "--quiet",
            "HEAD",
            allowed_returncodes=frozenset({0, 1}),
        ).strip()
        if symbolic:
            raise GitCandidateError("candidate worktree must remain detached")
        resolved_base = self._run_git(
            root,
            "rev-parse",
            "--verify",
            f"{base_commit}^{{commit}}",
        ).strip()
        if resolved_base != base_commit:
            raise GitCandidateError("candidate base commit is invalid")
        return root

    def _is_full_repository(self, root: Path, workspace: CandidateWorkspace) -> bool:
        if workspace.repository_kind != "full_repository":
            return False
        try:
            candidate_repository_directory(
                root,
                candidate_id=workspace.candidate_id,
                base_commit=workspace.base_commit,
                worktrees_root=self._worktrees_root,
            )
        except GitCandidateError:
            return False
        return True

    def _candidate_path(self, candidate_id: str) -> Path:
        return self._worktrees_root / self._candidate_id(candidate_id)

    @staticmethod
    def _candidate_id(value: str) -> str:
        safe = str(value or "").strip()
        if not _CANDIDATE_ID_RE.fullmatch(safe):
            raise ValueError("candidate_id is invalid")
        return safe

    @staticmethod
    def _git_ref(value: str) -> str:
        safe = str(value or "").strip()
        if safe == "HEAD" or _COMMIT_RE.fullmatch(safe):
            return safe
        components = safe.split("/")
        if (
            not safe.startswith("refs/")
            or len(safe) > 500
            or any(not part or part.startswith(".") or part.endswith((".", ".lock")) for part in components)
            or any(ord(character) < 32 or ord(character) == 127 for character in safe)
            or ".." in safe
            or "@{" in safe
            or any(character in safe for character in " ~^:?*[\\")
        ):
            raise ValueError("Git ref is invalid")
        return safe

    def _commit(self, value: str) -> str:
        safe = str(value or "").strip().lower()
        if not _COMMIT_RE.fullmatch(safe) or len(safe) != self._oid_length:
            raise ValueError("Git commit is invalid")
        return safe

    @staticmethod
    def _validate_changed_paths(paths: tuple[str, ...]) -> None:
        for raw_path in paths:
            path = PurePosixPath(raw_path)
            if path.is_absolute() or ".." in path.parts:
                raise GitCandidateError("candidate changed an unsafe path")
            for component in path.parts:
                lowered = component.casefold()
                if lowered in _PUBLIC_ENV_TEMPLATES:
                    continue
                if (
                    lowered in _SENSITIVE_NAMES
                    or lowered.startswith(".env.")
                    or lowered.endswith((".key", ".p12", ".pem", ".pfx"))
                ):
                    raise GitCandidateError("candidate attempted to commit a sensitive path")

    @staticmethod
    def _decode_git_paths(raw: bytes) -> tuple[str, ...]:
        try:
            return tuple(item.decode("utf-8", errors="strict") for item in raw.split(b"\0") if item)
        except UnicodeError:
            raise GitCandidateError("candidate contains a non-UTF-8 path") from None

    @staticmethod
    def _decode_diff(raw: bytes) -> str:
        try:
            return raw.decode("utf-8", errors="strict")
        except UnicodeError:
            raise GitCandidateError("candidate diff is not valid UTF-8") from None

    def _reject_unmerged_index(self, root: Path) -> None:
        if self._run_git_bytes(root, "ls-files", "--unmerged", "-z"):
            raise GitCandidateError("candidate index contains unresolved merge entries")

    def _staged_changed_paths(self, root: Path) -> tuple[str, ...]:
        raw = self._run_git_bytes(
            root,
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            "-z",
            "HEAD",
            "--",
        )
        return self._decode_git_paths(raw)

    def _changed_paths(self, root: Path, *commits: str) -> tuple[str, ...]:
        raw = self._run_git_bytes(
            root,
            "diff",
            "--name-only",
            "--diff-filter=ACDMRTUXB",
            "-z",
            *commits,
            "--",
        )
        return self._decode_git_paths(raw)

    def _tree_entries(self, root: Path, treeish: str) -> dict[str, tuple[str, str, str]]:
        if not _COMMIT_RE.fullmatch(treeish) or len(treeish) != self._oid_length:
            raise GitCandidateError("candidate tree evidence is invalid")
        cache = cast(
            dict[tuple[Path, str], dict[str, tuple[str, str, str]]] | None,
            getattr(self._operation_state, "tree_entries", None),
        )
        cache_key = (root, treeish)
        if cache is not None and cache_key in cache:
            return cache[cache_key]
        raw = self._run_git_bytes(
            root,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            treeish,
        )
        entries: dict[str, tuple[str, str, str]] = {}
        for raw_entry in raw.split(b"\0"):
            if not raw_entry:
                continue
            try:
                metadata, raw_path = raw_entry.split(b"\t", 1)
                raw_mode, raw_type, raw_oid = metadata.split(b" ", 2)
                path = raw_path.decode("utf-8", errors="strict")
                mode = raw_mode.decode("ascii", errors="strict")
                object_type = raw_type.decode("ascii", errors="strict")
                object_id = raw_oid.decode("ascii", errors="strict")
            except (UnicodeError, ValueError):
                raise GitCandidateError("candidate tree evidence is invalid") from None
            if (
                path in entries
                or object_type not in {"blob", "commit"}
                or mode not in {"100644", "100755", "120000", "160000"}
                or not _COMMIT_RE.fullmatch(object_id)
                or len(object_id) != self._oid_length
            ):
                raise GitCandidateError("candidate tree evidence is invalid")
            entries[path] = (mode, object_type, object_id)
        if cache is not None:
            cache[cache_key] = entries
        return entries

    def _read_blobs(self, root: Path, object_ids: Iterator[str]) -> dict[str, bytes]:
        requested = tuple(dict.fromkeys(object_ids))
        if not requested:
            return {}
        cache = cast(
            dict[tuple[Path, str], bytes] | None,
            getattr(self._operation_state, "blobs", None),
        )
        missing = tuple(
            object_id
            for object_id in requested
            if cache is None or (root, object_id) not in cache
        )
        if not missing:
            assert cache is not None
            return {object_id: cache[(root, object_id)] for object_id in requested}
        raw = self._run_git_bytes(
            root,
            "cat-file",
            "--batch",
            input_bytes=b"".join(f"{object_id}\n".encode("ascii") for object_id in missing),
        )
        blobs: dict[str, bytes] = {}
        offset = 0
        for expected in missing:
            header_end = raw.find(b"\n", offset)
            if header_end < 0:
                raise GitCandidateError("candidate blob evidence is invalid")
            fields = raw[offset:header_end].split()
            if len(fields) != 3 or fields[0] != expected.encode("ascii") or fields[1] != b"blob":
                raise GitCandidateError("candidate blob evidence is invalid")
            try:
                size = int(fields[2])
            except ValueError:
                raise GitCandidateError("candidate blob evidence is invalid") from None
            content_start = header_end + 1
            content_end = content_start + size
            if size < 0 or content_end >= len(raw) or raw[content_end : content_end + 1] != b"\n":
                raise GitCandidateError("candidate blob evidence is invalid")
            blobs[expected] = raw[content_start:content_end]
            offset = content_end + 1
        if offset != len(raw):
            raise GitCandidateError("candidate blob evidence is invalid")
        if cache is not None:
            cache.update({(root, object_id): content for object_id, content in blobs.items()})
            return {object_id: cache[(root, object_id)] for object_id in requested}
        return {object_id: blobs[object_id] for object_id in requested}

    @staticmethod
    def _tree_changed_paths(
        base_entries: dict[str, tuple[str, str, str]],
        final_entries: dict[str, tuple[str, str, str]],
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                path
                for path in base_entries.keys() | final_entries.keys()
                if base_entries.get(path) != final_entries.get(path)
            )
        )

    def _immutable_review(
        self,
        root: Path,
        base_treeish: str,
        final_treeish: str,
    ) -> tuple[bytes, tuple[str, ...]]:
        base_entries = self._tree_entries(root, base_treeish)
        final_entries = self._tree_entries(root, final_treeish)
        paths = self._tree_changed_paths(base_entries, final_entries)
        blobs = self._read_blobs(
            root,
            (
                entry[2]
                for path in paths
                for entry in (base_entries.get(path), final_entries.get(path))
                if entry is not None and entry[1] == "blob"
            ),
        )
        evidence = bytearray(_IMMUTABLE_REVIEW_HEADER)
        for path in paths:
            record = {
                "new": self._review_entry(final_entries.get(path), blobs),
                "old": self._review_entry(base_entries.get(path), blobs),
                "path": path,
            }
            evidence.extend(
                json.dumps(
                    record,
                    ensure_ascii=True,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            )
            evidence.extend(b"\n")
        return bytes(evidence), paths

    def _review_entry(
        self,
        entry: tuple[str, str, str] | None,
        blobs: dict[str, bytes],
    ) -> dict[str, str] | None:
        if entry is None:
            return None
        mode, object_type, object_id = entry
        result = {"mode": mode, "oid": object_id, "type": object_type}
        if object_type == "blob":
            blob = blobs[object_id]
            result["base64"] = base64.b64encode(blob).decode("ascii")
            with suppress(UnicodeError):
                result["text"] = blob.decode("utf-8", errors="strict")
        return result

    def _validate_commit_lineage(self, root: Path, base: str, head: str) -> None:
        try:
            self._run_git(root, "merge-base", "--is-ancestor", base, head)
        except GitCandidateError:
            raise GitCandidateError("candidate base is not an ancestor of HEAD") from None
        base_entries = self._tree_entries(root, base)
        raw_commits = self._run_git_bytes(
            root,
            "rev-list",
            "--reverse",
            "--topo-order",
            f"{base}..{head}",
        )
        try:
            commits = tuple(line for line in raw_commits.decode("ascii", errors="strict").splitlines())
        except UnicodeError:
            raise GitCandidateError("candidate commit lineage is invalid") from None
        for commit in commits:
            try:
                exact = self._commit(commit)
            except ValueError:
                raise GitCandidateError("candidate commit lineage is invalid") from None
            entries = self._tree_entries(root, exact)
            paths = self._tree_changed_paths(base_entries, entries)
            self._validate_changed_paths(paths)
            self._validate_tree_entries_content(root, entries, paths)

    def _validate_tree_entries_content(
        self,
        root: Path,
        entries: dict[str, tuple[str, str, str]],
        paths: tuple[str, ...],
    ) -> None:
        blobs = self._read_blobs(
            root,
            (
                entry[2]
                for path in paths
                if (entry := entries.get(path)) is not None and entry[1] == "blob"
            ),
        )
        for path in paths:
            entry = entries.get(path)
            if entry is None or entry[1] != "blob":
                continue
            content = blobs[entry[2]]
            if candidate_content_contains_secret(path, content):
                raise GitCandidateError("candidate commit contains credential material")

    def _write_tree(self, root: Path) -> str:
        value = self._run_git(root, "write-tree").strip().lower()
        if not _COMMIT_RE.fullmatch(value) or len(value) != self._oid_length:
            raise GitCandidateError("candidate index tree is invalid")
        return value

    def _validate_index_content(self, root: Path, paths: tuple[str, ...]) -> None:
        raw = self._run_git_bytes(root, "ls-files", "--stage", "-z", "--", *paths)
        requested = set(paths)
        indexed: dict[str, tuple[str, str]] = {}
        for entry in (entry for entry in raw.split(b"\0") if entry):
            if b"\t" not in entry:
                raise GitCandidateError("candidate index evidence is ambiguous")
            metadata, raw_path = entry.split(b"\t", 1)
            fields = metadata.split()
            if len(fields) != 3 or fields[2] != b"0":
                raise GitCandidateError("candidate index evidence is ambiguous")
            try:
                indexed_path = raw_path.decode("utf-8", errors="strict")
                mode = fields[0].decode("ascii", errors="strict")
                object_id = fields[1].decode("ascii", errors="strict")
            except UnicodeError:
                raise GitCandidateError("candidate index evidence is invalid") from None
            if (
                indexed_path not in requested
                or indexed_path in indexed
                or mode not in {"100644", "100755", "120000", "160000"}
                or not _COMMIT_RE.fullmatch(object_id)
                or len(object_id) != self._oid_length
            ):
                raise GitCandidateError("candidate index evidence is invalid")
            indexed[indexed_path] = (mode, object_id)
        blobs = self._read_blobs(
            root,
            (object_id for mode, object_id in indexed.values() if mode != "160000"),
        )
        for path in paths:
            indexed_entry = indexed.get(path)
            if indexed_entry is None or indexed_entry[0] == "160000":
                continue
            content = blobs[indexed_entry[1]]
            if candidate_content_contains_secret(path, content):
                raise GitCandidateError("candidate attempted to commit credential material")

    def _validate_tree_content(
        self,
        root: Path,
        commit: str,
        paths: tuple[str, ...],
    ) -> None:
        entries = self._tree_entries(root, commit)
        self._validate_tree_entries_content(root, entries, paths)

    def _require_unchanged_index(
        self,
        root: Path,
        changed_paths: tuple[str, ...],
        expected_tree: str,
    ) -> None:
        self._reject_unmerged_index(root)
        if self._staged_changed_paths(root) != changed_paths or self._write_tree(root) != expected_tree:
            raise GitCandidateError("candidate index changed while preparing its commit")

    @staticmethod
    def _regular_directory(path: str | Path, *, create: bool) -> Path:
        raw = Path(path).expanduser()
        if raw.is_symlink():
            raise ValueError("Git workspace roots cannot be symlinks")
        try:
            if create:
                raw.mkdir(parents=True, exist_ok=True)
            resolved = raw.resolve(strict=True)
        except OSError:
            raise ValueError("Git workspace root is unavailable") from None
        if not resolved.is_dir():
            raise ValueError("Git workspace root must be a directory")
        return resolved

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    @contextmanager
    def _mutation(self) -> Iterator[None]:
        with self._lock:
            try:
                with repository_mutation_lock(
                    self._git_common_directory,
                    timeout_seconds=self._timeout_seconds,
                ):
                    depth = getattr(self._operation_state, "depth", 0)
                    if depth == 0:
                        self._operation_state.validated_directories = set()
                        self._operation_state.tree_entries = {}
                        self._operation_state.blobs = {}
                        try:
                            self._assert_safe_git_configuration(self._source_repository)
                        except Exception:
                            del self._operation_state.validated_directories
                            del self._operation_state.tree_entries
                            del self._operation_state.blobs
                            raise
                    self._operation_state.depth = depth + 1
                    try:
                        yield
                    finally:
                        if depth == 0:
                            del self._operation_state.validated_directories
                            del self._operation_state.tree_entries
                            del self._operation_state.blobs
                            del self._operation_state.depth
                        else:
                            self._operation_state.depth = depth
            except RepositoryMutationLockError:
                raise GitCandidateError("Git candidate mutation lock failed") from None

    def _assert_safe_git_configuration(self, cwd: Path) -> None:
        git_directory, common_directory = _discover_git_directories(cwd)
        independent = common_directory == git_directory and cwd != self._source_repository
        if common_directory != self._git_common_directory and not independent:
            raise GitCandidateError("candidate belongs to another repository")
        key = (git_directory, common_directory)
        validated = getattr(self._operation_state, "validated_directories", None)
        if validated is not None and key in validated:
            return
        if _repository_git_configuration_is_unsafe(
            cwd,
            git_directory,
            common_directory,
            timeout_seconds=self._timeout_seconds,
            max_output_bytes=self._max_git_output_bytes,
        ):
            raise GitCandidateError("repository Git configuration is unsafe")
        if independent and self._run_git(cwd, "remote").strip():
            raise GitCandidateError("candidate repository must not have remotes")
        if validated is not None:
            validated.add(key)

    def _run_git(
        self,
        cwd: Path,
        *arguments: str,
        env: dict[str, str] | None = None,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> str:
        output = self._run_git_bytes(
            cwd,
            *arguments,
            env=env,
            allowed_returncodes=allowed_returncodes,
        )
        return output.decode("utf-8", errors="replace")

    def _run_git_bytes(
        self,
        cwd: Path,
        *arguments: str,
        env: dict[str, str] | None = None,
        input_bytes: bytes | None = None,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> bytes:
        if getattr(self._operation_state, "depth", 0) <= 0:
            raise GitCandidateError("Git candidate operation lacks a security context")
        try:
            completed = _run_trusted_git_process(
                cwd,
                arguments,
                timeout_seconds=self._timeout_seconds,
                max_output_bytes=self._max_git_output_bytes,
                env=env,
                input_bytes=input_bytes,
            )
        except OSError:
            raise GitCandidateError("Git candidate operation failed") from None
        if (
            completed.returncode not in allowed_returncodes
            or completed.truncated
            or completed.timed_out
        ):
            raise GitCandidateError("Git candidate operation failed")
        return completed.output


def candidate_path_is_promotable(raw_path: str) -> bool:
    """Return whether a normal, non-secret repository path may be released."""

    return candidate_path_is_runtime_overlay(raw_path)


def candidate_path_is_runtime_overlay(raw_path: str) -> bool:
    """Return whether a path may be copied over the immutable dependency base."""

    path = PurePosixPath(str(raw_path or ""))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return False
    for component in path.parts:
        lowered = component.casefold()
        if lowered in _RESERVED_SOURCE_COMPONENTS:
            return False
        if lowered in _PUBLIC_ENV_TEMPLATES:
            continue
        if (
            lowered in _SENSITIVE_NAMES
            or lowered.startswith(".env.")
            or lowered.endswith((".key", ".p12", ".pem", ".pfx"))
        ):
            return False
    return True


def candidate_content_contains_secret(raw_path: str, content: bytes) -> bool:
    """Detect credential material in a staged or committed Git blob."""

    del raw_path
    if _PRIVATE_KEY_BEGIN_RE.search(content) and _PRIVATE_KEY_END_RE.search(content):
        return True
    for pattern in (*_PROVIDER_TOKEN_RES, _BEARER_RE, _ASSIGNED_CREDENTIAL_RE):
        if pattern.search(content):
            return True
    return False


__all__ = [
    "CandidateCommit",
    "CandidateWorkspace",
    "ContributionArtifact",
    "GitCandidateError",
    "GitCandidateWorkspace",
    "RepositoryMutationLockError",
    "ReviewArtifact",
    "candidate_path_is_promotable",
    "candidate_path_is_runtime_overlay",
    "candidate_content_contains_secret",
    "candidate_repository_directory",
    "repository_mutation_lock",
]
