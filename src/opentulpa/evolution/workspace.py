"""Git-native candidate worktrees that never mutate the running checkout."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from opentulpa.evolution.process import run_bounded_process

_CANDIDATE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40,64}\Z")
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
_TEST_CREDENTIAL_MARKER = "opentulpa: allow-test-credential"
_PRIVATE_KEY_BEGIN_RE = re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
_PRIVATE_KEY_END_RE = re.compile(rb"-----END (?:[A-Z0-9 ]+ )?PRIVATE KEY-----")
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
_PLACEHOLDER_MARKERS = (
    b"changeme",
    b"disabled",
    b"dummy",
    b"example",
    b"fake",
    b"invalid",
    b"not-a-real",
    b"placeholder",
    b"redacted",
    b"test",
    b"abcdefghijklmnopqrstuvwxyz",
)


class GitCandidateError(RuntimeError):
    """A candidate repository operation failed without exposing command output."""


@dataclass(frozen=True, slots=True)
class CandidateWorkspace:
    candidate_id: str
    path: Path
    base_commit: str


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
        self._run_git(self._source_repository, "rev-parse", "--git-dir")

    @property
    def source_repository(self) -> Path:
        return self._source_repository

    @property
    def worktrees_root(self) -> Path:
        return self._worktrees_root

    def create(self, *, candidate_id: str, base_ref: str = "HEAD") -> CandidateWorkspace:
        safe_id = self._candidate_id(candidate_id)
        safe_ref = self._git_ref(base_ref)
        with self._lock:
            base_commit = self._resolve_commit(safe_ref)
            path = self._candidate_path(safe_id)
            if os.path.lexists(path):
                raise GitCandidateError("candidate worktree already exists")
            self._run_git(
                self._source_repository,
                "worktree",
                "add",
                "--detach",
                str(path),
                base_commit,
            )
            resolved = path.resolve(strict=True)
            if not self._is_relative_to(resolved, self._worktrees_root):
                self._remove_path(safe_id, resolved)
                raise GitCandidateError("candidate worktree escaped its configured root")
            return CandidateWorkspace(
                candidate_id=safe_id,
                path=resolved,
                base_commit=base_commit,
            )

    def head(self, workspace: CandidateWorkspace) -> str:
        root = self._validate_workspace(workspace)
        commit = self._run_git(root, "rev-parse", "--verify", "HEAD^{commit}").strip()
        if not _COMMIT_RE.fullmatch(commit):
            raise GitCandidateError("candidate HEAD is invalid")
        return commit

    def status(self, workspace: CandidateWorkspace) -> tuple[str, ...]:
        root = self._validate_workspace(workspace)
        raw = self._run_git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z")
        return tuple(item for item in raw.split("\0") if item)

    def diff(self, workspace: CandidateWorkspace) -> str:
        root = self._validate_workspace(workspace)
        with self._lock:
            self._run_git(root, "add", "--intent-to-add", "--all")
            return self._run_git(root, "diff", "--binary", "--no-ext-diff", "HEAD", "--")

    def full_diff(self, workspace: CandidateWorkspace) -> str:
        """Return one canonical base-to-working-tree diff across intermediate commits."""

        root = self._validate_workspace(workspace)
        with self._lock:
            self._run_git(root, "add", "--intent-to-add", "--all")
            return self._run_git(
                root,
                "diff",
                "--binary",
                "--no-ext-diff",
                workspace.base_commit,
                "--",
            )

    def commit(
        self,
        workspace: CandidateWorkspace,
        *,
        message: str,
    ) -> CandidateCommit:
        root = self._validate_workspace(workspace)
        safe_message = " ".join(str(message or "").split())[:500]
        if not safe_message:
            raise ValueError("candidate commit message is required")
        with self._lock:
            self._run_git(root, "add", "--intent-to-add", "--all")
            names = self._run_git(
                root,
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                "-z",
                "HEAD",
                "--",
            )
            changed_paths = tuple(path for path in names.split("\0") if path)
            if not changed_paths:
                raise GitCandidateError("candidate did not change source files")
            self._validate_changed_paths(changed_paths)
            self._validate_changed_content(root, changed_paths)
            self._run_git(root, "add", "--all")
            staged_names = self._run_git(
                root,
                "diff",
                "--cached",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                "-z",
                "--",
            )
            if tuple(path for path in staged_names.split("\0") if path) != changed_paths:
                raise GitCandidateError("candidate source changed while preparing its commit")
            env = {
                "PATH": os.environ.get("PATH", os.defpath),
                "HOME": "/tmp",
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
            source_commit = self.head(workspace)
            diff = self._run_git(
                root,
                "diff",
                "--binary",
                "--no-ext-diff",
                workspace.base_commit,
                source_commit,
                "--",
            )
            diff_sha256 = hashlib.sha256(diff.encode("utf-8")).hexdigest()
            self._run_git(
                self._source_repository,
                "update-ref",
                f"refs/opentulpa/candidates/{workspace.candidate_id}",
                source_commit,
            )
            return CandidateCommit(
                candidate_id=workspace.candidate_id,
                base_commit=workspace.base_commit,
                source_commit=source_commit,
                diff_sha256=diff_sha256,
                changed_paths=changed_paths,
                promotion_eligible=all(
                    candidate_path_is_promotable(path) for path in changed_paths
                ),
            )

    def recover_commit(self, workspace: CandidateWorkspace) -> CandidateCommit:
        """Rebind a clean descendant commit after a crash before archive persistence."""

        root = self._validate_workspace(workspace)
        with self._lock:
            if self.status(workspace):
                raise GitCandidateError("candidate worktree is not clean")
            source_commit = self.head(workspace)
            base_commit = self._commit(workspace.base_commit)
            if source_commit == base_commit:
                raise GitCandidateError("candidate has no committed source changes")
            self._run_git(root, "merge-base", "--is-ancestor", base_commit, source_commit)
            names = self._run_git(
                root,
                "diff",
                "--name-only",
                "--diff-filter=ACDMRTUXB",
                "-z",
                base_commit,
                source_commit,
                "--",
            )
            changed_paths = tuple(path for path in names.split("\0") if path)
            if not changed_paths:
                raise GitCandidateError("candidate has no committed source changes")
            self._validate_changed_paths(changed_paths)
            self._validate_changed_content(root, changed_paths)
            diff = self._run_git(
                root,
                "diff",
                "--binary",
                "--no-ext-diff",
                base_commit,
                source_commit,
                "--",
            )
            diff_sha256 = hashlib.sha256(diff.encode("utf-8")).hexdigest()
            self._run_git(
                self._source_repository,
                "update-ref",
                f"refs/opentulpa/candidates/{workspace.candidate_id}",
                source_commit,
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
        with self._lock:
            self._run_git(
                self._source_repository,
                "merge-base",
                "--is-ancestor",
                safe_base,
                safe_head,
            )
            patch = self._run_git(
                self._source_repository,
                "format-patch",
                "--stdout",
                "--no-signature",
                f"{safe_base}..{safe_head}",
            )
            if not patch.strip():
                raise GitCandidateError("candidate contribution patch is empty")
            digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
            patch_path = self._artifacts_root / f"{safe_id}-{digest[:16]}.patch"
            temporary = patch_path.with_suffix(".patch.tmp")
            temporary.write_text(patch, encoding="utf-8")
            os.replace(temporary, patch_path)
            branch_name = f"opentulpa/candidate-{safe_id}"
            self._run_git(
                self._source_repository,
                "update-ref",
                f"refs/opentulpa/contributions/{safe_id}",
                safe_head,
            )
            return ContributionArtifact(
                candidate_id=safe_id,
                base_commit=safe_base,
                head_commit=safe_head,
                branch_name=branch_name,
                patch_path=patch_path,
                patch_sha256=digest,
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
        with self._lock:
            self._run_git(
                self._source_repository,
                "merge-base",
                "--is-ancestor",
                safe_base,
                safe_head,
            )
            patch = self._run_git(
                self._source_repository,
                "diff",
                "--binary",
                "--no-ext-diff",
                safe_base,
                safe_head,
                "--",
            )
            if not patch.strip():
                raise GitCandidateError("candidate review patch is empty")
            digest = hashlib.sha256(patch.encode("utf-8")).hexdigest()
            patch_path = self._artifacts_root / f"review-{safe_id}-{digest[:16]}.patch"
            temporary = patch_path.with_suffix(".patch.tmp")
            temporary.write_text(patch, encoding="utf-8")
            temporary.chmod(0o600)
            os.replace(temporary, patch_path)
            return ReviewArtifact(
                candidate_id=safe_id,
                base_commit=safe_base,
                head_commit=safe_head,
                patch_path=patch_path,
                patch_sha256=digest,
            )

    def remove(self, workspace: CandidateWorkspace) -> None:
        root = self._validate_workspace(workspace)
        with self._lock:
            self._remove_path(workspace.candidate_id, root)

    def _remove_path(self, candidate_id: str, root: Path) -> None:
        expected = self._candidate_path(candidate_id).resolve(strict=False)
        if root.resolve(strict=False) != expected:
            raise GitCandidateError("refusing to remove an unknown worktree")
        self._run_git(
            self._source_repository,
            "worktree",
            "remove",
            "--force",
            str(root),
        )
        self._run_git(self._source_repository, "worktree", "prune")

    def _resolve_commit(self, ref: str) -> str:
        value = self._run_git(
            self._source_repository,
            "rev-parse",
            "--verify",
            f"{ref}^{{commit}}",
        ).strip()
        return self._commit(value)

    def _validate_workspace(self, workspace: CandidateWorkspace) -> Path:
        safe_id = self._candidate_id(workspace.candidate_id)
        expected = self._candidate_path(safe_id).resolve(strict=False)
        root = workspace.path.expanduser()
        if root.is_symlink() or not root.is_dir():
            raise GitCandidateError("candidate worktree is unavailable")
        resolved = root.resolve(strict=True)
        if resolved != expected or not self._is_relative_to(resolved, self._worktrees_root):
            raise GitCandidateError("candidate worktree escaped its configured root")
        return resolved

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
        if (
            not safe
            or len(safe) > 500
            or safe.startswith("-")
            or any(ord(character) < 32 or ord(character) == 127 for character in safe)
            or ".." in safe
            or safe.endswith((".", "/"))
            or "@{" in safe
            or "\\" in safe
        ):
            raise ValueError("Git ref is invalid")
        return safe

    @staticmethod
    def _commit(value: str) -> str:
        safe = str(value or "").strip().lower()
        if not _COMMIT_RE.fullmatch(safe):
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
    def _validate_changed_content(root: Path, paths: tuple[str, ...]) -> None:
        for raw_path in paths:
            path = root.joinpath(*PurePosixPath(raw_path).parts)
            if not os.path.lexists(path):
                continue
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                continue
            try:
                content = path.read_bytes()
            except OSError as exc:
                raise GitCandidateError("candidate source content could not be inspected") from exc
            if candidate_content_contains_secret(raw_path, content):
                raise GitCandidateError("candidate attempted to commit credential material")

    @staticmethod
    def _regular_directory(path: str | Path, *, create: bool) -> Path:
        raw = Path(path).expanduser()
        if raw.is_symlink():
            raise ValueError("Git workspace roots cannot be symlinks")
        if create:
            raw.mkdir(parents=True, exist_ok=True)
        resolved = raw.resolve(strict=True)
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

    def _run_git(
        self,
        cwd: Path,
        *arguments: str,
        env: dict[str, str] | None = None,
    ) -> str:
        process_env = {
            "PATH": os.environ.get("PATH", os.defpath),
            "HOME": "/tmp",
            **(env or {}),
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_PROTOCOL_FROM_USER": "0",
        }
        try:
            completed = run_bounded_process(
                [
                    "git",
                    "-C",
                    str(cwd),
                    "-c",
                    f"core.hooksPath={os.devnull}",
                    *arguments,
                ],
                cwd=cwd,
                env=process_env,
                timeout_seconds=self._timeout_seconds,
                max_output_bytes=self._max_git_output_bytes,
            )
        except OSError as exc:
            raise GitCandidateError("Git candidate operation failed") from exc
        if completed.returncode != 0 or completed.truncated:
            raise GitCandidateError("Git candidate operation failed")
        return completed.output.decode("utf-8", errors="replace")


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
    """Detect credential material while allowing explicit, obvious test fixtures."""

    normalized_path = PurePosixPath(str(raw_path or ""))
    fixture_allowed = (
        _TEST_CREDENTIAL_MARKER.encode("ascii") in content.lower()
        and any(part.casefold() in {"test", "tests", "fixture", "fixtures"} for part in normalized_path.parts)
    )
    if fixture_allowed:
        return False
    if _PRIVATE_KEY_BEGIN_RE.search(content) and _PRIVATE_KEY_END_RE.search(content):
        return True
    for pattern in (*_PROVIDER_TOKEN_RES, _BEARER_RE, _ASSIGNED_CREDENTIAL_RE):
        for match in pattern.finditer(content):
            value = next((group for group in match.groups() if group is not None), match.group(0))
            if not _credential_is_placeholder(value):
                return True
    return False


def _credential_is_placeholder(value: bytes) -> bool:
    lowered = value.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return True
    meaningful = bytes(character for character in lowered if chr(character).isalnum())
    return bool(meaningful) and len(set(meaningful)) <= 2


__all__ = [
    "CandidateCommit",
    "CandidateWorkspace",
    "ContributionArtifact",
    "GitCandidateError",
    "GitCandidateWorkspace",
    "ReviewArtifact",
    "candidate_path_is_promotable",
    "candidate_path_is_runtime_overlay",
    "candidate_content_contains_secret",
]
