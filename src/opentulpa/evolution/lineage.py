"""Native Git refs and merge state for instance/upstream lineage."""

from __future__ import annotations

import os
import re
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path, PurePosixPath
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

from opentulpa.evolution.git_security import (
    GitSecurityError,
    RepositoryMutationLockError,
    candidate_worktree_directories,
    discover_git_directories,
    repository_git_configuration_is_unsafe,
    repository_mutation_lock,
    run_hardened_git,
)
from opentulpa.evolution.workspace import (
    CandidateWorkspace,
    GitCandidateError,
    candidate_content_contains_secret,
    candidate_path_is_promotable,
    candidate_repository_directory,
)

UPSTREAM_REF = "refs/heads/upstream"
INSTANCE_REF = "refs/opentulpa/instance"
ACCEPTED_UPSTREAM_REF = "refs/opentulpa/upstream/accepted"
DEFAULT_UPSTREAM_REF = UPSTREAM_REF
DEFAULT_INSTANCE_REF = INSTANCE_REF
DEFAULT_ACCEPTED_UPSTREAM_REF = ACCEPTED_UPSTREAM_REF
UPSTREAM_LINEAGE_METADATA_KEY = "opentulpa.evolution.upstream_lineage"

_GIT_IDENTITY_ENV = {
    "GIT_AUTHOR_NAME": "OpenTulpa Candidate",
    "GIT_AUTHOR_EMAIL": "candidate@opentulpa.local",
    "GIT_COMMITTER_NAME": "OpenTulpa Supervisor",
    "GIT_COMMITTER_EMAIL": "supervisor@opentulpa.local",
}
_OID_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_COMMIT_PATTERN = r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$"
_CONFLICT_MARKER_RE = re.compile(
    rb"(?m)^(?:<<<<<<<(?: .*)?\r?$|\|\|\|\|\|\|\|(?: .*)?\r?$|=======\r?$|>>>>>>>(?: .*)?\r?$)"
)


class GitLineageError(RuntimeError):
    """A lineage operation failed without exposing Git output or local paths."""


class UpstreamLineage(BaseModel):
    """Rollback-safe source lineage stored under ``UPSTREAM_LINEAGE_METADATA_KEY``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    upstream_commit: str | None = Field(default=None, pattern=_COMMIT_PATTERN)
    merge_base_commit: str | None = Field(default=None, pattern=_COMMIT_PATTERN)

    @model_validator(mode="after")
    def _coupled_commits(self) -> Self:
        if (self.upstream_commit is None) != (self.merge_base_commit is None):
            raise ValueError("upstream_commit and merge_base_commit must be recorded together")
        if (
            self.upstream_commit is not None
            and self.merge_base_commit is not None
            and len(self.upstream_commit) != len(self.merge_base_commit)
        ):
            raise ValueError("upstream lineage commits must use the same object format")
        return self


@dataclass(frozen=True, slots=True)
class _GitResult:
    returncode: int
    output: str


class ConflictStage(IntEnum):
    """Git's native unmerged index stages for a two-parent instance merge."""

    BASE = 1
    INSTANCE = 2
    UPSTREAM = 3


@dataclass(frozen=True, slots=True)
class GitLineageSnapshot:
    instance_commit: str
    upstream_commit: str
    accepted_upstream_commit: str
    merge_base_commit: str

    @property
    def upstream_lineage(self) -> UpstreamLineage:
        return UpstreamLineage(
            upstream_commit=self.upstream_commit,
            merge_base_commit=self.merge_base_commit,
        )


@dataclass(frozen=True, slots=True)
class NativeMerge:
    instance_commit: str
    upstream_commit: str
    merge_base_commit: str
    conflicted_paths: tuple[str, ...]

    @property
    def upstream_lineage(self) -> UpstreamLineage:
        return UpstreamLineage(
            upstream_commit=self.upstream_commit,
            merge_base_commit=self.merge_base_commit,
        )


@dataclass(frozen=True, slots=True)
class UpstreamSync:
    previous_commit: str
    upstream_commit: str

    @property
    def changed(self) -> bool:
        return self.previous_commit != self.upstream_commit


class GitLineage:
    """Project instance refs and prepare durable native merges in candidate worktrees."""

    def __init__(
        self,
        repository: str | Path,
        *,
        worktrees_root: str | Path,
        upstream_ref: str = DEFAULT_UPSTREAM_REF,
        instance_ref: str = DEFAULT_INSTANCE_REF,
        accepted_upstream_ref: str = DEFAULT_ACCEPTED_UPSTREAM_REF,
        timeout_seconds: int = 120,
        max_git_output_bytes: int = 50 * 1024 * 1024,
    ) -> None:
        if timeout_seconds < 1 or timeout_seconds > 3_600:
            raise ValueError("Git timeout must be between 1 and 3600 seconds")
        if max_git_output_bytes < 1_024:
            raise ValueError("Git output limit is too small")
        self._repository = self._regular_directory(repository)
        self._worktrees_root = self._regular_directory(worktrees_root)
        if self._is_relative_to(self._repository, self._worktrees_root):
            raise ValueError("repository cannot be inside the candidate worktree root")
        self.upstream_ref = self._ref(upstream_ref)
        self.instance_ref = self._ref(instance_ref)
        self.accepted_upstream_ref = self._ref(accepted_upstream_ref)
        if len({self.upstream_ref, self.instance_ref, self.accepted_upstream_ref}) != 3:
            raise ValueError("lineage refs must be distinct")
        self._timeout_seconds = timeout_seconds
        self._max_git_output_bytes = max_git_output_bytes
        self._lock = threading.RLock()
        self._operation_state = threading.local()
        try:
            self._git_directory, self._git_common_directory = discover_git_directories(
                self._repository
            )
        except GitSecurityError:
            raise GitLineageError("Git repository metadata is unsafe") from None
        with self._mutation():
            object_format = self._run_git(
                self._repository,
                "rev-parse",
                "--show-object-format",
            ).output.strip()
        if object_format not in {"sha1", "sha256"}:
            raise GitLineageError("repository object format is unsupported")
        self._oid_length = 40 if object_format == "sha1" else 64

    def resolve_ref(self, ref: str) -> str:
        with self._mutation():
            return self._resolve_ref(ref)

    def initialize(
        self,
        instance_commit: str,
        accepted_upstream_commit: str | None = None,
    ) -> GitLineageSnapshot:
        """Create both private refs atomically, requiring that neither already exists."""
        with self._mutation():
            instance = self._resolve_commit(instance_commit)
            upstream = self._resolve_ref(self.upstream_ref)
            accepted = self._resolve_commit(accepted_upstream_commit or upstream)
            if not self._is_ancestor(accepted, instance):
                raise GitLineageError("accepted upstream is not in the instance")
            if not self._is_ancestor(accepted, upstream):
                raise GitLineageError("accepted commit is not in the upstream lineage")
            self._update_refs(
                instance_commit=instance,
                accepted_upstream_commit=accepted,
                expected_instance_commit=None,
                expected_accepted_upstream_commit=None,
                expected_upstream_commit=upstream,
            )
            return self._snapshot()

    def project(
        self,
        instance_commit: str,
        accepted_upstream_commit: str,
        *,
        expected_instance_commit: str,
        expected_accepted_upstream_commit: str,
    ) -> GitLineageSnapshot:
        """Atomically project active/accepted refs with expected-old compare-and-swap."""
        with self._mutation():
            instance = self._resolve_commit(instance_commit)
            accepted = self._resolve_commit(accepted_upstream_commit)
            expected_instance = self._resolve_commit(expected_instance_commit)
            expected_accepted = self._resolve_commit(expected_accepted_upstream_commit)
            upstream = self._resolve_ref(self.upstream_ref)
            if not self._is_ancestor(accepted, instance):
                raise GitLineageError("accepted upstream is not in the instance")
            if not self._is_ancestor(accepted, upstream):
                raise GitLineageError("accepted commit is not in the upstream lineage")
            self._update_refs(
                instance_commit=instance,
                accepted_upstream_commit=accepted,
                expected_instance_commit=expected_instance,
                expected_accepted_upstream_commit=expected_accepted,
                expected_upstream_commit=upstream,
            )
            return self._snapshot()

    def is_ancestor(self, ancestor_commit: str, descendant_commit: str) -> bool:
        with self._mutation():
            ancestor = self._resolve_commit(ancestor_commit)
            descendant = self._resolve_commit(descendant_commit)
            return self._is_ancestor(ancestor, descendant)

    def _is_ancestor(self, ancestor: str, descendant: str) -> bool:
        result = self._run_git(
            self._repository,
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
            allowed_returncodes=frozenset({0, 1}),
        )
        return result.returncode == 0

    def merge_base(self, instance_commit: str, upstream_commit: str) -> str:
        with self._mutation():
            instance = self._resolve_commit(instance_commit)
            upstream = self._resolve_commit(upstream_commit)
            return self._merge_base(instance, upstream)

    def _merge_base(self, instance: str, upstream: str) -> str:
        values = self._run_git(
            self._repository,
            "merge-base",
            "--all",
            instance,
            upstream,
        ).output.splitlines()
        if len(values) != 1:
            raise GitLineageError("lineage does not have one exact merge base")
        return self._object_id(values[0].strip())

    def snapshot(self) -> GitLineageSnapshot:
        """Resolve all lineage refs and their exact candidate merge base."""
        with self._mutation():
            return self._snapshot()

    def _snapshot(self) -> GitLineageSnapshot:
        instance = self._resolve_ref(self.instance_ref)
        upstream = self._resolve_ref(self.upstream_ref)
        accepted = self._resolve_ref(self.accepted_upstream_ref)
        return GitLineageSnapshot(
            instance_commit=instance,
            upstream_commit=upstream,
            accepted_upstream_commit=accepted,
            merge_base_commit=self._merge_base(instance, upstream),
        )

    def snapshot_upstream_lineage(self) -> UpstreamLineage:
        return self.snapshot().upstream_lineage

    def sync_upstream(self, repository_url: str, remote_ref: str) -> UpstreamSync:
        """Fetch one HTTPS branch and atomically advance the monotonic upstream ref."""

        url = self._https_repository_url(repository_url)
        source_ref = self._remote_branch_ref(remote_ref)
        fetched_ref = "refs/opentulpa/upstream/fetched"
        with self._mutation():
            previous = self._resolve_ref(self.upstream_ref)
            self._run_git(
                self._repository,
                "update-ref",
                "-d",
                fetched_ref,
                allowed_returncodes=frozenset({0, 1}),
            )
            self._fetch_https_upstream(url, source_ref, fetched_ref)
            fetched = self._resolve_ref(fetched_ref)
            if not self._is_ancestor(previous, fetched):
                self._run_git(self._repository, "update-ref", "-d", fetched_ref)
                raise GitLineageError("remote upstream rewrote previously imported history")
            if fetched == previous:
                self._run_git(self._repository, "update-ref", "-d", fetched_ref, fetched)
                return UpstreamSync(previous_commit=previous, upstream_commit=fetched)
            commands = (
                "start\n"
                f"update {self.upstream_ref} {fetched} {previous}\n"
                f"delete {fetched_ref} {fetched}\n"
                "prepare\n"
                "commit\n"
            ).encode("ascii")
            self._run_git(
                self._repository,
                "update-ref",
                "--stdin",
                input_bytes=commands,
            )
            return UpstreamSync(previous_commit=previous, upstream_commit=fetched)

    def _fetch_https_upstream(self, repository_url: str, remote_ref: str, target_ref: str) -> None:
        self._run_git(
            self._repository,
            "fetch",
            "--no-tags",
            "--no-recurse-submodules",
            "--no-write-fetch-head",
            repository_url,
            f"{remote_ref}:{target_ref}",
            allow_https=True,
        )

    def prepare_merge(
        self,
        workspace: CandidateWorkspace,
        lineage: UpstreamLineage | None = None,
    ) -> NativeMerge:
        """Prepare a no-commit merge; conflicts remain in MERGE_HEAD and the index."""
        with self._mutation():
            root = self._workspace_root(workspace)
            expected = lineage or self._snapshot().upstream_lineage
            if expected.upstream_commit is None or expected.merge_base_commit is None:
                raise GitLineageError("candidate upstream lineage is unavailable")
            upstream = self._resolve_commit(expected.upstream_commit)
            expected_base = self._resolve_commit(expected.merge_base_commit)
            workspace_base = self._resolve_commit(workspace.base_commit)
            if self._resolve_ref(self.instance_ref) != workspace_base:
                raise GitLineageError("candidate is based on a stale instance")
            if self._resolve_ref(self.upstream_ref) != upstream:
                raise GitLineageError("candidate upstream lineage is stale")
            actual_base = self._merge_base(workspace_base, upstream)
            if actual_base != expected_base:
                raise GitLineageError("candidate merge base is stale")
            if self._head(root) != workspace_base:
                raise GitLineageError("candidate HEAD is not its instance base")
            if self._merge_head(root, required=False) is not None:
                raise GitLineageError("candidate already has a native merge in progress")
            status = self._run_git(
                root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "-z",
            ).output
            if status:
                raise GitLineageError("candidate worktree is not clean")
            result = self._run_git(
                root,
                "merge",
                "--no-ff",
                "--no-commit",
                "--no-edit",
                upstream,
                allowed_returncodes=frozenset({0, 1}),
            )
            if self._merge_head(root, required=True) != upstream:
                raise GitLineageError("native merge did not retain the exact upstream")
            conflicts = self._conflicted_paths(root)
            if result.returncode != 0 and not conflicts:
                raise GitLineageError("native merge failed without durable conflicts")
            return NativeMerge(
                instance_commit=workspace_base,
                upstream_commit=upstream,
                merge_base_commit=actual_base,
                conflicted_paths=conflicts,
            )

    prepare_native_merge = prepare_merge

    def merge_state(
        self,
        workspace: CandidateWorkspace,
        lineage: UpstreamLineage,
    ) -> NativeMerge:
        """Recover an in-progress merge solely from Git's native durable state."""
        if lineage.upstream_commit is None or lineage.merge_base_commit is None:
            raise GitLineageError("candidate upstream lineage is unavailable")
        with self._mutation():
            root = self._workspace_root(workspace)
            workspace_base = self._resolve_commit(workspace.base_commit)
            upstream = self._resolve_commit(lineage.upstream_commit)
            expected_base = self._resolve_commit(lineage.merge_base_commit)
            if self._resolve_ref(self.instance_ref) != workspace_base:
                raise GitLineageError("candidate is based on a stale instance")
            if self._head(root) != workspace_base:
                raise GitLineageError("candidate HEAD is not its instance base")
            if self._merge_head(root, required=True) != upstream:
                raise GitLineageError("candidate pending upstream is unexpected")
            actual_base = self._merge_base(workspace_base, upstream)
            if actual_base != expected_base:
                raise GitLineageError("candidate merge base is stale")
            return NativeMerge(
                instance_commit=workspace_base,
                upstream_commit=upstream,
                merge_base_commit=actual_base,
                conflicted_paths=self._conflicted_paths(root),
            )

    def inspect_native_merge(self, workspace: CandidateWorkspace) -> NativeMerge | None:
        """Derive an in-progress merge from managed Git state without metadata."""

        with self._mutation():
            root = self._workspace_root(workspace)
            instance = self._resolve_commit(workspace.base_commit)
            if self._resolve_ref(self.instance_ref) != instance:
                raise GitLineageError("candidate is based on a stale instance")
            upstream = self._merge_head(root, required=False)
            if upstream is None:
                return None
            if self._head(root) != instance:
                raise GitLineageError("candidate HEAD is not its instance base")
            merge_base = self._merge_base(instance, upstream)
            return NativeMerge(
                instance_commit=instance,
                upstream_commit=upstream,
                merge_base_commit=merge_base,
                conflicted_paths=self._conflicted_paths(root),
            )

    def stage_resolved_conflicts(
        self,
        workspace: CandidateWorkspace,
        previously_conflicted_paths: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Stage safe resolutions only for paths conflicted before an agent command."""

        with self._mutation():
            root = self._workspace_root(workspace)
            previous = tuple(dict.fromkeys(previously_conflicted_paths))
            current = set(self._conflicted_paths(root))
            if any(path not in current for path in previous):
                raise GitLineageError("candidate conflict index changed unexpectedly")
            for path in previous:
                if not self._resolved_path_is_safe(root, path):
                    continue
                self._run_git(root, "add", "--all", "--", path)
                if path in self._conflicted_paths(root):
                    raise GitLineageError("candidate conflict resolution remained ambiguous")
            return self._conflicted_paths(root)

    def conflicted_paths(self, workspace: CandidateWorkspace) -> tuple[str, ...]:
        with self._mutation():
            return self._conflicted_paths(self._workspace_root(workspace))

    def _conflicted_paths(self, root: Path) -> tuple[str, ...]:
        raw = self._run_git(
            root,
            "diff",
            "--name-only",
            "--diff-filter=U",
            "--no-ext-diff",
            "-z",
            "--",
        ).output
        return tuple(path for path in raw.split("\0") if path)

    def verify_merge_commit(
        self,
        commit: str,
        *,
        instance_commit: str,
        upstream_commit: str,
    ) -> str:
        with self._mutation():
            merged = self._resolve_commit(commit)
            instance = self._resolve_commit(instance_commit)
            upstream = self._resolve_commit(upstream_commit)
            return self._verify_merge_commit(merged, instance, upstream)

    def verify_merged_tip(
        self,
        tip_commit: str,
        *,
        instance_commit: str,
        upstream_commit: str,
        expected_merge_commit: str | None = None,
    ) -> str:
        """Verify that a tip contains the exact instance/upstream merge in its history."""

        with self._mutation():
            tip = self._resolve_commit(tip_commit)
            instance = self._resolve_commit(instance_commit)
            upstream = self._resolve_commit(upstream_commit)
            expected = (
                self._resolve_commit(expected_merge_commit)
                if expected_merge_commit is not None
                else None
            )
            return self._verify_merged_tip(tip, instance, upstream, expected)

    def _verify_merged_tip(
        self,
        tip: str,
        instance: str,
        upstream: str,
        expected_merge: str | None,
    ) -> str:
        if not self._is_ancestor(instance, tip) or not self._is_ancestor(upstream, tip):
            raise GitLineageError("final candidate does not contain both merge parents")
        if expected_merge is not None:
            self._verify_merge_commit(expected_merge, instance, upstream)
            if not self._is_ancestor(expected_merge, tip):
                raise GitLineageError("verified merge is not reachable from final candidate")
            return expected_merge
        commits = self._run_git(
            self._repository,
            "rev-list",
            "--parents",
            "--topo-order",
            f"{instance}..{tip}",
        ).output.splitlines()
        for record in commits:
            fields = record.split()
            if len(fields) == 3 and fields[1:] == [instance, upstream]:
                return self._resolve_commit(fields[0])
        raise GitLineageError("exact instance/upstream merge is not reachable")

    def _verify_merge_commit(self, merged: str, instance: str, upstream: str) -> str:
        parents = self._run_git(
            self._repository,
            "show",
            "--no-ext-diff",
            "--no-patch",
            "--format=%P",
            merged,
        ).output.strip().split()
        if parents != [instance, upstream]:
            raise GitLineageError("merge parent order is not INSTANCE then UPSTREAM")
        return merged

    def _resolved_path_is_safe(self, root: Path, raw_path: str) -> bool:
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
            or not candidate_path_is_promotable(path.as_posix())
        ):
            raise GitLineageError("candidate conflict path is unsafe")
        parent = root
        for component in path.parts[:-1]:
            parent /= component
            try:
                metadata = parent.lstat()
            except FileNotFoundError:
                return True
            except OSError as exc:
                raise GitLineageError("candidate conflict path is unavailable") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise GitLineageError("candidate conflict path is unsafe")
        target = root.joinpath(*path.parts)
        if not os.path.lexists(target):
            return True
        try:
            metadata = target.lstat()
        except OSError as exc:
            raise GitLineageError("candidate conflict resolution is unavailable") from exc
        if not stat.S_ISREG(metadata.st_mode):
            return False
        if metadata.st_size > self._max_git_output_bytes:
            return False
        try:
            content = target.read_bytes()
        except OSError as exc:
            raise GitLineageError("candidate conflict resolution is unavailable") from exc
        return not _CONFLICT_MARKER_RE.search(content) and not candidate_content_contains_secret(
            raw_path,
            content,
        )

    def verify_final_merge(self, workspace: CandidateWorkspace, merge: NativeMerge) -> str:
        with self._mutation():
            root = self._workspace_root(workspace)
            tip = self._head(root)
            self._verify_merged_tip(
                tip,
                self._resolve_commit(merge.instance_commit),
                self._resolve_commit(merge.upstream_commit),
                None,
            )
            return tip

    def _update_refs(
        self,
        *,
        instance_commit: str,
        accepted_upstream_commit: str,
        expected_instance_commit: str | None,
        expected_accepted_upstream_commit: str | None,
        expected_upstream_commit: str,
    ) -> None:
        zero = "0" * self._oid_length
        expected_instance = expected_instance_commit or zero
        expected_accepted = expected_accepted_upstream_commit or zero
        commands = (
            "start\n"
            f"verify {self.upstream_ref} {expected_upstream_commit}\n"
            f"update {self.instance_ref} {instance_commit} {expected_instance}\n"
            f"update {self.accepted_upstream_ref} {accepted_upstream_commit} {expected_accepted}\n"
            "prepare\n"
            "commit\n"
        ).encode("ascii")
        self._run_git(
            self._repository,
            "update-ref",
            "--no-deref",
            "--stdin",
            input_bytes=commands,
        )

    def _resolve_commit(self, value: str) -> str:
        exact = self._object_id(value)
        resolved = self._run_git(
            self._repository,
            "rev-parse",
            "--verify",
            f"{exact}^{{commit}}",
        ).output.strip()
        if self._object_id(resolved) != exact:
            raise GitLineageError("Git commit did not resolve exactly")
        return exact

    def _resolve_ref(self, ref: str) -> str:
        exact_ref = self._ref(ref)
        value = self._run_git(
            self._repository,
            "for-each-ref",
            "--format=%(refname)%00%(objectname)%00%(objecttype)%00%(symref)",
            exact_ref,
        ).output.rstrip("\n")
        fields = value.split("\0")
        if len(fields) != 4 or fields[0] != exact_ref:
            raise GitLineageError("lineage ref did not resolve exactly")
        if fields[3]:
            raise GitLineageError("lineage ref must not be symbolic")
        if fields[2] != "commit":
            raise GitLineageError("lineage ref does not point to a commit")
        return self._object_id(fields[1])

    def _head(self, root: Path) -> str:
        value = self._run_git(root, "rev-parse", "--verify", "HEAD^{commit}").output.strip()
        return self._object_id(value)

    def _merge_head(self, root: Path, *, required: bool) -> str | None:
        result = self._run_git(
            root,
            "rev-parse",
            "--quiet",
            "--verify",
            "MERGE_HEAD^{commit}",
            allowed_returncodes=frozenset({0, 1}),
        )
        if result.returncode != 0:
            if required:
                raise GitLineageError("candidate has no native merge in progress")
            return None
        return self._object_id(result.output.strip())

    def _workspace_root(self, workspace: CandidateWorkspace) -> Path:
        self._object_id(workspace.base_commit)
        candidate_id = str(workspace.candidate_id or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", candidate_id):
            raise GitLineageError("candidate_id is invalid")
        try:
            if workspace.repository_kind == "full_repository":
                root = candidate_repository_directory(
                    workspace.path,
                    candidate_id=candidate_id,
                    base_commit=workspace.base_commit,
                    worktrees_root=self._worktrees_root,
                )
            else:
                root, _ = candidate_worktree_directories(
                    workspace.path,
                    candidate_id=candidate_id,
                    base_commit=workspace.base_commit,
                    worktrees_root=self._worktrees_root,
                    common_directory=self._git_common_directory,
                )
        except (GitSecurityError, GitCandidateError):
            raise GitLineageError("candidate is not a managed detached worktree") from None
        self._assert_safe_git_configuration(root)
        symbolic = self._run_git(
            root,
            "symbolic-ref",
            "--quiet",
            "HEAD",
            allowed_returncodes=frozenset({0, 1}),
        ).output.strip()
        if symbolic:
            raise GitLineageError("candidate worktree must remain detached")
        return root

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
                        try:
                            self._assert_safe_git_configuration(self._repository)
                        except Exception:
                            del self._operation_state.validated_directories
                            raise
                    self._operation_state.depth = depth + 1
                    try:
                        yield
                    finally:
                        if depth == 0:
                            del self._operation_state.validated_directories
                            del self._operation_state.depth
                        else:
                            self._operation_state.depth = depth
            except RepositoryMutationLockError:
                raise GitLineageError("Git lineage mutation lock failed") from None

    def _assert_safe_git_configuration(self, cwd: Path) -> None:
        try:
            git_directory, common_directory = discover_git_directories(cwd)
        except GitSecurityError:
            raise GitLineageError("Git repository metadata is unsafe") from None
        independent = common_directory == git_directory and cwd != self._repository
        if common_directory != self._git_common_directory and not independent:
            raise GitLineageError("candidate belongs to another repository")
        key = (git_directory, common_directory)
        validated = getattr(self._operation_state, "validated_directories", None)
        if validated is not None and key in validated:
            return
        if repository_git_configuration_is_unsafe(
            cwd,
            git_directory,
            self._git_common_directory,
            timeout_seconds=self._timeout_seconds,
            max_output_bytes=self._max_git_output_bytes,
        ):
            raise GitLineageError("repository Git configuration is unsafe")
        if validated is not None:
            validated.add(key)

    def _run_git(
        self,
        cwd: Path,
        *arguments: str,
        allowed_returncodes: frozenset[int] = frozenset({0}),
        input_bytes: bytes | None = None,
        allow_https: bool = False,
    ) -> _GitResult:
        if getattr(self._operation_state, "depth", 0) <= 0:
            raise GitLineageError("Git lineage operation lacks a security context")
        try:
            result = run_hardened_git(
                cwd,
                arguments,
                timeout_seconds=self._timeout_seconds,
                max_output_bytes=self._max_git_output_bytes,
                input_bytes=input_bytes,
                env=_GIT_IDENTITY_ENV,
                allow_https=allow_https,
            )
        except OSError:
            raise GitLineageError("Git lineage operation failed") from None
        if result.returncode not in allowed_returncodes or result.truncated or result.timed_out:
            raise GitLineageError("Git lineage operation failed")
        return _GitResult(
            returncode=result.returncode,
            output=result.output.decode("utf-8", errors="replace"),
        )

    def _object_id(self, value: str) -> str:
        exact = str(value or "").strip().lower()
        if not _OID_RE.fullmatch(exact) or len(exact) != self._oid_length:
            raise GitLineageError("Git object ID is invalid")
        return exact

    @staticmethod
    def _ref(value: str) -> str:
        ref = str(value or "").strip()
        components = ref.split("/")
        if (
            not ref.startswith("refs/")
            or len(ref) > 500
            or any(not part or part.startswith(".") or part.endswith((".", ".lock")) for part in components)
            or ".." in ref
            or "@{" in ref
            or any(character in ref for character in " ~^:?*[\\")
            or any(ord(character) < 32 or ord(character) == 127 for character in ref)
        ):
            raise ValueError("Git lineage ref is invalid")
        return ref

    @staticmethod
    def _remote_branch_ref(value: str) -> str:
        ref = GitLineage._ref(value)
        if not ref.startswith("refs/heads/"):
            raise ValueError("remote upstream ref must be a branch")
        return ref

    @staticmethod
    def _https_repository_url(value: str) -> str:
        url = str(value or "").strip()
        parsed = urlsplit(url)
        if (
            len(url) > 2_000
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "\\" in url
            or any(ord(character) < 32 or ord(character) == 127 for character in url)
        ):
            raise ValueError("upstream repository must be an unauthenticated HTTPS URL")
        return url

    @staticmethod
    def _regular_directory(value: str | Path) -> Path:
        raw = Path(value).expanduser()
        if raw.is_symlink():
            raise ValueError("Git lineage directories cannot be symlinks")
        try:
            path = raw.resolve(strict=True)
        except OSError:
            raise ValueError("Git lineage directory is unavailable") from None
        if not path.is_dir():
            raise ValueError("Git lineage directory must be a directory")
        return path

    @staticmethod
    def _is_relative_to(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True


InstanceGitLineage = GitLineage
