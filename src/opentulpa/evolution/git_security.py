"""POSIX-only Git execution, repository validation, and mutation locking."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import re
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from pathlib import Path
from urllib.parse import urlsplit

from opentulpa.evolution.process import BoundedProcessResult, run_bounded_process

_TRUSTED_GIT_CONFIG = (
    f"core.hooksPath={os.devnull}",
    "core.fsmonitor=false",
    "core.untrackedCache=false",
    f"core.attributesFile={os.devnull}",
    f"core.excludesFile={os.devnull}",
    "diff.external=",
    "credential.helper=",
    "protocol.allow=never",
    "protocol.file.allow=never",
    "maintenance.auto=false",
    "gc.auto=0",
    "submodule.recurse=false",
    "fetch.recurseSubmodules=false",
)
_SAFE_REPOSITORY_CONFIG = frozenset(
    {
        "core.bare",
        "core.filemode",
        "core.ignorecase",
        "core.logallrefupdates",
        "core.precomposeunicode",
        "core.repositoryformatversion",
        "core.symlinks",
        "extensions.compatobjectformat",
        "extensions.objectformat",
        "user.email",
        "user.name",
    }
)
_REPOSITORY_LOCKS_GUARD = threading.Lock()
_REPOSITORY_LOCKS: dict[Path, threading.RLock] = {}
_REPOSITORY_LOCK_DEPTH = threading.local()
_REPOSITORY_LOCK_PID = os.getpid()
_OPEN_LOCK_DESCRIPTORS: set[int] = set()
_FORK_DESCRIPTOR_GUARD = threading.Lock()
_REPOSITORY_SECURITY_CACHE_GUARD = threading.Lock()
_REPOSITORY_SECURITY_CACHE: dict[
    tuple[Path, Path],
    tuple[
        tuple[str, tuple[int, int, int, int, int, int] | None, str | None],
        ...,
    ],
] = {}
_MAX_REPOSITORY_SECURITY_CACHE_ENTRIES = 1_024
_REMOTE_CONFIG_RE = re.compile(r"remote\.([a-z0-9][a-z0-9._/-]{0,199})\.(url|fetch)\Z")
_BRANCH_CONFIG_RE = re.compile(r"branch\.([a-z0-9][a-z0-9._/-]{0,199})\.(remote|merge)\Z")
_SCP_REMOTE_RE = re.compile(
    r"[A-Za-z0-9._-]+@[A-Za-z0-9.-]+:[A-Za-z0-9._~!$&'()+,;=:@%/-]+\Z"
)


class GitSecurityError(RuntimeError):
    """Git trust-boundary validation failed without exposing a host path."""


class RepositoryMutationLockError(GitSecurityError):
    """A repository mutation lock could not be acquired safely."""


def _reset_repository_locks_after_fork() -> None:
    global _REPOSITORY_LOCK_DEPTH, _REPOSITORY_LOCK_PID, _REPOSITORY_LOCKS_GUARD
    global _REPOSITORY_SECURITY_CACHE_GUARD

    for descriptor in tuple(_OPEN_LOCK_DESCRIPTORS):
        with suppress(OSError):
            os.close(descriptor)
    _OPEN_LOCK_DESCRIPTORS.clear()
    _REPOSITORY_LOCKS.clear()
    _REPOSITORY_LOCKS_GUARD = threading.Lock()
    _REPOSITORY_LOCK_DEPTH = threading.local()
    _REPOSITORY_LOCK_PID = os.getpid()
    _REPOSITORY_SECURITY_CACHE.clear()
    _REPOSITORY_SECURITY_CACHE_GUARD = threading.Lock()


def _before_fork() -> None:
    _FORK_DESCRIPTOR_GUARD.acquire()


def _after_fork_parent() -> None:
    _FORK_DESCRIPTOR_GUARD.release()


def _after_fork_child() -> None:
    try:
        _reset_repository_locks_after_fork()
    finally:
        _FORK_DESCRIPTOR_GUARD.release()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_parent,
    after_in_child=_after_fork_child,
)


@contextmanager
def repository_mutation_lock(
    git_common_directory: str | Path,
    *,
    timeout_seconds: float = 30,
) -> Iterator[None]:
    """Serialize repository mutations across threads and POSIX processes."""

    if timeout_seconds <= 0 or timeout_seconds > 3_600:
        raise ValueError("repository lock timeout must be between 0 and 3600 seconds")
    if os.getpid() != _REPOSITORY_LOCK_PID:
        _reset_repository_locks_after_fork()
    raw_directory = Path(git_common_directory).expanduser()
    if raw_directory.is_symlink():
        raise RepositoryMutationLockError("Git common directory cannot be a symlink")
    try:
        common_directory = raw_directory.resolve(strict=True)
    except OSError:
        raise RepositoryMutationLockError("Git common directory is unavailable") from None
    if not common_directory.is_dir():
        raise RepositoryMutationLockError("Git common directory is unavailable")
    lock_path = common_directory / "opentulpa-mutation.lock"
    with _REPOSITORY_LOCKS_GUARD:
        thread_lock = _REPOSITORY_LOCKS.setdefault(lock_path, threading.RLock())
    deadline = time.monotonic() + timeout_seconds
    if not thread_lock.acquire(timeout=timeout_seconds):
        raise RepositoryMutationLockError("repository mutation lock timed out")
    try:
        depths = getattr(_REPOSITORY_LOCK_DEPTH, "depths", None)
        if depths is None:
            depths = {}
            _REPOSITORY_LOCK_DEPTH.depths = depths
        identity = (os.getpid(), threading.get_ident())
        owner, depth = depths.get(lock_path, (None, 0))
        if owner == identity and depth:
            depths[lock_path] = (identity, depth + 1)
            try:
                yield
            finally:
                depths[lock_path] = (identity, depth)
            return

        flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            with _FORK_DESCRIPTOR_GUARD:
                descriptor = os.open(lock_path, flags, 0o600)
                _OPEN_LOCK_DESCRIPTORS.add(descriptor)
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or metadata.st_nlink != 1
            ):
                raise OSError("unsafe lock file")
            os.fchmod(descriptor, 0o600)
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise RepositoryMutationLockError(
                            "repository mutation lock timed out"
                        ) from None
                    time.sleep(min(0.05, remaining))
        except RepositoryMutationLockError:
            if descriptor >= 0:
                with _FORK_DESCRIPTOR_GUARD:
                    _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
                    os.close(descriptor)
            raise
        except OSError:
            if descriptor >= 0:
                with _FORK_DESCRIPTOR_GUARD:
                    _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
                    os.close(descriptor)
            raise RepositoryMutationLockError("repository mutation lock failed") from None
        depths[lock_path] = (identity, 1)
        try:
            yield
        finally:
            depths.pop(lock_path, None)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                with _FORK_DESCRIPTOR_GUARD:
                    _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
                    os.close(descriptor)
    finally:
        thread_lock.release()


def _trusted_git_environment(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = {
        "PATH": os.environ.get("PATH", os.defpath),
        "HOME": "/tmp",
        "XDG_CONFIG_HOME": "/tmp/opentulpa-git-config",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": os.devnull,
        "SSH_ASKPASS": os.devnull,
        "GIT_MERGE_AUTOEDIT": "no",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }
    for key in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_DATE",
    ):
        if overrides is not None and key in overrides:
            environment[key] = overrides[key]
    return environment


def validate_git_executable(value: str | Path) -> str:
    executable = str(value)
    if (
        not executable
        or "\x00" in executable
        or any(ord(character) < 32 or ord(character) == 127 for character in executable)
        or Path(executable).name != "git"
    ):
        raise ValueError("Git executable is invalid")
    if executable == "git":
        return executable
    raw = Path(executable)
    if not raw.is_absolute():
        raise ValueError("configured Git executable must be an absolute regular file")
    try:
        resolved = raw.resolve(strict=True)
    except OSError:
        raise ValueError("configured Git executable is unavailable") from None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("configured Git executable is unavailable")
    return str(resolved)


def _trusted_git_command(
    cwd: Path,
    arguments: tuple[str, ...],
    *,
    git_executable: str | Path,
    allow_https: bool = False,
) -> list[str]:
    executable = validate_git_executable(git_executable)
    command = [executable, "--no-replace-objects", "--literal-pathspecs", "-C", str(cwd)]
    for setting in _TRUSTED_GIT_CONFIG:
        command.extend(("-c", setting))
    if allow_https:
        command.extend(("-c", "protocol.https.allow=always"))
    command.extend(arguments)
    return command


def _run_bounded_process_with_input(
    argv: list[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_seconds: int,
    max_output_bytes: int,
    input_bytes: bytes,
) -> BoundedProcessResult:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    stream = process.stdout
    input_stream = process.stdin
    assert stream is not None and input_stream is not None
    retained = bytearray()
    output_size = 0

    def drain() -> None:
        nonlocal output_size
        while chunk := stream.read(64 * 1024):
            output_size += len(chunk)
            remaining = max_output_bytes - len(retained)
            if remaining > 0:
                retained.extend(chunk[:remaining])

    reader = threading.Thread(target=drain, name="opentulpa-git-output", daemon=True)
    reader.start()
    timed_out = False
    try:
        try:
            input_stream.write(input_bytes)
        except BrokenPipeError:
            pass
        finally:
            input_stream.close()
        deadline = time.monotonic() + timeout_seconds
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                break
            try:
                process.wait(timeout=min(0.1, remaining))
            except subprocess.TimeoutExpired:
                continue
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    finally:
        reader.join(timeout=5)
        stream.close()
        if not input_stream.closed:
            input_stream.close()
    return BoundedProcessResult(
        returncode=124 if timed_out else int(process.returncode),
        output=bytes(retained),
        truncated=output_size > len(retained),
        timed_out=timed_out,
    )


def run_hardened_git(
    cwd: Path,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: int,
    max_output_bytes: int,
    input_bytes: bytes | None = None,
    env: Mapping[str, str] | None = None,
    git_executable: str | Path = "git",
    allow_https: bool = False,
) -> BoundedProcessResult:
    command = _trusted_git_command(
        cwd,
        arguments,
        git_executable=git_executable,
        allow_https=allow_https,
    )
    environment = _trusted_git_environment(env)
    if input_bytes is not None:
        return _run_bounded_process_with_input(
            command,
            cwd=cwd,
            env=environment,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            input_bytes=input_bytes,
        )
    return run_bounded_process(
        command,
        cwd=cwd,
        env=environment,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )


run_trusted_git_process = run_hardened_git


def read_git_admin_file(path: Path) -> str:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4_096:
            raise OSError("unsafe Git administrative file")
        value = path.read_bytes().decode("utf-8", errors="strict")
    except (OSError, UnicodeError):
        raise GitSecurityError("Git repository metadata is unsafe") from None
    lines = value.splitlines()
    if len(lines) != 1 or not lines[0] or "\x00" in lines[0]:
        raise GitSecurityError("Git repository metadata is unsafe")
    return lines[0]


def discover_git_directories(repository: Path) -> tuple[Path, Path]:
    marker = repository / ".git"
    try:
        metadata = marker.lstat()
    except OSError:
        raise GitSecurityError("Git repository metadata is unavailable") from None
    if stat.S_ISDIR(metadata.st_mode):
        try:
            git_directory = marker.resolve(strict=True)
        except OSError:
            raise GitSecurityError("Git repository metadata is unavailable") from None
    elif stat.S_ISREG(metadata.st_mode):
        value = read_git_admin_file(marker)
        if not value.startswith("gitdir: "):
            raise GitSecurityError("Git repository metadata is unsafe")
        raw_directory = Path(value.removeprefix("gitdir: "))
        if not raw_directory.is_absolute():
            raw_directory = repository / raw_directory
        if raw_directory.is_symlink():
            raise GitSecurityError("Git repository metadata is unsafe")
        try:
            git_directory = raw_directory.resolve(strict=True)
        except OSError:
            raise GitSecurityError("Git repository metadata is unavailable") from None
    else:
        raise GitSecurityError("Git repository metadata is unsafe")
    if git_directory.is_symlink() or not git_directory.is_dir():
        raise GitSecurityError("Git repository metadata is unsafe")

    commondir = git_directory / "commondir"
    if os.path.lexists(commondir):
        value = read_git_admin_file(commondir)
        raw_common = Path(value)
        if not raw_common.is_absolute():
            raw_common = git_directory / raw_common
        if raw_common.is_symlink():
            raise GitSecurityError("Git repository metadata is unsafe")
        try:
            common_directory = raw_common.resolve(strict=True)
        except OSError:
            raise GitSecurityError("Git repository metadata is unavailable") from None
    else:
        common_directory = git_directory
    if common_directory.is_symlink() or not common_directory.is_dir():
        raise GitSecurityError("Git repository metadata is unsafe")
    return git_directory, common_directory


def candidate_worktree_directories(
    path: Path,
    *,
    candidate_id: str,
    base_commit: str | None,
    worktrees_root: Path,
    common_directory: Path,
    require_registration: bool = True,
) -> tuple[Path, Path]:
    try:
        expected = (worktrees_root / candidate_id).resolve(strict=False)
    except OSError:
        raise GitSecurityError("candidate worktree path is unsafe") from None
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise GitSecurityError("candidate worktree is unavailable")
    try:
        root = raw.resolve(strict=True)
    except OSError:
        raise GitSecurityError("candidate worktree is unavailable") from None
    try:
        root.relative_to(worktrees_root)
    except ValueError:
        raise GitSecurityError("candidate worktree escaped its configured root") from None
    if root != expected or root == worktrees_root:
        raise GitSecurityError("candidate worktree identity does not match its path")
    git_directory, discovered_common = discover_git_directories(root)
    if (
        discovered_common != common_directory
        or git_directory == common_directory
        or git_directory.parent != common_directory / "worktrees"
    ):
        raise GitSecurityError("candidate is not a managed detached worktree")
    value = read_git_admin_file(git_directory / "gitdir")
    raw_marker = Path(value)
    if not raw_marker.is_absolute():
        raw_marker = git_directory / raw_marker
    try:
        registered_marker = raw_marker.resolve(strict=True)
    except OSError:
        raise GitSecurityError("candidate worktree registration is invalid") from None
    try:
        expected_marker = (root / ".git").resolve(strict=True)
    except OSError:
        raise GitSecurityError("candidate worktree registration is invalid") from None
    if registered_marker != expected_marker:
        raise GitSecurityError("candidate worktree registration is invalid")
    registration = git_directory / "opentulpa-candidate"
    expected_registration = (
        f"{candidate_id} {base_commit}" if base_commit is not None else candidate_id
    )
    if require_registration and read_git_admin_file(registration) != expected_registration:
        raise GitSecurityError("candidate worktree registration is invalid")
    return root, git_directory


def _config_entries(
    cwd: Path,
    config_path: Path,
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> tuple[tuple[str, str], ...] | None:
    try:
        metadata = config_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            return None
        result = run_trusted_git_process(
            cwd,
            (
                "config",
                "--file",
                str(config_path),
                "--no-includes",
                "--null",
                "--list",
            ),
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
        if result.returncode != 0 or result.truncated or result.timed_out:
            return None
        records = tuple(record for record in result.output.split(b"\0") if record)
        entries = []
        for record in records:
            raw_name, separator, raw_value = record.partition(b"\n")
            if not separator:
                return None
            entries.append(
                (
                    raw_name.decode("utf-8", errors="strict").casefold(),
                    raw_value.decode("utf-8", errors="strict"),
                )
            )
    except (OSError, UnicodeError):
        return None
    return tuple(entries)


def _repository_config_is_safe(entries: tuple[tuple[str, str], ...]) -> bool:
    booleans = {
        "core.filemode",
        "core.ignorecase",
        "core.logallrefupdates",
        "core.precomposeunicode",
        "core.symlinks",
    }
    seen: set[str] = set()
    for name, value in entries:
        lowered = value.casefold()
        remote_match = _REMOTE_CONFIG_RE.fullmatch(name)
        branch_match = _BRANCH_CONFIG_RE.fullmatch(name)
        repeatable = remote_match is not None and remote_match.group(2) in {"fetch", "url"}
        if name in seen and not repeatable:
            return False
        seen.add(name)
        if "\x00" in value:
            return False
        if remote_match is not None:
            field = remote_match.group(2)
            if field == "url" and not _remote_url_is_safe(value):
                return False
            if field == "fetch" and not _fetch_refspec_is_safe(value):
                return False
            continue
        if branch_match is not None:
            field = branch_match.group(2)
            if field == "remote" and not _remote_name_is_safe(value):
                return False
            if field == "merge" and not _branch_merge_ref_is_safe(value):
                return False
            continue
        if name not in _SAFE_REPOSITORY_CONFIG:
            return False
        if name in booleans and lowered not in {"true", "false"}:
            return False
        if name == "core.bare" and lowered != "false":
            return False
        if name == "core.repositoryformatversion" and value not in {"0", "1"}:
            return False
        if name in {"extensions.objectformat", "extensions.compatobjectformat"} and lowered not in {
            "sha1",
            "sha256",
        }:
            return False
        if name in {"user.email", "user.name"} and (
            not value or len(value) > 500 or any(ord(character) < 32 for character in value)
        ):
            return False
    return True


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_ino,
        metadata.st_dev,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
        metadata.st_mode,
    )


def _fingerprint_admin_path(
    path: Path,
    *,
    recursive: bool = False,
) -> tuple[
    tuple[str, tuple[int, int, int, int, int, int] | None, str | None], ...
]:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return ((str(path), None, None),)
    identity = _stat_identity(metadata)
    digest: str | None = None
    if stat.S_ISREG(metadata.st_mode):
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = -1
        try:
            descriptor = os.open(path, flags)
            if _stat_identity(os.fstat(descriptor)) != identity:
                raise OSError("Git administrative file changed during inspection")
            hasher = hashlib.sha256()
            while chunk := os.read(descriptor, 64 * 1024):
                hasher.update(chunk)
            if _stat_identity(os.fstat(descriptor)) != identity:
                raise OSError("Git administrative file changed during inspection")
            digest = hasher.hexdigest()
        except OSError:
            raise GitSecurityError("Git repository metadata is unsafe") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    records: list[
        tuple[str, tuple[int, int, int, int, int, int] | None, str | None]
    ] = [
        (str(path), identity, digest)
    ]
    if not recursive or not stat.S_ISDIR(metadata.st_mode):
        return tuple(records)
    try:
        children = sorted(path.iterdir(), key=lambda child: child.name)
    except OSError:
        raise GitSecurityError("Git repository metadata is unsafe") from None
    for child in children:
        records.extend(_fingerprint_admin_path(child, recursive=True))
    return tuple(records)


def _repository_security_fingerprint(
    git_directory: Path,
    common_directory: Path,
) -> tuple[
    tuple[str, tuple[int, int, int, int, int, int] | None, str | None], ...
]:
    static_paths = (
        common_directory / "config",
        common_directory / "config.worktree",
        git_directory / "config.worktree",
        common_directory / "shallow",
        git_directory / "shallow",
        common_directory / "info" / "grafts",
        git_directory / "info" / "grafts",
        common_directory / "objects" / "info" / "alternates",
        common_directory / "objects" / "info" / "http-alternates",
        common_directory / "info" / "attributes",
        git_directory / "info" / "attributes",
        common_directory / "packed-refs",
    )
    records: list[
        tuple[str, tuple[int, int, int, int, int, int] | None, str | None]
    ] = []
    for path in dict.fromkeys(static_paths):
        records.extend(_fingerprint_admin_path(path))
    replace_roots = (
        common_directory / "refs" / "replace",
        git_directory / "refs" / "replace",
    )
    for path in dict.fromkeys(replace_roots):
        records.extend(_fingerprint_admin_path(path, recursive=True))
    return tuple(records)


def _remote_name_is_safe(value: str) -> bool:
    return value == "." or bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", value)
        and ".." not in value
        and not value.endswith((".", "/"))
    )


def _branch_merge_ref_is_safe(value: str) -> bool:
    return bool(
        re.fullmatch(r"refs/heads/[A-Za-z0-9._/-]+", value)
        and len(value) <= 500
        and ".." not in value
        and "@{" not in value
        and not value.endswith((".", "/"))
        and not any(character in value for character in " ~^:?*[\\")
        and all(31 < ord(character) < 127 for character in value)
    )


def _fetch_refspec_is_safe(value: str) -> bool:
    refspec = value.removeprefix("+")
    if refspec.count(":") != 1:
        return False
    source, destination = refspec.split(":", 1)
    if source.count("*") != destination.count("*") or source.count("*") > 1:
        return False
    return bool(
        re.fullmatch(r"refs/[A-Za-z0-9._/*-]+", source)
        and re.fullmatch(r"refs/remotes/[A-Za-z0-9._/*-]+", destination)
        and ".." not in refspec
        and "@{" not in refspec
        and not source.endswith((".", "/"))
        and not destination.endswith((".", "/"))
    )


def _remote_url_is_safe(value: str) -> bool:
    if (
        not value
        or len(value) > 2_000
        or value.startswith(("-", "~", "!", "|"))
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    if _SCP_REMOTE_RE.fullmatch(value):
        return True
    parsed = urlsplit(value)
    if not parsed.scheme:
        return "://" not in value and "::" not in value and "\\" not in value
    scheme = parsed.scheme.casefold()
    if scheme not in {"https", "ssh"} or not parsed.hostname or parsed.password is not None:
        return False
    if parsed.query or parsed.fragment:
        return False
    if scheme == "https" and parsed.username is not None:
        return False
    return parsed.username is None or bool(re.fullmatch(r"[A-Za-z0-9._-]+", parsed.username))


def repository_git_configuration_is_unsafe(
    cwd: Path,
    git_directory: Path,
    common_directory: Path,
    *,
    timeout_seconds: int,
    max_output_bytes: int,
) -> bool:
    cache_key = (git_directory, common_directory)
    try:
        fingerprint = _repository_security_fingerprint(git_directory, common_directory)
    except (GitSecurityError, OSError):
        return True
    with _REPOSITORY_SECURITY_CACHE_GUARD:
        if _REPOSITORY_SECURITY_CACHE.get(cache_key) == fingerprint:
            return False
    unsafe_state = (
        common_directory / "shallow",
        common_directory / "info" / "grafts",
        common_directory / "objects" / "info" / "alternates",
        common_directory / "objects" / "info" / "http-alternates",
        git_directory / "shallow",
        git_directory / "info" / "grafts",
        common_directory / "config.worktree",
        git_directory / "config.worktree",
    )
    if any(os.path.lexists(path) for path in unsafe_state):
        return True
    entries = _config_entries(
        cwd,
        common_directory / "config",
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
    if entries is None or not _repository_config_is_safe(entries):
        return True
    try:
        validated_fingerprint = _repository_security_fingerprint(
            git_directory,
            common_directory,
        )
    except (GitSecurityError, OSError):
        return True
    if validated_fingerprint != fingerprint:
        return True
    with _REPOSITORY_SECURITY_CACHE_GUARD:
        if (
            cache_key not in _REPOSITORY_SECURITY_CACHE
            and len(_REPOSITORY_SECURITY_CACHE) >= _MAX_REPOSITORY_SECURITY_CACHE_ENTRIES
        ):
            del _REPOSITORY_SECURITY_CACHE[next(iter(_REPOSITORY_SECURITY_CACHE))]
        _REPOSITORY_SECURITY_CACHE[cache_key] = validated_fingerprint
    return False
