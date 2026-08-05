"""Locked integrity verification for immutable Python generations.

Hashes detect storage tampering. Linux child-UID separation remains the strong
boundary that prevents a candidate runtime from rewriting controller-owned files.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform as platform_module
import re
import shutil
import stat
import sys
import sysconfig
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from uuid import uuid4

from pydantic import ValidationError

from opentulpa.evolution.generation import GenerationManifest, canonical_json_bytes

_GENERATION_ID_RE = re.compile(r"[0-9a-f]{64}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MANIFEST_NAME = "manifest.json"
_COMPLETE_NAME = "COMPLETE"
_BUILDING_NAME = "BUILDING"
_RUNTIME_HASH_EXCLUSIONS = frozenset({_BUILDING_NAME, _MANIFEST_NAME, _COMPLETE_NAME})
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[Path, threading.RLock] = {}
_LOCK_DEPTH = threading.local()
_LOCK_PID = os.getpid()
_OPEN_LOCK_DESCRIPTORS: set[int] = set()


def _reset_locks_after_fork() -> None:
    global _LOCK_DEPTH, _LOCK_PID, _LOCKS_GUARD
    for descriptor in tuple(_OPEN_LOCK_DESCRIPTORS):
        with suppress(OSError):
            os.close(descriptor)
    _OPEN_LOCK_DESCRIPTORS.clear()
    _LOCKS.clear()
    _LOCKS_GUARD = threading.Lock()
    _LOCK_DEPTH = threading.local()
    _LOCK_PID = os.getpid()


os.register_at_fork(after_in_child=_reset_locks_after_fork)


class GenerationStoreError(RuntimeError):
    """A generation failed immutable-store verification."""


@dataclass(frozen=True, slots=True)
class InstalledGeneration:
    """A fully verified generation with final executable paths."""

    generation_id: str
    path: Path
    manifest: GenerationManifest
    manifest_digest: str
    interpreter_path: Path
    entrypoint_path: Path

    @property
    def interpreter(self) -> Path:
        return self.interpreter_path

    @property
    def entrypoint(self) -> Path:
        return self.entrypoint_path

    @property
    def entrypoint_argv(self) -> tuple[str, ...]:
        return (str(self.entrypoint_path), *self.manifest.identity.entrypoint[1:])


def runtime_tree_sha256(root: Path, *, require_read_only: bool = False) -> str:
    """Hash every descendant directory and regular file, including empty directories."""

    digest = hashlib.sha256()
    _hash_runtime_directory(
        root,
        root,
        digest,
        require_read_only=require_read_only,
        is_root=True,
    )
    return digest.hexdigest()


def _hash_runtime_directory(
    root: Path,
    directory: Path,
    digest: hashlib._Hash,
    *,
    require_read_only: bool,
    is_root: bool,
) -> None:
    try:
        directory_metadata = directory.lstat()
    except OSError as exc:
        raise GenerationStoreError("runtime directory is unavailable") from exc
    if stat.S_ISLNK(directory_metadata.st_mode) or not stat.S_ISDIR(directory_metadata.st_mode):
        raise GenerationStoreError("runtime tree contains a link or special directory")
    if directory_metadata.st_uid != os.geteuid():
        raise GenerationStoreError("runtime tree is not controller-owned")
    if require_read_only and stat.S_IMODE(directory_metadata.st_mode) != 0o555:
        raise GenerationStoreError("complete generation directory mode is not 0555")
    if not is_root:
        try:
            relative = directory.relative_to(root).as_posix().encode("utf-8")
        except (UnicodeEncodeError, ValueError) as exc:
            raise GenerationStoreError("runtime directory path is invalid") from exc
        digest.update(b"directory\0")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(f"{stat.S_IMODE(directory_metadata.st_mode):04o}".encode("ascii"))
        digest.update(b"\0")
    try:
        entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
    except OSError as exc:
        raise GenerationStoreError("runtime tree could not be enumerated") from exc
    for entry in entries:
        if is_root and entry.name in _RUNTIME_HASH_EXCLUSIONS:
            continue
        path = Path(entry.path)
        try:
            metadata = entry.stat(follow_symlinks=False)
        except OSError as exc:
            raise GenerationStoreError("runtime tree entry could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise GenerationStoreError("runtime tree contains a symbolic link")
        if stat.S_ISDIR(metadata.st_mode):
            _hash_runtime_directory(
                root,
                path,
                digest,
                require_read_only=require_read_only,
                is_root=False,
            )
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise GenerationStoreError("runtime tree contains a special file")
        if metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise GenerationStoreError("runtime file ownership or link count is unsafe")
        mode = stat.S_IMODE(metadata.st_mode)
        if require_read_only and mode not in {0o444, 0o555}:
            raise GenerationStoreError("complete generation file mode is not published read-only")
        try:
            relative = path.relative_to(root).as_posix().encode("utf-8")
        except (UnicodeEncodeError, ValueError) as exc:
            raise GenerationStoreError("runtime tree path is invalid") from exc
        digest.update(b"file\0")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(f"{mode:04o}".encode("ascii"))
        digest.update(b"\0")
        digest.update(str(metadata.st_size).encode("ascii"))
        digest.update(b"\0")
        with _open_regular_file(path, label="runtime file") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")


class GenerationStore:
    """Open generations under a dedicated, non-listable ``0711`` runtime root.

    Composition must keep this root outside private bootstrap state. Control files
    remain controller-only while children can traverse known generation paths.
    """

    def __init__(
        self,
        generations_root: str | Path,
        *,
        control_root: str | Path | None = None,
        quarantine_root: str | Path | None = None,
        max_manifest_bytes: int = 1024 * 1024,
    ) -> None:
        if max_manifest_bytes < 1024:
            raise ValueError("generation manifest limit is too small")
        self._root = _secure_directory(
            generations_root,
            create=True,
            label="generations root",
            required_mode=0o711,
            allow_root_owner=True,
        )
        raw_control = (
            Path(control_root).expanduser()
            if control_root is not None
            else self._root.parent / f".{self._root.name}-control"
        )
        self._control_root = _secure_directory(
            raw_control,
            create=True,
            label="generation control root",
            required_mode=0o700,
            allow_root_owner=False,
        )
        if self._control_root == self._root or _is_relative_to(self._control_root, self._root):
            raise ValueError("generation control root must be outside the runtime store")
        raw_quarantine = (
            Path(quarantine_root).expanduser()
            if quarantine_root is not None
            else self._control_root / "quarantine"
        )
        self._quarantine_root = _absolute_without_symlinks(
            raw_quarantine,
            label="generation quarantine root",
        )
        if self._quarantine_root == self._root or _is_relative_to(
            self._quarantine_root, self._root
        ):
            raise ValueError("generation quarantine must be outside the runtime store")
        self._max_manifest_bytes = max_manifest_bytes
        self._lock_path = self._control_root / "generation-store.lock"

    @property
    def root(self) -> Path:
        return self._root

    @property
    def quarantine_root(self) -> Path:
        return self._quarantine_root

    @property
    def control_root(self) -> Path:
        return self._control_root

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Serialize builders, verification, cleanup, and publication."""

        if os.getpid() != _LOCK_PID:
            _reset_locks_after_fork()
        with _LOCKS_GUARD:
            thread_lock = _LOCKS.setdefault(self._lock_path, threading.RLock())
        with thread_lock:
            depths = getattr(_LOCK_DEPTH, "depths", None)
            if depths is None:
                depths = {}
                _LOCK_DEPTH.depths = depths
            depth = depths.get(self._lock_path, 0)
            if depth:
                depths[self._lock_path] = depth + 1
                try:
                    yield
                finally:
                    depths[self._lock_path] -= 1
                return
            descriptor: int | None = None
            try:
                flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(self._lock_path, flags, 0o600)
                metadata = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or stat.S_IMODE(metadata.st_mode) & 0o077
                    or metadata.st_uid != os.geteuid()
                    or metadata.st_nlink != 1
                ):
                    raise GenerationStoreError("generation store lock is unsafe")
                _OPEN_LOCK_DESCRIPTORS.add(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._verify_root()
                depths[self._lock_path] = 1
                yield
            except OSError as exc:
                raise GenerationStoreError("generation store lock failed") from exc
            finally:
                depths.pop(self._lock_path, None)
                if descriptor is not None:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    finally:
                        _OPEN_LOCK_DESCRIPTORS.discard(descriptor)
                        os.close(descriptor)

    def open(
        self,
        generation_id: str,
        *,
        expected_manifest_digest: str,
        expected_state_contract_digest: str,
        expected_evaluator_fingerprint: str,
        expected_install_profile: str,
        controller_protocol: int,
    ) -> InstalledGeneration:
        """Open a generation only when controller-held provenance matches."""

        if not _DIGEST_RE.fullmatch(expected_manifest_digest):
            raise GenerationStoreError("expected manifest digest is invalid")
        if not _SHA256_RE.fullmatch(expected_state_contract_digest):
            raise GenerationStoreError("expected state contract digest is invalid")
        if not _DIGEST_RE.fullmatch(expected_evaluator_fingerprint):
            raise GenerationStoreError("expected evaluator fingerprint is invalid")
        if not expected_install_profile or controller_protocol < 1:
            raise GenerationStoreError("expected generation policy is invalid")
        with self.locked():
            return self._open_locked(
                generation_id,
                expected_manifest_digest=expected_manifest_digest,
                expected_state_contract_digest=expected_state_contract_digest,
                expected_evaluator_fingerprint=expected_evaluator_fingerprint,
                expected_install_profile=expected_install_profile,
                controller_protocol=controller_protocol,
            )

    def verify(
        self,
        generation_id: str,
        *,
        expected_manifest_digest: str,
        expected_state_contract_digest: str,
        expected_evaluator_fingerprint: str,
        expected_install_profile: str,
        controller_protocol: int,
    ) -> InstalledGeneration:
        """Alias production verification to the strict open path."""

        return self.open(
            generation_id,
            expected_manifest_digest=expected_manifest_digest,
            expected_state_contract_digest=expected_state_contract_digest,
            expected_evaluator_fingerprint=expected_evaluator_fingerprint,
            expected_install_profile=expected_install_profile,
            controller_protocol=controller_protocol,
        )

    def open_for_builder_reuse(
        self,
        generation_id: str,
        *,
        expected_state_contract_digest: str,
        expected_evaluator_fingerprint: str,
        expected_install_profile: str,
        controller_protocol: int,
    ) -> InstalledGeneration:
        """Verify a same-input build result before it has external release metadata."""

        if (
            not _SHA256_RE.fullmatch(expected_state_contract_digest)
            or not _DIGEST_RE.fullmatch(expected_evaluator_fingerprint)
            or not expected_install_profile
            or controller_protocol < 1
        ):
            raise GenerationStoreError("expected generation policy is invalid")
        with self.locked():
            return self._open_locked(
                generation_id,
                expected_state_contract_digest=expected_state_contract_digest,
                expected_evaluator_fingerprint=expected_evaluator_fingerprint,
                expected_install_profile=expected_install_profile,
                controller_protocol=controller_protocol,
            )

    def open_untrusted_for_test(self, generation_id: str) -> InstalledGeneration:
        """Verify self-consistency without external provenance; never use for launch."""

        with self.locked():
            return self._open_locked(generation_id)

    def cleanup_incomplete(
        self,
        *,
        quarantine: bool = True,
        stale_after_seconds: int = 3600,
    ) -> tuple[Path, ...]:
        """Validate COMPLETE entries and recover stale or malformed generations."""

        if stale_after_seconds < 0:
            raise ValueError("stale generation age cannot be negative")
        with self.locked():
            affected: list[Path] = []
            for candidate in sorted(self._root.iterdir(), key=lambda item: item.name):
                if not _GENERATION_ID_RE.fullmatch(candidate.name):
                    continue
                complete = os.path.lexists(candidate / _COMPLETE_NAME)
                if complete:
                    try:
                        self._open_locked(candidate.name)
                    except GenerationStoreError:
                        affected.append(self._discard_locked(candidate, quarantine=quarantine))
                    continue
                if self._building_is_live(candidate, stale_after_seconds=stale_after_seconds):
                    continue
                affected.append(self._discard_locked(candidate, quarantine=quarantine))
            return tuple(affected)

    def quarantine(self, generation_id: str) -> Path:
        """Quarantine an invalid generation, including a failed published entry."""

        safe_id = _generation_id(generation_id)
        with self.locked():
            candidate = self._root / safe_id
            if not os.path.lexists(candidate):
                raise GenerationStoreError("generation is unavailable for quarantine")
            return self._discard_locked(candidate, quarantine=True)

    def quarantine_incomplete(self) -> tuple[Path, ...]:
        return self.cleanup_incomplete(quarantine=True)

    def clean_incomplete(self) -> tuple[Path, ...]:
        return self.cleanup_incomplete(quarantine=False)

    def _open_locked(
        self,
        generation_id: str,
        *,
        expected_manifest_digest: str | None = None,
        expected_state_contract_digest: str | None = None,
        expected_evaluator_fingerprint: str | None = None,
        expected_install_profile: str | None = None,
        controller_protocol: int | None = None,
    ) -> InstalledGeneration:
        safe_id = _generation_id(generation_id)
        generation_path = self._root / safe_id
        _require_directory(generation_path, label="generation", require_read_only=True)
        _require_empty_regular_file(
            generation_path / _COMPLETE_NAME,
            label="COMPLETE marker",
            require_read_only=True,
        )
        if os.path.lexists(generation_path / _BUILDING_NAME):
            raise GenerationStoreError("complete generation still has a BUILDING marker")
        manifest_path = _require_regular_file(
            generation_path / _MANIFEST_NAME,
            label="generation manifest",
            require_read_only=True,
        )
        with _open_regular_file(manifest_path, label="generation manifest") as stream:
            manifest_bytes = stream.read(self._max_manifest_bytes + 1)
        if not manifest_bytes or len(manifest_bytes) > self._max_manifest_bytes:
            raise GenerationStoreError("generation manifest size is invalid")
        try:
            manifest = GenerationManifest.model_validate_json(manifest_bytes)
        except (ValidationError, ValueError) as exc:
            raise GenerationStoreError("generation manifest is invalid") from exc
        canonical = canonical_json_bytes(manifest)
        if manifest_bytes != canonical:
            raise GenerationStoreError("generation manifest is not canonical")
        manifest_digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
        if manifest.identity.generation_id != safe_id:
            raise GenerationStoreError("generation directory does not match manifest identity")
        if expected_manifest_digest is not None and manifest_digest != expected_manifest_digest:
            raise GenerationStoreError("generation manifest provenance does not match")
        if (
            expected_state_contract_digest is not None
            and manifest.state_contract.sha256() != expected_state_contract_digest
        ):
            raise GenerationStoreError("generation state contract provenance does not match")
        if (
            expected_evaluator_fingerprint is not None
            and manifest.identity.evaluator_fingerprint != expected_evaluator_fingerprint
        ):
            raise GenerationStoreError("generation evaluator provenance does not match")
        if (
            expected_install_profile is not None
            and manifest.identity.install_profile != expected_install_profile
        ):
            raise GenerationStoreError("generation install profile provenance does not match")
        if controller_protocol is not None and not (
            manifest.identity.controller_min
            <= controller_protocol
            <= manifest.identity.controller_max
        ):
            raise GenerationStoreError("generation controller protocol is incompatible")
        _verify_runtime_platform(manifest)
        actual_runtime_tree = runtime_tree_sha256(generation_path, require_read_only=True)
        if actual_runtime_tree != manifest.runtime_tree_sha256:
            raise GenerationStoreError("generation runtime tree failed integrity verification")

        descriptor = manifest.descriptor
        wheel_path = _safe_relative_path(
            generation_path,
            descriptor.wheel_path,
            expect_directory=False,
            label="generation wheel",
        )
        lock_path = _safe_relative_path(
            generation_path,
            descriptor.uv_lock_path,
            expect_directory=False,
            label="generation lockfile",
        )
        _verify_artifact(
            wheel_path,
            expected_size=descriptor.wheel_size_bytes,
            expected_sha256=manifest.identity.wheel_sha256,
            label="generation wheel",
        )
        _verify_artifact(
            lock_path,
            expected_size=descriptor.uv_lock_size_bytes,
            expected_sha256=manifest.identity.uv_lock_sha256,
            label="generation lockfile",
        )
        venv_path = _safe_relative_path(
            generation_path,
            descriptor.venv_path,
            expect_directory=True,
            label="generation virtualenv",
        )
        interpreter_path = _safe_relative_path(
            generation_path,
            f"{descriptor.venv_path}/bin/python",
            expect_directory=False,
            label="generation interpreter",
        )
        entrypoint_path = _safe_relative_path(
            generation_path,
            manifest.identity.entrypoint[0],
            expect_directory=False,
            label="generation entrypoint",
        )
        if entrypoint_path.parent != venv_path / "bin":
            raise GenerationStoreError("generation entrypoint escaped its virtualenv bin directory")
        if not os.access(interpreter_path, os.X_OK) or not os.access(entrypoint_path, os.X_OK):
            raise GenerationStoreError("generation interpreter or entrypoint is not executable")
        return InstalledGeneration(
            generation_id=safe_id,
            path=generation_path,
            manifest=manifest,
            manifest_digest=manifest_digest,
            interpreter_path=interpreter_path,
            entrypoint_path=entrypoint_path,
        )

    def _building_is_live(self, candidate: Path, *, stale_after_seconds: int) -> bool:
        marker = candidate / _BUILDING_NAME
        try:
            _require_regular_file(marker, label="BUILDING marker", require_read_only=False)
            with _open_regular_file(marker, label="BUILDING marker") as stream:
                payload = json.load(stream)
            pid = int(payload["pid"])
            started_at = float(payload["started_at"])
            if pid < 1 or started_at > time.time() + 60:
                return False
        except (GenerationStoreError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return False
        age = max(0.0, time.time() - started_at)
        if age > stale_after_seconds:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _discard_locked(self, candidate: Path, *, quarantine: bool) -> Path:
        if quarantine:
            quarantine_root = _secure_directory(
                self._quarantine_root,
                create=True,
                label="generation quarantine",
                required_mode=0o700,
                allow_root_owner=False,
            )
            destination = quarantine_root / f"{candidate.name}.{uuid4().hex}"
            try:
                metadata = candidate.lstat()
                if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                    candidate.chmod(0o700)
                os.replace(candidate, destination)
            except OSError as exc:
                raise GenerationStoreError("generation could not be quarantined") from exc
            return destination
        try:
            metadata = candidate.lstat()
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
        except OSError as exc:
            raise GenerationStoreError("generation could not be removed") from exc
        return candidate

    def _verify_root(self) -> None:
        _require_directory(self._root, label="generations root", require_read_only=False)
        metadata = self._root.stat(follow_symlinks=False)
        if (
            stat.S_IMODE(metadata.st_mode) != 0o711
            or metadata.st_uid not in {0, os.geteuid()}
        ):
            raise GenerationStoreError("generations root must remain trusted mode 0711")


def _absolute_without_symlinks(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            continue
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise ValueError(f"{label} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} has a symbolic-link ancestor")
    return absolute


def _secure_directory(
    path: str | Path,
    *,
    create: bool,
    label: str,
    required_mode: int,
    allow_root_owner: bool,
) -> Path:
    absolute = _absolute_without_symlinks(Path(path).expanduser(), label=label)
    existed = os.path.lexists(absolute)
    try:
        if create:
            absolute.mkdir(parents=True, exist_ok=True, mode=required_mode)
            if not existed:
                absolute.chmod(required_mode)
        metadata = absolute.lstat()
    except OSError as exc:
        raise ValueError(f"{label} is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in ({0, os.geteuid()} if allow_root_owner else {os.geteuid()})
    ):
        raise ValueError(f"{label} is not a regular directory")
    if stat.S_IMODE(metadata.st_mode) != required_mode:
        raise ValueError(f"{label} must have mode {required_mode:04o}")
    return absolute


def _generation_id(value: str) -> str:
    if not _GENERATION_ID_RE.fullmatch(str(value or "")):
        raise GenerationStoreError("generation ID is invalid")
    return value


def _require_directory(path: Path, *, label: str, require_read_only: bool) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GenerationStoreError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GenerationStoreError(f"{label} is not a regular directory")
    if require_read_only and stat.S_IMODE(metadata.st_mode) != 0o555:
        raise GenerationStoreError(f"{label} does not have published directory mode 0555")
    return path


def _require_regular_file(path: Path, *, label: str, require_read_only: bool) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GenerationStoreError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GenerationStoreError(f"{label} is not a regular file")
    if require_read_only and stat.S_IMODE(metadata.st_mode) not in {0o444, 0o555}:
        raise GenerationStoreError(f"{label} does not have a published read-only mode")
    return path


def _require_empty_regular_file(path: Path, *, label: str, require_read_only: bool) -> None:
    marker = _require_regular_file(path, label=label, require_read_only=require_read_only)
    if marker.stat(follow_symlinks=False).st_size != 0:
        raise GenerationStoreError(f"{label} is malformed")


def _safe_relative_path(
    generation_path: Path,
    raw_relative: str,
    *,
    expect_directory: bool,
    label: str,
) -> Path:
    relative = PurePosixPath(raw_relative)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise GenerationStoreError(f"{label} path is unsafe")
    current = generation_path
    for index, component in enumerate(relative.parts):
        current /= component
        final = index == len(relative.parts) - 1
        if final and not expect_directory:
            _require_regular_file(current, label=label, require_read_only=True)
        else:
            _require_directory(current, label=label, require_read_only=True)
    if not _is_relative_to(current, generation_path):
        raise GenerationStoreError(f"{label} escaped its generation")
    return current


def _verify_artifact(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> None:
    digest = hashlib.sha256()
    size = 0
    with _open_regular_file(path, label=label) as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    if size != expected_size or digest.hexdigest() != expected_sha256:
        raise GenerationStoreError(f"{label} hash or size does not match its manifest")


def _verify_runtime_platform(manifest: GenerationManifest) -> None:
    identity = manifest.identity
    expected = {
        "cpython_version": platform_module.python_version(),
        "cpython_cache_tag": str(sys.implementation.cache_tag or ""),
        "cpython_abi_tag": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "os_name": os.name,
        "platform": sysconfig.get_platform(),
        "machine": platform_module.machine(),
    }
    if sys.implementation.name != "cpython" or any(
        getattr(identity, field) != value for field, value in expected.items()
    ):
        raise GenerationStoreError("generation is incompatible with this Python platform")


def _open_regular_file(path: Path, *, label: str) -> BinaryIO:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GenerationStoreError(f"{label} is not a regular file")
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
        return stream
    except GenerationStoreError:
        raise
    except OSError as exc:
        raise GenerationStoreError(f"{label} could not be opened safely") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "GenerationStore",
    "GenerationStoreError",
    "InstalledGeneration",
    "runtime_tree_sha256",
]
