"""Trusted host control and mutable product root layout."""

from __future__ import annotations

import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_RUNTIME_UID = 65_532
_RUNTIME_GID = 65_532
_CANDIDATE_UID = 65_533
_CANDIDATE_GID = 65_533
_MAX_MIGRATION_ENTRIES = 100_000
_MAX_MIGRATION_BYTES = 2 * 1024 * 1024 * 1024
_PRODUCT_ENTRIES = {
    ".opentulpa": Path(".opentulpa"),
    "tulpa_stuff": Path("tulpa_stuff"),
    "notifications.db": Path(".opentulpa/notifications.db"),
    "customer_profiles.db": Path("customer_profiles.db"),
    "deepagents": Path("deepagents"),
    "file_vault": Path("file_vault"),
    "file_vault.db": Path("file_vault.db"),
    "intake_sinks": Path("intake_sinks"),
    "intake_workflows.db": Path("intake_workflows.db"),
    "knowledge": Path("knowledge"),
    "telegram_business.db": Path("telegram_business.db"),
}
_CONTROLLER_ENTRIES = frozenset(
    {
        "bootstrap",
        "install",
        "product",
        "source",
        "bun",
        "lost+found",
        ".runtime-generations-control",
        "runtime-generations",
        "runtime-source-envs",
        "sandbox-host",
        "sandbox_worker",
    }
)


class HostPathError(RuntimeError):
    """The persistent host layout is unsafe or has an ambiguous migration."""


@dataclass(frozen=True, slots=True)
class HostPaths:
    """Resolved roots with controller and mutable-product ownership kept separate."""

    data_root: Path
    control_root: Path
    product_root: Path
    runtime_uid: int | None
    runtime_gid: int | None
    candidate_uid: int | None
    candidate_gid: int | None

    @property
    def runtime_control_path(self) -> Path:
        return self.control_root / "runtime-child.json"

    @property
    def notification_store_path(self) -> Path:
        return self.product_root / ".opentulpa" / "notifications.db"

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
        *,
        home: Path | None = None,
    ) -> HostPaths:
        values = os.environ if environment is None else environment
        configured_data = str(values.get("OPENTULPA_DATA_ROOT") or "").strip()
        default_data = (Path.home() if home is None else home) / ".local" / "share" / "opentulpa"
        data_root = _safe_absolute_path(
            configured_data or str(default_data),
            label="OPENTULPA_DATA_ROOT",
        )
        configured_control = str(values.get("OPENTULPA_CONTROL_ROOT") or "").strip()
        configured_product = str(values.get("OPENTULPA_PRODUCT_ROOT") or "").strip()
        control_root = _safe_absolute_path(
            configured_control or str(data_root / "bootstrap"),
            label="OPENTULPA_CONTROL_ROOT",
        )
        product_root = _safe_absolute_path(
            configured_product or str(data_root / "product"),
            label="OPENTULPA_PRODUCT_ROOT",
        )
        _require_separate_roots(
            control_root,
            product_root,
        )
        isolated = sys.platform.startswith("linux") and hasattr(os, "geteuid") and os.geteuid() == 0
        if isolated and (_RUNTIME_UID == _CANDIDATE_UID or _RUNTIME_GID == _CANDIDATE_GID):
            raise HostPathError("runtime and candidate identities must be distinct")
        return cls(
            data_root=data_root,
            control_root=control_root,
            product_root=product_root,
            runtime_uid=_RUNTIME_UID if isolated else None,
            runtime_gid=_RUNTIME_GID if isolated else None,
            candidate_uid=_CANDIDATE_UID if isolated else None,
            candidate_gid=_CANDIDATE_GID if isolated else None,
        )

    def provision(self) -> None:
        """Create, migrate, validate, and assign the persistent root layout."""

        _secure_directory(self.data_root, mode=0o711 if self.runtime_uid is not None else 0o700)
        _secure_directory(self.control_root, mode=0o700)
        _secure_product_directory(self.product_root, runtime_uid=self.runtime_uid)
        self._migrate_legacy_product_entries()
        _validate_regular_tree(self.product_root)
        if self.runtime_uid is not None and self.runtime_gid is not None:
            _assign_product_ownership(self.product_root, self.runtime_uid, self.runtime_gid)
        else:
            _make_product_writable(self.product_root)

    def _migrate_legacy_product_entries(self) -> None:
        product_name = self.product_root.name if self.product_root.parent == self.data_root else ""
        control_name = self.control_root.name if self.control_root.parent == self.data_root else ""
        allowed = _CONTROLLER_ENTRIES | {
            product_name,
            control_name,
        }
        sources: list[tuple[Path, Path]] = []
        try:
            entries = tuple(os.scandir(self.data_root))
        except OSError as exc:
            raise HostPathError("host data root could not be inspected") from exc
        product_entries = dict(_PRODUCT_ENTRIES)
        legacy_notifications = self.data_root / "notifications.db"
        legacy_state_notifications = self.data_root / ".opentulpa" / "notifications.db"
        if os.path.lexists(legacy_notifications) and os.path.lexists(legacy_state_notifications):
            archived = Path(".opentulpa/notifications.legacy-from-data-root.db")
            if os.path.lexists(self.data_root / archived):
                raise HostPathError("legacy notification archive conflicts during product migration")
            product_entries["notifications.db"] = archived
        if len(entries) > len(allowed) + len(_PRODUCT_ENTRIES):
            raise HostPathError(
                "host data root has unknown ambiguous migration entries: "
                f"{_entry_names_for_error(entries)}"
            )
        for entry in entries:
            path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise HostPathError("host data root entry could not be inspected") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise HostPathError("host data root contains a symbolic-link entry")
            destination = product_entries.get(entry.name)
            if destination is not None:
                sources.append((path, self.product_root / destination))
                continue
            if entry.name not in allowed:
                raise HostPathError(
                    "host data root has unknown ambiguous migration entry: "
                    f"{_entry_name_for_error(entry)}"
                )
            if not stat.S_ISDIR(metadata.st_mode):
                raise HostPathError("host data root contains a special controller entry")

        for source, destination in sources:
            _validate_regular_tree(source)
            if os.path.lexists(destination):
                raise HostPathError("legacy product entry conflicts with the product root")
        for source, destination in sorted(sources, key=lambda item: len(item[1].parts)):
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if destination.parent.is_symlink():
                raise HostPathError("product migration destination is a symbolic link")
            try:
                os.replace(source, destination)
            except OSError as exc:
                raise HostPathError(
                    "legacy product entry could not be migrated atomically"
                ) from exc


def _safe_absolute_path(raw_value: str, *, label: str) -> Path:
    if not raw_value or "\x00" in raw_value:
        raise HostPathError(f"{label} is invalid")
    raw = Path(raw_value).expanduser()
    path = raw if raw.is_absolute() else Path.cwd() / raw
    path = path.absolute()
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            continue
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise HostPathError(f"{label} could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise HostPathError(f"{label} has a symbolic-link ancestor")
    return path


def _entry_names_for_error(entries: tuple[os.DirEntry[str], ...]) -> str:
    return ", ".join(_entry_name_for_error(entry) for entry in sorted(entries, key=lambda item: item.name))


def _entry_name_for_error(entry: os.DirEntry[str]) -> str:
    try:
        metadata = entry.stat(follow_symlinks=False)
    except OSError:
        kind = "unavailable"
    else:
        if stat.S_ISDIR(metadata.st_mode):
            kind = "dir"
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "link"
        else:
            kind = "special"
    return f"{entry.name} ({kind})"


def _require_separate_roots(*roots: Path) -> None:
    for index, root in enumerate(roots):
        for other in roots[index + 1 :]:
            if root == other or _is_relative_to(root, other) or _is_relative_to(other, root):
                raise HostPathError("host control, product, and generation roots must be separate")


def _secure_directory(path: Path, *, mode: int) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        metadata = path.lstat()
    except OSError as exc:
        raise HostPathError("host root could not be provisioned") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise HostPathError("host root is not a controller-owned directory")
    try:
        path.chmod(mode)
    except OSError as exc:
        raise HostPathError("host root permissions could not be applied") from exc


def _secure_product_directory(path: Path, *, runtime_uid: int | None) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = path.lstat()
    except OSError as exc:
        raise HostPathError("host product root could not be provisioned") from exc
    allowed_owners = {os.geteuid()}
    if runtime_uid is not None:
        allowed_owners.add(runtime_uid)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid not in allowed_owners
    ):
        raise HostPathError("host product root has an unexpected owner or type")
    try:
        path.chmod(0o700)
    except OSError as exc:
        raise HostPathError("host product root permissions could not be applied") from exc


def _validate_regular_tree(root: Path) -> None:
    entries = 0
    total_bytes = 0
    pending = [root]
    while pending:
        path = pending.pop()
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise HostPathError("product migration entry could not be inspected") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise HostPathError("product migration cannot contain symbolic links")
        if stat.S_ISREG(metadata.st_mode):
            if metadata.st_nlink != 1:
                raise HostPathError("product migration cannot contain hard-linked files")
            total_bytes += metadata.st_size
            if total_bytes > _MAX_MIGRATION_BYTES:
                raise HostPathError("legacy product data exceeds the migration size limit")
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise HostPathError("product migration cannot contain special files")
        try:
            children = tuple(Path(entry.path) for entry in os.scandir(path))
        except OSError as exc:
            raise HostPathError("product migration directory could not be inspected") from exc
        entries += len(children)
        if entries > _MAX_MIGRATION_ENTRIES:
            raise HostPathError("legacy product data has too many entries")
        pending.extend(children)


def _assign_product_ownership(root: Path, uid: int, gid: int) -> None:
    pending = [root]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise HostPathError("product root cannot contain symbolic links")
        os.chown(path, uid, gid, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o700)
            pending.extend(Path(entry.path) for entry in os.scandir(path))
        else:
            path.chmod(0o700 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o600)


def _make_product_writable(root: Path) -> None:
    pending = [root]
    while pending:
        path = pending.pop()
        metadata = path.lstat()
        if stat.S_ISDIR(metadata.st_mode):
            path.chmod(0o700)
            pending.extend(Path(entry.path) for entry in os.scandir(path))
        else:
            path.chmod(0o700 if stat.S_IMODE(metadata.st_mode) & 0o111 else 0o600)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["HostPathError", "HostPaths"]
