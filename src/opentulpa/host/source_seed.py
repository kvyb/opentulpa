"""Deterministic source-seed digests for controller install artifacts."""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path


def source_seed_sha256(root: Path) -> str:
    """Hash a regular-file source tree without trusting platform metadata noise."""

    if root.is_symlink() or not root.is_dir():
        raise ValueError("source seed root must be a directory")
    digest = hashlib.sha256()
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        paths.extend(Path(directory) / name for name in (*directory_names, *file_names))
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        payload = b""
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"D"
            mode = 0o755
        elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
            kind = b"F"
            payload = path.read_bytes()
            mode = 0o755 if metadata.st_mode & 0o111 else 0o644
        else:
            raise ValueError(f"source seed contains a link, hard link, or special file: {relative}")
        digest.update(kind + b"\0")
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(f"{mode:o}".encode("ascii") + b"\0")
        digest.update(str(len(payload)).encode("ascii") + b"\0")
        digest.update(payload + b"\0")
    return digest.hexdigest()
