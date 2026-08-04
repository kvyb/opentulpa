"""Safely materialize uploaded archives as ordinary workspace files."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

MAX_ZIP_ENTRIES = 512
MAX_ZIP_EXPANDED_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ExtractedZip:
    files: tuple[tuple[str, bytes], ...]
    skipped: tuple[str, ...]


def extract_zip_files(
    raw_bytes: bytes,
    *,
    max_file_bytes: int,
    max_entries: int = MAX_ZIP_ENTRIES,
    max_expanded_bytes: int = MAX_ZIP_EXPANDED_BYTES,
) -> ExtractedZip:
    """Return regular ZIP members without interpreting their contents."""

    try:
        archive = ZipFile(BytesIO(raw_bytes))
    except BadZipFile as exc:
        raise ValueError("attachment is not a valid ZIP archive") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > max_entries:
            raise ValueError(f"ZIP archive has more than {max_entries} entries")
        if sum(max(0, int(info.file_size)) for info in infos) > max_expanded_bytes:
            raise ValueError("ZIP archive expands beyond the sandbox limit")

        files: list[tuple[str, bytes]] = []
        skipped: list[str] = []
        seen: set[str] = set()
        expanded_bytes = 0
        for info in infos:
            if info.is_dir():
                continue
            member_name = _safe_member_name(info)
            if member_name is None or member_name in seen:
                skipped.append(str(info.filename or "unnamed member"))
                continue
            if info.flag_bits & 0x1 or info.file_size > max_file_bytes:
                skipped.append(member_name)
                continue
            try:
                read_limit = min(
                    max_file_bytes + 1,
                    max_expanded_bytes - expanded_bytes + 1,
                )
                with archive.open(info) as member:
                    content = member.read(read_limit)
            except (BadZipFile, NotImplementedError, OSError, RuntimeError):
                skipped.append(member_name)
                continue
            expanded_bytes += len(content)
            if expanded_bytes > max_expanded_bytes:
                raise ValueError("ZIP archive expands beyond the sandbox limit")
            if len(content) > max_file_bytes:
                skipped.append(member_name)
                continue
            seen.add(member_name)
            files.append((member_name, content))

    return ExtractedZip(files=tuple(files), skipped=tuple(skipped))


def _safe_member_name(info: ZipInfo) -> str | None:
    raw = str(info.filename or "").replace("\\", "/")
    if not raw or len(raw) > 4_096 or any(ord(char) < 32 for char in raw):
        return None
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        return None
    if path.parts[0].endswith(":") or len(path.parts) > 64:
        return None
    file_type = stat.S_IFMT(int(info.external_attr) >> 16)
    if file_type not in {0, stat.S_IFREG}:
        return None
    return str(path)
