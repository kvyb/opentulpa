from __future__ import annotations

import stat
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from opentulpa.deep_agent.attachments import extract_zip_files


def _zip_bytes(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        for name, content in files:
            archive.writestr(name, content)
    return output.getvalue()


def test_extract_zip_files_preserves_paths_order_and_binary_content() -> None:
    nested = _zip_bytes([("inside.bin", b"nested")])
    extracted = extract_zip_files(
        _zip_bytes(
            [
                ("assets/pixel.bin", b"\x00\xff\x10"),
                ("README.md", b"read me"),
                ("nested.zip", nested),
            ]
        ),
        max_file_bytes=1024,
    )

    assert extracted.files == (
        ("assets/pixel.bin", b"\x00\xff\x10"),
        ("README.md", b"read me"),
        ("nested.zip", nested),
    )
    assert extracted.skipped == ()


def test_extract_zip_files_skips_traversal_symlinks_duplicates_and_oversized_files() -> None:
    output = BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("safe.txt", b"safe")
        archive.writestr("../escape.txt", b"escape")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("safe.txt", b"duplicate")
        archive.writestr("large.bin", b"12345")
        symlink = ZipInfo("link")
        symlink.create_system = 3
        symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(symlink, "safe.txt")

    extracted = extract_zip_files(output.getvalue(), max_file_bytes=4)

    assert extracted.files == (("safe.txt", b"safe"),)
    assert set(extracted.skipped) == {"../escape.txt", "safe.txt", "large.bin", "link"}


def test_extract_zip_files_rejects_invalid_and_expansion_limit() -> None:
    with pytest.raises(ValueError, match="valid ZIP"):
        extract_zip_files(b"not-a-zip", max_file_bytes=100)

    with pytest.raises(ValueError, match="sandbox limit"):
        extract_zip_files(
            _zip_bytes([("one.bin", b"123"), ("two.bin", b"456")]),
            max_file_bytes=100,
            max_expanded_bytes=5,
        )
