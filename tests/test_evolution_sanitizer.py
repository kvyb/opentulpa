from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from opentulpa.evolution.sanitizer import (
    ContributionSanitizationError,
    sanitize_contribution_patch,
)


def _write_patch(path: Path, value: bytes) -> str:
    path.write_bytes(value)
    return hashlib.sha256(value).hexdigest()


def test_contribution_sanitizer_attests_to_exact_safe_patch(tmp_path: Path) -> None:
    path = tmp_path / "candidate.patch"
    digest = _write_patch(
        path,
        b"diff --git a/a.py b/a.py\n+print('safe improvement')\n",
    )

    result = sanitize_contribution_patch(path, expected_sha256=digest)

    assert result.patch_sha256 == digest
    assert result.bytes_scanned == path.stat().st_size
    assert result.scanner_version.startswith("opentulpa-contribution-sanitizer-")


@pytest.mark.parametrize(
    "unsafe",
    (
        b"+-----BEGIN OPENSSH PRIVATE KEY-----\n",
        b"+token = 'ghp_abcdefghijklmnopqrstuvwxyzABCDEFGHIJ'\n",
        b"+bot = '123456789:abcdefghijklmnopqrstuvwxyzABCDE12345'\n",
        b"+path = '/Users/alice/private/project'\n",
        b"GIT binary patch\nliteral 3\nabc\n",
    ),
)
def test_contribution_sanitizer_rejects_sensitive_or_binary_patch(
    tmp_path: Path,
    unsafe: bytes,
) -> None:
    path = tmp_path / "candidate.patch"
    digest = _write_patch(path, b"diff --git a/a b/a\n" + unsafe)

    with pytest.raises(ContributionSanitizationError):
        sanitize_contribution_patch(path, expected_sha256=digest)


def test_contribution_sanitizer_rejects_digest_change(tmp_path: Path) -> None:
    path = tmp_path / "candidate.patch"
    _write_patch(path, b"diff --git a/a b/a\n+safe\n")

    with pytest.raises(ContributionSanitizationError, match="digest changed"):
        sanitize_contribution_patch(path, expected_sha256="0" * 64)
