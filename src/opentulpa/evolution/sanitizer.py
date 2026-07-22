"""Supervisor-owned sanitation gate for candidate contribution patches."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


class ContributionSanitizationError(RuntimeError):
    """A patch cannot be exported without exposing unsafe material."""


@dataclass(frozen=True, slots=True)
class ContributionAttestation:
    patch_sha256: str
    scanner_version: str
    bytes_scanned: int


_SCANNER_VERSION = "opentulpa-contribution-sanitizer-v1"
_MAX_PATCH_BYTES = 50 * 1024 * 1024
_UNSAFE_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "private_key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    ("aws_access_key", re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    (
        "github_token",
        re.compile(rb"(?<![A-Za-z0-9])(?:gh[opusr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{50,255})"),
    ),
    ("slack_token", re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,255}")),
    (
        "openai_style_token",
        re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{32,255}"),
    ),
    (
        "telegram_bot_token",
        re.compile(rb"(?<![0-9])[0-9]{8,12}:[A-Za-z0-9_-]{30,60}(?![A-Za-z0-9_-])"),
    ),
    (
        "host_private_path",
        re.compile(rb"(?:/Users/[^/\s]+|/home/[^/\s]+|[A-Za-z]:\\Users\\[^\\\s]+)(?:[/\\])"),
    ),
)


def sanitize_contribution_patch(
    patch_path: Path,
    *,
    expected_sha256: str,
) -> ContributionAttestation:
    """Verify one immutable text patch and return a content-bound attestation."""

    path = patch_path.expanduser()
    if path.is_symlink() or not path.is_file():
        raise ContributionSanitizationError("contribution patch is unavailable")
    path = path.resolve(strict=True)
    size = path.stat().st_size
    if size < 1 or size > _MAX_PATCH_BYTES:
        raise ContributionSanitizationError("contribution patch exceeds its sanitation limit")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256 or "")) or digest != expected_sha256:
        raise ContributionSanitizationError("contribution patch digest changed before sanitation")
    if b"GIT binary patch" in payload or b"Binary files " in payload:
        raise ContributionSanitizationError("binary contribution patches require manual export")
    for _label, pattern in _UNSAFE_PATTERNS:
        if pattern.search(payload):
            raise ContributionSanitizationError("contribution patch contains sensitive material")
    return ContributionAttestation(
        patch_sha256=digest,
        scanner_version=_SCANNER_VERSION,
        bytes_scanned=len(payload),
    )


__all__ = [
    "ContributionAttestation",
    "ContributionSanitizationError",
    "sanitize_contribution_patch",
]
