"""Load reviewed, source-bundled capability manifests without importing extension code."""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

from pydantic import ValidationError

from opentulpa.capabilities.models import (
    CapabilityManifest,
    SecretSource,
    WorkerRuntime,
)

_MANIFEST_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}\.json\Z")
_WORKER_MODULE = re.compile(
    r"opentulpa\.capability_workers\.[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*\Z"
)
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_MANIFESTS = 100


class DeclarativeCapabilityError(RuntimeError):
    """A source-bundled manifest violates the fixed extension contract."""


def default_manifest_root() -> Path:
    """Return the non-importable package directory used for reviewed manifests."""

    return Path(__file__).resolve().parents[1] / "capability_manifests"


def load_declarative_capabilities(
    root: str | Path | None = None,
) -> tuple[CapabilityManifest, ...]:
    """Parse source-bundled manifests through a narrow, import-free contract.

    These files are allowed to select reviewed worker modules. They cannot declare an
    in-process entrypoint, install dependencies, reference an arbitrary executable, or
    request host-owned credentials.
    """

    directory = Path(root) if root is not None else default_manifest_root()
    if not directory.exists():
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise DeclarativeCapabilityError("capability manifest root must be a directory")

    paths = sorted(directory.iterdir(), key=lambda item: item.name)
    if len(paths) > _MAX_MANIFESTS:
        raise DeclarativeCapabilityError("too many declarative capability manifests")

    manifests: list[CapabilityManifest] = []
    for path in paths:
        if path.name == "README.md":
            continue
        if not _MANIFEST_NAME.fullmatch(path.name):
            raise DeclarativeCapabilityError("capability manifest filename is invalid")
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DeclarativeCapabilityError("capability manifest must be a regular file")
        if metadata.st_size > _MAX_MANIFEST_BYTES:
            raise DeclarativeCapabilityError("capability manifest exceeds its byte limit")
        try:
            raw = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(raw, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
            manifest = CapabilityManifest.model_validate(payload)
        except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as exc:
            raise DeclarativeCapabilityError("capability manifest is invalid") from exc
        _validate_manifest(path, manifest)
        manifests.append(manifest.model_copy(update={"seed": True}))
    return tuple(manifests)


def _validate_manifest(path: Path, manifest: CapabilityManifest) -> None:
    if path.stem != manifest.name:
        raise DeclarativeCapabilityError("capability manifest filename must match its name")
    if manifest.module is not None or manifest.entrypoint is not None:
        raise DeclarativeCapabilityError("declarative capabilities cannot run in process")
    if manifest.dependencies:
        raise DeclarativeCapabilityError(
            "declarative capabilities cannot install runtime dependencies"
        )
    if manifest.artifact_digest is not None:
        raise DeclarativeCapabilityError("source-bundled capabilities cannot select OCI images")
    for secret in (
        *manifest.secrets,
        *(item for worker in manifest.workers for item in worker.secrets),
    ):
        if secret.source is SecretSource.HOST:
            raise DeclarativeCapabilityError(
                "declarative capabilities cannot request host-owned secrets"
            )
    for worker in manifest.workers:
        if worker.runtime is not WorkerRuntime.SUBPROCESS:
            raise DeclarativeCapabilityError(
                "source-bundled capabilities require reviewed subprocess workers"
            )
        command = worker.command
        if len(command) < 3 or command[0:2] != ("python", "-m"):
            raise DeclarativeCapabilityError(
                "declarative worker commands must use python -m"
            )
        if _WORKER_MODULE.fullmatch(command[2]) is None:
            raise DeclarativeCapabilityError(
                "declarative worker module must be inside opentulpa.capability_workers"
            )


__all__ = [
    "DeclarativeCapabilityError",
    "default_manifest_root",
    "load_declarative_capabilities",
]
