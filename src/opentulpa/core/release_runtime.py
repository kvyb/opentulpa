"""Application-side release runtime identity and probation controls."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
_GENERATION_ID = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_GENERATION_MARKERS = (
    "OPENTULPA_GENERATION_ID",
    "OPENTULPA_GENERATION_MANIFEST_DIGEST",
    "OPENTULPA_GENERATION_SOURCE_COMMIT",
    "OPENTULPA_GENERATION_SOURCE_TREE_SHA256",
)
_LIVE_SOURCE_MARKERS = (
    "OPENTULPA_SOURCE_COMMIT",
    "OPENTULPA_LIVE_SOURCE_ROOT",
)


def release_consumers_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether this application process may start external consumers."""

    source = os.environ if environ is None else environ
    disabled = str(source.get("OPENTULPA_DISABLE_CONSUMERS", "") or "").strip().casefold()
    if not disabled or disabled in _FALSE_VALUES:
        return True
    if disabled in _TRUE_VALUES:
        return False
    raise RuntimeError("OPENTULPA_DISABLE_CONSUMERS must be an explicit boolean value")


@dataclass(frozen=True, slots=True)
class ReleaseRuntimeIdentity:
    """Exact child identity echoed to the stable runtime supervisor."""

    generation_id: str | None
    source_commit: str | None
    launch_nonce: str | None

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> ReleaseRuntimeIdentity:
        source = os.environ if environ is None else environ
        generation_mode = any(name in source for name in _GENERATION_MARKERS)
        live_source_mode = any(name in source for name in _LIVE_SOURCE_MARKERS)
        generation_id = source.get("OPENTULPA_GENERATION_ID")
        source_commit = source.get("OPENTULPA_SOURCE_COMMIT")
        launch_nonce = source.get("OPENTULPA_LAUNCH_NONCE")

        if generation_mode and live_source_mode:
            raise RuntimeError("runtime cannot declare both generation and live source identity")
        if generation_mode:
            if generation_id is None or _GENERATION_ID.fullmatch(generation_id) is None:
                raise RuntimeError("generation runtime identity is missing or invalid")
            if launch_nonce is None or not 16 <= len(launch_nonce) <= 200:
                raise RuntimeError("generation launch nonce is missing or invalid")
            if source_commit is not None:
                raise RuntimeError("generation runtime cannot declare a live source identity")
        elif live_source_mode:
            if generation_id is not None:
                raise RuntimeError("live source runtime cannot declare a generation identity")
            if source_commit is None or _SOURCE_COMMIT.fullmatch(source_commit) is None:
                raise RuntimeError("live source runtime identity is missing or invalid")
            if launch_nonce is None or not 16 <= len(launch_nonce) <= 200:
                raise RuntimeError("live source launch nonce is missing or invalid")
        else:
            if generation_id is not None:
                raise RuntimeError("legacy runtime cannot declare a generation identity")
            if source_commit is not None:
                raise RuntimeError("legacy runtime cannot declare a live source identity")
            if launch_nonce is not None and not 16 <= len(launch_nonce) <= 200:
                raise RuntimeError("legacy launch nonce is invalid")

        return cls(
            generation_id=generation_id,
            source_commit=source_commit,
            launch_nonce=launch_nonce,
        )


__all__ = ["ReleaseRuntimeIdentity", "release_consumers_enabled"]
