from __future__ import annotations

from pathlib import Path

import pytest

from opentulpa.evolution.models import Release
from opentulpa.evolution.release import AtomicReleasePointer


@pytest.mark.asyncio
async def test_release_pointer_is_atomic_typed_and_clearable(tmp_path: Path) -> None:
    pointer = AtomicReleasePointer(tmp_path / "state" / "current.json")
    release = Release(
        candidate_id="candidate_1",
        source_commit="a" * 40,
        artifact_digest="sha256:" + "b" * 64,
    )

    assert await pointer.current() is None
    await pointer.activate(release)
    assert await pointer.current() == release
    assert pointer.path.stat().st_mode & 0o777 == 0o600

    await pointer.clear()
    assert await pointer.current() is None
