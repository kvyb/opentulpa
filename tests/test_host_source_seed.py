from __future__ import annotations

import os
from pathlib import Path

import pytest

from opentulpa.host.source_seed import source_seed_sha256


def test_source_seed_sha256_is_stable_for_sorted_regular_tree(tmp_path: Path) -> None:
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "b.py").write_text("print('b')\n", encoding="utf-8")
    (root / "a.txt").write_text("a\n", encoding="utf-8")

    first = source_seed_sha256(root)
    second = source_seed_sha256(root)

    assert first == second
    assert len(first) == 64


def test_source_seed_sha256_rejects_links(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    target = tmp_path / "target"
    target.write_text("target\n", encoding="utf-8")
    os.symlink(target, root / "link")

    with pytest.raises(ValueError, match="link, hard link, or special file"):
        source_seed_sha256(root)
