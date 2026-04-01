from __future__ import annotations

import uuid

import pytest

from opentulpa.tasks import sandbox


def test_read_file_rejects_paths_outside_allowed_roots() -> None:
    with pytest.raises(PermissionError, match="allowed read roots"):
        sandbox.read_file("README.md")


def test_read_file_reports_missing_file_with_requested_path() -> None:
    missing_rel = f"tulpa_stuff/{uuid.uuid4().hex}.txt"

    with pytest.raises(FileNotFoundError, match=missing_rel):
        sandbox.read_file(missing_rel)


def test_read_file_normalizes_duplicate_tulpa_stuff_prefix(tmp_path, monkeypatch) -> None:
    tulpa_dir = tmp_path / "tulpa_stuff"
    tulpa_dir.mkdir()
    target = tulpa_dir / "solana_trading_wallet.json"
    target.write_text('{"ok":true}', encoding="utf-8")

    monkeypatch.setattr(sandbox, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sandbox, "TULPA_STUFF_DIR", tulpa_dir)
    monkeypatch.setitem(sandbox.ALLOWED_READ_DIRS, "tulpa_stuff", tulpa_dir)

    content = sandbox.read_file("tulpa_stuff/tulpa_stuff/solana_trading_wallet.json")

    assert content == '{"ok":true}'
