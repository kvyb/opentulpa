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
