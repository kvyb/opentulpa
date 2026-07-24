from __future__ import annotations

import inspect
from importlib.metadata import version

import pytest

daytona = pytest.importorskip("daytona")
langchain_daytona = pytest.importorskip("langchain_daytona")


def test_pinned_daytona_adapter_matches_repository_provider_contract() -> None:
    assert version("daytona") == "0.200.1"
    assert version("langchain-daytona") == "0.0.7"
    assert {"api_key", "api_url", "target"} <= set(
        inspect.signature(daytona.DaytonaConfig).parameters
    )
    assert {"snapshot", "labels", "auto_stop_interval", "network_block_all"} <= set(
        inspect.signature(daytona.CreateSandboxFromSnapshotParams).parameters
    )
    assert {"params", "timeout"} <= set(inspect.signature(daytona.Daytona.create).parameters)
    assert {"sandbox", "timeout"} <= set(inspect.signature(daytona.Daytona.stop).parameters)
    for name in (
        "execute",
        "aexecute",
        "ls",
        "read",
        "write",
        "edit",
        "glob",
        "grep",
        "upload_files",
        "download_files",
    ):
        assert callable(getattr(langchain_daytona.DaytonaSandbox, name))
