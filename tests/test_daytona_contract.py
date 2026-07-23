from __future__ import annotations

import inspect
from importlib.metadata import version

from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig
from langchain_daytona import DaytonaSandbox


def test_pinned_daytona_adapter_matches_repository_provider_contract() -> None:
    assert version("daytona") == "0.200.1"
    assert version("langchain-daytona") == "0.0.7"
    assert {"api_key", "api_url", "target"} <= set(inspect.signature(DaytonaConfig).parameters)
    assert {"snapshot", "labels", "auto_stop_interval", "network_block_all"} <= set(
        inspect.signature(CreateSandboxFromSnapshotParams).parameters
    )
    assert {"params", "timeout"} <= set(inspect.signature(Daytona.create).parameters)
    assert {"sandbox", "timeout"} <= set(inspect.signature(Daytona.stop).parameters)
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
        assert callable(getattr(DaytonaSandbox, name))
