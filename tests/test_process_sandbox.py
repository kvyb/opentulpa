from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from deepagents.backends.protocol import ExecuteResponse

from opentulpa.deep_agent import process_sandbox
from opentulpa.deep_agent.process_sandbox import RestrictedProcessExecutionProvider
from opentulpa.deep_agent.sandbox import TenantContainerPolicy


def test_restricted_process_provider_maps_tenant_policy_and_cancellation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeBackend:
        @staticmethod
        def supported() -> bool:
            return True

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def execute(self, command: str, **kwargs: Any) -> ExecuteResponse:
            captured["command"] = command
            captured.update(kwargs)
            return ExecuteResponse(output="ok", exit_code=0, truncated=False)

    monkeypatch.setattr(process_sandbox, "CandidateProcessBackend", FakeBackend)
    policy = TenantContainerPolicy(
        image="opentulpa-tenant-sandbox:test",
        memory_limit="256m",
        pid_limit=64,
        timeout_seconds=30,
        max_output_bytes=8_192,
        max_file_bytes=4_096,
        max_workspace_entries=250,
        network_enabled=True,
    )
    provider = RestrictedProcessExecutionProvider(
        policy=policy,
        max_workspace_bytes=1_000_000,
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cancellation = threading.Event()

    result = provider.execute(
        tenant_id="tenant-a",
        command="pwd",
        timeout=12,
        workspace=workspace,
        cancel_event=cancellation,
    )

    candidate_policy = captured["policy"]
    assert result.output == "ok"
    assert captured["workspace"] == workspace
    assert captured["allowed_root"] == tmp_path
    assert candidate_policy.memory_limit == "256m"
    assert candidate_policy.pid_limit == 64
    assert candidate_policy.max_total_bytes == 1_000_000
    assert captured["command"] == "pwd"
    assert captured["timeout"] == 12
    assert captured["cancel_event"] is cancellation
