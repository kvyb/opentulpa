"""Restricted zero-config command execution for single-host deployments."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from deepagents.backends.protocol import ExecuteResponse

from opentulpa.deep_agent.sandbox import TenantContainerPolicy
from opentulpa.evolution.sandbox import (
    CandidateProcessBackend,
    CandidateSandboxPolicy,
)


class RestrictedProcessExecutionProvider:
    """Run tenant commands as a resource-limited user with a scrubbed environment."""

    def __init__(
        self,
        *,
        policy: TenantContainerPolicy,
        max_workspace_bytes: int,
    ) -> None:
        self._policy = policy
        self._max_workspace_bytes = max(max_workspace_bytes, policy.max_file_bytes)

    @staticmethod
    def supported() -> bool:
        return CandidateProcessBackend.supported()

    def execute(
        self,
        *,
        tenant_id: str,
        command: str,
        timeout: int,
        workspace: Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecuteResponse:
        del tenant_id
        if workspace is not None:
            return self._execute(
                workspace=workspace,
                command=command,
                timeout=timeout,
                cancel_event=cancel_event,
            )
        with tempfile.TemporaryDirectory(prefix="opentulpa-tenant-process-") as temporary:
            root = Path(temporary)
            workspace_root = root / "workspace"
            workspace_root.mkdir()
            return self._execute(
                workspace=workspace_root,
                command=command,
                timeout=timeout,
                cancel_event=cancel_event,
            )

    def _execute(
        self,
        *,
        workspace: Path,
        command: str,
        timeout: int,
        cancel_event: threading.Event | None,
    ) -> ExecuteResponse:
        backend = CandidateProcessBackend(
            workspace=workspace,
            allowed_root=workspace.parent,
            policy=CandidateSandboxPolicy(
                image=self._policy.image,
                cpu_limit=self._policy.cpu_limit,
                memory_limit=self._policy.memory_limit,
                pid_limit=self._policy.pid_limit,
                timeout_seconds=self._policy.timeout_seconds,
                max_output_bytes=self._policy.max_output_bytes,
                max_file_bytes=self._policy.max_file_bytes,
                max_total_bytes=self._max_workspace_bytes,
                max_entries=max(100, self._policy.max_workspace_entries),
                network_enabled=True,
            ),
        )
        return backend.execute(
            command,
            timeout=timeout,
            cancel_event=cancel_event,
        )


__all__ = ["RestrictedProcessExecutionProvider"]
