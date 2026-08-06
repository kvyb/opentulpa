"""Interactive source-session authority used by the evolution supervisor."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from typing import Any

from opentulpa.evolution.lineage import GitLineageError
from opentulpa.evolution.models import SourceReleaseOperation, SourceReleaseOperationStatus


class SourceSessionService:
    """Own source-session requests while resolving shared behavior on the supervisor.

    The owner lookup is deliberate: tests and recovery code patch supervisor methods
    after construction, and those seams must remain observable by source operations.
    """

    def __init__(self, supervisor: Any) -> None:
        self._supervisor = supervisor

    async def source_status(
        self,
        *,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        supervisor = self._supervisor
        supervisor._require_started()
        if not supervisor._source_mutation_enabled:
            return {
                "available": False,
                "active": False,
                "session_active": False,
                "candidate_id": None,
                "reason": supervisor._source_mutation_unavailable_reason,
                "source_mutation_enabled": False,
                "dependency_resolution_available": False,
                "diff_sha256": hashlib.sha256(b"").hexdigest(),
                "conflict_paths": [],
            }
        session_key, audit = supervisor._source_context(audit_context)
        async with supervisor._source_lock(session_key):
            lineage = await supervisor._source_lineage()
            candidate = await supervisor._find_source_session(
                session_key,
                tenant_id=str(audit["tenant_id"]),
            )
            if candidate is None:
                return {
                    "available": True,
                    "source_mutation_enabled": True,
                    "dependency_resolution_available": supervisor._dependency_resolver is not None,
                    "active": False,
                    "session_active": False,
                    "candidate_id": None,
                    "diff_sha256": hashlib.sha256(b"").hexdigest(),
                    "conflict_paths": [],
                    **lineage,
                }
            workspace = supervisor._source_workspace(candidate)
            return {
                "available": True,
                "source_mutation_enabled": True,
                "dependency_resolution_available": supervisor._dependency_resolver is not None,
                "session_active": True,
                **await supervisor._source_snapshot(candidate, workspace),
                **lineage,
            }

    async def source_shell(
        self,
        *,
        command: str,
        timeout_seconds: int = 300,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        supervisor = self._supervisor
        supervisor._require_started()
        supervisor._require_source_mutation()
        safe_command = str(command or "").strip()
        if not safe_command or "\x00" in safe_command or len(safe_command) > 100_000:
            raise ValueError("source shell command is invalid")
        timeout = int(timeout_seconds)
        if timeout < 1 or timeout > 3_600:
            raise ValueError("source shell timeout must be between 1 and 3600 seconds")
        backend_factory = supervisor._candidate_backend_factory
        if backend_factory is None:
            raise supervisor._evolution_error("source shell is unavailable")
        session_key, audit = supervisor._source_context(audit_context)
        async with supervisor._source_lock(session_key):
            candidate, workspace = await supervisor._open_source_session(
                session_key=session_key,
                audit=audit,
            )
            await supervisor._require_current_source_base(candidate)
            conflict_paths = await supervisor._source_conflict_paths(candidate, workspace)
            backend = backend_factory(workspace.path)
            # Full candidate repositories deliberately expose normal Git metadata.
            response = await backend.aexecute(safe_command, timeout=timeout)
            if supervisor._lineage is not None and conflict_paths:
                try:
                    await asyncio.to_thread(
                        supervisor._lineage.stage_resolved_conflicts,
                        workspace,
                        conflict_paths,
                    )
                except GitLineageError as exc:
                    raise supervisor._evolution_error(
                        "source merge conflict resolution is invalid"
                    ) from exc
            current = await supervisor._required_candidate(candidate.id)
            snapshot = await supervisor._source_snapshot(current, workspace, include_diff=False)
            output, output_truncated = supervisor._bounded_text(response.output)
            return {
                **snapshot,
                "exit_code": int(response.exit_code),
                "output": output,
                "output_truncated": bool(response.truncated or output_truncated),
            }

    async def source_sync_upstream(
        self,
        *,
        expected_active_release_id: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        supervisor = self._supervisor
        supervisor._require_started()
        supervisor._require_source_mutation()
        expected_release = str(expected_active_release_id or "").strip()
        if not expected_release or len(expected_release) > 100:
            raise ValueError("expected active release is invalid")
        lineage = supervisor._lineage
        if lineage is None:
            raise supervisor._evolution_error("source lineage is unavailable")
        session_key, audit = supervisor._source_context(audit_context)
        tenant_id = str(audit["tenant_id"])
        async with supervisor._source_lock(session_key):
            current = await supervisor._archive.get_current_release()
            if current is None or current.id != expected_release:
                raise supervisor._evolution_error("active release changed before upstream sync")
            if await supervisor._find_source_session(session_key, tenant_id=tenant_id) is not None:
                raise supervisor._evolution_error(
                    "finish or release the active source session before upstream sync"
                )
            try:
                synced = await asyncio.to_thread(
                    lineage.sync_upstream,
                    supervisor._upstream_repository,
                    supervisor._upstream_ref,
                )
            except (GitLineageError, ValueError) as exc:
                raise supervisor._evolution_error("remote upstream synchronization failed") from exc
            values = {
                "synced": synced.changed,
                "previous_upstream_commit": synced.previous_commit,
                "upstream_commit": synced.upstream_commit,
            }
            if not synced.changed:
                return {**values, **await supervisor._source_lineage()}
            candidate, workspace = await supervisor._open_source_session(
                session_key=session_key,
                audit=audit,
            )
            return {
                **values,
                "session_active": True,
                **await supervisor._source_snapshot(candidate, workspace),
                **await supervisor._source_lineage(),
            }

    async def source_resolve_dependencies(
        self,
        *,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        supervisor = self._supervisor
        supervisor._require_started()
        supervisor._require_source_mutation()
        resolver = supervisor._dependency_resolver
        if resolver is None:
            raise supervisor._evolution_error("autonomous dependency resolution is unavailable")
        candidate_id = str(expected_candidate_id or "").strip()
        diff_sha256 = str(expected_diff_sha256 or "").strip().lower()
        if not candidate_id or len(candidate_id) > 100:
            raise ValueError("expected source candidate is invalid")
        if len(diff_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in diff_sha256
        ):
            raise ValueError("expected source diff digest is invalid")
        session_key, audit = supervisor._source_context(audit_context)
        async with supervisor._source_lock(session_key):
            candidate, workspace = await supervisor._open_source_session(
                session_key=session_key,
                audit=audit,
            )
            if candidate.id != candidate_id:
                raise supervisor._evolution_error("source dependency proposal identity changed")
            await supervisor._require_current_source_base(candidate)
            before = await supervisor._source_snapshot(candidate, workspace, include_diff=False)
            if before["diff_sha256"] != diff_sha256 or not before["dirty"]:
                raise supervisor._evolution_error("source dependency proposal changed")
            if before["conflict_paths"]:
                raise supervisor._evolution_error(
                    "source dependency proposal has unresolved conflicts"
                )
            resolved = await resolver.resolve(workspace.path)
            await asyncio.to_thread(supervisor._install_resolved_lock, workspace.path, resolved)
            current = await supervisor._required_candidate(candidate.id)
            after = await supervisor._source_snapshot(current, workspace)
            return {
                **after,
                "dependency_base_id": resolved.id,
                "dependency_lock_hash": resolved.lock_sha256,
                "dependency_inventory_sha256": resolved.inventory_sha256,
                "dependency_resolver_fingerprint": resolved.resolver_fingerprint,
                "dependency_wheelhouse_sha256": resolved.wheelhouse_sha256,
            }

    async def source_release(
        self,
        *,
        idempotency_key: str,
        expected_candidate_id: str,
        expected_diff_sha256: str,
        message: str = "OpenTulpa self-update",
        audit_context: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        supervisor = self._supervisor
        supervisor._require_started()
        supervisor._require_source_mutation()
        safe_key = str(idempotency_key or "").strip()
        safe_candidate_id = str(expected_candidate_id or "").strip()
        safe_diff_sha256 = str(expected_diff_sha256 or "").strip().lower()
        if not safe_key or len(safe_key) > 200:
            raise ValueError("source release idempotency key is invalid")
        if not safe_candidate_id or len(safe_candidate_id) > 100:
            raise ValueError("expected source candidate is invalid")
        if len(safe_diff_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in safe_diff_sha256
        ):
            raise ValueError("expected source diff digest is invalid")
        safe_message = " ".join(str(message or "").split())[:500] or "OpenTulpa self-update"
        session_key, audit = supervisor._source_context(audit_context)
        tenant_id = str(audit["tenant_id"])
        async with supervisor._source_lock(session_key):
            operation = await supervisor._archive.get_source_release_operation(
                tenant_id=tenant_id,
                idempotency_key=safe_key,
            )
            if operation is not None:
                if (
                    operation.message != safe_message
                    or operation.candidate_id != safe_candidate_id
                    or operation.expected_diff_sha256 != safe_diff_sha256
                ):
                    raise supervisor._evolution_error(
                        "source release idempotency key was used for another request"
                    )
                if operation.status is SourceReleaseOperationStatus.COMPLETED:
                    return dict(operation.result or {})
                return dict(await supervisor._execute_source_release(operation))
            candidate = await supervisor._find_source_session(session_key, tenant_id=tenant_id)
            if candidate is None:
                raise supervisor._evolution_error("no active source session exists")
            if candidate.id != safe_candidate_id:
                raise supervisor._evolution_error("source session changed before approval")
            await supervisor._require_current_source_base(candidate)
            workspace = supervisor._source_workspace(candidate)
            supervisor._require_source_diff_binding(workspace, expected_diff_sha256=safe_diff_sha256)
            current_release = await supervisor._archive.get_current_release()
            operation_digest = hashlib.sha256(f"{tenant_id}\x00{safe_key}".encode()).hexdigest()
            operation = await supervisor._archive.create_source_release_operation(
                SourceReleaseOperation(
                    id=f"source_release_{operation_digest[:48]}",
                    tenant_id=tenant_id,
                    idempotency_key=safe_key,
                    candidate_id=candidate.id,
                    expected_diff_sha256=safe_diff_sha256,
                    base_release_id=(current_release.id if current_release is not None else None),
                    message=safe_message,
                    audit_context=dict(audit),
                )
            )
            return dict(await supervisor._execute_source_release(operation))

__all__ = ["SourceSessionService"]
