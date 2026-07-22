"""Fail-closed broker for dynamically discovered MCP tools."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4

from opentulpa.capabilities.models import (
    CapabilityManifest,
    ToolExport,
    WorkerKind,
    WorkerSpec,
)
from opentulpa.logging.langfuse import redact_for_langfuse
from opentulpa.mcp.adapter import MCPTransportAdapter
from opentulpa.mcp.models import (
    MCPAuditEvent,
    MCPBrokerError,
    MCPBrokerResult,
    MCPCallMetadata,
    MCPRemoteTool,
    MCPToolDescriptor,
)
from opentulpa.mcp.schema import tool_schema_digest
from opentulpa.tooling.contract import (
    AgentRunContext,
    ApprovalMode,
    IdempotencyMode,
    ToolEffect,
)

_RESERVED_ARGUMENTS = frozenset(
    {
        "actor_id",
        "audit_id",
        "channel",
        "correlation_id",
        "customer_id",
        "idempotency_key",
        "run_kind",
        "tenant_id",
        "thread_id",
        "tool_call_id",
    }
)


class MCPBrokerRegistrationError(RuntimeError):
    """A worker's discovered tools do not match its approved manifest."""


class MCPApprovalHook(Protocol):
    """Optional policy hook that may only add approval requirements."""

    def requires_approval(
        self,
        descriptor: MCPToolDescriptor,
        context: AgentRunContext,
        arguments: Mapping[str, Any],
    ) -> bool: ...


class MCPAuditSink(Protocol):
    async def record(self, event: MCPAuditEvent) -> None: ...


class MCPIdempotencyStore(Protocol):
    async def get(self, key: str) -> MCPBrokerResult | None: ...

    async def put(self, key: str, result: MCPBrokerResult) -> None: ...


class InMemoryMCPAuditSink:
    """Test/dev audit sink; production composition should provide durable storage."""

    def __init__(self) -> None:
        self.events: list[MCPAuditEvent] = []
        self._lock = asyncio.Lock()

    async def record(self, event: MCPAuditEvent) -> None:
        async with self._lock:
            self.events.append(event)


class InMemoryMCPIdempotencyStore:
    """Reference idempotency hook used when no durable adapter is configured."""

    def __init__(self) -> None:
        self._values: dict[str, MCPBrokerResult] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> MCPBrokerResult | None:
        async with self._lock:
            return self._values.get(key)

    async def put(self, key: str, result: MCPBrokerResult) -> None:
        async with self._lock:
            self._values.setdefault(key, result)


@dataclass(frozen=True, slots=True)
class _Binding:
    instance_id: str
    manifest: CapabilityManifest
    worker: WorkerSpec
    policy: ToolExport
    remote: MCPRemoteTool
    adapter: MCPTransportAdapter

    @property
    def descriptor(self) -> MCPToolDescriptor:
        return MCPToolDescriptor(
            instance_id=self.instance_id,
            capability_name=self.manifest.name,
            capability_revision=self.manifest.revision,
            worker_name=self.worker.name,
            name=self.remote.name,
            description=self.remote.description or self.policy.description,
            input_schema=self.remote.input_schema,
            policy=self.policy,
        )


class MCPToolBroker:
    """Bind MCP discovery to manifest policy and trusted run context."""

    def __init__(
        self,
        *,
        audit_sink: MCPAuditSink | None = None,
        idempotency_store: MCPIdempotencyStore | None = None,
        approval_hook: MCPApprovalHook | None = None,
        redactor: Callable[[Any], Any] = redact_for_langfuse,
    ) -> None:
        self._audit = audit_sink or InMemoryMCPAuditSink()
        self._idempotency = idempotency_store or InMemoryMCPIdempotencyStore()
        self._approval_hook = approval_hook
        self._redact = redactor
        self._bindings: dict[tuple[str, str], _Binding] = {}
        self._execution_locks: dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        *,
        instance_id: str,
        manifest: CapabilityManifest,
        worker_name: str,
        adapter: MCPTransportAdapter,
    ) -> tuple[MCPToolDescriptor, ...]:
        """Discover one worker and expose nothing unless its contract matches exactly."""

        safe_instance_id = str(instance_id or "").strip()
        if not safe_instance_id:
            raise ValueError("MCP capability instance_id is required")
        worker = next((item for item in manifest.workers if item.name == worker_name), None)
        if worker is None or worker.kind is not WorkerKind.MCP:
            raise MCPBrokerRegistrationError("manifest MCP worker was not found")
        discovered_values = tuple(await adapter.discover(worker))
        if any(not isinstance(item, MCPRemoteTool) for item in discovered_values):
            raise MCPBrokerRegistrationError("MCP adapter returned invalid discovery metadata")
        discovered = {item.name: item for item in discovered_values}
        if len(discovered) != len(discovered_values):
            raise MCPBrokerRegistrationError("MCP server returned duplicate tool names")
        declared = {item.name: item for item in worker.tools}
        if set(discovered) != set(declared):
            raise MCPBrokerRegistrationError(
                "MCP discovered tools do not exactly match the capability manifest"
            )
        for name, remote in discovered.items():
            properties = remote.input_schema.get("properties", {})
            reserved_properties = (
                _RESERVED_ARGUMENTS.intersection(key.lower() for key in properties)
                if isinstance(properties, dict)
                else set()
            )
            if reserved_properties:
                raise MCPBrokerRegistrationError(
                    f"MCP tool {name!r} exposes reserved runtime fields"
                )
            discovered_digest = tool_schema_digest(remote.input_schema)
            if remote.schema_digest != discovered_digest:
                raise MCPBrokerRegistrationError(
                    f"MCP tool {name!r} reported an invalid input schema digest"
                )
            if discovered_digest != declared[name].schema_digest:
                raise MCPBrokerRegistrationError(
                    f"MCP tool {name!r} input schema digest does not match the manifest"
                )

        bindings = tuple(
            _Binding(
                instance_id=safe_instance_id,
                manifest=manifest,
                worker=worker,
                policy=declared[name],
                remote=discovered[name],
                adapter=adapter,
            )
            for name in sorted(discovered)
        )
        async with self._lock:
            collisions = [
                binding.remote.name
                for binding in bindings
                if (safe_instance_id, binding.remote.name) in self._bindings
            ]
            if collisions:
                raise MCPBrokerRegistrationError(
                    f"MCP tools are already registered: {', '.join(collisions)}"
                )
            self._bindings.update(
                {
                    (safe_instance_id, binding.remote.name): binding
                    for binding in bindings
                }
            )
        return tuple(binding.descriptor for binding in bindings)

    async def unregister(self, instance_id: str) -> None:
        async with self._lock:
            self._bindings = {
                key: value for key, value in self._bindings.items() if key[0] != instance_id
            }

    def descriptors(self, instance_id: str) -> tuple[MCPToolDescriptor, ...]:
        return tuple(
            binding.descriptor
            for (registered_instance, _), binding in sorted(self._bindings.items())
            if registered_instance == instance_id
        )

    def interrupt_on(self, instance_id: str) -> dict[str, bool]:
        """Return the conservative static Deep Agents interrupt policy."""

        return {
            descriptor.name: (
                self._core_requires_approval(descriptor.policy)
                or self._approval_hook is not None
            )
            for descriptor in self.descriptors(instance_id)
        }

    async def invoke(
        self,
        *,
        instance_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        context: AgentRunContext,
        tool_call_id: str,
        approval_granted: bool = False,
        idempotency_key: str | None = None,
    ) -> MCPBrokerResult:
        audit_id = f"audit_{uuid4().hex}"
        binding = self._bindings.get((instance_id, tool_name))
        if binding is None:
            return self._error(
                audit_id=audit_id,
                code="tool_unavailable",
                message="The requested capability tool is unavailable.",
            )
        safe_arguments = dict(arguments)
        reserved = _RESERVED_ARGUMENTS.intersection(key.lower() for key in safe_arguments)
        if reserved:
            return await self._reject(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                audit_id=audit_id,
                arguments=safe_arguments,
                code="reserved_argument",
                message="Trusted runtime fields cannot be supplied as tool arguments.",
                idempotency_key=None,
                approval_granted=approval_granted,
            )
        argument_error = self._validate_arguments(binding.remote, safe_arguments)
        if argument_error is not None:
            return await self._reject(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                audit_id=audit_id,
                arguments=safe_arguments,
                code="invalid_arguments",
                message=argument_error,
                idempotency_key=None,
                approval_granted=approval_granted,
            )

        try:
            safe_key = self._idempotency_key(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                arguments=safe_arguments,
                supplied=idempotency_key,
            )
        except ValueError:
            return await self._reject(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                audit_id=audit_id,
                arguments=safe_arguments,
                code="invalid_idempotency_key",
                message="The trusted idempotency key is invalid.",
                idempotency_key=None,
                approval_granted=approval_granted,
            )
        descriptor = binding.descriptor
        requires_approval = self._core_requires_approval(binding.policy)
        if self._approval_hook is not None:
            requires_approval = requires_approval or self._approval_hook.requires_approval(
                descriptor,
                context,
                safe_arguments,
            )
        if requires_approval and not approval_granted:
            result = MCPBrokerResult(
                status="approval_required",
                audit_id=audit_id,
                idempotency_key=safe_key,
            )
            await self._record(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                result=result,
                arguments=safe_arguments,
                approval_granted=False,
            )
            return result

        return await self._execute_binding(
            binding=binding,
            context=context,
            tool_call_id=tool_call_id,
            arguments=safe_arguments,
            approval_granted=approval_granted,
            audit_id=audit_id,
            idempotency_key=safe_key,
        )

    async def _execute_binding(
        self,
        *,
        binding: _Binding,
        context: AgentRunContext,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        approval_granted: bool,
        audit_id: str,
        idempotency_key: str | None,
    ) -> MCPBrokerResult:
        execution_lock: asyncio.Lock | None = None
        if idempotency_key is not None:
            async with self._lock:
                execution_lock = self._execution_locks.setdefault(
                    idempotency_key,
                    asyncio.Lock(),
                )
            await execution_lock.acquire()
        try:
            if idempotency_key is not None:
                cached = await self._idempotency.get(idempotency_key)
                if cached is not None:
                    replay = cached.model_copy(update={"audit_id": audit_id, "replayed": True})
                    await self._record(
                        binding=binding,
                        context=context,
                        tool_call_id=tool_call_id,
                        result=replay,
                        arguments=arguments,
                        approval_granted=approval_granted,
                        outcome="replayed",
                    )
                    return replay

            metadata = MCPCallMetadata(
                tenant_id=context.tenant_id,
                actor_id=context.actor_id,
                thread_id=context.thread_id,
                correlation_id=context.correlation_id,
                tool_call_id=tool_call_id,
                audit_id=audit_id,
                idempotency_key=idempotency_key,
            )
            started = MCPBrokerResult(
                status="ok",
                audit_id=audit_id,
                idempotency_key=idempotency_key,
            )
            await self._record(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                result=started,
                arguments=arguments,
                approval_granted=approval_granted,
                outcome="started",
            )
            try:
                async with asyncio.timeout(binding.policy.timeout_seconds):
                    raw_result = await binding.adapter.invoke(
                        binding.worker,
                        binding.remote.name,
                        arguments,
                        metadata,
                    )
                result = MCPBrokerResult(
                    status="ok",
                    data=self._redact(raw_result),
                    audit_id=audit_id,
                    idempotency_key=idempotency_key,
                )
                if idempotency_key is not None:
                    await self._idempotency.put(idempotency_key, result)
            except TimeoutError:
                result = self._error(
                    audit_id=audit_id,
                    code="timeout",
                    message="The capability tool timed out.",
                    retryable=True,
                    idempotency_key=idempotency_key,
                )
            except Exception:
                result = self._error(
                    audit_id=audit_id,
                    code="operation_failed",
                    message="The capability tool could not be completed.",
                    idempotency_key=idempotency_key,
                )
            await self._record(
                binding=binding,
                context=context,
                tool_call_id=tool_call_id,
                result=result,
                arguments=arguments,
                approval_granted=approval_granted,
            )
            return result
        finally:
            if execution_lock is not None:
                execution_lock.release()

    @staticmethod
    def _core_requires_approval(policy: ToolExport) -> bool:
        return (
            policy.approval is not ApprovalMode.AUTO
            or policy.effect is not ToolEffect.READ
        )

    @staticmethod
    def _validate_arguments(remote: MCPRemoteTool, arguments: Mapping[str, Any]) -> str | None:
        schema = remote.input_schema
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [name for name in required if name not in arguments]
        if missing:
            return f"Missing required arguments: {', '.join(sorted(missing))}."
        if schema.get("additionalProperties") is False:
            unknown = set(arguments).difference(properties)
            if unknown:
                return f"Unknown arguments: {', '.join(sorted(unknown))}."
        try:
            json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return "Tool arguments must contain only JSON values."
        return None

    @staticmethod
    def _idempotency_key(
        *,
        binding: _Binding,
        context: AgentRunContext,
        tool_call_id: str,
        arguments: Mapping[str, Any],
        supplied: str | None,
    ) -> str | None:
        if binding.policy.idempotency is IdempotencyMode.NONE:
            return None
        if supplied is not None:
            safe = str(supplied).strip()
            if not safe or len(safe) > 500:
                raise ValueError("idempotency_key must contain 1 to 500 characters")
            key_material: dict[str, Any] = {"supplied_key": safe}
        else:
            key_material = {
                "tool_call_id": tool_call_id,
                "arguments": arguments,
            }
        canonical = json.dumps(
            {
                "tenant_id": context.tenant_id,
                "instance_id": binding.instance_id,
                "capability": binding.manifest.name,
                "revision": binding.manifest.revision,
                "worker": binding.worker.name,
                "tool": binding.remote.name,
                **key_material,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return f"mcp_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    async def _reject(
        self,
        *,
        binding: _Binding,
        context: AgentRunContext,
        tool_call_id: str,
        audit_id: str,
        arguments: Mapping[str, Any],
        code: str,
        message: str,
        idempotency_key: str | None,
        approval_granted: bool,
    ) -> MCPBrokerResult:
        result = self._error(
            audit_id=audit_id,
            code=code,
            message=message,
            idempotency_key=idempotency_key,
        )
        await self._record(
            binding=binding,
            context=context,
            tool_call_id=tool_call_id,
            result=result,
            arguments=arguments,
            approval_granted=approval_granted,
        )
        return result

    async def _record(
        self,
        *,
        binding: _Binding,
        context: AgentRunContext,
        tool_call_id: str,
        result: MCPBrokerResult,
        arguments: Mapping[str, Any],
        approval_granted: bool,
        outcome: str | None = None,
    ) -> None:
        resolved_outcome = outcome or result.status
        await self._audit.record(
            MCPAuditEvent(
                audit_id=result.audit_id,
                instance_id=binding.instance_id,
                capability_name=binding.manifest.name,
                capability_revision=binding.manifest.revision,
                worker_name=binding.worker.name,
                tool_name=binding.remote.name,
                tenant_id=context.tenant_id,
                actor_id=context.actor_id,
                thread_id=context.thread_id,
                correlation_id=context.correlation_id,
                tool_call_id=tool_call_id,
                idempotency_key=result.idempotency_key,
                approval_granted=approval_granted,
                outcome=resolved_outcome,  # type: ignore[arg-type]
                arguments=self._redact(arguments),
                result=self._redact(result.data),
                error_code=result.error.code if result.error is not None else None,
            )
        )

    @staticmethod
    def _error(
        *,
        audit_id: str,
        code: str,
        message: str,
        retryable: bool = False,
        idempotency_key: str | None = None,
    ) -> MCPBrokerResult:
        return MCPBrokerResult(
            status="error",
            error=MCPBrokerError(code=code, message=message, retryable=retryable),
            audit_id=audit_id,
            idempotency_key=idempotency_key,
        )


__all__ = [
    "InMemoryMCPAuditSink",
    "InMemoryMCPIdempotencyStore",
    "MCPApprovalHook",
    "MCPAuditSink",
    "MCPBrokerRegistrationError",
    "MCPIdempotencyStore",
    "MCPToolBroker",
]
