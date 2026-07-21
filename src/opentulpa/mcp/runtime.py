"""Lifecycle bridge from active MCP capability workers into Deep Agents tools."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from urllib.parse import urlsplit

from opentulpa.capabilities.models import CapabilityManifest, WorkerKind, WorkerSpec
from opentulpa.deep_agent.dynamic_tools import TenantDynamicToolRegistry
from opentulpa.mcp.adapter import LangChainMCPAdapter, MCPTransportAdapter
from opentulpa.mcp.broker import MCPToolBroker
from opentulpa.mcp.langchain_tools import build_mcp_tool_bundle

AdapterFactory = Callable[..., MCPTransportAdapter]


class MCPToolRuntime:
    """Discover exact manifest tools and publish one atomic tenant bundle."""

    def __init__(
        self,
        *,
        broker: MCPToolBroker,
        tools: TenantDynamicToolRegistry,
        adapter_factory: AdapterFactory | None = None,
    ) -> None:
        self._broker = broker
        self._tools = tools
        self._adapter_factory = adapter_factory or self._default_adapter
        self._tenants: dict[str, str] = {}

    async def start(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        config: Mapping[str, object],
        secrets: Mapping[str, str],
        worker_endpoints: Mapping[str, str],
        worker_endpoint_headers: Mapping[str, Mapping[str, str]] | None = None,
        replace_instance_id: str | None = None,
    ) -> None:
        workers = tuple(worker for worker in manifest.workers if worker.kind is WorkerKind.MCP)
        if not workers:
            return
        if instance_id in self._tenants:
            raise RuntimeError("MCP capability instance is already published")
        try:
            for worker in workers:
                endpoint = worker_endpoints.get(worker.name)
                resolved_worker = (
                    worker.model_copy(update={"endpoint": endpoint})
                    if endpoint is not None
                    else worker
                )
                environment = {
                    **secrets,
                    "OPENTULPA_CAPABILITY_CONFIG": json.dumps(
                        dict(config),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                    "OPENTULPA_CAPABILITY_INSTANCE_ID": instance_id,
                    "OPENTULPA_CAPABILITY_NAME": manifest.name,
                    "OPENTULPA_CAPABILITY_REVISION": str(manifest.revision),
                    "OPENTULPA_WORKER_NAME": worker.name,
                }
                endpoint_headers = dict((worker_endpoint_headers or {}).get(worker.name, {}))
                parameters = inspect.signature(self._adapter_factory).parameters
                if endpoint_headers and len(parameters) < 4:
                    raise ValueError("the MCP adapter factory cannot authenticate the worker endpoint")
                adapter = (
                    self._adapter_factory(  # type: ignore[call-arg]
                        resolved_worker,
                        config,
                        environment,
                        endpoint_headers,
                    )
                    if len(parameters) >= 4
                    else self._adapter_factory(resolved_worker, config, environment)
                )
                await self._broker.register(
                    instance_id=instance_id,
                    manifest=manifest,
                    worker_name=worker.name,
                    adapter=adapter,
                )
            bundle = build_mcp_tool_bundle(self._broker, instance_id=instance_id)
            self._tools.register(
                tenant_id=tenant_id,
                instance_id=instance_id,
                tools=bundle.tools,
                interrupt_on=bundle.interrupt_on,
                capability_name=manifest.name,
                replace_instance_id=replace_instance_id,
                namespace_reserved=True,
            )
            self._tenants[instance_id] = tenant_id
            if replace_instance_id is not None and replace_instance_id != instance_id:
                self._tenants.pop(replace_instance_id, None)
                await self._broker.unregister(replace_instance_id)
        except Exception:
            await self._broker.unregister(instance_id)
            raise

    async def replace(
        self,
        *,
        tenant_id: str,
        previous_instance_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        config: Mapping[str, object],
        secrets: Mapping[str, str],
        worker_endpoints: Mapping[str, str],
        worker_endpoint_headers: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        """Atomically replace one capability's model-visible MCP generation."""

        workers = tuple(worker for worker in manifest.workers if worker.kind is WorkerKind.MCP)
        if not workers:
            await self.stop(previous_instance_id)
            return
        replacement = (
            previous_instance_id if previous_instance_id in self._tenants else None
        )
        await self.start(
            tenant_id=tenant_id,
            instance_id=instance_id,
            manifest=manifest,
            config=config,
            secrets=secrets,
            worker_endpoints=worker_endpoints,
            worker_endpoint_headers=worker_endpoint_headers,
            replace_instance_id=replacement,
        )
        if replacement is None:
            await self.stop(previous_instance_id)

    async def stop(self, instance_id: str) -> None:
        tenant_id = self._tenants.pop(instance_id, None)
        if tenant_id is not None:
            self._tools.unregister(tenant_id=tenant_id, instance_id=instance_id)
        await self._broker.unregister(instance_id)

    @staticmethod
    def _default_adapter(
        worker: WorkerSpec,
        config: Mapping[str, object],
        environment: Mapping[str, str],
        headers: Mapping[str, str] | None = None,
    ) -> MCPTransportAdapter:
        del config
        if worker.endpoint is not None:
            parsed = urlsplit(worker.endpoint)
            loopback = parsed.hostname in {"127.0.0.1", "localhost"}
            stable_proxy = bool(
                headers
                and headers.get("X-OpenTulpa-Capability-Worker-Token")
                and headers.get("X-OpenTulpa-Release-ID")
                and headers.get("X-OpenTulpa-Lease-Epoch")
                and headers.get("X-OpenTulpa-Control-Token")
            )
            if parsed.scheme != "http" or not (loopback or stable_proxy):
                raise ValueError(
                    "MCP HTTP workers require loopback or the authenticated stable proxy"
                )
            return LangChainMCPAdapter.from_worker(worker, headers=headers)
        return LangChainMCPAdapter.from_worker(worker, environment=environment)


__all__ = ["AdapterFactory", "MCPToolRuntime"]
