"""MCP transport adapter and optional LangChain implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from opentulpa.capabilities.models import WorkerSpec, WorkerTransport
from opentulpa.mcp.models import MCPCallMetadata, MCPRemoteTool
from opentulpa.mcp.schema import normalize_tool_schema, tool_schema_digest


class MCPAdapterError(RuntimeError):
    """MCP discovery or invocation failed at the transport boundary."""


class MCPAdapterUnavailableError(MCPAdapterError):
    """The optional LangChain MCP adapter dependency is not installed."""


class MCPTransportAdapter(Protocol):
    async def discover(self, worker: WorkerSpec) -> Sequence[MCPRemoteTool]: ...

    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any: ...


class LangChainMCPAdapter:
    """Lazy adapter over ``langchain-mcp-adapters``.

    The trusted metadata is passed through LangChain ``configurable`` state rather
    than MCP arguments. A configured MCP interceptor can turn it into authenticated
    transport headers without exposing ownership fields to the model schema.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._tools: dict[tuple[str, str], Any] = {}

    @classmethod
    def from_worker(
        cls,
        worker: WorkerSpec,
        *,
        environment: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> LangChainMCPAdapter:
        try:
            from langchain_mcp_adapters.client import (  # type: ignore[import-not-found]
                MultiServerMCPClient,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MCPAdapterUnavailableError(
                "install langchain-mcp-adapters to host MCP workers"
            ) from exc

        if worker.transport is WorkerTransport.STDIO:
            server: dict[str, Any] = {
                "transport": "stdio",
                "command": worker.command[0],
                "args": list(worker.command[1:]),
            }
            if environment:
                server["env"] = dict(environment)
        else:
            server = {
                "transport": "http",
                "url": worker.endpoint,
            }
            if headers:
                server["headers"] = dict(headers)
        return cls(MultiServerMCPClient(cast(Any, {worker.name: server})))

    async def discover(self, worker: WorkerSpec) -> Sequence[MCPRemoteTool]:
        try:
            try:
                tools = await self._client.get_tools(server_name=worker.name)
            except TypeError:
                tools = await self._client.get_tools()
        except Exception as exc:
            raise MCPAdapterError("MCP tool discovery failed") from exc

        discovered: list[MCPRemoteTool] = []
        for tool in tools:
            name = str(getattr(tool, "name", "") or "").strip()
            schema_model = getattr(tool, "args_schema", None)
            if schema_model is not None and hasattr(schema_model, "model_json_schema"):
                raw_schema = schema_model.model_json_schema()
            else:
                raw_schema = getattr(tool, "args", None)
                if isinstance(raw_schema, dict) and raw_schema.get("type") != "object":
                    raw_schema = {"type": "object", "properties": raw_schema}
            if not isinstance(raw_schema, dict):
                raise MCPAdapterError(f"MCP tool {name!r} has no object input schema")
            schema = normalize_tool_schema(raw_schema)
            discovered.append(
                MCPRemoteTool(
                    name=name,
                    description=str(getattr(tool, "description", "") or ""),
                    input_schema=schema,
                    schema_digest=tool_schema_digest(schema),
                )
            )
            self._tools[(worker.name, name)] = tool
        return tuple(discovered)

    async def invoke(
        self,
        worker: WorkerSpec,
        tool_name: str,
        arguments: Mapping[str, Any],
        metadata: MCPCallMetadata,
    ) -> Any:
        tool = self._tools.get((worker.name, tool_name))
        if tool is None:
            raise MCPAdapterError("MCP tool was not discovered before invocation")
        try:
            return await tool.ainvoke(
                dict(arguments),
                config={
                    "configurable": {
                        "opentulpa_mcp_context": metadata.model_dump(mode="json")
                    }
                },
            )
        except Exception as exc:
            raise MCPAdapterError("MCP tool invocation failed") from exc


__all__ = [
    "LangChainMCPAdapter",
    "MCPAdapterError",
    "MCPAdapterUnavailableError",
    "MCPTransportAdapter",
]
