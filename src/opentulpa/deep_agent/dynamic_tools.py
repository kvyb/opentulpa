"""Tenant-scoped tools supplied by active out-of-process capabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Protocol

from langchain_core.tools import BaseTool


@dataclass(frozen=True, slots=True)
class DynamicToolSnapshot:
    """One atomic generation of model-visible tools and approval policy."""

    generation: int
    tools: tuple[BaseTool, ...]
    interrupt_on: Mapping[str, bool]
    capability_name: str | None = None


class DynamicToolProvider(Protocol):
    def snapshot(self, tenant_id: str) -> DynamicToolSnapshot: ...


class TenantDynamicToolRegistry:
    """Atomically publish capability tool bundles without recompiling the runtime."""

    def __init__(self, *, reserved_names: Sequence[str] = ()) -> None:
        self._reserved = frozenset(reserved_names)
        self._instances: dict[str, dict[str, DynamicToolSnapshot]] = {}
        self._generation: dict[str, int] = {}
        self._lock = RLock()

    def register(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        tools: Sequence[BaseTool],
        interrupt_on: Mapping[str, bool],
        capability_name: str | None = None,
        replace_instance_id: str | None = None,
        namespace_reserved: bool = False,
    ) -> DynamicToolSnapshot:
        tenant = self._identity(tenant_id, "tenant_id")
        instance = self._identity(instance_id, "instance_id")
        capability = (
            self._identity(capability_name, "capability_name")
            if capability_name is not None
            else None
        )
        replacement = (
            self._identity(replace_instance_id, "replace_instance_id")
            if replace_instance_id is not None
            else None
        )
        values = tuple(tools)
        names = [tool.name for tool in values]
        if not values:
            raise ValueError("a dynamic tool bundle cannot be empty")
        if len(names) != len(set(names)):
            raise ValueError("a dynamic tool bundle contains duplicate names")
        collisions = sorted(set(names).intersection(self._reserved))
        if collisions:
            if not namespace_reserved or capability is None:
                raise ValueError("dynamic tools collide with kernel tools: " + ", ".join(collisions))
            renamed: dict[str, str] = {
                name: f"{capability}__{name}" if name in self._reserved else name
                for name in names
            }
            values = tuple(
                tool.model_copy(update={"name": renamed[tool.name]})
                for tool in values
            )
            names = [tool.name for tool in values]
        else:
            renamed = {name: name for name in names}
        supplied_policy = dict(interrupt_on)
        if set(supplied_policy) != set(renamed):
            raise ValueError("dynamic approval policy must cover every tool exactly")
        policy = {renamed[name]: value for name, value in supplied_policy.items()}
        if any(not isinstance(value, bool) for value in policy.values()):
            raise ValueError("dynamic approval policy values must be booleans")
        with self._lock:
            current = dict(self._instances.get(tenant, {}))
            if replacement is not None and replacement != instance:
                previous = current.get(replacement)
                if previous is None:
                    raise ValueError("dynamic replacement instance is not registered")
                if capability is None or previous.capability_name != capability:
                    raise ValueError("dynamic replacement must belong to the same capability")
                current.pop(replacement)
            other_names = {
                tool.name
                for key, snapshot in current.items()
                if key != instance
                for tool in snapshot.tools
            }
            overlap = sorted(set(names).intersection(other_names))
            if overlap:
                raise ValueError("dynamic tools collide across capabilities: " + ", ".join(overlap))
            generation = self._generation.get(tenant, 0) + 1
            current[instance] = DynamicToolSnapshot(
                generation=generation,
                tools=values,
                interrupt_on=MappingProxyType(policy),
                capability_name=capability,
            )
            self._instances[tenant] = current
            self._generation[tenant] = generation
            return self._snapshot_locked(tenant)

    def unregister(self, *, tenant_id: str, instance_id: str) -> DynamicToolSnapshot:
        tenant = self._identity(tenant_id, "tenant_id")
        instance = self._identity(instance_id, "instance_id")
        with self._lock:
            current = dict(self._instances.get(tenant, {}))
            if current.pop(instance, None) is None:
                return self._snapshot_locked(tenant)
            generation = self._generation.get(tenant, 0) + 1
            self._generation[tenant] = generation
            if current:
                self._instances[tenant] = current
            else:
                self._instances.pop(tenant, None)
            return self._snapshot_locked(tenant)

    def snapshot(self, tenant_id: str) -> DynamicToolSnapshot:
        tenant = self._identity(tenant_id, "tenant_id")
        with self._lock:
            return self._snapshot_locked(tenant)

    def _snapshot_locked(self, tenant_id: str) -> DynamicToolSnapshot:
        instances = self._instances.get(tenant_id, {})
        ordered = [instances[key] for key in sorted(instances)]
        tools = tuple(tool for snapshot in ordered for tool in snapshot.tools)
        policy = {
            name: required
            for snapshot in ordered
            for name, required in snapshot.interrupt_on.items()
        }
        return DynamicToolSnapshot(
            generation=self._generation.get(tenant_id, 0),
            tools=tools,
            interrupt_on=MappingProxyType(policy),
            capability_name=None,
        )

    @staticmethod
    def _identity(value: str, label: str) -> str:
        safe = str(value or "").strip()
        if not safe or len(safe) > 200 or any(ord(char) < 32 for char in safe):
            raise ValueError(f"{label} is invalid")
        return safe


__all__ = [
    "DynamicToolProvider",
    "DynamicToolSnapshot",
    "TenantDynamicToolRegistry",
]
