"""Closed registry for declarative capability manifests."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any, cast

from opentulpa.capabilities.models import CapabilityManifest


class CapabilityRegistryError(ValueError):
    """Capability manifests conflict or violate the closed registry contract."""


class CapabilityNotFoundError(KeyError):
    """A requested capability is not explicitly registered."""


class CapabilityLoadError(RuntimeError):
    """A registered capability entrypoint cannot be safely loaded."""


class CapabilityRegistry:
    """Immutable registry that rejects ambiguous names, tools, and services."""

    def __init__(self, manifests: Iterable[CapabilityManifest] = ()) -> None:
        by_name: dict[str, CapabilityManifest] = {}
        tool_owners: dict[str, str] = {}
        service_owners: dict[str, str] = {}
        ordered: list[CapabilityManifest] = []

        for manifest in manifests:
            if not isinstance(manifest, CapabilityManifest):
                raise CapabilityRegistryError(
                    "registry entries must be CapabilityManifest instances"
                )
            if manifest.name in by_name:
                raise CapabilityRegistryError(f"capability {manifest.name!r} is already registered")
            self._claim_exports("tool", manifest, manifest.exported_tools, tool_owners)
            self._claim_exports("service", manifest, manifest.services, service_owners)
            by_name[manifest.name] = manifest
            ordered.append(manifest)

        self._manifests = tuple(ordered)
        self._by_name: Mapping[str, CapabilityManifest] = MappingProxyType(by_name)
        self._tool_owners: Mapping[str, str] = MappingProxyType(tool_owners)
        self._service_owners: Mapping[str, str] = MappingProxyType(service_owners)

    @staticmethod
    def _claim_exports(
        export_type: str,
        manifest: CapabilityManifest,
        exports: tuple[str, ...],
        owners: dict[str, str],
    ) -> None:
        for export in exports:
            owner = owners.get(export)
            if owner is not None:
                raise CapabilityRegistryError(
                    f"{export_type} {export!r} is exported by both {owner!r} and {manifest.name!r}"
                )
            owners[export] = manifest.name

    @property
    def manifests(self) -> tuple[CapabilityManifest, ...]:
        return self._manifests

    def names(self) -> tuple[str, ...]:
        return tuple(manifest.name for manifest in self._manifests)

    def get(self, name: str) -> CapabilityManifest:
        safe_name = str(name or "").strip().lower()
        try:
            return self._by_name[safe_name]
        except KeyError as exc:
            raise CapabilityNotFoundError(safe_name) from exc

    def owner_for_tool(self, name: str) -> str:
        return self._owner_for_export("tool", name, self._tool_owners)

    def owner_for_service(self, name: str) -> str:
        return self._owner_for_export("service", name, self._service_owners)

    @staticmethod
    def _owner_for_export(export_type: str, name: str, owners: Mapping[str, str]) -> str:
        safe_name = str(name or "").strip().lower()
        try:
            return owners[safe_name]
        except KeyError as exc:
            raise CapabilityNotFoundError(f"{export_type}:{safe_name}") from exc

    def load_entrypoint(self, name: str) -> Callable[..., Any]:
        """Import a registered implementation only after the host enables it."""

        manifest = self.get(name)
        if manifest.module is None or manifest.entrypoint is None:
            raise CapabilityLoadError(
                f"capability {manifest.name!r} is worker-hosted and has no in-process entrypoint"
            )
        try:
            module = importlib.import_module(manifest.module)
        except Exception as exc:
            raise CapabilityLoadError(
                f"failed to import capability {manifest.name!r} module {manifest.module!r}"
            ) from exc
        entrypoint = getattr(module, manifest.entrypoint, None)
        if not callable(entrypoint):
            raise CapabilityLoadError(
                f"capability {manifest.name!r} entrypoint "
                f"{manifest.module_entrypoint!r} is not callable"
            )
        return cast(Callable[..., Any], entrypoint)


__all__ = [
    "CapabilityLoadError",
    "CapabilityNotFoundError",
    "CapabilityRegistry",
    "CapabilityRegistryError",
]
