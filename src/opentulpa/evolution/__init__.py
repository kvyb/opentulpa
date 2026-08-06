"""Versioned, evaluated, owner-approved OpenTulpa source evolution."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BootstrapReleaseActivator": "opentulpa.evolution.activation",
    "ReleaseActivationResult": "opentulpa.evolution.activation",
    "ReleaseActivationStatus": "opentulpa.evolution.activation",
    "ReleaseActivator": "opentulpa.evolution.activation",
    "DependencyResolutionError": "opentulpa.evolution.dependency_resolver",
    "DependencyResolverPolicy": "opentulpa.evolution.dependency_resolver",
    "ResolvedDependencyBase": "opentulpa.evolution.dependency_resolver",
    "TrustedDependencyResolver": "opentulpa.evolution.dependency_resolver",
    "OciReleaseArtifact": "opentulpa.evolution.release_builder",
    "OciReleaseBuildPolicy": "opentulpa.evolution.release_builder",
    "ReleaseBuilder": "opentulpa.evolution.release_builder",
    "ReleaseBuildError": "opentulpa.evolution.release_builder",
    "ReleaseBuildRequest": "opentulpa.evolution.release_builder",
    "TrustedOciReleaseBuilder": "opentulpa.evolution.release_builder",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
