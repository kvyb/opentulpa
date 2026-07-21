"""Ephemeral secret grants for trusted capability worker launches."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from opentulpa.capabilities.credentials import IssuedCapabilityCredential
from opentulpa.capabilities.models import (
    CapabilityManifest,
    CapabilitySecretBinding,
    SecretRequirement,
    SecretSource,
)
from opentulpa.secrets.models import SecretState
from opentulpa.secrets.vault import SecretGrantError, SecretVault
from opentulpa.specs import AgentRunBinding


class CapabilityCredentialLifecycle(Protocol):
    """Host-issued bearer lifecycle used while capability workers are running."""

    def resolve_agent_binding(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> AgentRunBinding | None: ...

    def issue_for_capability(
        self,
        *,
        tenant_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        agent_binding: AgentRunBinding | None,
    ) -> IssuedCapabilityCredential | None: ...

    def revoke_instance(self, *, tenant_id: str, instance_id: str) -> int: ...


class VaultCapabilitySecretResolver:
    """Redeem tenant handles once and add explicitly configured host grants."""

    def __init__(
        self,
        vault: SecretVault,
        *,
        host_secrets: Mapping[str, str] | None = None,
        capability_credentials: CapabilityCredentialLifecycle | None = None,
    ) -> None:
        self._vault = vault
        self._capability_credentials = capability_credentials
        self._host_secrets = {
            str(name): str(value)
            for name, value in dict(host_secrets or {}).items()
            if str(name) and str(value)
        }

    async def bind(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
        secret_handles: Mapping[str, str],
    ) -> Mapping[str, CapabilitySecretBinding]:
        """Bind every tenant handle to its exact active revision and required scopes."""

        requirements = _requirements(manifest)
        bindings: dict[str, CapabilitySecretBinding] = {}
        for environment_name, secret_id in secret_handles.items():
            requirement = requirements.get(environment_name)
            if requirement is None or requirement.source is not SecretSource.TENANT_HANDLE:
                raise ValueError("capability requested an undeclared tenant secret")
            handle = self._vault.get_handle(
                tenant_id=tenant_id,
                secret_id=secret_id,
            )
            if handle is None or handle.state is not SecretState.ACTIVE:
                raise SecretGrantError("capability secret handle is unavailable")
            if not set(requirement.scopes).issubset(handle.scopes):
                raise SecretGrantError("capability secret handle has insufficient scopes")
            bindings[environment_name] = CapabilitySecretBinding(
                handle_id=handle.id,
                revision=handle.revision,
                scopes=requirement.scopes,
            )
        return bindings

    async def bind_agent_run(
        self,
        *,
        tenant_id: str,
        manifest: CapabilityManifest,
    ) -> AgentRunBinding | None:
        """Resolve an interface's immutable AgentSpec reference for a new generation."""

        issued = _requirements(manifest).get("OPENTULPA_AGENT_API_TOKEN")
        if issued is None or issued.source is not SecretSource.ISSUED:
            return None
        if self._capability_credentials is None:
            raise SecretGrantError("Agent API credential issuance is unavailable")
        return self._capability_credentials.resolve_agent_binding(
            tenant_id=tenant_id,
            manifest=manifest,
        )

    async def resolve(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        instance_id: str,
        manifest: CapabilityManifest,
        secret_bindings: Mapping[str, CapabilitySecretBinding],
        agent_binding: AgentRunBinding | None = None,
    ) -> Mapping[str, str]:
        del actor_id
        requirements = _requirements(manifest)
        resolved = {
            name: value
            for name, value in self._host_secrets.items()
            if (requirement := requirements.get(name)) is not None
            and requirement.source is SecretSource.HOST
        }
        for environment_name, binding in secret_bindings.items():
            requirement = requirements.get(environment_name)
            if requirement is None or requirement.source is not SecretSource.TENANT_HANDLE:
                raise ValueError("capability requested an undeclared tenant secret")
            if binding.scopes != requirement.scopes:
                raise SecretGrantError("capability secret binding scopes do not match")
            handle = self._vault.get_handle(
                tenant_id=tenant_id,
                secret_id=binding.handle_id,
            )
            if handle is None or handle.state is not SecretState.ACTIVE:
                raise SecretGrantError("capability secret handle is unavailable")
            if handle.revision != binding.revision:
                raise SecretGrantError("capability secret binding revision changed")
            grant = self._vault.issue_grant(
                tenant_id=tenant_id,
                secret_id=handle.id,
                capability_id=instance_id,
                scopes=binding.scopes,
                ttl_seconds=60,
            )
            material = self._vault.redeem_grant(
                token=grant.token,
                capability_id=instance_id,
                scope=binding.scopes[0],
            )
            resolved[environment_name] = material.value.get_secret_value()

        issued_requirement = requirements.get("OPENTULPA_AGENT_API_TOKEN")
        if (
            self._capability_credentials is not None
            and issued_requirement is not None
            and issued_requirement.source is SecretSource.ISSUED
        ):
            issued = self._capability_credentials.issue_for_capability(
                tenant_id=tenant_id,
                instance_id=instance_id,
                manifest=manifest,
                agent_binding=agent_binding,
            )
            if issued is not None:
                resolved["OPENTULPA_AGENT_API_TOKEN"] = issued.token.get_secret_value()

        if manifest.name == "telegram" and "TELEGRAM_BOT_TOKEN" in resolved:
            pairing = requirements.get("OPENTULPA_TELEGRAM_PAIRING_CODE")
            if pairing is not None and pairing.source is SecretSource.HOST:
                resolved.setdefault(
                    "OPENTULPA_TELEGRAM_PAIRING_CODE",
                    resolved["TELEGRAM_BOT_TOKEN"][-8:],
                )
        return resolved

    async def revoke(self, *, tenant_id: str, instance_id: str) -> None:
        if self._capability_credentials is not None:
            self._capability_credentials.revoke_instance(
                tenant_id=tenant_id,
                instance_id=instance_id,
            )


def _requirements(manifest: CapabilityManifest) -> dict[str, SecretRequirement]:
    requirements: dict[str, SecretRequirement] = {}
    for requirement in (
        *manifest.secrets,
        *(item for worker in manifest.workers for item in worker.secrets),
    ):
        existing = requirements.get(requirement.name)
        if existing is None or (requirement.required and not existing.required):
            requirements[requirement.name] = requirement
    return requirements


__all__ = ["CapabilityCredentialLifecycle", "VaultCapabilitySecretResolver"]
