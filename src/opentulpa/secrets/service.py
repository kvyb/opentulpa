"""Authenticated control-plane operations over the encrypted secret vault."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from pydantic import SecretStr

from opentulpa.secrets.models import SecretHandle, SecretState
from opentulpa.secrets.vault import (
    SecretGrantError,
    SecretVault,
    SecretVaultConflictError,
    SecretVaultNotFoundError,
)

logger = logging.getLogger(__name__)

SecretChangeListener = Callable[..., None]


class SecretVaultService:
    """Manage public-safe handles without exposing a plaintext read operation."""

    def __init__(self, vault: SecretVault) -> None:
        self._vault = vault
        self._change_listeners: list[SecretChangeListener] = []

    def add_change_listener(self, listener: SecretChangeListener) -> None:
        """Observe public-safe revision changes without exposing secret material."""

        if listener not in self._change_listeners:
            self._change_listeners.append(listener)

    def list(self, *, tenant_id: str) -> list[SecretHandle]:
        return self._vault.list_handles(tenant_id=tenant_id)

    def get(self, *, tenant_id: str, secret_id: str) -> SecretHandle | None:
        return self._vault.get_handle(tenant_id=tenant_id, secret_id=secret_id)

    def resolve_for_sandbox(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
        scope: str,
        mount_type: str,
    ) -> SecretStr:
        """Redeem one tenant secret for a trusted one-shot sandbox mount."""

        del actor_id
        if str(mount_type or "").strip() not in {"ssh_password", "ssh_private_key"}:
            raise SecretGrantError("sandbox secret mount type is unsupported")
        grant = self._vault.issue_grant(
            tenant_id=tenant_id,
            secret_id=secret_id,
            capability_id="sandbox_ssh_diagnostic",
            scopes=(scope,),
            ttl_seconds=60,
        )
        material = self._vault.redeem_grant(
            token=grant.token,
            capability_id="sandbox_ssh_diagnostic",
            scope=scope,
        )
        return material.value

    def create_pending(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        name: str,
        scopes: Sequence[str],
        secret_id: str | None = None,
    ) -> SecretHandle:
        return self._vault.create_pending(
            tenant_id=tenant_id,
            secret_id=secret_id,
            name=name,
            scopes=scopes,
            created_by=actor_id,
        )

    def store(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
        expected_revision: int,
        value: SecretStr,
        scopes: Sequence[str] | None = None,
    ) -> SecretHandle:
        handle = self._vault.get_handle(tenant_id=tenant_id, secret_id=secret_id)
        if handle is None:
            raise SecretVaultNotFoundError(secret_id)
        if handle.state is SecretState.PENDING:
            if scopes is not None:
                raise SecretVaultConflictError(
                    "pending secret scopes cannot change while storing its first value"
                )
            stored = self._vault.fulfill(
                tenant_id=tenant_id,
                secret_id=secret_id,
                expected_revision=expected_revision,
                value=value,
                updated_by=actor_id,
            )
            self._notify_changed(stored, actor_id=actor_id)
            return stored
        if handle.state is SecretState.ACTIVE:
            stored = self._vault.rotate(
                tenant_id=tenant_id,
                secret_id=secret_id,
                expected_revision=expected_revision,
                value=value,
                scopes=scopes,
                updated_by=actor_id,
            )
            self._notify_changed(stored, actor_id=actor_id)
            return stored
        raise SecretVaultConflictError("revoked secret handles cannot be restored")

    def revoke(
        self,
        *,
        tenant_id: str,
        actor_id: str,
        secret_id: str,
        expected_revision: int,
    ) -> SecretHandle:
        revoked = self._vault.revoke(
            tenant_id=tenant_id,
            secret_id=secret_id,
            expected_revision=expected_revision,
            updated_by=actor_id,
        )
        self._notify_changed(revoked, actor_id=actor_id)
        return revoked

    def _notify_changed(self, handle: SecretHandle, *, actor_id: str) -> None:
        for listener in tuple(self._change_listeners):
            try:
                listener(
                    tenant_id=handle.tenant_id,
                    actor_id=actor_id,
                    secret_id=handle.id,
                )
            except Exception:
                logger.exception(
                    "Secret revision listener failed: tenant=%s secret=%s",
                    handle.tenant_id,
                    handle.id,
                )


__all__ = ["SecretChangeListener", "SecretVaultService"]
