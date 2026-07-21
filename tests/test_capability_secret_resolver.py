from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from opentulpa.capabilities import (
    CAPABILITY_API_SCOPES,
    CAPABILITY_CREDENTIAL_PREFIX,
    TELEGRAM_CAPABILITY,
    CapabilityAPICredentialService,
    CapabilityCredentialStore,
)
from opentulpa.secrets import (
    AesGcmHostKeyCipher,
    SecretGrantError,
    SecretVault,
    VaultCapabilitySecretResolver,
)
from opentulpa.specs import AgentRunBinding, AgentSpecRef


def _vault(tmp_path: Path) -> SecretVault:
    return SecretVault(
        tmp_path / "secrets.db",
        cipher=AesGcmHostKeyCipher(b"x" * 32),
    )


@pytest.mark.asyncio
async def test_resolver_scopes_handle_and_ignores_wrong_source_host_secrets(
    tmp_path: Path,
) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-1",
        secret_id="telegram_bot_token",
        name="telegram_bot_token",
        scopes=("telegram.receive", "telegram.send"),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-1",
        secret_id=pending.id,
        expected_revision=1,
        value=SecretStr("123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        updated_by="owner",
    )
    resolver = VaultCapabilitySecretResolver(
        vault,
        host_secrets={
            "OPENTULPA_AGENT_API_TOKEN": "agent-api-token",
            "UNDECLARED": "must-not-leak",
        },
    )

    bindings = await resolver.bind(
        tenant_id="tenant-1",
        manifest=TELEGRAM_CAPABILITY,
        secret_handles={"TELEGRAM_BOT_TOKEN": pending.id},
    )
    values = await resolver.resolve(
        tenant_id="tenant-1",
        actor_id="owner",
        instance_id="cap_tenant_telegram_g1",
        manifest=TELEGRAM_CAPABILITY,
        secret_bindings=bindings,
    )

    assert values == {
        "TELEGRAM_BOT_TOKEN": "123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH",
        "OPENTULPA_TELEGRAM_PAIRING_CODE": "ABCDEFGH",
    }


@pytest.mark.asyncio
async def test_resolver_rejects_cross_tenant_handle(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="telegram_bot_token",
        name="telegram_bot_token",
        scopes=("telegram.send",),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value=SecretStr("123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        updated_by="owner",
    )

    with pytest.raises(SecretGrantError, match="unavailable"):
        await VaultCapabilitySecretResolver(vault).bind(
            tenant_id="tenant-b",
            manifest=TELEGRAM_CAPABILITY,
            secret_handles={"TELEGRAM_BOT_TOKEN": pending.id},
        )


@pytest.mark.asyncio
async def test_resolver_rejects_handle_without_all_manifest_scopes(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="telegram_bot_token",
        name="telegram_bot_token",
        scopes=("telegram.send",),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value=SecretStr("123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        updated_by="owner",
    )

    with pytest.raises(SecretGrantError, match="insufficient scopes"):
        await VaultCapabilitySecretResolver(vault).bind(
            tenant_id="tenant-a",
            manifest=TELEGRAM_CAPABILITY,
            secret_handles={"TELEGRAM_BOT_TOKEN": pending.id},
        )


@pytest.mark.asyncio
async def test_resolver_rejects_binding_after_secret_revision_rotates(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="telegram_bot_token",
        name="telegram_bot_token",
        scopes=("telegram.receive", "telegram.send"),
        created_by="owner",
    )
    active = vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value=SecretStr("123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        updated_by="owner",
    )
    resolver = VaultCapabilitySecretResolver(vault)
    bindings = await resolver.bind(
        tenant_id="tenant-a",
        manifest=TELEGRAM_CAPABILITY,
        secret_handles={"TELEGRAM_BOT_TOKEN": active.id},
    )
    vault.rotate(
        tenant_id="tenant-a",
        secret_id=active.id,
        expected_revision=active.revision,
        value=SecretStr("987654321:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        updated_by="owner",
    )

    with pytest.raises(SecretGrantError, match="revision changed"):
        await resolver.resolve(
            tenant_id="tenant-a",
            actor_id="owner",
            instance_id="cap_tenant_telegram_g1",
            manifest=TELEGRAM_CAPABILITY,
            secret_bindings=bindings,
        )


@pytest.mark.asyncio
async def test_resolver_issues_and_revokes_generation_scoped_agent_api_credential(
    tmp_path: Path,
) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-1",
        secret_id="telegram_bot_token",
        name="telegram_bot_token",
        scopes=("telegram.receive", "telegram.send"),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-1",
        secret_id=pending.id,
        expected_revision=1,
        value=SecretStr("123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        updated_by="owner",
    )
    credential_store = CapabilityCredentialStore(tmp_path / "capability_credentials.db")
    credential_service = CapabilityAPICredentialService(
        credential_store,
        resolve_agent_spec=lambda tenant_id, spec_id: AgentSpecRef(
            tenant_id=tenant_id,
            spec_id=spec_id,
            revision=3,
        ),
    )
    resolver = VaultCapabilitySecretResolver(
        vault,
        host_secrets={"OPENTULPA_AGENT_API_TOKEN": "full-owner-token"},
        capability_credentials=credential_service,
    )

    bindings = await resolver.bind(
        tenant_id="tenant-1",
        manifest=TELEGRAM_CAPABILITY,
        secret_handles={"TELEGRAM_BOT_TOKEN": pending.id},
    )
    agent_binding = await resolver.bind_agent_run(
        tenant_id="tenant-1",
        manifest=TELEGRAM_CAPABILITY,
    )
    assert agent_binding == AgentRunBinding(
        agent_spec=AgentSpecRef(
            tenant_id="tenant-1",
            spec_id="owner",
            revision=3,
        ),
        run_kind="owner",
        trust_class="owner",
    )
    values = await resolver.resolve(
        tenant_id="tenant-1",
        actor_id="owner",
        instance_id="cap_tenant_telegram_g7",
        manifest=TELEGRAM_CAPABILITY,
        secret_bindings=bindings,
        agent_binding=agent_binding,
    )

    token = values["OPENTULPA_AGENT_API_TOKEN"]
    assert token.startswith(CAPABILITY_CREDENTIAL_PREFIX)
    assert token != "full-owner-token"
    credential = credential_store.authenticate(token)
    assert credential is not None
    assert credential.tenant_id == "tenant-1"
    assert credential.capability_instance_id == "cap_tenant_telegram_g7"
    assert credential.scopes == CAPABILITY_API_SCOPES
    assert credential.agent_spec == AgentSpecRef(
        tenant_id="tenant-1",
        spec_id="owner",
        revision=3,
    )
    assert credential.run_kind == "owner"
    assert credential.trust_class == "owner"

    await resolver.revoke(
        tenant_id="tenant-1",
        instance_id="cap_tenant_telegram_g7",
    )
    assert credential_store.authenticate(token) is None
