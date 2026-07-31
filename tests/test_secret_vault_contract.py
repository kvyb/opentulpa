from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from opentulpa.secrets import (
    AesGcmHostKeyCipher,
    SecretGrantError,
    SecretState,
    SecretVault,
    SecretVaultService,
)

NOW = datetime(2026, 7, 20, 12, tzinfo=UTC)


def _vault(tmp_path: Path) -> SecretVault:
    return SecretVault(
        tmp_path / "secrets.db",
        cipher=AesGcmHostKeyCipher(b"k" * 32),
        clock=lambda: NOW,
    )


def test_pending_handle_and_encrypted_storage_never_expose_plaintext(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        name="telegram_bot_token",
        scopes=("telegram.receive", "telegram.send"),
        created_by="owner",
    )

    assert pending.state is SecretState.PENDING
    assert set(pending.model_dump()) == {
        "tenant_id",
        "id",
        "revision",
        "name",
        "state",
        "scopes",
        "created_at",
        "created_by",
    }

    plaintext = "123456:telegram-super-secret-token"
    active = vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=pending.revision,
        value=plaintext,
        updated_by="owner",
    )
    assert active.state is SecretState.ACTIVE
    assert active.revision == 2
    assert plaintext.encode() not in vault.db_path.read_bytes()

    with sqlite3.connect(vault.db_path) as conn:
        row = conn.execute(
            "SELECT key_id, nonce, ciphertext FROM secret_revisions WHERE revision = 2"
        ).fetchone()
    assert row is not None
    assert row[0] == "host-v1"
    assert len(row[1]) == 12
    assert plaintext.encode() not in row[2]


def test_secret_plaintext_requires_scoped_one_time_capability_grant(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        name="telegram_bot_token",
        scopes=("telegram.receive", "telegram.send"),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value="telegram-secret",
        updated_by="owner",
    )

    grant = vault.issue_grant(
        tenant_id="tenant-a",
        secret_id=pending.id,
        capability_id="telegram",
        scopes=("telegram.send",),
    )
    assert grant.model_dump(mode="json")["token"] == "**********"

    with pytest.raises(SecretGrantError, match="another capability"):
        vault.redeem_grant(
            token=grant.token,
            capability_id="browser",
            scope="telegram.send",
        )

    material = vault.redeem_grant(
        token=grant.token,
        capability_id="telegram",
        scope="telegram.send",
    )
    assert material.value.get_secret_value() == "telegram-secret"
    assert material.model_dump(mode="json")["value"] == "**********"

    with pytest.raises(SecretGrantError, match="already consumed"):
        vault.redeem_grant(
            token=grant.token,
            capability_id="telegram",
            scope="telegram.send",
        )

    with pytest.raises(SecretGrantError, match="not allowed"):
        vault.issue_grant(
            tenant_id="tenant-a",
            secret_id=pending.id,
            capability_id="telegram",
            scopes=("browser.control",),
        )


def test_secret_service_resolves_one_shot_sandbox_mount_material(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    service = SecretVaultService(vault)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="ssh_private_key",
        name="ssh_private_key",
        scopes=("ssh.connect",),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value="private-key",
        updated_by="owner",
    )

    material = service.resolve_for_sandbox(
        tenant_id="tenant-a",
        actor_id="owner",
        secret_id=pending.id,
        scope="ssh.connect",
        mount_type="ssh_private_key",
    )

    assert material.get_secret_value() == "private-key"


def test_secret_service_resolves_one_shot_ssh_password_mount(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    service = SecretVaultService(vault)
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="ssh_password",
        name="ssh_password",
        scopes=("ssh.connect",),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value="test-password",
        updated_by="owner",
    )

    material = service.resolve_for_sandbox(
        tenant_id="tenant-a",
        actor_id="owner",
        secret_id=pending.id,
        scope="ssh.connect",
        mount_type="ssh_password",
    )

    assert material.get_secret_value() == "test-password"
