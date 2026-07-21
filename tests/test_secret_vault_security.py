from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from opentulpa.secrets import (
    AesGcmHostKeyCipher,
    SecretCipherError,
    SecretGrantError,
    SecretState,
    SecretVault,
    SecretVaultConflictError,
)


class _Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 7, 20, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _active_vault(tmp_path: Path) -> tuple[SecretVault, _Clock]:
    clock = _Clock()
    vault = SecretVault(
        tmp_path / "secrets.db",
        cipher=AesGcmHostKeyCipher(b"k" * 32),
        clock=clock,
    )
    pending = vault.create_pending(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        name="telegram_bot_token",
        scopes=("telegram.send",),
        created_by="owner",
    )
    vault.fulfill(
        tenant_id="tenant-a",
        secret_id=pending.id,
        expected_revision=1,
        value="old-telegram-token",
        updated_by="owner",
    )
    return vault, clock


def test_rotation_and_revocation_invalidate_existing_grants(tmp_path: Path) -> None:
    vault, _ = _active_vault(tmp_path)
    stale = vault.issue_grant(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        capability_id="telegram",
        scopes=("telegram.send",),
    )
    rotated = vault.rotate(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        expected_revision=2,
        value="new-telegram-token",
        updated_by="owner",
    )
    assert rotated.revision == 3

    with pytest.raises(SecretGrantError, match="newer revision"):
        vault.redeem_grant(
            token=stale.token,
            capability_id="telegram",
            scope="telegram.send",
        )

    current = vault.issue_grant(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        capability_id="telegram",
        scopes=("telegram.send",),
    )
    revoked = vault.revoke(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        expected_revision=3,
        updated_by="owner",
    )
    assert revoked.state is SecretState.REVOKED
    assert vault.redact_text(
        tenant_id="tenant-a",
        text="old-telegram-token new-telegram-token",
    ) == "[redacted] [redacted]"

    with pytest.raises(SecretGrantError, match="newer revision"):
        vault.redeem_grant(
            token=current.token,
            capability_id="telegram",
            scope="telegram.send",
        )
    with pytest.raises(SecretGrantError, match="not active"):
        vault.issue_grant(
            tenant_id="tenant-a",
            secret_id="telegram_bot",
            capability_id="telegram",
            scopes=("telegram.send",),
        )


def test_grants_expire_and_revision_transitions_use_cas(tmp_path: Path) -> None:
    vault, clock = _active_vault(tmp_path)
    grant = vault.issue_grant(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        capability_id="telegram",
        scopes=("telegram.send",),
        ttl_seconds=10,
    )
    clock.now += timedelta(seconds=11)

    with pytest.raises(SecretGrantError, match="expired"):
        vault.redeem_grant(
            token=grant.token,
            capability_id="telegram",
            scope="telegram.send",
        )

    with pytest.raises(SecretVaultConflictError, match="revision is 2"):
        vault.rotate(
            tenant_id="tenant-a",
            secret_id="telegram_bot",
            expected_revision=1,
            value="stale-update",
            updated_by="owner",
        )


def test_redaction_is_tenant_scoped_and_authenticated_ciphertext_detects_tampering(
    tmp_path: Path,
) -> None:
    vault, _ = _active_vault(tmp_path)
    assert vault.redact_payload(
        tenant_id="tenant-a",
        value={"message": "Bearer old-telegram-token", "nested": ["old-telegram-token"]},
    ) == {"message": "Bearer [redacted]", "nested": ["[redacted]"]}
    assert vault.redact_text(
        tenant_id="tenant-b",
        text="old-telegram-token",
    ) == "old-telegram-token"

    grant = vault.issue_grant(
        tenant_id="tenant-a",
        secret_id="telegram_bot",
        capability_id="telegram",
        scopes=("telegram.send",),
    )
    with sqlite3.connect(vault.db_path) as conn:
        conn.execute(
            """
            UPDATE secret_revisions SET ciphertext = ?
            WHERE tenant_id = 'tenant-a' AND id = 'telegram_bot' AND revision = 2
            """,
            (b"tampered",),
        )
        conn.commit()

    with pytest.raises(SecretCipherError, match="failed authentication"):
        vault.redeem_grant(
            token=grant.token,
            capability_id="telegram",
            scope="telegram.send",
        )
