from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import SecretStr

from opentulpa.host.models import HostConfigInput
from opentulpa.host.store import HostConfigConflictError, HostStore
from opentulpa.secrets.cipher import AesGcmHostKeyCipher


def _store(tmp_path: Path) -> HostStore:
    return HostStore(tmp_path / "host.db", cipher=AesGcmHostKeyCipher(b"k" * 32))


def test_host_claim_hashes_credentials_and_cannot_be_reclaimed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    setup = "setup-token-with-enough-entropy"
    owner = "owner-token-with-at-least-thirty-two-characters"
    store.configure_setup_token(setup)

    store.claim(setup_token=setup, owner_token=owner)

    assert store.claimed
    assert store.authorize_owner(owner)
    assert not store.authorize_owner("wrong")
    database = (tmp_path / "host.db").read_bytes()
    assert setup.encode() not in database
    assert owner.encode() not in database
    with pytest.raises(HostConfigConflictError):
        store.claim(setup_token=setup, owner_token="another-owner-token-with-enough-characters")


def test_configuration_is_encrypted_revisioned_and_atomic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.stage(
        HostConfigInput(
            api_key=SecretStr("provider-secret-one"),
            model="moonshotai/kimi-k3",
            telegram_bot_token=SecretStr("123:telegram-secret"),
            telegram_user_id=7,
        )
    )
    assert first.status == "staged"
    store.activate(first.revision)

    second = store.stage(
        HostConfigInput(
            expected_revision=first.revision,
            api_key=None,
            base_url="https://models.example/v1/",
            model="z-ai/glm-5.2",
            telegram_bot_token=None,
            telegram_user_id=7,
        )
    )
    store.fail(second.revision, "candidate failed")

    active = store.active()
    assert active is not None
    assert active.revision == first.revision
    assert active.api_key.get_secret_value() == "provider-secret-one"
    assert active.telegram_bot_token is not None
    assert active.telegram_bot_token.get_secret_value() == "123:telegram-secret"
    assert store.get(second.revision).status == "failed"  # type: ignore[union-attr]
    assert b"provider-secret-one" not in (tmp_path / "host.db").read_bytes()
    assert b"telegram-secret" not in (tmp_path / "host.db").read_bytes()

    with pytest.raises(HostConfigConflictError):
        store.stage(
            HostConfigInput(
                expected_revision=999,
                model="z-ai/glm-5.2",
            )
        )


def test_first_configuration_accepts_telegram_token_without_owner_id(tmp_path: Path) -> None:
    store = _store(tmp_path)

    config = store.stage(
        HostConfigInput(
            api_key=SecretStr("provider-secret"),
            telegram_bot_token=SecretStr("123456789:abcdefghijklmnopqrstuvwxyzABCDEFGH"),
        )
    )

    assert config.telegram_user_id is None
    assert config.telegram_pairing_code == SecretStr("ABCDEFGH")
    assert store.view(config).telegram_pairing_required is True


def test_host_database_has_exactly_one_active_revision(tmp_path: Path) -> None:
    store = _store(tmp_path)
    one = store.stage(HostConfigInput(api_key=SecretStr("secret-one")))
    store.activate(one.revision)
    two = store.stage(
        HostConfigInput(
            expected_revision=one.revision,
            api_key=SecretStr("secret-two"),
        )
    )
    store.activate(two.revision)

    with sqlite3.connect(tmp_path / "host.db") as connection:
        rows = connection.execute(
            "SELECT revision FROM host_config_revisions WHERE status='active'"
        ).fetchall()
    assert rows == [(two.revision,)]
    assert store.get(one.revision).status == "inactive"  # type: ignore[union-attr]
