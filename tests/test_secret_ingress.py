from __future__ import annotations

from pathlib import Path

from opentulpa.secrets import (
    AesGcmHostKeyCipher,
    SecretIngressHook,
    SecretIngressService,
    SecretVault,
    SecretVaultService,
)


def _ingress(tmp_path: Path) -> tuple[SecretIngressService, SecretVault]:
    vault = SecretVault(
        tmp_path / "secrets.db",
        cipher=AesGcmHostKeyCipher(b"k" * 32),
    )
    return SecretIngressService(SecretVaultService(vault)), vault


def _assert_hook(hook: SecretIngressHook) -> SecretIngressHook:
    return hook


def _assert_absent_from_database(vault: SecretVault, *values: str) -> None:
    for path in (vault.db_path, vault.db_path.with_name(f"{vault.db_path.name}-wal")):
        if not path.exists():
            continue
        stored = path.read_bytes()
        for value in values:
            assert value.encode() not in stored


def test_ingress_encrypts_pasted_credentials_and_returns_only_handles(tmp_path: Path) -> None:
    ingress, vault = _ingress(tmp_path)
    hook = _assert_hook(ingress)
    telegram = "1234567890:AAEabcdefghijklmnopqrstuvwxyz012345678"
    api_token = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789"

    sanitized = hook(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f"Use Telegram {telegram} and this API token {api_token}.",
    )

    assert sanitized == (
        "Use Telegram secret://telegram_bot_token and this API token "
        "secret://api_token."
    )
    assert telegram not in sanitized
    assert api_token not in sanitized
    telegram_handle = vault.get_handle(
        tenant_id="tenant-a",
        secret_id="telegram_bot_token",
    )
    api_handle = vault.get_handle(tenant_id="tenant-a", secret_id="api_token")
    assert telegram_handle is not None and telegram_handle.revision == 2
    assert api_handle is not None and api_handle.revision == 2
    _assert_absent_from_database(vault, telegram, api_token)


def test_ingress_rotates_existing_handles_and_leaves_normal_text_unchanged(
    tmp_path: Path,
) -> None:
    ingress, vault = _ingress(tmp_path)
    first_telegram = "1234567890:AAEabcdefghijklmnopqrstuvwxyz012345678"
    first_api = "sk-first_abcdefghijklmnopqrstuvwxyz0123456789"
    ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f"{first_telegram} {first_api}",
    )
    second_telegram = "9876543210:AAFabcdefghijklmnopqrstuvwxyz987654321"
    second_api = "sk-second_abcdefghijklmnopqrstuvwxyz9876543210"

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f"Replace them: {second_telegram} and {second_api}",
    )

    assert result.text == (
        "Replace them: secret://telegram_bot_token and secret://api_token"
    )
    assert {handle.id: handle.revision for handle in result.handles} == {
        "telegram_bot_token": 3,
        "api_token": 3,
    }
    assert ingress(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text="No credentials here.",
    ) == "No credentials here."
    _assert_absent_from_database(
        vault,
        first_telegram,
        first_api,
        second_telegram,
        second_api,
    )


def test_ingress_uses_separate_handles_for_multiple_tokens_of_one_kind(
    tmp_path: Path,
) -> None:
    ingress, vault = _ingress(tmp_path)
    first = "sk-first_abcdefghijklmnopqrstuvwxyz0123456789"
    second = "sk-second_abcdefghijklmnopqrstuvwxyz9876543210"

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f"primary={first} fallback={second}",
    )

    assert result.text == "primary=secret://api_token fallback=secret://api_token_2"
    assert [handle.id for handle in result.handles] == ["api_token", "api_token_2"]
    assert vault.get_handle(tenant_id="tenant-a", secret_id="api_token") is not None
    assert vault.get_handle(tenant_id="tenant-a", secret_id="api_token_2") is not None
    _assert_absent_from_database(vault, first, second)


def test_ingress_does_not_restore_a_revoked_handle(tmp_path: Path) -> None:
    ingress, vault = _ingress(tmp_path)
    first = "sk-first_abcdefghijklmnopqrstuvwxyz0123456789"
    ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=first,
    )
    revoked = vault.revoke(
        tenant_id="tenant-a",
        secret_id="api_token",
        expected_revision=2,
        updated_by="owner-a",
    )
    replacement = "sk-second_abcdefghijklmnopqrstuvwxyz9876543210"

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=replacement,
    )

    assert revoked.state == "revoked"
    assert result.text == "secret://api_token_replacement_2"
    assert result.handles[0].id == "api_token_replacement_2"
    assert vault.get_handle(tenant_id="tenant-a", secret_id="api_token") == revoked
    _assert_absent_from_database(vault, first, replacement)


def test_ingress_stores_arbitrary_named_credentials_before_model_input(
    tmp_path: Path,
) -> None:
    ingress, vault = _ingress(tmp_path)
    composio = "ak_live_composio_abcdefghijklmnopqrstuvwxyz"
    daytona = "dtn_live_abcdefghijklmnopqrstuvwxyz012345"
    github = "github_pat_11AA22BB33CC44DD55EE66FF"

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=(
            f"Configure COMPOSIO_API_KEY={composio} and "
            f"DAYTONA_API_KEY={daytona} and "
            f'GITHUB_TOKEN="{github}" for me.'
        ),
    )

    assert result.text == (
        "Configure COMPOSIO_API_KEY=secret://composio_api_key and "
        "DAYTONA_API_KEY=secret://daytona_api_key and "
        'GITHUB_TOKEN="secret://github_token" for me.'
    )
    assert [handle.id for handle in result.handles] == [
        "composio_api_key",
        "daytona_api_key",
        "github_token",
    ]
    composio_handle = vault.get_handle(
        tenant_id="tenant-a",
        secret_id="composio_api_key",
    )
    github_handle = vault.get_handle(tenant_id="tenant-a", secret_id="github_token")
    daytona_handle = vault.get_handle(tenant_id="tenant-a", secret_id="daytona_api_key")
    assert composio_handle is not None
    assert composio_handle.scopes == ("composio.manage", "composio.invoke")
    assert daytona_handle is not None
    assert daytona_handle.scopes == ("daytona.manage",)
    assert github_handle is not None
    assert github_handle.scopes == ("github.read", "github.write")
    _assert_absent_from_database(vault, composio, daytona, github)


def test_ingress_supports_named_multiline_secret_blocks(tmp_path: Path) -> None:
    ingress, vault = _ingress(tmp_path)
    private_key = (
        "-----BEGIN PRIVATE KEY-----\n"
        "not-a-real-private-key-for-tests\n"
        "-----END PRIVATE KEY-----"
    )

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=(
            "Store this:\n"
            '<secret name="DEPLOY_PRIVATE_KEY">\n'
            f"{private_key}\n"
            "</secret>\n"
            "Use it for the deployment capability."
        ),
    )

    assert result.text == (
        "Store this:\n"
        "secret://deploy_private_key\n"
        "Use it for the deployment capability."
    )
    assert result.handles[0].id == "deploy_private_key"
    assert result.handles[0].scopes == ("credential.use",)
    _assert_absent_from_database(vault, private_key)


def test_ingress_scopes_ssh_private_key_blocks_for_sandbox_connect(tmp_path: Path) -> None:
    ingress, _ = _ingress(tmp_path)
    private_key = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "not-a-real-private-key-for-tests\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f'<secret name="SSH_PRIVATE_KEY">\n{private_key}\n</secret>',
    )

    assert result.text == "secret://ssh_private_key"
    assert result.handles[0].id == "ssh_private_key"
    assert result.handles[0].scopes == ("ssh.connect",)


def test_ingress_infers_malformed_openssh_secret_block_for_sandbox_connect(
    tmp_path: Path,
) -> None:
    ingress, vault = _ingress(tmp_path)
    private_key = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "not-a-real-private-key-for-tests\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f"Store this:\n<secret [redacted]\n{private_key}\n</secret>\nThen connect.",
    )

    assert result.text == "Store this:\nsecret://ssh_private_key\nThen connect."
    assert result.handles[0].id == "ssh_private_key"
    assert result.handles[0].scopes == ("ssh.connect",)
    _assert_absent_from_database(vault, private_key)


def test_ingress_supports_bare_multiline_secret_name(tmp_path: Path) -> None:
    ingress, _ = _ingress(tmp_path)
    private_key = (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        "not-a-real-private-key-for-tests\n"
        "-----END OPENSSH PRIVATE KEY-----"
    )

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=f"<secret SSH_PRIVATE_KEY>\n{private_key}\n</secret>",
    )

    assert result.text == "secret://ssh_private_key"
    assert result.handles[0].id == "ssh_private_key"
    assert result.handles[0].scopes == ("ssh.connect",)


def test_ingress_ignores_unnamed_unrecognized_secret_blocks(tmp_path: Path) -> None:
    ingress, vault = _ingress(tmp_path)
    text = "<secret [redacted]\nthis-is-not-a-recognized-secret-block\n</secret>"

    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=text,
    )

    assert result.text == text
    assert result.handles == ()
    assert vault.list_handles(tenant_id="tenant-a") == []


def test_ingress_ignores_named_placeholders(tmp_path: Path) -> None:
    ingress, vault = _ingress(tmp_path)

    text = "COMPOSIO_API_KEY=changeme"
    result = ingress.ingest(
        tenant_id="tenant-a",
        actor_id="owner-a",
        text=text,
    )

    assert result.text == text
    assert result.handles == ()
    assert vault.get_handle(tenant_id="tenant-a", secret_id="composio_api_key") is None
