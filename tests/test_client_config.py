from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from opentulpa.client import config
from opentulpa.client.config import ClientConfigError


def _path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "client" / "connection.json"
    monkeypatch.setenv("OPENTULPA_CLIENT_CONFIG", str(path))
    return path


def test_connection_falls_back_to_private_file_and_restores_thread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _path(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "_keyring_set", lambda account, token: False)

    saved = config.save_connection("https://tulpa.example/", "owner-secret")
    updated = config.update_connection(
        saved,
        thread_id="cli-thread-two",
        last_run_id="run-9",
        last_sequence=17,
    )
    loaded = config.load_connection()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert loaded == updated
    assert loaded.url == "https://tulpa.example"
    assert loaded.token == "owner-secret"
    assert loaded.credential_storage == "file"
    config.clear_connection()
    assert not path.exists()


def test_connection_uses_keyring_without_writing_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _path(tmp_path, monkeypatch)
    keyring: dict[str, str] = {}

    def keyring_set(account: str, token: str) -> bool:
        keyring[account] = token
        return True

    monkeypatch.setattr(config, "_keyring_set", keyring_set)
    monkeypatch.setattr(config, "_keyring_get", keyring.get)
    monkeypatch.setattr(config, "_keyring_delete", lambda account: keyring.pop(account, None))

    saved = config.save_connection("https://tulpa.example", "owner-secret")

    assert saved.credential_storage == "keyring"
    assert config.load_connection().token == "owner-secret"
    assert "owner-secret" not in path.read_text(encoding="utf-8")
    config.clear_connection()
    assert not keyring


def test_legacy_private_config_migrates_on_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _path(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"url": "https://old.example/", "token": "old-token"}),
        encoding="utf-8",
    )
    path.chmod(0o600)
    monkeypatch.setattr(config, "_keyring_set", lambda account, token: False)

    loaded = config.load_connection()

    assert loaded.url == "https://old.example"
    assert loaded.token == "old-token"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 2


def test_client_config_rejects_public_permissions_and_unsafe_urls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _path(tmp_path, monkeypatch)
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(ClientConfigError, match="0600"):
        config.load_connection()
    with pytest.raises(ClientConfigError, match="http"):
        config.normalize_url("file:///tmp/server")
    with pytest.raises(ClientConfigError, match="origin"):
        config.normalize_url("https://owner:secret@example.test")
    with pytest.raises(ClientConfigError, match="origin"):
        config.normalize_url("https://example.test/opentulpa")
