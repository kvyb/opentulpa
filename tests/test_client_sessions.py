from __future__ import annotations

import stat
from pathlib import Path

import pytest

from opentulpa.client import config, sessions


def test_sessions_create_switch_and_restore_thread_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_path = tmp_path / "client" / "connection.json"
    monkeypatch.setenv("OPENTULPA_CLIENT_CONFIG", str(client_path))
    monkeypatch.setattr(config, "_keyring_set", lambda account, token: False)
    connection = config.save_connection(
        "https://tulpa.example",
        "owner-secret",
        thread_id="thread-main",
        last_run_id="run-main",
        last_sequence=8,
    )

    assert [item.name for item in sessions.list_sessions(connection)] == ["Main"]

    second = sessions.create_session(connection, name="Research")
    catalog = sessions.list_sessions(second)

    assert [item.name for item in catalog] == ["Main", "Research"]
    assert second.thread_id != connection.thread_id
    assert second.last_run_id is None
    assert stat.S_IMODE(sessions.sessions_path().stat().st_mode) == 0o600

    restored, selected = sessions.switch_session(second, "1")

    assert selected.name == "Main"
    assert restored.thread_id == "thread-main"
    assert restored.last_run_id == "run-main"
    assert restored.last_sequence == 8
    assert config.load_connection() == restored


def test_sessions_reject_duplicate_names_and_unknown_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENTULPA_CLIENT_CONFIG", str(tmp_path / "client.json"))
    monkeypatch.setattr(config, "_keyring_set", lambda account, token: False)
    connection = config.save_connection(
        "https://tulpa.example",
        "owner-secret",
        thread_id="thread-main",
    )
    second = sessions.create_session(connection, name="Research")

    with pytest.raises(sessions.SessionCatalogError, match="already exists"):
        sessions.create_session(second, name="research")
    with pytest.raises(sessions.SessionCatalogError, match="Unknown session"):
        sessions.switch_session(second, "missing")
