from __future__ import annotations

import os
from pathlib import Path

import pytest

from opentulpa.interfaces.telegram.state_store import TelegramStateStore


def test_state_save_failure_preserves_last_complete_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = TelegramStateStore(tmp_path / "telegram.json")
    store.save({"pending_approvals": {"first": {"run_id": "run-1"}}})
    real_replace = os.replace

    def fail_main_replace(source: str | bytes | Path, target: str | bytes | Path) -> None:
        if Path(target) == store.state_path:
            raise OSError("simulated replace failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", fail_main_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        store.save({"pending_approvals": {"second": {"run_id": "run-2"}}})

    assert store.load()["pending_approvals"] == {"first": {"run_id": "run-1"}}
    assert not list(tmp_path.glob("*.tmp"))


def test_state_recovers_from_last_good_backup(tmp_path: Path) -> None:
    store = TelegramStateStore(tmp_path / "telegram.json")
    first = {"pending_approvals": {"first": {"run_id": "run-1"}}}
    store.save(first)
    store.save({"pending_approvals": {"second": {"run_id": "run-2"}}})
    store.state_path.write_text("{partial", encoding="utf-8")

    assert store.load() == first
    assert TelegramStateStore(store.state_path).load() == first


def test_state_corruption_without_backup_fails_closed(tmp_path: Path) -> None:
    store = TelegramStateStore(tmp_path / "telegram.json")
    store.state_path.write_text("{partial", encoding="utf-8")

    with pytest.raises(RuntimeError, match="state is corrupt"):
        store.load()


def test_state_files_are_owner_only(tmp_path: Path) -> None:
    store = TelegramStateStore(tmp_path / "telegram.json")
    store.save({"sessions": {}})
    store.save({"sessions": {"1": {"customer_id": "tenant-1"}}})

    assert store.state_path.stat().st_mode & 0o777 == 0o600
    assert store.backup_path.stat().st_mode & 0o777 == 0o600


def test_owner_webhook_inbox_is_durable_deduplicated_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "telegram.json"
    store = TelegramStateStore(path)
    body = {"update_id": 7, "message": {"text": "hello"}}

    key, should_process = store.enqueue_owner_update(body)
    duplicate_key, duplicate_should_process = TelegramStateStore(path).enqueue_owner_update(body)

    assert (duplicate_key, duplicate_should_process) == (key, True)
    assert should_process is True
    assert TelegramStateStore(path).owner_update(key) == body
    assert TelegramStateStore(path).pending_owner_updates() == [(key, body)]
    assert TelegramStateStore(path).complete_owner_update(key) is True
    assert TelegramStateStore(path).owner_update(key) is None
    assert TelegramStateStore(path).enqueue_owner_update(body) == (key, False)


def test_owner_webhook_update_id_conflict_fails_closed(tmp_path: Path) -> None:
    store = TelegramStateStore(tmp_path / "telegram.json")
    store.enqueue_owner_update({"update_id": 7, "message": {"text": "first"}})

    with pytest.raises(ValueError, match="payload conflict"):
        store.enqueue_owner_update({"update_id": 7, "message": {"text": "different"}})
