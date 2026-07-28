from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from opentulpa.capability_workers.state import TelegramStateError, TelegramWorkerState


def test_state_persists_pairing_threads_dedupe_and_private_mode(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    state = TelegramWorkerState(path, max_seen_source_events=100)

    assert state.pair(user_id=7, chat_id=9)
    thread_id = state.thread_id(9)
    state.save_pending_run(
        source_event_id="telegram:1:12",
        update_id=12,
        run_id="run_1",
        chat_id=9,
        sequence=2,
        accumulated_text="hello",
    )
    state.complete_update(update_id=12, source_event_id="telegram:1:12")

    reloaded = TelegramWorkerState(path, max_seen_source_events=100)
    assert reloaded.paired_identity() == (7, 9)
    assert reloaded.thread_id(9) == thread_id
    assert reloaded.source_seen("telegram:1:12")
    assert reloaded.next_update_id == 13
    assert reloaded.pending_runs() == []
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_completion_atomically_consumes_approval_and_edit_marker(tmp_path: Path) -> None:
    state = TelegramWorkerState(tmp_path / "worker.json", max_seen_source_events=100)
    state.save_approval(
        token="token_1",
        run_id="run_1",
        approval_id="approval_1",
        chat_id=9,
        user_id=7,
        allowed_decisions=["approve", "reject"],
        tool_name="integration_invoke",
        description="Send an email",
    )
    state.await_edit(chat_id=9, token="token_1")

    state.complete_update(
        update_id=3,
        source_event_id="telegram:1:3",
        consumed_approval_token="token_1",
    )

    assert state.approval("token_1") is None
    assert state.awaiting_edit(9) is None
    assert state.source_seen("telegram:1:3")


def test_thread_replacement_is_committed_only_after_compare_and_swap(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    state = TelegramWorkerState(path)
    current = state.thread_id(9)
    replacement = state.new_thread_id(9)

    assert TelegramWorkerState(path).thread_id(9) == current

    state.replace_thread(
        9,
        expected_thread_id=current,
        replacement_thread_id=replacement,
    )

    assert TelegramWorkerState(path).thread_id(9) == replacement
    with pytest.raises(TelegramStateError, match="changed during replacement"):
        state.replace_thread(
            9,
            expected_thread_id=current,
            replacement_thread_id=state.new_thread_id(9),
        )


def test_state_rejects_corruption_and_existing_symlink(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text(json.dumps({"version": 2}), encoding="utf-8")
    with pytest.raises(TelegramStateError, match="unsupported schema"):
        TelegramWorkerState(corrupt)

    target = tmp_path / "target.json"
    target.write_text(json.dumps({"version": 1}), encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(TelegramStateError, match="regular file"):
        TelegramWorkerState(linked)


def test_state_refuses_to_repair_to_a_different_identity(tmp_path: Path) -> None:
    state = TelegramWorkerState(tmp_path / "worker.json")
    assert state.pair(user_id=7, chat_id=9)
    assert not state.pair(user_id=8, chat_id=9)
    assert not state.pair(user_id=7, chat_id=10)
