from __future__ import annotations

import sys
from pathlib import Path

_E2E_ROOT = Path(__file__).resolve().parent / "e2e"
if str(_E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(_E2E_ROOT))

from harness.runner import _is_owner_setup_interim_message  # noqa: E402


def test_owner_setup_interim_message_classifier_ignores_progress_updates() -> None:
    assert _is_owner_setup_interim_message(
        {"text": "I’m setting up the workflow now. I’ll send the proposal or exact blocker when validation finishes."},
        status_texts={
            "i’m setting up the workflow now. i’ll send the proposal or exact blocker when validation finishes."
        },
    )
    assert _is_owner_setup_interim_message(
        {"text": "Работаю над настройкой…"},
        status_texts={"работаю над настройкой…"},
    )


def test_owner_setup_interim_message_classifier_keeps_final_setup_replies() -> None:
    assert not _is_owner_setup_interim_message(
        {"text": "Here's your workflow proposal — ready to activate. Shall I save and activate this workflow?"},
        status_texts=set(),
    )
    assert not _is_owner_setup_interim_message(
        {"text": "E2E Quality Car Wash is now live and active. Workflow ID: iwf_123"},
        status_texts={"something else"},
    )
