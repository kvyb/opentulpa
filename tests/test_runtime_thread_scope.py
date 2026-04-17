from __future__ import annotations

import contextvars

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def test_runtime_active_thread_id_context_round_trip() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)

    token = OpenTulpaLangGraphRuntime.set_active_thread_id(runtime, "thread_123")
    try:
        assert OpenTulpaLangGraphRuntime.get_active_thread_id(runtime) == "thread_123"
        assert runtime._active_thread_id == "thread_123"
    finally:
        OpenTulpaLangGraphRuntime.reset_active_thread_id(runtime, token)

    assert OpenTulpaLangGraphRuntime.get_active_thread_id(runtime) == ""
    assert runtime._active_thread_id == ""


def test_runtime_active_thread_id_reset_tolerates_cross_context() -> None:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)

    other_context = contextvars.copy_context()
    token = other_context.run(OpenTulpaLangGraphRuntime.set_active_thread_id, runtime, "thread_123")

    assert OpenTulpaLangGraphRuntime.get_active_thread_id(runtime) == ""
    assert runtime._active_thread_id == "thread_123"

    OpenTulpaLangGraphRuntime.reset_active_thread_id(runtime, token)

    assert OpenTulpaLangGraphRuntime.get_active_thread_id(runtime) == ""
    assert runtime._active_thread_id == ""
