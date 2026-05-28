from __future__ import annotations

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


def test_recursion_limit_override_clamps_to_250() -> None:
    runtime = OpenTulpaLangGraphRuntime.__new__(OpenTulpaLangGraphRuntime)
    runtime.recursion_limit = 250

    assert runtime._effective_recursion_limit() == 250
    assert runtime._effective_recursion_limit(999) == 250
    assert runtime._effective_recursion_limit(3) == 5
