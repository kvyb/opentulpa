from __future__ import annotations

import pytest

from opentulpa.agent.runtime import OpenTulpaLangGraphRuntime


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, _messages: object) -> _FakeResponse:
        return _FakeResponse(self._content)


class _FailingModel:
    async def ainvoke(self, _messages: object) -> _FakeResponse:
        raise RuntimeError("model_down")


def _mk_runtime_with_model(model: object) -> OpenTulpaLangGraphRuntime:
    runtime = object.__new__(OpenTulpaLangGraphRuntime)
    runtime._guardrail_classifier_model = model
    return runtime


@pytest.mark.asyncio
async def test_verify_completion_claim_flags_mismatch_when_supported() -> None:
    runtime = _mk_runtime_with_model(
        _FakeModel(
            '{"ok": true, "applies": true, "mismatch": true, "confidence": 0.91, '
            '"reason": "claimed_sent_without_tool_success", '
            '"repair_instruction": "run the missing tool first"}'
        )
    )

    result = await runtime.verify_completion_claim(
        user_text="send that file now",
        assistant_text="I sent the file.",
        recent_tool_outputs=[],
    )

    assert result["mismatch"] is True
    assert result["applies"] is True
    assert result["confidence"] == pytest.approx(0.91)
    assert result["usable"] is True


@pytest.mark.asyncio
async def test_verify_completion_claim_is_conservative_when_not_applicable() -> None:
    runtime = _mk_runtime_with_model(
        _FakeModel(
            '{"ok": true, "applies": false, "mismatch": true, "confidence": 0.8, '
            '"reason": "future_tense", "repair_instruction": "none"}'
        )
    )

    result = await runtime.verify_completion_claim(
        user_text="set a schedule",
        assistant_text="I will run this at 16:00.",
        recent_tool_outputs=[],
    )

    assert result["applies"] is False
    assert result["mismatch"] is False
    assert result["usable"] is True


@pytest.mark.asyncio
async def test_verify_completion_claim_fails_open_on_classifier_error() -> None:
    runtime = _mk_runtime_with_model(_FailingModel())

    result = await runtime.verify_completion_claim(
        user_text="post now",
        assistant_text="Done, posted.",
        recent_tool_outputs=[],
    )

    assert result["ok"] is False
    assert result["mismatch"] is False
    assert "classifier_error" in result["reason"]
    assert result["usable"] is False


@pytest.mark.asyncio
async def test_verify_completion_claim_marks_unusable_when_json_missing() -> None:
    runtime = _mk_runtime_with_model(_FakeModel("not json"))

    result = await runtime.verify_completion_claim(
        user_text="run it now",
        assistant_text="I will retry now.",
        recent_tool_outputs=[],
    )

    assert result["usable"] is False
    assert result["mismatch"] is False
    assert result["reason"].startswith("classifier_error:")
